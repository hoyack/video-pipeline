"""Production-level video rendering service."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Production, Scene, Sequence, SoundMixArtifact, VoiceMixArtifact
from vidpipe.services.file_manager import FileManager
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend


@dataclass(frozen=True)
class SceneAudioStem:
    """A scene audio stem positioned on the production timeline."""

    path: Path
    scene_start_seconds: float
    duration_seconds: float
    kind: str


@dataclass(frozen=True)
class ProductionMasterResult:
    """Rendered production master metadata."""

    production_id: uuid.UUID
    video_path: str
    video_url: str
    scene_count: int
    duration_seconds: float | None
    audio_stem_count: int


class ProductionMasterError(RuntimeError):
    """Raised when a production master cannot be rendered."""


class ProductionMasterService:
    """Render a final production MP4 from completed scene videos and voice stems."""

    def __init__(self, file_manager: FileManager | None = None) -> None:
        self.file_manager = file_manager or FileManager()
        self.storage = get_storage_backend()

    async def render_master(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
    ) -> ProductionMasterResult:
        """Render and store a master video for a production."""
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise ProductionMasterError("ffmpeg and ffprobe are required to render a production master")

        production = await session.get(Production, production_id)
        if production is None:
            raise ValueError("Production not found")

        scenes = await self._list_completed_scenes(session, production_id)
        if not scenes:
            raise ProductionMasterError("No completed scenes with stitched output found")

        work_dir = self.file_manager.base_dir / "productions" / str(production_id) / "master"
        segment_dir = work_dir / "segments"
        input_dir = work_dir / "inputs"
        segment_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        segment_paths: list[Path] = []
        audio_stems: list[SceneAudioStem] = []
        audio_stem_count = 0
        scene_start_seconds = 0.0
        for index, scene in enumerate(scenes):
            assert scene.output_path is not None
            video_path = await self._materialize(scene.output_path, input_dir / f"scene_{index}.mp4")
            scene_duration = self._probe_duration(video_path)
            if scene_duration is None or scene_duration <= 0:
                raise ProductionMasterError(f"Could not determine video duration for {video_path}")
            scene_stems = await self._latest_scene_audio_stems(session, scene.id)
            for stem_index, stem in enumerate(scene_stems):
                audio_path = await self._materialize(
                    str(stem.path),
                    input_dir / f"scene_{index}_{stem.kind}_{stem_index}.m4a",
                )
                stem_duration = stem.duration_seconds or self._probe_duration(audio_path)
                if stem_duration is not None and stem_duration > 0:
                    audio_stems.append(
                        SceneAudioStem(
                            path=audio_path,
                            scene_start_seconds=scene_start_seconds,
                            duration_seconds=stem_duration,
                            kind=stem.kind,
                        )
                    )
                    audio_stem_count += 1

            segment_path = segment_dir / f"{index:03d}_{scene.id}.mp4"
            self._render_video_segment(video_path, segment_path)
            segment_paths.append(segment_path)
            scene_start_seconds += scene_duration

        video_timeline_path = work_dir / "master_video.mp4"
        self._concat_segments(segment_paths, video_timeline_path)
        video_duration = self._probe_duration(video_timeline_path)
        if video_duration is None or video_duration <= 0:
            raise ProductionMasterError("Could not determine production video duration")

        audio_timeline = self._position_audio_stems(audio_stems)
        audio_duration = max((start + stem.duration_seconds for start, stem in audio_timeline), default=0.0)
        output_duration = max(video_duration, audio_duration)

        if output_duration > video_duration + 0.05:
            extended_video_path = work_dir / "master_video_extended.mp4"
            self._extend_video_to_duration(video_timeline_path, extended_video_path, output_duration)
            video_timeline_path = extended_video_path

        audio_track_path = work_dir / "master_audio.m4a"
        self._build_audio_track(audio_timeline, output_duration, audio_track_path)

        output_path = work_dir / "master.mp4"
        self._mux_video_audio(video_timeline_path, audio_track_path, output_path, output_duration)

        storage_key = f"productions/{production_id}/output/master.mp4"
        await self.storage.put(storage_key, output_path.read_bytes(), "video/mp4")
        return ProductionMasterResult(
            production_id=production_id,
            video_path=storage_key,
            video_url=f"/api/productions/{production_id}/master-video",
            scene_count=len(scenes),
            duration_seconds=self._probe_duration(output_path),
            audio_stem_count=audio_stem_count,
        )

    async def get_existing_master(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
    ) -> ProductionMasterResult:
        """Return metadata for an already-rendered production master."""
        production = await session.get(Production, production_id)
        if production is None:
            raise ValueError("Production not found")

        storage_key = self.master_storage_key(production_id)
        if not await self.storage.exists(storage_key):
            raise FileNotFoundError("Production master not found")

        scenes = await self._list_completed_scenes(session, production_id)
        audio_stem_count = 0
        for scene in scenes:
            audio_stem_count += len(await self._latest_scene_audio_stems(session, scene.id))

        duration_seconds = None
        if shutil.which("ffprobe") is not None:
            try:
                if isinstance(self.storage, LocalStorageBackend):
                    master_path = self.storage.resolve_local_path(storage_key)
                else:
                    work_dir = self.file_manager.base_dir / "productions" / str(production_id) / "master"
                    master_path = await self._materialize(storage_key, work_dir / "master_existing.mp4")
                duration_seconds = self._probe_duration(master_path)
            except (FileNotFoundError, ProductionMasterError, subprocess.SubprocessError):
                duration_seconds = None

        return ProductionMasterResult(
            production_id=production_id,
            video_path=storage_key,
            video_url=f"/api/productions/{production_id}/master-video",
            scene_count=len(scenes),
            duration_seconds=duration_seconds,
            audio_stem_count=audio_stem_count,
        )

    def master_storage_key(self, production_id: uuid.UUID) -> str:
        """Return the canonical storage key for a production master video."""
        return f"productions/{production_id}/output/master.mp4"

    async def _list_completed_scenes(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
    ) -> list[Scene]:
        result = await session.execute(
            select(Scene)
            .outerjoin(Sequence, Scene.sequence_id == Sequence.id)
            .where(
                Scene.production_id == production_id,
                Scene.deleted_at.is_(None),
                Scene.status == "complete",
                Scene.output_path.isnot(None),
            )
            .order_by(
                Sequence.order,
                Scene.scene_order,
                Scene.screenplay_breakdown_index,
                Scene.created_at,
            )
        )
        return list(result.scalars().all())

    async def _latest_scene_stem(
        self,
        session: AsyncSession,
        scene_id: uuid.UUID,
    ) -> VoiceMixArtifact | None:
        result = await session.execute(
            select(VoiceMixArtifact)
            .where(
                VoiceMixArtifact.scene_id == scene_id,
                VoiceMixArtifact.artifact_type == "SCENE_VOICE_STEM",
                VoiceMixArtifact.status == "READY",
                VoiceMixArtifact.audio_path.isnot(None),
            )
            .order_by(desc(VoiceMixArtifact.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _latest_scene_sound_stem(
        self,
        session: AsyncSession,
        scene_id: uuid.UUID,
    ) -> SoundMixArtifact | None:
        result = await session.execute(
            select(SoundMixArtifact)
            .where(
                SoundMixArtifact.scene_id == scene_id,
                SoundMixArtifact.artifact_type == "SCENE_SFX_STEM",
                SoundMixArtifact.status == "READY",
                SoundMixArtifact.audio_path.isnot(None),
            )
            .order_by(desc(SoundMixArtifact.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _latest_scene_audio_stems(
        self,
        session: AsyncSession,
        scene_id: uuid.UUID,
    ) -> list[SceneAudioStem]:
        stems: list[SceneAudioStem] = []
        voice_stem = await self._latest_scene_stem(session, scene_id)
        if voice_stem is not None and voice_stem.audio_path:
            stems.append(
                SceneAudioStem(
                    path=Path(voice_stem.audio_path),
                    scene_start_seconds=0.0,
                    duration_seconds=voice_stem.duration_seconds or 0.0,
                    kind="voice",
                )
            )
        sound_stem = await self._latest_scene_sound_stem(session, scene_id)
        if sound_stem is not None and sound_stem.audio_path:
            stems.append(
                SceneAudioStem(
                    path=Path(sound_stem.audio_path),
                    scene_start_seconds=0.0,
                    duration_seconds=sound_stem.duration_seconds or 0.0,
                    kind="sfx",
                )
            )
        return stems

    async def _materialize(self, path_or_key: str, destination: Path) -> Path:
        path = Path(path_or_key)
        if path.is_absolute() and path.exists():
            return path

        if isinstance(self.storage, LocalStorageBackend) and not path.is_absolute():
            local_path = self.storage.resolve_local_path(path_or_key)
            if local_path.exists():
                return local_path

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(await self.file_manager.read_bytes(path_or_key))
        return destination

    def _render_video_segment(
        self,
        video_path: Path,
        output_path: Path,
    ) -> None:
        duration_seconds = self._probe_duration(video_path)
        if duration_seconds is None or duration_seconds <= 0:
            raise ProductionMasterError(f"Could not determine video duration for {video_path}")
        video_filter = (
            "[0:v:0]scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v]"
        )
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path)]
        cmd.extend([
            "-filter_complex",
            video_filter,
            "-map",
            "[v]",
            "-an",
            "-t",
            f"{duration_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ])
        self._run(cmd)

    def _concat_segments(self, segment_paths: list[Path], output_path: Path) -> None:
        concat_file = output_path.with_suffix(".txt")
        concat_file.write_text(
            "".join(f"file '{self._ffmpeg_concat_path(path)}'\n" for path in segment_paths),
            encoding="utf-8",
        )
        self._run([
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ])

    def _position_audio_stems(self, stems: list[SceneAudioStem]) -> list[tuple[float, SceneAudioStem]]:
        positioned: list[tuple[float, SceneAudioStem]] = []
        next_available_audio_time = 0.0
        for stem in stems:
            start_seconds = stem.scene_start_seconds
            if stem.kind == "voice":
                start_seconds = max(stem.scene_start_seconds, next_available_audio_time)
                next_available_audio_time = start_seconds + stem.duration_seconds
            positioned.append((start_seconds, stem))
        return positioned

    def _build_audio_track(
        self,
        audio_timeline: list[tuple[float, SceneAudioStem]],
        duration_seconds: float,
        output_path: Path,
    ) -> None:
        if not audio_timeline:
            self._run([
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t",
                f"{duration_seconds:.3f}",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_path),
            ])
            return

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for _, stem in audio_timeline:
            cmd.extend(["-i", str(stem.path)])

        filters = []
        labels = []
        for index, (start_seconds, _) in enumerate(audio_timeline):
            label = f"a{index}"
            delay_ms = max(0, round(start_seconds * 1000))
            filters.append(
                f"[{index}:a:0]adelay={delay_ms}:all=1,apad,"
                f"atrim=0:{duration_seconds:.3f}[{label}]"
            )
            labels.append(f"[{label}]")
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0:"
            f"dropout_transition=0,alimiter=limit=0.95,"
            f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[a]"
        )
        cmd.extend([
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[a]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ])
        self._run(cmd)

    def _extend_video_to_duration(
        self,
        input_path: Path,
        output_path: Path,
        duration_seconds: float,
    ) -> None:
        current_duration = self._probe_duration(input_path)
        if current_duration is None or current_duration <= 0:
            raise ProductionMasterError(f"Could not determine video duration for {input_path}")
        extension_seconds = max(0.0, duration_seconds - current_duration)
        self._run([
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={extension_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ])

    def _mux_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        duration_seconds: float,
    ) -> None:
        self._run([
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration_seconds:.3f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ])

    def _has_audio_stream(self, path: Path) -> bool:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def _probe_duration(self, path: Path) -> float | None:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return None

    def _run(self, cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = result.stderr.strip() or "ffmpeg command failed"
            raise ProductionMasterError(message[:1000])

    def _ffmpeg_concat_path(self, path: Path) -> str:
        return str(path).replace("'", "'\\''")
