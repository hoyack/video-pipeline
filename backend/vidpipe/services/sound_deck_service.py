"""Production sound-deck generation, SFX rendering, and stem mixing."""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.config import settings
from vidpipe.db.models import (
    Production,
    Scene,
    Shot,
    ShotAudioManifest,
    SoundEffectCue,
    SoundMixArtifact,
    SoundAsset,
    SoundBinding,
)
from vidpipe.schemas.sound_deck import GeneratedSoundDeck
from vidpipe.services.audio.base import AudioAdapter, AudioAdapterError
from vidpipe.services.audio.registry import get_audio_adapter
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend

logger = logging.getLogger(__name__)


SOUND_DECK_SYSTEM_PROMPT = """You design production sound effects for a video timeline.
Return only JSON matching the schema. Create concise, layerable SFX/ambience/foley
cues that support the visuals without competing with narration. Use scene_number and
shot_number from the provided list. Prompts must be ready for a text-to-sound model."""


class SoundDeckError(RuntimeError):
    """Raised when sound-deck generation or mixing fails."""


class SoundDeckMixer:
    """Build scene-level SFX stems from generated sound cue audio."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.storage = get_storage_backend()
        self.base_dir = base_dir or settings.storage.tmp_dir

    async def build_scene_stem(
        self,
        production_id: uuid.UUID,
        scene_id: uuid.UUID,
        cues: list[SoundEffectCue],
        *,
        duration_seconds: float,
    ) -> tuple[str, float | None]:
        ready_cues = [cue for cue in cues if cue.audio_path]
        if not ready_cues:
            raise SoundDeckError("No generated sound cue audio is available")

        output_key = f"productions/{production_id}/sound/mix/scene_{scene_id}.m4a"
        output_path = self.base_dir / output_key
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_paths = [await self._ensure_local(cue.audio_path) for cue in ready_cues if cue.audio_path]
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for path in input_paths:
            cmd.extend(["-i", str(path)])

        filters: list[str] = []
        labels: list[str] = []
        for index, cue in enumerate(ready_cues):
            label = f"sfx{index}"
            delay_ms = max(0, round((cue.start_time_seconds or 0.0) * 1000))
            volume_db = cue.volume_db if cue.volume_db is not None else -18.0
            filters.append(
                f"[{index}:a:0]adelay={delay_ms}:all=1,volume={volume_db}dB,apad,"
                f"atrim=0:{duration_seconds:.3f}[{label}]"
            )
            labels.append(f"[{label}]")
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0:"
            f"dropout_transition=0,atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[a]"
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
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SoundDeckError(f"Sound mix failed: {exc}") from exc

        await self.storage.put(output_key, output_path.read_bytes(), "audio/mp4")
        return output_key, _probe_duration(output_path)

    async def _ensure_local(self, audio_path: str) -> Path:
        path = Path(audio_path)
        if path.is_absolute() and path.exists():
            return path
        if isinstance(self.storage, LocalStorageBackend):
            local_path = self.storage.resolve_local_path(audio_path)
            if local_path.exists():
                return local_path
        local_path = self.base_dir / audio_path
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(await self.storage.get(audio_path))
        return local_path


class SoundDeckService:
    """Coordinates generated production SFX cues and mix artifacts."""

    async def list_cues(self, session: AsyncSession, production_id: uuid.UUID) -> list[SoundEffectCue]:
        result = await session.execute(
            select(SoundEffectCue)
            .where(SoundEffectCue.production_id == production_id)
            .order_by(
                SoundEffectCue.scene_number,
                SoundEffectCue.shot_number,
                SoundEffectCue.cue_index,
                SoundEffectCue.created_at,
            )
        )
        return list(result.scalars().all())

    async def list_mix_artifacts(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
    ) -> list[SoundMixArtifact]:
        result = await session.execute(
            select(SoundMixArtifact)
            .where(SoundMixArtifact.production_id == production_id)
            .order_by(desc(SoundMixArtifact.created_at))
        )
        return list(result.scalars().all())

    async def generate_from_production(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        llm_adapter: Any,
        *,
        text_model: str | None = None,
    ) -> list[SoundEffectCue]:
        production = await session.get(Production, production_id)
        if production is None:
            raise ValueError("Production not found")

        prompt = await self._build_generation_prompt(session, production)
        generated = await llm_adapter.generate_text(
            prompt,
            GeneratedSoundDeck,
            temperature=0.45,
            system_prompt=SOUND_DECK_SYSTEM_PROMPT,
        )
        if not isinstance(generated, GeneratedSoundDeck):
            generated = GeneratedSoundDeck.model_validate(generated)
        if len(generated.cues) < 2:
            generated = GeneratedSoundDeck(cues=await self._fallback_generated_cues(session, production))

        await session.execute(delete(SoundEffectCue).where(SoundEffectCue.production_id == production_id))
        await session.flush()

        cues: list[SoundEffectCue] = []
        for index, cue_data in enumerate(generated.cues):
            scene_id, shot_id, scene_start = await self._resolve_scene_shot(
                session,
                production_id,
                cue_data.scene_number,
                cue_data.shot_number,
            )
            cue = SoundEffectCue(
                production_id=production_id,
                scene_id=scene_id,
                shot_id=shot_id,
                scene_number=cue_data.scene_number,
                shot_number=cue_data.shot_number,
                cue_index=index,
                cue_type=cue_data.cue_type,
                name=cue_data.name,
                prompt=cue_data.prompt,
                timing_hint=cue_data.timing_hint,
                start_time_seconds=self._normalized_start_seconds(cue_data.start_time_seconds, scene_start),
                duration_seconds=cue_data.duration_seconds or self._default_duration(cue_data.cue_type),
                volume_db=cue_data.volume_db if cue_data.volume_db is not None else self._default_volume(cue_data.cue_type),
                generation_status="PENDING",
                provider_metadata={"text_model": text_model} if text_model else None,
            )
            cues.append(cue)
            session.add(cue)
        await session.commit()
        return cues

    async def update_cue(
        self,
        session: AsyncSession,
        cue_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> SoundEffectCue:
        cue = await session.get(SoundEffectCue, cue_id)
        if cue is None:
            raise ValueError("Sound cue not found")

        uuid_fields = {"sound_asset_id", "sfx_item_id"}
        dirty_generation_fields = {"prompt", "duration_seconds", "cue_type", "name"}
        should_reset_audio = bool(dirty_generation_fields.intersection(updates))
        for field_name, value in updates.items():
            if field_name in uuid_fields and value is not None:
                value = uuid.UUID(str(value))
            setattr(cue, field_name, value)

        if "scene_number" in updates or "shot_number" in updates:
            cue.scene_id, cue.shot_id, scene_start = await self._resolve_scene_shot(
                session,
                cue.production_id,
                cue.scene_number,
                cue.shot_number,
            )
            if cue.start_time_seconds is None:
                cue.start_time_seconds = scene_start

        if should_reset_audio:
            cue.generation_status = "PENDING"
            cue.audio_path = None
            cue.error_message = None
        await session.commit()
        await session.refresh(cue)
        return cue

    async def delete_cue(self, session: AsyncSession, cue_id: uuid.UUID) -> None:
        cue = await session.get(SoundEffectCue, cue_id)
        if cue is None:
            raise ValueError("Sound cue not found")
        await session.delete(cue)
        await session.commit()

    async def generate_cue_audio(
        self,
        session: AsyncSession,
        cue_id: uuid.UUID,
        *,
        audio_adapter: AudioAdapter | None = None,
        api_key: str | None = None,
    ) -> SoundEffectCue:
        cue = await session.get(SoundEffectCue, cue_id)
        if cue is None:
            raise ValueError("Sound cue not found")

        existing_audio_path = cue.audio_path
        existing_duration_seconds = cue.duration_seconds
        cue.generation_status = "GENERATING"
        cue.error_message = None
        await session.flush()
        try:
            adapter = audio_adapter or get_audio_adapter("ELEVENLABS", api_key=api_key)
            content = await adapter.generate_sfx(
                cue.prompt,
                duration_seconds=cue.duration_seconds,
            )
            audio_key = f"productions/{cue.production_id}/sound/cues/{cue.id}.mp3"
            storage = get_storage_backend()
            await storage.put(audio_key, content, "audio/mpeg")
            cue.audio_path = audio_key
            cue.audio_mime_type = "audio/mpeg"
            cue.duration_seconds = await self._probe_stored_audio_duration(audio_key) or cue.duration_seconds
            cue.generation_status = "READY"
            cue.error_message = None
        except (AudioAdapterError, Exception) as exc:
            if existing_audio_path:
                cue.audio_path = existing_audio_path
                cue.duration_seconds = existing_duration_seconds
                cue.generation_status = "READY"
                cue.error_message = f"Regeneration failed; existing audio retained: {exc}"[:500]
            else:
                cue.generation_status = "FAILED"
                cue.error_message = str(exc)[:500]
            logger.exception("Sound cue generation failed for %s", cue.id)
        await session.commit()
        await session.refresh(cue)
        return cue

    async def generate_pending_audio(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        *,
        audio_adapter: AudioAdapter | None = None,
        api_key: str | None = None,
    ) -> list[SoundEffectCue]:
        generated: list[SoundEffectCue] = []
        for cue in await self.list_cues(session, production_id):
            if cue.generation_status not in {"PENDING", "FAILED"}:
                continue
            generated.append(
                await self.generate_cue_audio(
                    session,
                    cue.id,
                    audio_adapter=audio_adapter,
                    api_key=api_key,
                )
            )
        return generated

    async def build_mix_artifacts(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        *,
        mixer: SoundDeckMixer | None = None,
    ) -> list[SoundMixArtifact]:
        production = await session.get(Production, production_id)
        if production is None:
            raise ValueError("Production not found")

        ready_cues = [cue for cue in await self.list_cues(session, production_id) if cue.audio_path]
        if not ready_cues:
            return []

        await session.execute(delete(SoundMixArtifact).where(SoundMixArtifact.production_id == production_id))
        groups: dict[uuid.UUID | None, list[SoundEffectCue]] = defaultdict(list)
        for cue in ready_cues:
            groups[cue.scene_id].append(cue)

        mixer = mixer or SoundDeckMixer()
        artifacts: list[SoundMixArtifact] = []
        for scene_id, cues in groups.items():
            duration = await self._scene_duration(session, scene_id) if scene_id else self._production_duration(session, production_id)
            artifact = SoundMixArtifact(
                production_id=production_id,
                scene_id=scene_id,
                artifact_type="SCENE_SFX_STEM" if scene_id else "PRODUCTION_SFX_STEM",
                status="GENERATING",
            )
            session.add(artifact)
            await session.flush()
            try:
                key, stem_duration = await mixer.build_scene_stem(
                    production_id,
                    scene_id or production_id,
                    sorted(cues, key=lambda item: item.cue_index),
                    duration_seconds=duration,
                )
                artifact.audio_path = key
                artifact.duration_seconds = stem_duration
                artifact.status = "READY"
            except SoundDeckError as exc:
                artifact.status = "FAILED"
                artifact.error_message = str(exc)[:500]
            artifacts.append(artifact)
        await session.commit()
        return artifacts

    async def _build_generation_prompt(self, session: AsyncSession, production: Production) -> str:
        scenes_payload = []
        result = await session.execute(
            select(Scene)
            .where(
                Scene.production_id == production.id,
                Scene.deleted_at.is_(None),
            )
            .order_by(Scene.scene_order, Scene.screenplay_breakdown_index, Scene.created_at)
        )
        scenes = list(result.scalars().all())
        for scene_number, scene in enumerate(scenes, start=1):
            shots_result = await session.execute(
                select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
            )
            shots_payload = []
            for shot in shots_result.scalars().all():
                manifest = await session.get(ShotAudioManifest, (scene.id, shot.shot_index))
                shots_payload.append(
                    {
                        "shot_number": shot.shot_index + 1,
                        "shot_start_seconds": (shot.shot_index or 0) * (scene.target_clip_duration or 5),
                        "description": shot.shot_description,
                        "motion": shot.video_motion_prompt,
                        "sfx_manifest": manifest.sfx_json if manifest else [],
                        "ambient_manifest": manifest.ambient_json if manifest else None,
                    }
                )
            scenes_payload.append(
                {
                    "scene_number": scene_number,
                    "title": scene.title,
                    "duration_seconds": (scene.target_clip_duration or 5) * max(len(shots_payload), 1),
                    "shots": shots_payload,
                }
            )

        sounds = []
        if production.production_bible_id:
            result = await session.execute(
                select(SoundBinding, SoundAsset)
                .join(SoundAsset, SoundBinding.sound_asset_id == SoundAsset.id)
                .where(SoundBinding.production_bible_id == production.production_bible_id)
                .order_by(SoundBinding.created_at)
            )
            sounds = [
                {
                    "tag": binding.tag,
                    "name": asset.name,
                    "category": asset.category,
                    "usage_notes": binding.usage_notes,
                    "prompt_tags": binding.prompt_tags,
                }
                for binding, asset in result.all()
            ]

        return (
            f"Production: {production.name}\n"
            f"Description: {production.description or ''}\n"
            f"Sound assets bound in bible: {sounds}\n"
            f"Scenes and shots: {scenes_payload}\n\n"
            "Create 1-3 sound cues per scene. Prefer ambience beds and concrete foley/SFX "
            "that support the visible action. Use scene-relative start_time_seconds. "
            "Use duration_seconds between 1 and 8 for spot effects, and up to the scene duration "
            "for ambience. Keep volume_db negative: around -24 for ambience, -18 for foley, "
            "and -12 for prominent impacts."
        )

    async def _fallback_generated_cues(self, session: AsyncSession, production: Production):
        from vidpipe.schemas.sound_deck import GeneratedSoundCue

        result = await session.execute(
            select(Scene)
            .where(Scene.production_id == production.id, Scene.deleted_at.is_(None))
            .order_by(Scene.scene_order, Scene.screenplay_breakdown_index, Scene.created_at)
        )
        cues: list[GeneratedSoundCue] = []
        for scene_number, scene in enumerate(result.scalars().all(), start=1):
            cues.append(
                GeneratedSoundCue(
                    scene_number=scene_number,
                    shot_number=1,
                    cue_type="AMBIENCE",
                    name=f"{scene.title or 'Scene'} ambience",
                    prompt=f"Subtle documentary ambience for {scene.title or scene.prompt}: low room tone, archival texture, restrained movement, no music, no speech.",
                    timing_hint="throughout scene",
                    start_time_seconds=0.0,
                    duration_seconds=(scene.target_clip_duration or 5) * max(scene.target_shot_count or 1, 1),
                    volume_db=-26.0,
                )
            )
        return cues

    async def _resolve_scene_shot(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        scene_number: int | None,
        shot_number: int | None,
    ) -> tuple[uuid.UUID | None, uuid.UUID | None, float | None]:
        if scene_number is None:
            return None, None, None
        scenes_result = await session.execute(
            select(Scene)
            .where(Scene.production_id == production_id, Scene.deleted_at.is_(None))
            .order_by(Scene.scene_order, Scene.screenplay_breakdown_index, Scene.created_at)
        )
        scenes = list(scenes_result.scalars().all())
        if scene_number < 1 or scene_number > len(scenes):
            return None, None, None
        scene = scenes[scene_number - 1]
        if shot_number is None:
            return scene.id, None, None
        shot_result = await session.execute(
            select(Shot).where(
                Shot.scene_id == scene.id,
                Shot.shot_index == shot_number - 1,
            )
        )
        shot = shot_result.scalar_one_or_none()
        scene_start = (shot_number - 1) * (scene.target_clip_duration or 5)
        return scene.id, shot.id if shot else None, scene_start

    def _normalized_start_seconds(self, value: float | None, fallback: float | None) -> float:
        if value is None:
            return max(0.0, fallback or 0.0)
        return max(0.0, value)

    def _default_duration(self, cue_type: str) -> float:
        return 6.0 if cue_type == "AMBIENCE" else 2.0

    def _default_volume(self, cue_type: str) -> float:
        if cue_type == "AMBIENCE":
            return -26.0
        if cue_type in {"IMPACT", "MECHANICAL"}:
            return -14.0
        return -18.0

    async def _scene_duration(self, session: AsyncSession, scene_id: uuid.UUID | None) -> float:
        if scene_id is None:
            return 1.0
        scene = await session.get(Scene, scene_id)
        if scene is None:
            return 1.0
        return float((scene.target_clip_duration or 5) * max(scene.target_shot_count or 1, 1))

    async def _production_duration(self, session: AsyncSession, production_id: uuid.UUID) -> float:
        result = await session.execute(
            select(Scene).where(Scene.production_id == production_id, Scene.deleted_at.is_(None))
        )
        return sum((scene.target_clip_duration or 5) * max(scene.target_shot_count or 1, 1) for scene in result.scalars().all())

    async def _probe_stored_audio_duration(self, audio_key: str) -> float | None:
        storage = get_storage_backend()
        if isinstance(storage, LocalStorageBackend):
            path = storage.resolve_local_path(audio_key)
        else:
            path = settings.storage.tmp_dir / audio_key
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(await storage.get(audio_key))
        return _probe_duration(path)


def _probe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
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
        return float(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        logger.debug("Could not probe duration for %s", path)
        return None
