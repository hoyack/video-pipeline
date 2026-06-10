"""Tests for production-level master rendering."""

import shutil
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.config import settings
from vidpipe.db.models import (
    Base,
    Production,
    Scene,
    Screenplay,
    SoundMixArtifact,
    VoiceMixArtifact,
    VoiceScript,
)
from vidpipe.services.production_master import ProductionMasterService
from vidpipe.services.storage_backend import get_storage_backend, reset_storage_backend


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for production master tests")


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _write_silent_video(path: Path, color: str) -> None:
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=320x180:d=1",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ])


def _write_tone_mp3(path: Path, frequency: int, duration: float = 1) -> None:
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={frequency}:duration={duration}",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "96k",
        str(path),
    ])


def _probe_streams(path: Path, stream_selector: str) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            stream_selector,
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
    return result.stdout.strip()


def _mean_volume(path: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def _parse_max_volume(output: str) -> float:
    for line in output.splitlines():
        if "max_volume:" in line:
            return float(line.split("max_volume:", 1)[1].strip().split()[0])
    raise AssertionError(f"No max_volume in ffmpeg output: {output}")


def _mean_volume_segment(path: Path, start_seconds: float, duration_seconds: float) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def _probe_duration(path: Path) -> float:
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


@pytest_asyncio.fixture
async def session_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage, "tmp_dir", tmp_path)
    monkeypatch.setattr(settings.storage, "storage_backend", "local")
    reset_storage_backend()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        reset_storage_backend()
        await engine.dispose()


@pytest.mark.asyncio
async def test_render_master_concats_scenes_and_overlays_voice_stems(
    session_factory,
    tmp_path,
):
    _require_ffmpeg()
    storage = get_storage_backend()

    scene_video_1 = tmp_path / "scene_1.mp4"
    scene_video_2 = tmp_path / "scene_2.mp4"
    stem_1 = tmp_path / "stem_1.mp3"
    stem_2 = tmp_path / "stem_2.mp3"
    sfx_stem = tmp_path / "sfx_stem.mp3"
    _write_silent_video(scene_video_1, "blue")
    _write_silent_video(scene_video_2, "red")
    _write_tone_mp3(stem_1, 440)
    _write_tone_mp3(stem_2, 660)
    _write_tone_mp3(sfx_stem, 880)

    scene_key_1 = await storage.put("fixtures/scene_1.mp4", scene_video_1.read_bytes(), "video/mp4")
    scene_key_2 = await storage.put("fixtures/scene_2.mp4", scene_video_2.read_bytes(), "video/mp4")
    stem_key_1 = await storage.put("fixtures/stem_1.mp3", stem_1.read_bytes(), "audio/mpeg")
    stem_key_2 = await storage.put("fixtures/stem_2.mp3", stem_2.read_bytes(), "audio/mpeg")
    sfx_stem_key = await storage.put("fixtures/sfx_stem.mp3", sfx_stem.read_bytes(), "audio/mpeg")

    service = ProductionMasterService()
    async with session_factory() as session:
        production = Production(name="Synthetic NASA Documentary")
        session.add(production)
        await session.flush()

        scene_1 = Scene(
            production_id=production.id,
            title="Mercury control room",
            prompt="A control room prepares for launch.",
            style="documentary",
            aspect_ratio="16:9",
            target_clip_duration=1,
            target_shot_count=1,
            status="complete",
            output_path=scene_key_1,
            scene_order=1,
            screenplay_breakdown_index=1,
        )
        scene_2 = Scene(
            production_id=production.id,
            title="Launch pad",
            prompt="A rocket waits on the launch pad.",
            style="documentary",
            aspect_ratio="16:9",
            target_clip_duration=1,
            target_shot_count=1,
            status="complete",
            output_path=scene_key_2,
            scene_order=2,
            screenplay_breakdown_index=2,
        )
        session.add_all([scene_1, scene_2])
        await session.flush()

        screenplay = Screenplay(production_id=production.id, title="Synthetic NASA Documentary")
        session.add(screenplay)
        await session.flush()
        voice_script = VoiceScript(screenplay_id=screenplay.id, production_id=production.id)
        session.add(voice_script)
        await session.flush()
        session.add_all([
            VoiceMixArtifact(
                voice_script_id=voice_script.id,
                scene_id=scene_1.id,
                artifact_type="SCENE_VOICE_STEM",
                audio_path=stem_key_1,
                duration_seconds=1,
                status="READY",
            ),
            VoiceMixArtifact(
                voice_script_id=voice_script.id,
                scene_id=scene_2.id,
                artifact_type="SCENE_VOICE_STEM",
                audio_path=stem_key_2,
                duration_seconds=1,
                status="READY",
            ),
            SoundMixArtifact(
                production_id=production.id,
                scene_id=scene_1.id,
                artifact_type="SCENE_SFX_STEM",
                audio_path=sfx_stem_key,
                duration_seconds=1,
                status="READY",
            ),
        ])
        await session.commit()

        result = await service.render_master(session, production.id)

    assert result.scene_count == 2
    assert result.audio_stem_count == 3
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 1.8
    assert await storage.exists(result.video_path)

    master_path = storage.resolve_local_path(result.video_path)
    assert _probe_streams(master_path, "v")
    assert _probe_streams(master_path, "a")
    volume_output = _mean_volume(master_path)
    assert "mean_volume: -inf" not in volume_output
    assert "max_volume: -inf" not in volume_output
    assert _parse_max_volume(volume_output) > -20.0


@pytest.mark.asyncio
async def test_render_master_layers_voice_after_video_concat_without_cutting_tail(
    session_factory,
    tmp_path,
):
    _require_ffmpeg()
    storage = get_storage_backend()

    scene_video_1 = tmp_path / "scene_1.mp4"
    scene_video_2 = tmp_path / "scene_2.mp4"
    stem_1 = tmp_path / "stem_1.mp3"
    stem_2 = tmp_path / "stem_2.mp3"
    _write_silent_video(scene_video_1, "blue")
    _write_silent_video(scene_video_2, "red")
    _write_tone_mp3(stem_1, 440, duration=1.4)
    _write_tone_mp3(stem_2, 660, duration=1.4)

    scene_key_1 = await storage.put("fixtures/tail_scene_1.mp4", scene_video_1.read_bytes(), "video/mp4")
    scene_key_2 = await storage.put("fixtures/tail_scene_2.mp4", scene_video_2.read_bytes(), "video/mp4")
    stem_key_1 = await storage.put("fixtures/tail_stem_1.mp3", stem_1.read_bytes(), "audio/mpeg")
    stem_key_2 = await storage.put("fixtures/tail_stem_2.mp3", stem_2.read_bytes(), "audio/mpeg")

    service = ProductionMasterService()
    async with session_factory() as session:
        production = Production(name="Synthetic NASA Documentary")
        session.add(production)
        await session.flush()

        scene_1 = Scene(
            production_id=production.id,
            title="Mercury control room",
            prompt="A control room prepares for launch.",
            style="documentary",
            aspect_ratio="16:9",
            target_clip_duration=1,
            target_shot_count=1,
            status="complete",
            output_path=scene_key_1,
            scene_order=1,
            screenplay_breakdown_index=1,
        )
        scene_2 = Scene(
            production_id=production.id,
            title="Launch pad",
            prompt="A rocket waits on the launch pad.",
            style="documentary",
            aspect_ratio="16:9",
            target_clip_duration=1,
            target_shot_count=1,
            status="complete",
            output_path=scene_key_2,
            scene_order=2,
            screenplay_breakdown_index=2,
        )
        session.add_all([scene_1, scene_2])
        await session.flush()

        screenplay = Screenplay(production_id=production.id, title="Synthetic NASA Documentary")
        session.add(screenplay)
        await session.flush()
        voice_script = VoiceScript(screenplay_id=screenplay.id, production_id=production.id)
        session.add(voice_script)
        await session.flush()
        session.add_all([
            VoiceMixArtifact(
                voice_script_id=voice_script.id,
                scene_id=scene_1.id,
                artifact_type="SCENE_VOICE_STEM",
                audio_path=stem_key_1,
                duration_seconds=1.4,
                status="READY",
            ),
            VoiceMixArtifact(
                voice_script_id=voice_script.id,
                scene_id=scene_2.id,
                artifact_type="SCENE_VOICE_STEM",
                audio_path=stem_key_2,
                duration_seconds=1.4,
                status="READY",
            ),
        ])
        await session.commit()

        result = await service.render_master(session, production.id)

    master_path = storage.resolve_local_path(result.video_path)
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 2.7
    assert _probe_duration(master_path) >= 2.7

    boundary_tail = _mean_volume_segment(master_path, 1.10, 0.20)
    final_tail = _mean_volume_segment(master_path, 2.35, 0.20)
    assert "mean_volume: -inf" not in boundary_tail
    assert "max_volume: -inf" not in boundary_tail
    assert "mean_volume: -inf" not in final_tail
    assert "max_volume: -inf" not in final_tail
