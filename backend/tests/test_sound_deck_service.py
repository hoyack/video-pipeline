"""Tests for production sound-deck cue generation, SFX audio, and mixing."""

import shutil
import subprocess
import re
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.config import settings
from vidpipe.db.models import Base, Production, Scene, Shot, SoundEffectCue
from vidpipe.services.audio.base import AudioAdapter, VoiceProfileInfo
from vidpipe.services.sound_deck_service import SoundDeckService
from vidpipe.services.storage_backend import get_storage_backend, reset_storage_backend


class FakeSoundDeckLLM:
    async def generate_text(self, prompt, schema, **kwargs):
        assert "Scenes and shots" in prompt
        return schema.model_validate({
            "cues": [
                {
                    "scene_number": 1,
                    "shot_number": 1,
                    "cue_type": "AMBIENCE",
                    "name": "Control room tone",
                    "prompt": "Subtle NASA mission control room tone, distant ventilation, soft console beeps, no speech.",
                    "timing_hint": "throughout opening shot",
                    "start_time_seconds": 0.0,
                    "duration_seconds": 0.5,
                    "volume_db": -24.0,
                },
                {
                    "scene_number": 1,
                    "shot_number": 2,
                    "cue_type": "MECHANICAL",
                    "name": "Checklist switch",
                    "prompt": "Single crisp analog switch click in a quiet spacecraft checklist environment.",
                    "timing_hint": "start of second shot",
                    "start_time_seconds": 2.0,
                    "duration_seconds": 1.0,
                    "volume_db": -16.0,
                },
            ]
        })


class FakeSFXAdapter(AudioAdapter):
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, float | None]] = []

    async def generate_sfx(self, prompt: str, *, duration_seconds: float | None = None) -> bytes:
        self.calls.append((prompt, duration_seconds))
        path = self.tmp_path / f"sfx_{len(self.calls)}.mp3"
        duration = duration_seconds or 1.0
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={600 + len(self.calls) * 100}:duration={duration}",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return path.read_bytes()

    async def generate_voice(self, voice_id: str, text: str, **kwargs) -> bytes:
        raise AssertionError("voice generation should not be used for SFX")

    async def list_voices(self, **kwargs) -> list[VoiceProfileInfo]:
        return []

    async def get_voice(self, voice_id: str) -> VoiceProfileInfo:
        raise AssertionError("voice lookup should not be used for SFX")


def _volume_segment(path: Path, start_seconds: float, duration_seconds: float) -> str:
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


def _max_volume_db(volume_output: str) -> float:
    match = re.search(r"max_volume:\s+(-?inf|-?\d+(?:\.\d+)?) dB", volume_output)
    assert match, volume_output
    if match.group(1) == "-inf":
        return float("-inf")
    return float(match.group(1))


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


async def _seed_production(session):
    production = Production(name="NASA Sound Deck")
    session.add(production)
    await session.flush()
    scene = Scene(
        production_id=production.id,
        title="Mission Control",
        prompt="Mission control consoles during Gemini.",
        style="documentary",
        aspect_ratio="16:9",
        target_clip_duration=2,
        target_shot_count=2,
        status="complete",
        scene_order=1,
        screenplay_breakdown_index=1,
    )
    session.add(scene)
    await session.flush()
    session.add_all([
        Shot(
            scene_id=scene.id,
            shot_index=0,
            shot_description="Controllers watch tracking displays.",
            start_frame_prompt="start",
            end_frame_prompt="end",
            video_motion_prompt="slow push over consoles",
            status="video_done",
        ),
        Shot(
            scene_id=scene.id,
            shot_index=1,
            shot_description="A hand checks a spacecraft procedure.",
            start_frame_prompt="start",
            end_frame_prompt="end",
            video_motion_prompt="close detail of checklist",
            status="video_done",
        ),
    ])
    await session.commit()
    return production


@pytest.mark.asyncio
async def test_sound_deck_generates_audio_and_scene_stem(session_factory, tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for sound deck tests")

    service = SoundDeckService()
    storage = get_storage_backend()
    async with session_factory() as session:
        production = await _seed_production(session)

        cues = await service.generate_from_production(
            session,
            production.id,
            FakeSoundDeckLLM(),
            text_model="test-model",
        )
        assert len(cues) == 2
        assert cues[0].scene_id is not None
        assert cues[0].shot_id is not None
        assert cues[1].start_time_seconds == 2.0
        assert cues[0].duration_seconds == 0.5

        adapter = FakeSFXAdapter(tmp_path)
        generated = await service.generate_pending_audio(
            session,
            production.id,
            audio_adapter=adapter,
        )
        assert len(generated) == 2
        assert len(adapter.calls) == 2

        rows = list((await session.execute(select(SoundEffectCue))).scalars().all())
        assert all(cue.generation_status == "READY" for cue in rows)
        for cue in rows:
            assert cue.audio_path is not None
            assert await storage.exists(cue.audio_path)

        artifacts = await service.build_mix_artifacts(session, production.id)
        assert len(artifacts) == 1
        assert artifacts[0].status == "READY"
        assert artifacts[0].audio_path is not None
        assert await storage.exists(artifacts[0].audio_path)

        stem_path = storage.resolve_local_path(artifacts[0].audio_path)
        assert artifacts[0].duration_seconds is not None
        assert artifacts[0].duration_seconds >= 3.0
        first_cue = _volume_segment(stem_path, 0.10, 0.25)
        gap_between_cues = _volume_segment(stem_path, 1.20, 0.25)
        delayed_cue = _volume_segment(stem_path, 2.10, 0.25)
        assert _max_volume_db(first_cue) > -80.0
        assert _max_volume_db(gap_between_cues) < -80.0
        assert _max_volume_db(delayed_cue) > -80.0
