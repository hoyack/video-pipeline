"""Tests for Phase 28 voice script, TTS, mix, and lip-sync services."""

import math
import shutil
import struct
import subprocess
import uuid
import wave
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.config import settings
from vidpipe.db.models import (
    Actor,
    ActorVoiceProfile,
    Base,
    CastBinding,
    LipSyncJob,
    Production,
    ProductionBible,
    Scene,
    Screenplay,
    Shot,
    VideoClip,
    VoiceLine,
    VoiceScript,
)
from vidpipe.services.audio.base import AudioAdapter, VoiceProfileInfo
from vidpipe.services.storage_backend import get_storage_backend, reset_storage_backend
from vidpipe.services.voice_script_service import VoiceScriptService


class FakeLLMAdapter:
    async def generate_text(self, prompt, schema, **kwargs):
        assert "Production Bible cast bindings" in prompt
        return schema.model_validate({
            "lines": [
                {
                    "scene_number": 1,
                    "shot_number": 1,
                    "line_type": "DIALOGUE",
                    "speaker_tag": "HERO",
                    "speaker_name": "Hero",
                    "text": "We have one shot.",
                    "delivery_notes": "steady",
                    "timing_hint": "after reveal",
                    "lip_sync_mode": "AUTO",
                }
            ]
        })


class FakeAudioAdapter(AudioAdapter):
    async def generate_voice(self, voice_id: str, text: str, **kwargs) -> bytes:
        assert voice_id == "voice-hero"
        assert text == "We have one shot."
        return b"fake-mp3"

    async def generate_sfx(self, prompt: str, **kwargs) -> bytes:
        raise NotImplementedError

    async def list_voices(self, **kwargs) -> list[VoiceProfileInfo]:
        return []

    async def get_voice(self, voice_id: str) -> VoiceProfileInfo:
        raise NotImplementedError


class FailingAudioAdapter(AudioAdapter):
    async def generate_voice(self, voice_id: str, text: str, **kwargs) -> bytes:
        raise RuntimeError("provider rejected voice id")

    async def generate_sfx(self, prompt: str, **kwargs) -> bytes:
        raise NotImplementedError

    async def list_voices(self, **kwargs) -> list[VoiceProfileInfo]:
        return []

    async def get_voice(self, voice_id: str) -> VoiceProfileInfo:
        raise NotImplementedError


def _write_tone_mp3(path: Path, duration_seconds: float = 0.5) -> bytes:
    """Create a small non-silent MP3 fixture that browser audio controls can decode."""
    sample_rate = 44_100
    wav_path = path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        total_samples = int(sample_rate * duration_seconds)
        for index in range(total_samples):
            sample = int(12_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav_file.writeframes(struct.pack("<h", sample))

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to build MP3 voice fixtures")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path.read_bytes()


def _assert_non_silent_audio(path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to inspect MP3 voice fixtures")
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
    assert "max_volume: -" in result.stderr
    assert "max_volume: -inf" not in result.stderr


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


async def _seed_voice_production(session):
    bible = ProductionBible(name="Voice Bible")
    actor = Actor(name="Actor Hero")
    session.add_all([bible, actor])
    await session.flush()

    production = Production(name="Voice Test", production_bible_id=bible.id)
    voice_profile = ActorVoiceProfile(actor_id=actor.id, voice_id="voice-hero", adapter_type="ELEVENLABS")
    session.add_all([production, voice_profile])
    await session.flush()

    binding = CastBinding(
        production_bible_id=bible.id,
        actor_id=actor.id,
        voice_profile_id=voice_profile.id,
        tag="HERO",
        character_name="Hero",
        role="LEAD",
    )
    screenplay = Screenplay(
        production_id=production.id,
        title="Voice Test",
        logline="A test hero speaks.",
        script="HERO: We have one shot.",
        shot_list=[{"scene_number": 1, "shot_number": 1}],
    )
    session.add_all([binding, screenplay])
    await session.flush()
    return production


@pytest.mark.asyncio
async def test_generate_from_screenplay_creates_bound_voice_line(session_factory):
    service = VoiceScriptService()
    async with session_factory() as session:
        production = await _seed_voice_production(session)
        await session.commit()

        voice_script = await service.generate_from_screenplay(
            session,
            production.id,
            FakeLLMAdapter(),
            text_model="test-model",
        )

        line = (
            await session.execute(select(VoiceLine).where(VoiceLine.voice_script_id == voice_script.id))
        ).scalar_one()
        assert line.speaker_tag == "HERO"
        assert line.voice_id == "voice-hero"
        assert line.cast_binding_id is not None
        assert line.generation_status == "PENDING"


@pytest.mark.asyncio
async def test_generate_line_audio_stores_audio_asset(session_factory):
    service = VoiceScriptService()
    async with session_factory() as session:
        production = await _seed_voice_production(session)
        await service.generate_from_screenplay(session, production.id, FakeLLMAdapter())
        line = (await session.execute(select(VoiceLine))).scalar_one()

        updated = await service.generate_line_audio(
            session,
            line.id,
            audio_adapter=FakeAudioAdapter(),
        )

        assert updated.generation_status == "READY"
        assert updated.audio_path is not None
        storage = get_storage_backend()
        assert await storage.exists(updated.audio_path)


@pytest.mark.asyncio
async def test_generate_line_audio_retains_existing_audio_on_regeneration_failure(session_factory):
    service = VoiceScriptService()
    async with session_factory() as session:
        production = await _seed_voice_production(session)
        await service.generate_from_screenplay(session, production.id, FakeLLMAdapter())
        line = (await session.execute(select(VoiceLine))).scalar_one()
        line.audio_path = f"productions/{production.id}/voice/test-existing.mp3"
        line.duration_seconds = 1.5
        line.generation_status = "READY"
        await session.commit()

        updated = await service.generate_line_audio(
            session,
            line.id,
            audio_adapter=FailingAudioAdapter(),
        )

        assert updated.generation_status == "READY"
        assert updated.audio_path == line.audio_path
        assert updated.duration_seconds == 1.5
        assert updated.error_message is not None
        assert "existing audio retained" in updated.error_message


@pytest.mark.asyncio
async def test_mix_and_fake_lip_sync_create_ui_visible_artifacts(session_factory):
    service = VoiceScriptService()
    async with session_factory() as session:
        production = await _seed_voice_production(session)
        scene = Scene(
            production_id=production.id,
            production_bible_id=production.production_bible_id,
            screenplay_breakdown_index=0,
            title="Scene 1",
            prompt="A test scene.",
            style="cinematic",
            aspect_ratio="16:9",
            target_clip_duration=4,
            target_shot_count=1,
            status="complete",
        )
        session.add(scene)
        await session.flush()

        shot = Shot(
            scene_id=scene.id,
            shot_index=0,
            shot_description="Hero speaks",
            start_frame_prompt="start",
            end_frame_prompt="end",
            video_motion_prompt="speak",
            status="complete",
        )
        session.add(shot)
        await session.flush()

        clip = VideoClip(
            shot_id=shot.id,
            status="complete",
            local_path=f"{scene.id}/clips/shot_0.mp4",
        )
        session.add(clip)
        await session.flush()

        storage = get_storage_backend()
        await storage.put(clip.local_path, b"fake-video", "video/mp4")

        voice_script = VoiceScript(
            screenplay_id=(await session.execute(select(Screenplay))).scalar_one().id,
            production_id=production.id,
        )
        session.add(voice_script)
        await session.flush()
        line = VoiceLine(
            voice_script_id=voice_script.id,
            production_id=production.id,
            scene_number=1,
            scene_id=scene.id,
            shot_number=1,
            shot_id=shot.id,
            line_index=0,
            line_type="DIALOGUE",
            speaker_tag="HERO",
            speaker_name="Hero",
            voice_id="voice-hero",
            text="We have one shot.",
            audio_path=f"productions/{production.id}/voice/{voice_script.id}/lines/{uuid.uuid4()}.mp3",
            generation_status="READY",
        )
        session.add(line)
        await session.flush()
        fixture_audio_path = settings.storage.tmp_dir / "audible-line.mp3"
        await storage.put(line.audio_path, _write_tone_mp3(fixture_audio_path), "audio/mpeg")

        artifacts = await service.build_mix_artifacts(session, voice_script.id)
        jobs = await service.queue_lip_sync_jobs(session, voice_script.id, adapter_type="FAKE")

        assert len(artifacts) == 1
        assert artifacts[0].status == "READY"
        assert artifacts[0].audio_path is not None
        _assert_non_silent_audio(settings.storage.tmp_dir / artifacts[0].audio_path)
        assert len(jobs) == 1
        assert jobs[0].status == "READY"
        assert jobs[0].output_clip_path is not None
        assert jobs[0].completed_at is not None
        assert jobs[0].completed_at.tzinfo is None
        assert (await session.execute(select(LipSyncJob))).scalars().first() is not None
