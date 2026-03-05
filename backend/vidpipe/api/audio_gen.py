"""Audio generation API routes.

Endpoints for ElevenLabs voice search/resolve, TTS voice sample generation,
and SFX audio generation. All endpoints require an ElevenLabs API key
configured in UserSettings.
"""

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from vidpipe.db import async_session
from vidpipe.db.models import (
    Character,
    SFXItem,
    UserSettings,
    VoiceProfile,
    DEFAULT_USER_ID,
)
from vidpipe.services.audio.base import (
    AudioAdapterError,
    AudioAuthError,
    AudioQuotaError,
)
from vidpipe.services.audio.registry import get_audio_adapter
from vidpipe.services.storage_backend import get_storage_backend, LocalStorageBackend

logger = logging.getLogger(__name__)

audio_gen_router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class GenerateVoiceSampleRequest(BaseModel):
    text: str = "Hello, this is a sample of my voice. How does it sound?"


class GenerateSFXRequest(BaseModel):
    pass  # uses SFXItem.generation_prompt from DB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_elevenlabs_key(session) -> str:
    """Load ElevenLabs API key from UserSettings. Raises 422 if not configured."""
    result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    settings = result.scalars().first()
    if settings is None or not settings.elevenlabs_api_key:
        raise HTTPException(
            status_code=422,
            detail="ElevenLabs API key not configured. Set it in Settings.",
        )
    return settings.elevenlabs_api_key


def _handle_adapter_error(e: AudioAdapterError) -> None:
    """Convert adapter errors to appropriate HTTP responses."""
    if isinstance(e, AudioAuthError):
        raise HTTPException(status_code=401, detail=str(e))
    if isinstance(e, AudioQuotaError):
        raise HTTPException(status_code=429, detail=str(e))
    raise HTTPException(status_code=502, detail=str(e))


async def _store_audio(
    content: bytes,
    storage_prefix: str,
    filename: str,
) -> str:
    """Store generated audio bytes and return the path/key."""
    storage = get_storage_backend()

    if isinstance(storage, LocalStorageBackend):
        from vidpipe.config import settings as _settings

        local_dir = _settings.storage.tmp_dir / storage_prefix
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename
        await asyncio.to_thread(local_path.write_bytes, content)
        return str(local_path)
    else:
        key = f"{storage_prefix}/{filename}"
        await storage.put(key, content, "audio/mpeg")
        from vidpipe.config import settings as _settings

        local_path = _settings.storage.tmp_dir / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(local_path.write_bytes, content)
        return key


# ---------------------------------------------------------------------------
# Voice search/resolve endpoints
# ---------------------------------------------------------------------------


@audio_gen_router.get("/elevenlabs/voices")
async def search_elevenlabs_voices(search: Optional[str] = None):
    """Search ElevenLabs voices by name, or list all if no search param."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)

    adapter = get_audio_adapter("ELEVENLABS", api_key)
    try:
        voices = await adapter.list_voices(search=search)
        return [v.model_dump() for v in voices]
    except AudioAdapterError as e:
        _handle_adapter_error(e)


@audio_gen_router.get("/elevenlabs/voices/{voice_id}")
async def resolve_elevenlabs_voice(voice_id: str):
    """Resolve a voice_id to name + labels."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)

    adapter = get_audio_adapter("ELEVENLABS", api_key)
    try:
        voice = await adapter.get_voice(voice_id)
        return voice.model_dump()
    except AudioAdapterError as e:
        _handle_adapter_error(e)


# ---------------------------------------------------------------------------
# Voice sample generation
# ---------------------------------------------------------------------------


@audio_gen_router.post("/characters/{character_id}/generate-voice-sample")
async def generate_voice_sample(character_id: str, body: GenerateVoiceSampleRequest):
    """Generate a TTS voice sample for a character's voice profile."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)

        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        result = await session.execute(
            select(VoiceProfile).where(
                VoiceProfile.character_id == uuid.UUID(character_id)
            )
        )
        vp = result.scalars().first()
        if vp is None:
            raise HTTPException(
                status_code=404, detail="Voice profile not found. Create one first."
            )
        if not vp.voice_id:
            raise HTTPException(
                status_code=422, detail="Voice ID not set on voice profile."
            )

        adapter = get_audio_adapter(vp.adapter_type, api_key)
        try:
            audio_bytes = await adapter.generate_voice(
                vp.voice_id,
                body.text,
                style_notes=vp.style_notes,
            )
        except AudioAdapterError as e:
            _handle_adapter_error(e)

        # Store audio
        storage_prefix = (
            f"manifests/{char.production_bible_id}/characters/{char.id}/voice_samples"
        )
        stored_path = await _store_audio(audio_bytes, storage_prefix, "sample.mp3")

        vp.sample_audio = stored_path
        await session.commit()
        await session.refresh(vp)

        return {
            "voice_profile_id": str(vp.id),
            "character_id": str(vp.character_id),
            "voice_id": vp.voice_id,
            "adapter_type": vp.adapter_type,
            "style_notes": vp.style_notes,
            "sample_audio": vp.sample_audio,
            "created_at": vp.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# SFX generation
# ---------------------------------------------------------------------------


@audio_gen_router.post("/sfx/{sfx_item_id}/generate")
async def generate_sfx(sfx_item_id: str):
    """Generate SFX audio from an SFX item's generation_prompt."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)

        item = await session.get(SFXItem, uuid.UUID(sfx_item_id))
        if item is None:
            raise HTTPException(status_code=404, detail="SFX item not found")
        if not item.generation_prompt:
            raise HTTPException(
                status_code=422,
                detail="SFX item has no generation_prompt set.",
            )

        adapter = get_audio_adapter("ELEVENLABS", api_key)
        try:
            audio_bytes = await adapter.generate_sfx(item.generation_prompt)
        except AudioAdapterError as e:
            _handle_adapter_error(e)

        # Store audio
        storage_prefix = (
            f"manifests/{item.production_bible_id}/sfx/{item.id}"
        )
        stored_path = await _store_audio(audio_bytes, storage_prefix, "generated.mp3")

        item.source_audio = stored_path
        await session.commit()
        await session.refresh(item)

        return {
            "sfx_item_id": str(item.id),
            "production_bible_id": str(item.production_bible_id),
            "name": item.name,
            "category": item.category,
            "source_audio": item.source_audio,
            "generation_prompt": item.generation_prompt,
            "tags": item.tags,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }
