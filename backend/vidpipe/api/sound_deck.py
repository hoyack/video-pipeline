"""Production sound-deck API routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy import select

from vidpipe.config import settings
from vidpipe.db import async_session
from vidpipe.db.models import DEFAULT_USER_ID, SoundEffectCue, SoundMixArtifact, UserSettings
from vidpipe.schemas.sound_deck import (
    GenerateSoundDeckRequest,
    SoundDeckActionResponse,
    SoundDeckResponse,
    SoundEffectCueResponse,
    SoundEffectCueUpdate,
    SoundMixArtifactResponse,
)
from vidpipe.services.audio.base import AudioAdapterError, AudioAuthError, AudioQuotaError
from vidpipe.services.audio.registry import get_audio_adapter
from vidpipe.services.llm import get_adapter
from vidpipe.services.sound_deck_service import SoundDeckService
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend

sound_deck_router = APIRouter(prefix="/api")
sound_deck_service = SoundDeckService()


@sound_deck_router.get(
    "/productions/{production_id}/sound-deck",
    response_model=SoundDeckResponse,
)
async def get_sound_deck(production_id: str):
    """List production SFX cues and mix artifacts."""
    async with async_session() as session:
        return await _sound_deck_response(session, uuid.UUID(production_id))


@sound_deck_router.post(
    "/productions/{production_id}/sound-deck/generate",
    response_model=SoundDeckActionResponse,
)
async def generate_sound_deck(production_id: str, request: GenerateSoundDeckRequest):
    """Generate editable sound-effect cues from production scenes and bible sound assets."""
    async with async_session() as session:
        user_settings = await _get_user_settings(session)
        model_id = request.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        try:
            await sound_deck_service.generate_from_production(
                session,
                uuid.UUID(production_id),
                adapter,
                text_model=model_id,
                include_music=request.include_music,
                music_prompt=request.music_prompt,
            )
            return SoundDeckActionResponse(
                sound_deck=await _sound_deck_response(session, uuid.UUID(production_id)),
                message="Sound deck generated",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@sound_deck_router.patch(
    "/sound-cues/{cue_id}",
    response_model=SoundEffectCueResponse,
)
async def update_sound_cue(cue_id: str, request: SoundEffectCueUpdate):
    """Patch one editable sound cue."""
    async with async_session() as session:
        try:
            cue = await sound_deck_service.update_cue(
                session,
                uuid.UUID(cue_id),
                {field: getattr(request, field) for field in request.model_fields_set},
            )
            return _cue_response(cue)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@sound_deck_router.delete("/sound-cues/{cue_id}", status_code=204)
async def delete_sound_cue(cue_id: str):
    """Delete one sound cue."""
    async with async_session() as session:
        try:
            await sound_deck_service.delete_cue(session, uuid.UUID(cue_id))
            return Response(status_code=204)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@sound_deck_router.post(
    "/sound-cues/{cue_id}/generate-audio",
    response_model=SoundEffectCueResponse,
)
async def generate_sound_cue_audio(cue_id: str):
    """Generate ElevenLabs SFX audio for one cue."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)
        try:
            cue = await sound_deck_service.generate_cue_audio(
                session,
                uuid.UUID(cue_id),
                api_key=api_key,
            )
            return _cue_response(cue)
        except AudioAdapterError as exc:
            _handle_adapter_error(exc)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@sound_deck_router.post(
    "/productions/{production_id}/sound-deck/generate-audio",
    response_model=SoundDeckActionResponse,
)
async def generate_sound_deck_audio(production_id: str):
    """Generate ElevenLabs SFX audio for all pending/failed cues."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)
        adapter = get_audio_adapter("ELEVENLABS", api_key)
        try:
            await sound_deck_service.generate_pending_audio(
                session,
                uuid.UUID(production_id),
                audio_adapter=adapter,
                api_key=api_key,
            )
            return SoundDeckActionResponse(
                sound_deck=await _sound_deck_response(session, uuid.UUID(production_id)),
                message="Sound cue audio generated",
            )
        except AudioAdapterError as exc:
            _handle_adapter_error(exc)


@sound_deck_router.post(
    "/productions/{production_id}/sound-deck/mix",
    response_model=SoundDeckActionResponse,
)
async def mix_sound_deck(production_id: str):
    """Build scene SFX stems from generated cue audio."""
    async with async_session() as session:
        try:
            await sound_deck_service.build_mix_artifacts(session, uuid.UUID(production_id))
            return SoundDeckActionResponse(
                sound_deck=await _sound_deck_response(session, uuid.UUID(production_id)),
                message="Sound mix artifacts updated",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@sound_deck_router.get("/sound-cues/{cue_id}/audio")
async def get_sound_cue_audio(cue_id: str):
    """Serve generated cue audio."""
    async with async_session() as session:
        cue = await session.get(SoundEffectCue, uuid.UUID(cue_id))
        if cue is None or not cue.audio_path:
            raise HTTPException(status_code=404, detail="Sound cue audio not found")
        return await _audio_response(cue.audio_path, cue.audio_mime_type)


@sound_deck_router.get("/sound-mix-artifacts/{artifact_id}/audio")
async def get_sound_mix_audio(artifact_id: str):
    """Serve generated sound stem audio."""
    async with async_session() as session:
        artifact = await session.get(SoundMixArtifact, uuid.UUID(artifact_id))
        if artifact is None or not artifact.audio_path:
            raise HTTPException(status_code=404, detail="Sound mix audio not found")
        return await _audio_response(artifact.audio_path, "audio/mp4")


async def _sound_deck_response(session, production_id: uuid.UUID) -> SoundDeckResponse:
    cues = await sound_deck_service.list_cues(session, production_id)
    artifacts = await sound_deck_service.list_mix_artifacts(session, production_id)
    return SoundDeckResponse(
        production_id=str(production_id),
        cues=[_cue_response(cue) for cue in cues],
        mix_artifacts=[_mix_artifact_response(artifact) for artifact in artifacts],
    )


def _cue_response(cue: SoundEffectCue) -> SoundEffectCueResponse:
    return SoundEffectCueResponse(
        id=str(cue.id),
        production_id=str(cue.production_id),
        scene_id=str(cue.scene_id) if cue.scene_id else None,
        shot_id=str(cue.shot_id) if cue.shot_id else None,
        scene_number=cue.scene_number,
        shot_number=cue.shot_number,
        cue_index=cue.cue_index,
        cue_type=cue.cue_type,
        name=cue.name,
        prompt=cue.prompt,
        timing_hint=cue.timing_hint,
        start_time_seconds=cue.start_time_seconds,
        duration_seconds=cue.duration_seconds,
        volume_db=cue.volume_db,
        sound_asset_id=str(cue.sound_asset_id) if cue.sound_asset_id else None,
        sfx_item_id=str(cue.sfx_item_id) if cue.sfx_item_id else None,
        audio_path=cue.audio_path,
        audio_url=f"/api/sound-cues/{cue.id}/audio" if cue.audio_path else None,
        audio_mime_type=cue.audio_mime_type,
        generation_status=cue.generation_status,
        provider_metadata=cue.provider_metadata,
        error_message=cue.error_message,
        created_at=cue.created_at,
        updated_at=cue.updated_at,
    )


def _mix_artifact_response(artifact: SoundMixArtifact) -> SoundMixArtifactResponse:
    return SoundMixArtifactResponse(
        id=str(artifact.id),
        production_id=str(artifact.production_id),
        scene_id=str(artifact.scene_id) if artifact.scene_id else None,
        artifact_type=artifact.artifact_type,
        audio_path=artifact.audio_path,
        audio_url=f"/api/sound-mix-artifacts/{artifact.id}/audio" if artifact.audio_path else None,
        duration_seconds=artifact.duration_seconds,
        status=artifact.status,
        error_message=artifact.error_message,
        created_at=artifact.created_at,
    )


async def _audio_response(audio_path: str, media_type: str):
    storage = get_storage_backend()
    if isinstance(storage, LocalStorageBackend):
        path = storage.resolve_local_path(audio_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(Path(path), media_type=media_type)
    try:
        return Response(await storage.get(audio_path), media_type=media_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Audio file not found") from exc


async def _get_user_settings(session) -> UserSettings | None:
    result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    return result.scalar_one_or_none()


async def _get_elevenlabs_key(session) -> str:
    user_settings = await _get_user_settings(session)
    if user_settings is None or not user_settings.elevenlabs_api_key:
        raise HTTPException(
            status_code=422,
            detail="ElevenLabs API key not configured. Set it in Settings.",
        )
    return user_settings.elevenlabs_api_key


def _handle_adapter_error(exc: AudioAdapterError) -> None:
    if isinstance(exc, AudioAuthError):
        raise HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, AudioQuotaError):
        raise HTTPException(status_code=429, detail=str(exc))
    raise HTTPException(status_code=502, detail=str(exc))
