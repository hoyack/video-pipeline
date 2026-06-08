"""Voice script, TTS, mixing, and lip-sync API routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy import select

from vidpipe.config import settings
from vidpipe.db import async_session
from vidpipe.db.models import DEFAULT_USER_ID, UserSettings, VoiceLine, VoiceMixArtifact
from vidpipe.schemas.voice import (
    GenerateVoiceScriptRequest,
    LipSyncJobResponse,
    VoiceLineResponse,
    VoiceLineUpdate,
    VoiceMixArtifactResponse,
    VoiceScriptActionResponse,
    VoiceScriptResponse,
)
from vidpipe.services.audio.base import AudioAdapterError
from vidpipe.services.audio.registry import get_audio_adapter
from vidpipe.services.llm import get_adapter
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend
from vidpipe.services.voice_script_service import VoiceScriptService

voice_script_router = APIRouter(prefix="/api")
voice_service = VoiceScriptService()


@voice_script_router.get(
    "/productions/{production_id}/voice-script",
    response_model=VoiceScriptResponse,
)
async def get_voice_script(production_id: str):
    """Get or create a production's voice script."""
    async with async_session() as session:
        try:
            voice_script = await voice_service.get_or_create_script(
                session,
                uuid.UUID(production_id),
            )
            await session.commit()
            return await _voice_script_response(session, voice_script.id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.post(
    "/productions/{production_id}/voice-script/generate",
    response_model=VoiceScriptActionResponse,
)
async def generate_voice_script(production_id: str, request: GenerateVoiceScriptRequest):
    """Generate editable narration/dialogue lines from the current screenplay."""
    async with async_session() as session:
        user_settings = await _get_user_settings(session)
        model_id = request.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        try:
            voice_script = await voice_service.generate_from_screenplay(
                session,
                uuid.UUID(production_id),
                adapter,
                text_model=model_id,
            )
            response = await _voice_script_response(session, voice_script.id)
            return VoiceScriptActionResponse(
                voice_script=response,
                message="Voice script generated",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.post(
    "/voice-scripts/{voice_script_id}/resolve-bindings",
    response_model=VoiceScriptActionResponse,
)
async def resolve_voice_bindings(voice_script_id: str):
    """Resolve all voice lines to Production Bible cast/voice profiles."""
    async with async_session() as session:
        try:
            await voice_service.resolve_all_bindings(session, uuid.UUID(voice_script_id))
            await session.commit()
            response = await _voice_script_response(session, uuid.UUID(voice_script_id))
            return VoiceScriptActionResponse(
                voice_script=response,
                message="Voice bindings resolved",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.post(
    "/voice-scripts/{voice_script_id}/generate-audio",
    response_model=VoiceScriptActionResponse,
)
async def generate_voice_script_audio(voice_script_id: str):
    """Generate TTS audio for pending voice lines."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)
        adapter = get_audio_adapter("ELEVENLABS", api_key)
        try:
            await voice_service.generate_pending_audio(
                session,
                uuid.UUID(voice_script_id),
                audio_adapter=adapter,
                api_key=api_key,
            )
            response = await _voice_script_response(session, uuid.UUID(voice_script_id))
            return VoiceScriptActionResponse(
                voice_script=response,
                message="Voice audio generated",
            )
        except AudioAdapterError as exc:
            _handle_adapter_error(exc)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.post(
    "/voice-lines/{voice_line_id}/generate-audio",
    response_model=VoiceLineResponse,
)
async def generate_voice_line_audio(voice_line_id: str):
    """Generate TTS audio for one voice line."""
    async with async_session() as session:
        api_key = await _get_elevenlabs_key(session)
        try:
            line = await voice_service.generate_line_audio(
                session,
                uuid.UUID(voice_line_id),
                api_key=api_key,
            )
            return _voice_line_response(line)
        except AudioAdapterError as exc:
            _handle_adapter_error(exc)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.patch(
    "/voice-lines/{voice_line_id}",
    response_model=VoiceLineResponse,
)
async def update_voice_line(voice_line_id: str, request: VoiceLineUpdate):
    """Patch an editable voice line."""
    async with async_session() as session:
        try:
            line = await voice_service.update_line(
                session,
                uuid.UUID(voice_line_id),
                {
                    field: getattr(request, field)
                    for field in request.model_fields_set
                },
            )
            return _voice_line_response(line)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.delete("/voice-lines/{voice_line_id}", status_code=204)
async def delete_voice_line(voice_line_id: str):
    """Delete one voice line."""
    async with async_session() as session:
        try:
            await voice_service.delete_line(session, uuid.UUID(voice_line_id))
            return Response(status_code=204)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.post(
    "/voice-scripts/{voice_script_id}/mix",
    response_model=VoiceScriptActionResponse,
)
async def mix_voice_script(voice_script_id: str):
    """Build scene-level voice stems from generated TTS line audio."""
    async with async_session() as session:
        try:
            await voice_service.build_mix_artifacts(session, uuid.UUID(voice_script_id))
            response = await _voice_script_response(session, uuid.UUID(voice_script_id))
            return VoiceScriptActionResponse(
                voice_script=response,
                message="Voice mix artifacts updated",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.post(
    "/voice-scripts/{voice_script_id}/lip-sync",
    response_model=VoiceScriptActionResponse,
)
async def lip_sync_voice_script(voice_script_id: str, adapter_type: str = "FAKE"):
    """Create lip-sync jobs for eligible rendered dialogue lines."""
    async with async_session() as session:
        try:
            await voice_service.queue_lip_sync_jobs(
                session,
                uuid.UUID(voice_script_id),
                adapter_type=adapter_type,
                run_now=True,
            )
            response = await _voice_script_response(session, uuid.UUID(voice_script_id))
            return VoiceScriptActionResponse(
                voice_script=response,
                message="Lip-sync jobs updated",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.get(
    "/voice-scripts/{voice_script_id}/jobs",
    response_model=list[LipSyncJobResponse],
)
async def get_voice_script_jobs(voice_script_id: str):
    """List lip-sync jobs for a voice script."""
    async with async_session() as session:
        try:
            return [
                _lip_sync_job_response(job)
                for job in await voice_service.list_lip_sync_jobs(
                    session,
                    uuid.UUID(voice_script_id),
                )
            ]
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@voice_script_router.get("/voice-lines/{voice_line_id}/audio")
async def get_voice_line_audio(voice_line_id: str):
    """Serve generated line audio."""
    async with async_session() as session:
        line = await session.get(VoiceLine, uuid.UUID(voice_line_id))
        if line is None or not line.audio_path:
            raise HTTPException(status_code=404, detail="Voice audio not found")
        return await _serve_storage_path(line.audio_path, line.audio_mime_type)


@voice_script_router.get("/voice-mix-artifacts/{artifact_id}/audio")
async def get_voice_mix_audio(artifact_id: str):
    """Serve generated voice stem audio."""
    async with async_session() as session:
        artifact = await session.get(VoiceMixArtifact, uuid.UUID(artifact_id))
        if artifact is None or not artifact.audio_path:
            raise HTTPException(status_code=404, detail="Voice mix audio not found")
        return await _serve_storage_path(artifact.audio_path, "audio/mpeg")


async def _get_user_settings(session):
    result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    return result.scalar_one_or_none()


async def _get_elevenlabs_key(session) -> str:
    settings_row = await _get_user_settings(session)
    if settings_row is None or not settings_row.elevenlabs_api_key:
        raise HTTPException(
            status_code=422,
            detail="ElevenLabs API key not configured. Set it in Settings.",
        )
    return settings_row.elevenlabs_api_key


def _handle_adapter_error(exc: AudioAdapterError) -> None:
    from vidpipe.services.audio.base import AudioAuthError, AudioQuotaError

    if isinstance(exc, AudioAuthError):
        raise HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, AudioQuotaError):
        raise HTTPException(status_code=429, detail=str(exc))
    raise HTTPException(status_code=502, detail=str(exc))


async def _voice_script_response(session, voice_script_id: uuid.UUID) -> VoiceScriptResponse:
    voice_script = await voice_service.get_script_by_id(session, voice_script_id)
    lines = await voice_service.list_lines(session, voice_script_id)
    artifacts = await voice_service.list_mix_artifacts(session, voice_script_id)
    jobs = await voice_service.list_lip_sync_jobs(session, voice_script_id)
    return VoiceScriptResponse(
        id=str(voice_script.id),
        screenplay_id=str(voice_script.screenplay_id),
        production_id=str(voice_script.production_id),
        status=voice_script.status,
        version=voice_script.version,
        script_model=voice_script.script_model,
        source_screenplay_updated_at=voice_script.source_screenplay_updated_at,
        voice_generation_status=voice_script.voice_generation_status,
        mix_status=voice_script.mix_status,
        lip_sync_status=voice_script.lip_sync_status,
        error_message=voice_script.error_message,
        lines=[_voice_line_response(line) for line in lines],
        mix_artifacts=[_mix_artifact_response(artifact) for artifact in artifacts],
        lip_sync_jobs=[_lip_sync_job_response(job) for job in jobs],
        created_at=voice_script.created_at,
        updated_at=voice_script.updated_at,
    )


def _voice_line_response(line: VoiceLine) -> VoiceLineResponse:
    return VoiceLineResponse(
        id=str(line.id),
        voice_script_id=str(line.voice_script_id),
        production_id=str(line.production_id),
        scene_number=line.scene_number,
        scene_id=str(line.scene_id) if line.scene_id else None,
        shot_number=line.shot_number,
        shot_id=str(line.shot_id) if line.shot_id else None,
        line_index=line.line_index,
        line_type=line.line_type,
        speaker_tag=line.speaker_tag,
        speaker_name=line.speaker_name,
        cast_binding_id=str(line.cast_binding_id) if line.cast_binding_id else None,
        actor_voice_profile_id=str(line.actor_voice_profile_id) if line.actor_voice_profile_id else None,
        character_voice_profile_id=str(line.character_voice_profile_id) if line.character_voice_profile_id else None,
        voice_id=line.voice_id,
        adapter_type=line.adapter_type,
        text=line.text,
        delivery_notes=line.delivery_notes,
        timing_hint=line.timing_hint,
        start_time_seconds=line.start_time_seconds,
        end_time_seconds=line.end_time_seconds,
        duration_seconds=line.duration_seconds,
        audio_path=line.audio_path,
        audio_url=f"/api/voice-lines/{line.id}/audio" if line.audio_path else None,
        audio_mime_type=line.audio_mime_type,
        generation_status=line.generation_status,
        lip_sync_mode=line.lip_sync_mode,
        lip_sync_status=line.lip_sync_status,
        provider_metadata=line.provider_metadata,
        error_message=line.error_message,
        warnings=_line_warnings(line),
        created_at=line.created_at,
        updated_at=line.updated_at,
    )


def _mix_artifact_response(artifact: VoiceMixArtifact) -> VoiceMixArtifactResponse:
    return VoiceMixArtifactResponse(
        id=str(artifact.id),
        voice_script_id=str(artifact.voice_script_id),
        scene_id=str(artifact.scene_id) if artifact.scene_id else None,
        shot_id=str(artifact.shot_id) if artifact.shot_id else None,
        artifact_type=artifact.artifact_type,
        audio_path=artifact.audio_path,
        audio_url=f"/api/voice-mix-artifacts/{artifact.id}/audio" if artifact.audio_path else None,
        video_path=artifact.video_path,
        duration_seconds=artifact.duration_seconds,
        status=artifact.status,
        error_message=artifact.error_message,
        created_at=artifact.created_at,
    )


def _lip_sync_job_response(job) -> LipSyncJobResponse:
    return LipSyncJobResponse(
        id=str(job.id),
        voice_line_id=str(job.voice_line_id) if job.voice_line_id else None,
        shot_id=str(job.shot_id),
        input_clip_id=str(job.input_clip_id) if job.input_clip_id else None,
        input_audio_path=job.input_audio_path,
        output_clip_path=job.output_clip_path,
        adapter_type=job.adapter_type,
        status=job.status,
        eligibility_reason=job.eligibility_reason,
        metrics_json=job.metrics_json,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


def _line_warnings(line: VoiceLine) -> list[str]:
    warnings = []
    if not line.voice_id:
        warnings.append("No voice assigned")
    if line.line_type == "DIALOGUE" and line.lip_sync_mode != "NEVER" and line.shot_id is None:
        warnings.append("No linked shot for lip-sync")
    if line.generation_status == "READY" and line.error_message:
        warnings.append(line.error_message)
    if line.generation_status == "FAILED" and line.error_message:
        warnings.append(line.error_message)
    return warnings


async def _serve_storage_path(path_or_key: str, media_type: str):
    storage = get_storage_backend()
    if isinstance(storage, LocalStorageBackend):
        path = Path(path_or_key)
        if not path.is_absolute():
            path = storage.resolve_local_path(path_or_key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path, media_type=media_type)

    return Response(await storage.get(path_or_key), media_type=media_type)
