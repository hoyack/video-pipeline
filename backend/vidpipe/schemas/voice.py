"""Voice script, TTS, mix, and lip-sync API schemas."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


VoiceLineType = Literal["NARRATION", "DIALOGUE"]
VoiceLineStatus = Literal["PENDING", "GENERATING", "READY", "FAILED", "SKIPPED"]
LipSyncMode = Literal["AUTO", "ALWAYS", "NEVER"]


class GeneratedVoiceLine(BaseModel):
    """Structured line returned by the screenplay-to-voice LLM prompt."""

    scene_number: Optional[int] = None
    shot_number: Optional[int] = None
    line_type: VoiceLineType = "DIALOGUE"
    speaker_tag: Optional[str] = None
    speaker_name: Optional[str] = None
    text: str = Field(min_length=1)
    delivery_notes: Optional[str] = None
    timing_hint: Optional[str] = None
    lip_sync_mode: LipSyncMode = "AUTO"


class GeneratedVoiceScript(BaseModel):
    """LLM output schema for generated voice scripts."""

    lines: list[GeneratedVoiceLine]


class GenerateVoiceScriptRequest(BaseModel):
    text_model: Optional[str] = None


class VoiceLineUpdate(BaseModel):
    scene_number: Optional[int] = None
    shot_number: Optional[int] = None
    line_type: Optional[VoiceLineType] = None
    speaker_tag: Optional[str] = None
    speaker_name: Optional[str] = None
    cast_binding_id: Optional[str] = None
    actor_voice_profile_id: Optional[str] = None
    character_voice_profile_id: Optional[str] = None
    voice_id: Optional[str] = None
    adapter_type: Optional[str] = None
    text: Optional[str] = None
    delivery_notes: Optional[str] = None
    timing_hint: Optional[str] = None
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None
    lip_sync_mode: Optional[LipSyncMode] = None


class VoiceLineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    voice_script_id: str
    production_id: str
    scene_number: Optional[int] = None
    scene_id: Optional[str] = None
    shot_number: Optional[int] = None
    shot_id: Optional[str] = None
    line_index: int
    line_type: str
    speaker_tag: Optional[str] = None
    speaker_name: Optional[str] = None
    cast_binding_id: Optional[str] = None
    actor_voice_profile_id: Optional[str] = None
    character_voice_profile_id: Optional[str] = None
    voice_id: Optional[str] = None
    adapter_type: str
    text: str
    delivery_notes: Optional[str] = None
    timing_hint: Optional[str] = None
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    audio_mime_type: str
    generation_status: str
    lip_sync_mode: str
    lip_sync_status: Optional[str] = None
    provider_metadata: Optional[dict] = None
    error_message: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class VoiceMixArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    voice_script_id: str
    scene_id: Optional[str] = None
    shot_id: Optional[str] = None
    artifact_type: str
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    video_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class LipSyncJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    voice_line_id: Optional[str] = None
    shot_id: str
    input_clip_id: Optional[str] = None
    input_audio_path: str
    output_clip_path: Optional[str] = None
    adapter_type: str
    status: str
    eligibility_reason: Optional[str] = None
    metrics_json: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class VoiceScriptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    screenplay_id: str
    production_id: str
    status: str
    version: int
    script_model: Optional[str] = None
    source_screenplay_updated_at: Optional[datetime] = None
    voice_generation_status: Optional[str] = None
    mix_status: Optional[str] = None
    lip_sync_status: Optional[str] = None
    error_message: Optional[str] = None
    lines: list[VoiceLineResponse] = Field(default_factory=list)
    mix_artifacts: list[VoiceMixArtifactResponse] = Field(default_factory=list)
    lip_sync_jobs: list[LipSyncJobResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class VoiceScriptActionResponse(BaseModel):
    voice_script: VoiceScriptResponse
    message: str
