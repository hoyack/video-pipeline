"""Sound deck schemas for production SFX cues and mix artifacts."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SoundCueType = Literal["SFX", "AMBIENCE", "FOLEY", "UI", "IMPACT", "MECHANICAL", "NATURAL"]
SoundCueStatus = Literal["PENDING", "GENERATING", "READY", "FAILED", "SKIPPED"]


class GeneratedSoundCue(BaseModel):
    scene_number: Optional[int] = None
    shot_number: Optional[int] = None
    cue_type: SoundCueType = "SFX"
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    timing_hint: Optional[str] = None
    start_time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    volume_db: Optional[float] = None


class GeneratedSoundDeck(BaseModel):
    cues: list[GeneratedSoundCue]


class GenerateSoundDeckRequest(BaseModel):
    text_model: Optional[str] = None


class SoundEffectCueUpdate(BaseModel):
    scene_number: Optional[int] = None
    shot_number: Optional[int] = None
    cue_type: Optional[SoundCueType] = None
    name: Optional[str] = None
    prompt: Optional[str] = None
    timing_hint: Optional[str] = None
    start_time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    volume_db: Optional[float] = None
    sound_asset_id: Optional[str] = None
    sfx_item_id: Optional[str] = None


class SoundEffectCueResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    production_id: str
    scene_id: Optional[str] = None
    shot_id: Optional[str] = None
    scene_number: Optional[int] = None
    shot_number: Optional[int] = None
    cue_index: int
    cue_type: str
    name: str
    prompt: str
    timing_hint: Optional[str] = None
    start_time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    volume_db: Optional[float] = None
    sound_asset_id: Optional[str] = None
    sfx_item_id: Optional[str] = None
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    audio_mime_type: str
    generation_status: str
    provider_metadata: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SoundMixArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    production_id: str
    scene_id: Optional[str] = None
    artifact_type: str
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class SoundDeckResponse(BaseModel):
    production_id: str
    cues: list[SoundEffectCueResponse] = Field(default_factory=list)
    mix_artifacts: list[SoundMixArtifactResponse] = Field(default_factory=list)


class SoundDeckActionResponse(BaseModel):
    sound_deck: SoundDeckResponse
    message: str
