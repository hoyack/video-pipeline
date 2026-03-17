"""API route handlers and Pydantic response schemas."""

import asyncio
import json
import logging
import math
import random
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import and_ as sa_and, case, func as sa_func, select, update
from sqlalchemy.orm import selectinload

from vidpipe.db import async_session
from vidpipe.db.models import Scene, Shot, Keyframe, VideoClip, ProductionBible, Asset, ShotManifest as ShotManifestModel, ShotAudioManifest as ShotAudioManifestModel, GenerationCandidate, UserSettings, DEFAULT_USER_ID, SceneCheckpoint, AssetAppearance, Production

# Backwards-compat alias (used in upload-video handler that calls session.get(Manifest, ...))
Manifest = ProductionBible
from vidpipe.orchestrator.pipeline import run_pipeline
from vidpipe.orchestrator.state import can_resume
from vidpipe.schemas.storyboard import ShotSchema
from vidpipe.schemas.storyboard_enhanced import EnhancedShotSchema
from vidpipe.services.file_manager import FileManager
from vidpipe.services.storage_backend import get_storage_backend, LocalStorageBackend, StorageBackend
from vidpipe.services import manifest_service
from vidpipe.workers.processing_tasks import process_manifest_task, extract_video_frames_task, TASK_STATUS

from collections import Counter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Allowed model IDs
# ---------------------------------------------------------------------------
ALLOWED_TEXT_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
}
ALLOWED_IMAGE_MODELS = {
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "qwen-fast",
    "qwen-image-edit",
    "flux-dev",
    "flux-dev-lora",
    "flux-dev-redux",
    "flux-dev-full",
}
ALLOWED_VIDEO_MODELS = {
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-fast-generate-001",
    "wan-2.2-i2v",
}

# Video models that support audio generation
AUDIO_CAPABLE_MODELS = ALLOWED_VIDEO_MODELS - {"veo-2.0-generate-001", "wan-2.2-i2v"}

# Allowed clip durations per video model
ALLOWED_DURATIONS: dict[str, list[int]] = {
    "veo-2.0-generate-001": [5, 6, 7, 8],
    "veo-3.0-generate-001": [4, 6, 8],
    "veo-3.0-fast-generate-001": [4, 6, 8],
    "veo-3.1-generate-preview": [4, 6, 8],
    "veo-3.1-generate-001": [4, 6, 8],
    "veo-3.1-fast-generate-preview": [4, 6, 8],
    "veo-3.1-fast-generate-001": [4, 6, 8],
    "wan-2.2-i2v": [5],
}

# ---------------------------------------------------------------------------
# Pricing (for server-side cost estimation)
# ---------------------------------------------------------------------------
TEXT_MODEL_COST: dict[str, float] = {
    "gemini-2.5-flash": 0.006,
    "gemini-2.5-flash-lite": 0.001,
    "gemini-2.5-pro": 0.023,
    "gemini-3-flash-preview": 0.007,
    "gemini-3-pro-preview": 0.028,
}

IMAGE_MODEL_COST: dict[str, float] = {
    "gemini-2.5-flash-image": 0.04,
    "gemini-3-pro-image-preview": 0.13,
    "qwen-fast": 0.0,
    "qwen-image-edit": 0.0,
    "flux-dev": 0.0,
    "flux-dev-lora": 0.0,
    "flux-dev-redux": 0.0,
    "flux-dev-full": 0.0,
}

VIDEO_MODEL_COST_SILENT: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-preview": 0.40,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-preview": 0.10,
    "veo-3.1-fast-generate-001": 0.10,
    "wan-2.2-i2v": 0.0,
}

VIDEO_MODEL_COST_AUDIO: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-preview": 0.40,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-preview": 0.15,
    "veo-3.1-fast-generate-001": 0.15,
    "wan-2.2-i2v": 0.0,
}


def _estimate_scene_cost(
    scene: "Scene",
    generated_keyframes: int = 0,
    completed_clips: int = 0,
    generated_clips: int = 0,
    has_storyboard: bool = False,
    billed_veo_submissions: int = 0,
    extra_image_regens: int = 0,
) -> float:
    """Estimate cost based on actual artifacts generated.

    For complete scenes, uses theoretical cost minus inherited assets,
    plus extra costs from escalation retries and safety regens.
    For incomplete scenes, uses billed_veo_submissions (all Veo ops
    that Google charged for, including failed/polling clips and retries).

    Args:
        generated_keyframes: Keyframes with source='generated'
        completed_clips: All completed clips (for backward compat)
        generated_clips: Clips with source='generated' (excludes inherited)
        has_storyboard: Whether a storyboard exists
        billed_veo_submissions: Total Veo submissions billed by Google
            (includes failed clips + escalation retries)
        extra_image_regens: Safety-regen image calls (from safety_regen_count)
    """
    clip_dur = scene.target_clip_duration or 6
    audio = scene.audio_enabled and scene.video_model in AUDIO_CAPABLE_MODELS
    vid_rate = (
        VIDEO_MODEL_COST_AUDIO.get(scene.video_model or "", 0.40)
        if audio
        else VIDEO_MODEL_COST_SILENT.get(scene.video_model or "", 0.40)
    )
    img_rate = IMAGE_MODEL_COST.get(scene.image_model or "", 0.04)
    text_cost_rate = TEXT_MODEL_COST.get(scene.text_model or "", 0.01)

    if scene.status == "complete":
        # Full theoretical cost minus inherited assets
        shot_count = scene.target_shot_count or math.ceil(
            (scene.total_duration or 15) / clip_dur
        )
        full_cost = (
            text_cost_rate
            + (shot_count + 1) * img_rate
            + shot_count * clip_dur * vid_rate
        )
        # Subtract inherited keyframe and clip costs
        inherited_kf = max(0, (shot_count + 1) - generated_keyframes) if generated_keyframes > 0 else 0
        inherited_clips = max(0, shot_count - generated_clips) if generated_clips > 0 else 0
        cost = full_cost - (inherited_kf * img_rate) - (inherited_clips * clip_dur * vid_rate)
        # Add extra costs from escalation retries beyond the base count
        extra_submissions = max(0, billed_veo_submissions - generated_clips)
        cost += extra_submissions * clip_dur * vid_rate
        # Add safety regen image costs
        cost += extra_image_regens * img_rate
        return cost

    if scene.status == "pending":
        return 0.0

    # Partial cost based on actual billed artifacts
    cost = 0.0
    if has_storyboard:
        cost += text_cost_rate
    cost += generated_keyframes * img_rate
    # Use billed_veo_submissions to capture failed/polling clips + retries
    veo_count = billed_veo_submissions if billed_veo_submissions > 0 else (
        generated_clips if generated_clips > 0 else completed_clips
    )
    cost += veo_count * clip_dur * vid_rate
    # Add safety regen image costs
    cost += extra_image_regens * img_rate
    return cost


# ============================================================================
# Pydantic Schemas
# ============================================================================

class GenerateRequest(BaseModel):
    """Request schema for POST /api/generate."""
    title: Optional[str] = None
    prompt: str
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    clip_duration: int = 6
    total_duration: int = 15
    text_model: str = "gemini-2.5-flash"
    image_model: str = "gemini-2.5-flash-image"
    video_model: str = "veo-3.1-fast-generate-001"
    enable_audio: bool = True
    production_bible_id: Optional[str] = None
    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: bool = False
    candidate_count: int = 1
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None
    # Selective stage execution
    run_through: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response schema for POST /api/generate."""
    scene_id: str
    status: str
    status_url: str


class StatusResponse(BaseModel):
    """Response schema for GET /api/scenes/{id}/status."""
    scene_id: str
    status: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class ShotReference(BaseModel):
    """Asset reference selected for a shot's Veo generation."""
    asset_id: str
    manifest_tag: str
    name: str
    asset_type: str
    thumbnail_url: Optional[str] = None
    reference_image_url: Optional[str] = None
    quality_score: Optional[float] = None
    is_face_crop: bool = False


class ShotDetail(BaseModel):
    """Shot detail within SceneDetail response."""
    shot_index: int
    description: str
    status: str
    has_start_keyframe: bool
    has_end_keyframe: bool
    has_clip: bool
    clip_status: Optional[str] = None
    start_frame_prompt: Optional[str] = None
    end_frame_prompt: Optional[str] = None
    video_motion_prompt: Optional[str] = None
    transition_notes: Optional[str] = None
    start_keyframe_url: Optional[str] = None
    end_keyframe_url: Optional[str] = None
    clip_url: Optional[str] = None
    selected_references: list[ShotReference] = []
    # PipeSVN staleness
    start_keyframe_staleness: Optional[str] = None
    end_keyframe_staleness: Optional[str] = None
    clip_staleness: Optional[str] = None
    start_keyframe_prompt_used: Optional[str] = None
    end_keyframe_prompt_used: Optional[str] = None
    clip_prompt_used: Optional[str] = None
    rewritten_keyframe_prompt: Optional[str] = None
    rewritten_video_prompt: Optional[str] = None
    is_empty_slot: bool = False
    generation_status: Optional[str] = None


class SceneDetail(BaseModel):
    """Response schema for GET /api/scenes/{id}."""
    scene_id: str
    title: Optional[str] = None
    prompt: str
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    status: str
    created_at: str
    updated_at: str
    shot_count: int
    shots: list[ShotDetail]
    error_message: Optional[str] = None
    total_duration: Optional[int] = None
    clip_duration: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    forked_from: Optional[str] = None
    production_bible_id: Optional[str] = None
    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: bool = False
    candidate_count: int = 1
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None
    # Selective stage execution
    run_through: Optional[str] = None
    # PipeSVN
    head_sha: Optional[str] = None


class ShotEditPayload(BaseModel):
    """Per-shot edits within an edit request."""
    shot_description: Optional[str] = None
    start_frame_prompt: Optional[str] = None
    end_frame_prompt: Optional[str] = None
    video_motion_prompt: Optional[str] = None
    transition_notes: Optional[str] = None


class EditSceneRequest(BaseModel):
    """Request body for PATCH /api/scenes/{id}/edit."""
    prompt: Optional[str] = None
    title: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    clip_duration: Optional[int] = None
    target_shot_count: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    vision_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    production_bible_id: Optional[str] = None
    shot_edits: Optional[dict[int, ShotEditPayload]] = None
    removed_shots: Optional[list[int]] = None
    shot_order: Optional[list[int]] = None
    commit_message: Optional[str] = None
    expected_sha: Optional[str] = None


class EditSceneResponse(BaseModel):
    """Response from PATCH /api/scenes/{id}/edit."""
    scene_id: str
    head_sha: str
    message: str
    changes_count: int


class SceneListItem(BaseModel):
    """Item in list response for GET /api/scenes."""
    scene_id: str
    title: Optional[str] = None
    prompt: str
    status: str
    created_at: str
    total_duration: Optional[int] = None
    clip_duration: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: bool = False
    candidate_count: int = 1
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None
    # Selective stage execution
    run_through: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    thumbnail_url: Optional[str] = None
    production_id: Optional[str] = None
    # Phase 16: Sequence grouping
    sequence_id: Optional[str] = None
    scene_order: Optional[int] = None
    # Phase 18: Screenplay linkage
    screenplay_breakdown_index: Optional[int] = None


class PaginatedScenes(BaseModel):
    """Paginated response envelope for GET /api/scenes."""
    items: list[SceneListItem]
    total: int
    page: int
    per_page: int


class ResumeResponse(BaseModel):
    """Response schema for POST /api/scenes/{id}/resume."""
    scene_id: str
    status: str
    status_url: str


class BatchAddScenesRequest(BaseModel):
    """Request body for POST /api/productions/{id}/scenes (batch add)."""
    scene_ids: list[str]


class ContinueRequest(BaseModel):
    """Optional body for POST /api/scenes/{id}/resume to advance run_through."""
    run_through: Optional[str] = None  # "keyframes", "video", or "all" (= run everything)
    # Model overrides — applied to the scene before resuming
    image_model: Optional[str] = None
    vision_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    clip_duration: Optional[int] = None


class ModifiedAsset(BaseModel):
    """Asset modification in a fork."""
    changes: dict  # {"reverse_prompt": "...", "name": "...", "reference_image": base64_str}


class NewUpload(BaseModel):
    """New reference image to add in fork."""
    image_data: str  # base64-encoded image
    name: str
    asset_type: str
    description: Optional[str] = None
    tags: Optional[list[str]] = None


class AssetChanges(BaseModel):
    """Asset changes for fork request."""
    modified_assets: dict[str, ModifiedAsset] = Field(default_factory=dict)  # asset_id -> changes
    removed_asset_ids: list[str] = Field(default_factory=list)
    new_uploads: list[NewUpload] = Field(default_factory=list)


class ForkRequest(BaseModel):
    """Request schema for POST /api/scenes/{id}/fork."""
    prompt: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    clip_duration: Optional[int] = None
    total_duration: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    shot_edits: Optional[dict[int, dict[str, str]]] = None
    deleted_shots: Optional[list[int]] = None
    clear_keyframes: Optional[list[int]] = None
    asset_changes: Optional[AssetChanges] = None
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None


class ForkResponse(BaseModel):
    """Response schema for POST /api/scenes/{id}/fork."""
    scene_id: str
    forked_from: str
    status: str
    status_url: str
    copied_shots: int
    resume_from: str


class StopResponse(BaseModel):
    """Response schema for POST /api/scenes/{id}/stop."""
    scene_id: str
    status: str


class MetricsResponse(BaseModel):
    """Response schema for GET /api/metrics."""
    total_scenes: int
    status_counts: dict[str, int]
    style_counts: dict[str, int]
    aspect_ratio_counts: dict[str, int]
    text_model_counts: dict[str, int]
    image_model_counts: dict[str, int]
    video_model_counts: dict[str, int]
    audio_counts: dict[str, int]
    shot_count_counts: dict[str, int]
    total_estimated_cost: float
    total_video_seconds: int
    avg_clip_duration: Optional[float] = None


# Production Bible System Schemas

class CreateProductionBibleRequest(BaseModel):
    """Request schema for POST /api/production-bibles."""
    name: str
    description: Optional[str] = None
    category: str = "CUSTOM"
    tags: Optional[list[str]] = Field(default=None)


# Backwards-compat aliases for internal use
CreateManifestRequest = CreateProductionBibleRequest


class UpdateProductionBibleRequest(BaseModel):
    """Request schema for PUT /api/production-bibles/{id}."""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None


UpdateManifestRequest = UpdateProductionBibleRequest


class CreateProductionBibleFromSceneRequest(BaseModel):
    """Request schema for POST /api/production-bibles/from-scene."""
    scene_id: str
    name: Optional[str] = None


CreateManifestFromSceneRequest = CreateProductionBibleFromSceneRequest


class ProductionBibleListItem(BaseModel):
    """Item in list response for GET /api/production-bibles."""
    production_bible_id: str
    name: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    category: str
    tags: Optional[list[str]]
    status: str
    asset_count: int
    times_used: int
    last_used_at: Optional[str]
    version: int
    created_at: str
    updated_at: str


ManifestListItem = ProductionBibleListItem


class ProductionBibleDetailResponse(BaseModel):
    """Response schema for GET /api/production-bibles/{id}."""
    production_bible_id: str
    name: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    category: str
    tags: Optional[list[str]]
    status: str
    processing_progress: Optional[dict]
    contact_sheet_url: Optional[str]
    asset_count: int
    total_processing_cost: float
    times_used: int
    last_used_at: Optional[str]
    version: int
    parent_production_bible_id: Optional[str]
    source_video_duration: Optional[float] = None
    created_at: str
    updated_at: str
    assets: list["AssetResponse"]


ManifestDetailResponse = ProductionBibleDetailResponse


class VideoUploadResponse(BaseModel):
    """Response schema for POST /api/production-bibles/{id}/upload-video."""
    task_id: str
    status: str
    production_bible_id: str


class CreateAssetRequest(BaseModel):
    """Request schema for POST /api/production-bibles/{id}/assets."""
    name: str
    asset_type: str
    description: Optional[str] = None
    user_tags: Optional[list[str]] = Field(default=None)


class UpdateAssetRequest(BaseModel):
    """Request schema for PUT /api/assets/{id}."""
    name: Optional[str] = None
    description: Optional[str] = None
    asset_type: Optional[str] = None
    user_tags: Optional[list[str]] = None
    sort_order: Optional[int] = None
    reverse_prompt: Optional[str] = None        # Phase 5: Stage 3 inline editing
    visual_description: Optional[str] = None     # Phase 5: Stage 3 inline editing


class AssetResponse(BaseModel):
    """Response schema for asset operations."""
    asset_id: str
    production_bible_id: str
    asset_type: str
    name: str
    manifest_tag: str
    user_tags: Optional[list[str]]
    reference_image_url: Optional[str]
    thumbnail_url: Optional[str]
    description: Optional[str]
    source: str
    sort_order: int
    created_at: str
    # Phase 5 fields
    reverse_prompt: Optional[str] = None
    visual_description: Optional[str] = None
    detection_class: Optional[str] = None
    detection_confidence: Optional[float] = None
    is_face_crop: bool = False
    quality_score: Optional[float] = None


class ProcessingProgressResponse(BaseModel):
    """Response schema for manifest processing progress."""
    status: str  # processing, complete, error, not_started
    current_step: Optional[str] = None
    progress: Optional[dict] = None
    error: Optional[str] = None


class CandidateResponse(BaseModel):
    """Response schema for a single generation candidate."""
    candidate_id: str
    candidate_number: int
    local_path: Optional[str] = None
    manifest_adherence_score: Optional[float] = None
    visual_quality_score: Optional[float] = None
    continuity_score: Optional[float] = None
    prompt_adherence_score: Optional[float] = None
    composite_score: Optional[float] = None
    scoring_details: Optional[dict] = None
    is_selected: bool = False
    selected_by: str = "auto"
    generation_cost: float = 0.0
    scoring_cost: float = 0.0
    created_at: str


# ---------------------------------------------------------------------------
# Phase 15: Video Generation Editor Schemas
# ---------------------------------------------------------------------------

class CreateSceneRequest(BaseModel):
    """Request schema for POST /api/scenes (draft scene creation)."""
    prompt: str = ""
    title: str = ""
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    clip_duration: Optional[int] = None
    shot_count: int = 1
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    enable_audio: bool = False
    production_bible_id: Optional[str] = None
    quality_mode: bool = False
    candidate_count: int = 1
    vision_model: Optional[str] = None


class CreateSceneResponse(BaseModel):
    """Response schema for POST /api/scenes."""
    scene_id: str
    status: str
    shot_count: int


class StartGenerationRequest(BaseModel):
    """Request schema for POST /api/scenes/{id}/generate."""
    run_through: Optional[str] = None  # "storyboard"|"keyframes"|"video"|None (all)
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    vision_model: Optional[str] = None
    clip_duration: Optional[int] = None
    enable_audio: Optional[bool] = None


class StartGenerationResponse(BaseModel):
    """Response schema for POST /api/scenes/{id}/generate."""
    scene_id: str
    status: str


# ============================================================================
# Background Task Wrapper
# ============================================================================

async def run_pipeline_background(scene_id: uuid.UUID):
    """Run pipeline in background with fresh session.

    CRITICAL: Never share session across async boundaries.
    Create fresh session inside background task.
    """
    async with async_session() as session:
        try:
            await run_pipeline(session, scene_id)
        except Exception as e:
            # Error already persisted to database by orchestrator
            logger.error(f"Background pipeline failed for {scene_id}: {type(e).__name__}: {str(e)}")


# ============================================================================
# Endpoint Handlers
# ============================================================================

@router.post("/generate", status_code=202, response_model=GenerateResponse)
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start video generation pipeline in background.

    Creates scene record with pending status and adds pipeline execution
    to background tasks. Returns 202 Accepted with scene_id and status URL.
    """
    # Validate aspect ratio (Veo supports 16:9 and 9:16 only)
    if request.aspect_ratio not in ("16:9", "9:16"):
        raise HTTPException(status_code=422, detail=f"aspect_ratio must be 16:9 or 9:16, got {request.aspect_ratio}")

    # Validate clip duration per video model
    allowed = ALLOWED_DURATIONS.get(request.video_model, [5, 6, 7, 8])
    if request.clip_duration not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"clip_duration {request.clip_duration} not supported for {request.video_model}. Allowed: {allowed}",
        )

    # Validate model IDs (ollama/ prefix accepted for text and vision models)
    if not (request.text_model in ALLOWED_TEXT_MODELS or request.text_model.startswith("ollama/")):
        raise HTTPException(status_code=422, detail=f"Invalid text_model: {request.text_model}")
    if request.image_model not in ALLOWED_IMAGE_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid image_model: {request.image_model}")
    if request.video_model not in ALLOWED_VIDEO_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid video_model: {request.video_model}")
    if request.vision_model is not None and not (
        request.vision_model in ALLOWED_TEXT_MODELS or request.vision_model.startswith("ollama/")
    ):
        raise HTTPException(status_code=422, detail=f"Invalid vision_model: {request.vision_model}")

    # Validate audio: reject enable_audio=True for models without audio support
    if request.enable_audio and request.video_model not in AUDIO_CAPABLE_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Audio generation not supported for {request.video_model}",
        )

    # Validate Phase 11: quality mode and candidate count
    if request.candidate_count < 1 or request.candidate_count > 4:
        raise HTTPException(status_code=422, detail="candidate_count must be 1-4")
    if request.quality_mode and request.candidate_count < 2:
        raise HTTPException(status_code=422, detail="Quality Mode requires candidate_count >= 2")

    # Validate run_through (selective stage execution)
    if request.run_through is not None and request.run_through not in ("storyboard", "keyframes", "video"):
        raise HTTPException(
            status_code=422,
            detail=f"run_through must be 'storyboard', 'keyframes', 'video', or null; got '{request.run_through}'",
        )

    # Derive shot count from total duration and clip duration
    shot_count = math.ceil(request.total_duration / request.clip_duration)

    async with async_session() as session:
        # Create scene record
        scene = Scene(
            title=request.title,
            prompt=request.prompt,
            style=request.style,
            aspect_ratio=request.aspect_ratio,
            target_clip_duration=request.clip_duration,
            target_shot_count=shot_count,
            total_duration=request.total_duration,
            text_model=request.text_model,
            image_model=request.image_model,
            video_model=request.video_model,
            audio_enabled=request.enable_audio,
            vision_model=request.vision_model,
            seed=random.randint(0, 2**32 - 1),
            quality_mode=request.quality_mode,
            candidate_count=request.candidate_count if request.quality_mode else 1,
            run_through=request.run_through,
            status="pending",
        )
        session.add(scene)
        await session.flush()  # Get scene.id before snapshot creation

        # Handle production_bible_id if provided
        if request.production_bible_id:
            pb_uuid = uuid.UUID(request.production_bible_id)

            # Validate production bible exists
            manifest = await manifest_service.get_manifest(session, pb_uuid)
            if not manifest:
                raise HTTPException(status_code=404, detail=f"Production Bible {request.production_bible_id} not found")

            # Set scene production bible fields
            scene.production_bible_id = pb_uuid
            scene.production_bible_version = manifest.version

            # Create snapshot
            await manifest_service.create_snapshot(session, pb_uuid, scene.id)

            # Increment usage tracking
            await manifest_service.increment_usage(session, pb_uuid)

            # Also set Production.production_bible_id for direct lookup
            if scene.production_id:
                from vidpipe.db.models import Production as ProdModel
                prod_obj = await session.get(ProdModel, scene.production_id)
                if prod_obj and not prod_obj.production_bible_id:
                    prod_obj.production_bible_id = pb_uuid

            logger.info(f"Scene {scene.id} using production bible {request.production_bible_id}, snapshot created")

            # Note: Conditional manifesting skip (Phase 6 success criteria #5) is achieved
            # by the presence of production_bible_id on the scene. When production_bible_id is set,
            # the pipeline knows assets are pre-processed (snapshot exists). The pipeline's
            # manifesting step (to be added in Phase 7+) will check scene.production_bible_id
            # and skip if present. For now, the pipeline has no manifesting step, so the
            # skip is implicit — pre-built production bibles just bypass the need for one.

        await session.commit()
        await session.refresh(scene)

        scene_id = scene.id
        logger.info(f"Created scene {scene_id} for prompt: {request.prompt[:50]}...")

    # Add background task AFTER committing scene
    background_tasks.add_task(run_pipeline_background, scene_id)

    return GenerateResponse(
        scene_id=str(scene_id),
        status="pending",
        status_url=f"/api/scenes/{scene_id}/status",
    )


# ---------------------------------------------------------------------------
# Phase 15: Video Generation Editor Endpoints
# ---------------------------------------------------------------------------

@router.post("/scenes", status_code=201, response_model=CreateSceneResponse)
async def create_draft_scene(request: CreateSceneRequest):
    """Create a draft scene with empty Shot rows.

    Does NOT start the pipeline. The scene remains in "draft" status until
    POST /api/scenes/{id}/generate is called.
    """
    # Validate aspect ratio (skip if None — draft may not have one yet)
    if request.aspect_ratio is not None:
        if request.aspect_ratio not in ("16:9", "9:16"):
            raise HTTPException(status_code=422, detail=f"aspect_ratio must be 16:9 or 9:16, got {request.aspect_ratio}")

    # Validate clip duration per video model (skip if either is None)
    if request.clip_duration is not None and request.video_model is not None:
        allowed = ALLOWED_DURATIONS.get(request.video_model, [5, 6, 7, 8])
        if request.clip_duration not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"clip_duration {request.clip_duration} not supported for {request.video_model}. Allowed: {allowed}",
            )

    # Validate model IDs (skip if None — draft may not have them yet)
    if request.text_model is not None:
        if not (request.text_model in ALLOWED_TEXT_MODELS or request.text_model.startswith("ollama/")):
            raise HTTPException(status_code=422, detail=f"Invalid text_model: {request.text_model}")
    if request.image_model is not None:
        if request.image_model not in ALLOWED_IMAGE_MODELS:
            raise HTTPException(status_code=422, detail=f"Invalid image_model: {request.image_model}")
    if request.video_model is not None:
        if request.video_model not in ALLOWED_VIDEO_MODELS:
            raise HTTPException(status_code=422, detail=f"Invalid video_model: {request.video_model}")
    if request.vision_model is not None and not (
        request.vision_model in ALLOWED_TEXT_MODELS or request.vision_model.startswith("ollama/")
    ):
        raise HTTPException(status_code=422, detail=f"Invalid vision_model: {request.vision_model}")

    # Validate audio (only check if video_model is also specified)
    if request.enable_audio and request.video_model is not None:
        if request.video_model not in AUDIO_CAPABLE_MODELS:
            raise HTTPException(
                status_code=422,
                detail=f"Audio generation not supported for {request.video_model}",
            )

    # Validate quality mode
    if request.candidate_count < 1 or request.candidate_count > 4:
        raise HTTPException(status_code=422, detail="candidate_count must be 1-4")
    if request.quality_mode and request.candidate_count < 2:
        raise HTTPException(status_code=422, detail="Quality Mode requires candidate_count >= 2")

    # Validate shot_count
    if request.shot_count < 1 or request.shot_count > 20:
        raise HTTPException(status_code=422, detail="shot_count must be 1-20")

    async with async_session() as session:
        clip_dur = request.clip_duration or 0
        scene = Scene(
            title=request.title or None,
            prompt=request.prompt,
            style=request.style or "",
            aspect_ratio=request.aspect_ratio or "",
            target_clip_duration=clip_dur,
            target_shot_count=request.shot_count,
            total_duration=request.shot_count * (request.clip_duration or 6),
            text_model=request.text_model,
            image_model=request.image_model,
            video_model=request.video_model,
            audio_enabled=request.enable_audio,
            vision_model=request.vision_model,
            seed=random.randint(0, 2**32 - 1),
            quality_mode=request.quality_mode,
            candidate_count=request.candidate_count if request.quality_mode else 1,
            status="draft",
        )
        session.add(scene)
        await session.flush()

        # Handle production_bible_id if provided (reuse same snapshot logic as POST /api/generate)
        if request.production_bible_id:
            pb_uuid = uuid.UUID(request.production_bible_id)
            manifest = await manifest_service.get_manifest(session, pb_uuid)
            if not manifest:
                raise HTTPException(status_code=404, detail=f"Production Bible {request.production_bible_id} not found")
            scene.production_bible_id = pb_uuid
            scene.production_bible_version = manifest.version
            await manifest_service.create_snapshot(session, pb_uuid, scene.id)
            await manifest_service.increment_usage(session, pb_uuid)
            # Also set Production.production_bible_id for direct lookup
            if scene.production_id:
                from vidpipe.db.models import Production as ProdModel
                prod_obj = await session.get(ProdModel, scene.production_id)
                if prod_obj and not prod_obj.production_bible_id:
                    prod_obj.production_bible_id = pb_uuid
            logger.info(f"Draft scene {scene.id} using production bible {request.production_bible_id}, snapshot created")

        # Create empty Shot rows
        for i in range(request.shot_count):
            shot = Shot(
                scene_id=scene.id,
                shot_index=i,
                shot_description="",
                start_frame_prompt="",
                end_frame_prompt="",
                video_motion_prompt="",
                transition_notes="",
                status="pending",
            )
            session.add(shot)

        await session.commit()
        await session.refresh(scene)

        scene_id = scene.id
        logger.info(f"Created draft scene {scene_id} with {request.shot_count} empty shots")

    return CreateSceneResponse(
        scene_id=str(scene_id),
        status="draft",
        shot_count=request.shot_count,
    )


@router.post("/scenes/{scene_id}/generate", status_code=202, response_model=StartGenerationResponse)
async def start_generation(
    scene_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    body: Optional[StartGenerationRequest] = None,
):
    """Start or resume pipeline generation for a scene.

    Transitions draft/stopped/staged/failed/complete scenes into pipeline
    execution with gap-filling semantics (empty shots get generated, filled
    shots are preserved).
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Only allow generation from specific states
        allowed_states = ("draft", "stopped", "staged", "failed", "complete")
        if scene.status not in allowed_states:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot start generation from status '{scene.status}'. Allowed: {', '.join(allowed_states)}",
            )

        # Apply optional overrides from request body
        if body:
            if body.run_through is not None:
                new_val = None if body.run_through in ("all", "") else body.run_through
                if new_val is not None and new_val not in ("storyboard", "keyframes", "video"):
                    raise HTTPException(
                        status_code=422,
                        detail=f"run_through must be 'storyboard', 'keyframes', 'video', or null; got '{body.run_through}'",
                    )
                scene.run_through = new_val
            else:
                # Default: clear run_through to run all stages
                scene.run_through = None

            if body.text_model is not None:
                if not (body.text_model in ALLOWED_TEXT_MODELS or body.text_model.startswith("ollama/")):
                    raise HTTPException(status_code=422, detail=f"Invalid text_model: {body.text_model}")
                scene.text_model = body.text_model
            if body.image_model is not None:
                if body.image_model not in ALLOWED_IMAGE_MODELS:
                    raise HTTPException(status_code=422, detail=f"Invalid image_model: {body.image_model}")
                scene.image_model = body.image_model
            if body.video_model is not None:
                if body.video_model not in ALLOWED_VIDEO_MODELS:
                    raise HTTPException(status_code=422, detail=f"Invalid video_model: {body.video_model}")
                scene.video_model = body.video_model
            if body.vision_model is not None:
                if not (body.vision_model in ALLOWED_TEXT_MODELS or body.vision_model.startswith("ollama/")):
                    raise HTTPException(status_code=422, detail=f"Invalid vision_model: {body.vision_model}")
                scene.vision_model = body.vision_model if body.vision_model else None
            if body.clip_duration is not None:
                dur_allowed = ALLOWED_DURATIONS.get(scene.video_model or "", [5, 6, 7, 8])
                if body.clip_duration not in dur_allowed:
                    raise HTTPException(
                        status_code=422,
                        detail=f"clip_duration {body.clip_duration} not supported for {scene.video_model}. Allowed: {dur_allowed}",
                    )
                scene.target_clip_duration = body.clip_duration
            if body.enable_audio is not None:
                if body.enable_audio and scene.video_model not in AUDIO_CAPABLE_MODELS:
                    raise HTTPException(status_code=422, detail=f"Audio not supported for {scene.video_model}")
                scene.audio_enabled = body.enable_audio

        # Verify scene is fully configured before launching pipeline
        run_through = scene.run_through
        required_fields: dict[str, object] = {
            "text_model": scene.text_model,
            "image_model": scene.image_model,
            "video_model": scene.video_model,
            "aspect_ratio": scene.aspect_ratio,
        }
        if run_through == "storyboard":
            required_fields = {"text_model": scene.text_model}
        elif run_through == "keyframes":
            required_fields = {
                "text_model": scene.text_model,
                "image_model": scene.image_model,
            }
        missing = [k for k, v in required_fields.items() if not v]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot start generation: missing required fields: {', '.join(missing)}",
            )

        # Ensure clip_duration is set (default to 6 if missing)
        if not scene.target_clip_duration:
            scene.target_clip_duration = 6
            scene.total_duration = (scene.target_shot_count or 1) * 6

        # Ensure style has a value (default to "cinematic")
        if not scene.style:
            scene.style = "cinematic"

        # Ensure aspect_ratio is set for non-storyboard runs
        if not scene.aspect_ratio and run_through != "storyboard":
            raise HTTPException(status_code=422, detail="Cannot start generation: aspect_ratio is required")

        # Transition status
        if scene.status == "draft":
            scene.status = "pending"
        # For stopped/staged/failed/complete: follow existing resume pattern
        # (pipeline's run_pipeline checks get_resume_step to find correct re-entry point)

        await session.commit()

        logger.info(f"Starting generation for scene {scene_id} (was {scene.status})")

    background_tasks.add_task(run_pipeline_background, scene_id)

    return StartGenerationResponse(
        scene_id=str(scene_id),
        status="pending",
    )


@router.put("/scenes/{scene_id}/final-video")
async def upload_final_video(scene_id: uuid.UUID, file: UploadFile = File(...)):
    """Upload a final video file for a scene.

    Accepts a multipart video upload and saves it as the scene's output file.
    """
    # Validate content type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=422,
            detail=f"Expected video/* content type, got {file.content_type}",
        )

    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        content = await file.read()
        file_mgr = FileManager()
        storage = get_storage_backend()

        if isinstance(storage, LocalStorageBackend):
            # Local: write to disk
            output_path = file_mgr.get_output_path(scene.id, "final.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(content)
            scene.output_path = str(output_path)
        else:
            # S3: upload to storage
            key = f"{scene.id}/output/final.mp4"
            await storage.put(key, content, "video/mp4")
            scene.output_path = key

        await session.commit()

        logger.info(f"Uploaded final video for scene {scene_id}: {scene.output_path}")

        return {
            "scene_id": str(scene_id),
            "output_path": scene.output_path,
            "status": scene.status,
        }


@router.get("/scenes/{scene_id}/status", response_model=StatusResponse)
async def get_scene_status(scene_id: uuid.UUID):
    """Get lightweight scene status for polling.

    Returns scene-level status only (no shot details).
    Use GET /api/scenes/{id} for full detail.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        return StatusResponse(
            scene_id=str(scene.id),
            status=scene.status,
            created_at=scene.created_at.isoformat(),
            updated_at=scene.updated_at.isoformat(),
            error_message=scene.error_message,
        )


@router.get("/scenes/{scene_id}", response_model=SceneDetail)
async def get_scene_detail(scene_id: uuid.UUID):
    """Get full scene detail with shot breakdown.

    Includes shot-level status, keyframe existence, and clip status.
    """
    async with async_session() as session:
        # Load scene
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Load shots with their keyframes and clips
        shots_result = await session.execute(
            select(Shot)
            .where(Shot.scene_id == scene_id)
            .order_by(Shot.shot_index)
        )
        shots = shots_result.scalars().all()

        # Build shot details
        shot_details = []
        for shot in shots:
            # Get keyframes for this shot
            keyframes_result = await session.execute(
                select(Keyframe).where(Keyframe.shot_id == shot.id)
            )
            keyframes = keyframes_result.scalars().all()

            has_start_keyframe = any(kf.position == "start" for kf in keyframes)
            has_end_keyframe = any(kf.position == "end" for kf in keyframes)

            # Get clip for this shot
            clip_result = await session.execute(
                select(VideoClip).where(VideoClip.shot_id == shot.id)
            )
            clip = clip_result.scalar_one_or_none()

            start_kf = next((kf for kf in keyframes if kf.position == "start"), None)
            end_kf = next((kf for kf in keyframes if kf.position == "end"), None)

            shot_details.append(ShotDetail(
                shot_index=shot.shot_index,
                description=shot.shot_description,
                status=shot.status,
                has_start_keyframe=has_start_keyframe,
                has_end_keyframe=has_end_keyframe,
                has_clip=clip is not None,
                clip_status=clip.status if clip else None,
                start_frame_prompt=shot.start_frame_prompt,
                end_frame_prompt=shot.end_frame_prompt,
                video_motion_prompt=shot.video_motion_prompt,
                transition_notes=shot.transition_notes,
                start_keyframe_url=f"/api/keyframes/{start_kf.id}" if start_kf else None,
                end_keyframe_url=f"/api/keyframes/{end_kf.id}" if end_kf else None,
                clip_url=f"/api/clips/{clip.id}" if clip and clip.status == "complete" and clip.local_path else None,
                generation_status=shot.generation_status,
            ))

        # Load selected reference tags for all shots (Phase 8)
        sm_result = await session.execute(
            select(ShotManifestModel).where(
                ShotManifestModel.scene_id == scene.id
            )
        )
        shot_manifests_by_index = {
            sm.shot_index: sm for sm in sm_result.scalars().all()
        }

        # If scene has production bible, load assets for reference resolution
        ref_assets_by_tag = {}
        if scene.production_bible_id:
            assets_result = await session.execute(
                select(Asset).where(Asset.production_bible_id == scene.production_bible_id)
            )
            ref_assets_by_tag = {
                a.manifest_tag: a for a in assets_result.scalars().all()
            }

        # Enrich shot_details with selected references
        for sd in shot_details:
            sm = shot_manifests_by_index.get(sd.shot_index)
            if sm and sm.selected_reference_tags:
                refs = []
                for tag in sm.selected_reference_tags:
                    asset = ref_assets_by_tag.get(tag)
                    if asset:
                        refs.append(ShotReference(
                            asset_id=str(asset.id),
                            manifest_tag=asset.manifest_tag,
                            name=asset.name,
                            asset_type=asset.asset_type,
                            thumbnail_url=asset.thumbnail_url,
                            reference_image_url=asset.reference_image_url,
                            quality_score=asset.quality_score,
                            is_face_crop=asset.is_face_crop,
                        ))
                sd.selected_references = refs

        # Enrich shot_details with staleness (PipeSVN)
        from vidpipe.services.checkpoint_service import compute_keyframe_staleness, compute_clip_staleness
        for sd in shot_details:
            sm = shot_manifests_by_index.get(sd.shot_index)
            # Find the matching shot object
            shot_obj = next((s for s in shots if s.shot_index == sd.shot_index), None)
            if not shot_obj:
                continue

            # Get keyframes and clip for this shot
            kf_result2 = await session.execute(
                select(Keyframe).where(Keyframe.shot_id == shot_obj.id)
            )
            kfs = kf_result2.scalars().all()
            start_kf = next((k for k in kfs if k.position == "start"), None)
            end_kf = next((k for k in kfs if k.position == "end"), None)

            clip_result2 = await session.execute(
                select(VideoClip).where(VideoClip.shot_id == shot_obj.id)
            )
            clip_obj = clip_result2.scalar_one_or_none()

            sd.start_keyframe_staleness = compute_keyframe_staleness(shot_obj, start_kf, sm)
            sd.end_keyframe_staleness = compute_keyframe_staleness(shot_obj, end_kf, sm)
            sd.clip_staleness = compute_clip_staleness(shot_obj, clip_obj, sm)
            sd.start_keyframe_prompt_used = start_kf.prompt_used if start_kf else None
            sd.end_keyframe_prompt_used = end_kf.prompt_used if end_kf else None
            sd.clip_prompt_used = clip_obj.prompt_used if clip_obj else None
            if sm:
                sd.rewritten_keyframe_prompt = sm.rewritten_keyframe_prompt
                sd.rewritten_video_prompt = sm.rewritten_video_prompt

        return SceneDetail(
            scene_id=str(scene.id),
            title=scene.title,
            prompt=scene.prompt,
            style=scene.style,
            aspect_ratio=scene.aspect_ratio,
            status=scene.status,
            created_at=scene.created_at.isoformat(),
            updated_at=scene.updated_at.isoformat(),
            shot_count=len(shots),
            shots=shot_details,
            error_message=scene.error_message,
            total_duration=scene.total_duration,
            clip_duration=scene.target_clip_duration,
            text_model=scene.text_model,
            image_model=scene.image_model,
            video_model=scene.video_model,
            audio_enabled=scene.audio_enabled,
            forked_from=str(scene.forked_from_id) if scene.forked_from_id else None,
            production_bible_id=str(scene.production_bible_id) if scene.production_bible_id else None,
            quality_mode=scene.quality_mode,
            candidate_count=scene.candidate_count,
            vision_model=scene.vision_model,
            run_through=scene.run_through,
            head_sha=scene.head_sha,
        )


@router.get("/scenes", response_model=PaginatedScenes)
async def list_scenes(
    page: int = 1,
    per_page: int = 12,
    view: Optional[str] = None,
    status: Optional[str] = None,
):
    """List scenes with server-side pagination (newest first).

    Query params:
      - page: 1-based page number (default 1)
      - per_page: items per page, must be 12, 24, 48, or 96 (default 12)
      - view: "cards" to include thumbnail_url (avoids extra query for list view)
      - status: filter by scene status (e.g. "complete", "failed", "stopped")
    """
    if per_page not in (12, 24, 48, 96):
        per_page = 12
    if page < 1:
        page = 1

    VALID_STATUSES = {"draft", "pending", "storyboarding", "keyframing", "video_gen", "stitching", "complete", "failed", "stopped", "staged"}

    async with async_session() as session:
        filters = [Scene.deleted_at.is_(None)]
        if status and status in VALID_STATUSES:
            filters.append(Scene.status == status)
        base_filter = filters[0] if len(filters) == 1 else sa_and(*filters)

        # Total count
        count_result = await session.execute(
            select(sa_func.count(Scene.id)).where(base_filter)
        )
        total = count_result.scalar() or 0

        # Paginated scenes
        result = await session.execute(
            select(Scene)
            .where(base_filter)
            .order_by(Scene.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
        scenes = result.scalars().all()

        # Build thumbnail map when cards view requested
        thumbnail_map: dict[str, str] = {}
        if view == "cards" and scenes:
            scene_ids = [p.id for p in scenes]
            thumb_q = await session.execute(
                select(Shot.scene_id, Keyframe.id)
                .join(Shot, Keyframe.shot_id == Shot.id)
                .where(Shot.scene_id.in_(scene_ids))
                .where(Shot.shot_index == 0)
                .where(Keyframe.position == "start")
            )
            for row in thumb_q:
                pid_str = str(row[0])
                if pid_str not in thumbnail_map:
                    thumbnail_map[pid_str] = f"/api/keyframes/{row[1]}"

        return PaginatedScenes(
            items=[
                SceneListItem(
                    scene_id=str(p.id),
                    title=p.title,
                    prompt=p.prompt,
                    status=p.status,
                    created_at=p.created_at.isoformat(),
                    total_duration=p.total_duration,
                    clip_duration=p.target_clip_duration,
                    text_model=p.text_model,
                    image_model=p.image_model,
                    video_model=p.video_model,
                    audio_enabled=p.audio_enabled,
                    vision_model=p.vision_model,
                    run_through=p.run_through,
                    style=p.style,
                    aspect_ratio=p.aspect_ratio,
                    thumbnail_url=thumbnail_map.get(str(p.id)),
                    production_id=str(p.production_id) if p.production_id else None,
                    sequence_id=str(p.sequence_id) if p.sequence_id else None,
                    scene_order=p.scene_order,
                    screenplay_breakdown_index=p.screenplay_breakdown_index,
                )
                for p in scenes
            ],
            total=total,
            page=page,
            per_page=per_page,
        )


@router.post("/scenes/{scene_id}/resume", status_code=202, response_model=ResumeResponse)
async def resume_scene(
    scene_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    body: Optional[ContinueRequest] = None,
):
    """Resume failed or interrupted pipeline in background.

    Accepts optional ContinueRequest body to advance run_through when
    continuing from a "staged" state.

    Returns 409 if scene is not in a resumable state (including complete).
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Check if scene can be resumed
        if not can_resume(scene.status):
            raise HTTPException(
                status_code=409,
                detail=f"Scene cannot be resumed from status '{scene.status}'"
            )

        # Apply updates from ContinueRequest body
        if body:
            # Update run_through when continuing from staged state
            if scene.status == "staged" and body.run_through is not None:
                new_val = None if body.run_through == "all" else body.run_through
                if new_val is not None and new_val not in ("storyboard", "keyframes", "video"):
                    raise HTTPException(
                        status_code=422,
                        detail=f"run_through must be 'storyboard', 'keyframes', 'video', or 'all'; got '{body.run_through}'",
                    )
                scene.run_through = new_val

            # Apply model overrides
            if body.image_model is not None:
                if body.image_model not in ALLOWED_IMAGE_MODELS:
                    raise HTTPException(status_code=422, detail=f"Invalid image_model: {body.image_model}")
                scene.image_model = body.image_model
            if body.vision_model is not None:
                if not (body.vision_model in ALLOWED_TEXT_MODELS or body.vision_model.startswith("ollama/")):
                    raise HTTPException(status_code=422, detail=f"Invalid vision_model: {body.vision_model}")
                scene.vision_model = body.vision_model if body.vision_model else None
            if body.video_model is not None:
                if body.video_model not in ALLOWED_VIDEO_MODELS:
                    raise HTTPException(status_code=422, detail=f"Invalid video_model: {body.video_model}")
                scene.video_model = body.video_model
            if body.audio_enabled is not None:
                if body.audio_enabled and scene.video_model not in AUDIO_CAPABLE_MODELS:
                    raise HTTPException(status_code=422, detail=f"Audio not supported for {scene.video_model}")
                scene.audio_enabled = body.audio_enabled
            if body.clip_duration is not None:
                allowed = ALLOWED_DURATIONS.get(scene.video_model or "", [5, 6, 7, 8])
                if body.clip_duration not in allowed:
                    raise HTTPException(
                        status_code=422,
                        detail=f"clip_duration {body.clip_duration} not supported for {scene.video_model}. Allowed: {allowed}",
                    )
                scene.target_clip_duration = body.clip_duration

            await session.commit()

        logger.info(f"Resuming scene {scene_id} from status {scene.status}")

    # Add background task
    background_tasks.add_task(run_pipeline_background, scene_id)

    return ResumeResponse(
        scene_id=str(scene_id),
        status=scene.status,
        status_url=f"/api/scenes/{scene_id}/status",
    )


@router.post("/scenes/{scene_id}/stop", response_model=StopResponse)
async def stop_scene(scene_id: uuid.UUID):
    """Stop a running pipeline.

    Sets the scene status to 'stopped'. The background pipeline checks this
    flag between steps and inside long-running loops, then exits gracefully.
    The scene can be resumed later with POST /resume.

    Returns 409 if scene is already in a terminal state (complete/failed/stopped).
    """
    ACTIVE_STATUSES = {"pending", "storyboarding", "keyframing", "video_gen", "stitching"}

    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        if scene.status not in ACTIVE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Scene cannot be stopped from status '{scene.status}'"
            )

        scene.status = "stopped"
        await session.commit()

        logger.info(f"Scene {scene_id} marked as stopped")

    return StopResponse(scene_id=str(scene_id), status="stopped")


class UpdateSceneRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@router.patch("/scenes/{scene_id}", response_model=dict)
async def update_scene(scene_id: uuid.UUID, body: UpdateSceneRequest):
    """Update mutable scene fields (currently just title)."""
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene or scene.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Scene not found")

        scene.title = body.title
        await session.commit()

    return {"scene_id": str(scene_id), "title": body.title}


@router.delete("/scenes/{scene_id}")
async def delete_scene(scene_id: uuid.UUID):
    """Soft-delete a scene: sets deleted_at, removes disk assets.

    Only terminal-status scenes (complete, failed, stopped) can be deleted.
    DB records are preserved for cost tracking.
    """
    DELETABLE_STATUSES = {"complete", "failed", "stopped", "draft"}

    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene or scene.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Scene not found")

        if scene.status not in DELETABLE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete scene with status '{scene.status}'. Stop it first.",
            )

        scene.deleted_at = sa_func.now()
        await session.commit()

    # Remove disk assets (keyframes, clips, output) after commit
    try:
        scene_dir = FileManager().base_dir / str(scene_id)
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
            logger.info(f"Removed disk assets for scene {scene_id}")
    except Exception:
        logger.warning(f"Failed to remove disk assets for scene {scene_id}", exc_info=True)

    # Remove S3 objects under this scene's prefix
    storage = get_storage_backend()
    if not storage.is_local():
        try:
            count = await storage.delete_prefix(str(scene_id))
            if count:
                logger.info(f"Removed {count} S3 objects for scene {scene_id}")
        except Exception:
            logger.warning(f"Failed to remove S3 objects for scene {scene_id}", exc_info=True)

    return {"status": "deleted", "scene_id": str(scene_id)}


class _ExpansionShots(BaseModel):
    """Wrapper schema for Gemini structured output of expansion shots."""
    shots: list[ShotSchema] = Field(description="New shots to append")


class _EnhancedExpansionShots(BaseModel):
    """Wrapper schema for Gemini structured output of manifest-aware expansion shots."""
    shots: list[EnhancedShotSchema] = Field(description="New shots to append with manifest and audio data")


async def _generate_expansion_shots(
    scene: Scene,
    kept_shots: list[dict],
    num_new_shots: int,
    start_index: int,
    asset_registry_block: Optional[str] = None,
    text_adapter=None,
) -> list[dict]:
    """Generate storyboard entries for new shots extending an existing storyboard.

    Uses the LLM adapter system (Vertex AI or Ollama) to create shot entries
    that continue the narrative from the kept shots.

    Args:
        scene: The new forked scene (has prompt, style, etc.)
        kept_shots: storyboard_raw["shots"] entries for kept shots
        num_new_shots: How many new shots to generate
        start_index: shot_index for the first new shot
        asset_registry_block: When provided, enables manifest-aware generation
            with shot_manifest and audio_manifest in the output schema.
        text_adapter: Optional LLMAdapter instance. If None, falls back to
            Vertex AI with the scene's text_model or gemini-2.5-flash.

    Returns:
        List of shot dicts ready for Shot record creation
    """
    from vidpipe.services.llm import get_adapter

    if text_adapter is None:
        model_id = scene.text_model or "gemini-2.5-flash"
        text_adapter = get_adapter(model_id)

    style_label = (scene.style or "cinematic").replace("_", " ")

    # Build context from kept shots
    kept_summary = json.dumps(kept_shots, indent=2) if kept_shots else "[]"

    # Choose schema based on whether asset registry is available
    use_enhanced = asset_registry_block is not None
    schema = _EnhancedExpansionShots if use_enhanced else _ExpansionShots

    manifest_instructions = ""
    if use_enhanced:
        manifest_instructions = f"""
AVAILABLE ASSETS (from Asset Registry):
{asset_registry_block}

SHOT MANIFEST INSTRUCTIONS:
- Reference registered assets by their [TAG] (e.g., [CHAR_01], [ENV_02])
- Use the asset's reverse_prompt for visual detail
- Assign roles: subject | background | prop | interaction_target | environment
- Specify spatial positions and actions for each placed asset
- Include composition metadata: shot_type, camera_movement, focal_point
- Add continuity_notes describing visual continuity with previous shots
- You MAY declare new_asset_declarations for assets NOT in the registry

AUDIO MANIFEST INSTRUCTIONS:
- dialogue_lines: Map speech to character tags (speaker_tag must be a registered [TAG])
- sfx: Sound effects with trigger descriptions and relative timing
- ambient: Base layer soundscape + environmental context
- music: Style, mood, tempo, instruments, and transition cues
- audio_continuity: What carries from previous shot, what's new, what cuts
"""

    prompt = f"""You are a storyboard director. You have an existing partial storyboard and need to generate {num_new_shots} NEW continuation shot(s).

VISUAL STYLE: {style_label}
ASPECT RATIO: {scene.aspect_ratio}
{manifest_instructions}
ORIGINAL SCRIPT:
{scene.prompt}

EXISTING SHOTS (already in the storyboard — do NOT repeat these):
{kept_summary}

Generate exactly {num_new_shots} new shot(s) that continue the narrative from where the existing shots leave off.
- Shot indices must start at {start_index} and increment by 1.
- Maintain visual consistency with the existing shots (same characters, style, color palette).
- Follow the same keyframe prompt format: "A {style_label} rendering of..." with subject, action, setting, lighting, camera, style cues, color palette.
- Motion prompts should describe ONLY motion/camera, not re-describe visuals.
- Transition notes should connect smoothly from the last existing shot to the first new shot, and between new shots.
- Include key_details for each shot (3-6 specific terms from the original script).
"""

    result = await text_adapter.generate_text(
        prompt=prompt,
        schema=schema,
        temperature=0.7,
        max_retries=3,
    )

    # Normalize shot indices to start at start_index
    shots_out = []
    for i, shot_entry in enumerate(result.shots[:num_new_shots]):
        sd = shot_entry.model_dump()
        sd["shot_index"] = start_index + i
        shots_out.append(sd)

    return shots_out


async def _copy_assets_for_fork(
    session,
    source_manifest_id,
    source_scene_id,
    new_scene_id,
    asset_changes,  # Optional[AssetChanges]
) -> tuple[list, list[str]]:
    """Copy parent manifest assets to forked scene with inheritance tracking.

    Returns:
        (list_of_new_assets, list_of_modified_asset_tags)
    """
    # Load canonical (non-inherited) assets for this manifest to avoid
    # cascading duplication when forking a scene that was itself forked
    result = await session.execute(
        select(Asset).where(
            Asset.production_bible_id == source_manifest_id,
            Asset.is_inherited == False,
        )
    )
    parent_assets = list(result.scalars().all())

    # Build removed set and modified map from asset_changes
    removed_set: set[str] = set()
    modified_map: dict = {}
    if asset_changes is not None:
        removed_set = set(asset_changes.removed_asset_ids)
        modified_map = asset_changes.modified_assets  # dict[str, ModifiedAsset]

    new_assets = []
    modified_asset_tags: list[str] = []

    for asset in parent_assets:
        asset_id_str = str(asset.id)

        # Skip removed assets
        if asset_id_str in removed_set:
            continue

        # Determine modification state
        is_modified = asset_id_str in modified_map

        # Create new Asset row with all fields copied
        new_asset = Asset(
            manifest_id=source_manifest_id,  # shared manifest
            asset_type=asset.asset_type,
            name=asset.name,
            manifest_tag=asset.manifest_tag,
            user_tags=asset.user_tags,
            reference_image_url=asset.reference_image_url,
            thumbnail_url=asset.thumbnail_url,
            description=asset.description,
            source=asset.source,
            sort_order=asset.sort_order,
            reverse_prompt=asset.reverse_prompt,
            visual_description=asset.visual_description,
            detection_class=asset.detection_class,
            detection_confidence=asset.detection_confidence,
            is_face_crop=asset.is_face_crop,
            crop_bbox=asset.crop_bbox,
            face_embedding=asset.face_embedding,
            clip_embedding=asset.clip_embedding,
            quality_score=asset.quality_score,
            source_asset_id=asset.source_asset_id,
            # Inheritance tracking
            is_inherited=not is_modified,
            inherited_from_asset=asset.id,
            inherited_from_scene=source_scene_id,
        )

        # Apply modifications if present
        if is_modified:
            mod = modified_map[asset_id_str]
            changes = mod.changes
            if "reverse_prompt" in changes:
                new_asset.reverse_prompt = changes["reverse_prompt"]
            if "name" in changes:
                new_asset.name = changes["name"]
            if "visual_description" in changes:
                new_asset.visual_description = changes["visual_description"]
            # Collect the original asset's tag for invalidation computation
            modified_asset_tags.append(asset.manifest_tag)

        session.add(new_asset)

    new_assets_result = await session.execute(
        select(Asset).where(Asset.inherited_from_scene == new_scene_id)
    )

    return new_assets, modified_asset_tags


async def _copy_shot_manifests(
    session,
    source_scene_id,
    new_scene_id,
    shot_boundary: int,
    deleted_set: set,
):
    """Copy shot manifests for all non-deleted source shots.

    Shots below the invalidation boundary get a full copy.
    Shots at/above the boundary get structural fields only (regen fields cleared).
    Uses post-deletion shot index mapping.
    """
    # Load source shot manifests
    result = await session.execute(
        select(ShotManifestModel).where(
            ShotManifestModel.scene_id == source_scene_id
        )
    )
    source_sms = list(result.scalars().all())

    def _old_to_new(old_idx: int) -> int:
        """Map original shot index to post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    for sm in source_sms:
        # Skip shots that were deleted
        if sm.shot_index in deleted_set:
            continue

        new_idx = _old_to_new(sm.shot_index)

        if new_idx < shot_boundary:
            # Full copy for shots below boundary
            new_sm = ShotManifestModel(
                scene_id=new_scene_id,
                shot_index=new_idx,
                manifest_json=sm.manifest_json,
                composition_shot_type=sm.composition_shot_type,
                composition_camera_movement=sm.composition_camera_movement,
                asset_tags=sm.asset_tags,
                new_asset_count=sm.new_asset_count,
                selected_reference_tags=sm.selected_reference_tags,
                cv_analysis_json=sm.cv_analysis_json,
                continuity_score=sm.continuity_score,
                rewritten_keyframe_prompt=sm.rewritten_keyframe_prompt,
                rewritten_video_prompt=sm.rewritten_video_prompt,
            )
        else:
            # Structural copy for shots at/above boundary — clear regen fields
            new_sm = ShotManifestModel(
                scene_id=new_scene_id,
                shot_index=new_idx,
                manifest_json=sm.manifest_json,
                composition_shot_type=sm.composition_shot_type,
                composition_camera_movement=sm.composition_camera_movement,
                asset_tags=sm.asset_tags,
                new_asset_count=sm.new_asset_count,
                selected_reference_tags=None,
                cv_analysis_json=None,
                continuity_score=None,
                rewritten_keyframe_prompt=None,
                rewritten_video_prompt=None,
            )
        session.add(new_sm)


async def _copy_shot_audio_manifests(
    session,
    source_scene_id,
    new_scene_id,
    deleted_set: set,
):
    """Copy shot audio manifests for all non-deleted source shots.

    Audio manifests have no regen-specific fields, so all columns are copied
    as-is with shot index remapping for deleted shots.
    """
    result = await session.execute(
        select(ShotAudioManifestModel).where(
            ShotAudioManifestModel.scene_id == source_scene_id
        )
    )
    source_sams = list(result.scalars().all())

    def _old_to_new(old_idx: int) -> int:
        """Map original shot index to post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    for sam in source_sams:
        if sam.shot_index in deleted_set:
            continue

        new_idx = _old_to_new(sam.shot_index)

        new_sam = ShotAudioManifestModel(
            scene_id=new_scene_id,
            shot_index=new_idx,
            dialogue_json=sam.dialogue_json,
            sfx_json=sam.sfx_json,
            ambient_json=sam.ambient_json,
            music_json=sam.music_json,
            audio_continuity_json=sam.audio_continuity_json,
            speaker_tags=sam.speaker_tags,
            has_dialogue=sam.has_dialogue,
            has_music=sam.has_music,
        )
        session.add(new_sam)


def _compute_asset_invalidation_point(
    shot_manifests: list,  # ShotManifest rows
    modified_asset_tags: set,
    deleted_set: set,
) -> int:
    """Find earliest shot using any modified asset.

    Returns:
        Shot index of first affected shot, or -1 if no match.
    """
    earliest = float("inf")
    for sm in shot_manifests:
        if sm.shot_index in deleted_set:
            continue
        if sm.asset_tags:
            for tag in sm.asset_tags:
                if tag in modified_asset_tags:
                    earliest = min(earliest, sm.shot_index)
                    break
    return int(earliest) if earliest != float("inf") else -1


def _compute_invalidation(
    source: Scene,
    overrides: dict,
    shot_edits: Optional[dict[int, dict[str, str]]],
    deleted_shots: Optional[list[int]] = None,
    clear_keyframes: Optional[list[int]] = None,
) -> tuple[str, int]:
    """Compute the pipeline resume point and shot copy boundary for a fork.

    Returns (resume_stage, shot_copy_boundary) where:
    - resume_stage: the pipeline status to start from
    - shot_copy_boundary: shots with index < boundary are fully copied,
      shots at/after boundary get partial or no copying depending on stage

    Shot boundary uses *new* (post-deletion) numbering when deleted_shots
    are present.

    Inherited shots are always preserved — global setting changes (prompt,
    style, models, aspect_ratio, etc.) never trigger re-generation of
    inherited content.  Only per-shot edits and explicit keyframe clears
    move the boundary.
    """
    shot_count = source.target_shot_count or 3

    # Compute deletion info
    deleted_set = set(deleted_shots) if deleted_shots else set()
    deleted_count = len(deleted_set)
    kept_count = shot_count - deleted_count

    # Compute new shot count from duration settings
    new_total = overrides.get("total_duration", source.total_duration) or source.total_duration or 15
    new_clip = overrides.get("clip_duration", source.target_clip_duration) or source.target_clip_duration or 6
    new_shot_count = math.ceil(new_total / new_clip)

    def _old_to_new(old_idx: int) -> int:
        """Map an original shot index to its post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    # Start with all shots preserved (boundaries at new_shot_count = no invalidation)
    min_keyframe_boundary = new_shot_count
    min_video_boundary = new_shot_count

    # Cleared keyframes → keyframe invalidation
    if clear_keyframes:
        for idx in clear_keyframes:
            if 0 <= idx < shot_count and idx not in deleted_set:
                new_idx = _old_to_new(idx)
                min_keyframe_boundary = min(min_keyframe_boundary, new_idx)

    # Shot-level edits
    if shot_edits:
        for idx, edits in shot_edits.items():
            if idx in deleted_set:
                continue
            for field in edits:
                if field in ("start_frame_prompt", "end_frame_prompt"):
                    new_idx = _old_to_new(idx)
                    min_keyframe_boundary = min(min_keyframe_boundary, new_idx)
                elif field == "video_motion_prompt":
                    new_idx = _old_to_new(idx)
                    min_video_boundary = min(min_video_boundary, new_idx)

    # Expansion: new shots need keyframes + video
    if new_shot_count > kept_count:
        min_keyframe_boundary = min(min_keyframe_boundary, kept_count)

    if min_keyframe_boundary < new_shot_count:
        return "keyframing", min_keyframe_boundary
    if min_video_boundary < new_shot_count:
        return "video_gen", min_video_boundary

    # No invalidation — just re-stitch
    return "stitching", new_shot_count


@router.post("/scenes/{scene_id}/fork", status_code=202, response_model=ForkResponse)
async def fork_scene(scene_id: uuid.UUID, request: ForkRequest, background_tasks: BackgroundTasks):
    """Fork a scene with optional edits and resume from the appropriate pipeline stage.

    Creates a new scene that copies existing assets up to the edit point, then
    resumes the pipeline from there. Copied assets are marked as 'inherited'.
    """
    async with async_session() as session:
        # Load source scene
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Must be terminal
        if source.status not in ("complete", "failed", "stopped"):
            raise HTTPException(status_code=409, detail=f"Can only fork terminal scenes, got '{source.status}'")

        # Collect overrides (only explicitly provided fields)
        overrides: dict = {}
        for field in ("prompt", "style", "aspect_ratio", "clip_duration", "total_duration",
                       "text_model", "image_model", "video_model", "audio_enabled", "vision_model"):
            val = getattr(request, field if field != "clip_duration" else field)
            if val is not None:
                overrides[field] = val

        # Validate overrides
        ar = overrides.get("aspect_ratio", source.aspect_ratio)
        if ar not in ("16:9", "9:16"):
            raise HTTPException(status_code=422, detail=f"aspect_ratio must be 16:9 or 9:16, got {ar}")

        vm = overrides.get("video_model", source.video_model)
        if vm and vm not in ALLOWED_VIDEO_MODELS:
            raise HTTPException(status_code=422, detail=f"Invalid video_model: {vm}")
        im = overrides.get("image_model", source.image_model)
        if im and im not in ALLOWED_IMAGE_MODELS:
            raise HTTPException(status_code=422, detail=f"Invalid image_model: {im}")
        tm = overrides.get("text_model", source.text_model)
        if tm and not (tm in ALLOWED_TEXT_MODELS or tm.startswith("ollama/")):
            raise HTTPException(status_code=422, detail=f"Invalid text_model: {tm}")
        vism = overrides.get("vision_model", source.vision_model)
        if vism and not (vism in ALLOWED_TEXT_MODELS or vism.startswith("ollama/")):
            raise HTTPException(status_code=422, detail=f"Invalid vision_model: {vism}")

        cd = overrides.get("clip_duration", source.target_clip_duration)
        allowed = ALLOWED_DURATIONS.get(vm or "", [5, 6, 7, 8])
        if cd not in allowed:
            raise HTTPException(status_code=422, detail=f"clip_duration {cd} not supported for {vm}. Allowed: {allowed}")

        ae = overrides.get("audio_enabled", source.audio_enabled)
        if ae and vm not in AUDIO_CAPABLE_MODELS:
            raise HTTPException(status_code=422, detail=f"Audio not supported for {vm}")

        # Validate deleted_shots — must leave at least 1 shot
        deleted_set = set(request.deleted_shots) if request.deleted_shots else set()
        src_shot_count = source.target_shot_count or 3
        if deleted_set and len(deleted_set) >= src_shot_count:
            raise HTTPException(status_code=422, detail="Cannot delete all shots; at least 1 must remain")

        # Validate asset_changes requires a production bible on the source scene
        if request.asset_changes is not None and source.production_bible_id is None:
            raise HTTPException(status_code=422, detail="Cannot apply asset_changes to a scene without a production bible")

        # Compute invalidation point
        resume_from, shot_boundary = _compute_invalidation(
            source, overrides, request.shot_edits,
            deleted_shots=request.deleted_shots,
            clear_keyframes=request.clear_keyframes,
        )

        # Phase 12: Asset modification invalidation
        # If asset_changes has modified assets, check which shots use them
        # and potentially tighten the invalidation boundary.
        if request.asset_changes is not None and source.production_bible_id is not None:
            modified_asset_ids = list((request.asset_changes.modified_assets or {}).keys())
            if modified_asset_ids:
                # Load parent assets to find their manifest_tags
                parent_assets_result = await session.execute(
                    select(Asset).where(Asset.production_bible_id == source.production_bible_id)
                )
                parent_assets_map = {
                    str(a.id): a.manifest_tag for a in parent_assets_result.scalars().all()
                }
                mod_tags = {
                    parent_assets_map[aid] for aid in modified_asset_ids if aid in parent_assets_map
                }

                if mod_tags:
                    # Load source shot manifests for asset invalidation check
                    src_sm_result = await session.execute(
                        select(ShotManifestModel).where(
                            ShotManifestModel.scene_id == source.id
                        )
                    )
                    src_shot_manifests = list(src_sm_result.scalars().all())

                    asset_inv_point = _compute_asset_invalidation_point(
                        src_shot_manifests, mod_tags, deleted_set
                    )

                    if asset_inv_point >= 0 and asset_inv_point < shot_boundary:
                        # Asset modification affects shots before current boundary —
                        # tighten to asset invalidation point (keyframe re-run needed)
                        shot_boundary = asset_inv_point
                        if resume_from == "stitching" or resume_from == "video_gen":
                            resume_from = "keyframing"

        # Merge settings
        new_total = overrides.get("total_duration", source.total_duration)
        new_clip = overrides.get("clip_duration", source.target_clip_duration)
        new_shot_count = math.ceil((new_total or 15) / (new_clip or 6))

        # Adjust shot count and total duration for deletions — but only when
        # the frontend did NOT explicitly send total_duration. When total_duration
        # is explicit, new_shot_count already reflects the user's desired count
        # (including any expansion after deletion).
        if deleted_set and "total_duration" not in overrides:
            new_shot_count = max(1, new_shot_count - len(deleted_set))
            new_total = new_shot_count * (new_clip or 6)

        # Create new scene
        new_scene = Scene(
            prompt=overrides.get("prompt", source.prompt),
            style=overrides.get("style", source.style),
            aspect_ratio=ar,
            target_clip_duration=new_clip,
            target_shot_count=new_shot_count,
            total_duration=new_total,
            text_model=tm,
            image_model=im,
            video_model=vm,
            audio_enabled=ae,
            vision_model=vism,
            seed=random.randint(0, 2**32 - 1),
            forked_from_id=source.id,
            status=resume_from,
        )

        # Always copy storyboard data — inherited shots are always preserved
        new_scene.storyboard_raw = source.storyboard_raw
        new_scene.style_guide = source.style_guide

        # Phase 12: Inherit production_bible_id and production_bible_version from source
        if source.production_bible_id is not None:
            new_scene.production_bible_id = source.production_bible_id
            new_scene.production_bible_version = source.production_bible_version

        session.add(new_scene)
        await session.commit()
        await session.refresh(new_scene)

        new_id = new_scene.id
        file_mgr = FileManager()
        file_mgr.get_scene_dir(new_id)  # ensure dirs exist

        # Build set of shots whose keyframes should be cleared
        clear_kf_set = set(request.clear_keyframes) if request.clear_keyframes else set()

        # Load source shots
        src_shots_result = await session.execute(
            select(Shot).where(Shot.scene_id == source.id).order_by(Shot.shot_index)
        )
        src_shots = src_shots_result.scalars().all()

        copied_shots = 0
        new_idx = 0  # renumbered index for non-deleted shots

        if resume_from != "pending":
            for src_shot in src_shots:
                idx = src_shot.shot_index

                # Skip deleted shots
                if idx in deleted_set:
                    continue

                # Apply shot-level edits
                shot_desc = src_shot.shot_description
                start_fp = src_shot.start_frame_prompt
                end_fp = src_shot.end_frame_prompt
                motion = src_shot.video_motion_prompt
                transition = src_shot.transition_notes

                if request.shot_edits and idx in request.shot_edits:
                    edits = request.shot_edits[idx]
                    shot_desc = edits.get("shot_description", shot_desc)
                    start_fp = edits.get("start_frame_prompt", start_fp)
                    end_fp = edits.get("end_frame_prompt", end_fp)
                    motion = edits.get("video_motion_prompt", motion)
                    transition = edits.get("transition_notes", transition)

                # Determine what to copy for this shot (using new_idx for boundary checks)
                if resume_from == "stitching":
                    new_shot_status = "video_done"
                    copy_keyframes = True
                    copy_clip = True
                elif resume_from == "video_gen":
                    if new_idx < shot_boundary:
                        new_shot_status = "video_done"
                        copy_keyframes = True
                        copy_clip = True
                    else:
                        new_shot_status = "keyframes_done"
                        copy_keyframes = True
                        copy_clip = False
                elif resume_from == "keyframing":
                    if new_idx < shot_boundary:
                        new_shot_status = "video_done"
                        copy_keyframes = True
                        copy_clip = True
                    else:
                        new_shot_status = "pending"
                        copy_keyframes = False
                        copy_clip = False
                else:
                    new_shot_status = "pending"
                    copy_keyframes = False
                    copy_clip = False

                # If keyframes are explicitly cleared for this shot, don't copy them
                if idx in clear_kf_set:
                    copy_keyframes = False
                    copy_clip = False
                    if new_shot_status not in ("pending",):
                        new_shot_status = "pending"

                new_shot = Shot(
                    scene_id=new_id,
                    shot_index=new_idx,
                    shot_description=shot_desc,
                    start_frame_prompt=start_fp,
                    end_frame_prompt=end_fp,
                    video_motion_prompt=motion,
                    transition_notes=transition,
                    status=new_shot_status,
                )
                session.add(new_shot)
                await session.flush()  # get new_shot.id

                # Copy keyframes
                if copy_keyframes:
                    kf_result = await session.execute(
                        select(Keyframe).where(Keyframe.shot_id == src_shot.id)
                    )
                    storage = get_storage_backend()
                    for kf in kf_result.scalars().all():
                        try:
                            src_data = await file_mgr.read_bytes(kf.file_path)
                        except FileNotFoundError:
                            continue
                        if isinstance(storage, LocalStorageBackend):
                            dst_path = file_mgr.get_scene_dir(new_id) / "keyframes" / f"shot_{new_idx}_{kf.position}.png"
                            dst_path.write_bytes(src_data)
                            stored_path = str(dst_path)
                        else:
                            kf_key = f"{new_id}/keyframes/shot_{new_idx}_{kf.position}.png"
                            await storage.put(kf_key, src_data, kf.mime_type or "image/png")
                            stored_path = kf_key
                        new_kf = Keyframe(
                            shot_id=new_shot.id,
                            position=kf.position,
                            prompt_used=kf.prompt_used,
                            file_path=stored_path,
                            mime_type=kf.mime_type,
                            source="inherited",
                        )
                        session.add(new_kf)

                # Copy clip
                if copy_clip:
                    clip_result = await session.execute(
                        select(VideoClip).where(VideoClip.shot_id == src_shot.id)
                    )
                    clip = clip_result.scalar_one_or_none()
                    if clip and clip.local_path:
                        try:
                            src_clip_data = await file_mgr.read_bytes(clip.local_path)
                        except FileNotFoundError:
                            src_clip_data = None
                        if src_clip_data:
                            storage = get_storage_backend()
                            if isinstance(storage, LocalStorageBackend):
                                dst_clip_path = file_mgr.get_scene_dir(new_id) / "clips" / f"shot_{new_idx}.mp4"
                                dst_clip_path.write_bytes(src_clip_data)
                                stored_clip_path = str(dst_clip_path)
                            else:
                                clip_key = f"{new_id}/clips/shot_{new_idx}.mp4"
                                await storage.put(clip_key, src_clip_data, "video/mp4")
                                stored_clip_path = clip_key
                            new_clip = VideoClip(
                                shot_id=new_shot.id,
                                source="inherited",
                                status="complete",
                                local_path=stored_clip_path,
                                gcs_uri=clip.gcs_uri,
                                duration_seconds=clip.duration_seconds,
                            )
                            session.add(new_clip)

                if new_shot_status != "pending":
                    copied_shots += 1

                new_idx += 1

            # Update storyboard_raw: apply edits, prune deleted shots, renumber
            if new_scene.storyboard_raw:
                sb = dict(new_scene.storyboard_raw)
                if "shots" in sb:
                    # Apply shot edits first (on original indices)
                    if request.shot_edits:
                        for idx_str, edits in request.shot_edits.items():
                            edit_idx = int(idx_str)
                            if edit_idx < len(sb["shots"]):
                                for field, value in edits.items():
                                    sb["shots"][edit_idx][field] = value
                    # Remove deleted shots (iterate in reverse to preserve indices)
                    if deleted_set:
                        sb["shots"] = [
                            s for i, s in enumerate(sb["shots"]) if i not in deleted_set
                        ]
                        # Renumber shot_index to match post-deletion DB indices
                        for new_idx_sb, shot_data in enumerate(sb["shots"]):
                            shot_data["shot_index"] = new_idx_sb
                    new_scene.storyboard_raw = sb

        # Phase 12: Copy production bible assets and shot manifests for forked scene
        if source.production_bible_id is not None:
            # Copy assets with inheritance tracking
            _, modified_asset_tags = await _copy_assets_for_fork(
                session,
                source.production_bible_id,
                source.id,
                new_id,
                request.asset_changes,
            )

            # Always copy shot manifests — inherited shots are always preserved
            await _copy_shot_manifests(
                session,
                source.id,
                new_id,
                shot_boundary,
                deleted_set,
            )

            await _copy_shot_audio_manifests(
                session,
                source.id,
                new_id,
                deleted_set,
            )

            # Process new uploads if any
            if (
                request.asset_changes is not None
                and request.asset_changes.new_uploads
            ):
                from vidpipe.services.manifesting_engine import ManifestingEngine

                # Collect existing face embeddings from inherited assets for cross-matching
                inherited_result = await session.execute(
                    select(Asset).where(
                        Asset.inherited_from_scene == new_id,
                        Asset.face_embedding.isnot(None),
                    )
                )
                inherited_face_assets = list(inherited_result.scalars().all())
                existing_face_embeddings = [
                    (a.id, a.face_embedding)
                    for a in inherited_face_assets
                    if a.face_embedding
                ]

                engine = ManifestingEngine(session)
                await engine.process_new_uploads(
                    manifest_id=source.production_bible_id,
                    new_uploads=request.asset_changes.new_uploads,
                    existing_face_embeddings=existing_face_embeddings,
                )

        await session.commit()

        # Sync local manifest files to S3 (no-op for local backend)
        from vidpipe.services.file_manager import sync_manifest_dir_to_s3
        await sync_manifest_dir_to_s3(str(source.production_bible_id), session)
        await session.commit()

    # Start pipeline in background
    background_tasks.add_task(run_pipeline_background, new_id)

    return ForkResponse(
        scene_id=str(new_id),
        forked_from=str(scene_id),
        status=resume_from,
        status_url=f"/api/scenes/{new_id}/status",
        copied_shots=copied_shots,
        resume_from=resume_from,
    )


@router.patch("/scenes/{scene_id}/edit", response_model=EditSceneResponse)
async def edit_scene_in_place(scene_id: uuid.UUID, body: EditSceneRequest):
    """Edit scene in-place with checkpoint (PipeSVN).

    Only allowed on terminal-status scenes. Uses optimistic concurrency
    via expected_sha to prevent lost updates.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Must be terminal
        if scene.status not in ("complete", "failed", "stopped", "staged", "draft"):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot edit scene in status '{scene.status}'"
            )

        # Optimistic concurrency check
        if body.expected_sha is not None and body.expected_sha != scene.head_sha:
            raise HTTPException(
                status_code=409,
                detail=f"Conflict: expected head_sha={body.expected_sha}, "
                       f"actual={scene.head_sha}. Another edit was committed."
            )

        changes = []

        # Apply scene-level field changes
        field_map = {
            "prompt": body.prompt,
            "title": body.title,
            "style": body.style,
            "aspect_ratio": body.aspect_ratio,
            "target_clip_duration": body.clip_duration,
            "target_shot_count": body.target_shot_count,
            "text_model": body.text_model,
            "image_model": body.image_model,
            "video_model": body.video_model,
            "vision_model": body.vision_model,
            "audio_enabled": body.audio_enabled,
        }
        for attr, value in field_map.items():
            if value is not None:
                old_val = getattr(scene, attr)
                if old_val != value:
                    setattr(scene, attr, value)
                    changes.append({"type": "scene_field", "field": attr, "old": str(old_val), "new": str(value)})

        # Handle production_bible_id change (requires UUID parsing + snapshot)
        if body.production_bible_id is not None:
            pb_uuid = uuid.UUID(body.production_bible_id)
            old_pb = scene.production_bible_id
            if old_pb != pb_uuid:
                manifest = await manifest_service.get_manifest(session, pb_uuid)
                if not manifest:
                    raise HTTPException(status_code=404, detail=f"Production Bible {body.production_bible_id} not found")
                scene.production_bible_id = pb_uuid
                scene.production_bible_version = manifest.version
                await manifest_service.create_snapshot(session, pb_uuid, scene.id)
                await manifest_service.increment_usage(session, pb_uuid)
                changes.append({"type": "scene_field", "field": "production_bible_id", "old": str(old_pb), "new": str(pb_uuid)})

        # Handle shot expansion (target_shot_count increase) — BEFORE shot edits
        # so that newly-created rows exist when shot_edits tries to find them.
        if body.target_shot_count is not None:
            existing_result = await session.execute(
                select(sa_func.count(Shot.id)).where(
                    Shot.scene_id == scene.id,
                    Shot.status != "removed",
                )
            )
            existing_count = existing_result.scalar() or 0
            if body.target_shot_count > existing_count:
                # Get max shot_index
                max_idx_result = await session.execute(
                    select(sa_func.max(Shot.shot_index)).where(
                        Shot.scene_id == scene.id
                    )
                )
                max_idx = max_idx_result.scalar() or 0
                for i in range(body.target_shot_count - existing_count):
                    new_idx = max_idx + 1 + i
                    new_shot = Shot(
                        scene_id=scene.id,
                        shot_index=new_idx,
                        shot_description="",
                        start_frame_prompt="",
                        end_frame_prompt="",
                        video_motion_prompt="",
                        status="pending",
                    )
                    session.add(new_shot)
                    changes.append({"type": "shot_added", "shot_index": new_idx})
            # Flush so new Shot rows are visible to subsequent queries
            await session.flush()

        # Apply shot edits
        if body.shot_edits:
            for shot_idx, edits in body.shot_edits.items():
                shot_result = await session.execute(
                    select(Shot).where(
                        Shot.scene_id == scene.id,
                        Shot.shot_index == int(shot_idx),
                    )
                )
                shot = shot_result.scalar_one_or_none()
                if not shot:
                    continue

                shot_field_map = {
                    "shot_description": edits.shot_description,
                    "start_frame_prompt": edits.start_frame_prompt,
                    "end_frame_prompt": edits.end_frame_prompt,
                    "video_motion_prompt": edits.video_motion_prompt,
                    "transition_notes": edits.transition_notes,
                }
                for attr, value in shot_field_map.items():
                    if value is not None:
                        old_val = getattr(shot, attr)
                        if old_val != value:
                            setattr(shot, attr, value)
                            changes.append({
                                "type": "shot_field",
                                "shot_index": int(shot_idx),
                                "field": attr,
                            })

        # Handle removed shots (set status to "removed")
        if body.removed_shots:
            for shot_idx in body.removed_shots:
                shot_result = await session.execute(
                    select(Shot).where(
                        Shot.scene_id == scene.id,
                        Shot.shot_index == shot_idx,
                    )
                )
                shot = shot_result.scalar_one_or_none()
                if shot:
                    shot.status = "removed"
                    changes.append({"type": "shot_removed", "shot_index": shot_idx})

        # Reorder shots
        if body.shot_order is not None:
            # Fetch all active shots for this scene
            shot_result = await session.execute(
                select(Shot).where(
                    Shot.scene_id == scene.id,
                    Shot.status != "removed",
                )
            )
            active_shots = {s.shot_index: s for s in shot_result.scalars().all()}

            # Validate: all indices in shot_order must exist
            for idx in body.shot_order:
                if idx not in active_shots:
                    raise HTTPException(400, f"shot_order references unknown shot_index {idx}")

            # Two-pass renumber to avoid composite PK collisions
            # Pass 1: assign temporary negative indices
            temp_mapping = {}  # old_index -> temp_index
            for i, old_idx in enumerate(body.shot_order):
                temp_idx = -(i + 1)
                temp_mapping[old_idx] = temp_idx

            for old_idx, temp_idx in temp_mapping.items():
                active_shots[old_idx].shot_index = temp_idx
                await session.execute(
                    update(ShotManifestModel).where(
                        ShotManifestModel.scene_id == scene.id,
                        ShotManifestModel.shot_index == old_idx,
                    ).values(shot_index=temp_idx)
                )
                await session.execute(
                    update(ShotAudioManifestModel).where(
                        ShotAudioManifestModel.scene_id == scene.id,
                        ShotAudioManifestModel.shot_index == old_idx,
                    ).values(shot_index=temp_idx)
                )
                await session.execute(
                    update(AssetAppearance).where(
                        AssetAppearance.scene_id == scene.id,
                        AssetAppearance.shot_index == old_idx,
                    ).values(shot_index=temp_idx)
                )

            await session.flush()

            # Pass 2: assign final indices (0, 1, 2, ...)
            for new_idx, old_idx in enumerate(body.shot_order):
                temp_idx = temp_mapping[old_idx]
                active_shots[old_idx].shot_index = new_idx
                await session.execute(
                    update(ShotManifestModel).where(
                        ShotManifestModel.scene_id == scene.id,
                        ShotManifestModel.shot_index == temp_idx,
                    ).values(shot_index=new_idx)
                )
                await session.execute(
                    update(ShotAudioManifestModel).where(
                        ShotAudioManifestModel.scene_id == scene.id,
                        ShotAudioManifestModel.shot_index == temp_idx,
                    ).values(shot_index=new_idx)
                )
                await session.execute(
                    update(AssetAppearance).where(
                        AssetAppearance.scene_id == scene.id,
                        AssetAppearance.shot_index == temp_idx,
                    ).values(shot_index=new_idx)
                )

            await session.flush()
            changes.append({"type": "shots_reordered", "new_order": body.shot_order})

        if not changes:
            raise HTTPException(status_code=400, detail="No changes to commit")

        # Create checkpoint
        from vidpipe.services.checkpoint_service import create_checkpoint
        message = body.commit_message or f"Edit: {len(changes)} change(s)"
        checkpoint = await create_checkpoint(
            session, scene, message, metadata={"changes": changes}
        )

        await session.commit()

        return EditSceneResponse(
            scene_id=str(scene.id),
            head_sha=checkpoint.sha,
            message=message,
            changes_count=len(changes),
        )


@router.get("/scenes/{scene_id}/download")
async def download_video(scene_id: uuid.UUID, dl: int = 0):
    """Download or stream final MP4 video file.

    Query params:
    - dl=1: force Content-Disposition: attachment (browser download).
      Default serves inline for <video> element streaming.
    - v=<sha>: ignored, used as client-side cache buster.

    Returns 409 if scene is not complete.
    Returns 404 if output file does not exist.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()

        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Check if scene is complete
        if scene.status != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Scene not ready for download (status: {scene.status})"
            )

        # Check if output file exists
        if not scene.output_path:
            raise HTTPException(status_code=404, detail="Output file path not set")

        storage = get_storage_backend()
        filename = f"video_{scene_id}.mp4"
        disposition = "attachment" if dl else "inline"

        if isinstance(storage, LocalStorageBackend):
            output_path = Path(scene.output_path)
            if not output_path.exists():
                raise HTTPException(status_code=404, detail="Output file not found on disk")
            return FileResponse(
                path=str(output_path),
                media_type="video/mp4",
                filename=filename,
                headers={
                    "Content-Disposition": f'{disposition}; filename="{filename}"'
                }
            )
        else:
            file_mgr = FileManager()
            data = await file_mgr.read_bytes(scene.output_path)
            return Response(
                content=data,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f'{disposition}; filename="{filename}"'
                },
            )


@router.get("/keyframes/{keyframe_id}")
async def get_keyframe_image(keyframe_id: uuid.UUID):
    """Serve a keyframe image by its database ID."""
    async with async_session() as session:
        result = await session.execute(
            select(Keyframe).where(Keyframe.id == keyframe_id)
        )
        keyframe = result.scalar_one_or_none()

        if not keyframe:
            raise HTTPException(status_code=404, detail="Keyframe not found")

        storage = get_storage_backend()
        if isinstance(storage, LocalStorageBackend):
            file_path = Path(keyframe.file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Keyframe file not found on disk")
            return FileResponse(
                path=str(file_path),
                media_type=keyframe.mime_type,
            )
        else:
            file_mgr = FileManager()
            data = await file_mgr.read_bytes(keyframe.file_path)
            return Response(content=data, media_type=keyframe.mime_type)


@router.get("/clips/{clip_id}")
async def get_clip_video(clip_id: uuid.UUID):
    """Serve a video clip by its database ID."""
    async with async_session() as session:
        result = await session.execute(
            select(VideoClip).where(VideoClip.id == clip_id)
        )
        clip = result.scalar_one_or_none()

        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")

        if not clip.local_path:
            raise HTTPException(status_code=404, detail="Clip file path not set")

        storage = get_storage_backend()
        if isinstance(storage, LocalStorageBackend):
            file_path = Path(clip.local_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Clip file not found on disk")
            return FileResponse(
                path=str(file_path),
                media_type="video/mp4",
            )
        else:
            file_mgr = FileManager()
            data = await file_mgr.read_bytes(clip.local_path)
            return Response(content=data, media_type="video/mp4")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Aggregate metrics across all scenes.

    Cost estimation is artifact-based: complete scenes use theoretical full
    cost; incomplete scenes count only actually-generated keyframes and
    completed video clips.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.deleted_at.is_(None))
        )
        scenes = result.scalars().all()

        # Build lookup of actual artifacts per scene for cost accuracy.
        # generated keyframes per scene (only source='generated', not 'inherited')
        kf_q = await session.execute(
            select(
                Shot.scene_id,
                sa_func.count(Keyframe.id),
            )
            .join(Shot, Keyframe.shot_id == Shot.id)
            .where(Keyframe.source == "generated")
            .group_by(Shot.scene_id)
        )
        kf_counts: dict[uuid.UUID, int] = {row[0]: row[1] for row in kf_q}

        # completed video clips per scene (only source='generated')
        vc_q = await session.execute(
            select(
                Shot.scene_id,
                sa_func.count(VideoClip.id),
            )
            .join(Shot, VideoClip.shot_id == Shot.id)
            .where(VideoClip.status == "complete")
            .where(VideoClip.source == "generated")
            .group_by(Shot.scene_id)
        )
        vc_counts: dict[uuid.UUID, int] = {row[0]: row[1] for row in vc_q}

        # Billed Veo submissions per scene: all clips with operation_name
        # (Google charges even for failed/filtered operations)
        submission_expr = case(
            (VideoClip.veo_submission_count > 0, VideoClip.veo_submission_count),
            else_=1,  # backward compat: old rows with operation_name but count=0
        )
        billed_q = await session.execute(
            select(
                Shot.scene_id,
                sa_func.sum(submission_expr),
            )
            .join(Shot, VideoClip.shot_id == Shot.id)
            .where(VideoClip.operation_name.isnot(None))
            .where(VideoClip.source == "generated")
            .group_by(Shot.scene_id)
        )
        billed_counts: dict[uuid.UUID, int] = {row[0]: int(row[1]) for row in billed_q}

        # Safety regen image calls per scene
        regen_q = await session.execute(
            select(
                Shot.scene_id,
                sa_func.sum(VideoClip.safety_regen_count),
            )
            .join(Shot, VideoClip.shot_id == Shot.id)
            .where(VideoClip.safety_regen_count > 0)
            .group_by(Shot.scene_id)
        )
        regen_counts: dict[uuid.UUID, int] = {row[0]: int(row[1]) for row in regen_q}

        # actual shot counts per scene
        sc_q = await session.execute(
            select(
                Shot.scene_id,
                sa_func.count(Shot.id),
            )
            .group_by(Shot.scene_id)
        )
        shot_counts_per_scene: dict[uuid.UUID, int] = {row[0]: row[1] for row in sc_q}

    status_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    aspect_ratio_counts: Counter[str] = Counter()
    text_model_counts: Counter[str] = Counter()
    image_model_counts: Counter[str] = Counter()
    video_model_counts: Counter[str] = Counter()
    audio_counts: Counter[str] = Counter()
    shot_count_counts: Counter[str] = Counter()
    total_cost = 0.0
    total_seconds = 0
    clip_durations: list[int] = []

    for p in scenes:
        status_counts[p.status] += 1
        style_counts[p.style] += 1
        aspect_ratio_counts[p.aspect_ratio] += 1
        if p.text_model:
            text_model_counts[p.text_model] += 1
        if p.image_model:
            image_model_counts[p.image_model] += 1
        if p.video_model:
            video_model_counts[p.video_model] += 1
        audio_counts["enabled" if p.audio_enabled else "disabled"] += 1
        sc = shot_counts_per_scene.get(p.id, 0)
        if sc > 0:
            shot_count_counts[str(sc)] += 1
        if p.total_duration:
            total_seconds += p.total_duration
        if p.target_clip_duration:
            clip_durations.append(p.target_clip_duration)
        total_cost += _estimate_scene_cost(
            p,
            generated_keyframes=kf_counts.get(p.id, 0),
            completed_clips=vc_counts.get(p.id, 0),
            generated_clips=vc_counts.get(p.id, 0),
            has_storyboard=p.storyboard_raw is not None,
            billed_veo_submissions=billed_counts.get(p.id, 0),
            extra_image_regens=regen_counts.get(p.id, 0),
        )

    avg_clip = sum(clip_durations) / len(clip_durations) if clip_durations else None

    return MetricsResponse(
        total_scenes=len(scenes),
        status_counts=dict(status_counts),
        style_counts=dict(style_counts),
        aspect_ratio_counts=dict(aspect_ratio_counts),
        text_model_counts=dict(text_model_counts),
        image_model_counts=dict(image_model_counts),
        video_model_counts=dict(video_model_counts),
        audio_counts=dict(audio_counts),
        shot_count_counts=dict(shot_count_counts),
        total_estimated_cost=round(total_cost, 2),
        total_video_seconds=total_seconds,
        avg_clip_duration=round(avg_clip, 1) if avg_clip is not None else None,
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0"
    }


# ============================================================================
# Production Bible API Endpoints
# ============================================================================

def _manifest_to_list_item(m: ProductionBible) -> ProductionBibleListItem:
    """Convert ProductionBible ORM model to ProductionBibleListItem response."""
    return ProductionBibleListItem(
        production_bible_id=str(m.id),
        name=m.name,
        description=m.description,
        thumbnail_url=m.thumbnail_url,
        category=m.category,
        tags=m.tags,
        status=m.status,
        asset_count=m.asset_count,
        times_used=m.times_used,
        last_used_at=m.last_used_at.isoformat() if m.last_used_at else None,
        version=m.version,
        created_at=m.created_at.isoformat(),
        updated_at=m.updated_at.isoformat(),
    )


def _asset_to_response(a: Asset) -> AssetResponse:
    """Convert Asset ORM model to AssetResponse."""
    return AssetResponse(
        asset_id=str(a.id),
        production_bible_id=str(a.production_bible_id),
        asset_type=a.asset_type,
        name=a.name,
        manifest_tag=a.manifest_tag,
        user_tags=a.user_tags,
        reference_image_url=f"/api/assets/{a.id}/image" if a.reference_image_url else None,
        thumbnail_url=a.thumbnail_url,
        description=a.description,
        source=a.source,
        sort_order=a.sort_order,
        created_at=a.created_at.isoformat(),
        # Phase 5 fields
        reverse_prompt=a.reverse_prompt,
        visual_description=a.visual_description,
        detection_class=a.detection_class,
        detection_confidence=a.detection_confidence,
        is_face_crop=a.is_face_crop,
        quality_score=a.quality_score,
    )


@router.post("/production-bibles", status_code=201, response_model=ProductionBibleListItem)
async def create_production_bible(request: CreateProductionBibleRequest):
    """Create a new production bible in DRAFT status."""
    async with async_session() as session:
        try:
            manifest = await manifest_service.create_manifest(
                session,
                name=request.name,
                description=request.description,
                category=request.category,
                tags=request.tags,
            )
            await session.commit()
            await session.refresh(manifest)
            return _manifest_to_list_item(manifest)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))


@router.post("/production-bibles/from-scene", status_code=201, response_model=ProductionBibleDetailResponse)
async def create_production_bible_from_scene(request: CreateProductionBibleFromSceneRequest):
    """Create a production bible pre-populated from a scene's storyboard data."""
    try:
        scene_id = uuid.UUID(request.scene_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid scene_id format")

    async with async_session() as session:
        try:
            manifest, assets = await manifest_service.create_manifest_from_scene(
                session,
                scene_id=scene_id,
                name=request.name,
            )
            await session.commit()
            await session.refresh(manifest)
            for a in assets:
                await session.refresh(a)

            return ProductionBibleDetailResponse(
                production_bible_id=str(manifest.id),
                name=manifest.name,
                description=manifest.description,
                thumbnail_url=manifest.thumbnail_url,
                category=manifest.category,
                tags=manifest.tags,
                status=manifest.status,
                processing_progress=manifest.processing_progress,
                contact_sheet_url=manifest.contact_sheet_url,
                asset_count=manifest.asset_count,
                total_processing_cost=manifest.total_processing_cost,
                times_used=manifest.times_used,
                last_used_at=manifest.last_used_at.isoformat() if manifest.last_used_at else None,
                version=manifest.version,
                parent_production_bible_id=str(manifest.parent_production_bible_id) if manifest.parent_production_bible_id else None,
                source_video_duration=manifest.source_video_duration,
                created_at=manifest.created_at.isoformat(),
                updated_at=manifest.updated_at.isoformat(),
                assets=[_asset_to_response(a) for a in assets],
            )
        except ValueError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@router.get("/production-bibles", response_model=list[ProductionBibleListItem])
async def list_production_bibles(
    category: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "updated_at",
    sort_order: str = "desc",
):
    """List production bibles with optional filters and sorting."""
    async with async_session() as session:
        manifests = await manifest_service.list_manifests(
            session,
            category=category,
            status=status,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return [_manifest_to_list_item(m) for m in manifests]


@router.get("/production-bibles/{production_bible_id}", response_model=ProductionBibleDetailResponse)
async def get_production_bible_detail(production_bible_id: uuid.UUID):
    """Get production bible detail with assets."""
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, production_bible_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        assets = await manifest_service.list_assets(session, production_bible_id)

        return ProductionBibleDetailResponse(
            production_bible_id=str(manifest.id),
            name=manifest.name,
            description=manifest.description,
            thumbnail_url=manifest.thumbnail_url,
            category=manifest.category,
            tags=manifest.tags,
            status=manifest.status,
            processing_progress=manifest.processing_progress,
            contact_sheet_url=manifest.contact_sheet_url,
            asset_count=manifest.asset_count,
            total_processing_cost=manifest.total_processing_cost,
            times_used=manifest.times_used,
            last_used_at=manifest.last_used_at.isoformat() if manifest.last_used_at else None,
            version=manifest.version,
            parent_production_bible_id=str(manifest.parent_production_bible_id) if manifest.parent_production_bible_id else None,
            source_video_duration=manifest.source_video_duration,
            created_at=manifest.created_at.isoformat(),
            updated_at=manifest.updated_at.isoformat(),
            assets=[_asset_to_response(a) for a in assets],
        )


@router.put("/production-bibles/{production_bible_id}", response_model=ProductionBibleListItem)
async def update_production_bible(production_bible_id: uuid.UUID, request: UpdateProductionBibleRequest):
    """Update production bible fields."""
    async with async_session() as session:
        try:
            # Build kwargs from non-None fields
            kwargs = {k: v for k, v in request.model_dump().items() if v is not None}
            manifest = await manifest_service.update_manifest(
                session,
                production_bible_id,
                **kwargs,
            )
            await session.commit()
            await session.refresh(manifest)
            return _manifest_to_list_item(manifest)
        except ValueError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@router.delete("/production-bibles/{production_bible_id}")
async def delete_production_bible(production_bible_id: uuid.UUID):
    """Soft delete production bible. Returns 409 if referenced by scenes."""
    async with async_session() as session:
        try:
            await manifest_service.delete_manifest(session, production_bible_id)
            await session.commit()
            return {"status": "deleted", "production_bible_id": str(production_bible_id)}
        except ValueError as e:
            error_msg = str(e)
            if "not found" in error_msg:
                raise HTTPException(status_code=404, detail=error_msg)
            if "referenced by" in error_msg:
                raise HTTPException(status_code=409, detail=error_msg)
            raise HTTPException(status_code=422, detail=error_msg)


@router.post("/production-bibles/{production_bible_id}/duplicate", status_code=201, response_model=ProductionBibleListItem)
async def duplicate_production_bible(production_bible_id: uuid.UUID, name: Optional[str] = None):
    """Duplicate production bible with all assets."""
    async with async_session() as session:
        try:
            new_manifest = await manifest_service.duplicate_manifest(
                session,
                production_bible_id,
                new_name=name,
            )
            await session.commit()
            await session.refresh(new_manifest)
            return _manifest_to_list_item(new_manifest)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("/production-bibles/{production_bible_id}/assets", status_code=201, response_model=AssetResponse)
async def create_asset(production_bible_id: uuid.UUID, request: CreateAssetRequest):
    """Create asset within production bible."""
    async with async_session() as session:
        try:
            asset = await manifest_service.create_asset(
                session,
                manifest_id=production_bible_id,
                name=request.name,
                asset_type=request.asset_type,
                description=request.description,
                user_tags=request.user_tags,
            )
            await session.commit()
            await session.refresh(asset)
            return _asset_to_response(asset)
        except ValueError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@router.get("/production-bibles/{production_bible_id}/assets", response_model=list[AssetResponse])
async def list_production_bible_assets(production_bible_id: uuid.UUID):
    """List assets for production bible."""
    async with async_session() as session:
        assets = await manifest_service.list_assets(session, production_bible_id)
        return [_asset_to_response(a) for a in assets]


@router.put("/assets/{asset_id}", response_model=AssetResponse)
async def update_asset(asset_id: uuid.UUID, request: UpdateAssetRequest):
    """Update asset fields."""
    async with async_session() as session:
        try:
            kwargs = {k: v for k, v in request.model_dump().items() if v is not None}
            asset = await manifest_service.update_asset(
                session,
                asset_id,
                **kwargs,
            )
            await session.commit()
            await session.refresh(asset)
            return _asset_to_response(asset)
        except ValueError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: uuid.UUID):
    """Delete asset."""
    async with async_session() as session:
        try:
            await manifest_service.delete_asset(session, asset_id)
            await session.commit()
            return {"status": "deleted", "asset_id": str(asset_id)}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("/assets/{asset_id}/upload", response_model=AssetResponse)
async def upload_asset_image(asset_id: uuid.UUID, file: UploadFile = File(...)):
    """Upload image file for an asset."""
    # Validate content type
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be image/png, image/jpeg, or image/webp",
        )

    # Read file content
    content = await file.read()

    # Validate file size (max 10MB)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 10MB")

    async with async_session() as session:
        # Get asset to retrieve manifest_id
        asset = await manifest_service.get_asset(session, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        storage = get_storage_backend()
        filename = file.filename or "upload.png"

        if isinstance(storage, LocalStorageBackend):
            # Local: write to disk (sync)
            file_path = await asyncio.to_thread(
                manifest_service.save_asset_image,
                asset.production_bible_id,
                asset_id,
                content,
                filename,
            )
            # reference_image_url stays as API path for local
            asset.reference_image_url = f"/api/assets/{asset_id}/image"
        else:
            # S3: upload to storage + dual-write local for pipeline reads
            key = f"manifests/{asset.production_bible_id}/uploads/{asset_id}_{filename}"
            await storage.put(key, content, file.content_type or "image/png")
            # Keep local copy so pipeline (CV, ffmpeg) can read without S3 fetch
            from vidpipe.config import settings as _settings
            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            # Store the S3 key — GET endpoint will resolve to public URL
            asset.reference_image_url = key

        await session.commit()
        await session.refresh(asset)

        return _asset_to_response(asset)


async def _find_asset_s3_key(
    storage: StorageBackend, pb_id: uuid.UUID, asset_id: uuid.UUID
) -> str | None:
    """Search S3 for an asset image by probing uploads/ and crops/ prefixes."""
    for subdir in ("uploads", "crops"):
        prefix = f"manifests/{pb_id}/{subdir}/{asset_id}_"
        keys = await storage.list_prefix(prefix, limit=1)
        if keys:
            return keys[0]
    return None


def _media_type_from_key(key: str) -> str:
    """Derive MIME type from a storage key's extension."""
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")


@router.get("/assets/{asset_id}/image")
async def get_asset_image(asset_id: uuid.UUID):
    """Serve an asset's uploaded image by asset ID."""
    async with async_session() as session:
        asset = await manifest_service.get_asset(session, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        storage = get_storage_backend()

        if isinstance(storage, LocalStorageBackend):
            # Local backend: find file in uploads or crops directory
            manifest_dir = Path("tmp/manifests") / str(asset.production_bible_id)
            matches = []
            for subdir in ("uploads", "crops"):
                d = manifest_dir / subdir
                if d.exists():
                    matches = list(d.glob(f"{asset_id}_*"))
                    if matches:
                        break
            if not matches:
                raise HTTPException(status_code=404, detail="Asset image not found on disk")

            file_path = matches[0]
            return FileResponse(
                path=str(file_path),
                media_type=_media_type_from_key(file_path.name),
            )
        else:
            # S3 backend: if reference_image_url is already an S3 key, use it directly
            ref_url = asset.reference_image_url
            if ref_url and not ref_url.startswith("/"):
                try:
                    data = await storage.get(ref_url)
                    return Response(content=data, media_type=_media_type_from_key(ref_url))
                except FileNotFoundError:
                    pass  # Fall through to prefix search

            # Legacy /api/assets/... URL or missing key — search S3 by prefix
            s3_key = await _find_asset_s3_key(storage, asset.production_bible_id, asset_id)
            if s3_key:
                # Lazy-update the DB to the S3 key for future requests
                if ref_url != s3_key:
                    asset.reference_image_url = s3_key
                    await session.commit()
                data = await storage.get(s3_key)
                return Response(content=data, media_type=_media_type_from_key(s3_key))

            raise HTTPException(status_code=404, detail="Asset image not found")


# ============================================================================
# Phase 5: Manifesting Engine Endpoints
# ============================================================================

@router.post("/production-bibles/{production_bible_id}/process", status_code=202)
async def process_production_bible(production_bible_id: uuid.UUID):
    """Trigger manifesting pipeline for a production bible.

    Returns 202 Accepted immediately and runs processing in background.
    """
    manifest_id = production_bible_id
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        # Verify status allows processing (DRAFT, READY, or ERROR for retry)
        if manifest.status not in ("DRAFT", "READY", "ERROR"):
            raise HTTPException(
                status_code=422,
                detail=f"Cannot process production bible in status {manifest.status}. Must be DRAFT, READY, or ERROR."
            )

        # Set status to PROCESSING
        manifest.status = "PROCESSING"
        await session.commit()

    # Start background task
    asyncio.create_task(process_manifest_task(str(manifest_id)))

    return {
        "task_id": f"manifest_{manifest_id}",
        "status": "started",
        "production_bible_id": str(manifest_id),
    }


@router.post("/production-bibles/{production_bible_id}/finalize", status_code=200)
async def finalize_production_bible(production_bible_id: uuid.UUID):
    """Finalize a bindings-only production bible (DRAFT -> READY).

    Use when a bible has Asset Library bindings but no uploaded images
    that need the full manifesting pipeline.
    """
    from sqlalchemy import func as sa_func
    from vidpipe.db.models import CastBinding, SetBinding, PropBinding, SoundBinding

    manifest_id = production_bible_id
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        if manifest.status not in ("DRAFT", "ERROR"):
            raise HTTPException(
                status_code=422,
                detail=f"Cannot finalize production bible in status {manifest.status}. Must be DRAFT or ERROR.",
            )

        # Count bindings to ensure there's something to finalize
        binding_count = 0
        for model in (CastBinding, SetBinding, PropBinding, SoundBinding):
            result = await session.execute(
                select(sa_func.count(model.id)).where(model.production_bible_id == manifest_id)
            )
            binding_count += result.scalar() or 0

        # Also count uploaded assets
        asset_count_result = await session.execute(
            select(sa_func.count(Asset.id)).where(Asset.production_bible_id == manifest_id)
        )
        asset_count = asset_count_result.scalar() or 0

        if binding_count == 0 and asset_count == 0:
            raise HTTPException(
                status_code=422,
                detail="Nothing to finalize — add bindings or assets first.",
            )

        manifest.status = "READY"
        manifest.asset_count = asset_count
        await session.commit()

    return {
        "status": "ready",
        "production_bible_id": str(manifest_id),
    }


@router.get("/production-bibles/{production_bible_id}/progress", response_model=ProcessingProgressResponse)
async def get_production_bible_progress(production_bible_id: uuid.UUID):
    """Get processing progress for a production bible."""
    manifest_id = production_bible_id
    task_id = f"manifest_{manifest_id}"

    # Check in-memory task status
    if task_id in TASK_STATUS:
        return ProcessingProgressResponse(**TASK_STATUS[task_id])

    # If not in memory, check database status
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        if manifest.status == "READY":
            return ProcessingProgressResponse(
                status="complete",
                current_step="complete",
                progress={},
            )
        elif manifest.status == "ERROR":
            return ProcessingProgressResponse(
                status="error",
                current_step="unknown",
                error="Processing failed (see server logs)",
            )
        else:
            return ProcessingProgressResponse(
                status="not_started",
                current_step=None,
                progress={},
            )


ALLOWED_VIDEO_MIMES = {"video/mp4", "video/quicktime", "video/webm"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


@router.post("/production-bibles/{production_bible_id}/upload-video", status_code=202, response_model=VideoUploadResponse)
async def upload_video_for_production_bible(
    production_bible_id: uuid.UUID,
    file: UploadFile = File(...),
):
    """Upload a video file to extract frames for production bible creation.

    Accepts MP4, MOV, WebM up to 200MB. Returns 202 and runs extraction
    in background.
    """
    from vidpipe.config import settings

    manifest_id = production_bible_id
    max_size = settings.cv_analysis.max_video_file_size_mb * 1024 * 1024

    # Validate production bible exists and is DRAFT
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Production Bible not found")
        if manifest.status not in ("DRAFT", "READY", "ERROR"):
            raise HTTPException(
                status_code=422,
                detail=f"Cannot upload video to production bible in status {manifest.status}. Must be DRAFT, READY, or ERROR."
            )

    # Validate file type
    content_type = file.content_type or ""
    ext = Path(file.filename or "").suffix.lower()
    if content_type not in ALLOWED_VIDEO_MIMES and ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format. Accepted: MP4, MOV, WebM"
        )

    # Read and validate size
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=422,
            detail=f"Video exceeds {settings.cv_analysis.max_video_file_size_mb}MB limit"
        )

    # Save video to disk
    video_dir = Path("tmp/manifests") / str(manifest_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_ext = ext if ext in ALLOWED_VIDEO_EXTENSIONS else ".mp4"
    video_path = video_dir / f"source_video{video_ext}"
    video_path.write_bytes(content)

    # Update production bible status
    async with async_session() as session:
        pb = await session.get(ProductionBible, manifest_id)
        if pb:
            pb.status = "EXTRACTING"
            pb.source_video_path = str(video_path)
            await session.commit()

    # Spawn background task
    asyncio.create_task(extract_video_frames_task(str(manifest_id), str(video_path)))

    return VideoUploadResponse(
        task_id=f"extract_{manifest_id}",
        status="started",
        production_bible_id=str(manifest_id),
    )


@router.get("/production-bibles/{production_bible_id}/contact-sheet")
async def get_production_bible_contact_sheet(production_bible_id: uuid.UUID):
    """Serve the contact sheet image for a production bible."""
    contact_sheet_path = Path("tmp/manifests") / str(production_bible_id) / "contact_sheet.jpg"
    if contact_sheet_path.exists():
        return FileResponse(path=str(contact_sheet_path), media_type="image/jpeg")

    # S3 fallback: try storage backend
    storage = get_storage_backend()
    if not storage.is_local():
        s3_key = f"manifests/{production_bible_id}/contact_sheet.jpg"
        try:
            data = await storage.get(s3_key)
            return Response(content=data, media_type="image/jpeg")
        except FileNotFoundError:
            pass

    raise HTTPException(status_code=404, detail="Contact sheet not found. Run /process first.")


@router.get("/production-bibles/{production_bible_id}/extraction-progress", response_model=ProcessingProgressResponse)
async def get_production_bible_extraction_progress(production_bible_id: uuid.UUID):
    """Get video frame extraction progress for a production bible."""
    manifest_id = production_bible_id
    task_id = f"extract_{manifest_id}"

    if task_id in TASK_STATUS:
        task_data = TASK_STATUS[task_id]
        return ProcessingProgressResponse(
            status=task_data.get("status", "extracting"),
            current_step=task_data.get("current_step"),
            progress=task_data.get("progress"),
            error=task_data.get("error"),
        )

    # Check database status
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        if manifest.status == "EXTRACTING":
            return ProcessingProgressResponse(
                status="extracting",
                current_step="initializing",
                progress={},
            )
        elif manifest.status == "ERROR":
            return ProcessingProgressResponse(
                status="error",
                current_step="unknown",
                error="Extraction failed (see server logs)",
            )
        else:
            # DRAFT with video means extraction completed
            return ProcessingProgressResponse(
                status="complete",
                current_step="complete",
                progress={},
            )


@router.post("/assets/{asset_id}/reprocess", response_model=AssetResponse)
async def reprocess_asset(asset_id: uuid.UUID):
    """Reprocess a single asset (re-run YOLO detection and reverse-prompting)."""
    from vidpipe.services.manifesting_engine import ManifestingEngine

    async with async_session() as session:
        asset = await manifest_service.get_asset(session, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        # Create engine with fresh session and reprocess
        engine = ManifestingEngine(session)
        updated = await engine.reprocess_asset(asset_id)

        return _asset_to_response(updated)


# ============================================================================
# Phase 11: Multi-Candidate Quality Mode Endpoints
# ============================================================================

@router.get("/scenes/{scene_id}/shots/{shot_idx}/candidates")
async def list_candidates(scene_id: str, shot_idx: int):
    """List all generation candidates for a specific shot with scores."""
    async with async_session() as session:
        result = await session.execute(
            select(GenerationCandidate)
            .where(
                GenerationCandidate.scene_id == uuid.UUID(scene_id),
                GenerationCandidate.shot_index == shot_idx,
            )
            .order_by(GenerationCandidate.candidate_number)
        )
        candidates = result.scalars().all()
        return [
            CandidateResponse(
                candidate_id=str(c.id),
                candidate_number=c.candidate_number,
                local_path=c.local_path,
                manifest_adherence_score=c.manifest_adherence_score,
                visual_quality_score=c.visual_quality_score,
                continuity_score=c.continuity_score,
                prompt_adherence_score=c.prompt_adherence_score,
                composite_score=c.composite_score,
                scoring_details=c.scoring_details,
                is_selected=c.is_selected,
                selected_by=c.selected_by,
                generation_cost=c.generation_cost,
                scoring_cost=c.scoring_cost,
                created_at=str(c.created_at),
            )
            for c in candidates
        ]


@router.put("/scenes/{scene_id}/shots/{shot_idx}/candidates/{candidate_id}/select")
async def select_candidate(scene_id: str, shot_idx: int, candidate_id: str):
    """Manually override auto-selection for a shot's candidate.

    CRITICAL: Updates BOTH GenerationCandidate.is_selected AND VideoClip.local_path.
    The stitcher reads VideoClip.local_path, so both must be consistent.
    """
    async with async_session() as session:
        # Load all candidates for this shot
        all_result = await session.execute(
            select(GenerationCandidate).where(
                GenerationCandidate.scene_id == uuid.UUID(scene_id),
                GenerationCandidate.shot_index == shot_idx,
            )
        )
        all_candidates = all_result.scalars().all()
        if not all_candidates:
            raise HTTPException(404, "No candidates found for this shot")

        # Find the chosen candidate
        chosen = next(
            (c for c in all_candidates if str(c.id) == candidate_id),
            None,
        )
        if not chosen:
            raise HTTPException(404, "Candidate not found")

        # Deselect all, then select chosen
        for c in all_candidates:
            c.is_selected = False
        chosen.is_selected = True
        chosen.selected_by = "user"

        # CRITICAL: Update VideoClip.local_path to point to selected candidate
        shot_result = await session.execute(
            select(Shot).where(
                Shot.scene_id == uuid.UUID(scene_id),
                Shot.shot_index == shot_idx,
            )
        )
        shot = shot_result.scalar_one_or_none()
        if shot:
            clip_result = await session.execute(
                select(VideoClip).where(VideoClip.shot_id == shot.id)
            )
            clip = clip_result.scalar_one_or_none()
            if clip:
                clip.local_path = chosen.local_path

        await session.commit()
        return {"selected": candidate_id, "selected_by": "user"}


# ============================================================================
# User Settings
# ============================================================================

class UserSettingsResponse(BaseModel):
    """Response schema for GET /api/settings."""
    enabled_text_models: Optional[list[str]] = None
    enabled_image_models: Optional[list[str]] = None
    enabled_video_models: Optional[list[str]] = None
    default_text_model: Optional[str] = None
    default_image_model: Optional[str] = None
    default_video_model: Optional[str] = None
    default_vision_model: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    has_api_key: bool = False
    has_gcp_service_account: bool = False
    comfyui_host: Optional[str] = None
    has_comfyui_key: bool = False
    comfyui_cost_per_second: Optional[float] = None
    # Phase 13: Ollama configuration
    ollama_use_cloud: bool = False
    has_ollama_key: bool = False
    ollama_endpoint: Optional[str] = None
    ollama_models: Optional[list] = None
    # Display preferences
    show_cost: bool = True
    # ElevenLabs configuration
    has_elevenlabs_key: bool = False
    default_voice_id: Optional[str] = None


class UserSettingsUpdate(BaseModel):
    """Request schema for PUT /api/settings."""
    enabled_text_models: Optional[list[str]] = None
    enabled_image_models: Optional[list[str]] = None
    enabled_video_models: Optional[list[str]] = None
    default_text_model: Optional[str] = None
    default_image_model: Optional[str] = None
    default_video_model: Optional[str] = None
    default_vision_model: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    vertex_api_key: Optional[str] = None
    clear_api_key: bool = False
    gcp_service_account_json: Optional[str] = None
    clear_gcp_service_account: bool = False
    comfyui_host: Optional[str] = None
    comfyui_api_key: Optional[str] = None
    clear_comfyui_key: bool = False
    comfyui_cost_per_second: Optional[float] = None
    # Phase 13: Ollama configuration
    ollama_use_cloud: Optional[bool] = None
    ollama_api_key: Optional[str] = None
    clear_ollama_key: Optional[bool] = None
    ollama_endpoint: Optional[str] = None
    ollama_models: Optional[list] = None
    # Display preferences
    show_cost: Optional[bool] = None
    # ElevenLabs configuration
    elevenlabs_api_key: Optional[str] = None
    clear_elevenlabs_key: bool = False
    default_voice_id: Optional[str] = None


class EnabledModelsResponse(BaseModel):
    """Lightweight response for GenerateForm model filtering."""
    enabled_text_models: Optional[list[str]] = None
    enabled_image_models: Optional[list[str]] = None
    enabled_video_models: Optional[list[str]] = None
    default_text_model: Optional[str] = None
    default_image_model: Optional[str] = None
    default_video_model: Optional[str] = None
    default_vision_model: Optional[str] = None
    comfyui_cost_per_second: Optional[float] = None
    # Phase 13: Ollama models for dynamic model list
    ollama_models: Optional[list] = None
    # Display preferences
    show_cost: bool = True
    default_voice_id: Optional[str] = None


@router.get("/settings")
async def get_settings() -> UserSettingsResponse:
    """Get current user settings."""
    async with async_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        settings = result.scalar_one_or_none()
        if not settings:
            return UserSettingsResponse()
        return UserSettingsResponse(
            enabled_text_models=settings.enabled_text_models,
            enabled_image_models=settings.enabled_image_models,
            enabled_video_models=settings.enabled_video_models,
            default_text_model=settings.default_text_model,
            default_image_model=settings.default_image_model,
            default_video_model=settings.default_video_model,
            default_vision_model=settings.default_vision_model,
            gcp_project_id=settings.gcp_project_id,
            gcp_location=settings.gcp_location,
            has_api_key=bool(settings.vertex_api_key),
            has_gcp_service_account=bool(settings.gcp_service_account_json),
            comfyui_host=settings.comfyui_host,
            has_comfyui_key=bool(settings.comfyui_api_key),
            comfyui_cost_per_second=settings.comfyui_cost_per_second,
            ollama_use_cloud=bool(settings.ollama_use_cloud),
            has_ollama_key=bool(settings.ollama_api_key),
            ollama_endpoint=settings.ollama_endpoint,
            ollama_models=settings.ollama_models,
            show_cost=bool(settings.show_cost),
            has_elevenlabs_key=bool(settings.elevenlabs_api_key),
            default_voice_id=settings.default_voice_id,
        )


@router.put("/settings")
async def update_settings(body: UserSettingsUpdate) -> UserSettingsResponse:
    """Update user settings."""
    # Validate model IDs if provided
    if body.enabled_text_models is not None:
        invalid = set(body.enabled_text_models) - ALLOWED_TEXT_MODELS
        if invalid:
            raise HTTPException(400, f"Invalid text model IDs: {invalid}")
    if body.enabled_image_models is not None:
        invalid = set(body.enabled_image_models) - ALLOWED_IMAGE_MODELS
        if invalid:
            raise HTTPException(400, f"Invalid image model IDs: {invalid}")
    if body.enabled_video_models is not None:
        invalid = set(body.enabled_video_models) - ALLOWED_VIDEO_MODELS
        if invalid:
            raise HTTPException(400, f"Invalid video model IDs: {invalid}")
    if body.default_text_model and body.default_text_model not in ALLOWED_TEXT_MODELS:
        if not body.default_text_model.startswith("ollama/"):
            raise HTTPException(400, f"Invalid default text model: {body.default_text_model}")
    if body.default_image_model and body.default_image_model not in ALLOWED_IMAGE_MODELS:
        raise HTTPException(400, f"Invalid default image model: {body.default_image_model}")
    if body.default_video_model and body.default_video_model not in ALLOWED_VIDEO_MODELS:
        raise HTTPException(400, f"Invalid default video model: {body.default_video_model}")
    if body.default_vision_model and body.default_vision_model not in ALLOWED_TEXT_MODELS:
        if not body.default_vision_model.startswith("ollama/"):
            raise HTTPException(400, f"Invalid default vision model: {body.default_vision_model}")

    async with async_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        settings = result.scalar_one_or_none()
        if not settings:
            raise HTTPException(500, "Default user settings not found")

        # Update all fields from the request body
        settings.enabled_text_models = body.enabled_text_models
        settings.enabled_image_models = body.enabled_image_models
        settings.enabled_video_models = body.enabled_video_models
        settings.default_text_model = body.default_text_model
        settings.default_image_model = body.default_image_model
        settings.default_video_model = body.default_video_model
        if body.default_vision_model is not None:
            settings.default_vision_model = body.default_vision_model or None
        settings.gcp_project_id = body.gcp_project_id
        settings.gcp_location = body.gcp_location

        # Only overwrite API key if explicitly provided (non-empty string)
        if body.clear_api_key:
            settings.vertex_api_key = None
        elif body.vertex_api_key:
            settings.vertex_api_key = body.vertex_api_key

        # GCP service account JSON
        if body.clear_gcp_service_account:
            settings.gcp_service_account_json = None
        elif body.gcp_service_account_json:
            # Validate JSON structure
            import json as _json
            try:
                sa_info = _json.loads(body.gcp_service_account_json)
            except _json.JSONDecodeError:
                raise HTTPException(400, "Invalid JSON in gcp_service_account_json")
            required_keys = {"type", "project_id", "private_key"}
            missing = required_keys - set(sa_info.keys())
            if missing:
                raise HTTPException(400, f"Service account JSON missing keys: {missing}")
            settings.gcp_service_account_json = body.gcp_service_account_json
            # Auto-populate project_id from service account if not explicitly set
            if not body.gcp_project_id and sa_info.get("project_id"):
                settings.gcp_project_id = sa_info["project_id"]

        # ComfyUI fields
        if body.comfyui_host is not None:
            settings.comfyui_host = body.comfyui_host or None
        if body.clear_comfyui_key:
            settings.comfyui_api_key = None
        elif body.comfyui_api_key:
            settings.comfyui_api_key = body.comfyui_api_key
        if body.comfyui_cost_per_second is not None:
            settings.comfyui_cost_per_second = body.comfyui_cost_per_second

        # Phase 13: Ollama fields
        if body.ollama_use_cloud is not None:
            settings.ollama_use_cloud = body.ollama_use_cloud
        if body.clear_ollama_key:
            settings.ollama_api_key = None
        elif body.ollama_api_key is not None:
            settings.ollama_api_key = body.ollama_api_key
        if body.ollama_endpoint is not None:
            settings.ollama_endpoint = body.ollama_endpoint or None
        if body.ollama_models is not None:
            settings.ollama_models = body.ollama_models
        if body.show_cost is not None:
            settings.show_cost = body.show_cost

        # ElevenLabs fields
        if body.clear_elevenlabs_key:
            settings.elevenlabs_api_key = None
        elif body.elevenlabs_api_key:
            settings.elevenlabs_api_key = body.elevenlabs_api_key
        if body.default_voice_id is not None:
            settings.default_voice_id = body.default_voice_id or None

        await session.commit()
        await session.refresh(settings)

        # Reload credentials for any changed GCP or ComfyUI fields
        gcp_changed = any([
            body.gcp_service_account_json, body.clear_gcp_service_account,
            body.gcp_project_id is not None, body.gcp_location is not None,
            body.vertex_api_key, body.clear_api_key,
        ])
        comfyui_changed = any([
            body.comfyui_host is not None, body.comfyui_api_key,
            body.clear_comfyui_key,
        ])
        if gcp_changed:
            from vidpipe.services.vertex_client import (
                invalidate_vertex_clients, load_vertex_credentials_from_db,
            )
            invalidate_vertex_clients()
            await load_vertex_credentials_from_db()
        if comfyui_changed:
            from vidpipe.services.comfyui_client import (
                invalidate_comfyui_client, load_comfyui_credentials_from_db,
            )
            invalidate_comfyui_client()
            await load_comfyui_credentials_from_db()

        return UserSettingsResponse(
            enabled_text_models=settings.enabled_text_models,
            enabled_image_models=settings.enabled_image_models,
            enabled_video_models=settings.enabled_video_models,
            default_text_model=settings.default_text_model,
            default_image_model=settings.default_image_model,
            default_video_model=settings.default_video_model,
            default_vision_model=settings.default_vision_model,
            gcp_project_id=settings.gcp_project_id,
            gcp_location=settings.gcp_location,
            has_api_key=bool(settings.vertex_api_key),
            has_gcp_service_account=bool(settings.gcp_service_account_json),
            comfyui_host=settings.comfyui_host,
            has_comfyui_key=bool(settings.comfyui_api_key),
            comfyui_cost_per_second=settings.comfyui_cost_per_second,
            ollama_use_cloud=bool(settings.ollama_use_cloud),
            has_ollama_key=bool(settings.ollama_api_key),
            ollama_endpoint=settings.ollama_endpoint,
            ollama_models=settings.ollama_models,
            show_cost=bool(settings.show_cost),
            has_elevenlabs_key=bool(settings.elevenlabs_api_key),
            default_voice_id=settings.default_voice_id,
        )


@router.get("/settings/models")
async def get_enabled_models() -> EnabledModelsResponse:
    """Lightweight endpoint for GenerateForm model filtering."""
    async with async_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        settings = result.scalar_one_or_none()
        if not settings:
            return EnabledModelsResponse()
        return EnabledModelsResponse(
            enabled_text_models=settings.enabled_text_models,
            enabled_image_models=settings.enabled_image_models,
            enabled_video_models=settings.enabled_video_models,
            default_text_model=settings.default_text_model,
            default_image_model=settings.default_image_model,
            default_video_model=settings.default_video_model,
            default_vision_model=settings.default_vision_model,
            comfyui_cost_per_second=settings.comfyui_cost_per_second,
            ollama_models=settings.ollama_models,
            show_cost=bool(settings.show_cost),
            default_voice_id=settings.default_voice_id,
        )


# ============================================================================
# PipeSVN: Checkpoint CRUD
# ============================================================================

class CheckpointListItem(BaseModel):
    sha: str
    parent_sha: Optional[str] = None
    message: str
    changes_count: int = 0
    created_at: str


class CheckpointDetail(BaseModel):
    sha: str
    parent_sha: Optional[str] = None
    message: str
    snapshot_data: dict
    metadata_json: Optional[dict] = None
    created_at: str


class CheckpointDiff(BaseModel):
    sha: str
    message: str
    changes: list[dict]


@router.get("/scenes/{scene_id}/checkpoints", response_model=list[CheckpointListItem])
async def list_checkpoints(scene_id: uuid.UUID):
    """List all checkpoints for a scene, newest first."""
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Scene not found")

        cp_result = await session.execute(
            select(SceneCheckpoint)
            .where(SceneCheckpoint.scene_id == scene_id)
            .order_by(SceneCheckpoint.created_at.desc())
        )
        checkpoints = cp_result.scalars().all()

        return [
            CheckpointListItem(
                sha=cp.sha,
                parent_sha=cp.parent_sha,
                message=cp.message,
                changes_count=len((cp.metadata_json or {}).get("changes", [])),
                created_at=cp.created_at.isoformat(),
            )
            for cp in checkpoints
        ]


@router.get("/scenes/{scene_id}/checkpoints/{sha}", response_model=CheckpointDetail)
async def get_checkpoint(scene_id: uuid.UUID, sha: str):
    """Get checkpoint detail including snapshot."""
    async with async_session() as session:
        result = await session.execute(
            select(SceneCheckpoint).where(
                SceneCheckpoint.scene_id == scene_id,
                SceneCheckpoint.sha == sha,
            )
        )
        cp = result.scalar_one_or_none()
        if not cp:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        return CheckpointDetail(
            sha=cp.sha,
            parent_sha=cp.parent_sha,
            message=cp.message,
            snapshot_data=cp.snapshot_data,
            metadata_json=cp.metadata_json,
            created_at=cp.created_at.isoformat(),
        )


@router.get("/scenes/{scene_id}/checkpoints/{sha}/diff", response_model=CheckpointDiff)
async def get_checkpoint_diff(scene_id: uuid.UUID, sha: str):
    """Get structured diff for a checkpoint."""
    async with async_session() as session:
        result = await session.execute(
            select(SceneCheckpoint).where(
                SceneCheckpoint.scene_id == scene_id,
                SceneCheckpoint.sha == sha,
            )
        )
        cp = result.scalar_one_or_none()
        if not cp:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # If checkpoint has metadata with changes, use that
        changes = (cp.metadata_json or {}).get("changes", [])

        # Otherwise compute diff from parent
        if not changes and cp.parent_sha:
            parent_result = await session.execute(
                select(SceneCheckpoint).where(
                    SceneCheckpoint.scene_id == scene_id,
                    SceneCheckpoint.sha == cp.parent_sha,
                )
            )
            parent_cp = parent_result.scalar_one_or_none()
            if parent_cp:
                from vidpipe.services.checkpoint_service import compute_diff
                changes = compute_diff(parent_cp.snapshot_data, cp.snapshot_data)

        return CheckpointDiff(
            sha=cp.sha,
            message=cp.message,
            changes=changes,
        )


@router.post("/scenes/{scene_id}/checkpoints")
async def create_manual_checkpoint(scene_id: uuid.UUID):
    """Create a manual checkpoint of current state."""
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        from vidpipe.services.checkpoint_service import create_checkpoint
        cp = await create_checkpoint(session, scene, "Manual checkpoint")
        await session.commit()

        return {"sha": cp.sha, "message": cp.message}


@router.delete("/scenes/{scene_id}/checkpoints/{sha}")
async def delete_checkpoint(scene_id: uuid.UUID, sha: str):
    """Delete a checkpoint. Splices the chain (updates child's parent_sha)."""
    async with async_session() as session:
        result = await session.execute(
            select(SceneCheckpoint).where(
                SceneCheckpoint.scene_id == scene_id,
                SceneCheckpoint.sha == sha,
            )
        )
        cp = result.scalar_one_or_none()
        if not cp:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # Don't allow deleting the head checkpoint
        proj_result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = proj_result.scalar_one_or_none()
        if scene and scene.head_sha == sha:
            raise HTTPException(status_code=400, detail="Cannot delete the current head checkpoint")

        # Splice: find child that points to this checkpoint, update its parent_sha
        child_result = await session.execute(
            select(SceneCheckpoint).where(
                SceneCheckpoint.scene_id == scene_id,
                SceneCheckpoint.parent_sha == sha,
            )
        )
        child = child_result.scalar_one_or_none()
        if child:
            child.parent_sha = cp.parent_sha

        await session.delete(cp)
        await session.commit()

        return {"status": "deleted", "sha": sha}


# ============================================================================
# PipeSVN: Revert
# ============================================================================

@router.post("/scenes/{scene_id}/revert")
async def revert_to_checkpoint(scene_id: uuid.UUID, body: dict):
    """Revert scene state to a specific checkpoint.

    Body: { "sha": "abc123..." }
    Creates a forward-commit checkpoint "Revert to {sha[:8]}".
    """
    target_sha = body.get("sha")
    if not target_sha:
        raise HTTPException(status_code=400, detail="Missing 'sha' in request body")

    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        cp_result = await session.execute(
            select(SceneCheckpoint).where(
                SceneCheckpoint.scene_id == scene_id,
                SceneCheckpoint.sha == target_sha,
            )
        )
        target_cp = cp_result.scalar_one_or_none()
        if not target_cp:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        from vidpipe.services.checkpoint_service import restore_from_snapshot, create_checkpoint
        await restore_from_snapshot(session, scene, target_cp.snapshot_data)

        revert_cp = await create_checkpoint(
            session, scene,
            f"Revert to {target_sha[:8]}",
            metadata={"reverted_to": target_sha},
        )
        await session.commit()

        return {
            "status": "reverted",
            "head_sha": revert_cp.sha,
            "reverted_to": target_sha,
        }


# ============================================================================
# PipeSVN: Shot Regeneration
# ============================================================================

class RegenerateShotRequest(BaseModel):
    targets: list[str]  # ["start_keyframe", "end_keyframe", "video_clip"]
    prompt_overrides: Optional[dict[str, str]] = None
    skip_checkpoint: bool = False
    video_model: Optional[str] = None
    image_model: Optional[str] = None
    shot_edits: Optional[dict[str, str]] = None


class RegenerateTextRequest(BaseModel):
    field: str  # "shot_description" | "start_frame_prompt" | "end_frame_prompt" | "video_motion_prompt" | "transition_notes"
    extra_context: str = ""
    text_model: Optional[str] = None  # override scene's text_model (use current edit-mode selection)
    shot_edits: Optional[dict[str, str]] = None
    prompt: Optional[str] = None  # override scene prompt (use current edit-mode value)


class _RegenTextResult(BaseModel):
    text: str


@router.post("/scenes/{scene_id}/shots/{shot_idx}/regenerate")
async def regenerate_shot_assets(
    scene_id: uuid.UUID,
    shot_idx: int,
    body: RegenerateShotRequest,
    background_tasks: BackgroundTasks,
):
    """Regenerate specific assets for a shot (keyframes and/or clip).

    Returns 202 --- regeneration runs in background.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot).where(
                Shot.scene_id == scene.id,
                Shot.shot_index == shot_idx,
            )
        )
        shot = shot_result.scalar_one_or_none()
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")

    # Queue background task
    background_tasks.add_task(
        _run_shot_regeneration, scene_id, shot_idx, body.targets, body.prompt_overrides, body.skip_checkpoint,
        video_model_override=body.video_model, image_model_override=body.image_model,
        shot_edits=body.shot_edits,
    )

    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "targets": body.targets,
            "shot_index": shot_idx,
            "head_sha": scene.head_sha,
        },
    )


_REGEN_TEXT_VALID_FIELDS = {
    "shot_description",
    "start_frame_prompt",
    "end_frame_prompt",
    "video_motion_prompt",
    "transition_notes",
}

_REGEN_FIELD_INSTRUCTIONS = {
    "shot_description": (
        "Write a vivid narrative shot description capturing the key visual story beat. "
        "Include specific details about characters, setting, mood, and action."
    ),
    "start_frame_prompt": (
        "Write a keyframe image prompt following this structure:\n"
        "1. MEDIUM DECLARATION: 'A {style} rendering of...'\n"
        "2. SUBJECT: Detailed character description matching the character bible\n"
        "3. ACTION/POSE: What the character is doing, body position\n"
        "4. SETTING: Environment, background elements, any visible text/signage\n"
        "5. LIGHTING: Light source direction, quality, mood\n"
        "6. CAMERA: Shot type (wide/medium/close-up), angle, lens\n"
        "7. STYLE CUES: Rendering technique details specific to {style}\n"
        "8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic"
    ),
    "end_frame_prompt": (
        "Write a keyframe image prompt following this structure:\n"
        "1. MEDIUM DECLARATION: 'A {style} rendering of...'\n"
        "2. SUBJECT: Detailed character description matching the character bible\n"
        "3. ACTION/POSE: What the character is doing, body position\n"
        "4. SETTING: Environment, background elements, any visible text/signage\n"
        "5. LIGHTING: Light source direction, quality, mood\n"
        "6. CAMERA: Shot type (wide/medium/close-up), angle, lens\n"
        "7. STYLE CUES: Rendering technique details specific to {style}\n"
        "8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic"
    ),
    "video_motion_prompt": (
        "Write a video motion prompt describing ONLY motion and camera movement. "
        "Do NOT re-describe characters, setting, or style — the keyframe images "
        "already provide visual context. Focus on: camera movement (pan, dolly, "
        "track, crane), subject animation, environmental animation.\n"
        "Good: 'Slow dolly forward as the subject turns to face the camera, "
        "hair gently blowing in the breeze'\n"
        "Bad: 'A blonde woman in anime style turns around in a room' (re-describes visuals)"
    ),
    "transition_notes": (
        "Write brief transition/continuity notes describing how this shot connects "
        "visually to its neighbors. Focus on visual elements that carry over: "
        "character position, lighting direction, color temperature, camera angle."
    ),
}


@router.post("/scenes/{scene_id}/shots/{shot_idx}/regenerate-text")
async def regenerate_shot_text(
    scene_id: uuid.UUID,
    shot_idx: int,
    body: RegenerateTextRequest,
):
    """Regenerate a single text field for a shot via LLM.

    Returns the generated text directly (synchronous — no background task).
    The text is NOT saved to DB; the frontend applies it as an edit.
    """
    if body.field not in _REGEN_TEXT_VALID_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field: {body.field}. Must be one of: {', '.join(sorted(_REGEN_TEXT_VALID_FIELDS))}",
        )

    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot)
            .where(Shot.scene_id == scene.id)
            .order_by(Shot.shot_index)
        )
        all_shots = list(shot_result.scalars().all())
        shot = next((s for s in all_shots if s.shot_index == shot_idx), None)
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")

        # Build neighbor context
        prev_shot = next((s for s in all_shots if s.shot_index == shot_idx - 1), None)
        next_shot = next((s for s in all_shots if s.shot_index == shot_idx + 1), None)

        # Load user settings for LLM adapter (needed for Ollama config)
        us_result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        user_settings = us_result.scalar_one_or_none()

    # Build system prompt
    style = scene.style or "cinematic"
    field_instructions = _REGEN_FIELD_INSTRUCTIONS[body.field].replace("{style}", style)
    system_prompt = (
        f"You are a storyboard director specializing in short-form video content.\n"
        f"Visual style: {style}\n"
        f"Aspect ratio: {scene.aspect_ratio or '16:9'}\n\n"
        f"TASK: Regenerate the '{body.field}' field for shot {shot_idx + 1}.\n\n"
        f"FORMAT INSTRUCTIONS:\n{field_instructions}\n\n"
        f"Return a JSON object with a single 'text' field containing the regenerated content."
    )

    # Build user prompt with context — prefer in-flight prompt from edit form
    effective_prompt = body.prompt or scene.prompt
    parts = []
    parts.append(f"SCENE CONCEPT: {effective_prompt}")
    if scene.style_guide:
        sg = scene.style_guide
        if isinstance(sg, dict):
            sg_text = sg.get("description") or json.dumps(sg)
        else:
            sg_text = str(sg)
        parts.append(f"STYLE GUIDE: {sg_text}")

    # Character bible from storyboard_raw
    if scene.storyboard_raw and isinstance(scene.storyboard_raw, dict):
        characters = scene.storyboard_raw.get("characters")
        if characters:
            parts.append(f"CHARACTERS: {json.dumps(characters)}")

    # Current shot context (sibling fields) — prefer in-flight edits over committed DB values
    se = body.shot_edits or {}
    parts.append(f"\nCURRENT SHOT {shot_idx + 1}:")
    parts.append(f"  Description: {se.get('shot_description', shot.shot_description)}")
    parts.append(f"  Start Frame Prompt: {se.get('start_frame_prompt', shot.start_frame_prompt)}")
    parts.append(f"  End Frame Prompt: {se.get('end_frame_prompt', shot.end_frame_prompt)}")
    parts.append(f"  Motion Prompt: {se.get('video_motion_prompt', shot.video_motion_prompt)}")
    tn = se.get('transition_notes', shot.transition_notes)
    if tn:
        parts.append(f"  Transition Notes: {tn}")

    # Neighbor context
    if prev_shot:
        parts.append(f"\nPREVIOUS SHOT {prev_shot.shot_index + 1}: {prev_shot.shot_description}")
    if next_shot:
        parts.append(f"\nNEXT SHOT {next_shot.shot_index + 1}: {next_shot.shot_description}")

    # Extra context from user
    if body.extra_context and body.extra_context.strip():
        parts.append(f"\n[Additional direction: {body.extra_context.strip()}]")

    user_prompt = "\n".join(parts)

    # Call LLM — prefer request override (edit-mode selection) over saved scene model
    from vidpipe.services.llm import get_adapter
    model_id = body.text_model or scene.text_model or "gemini-2.5-flash"
    adapter = get_adapter(model_id, user_settings=user_settings)

    try:
        result = await adapter.generate_text(
            prompt=user_prompt,
            schema=_RegenTextResult,
            temperature=0.7,
            system_prompt=system_prompt,
        )
        return {"field": body.field, "text": result.text}
    except Exception as e:
        logger.error("Text regen failed for shot %d field %s: %s", shot_idx, body.field, e)
        raise HTTPException(status_code=500, detail=f"Text regeneration failed: {e}")


class GenerateShotFieldsRequest(BaseModel):
    shot_index: int
    all_shot_edits: Optional[dict[int, dict[str, str]]] = None
    text_model: Optional[str] = None
    prompt: Optional[str] = None  # override scene prompt (use current edit-mode value)


class GenerateShotFieldsResponse(BaseModel):
    shot_description: str
    start_frame_prompt: str
    end_frame_prompt: str
    video_motion_prompt: str
    transition_notes: str


@router.post("/scenes/{scene_id}/generate-shot-fields")
async def generate_shot_fields(
    scene_id: uuid.UUID,
    body: GenerateShotFieldsRequest,
):
    """Generate all 5 text fields for a new/empty shot via a single LLM call.

    Returns the generated fields directly (synchronous — no background task).
    The text is NOT saved to DB; the frontend applies fields as edits.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot)
            .where(Shot.scene_id == scene.id)
            .order_by(Shot.shot_index)
        )
        all_shots = list(shot_result.scalars().all())

        # Load user settings for LLM adapter (needed for Ollama config)
        us_result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        user_settings = us_result.scalar_one_or_none()

    # Helper: get effective field value considering in-flight edits
    edits_map = body.all_shot_edits or {}

    def shot_field(shot_obj, field_db: str, shot_idx: int) -> str:
        """Return in-flight edit if present, else DB value."""
        idx_edits = edits_map.get(shot_idx, {})
        if field_db in idx_edits:
            return idx_edits[field_db]
        if shot_obj is None:
            return ""
        return getattr(shot_obj, field_db, "") or ""

    # Build neighbor context
    prev_db = next((s for s in all_shots if s.shot_index == body.shot_index - 1), None)
    next_db = next((s for s in all_shots if s.shot_index == body.shot_index + 1), None)
    prev_idx = body.shot_index - 1
    next_idx = body.shot_index + 1

    style = scene.style or "cinematic"

    # Build system prompt
    field_blocks = []
    for field_name, instructions in _REGEN_FIELD_INSTRUCTIONS.items():
        field_blocks.append(f"  {field_name}: {instructions.replace('{style}', style)}")
    all_field_instructions = "\n".join(field_blocks)

    system_prompt = (
        f"You are a storyboard director specializing in short-form video content.\n"
        f"Visual style: {style}\n"
        f"Aspect ratio: {scene.aspect_ratio or '16:9'}\n\n"
        f"TASK: Generate ALL 5 text fields for a NEW shot at position {body.shot_index + 1}.\n\n"
        f"FIELD INSTRUCTIONS:\n{all_field_instructions}\n\n"
        f"Return a JSON object with exactly these 5 fields:\n"
        f"  shot_description, start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes"
    )

    # Build user prompt with context — prefer in-flight prompt from edit form
    effective_prompt = body.prompt or scene.prompt
    parts = []
    parts.append(f"SCENE CONCEPT: {effective_prompt}")
    if scene.style_guide:
        sg = scene.style_guide
        if isinstance(sg, dict):
            sg_text = sg.get("description") or json.dumps(sg)
        else:
            sg_text = str(sg)
        parts.append(f"STYLE GUIDE: {sg_text}")

    # Character bible from storyboard_raw
    if scene.storyboard_raw and isinstance(scene.storyboard_raw, dict):
        characters = scene.storyboard_raw.get("characters")
        if characters:
            parts.append(f"CHARACTERS: {json.dumps(characters)}")

    # Previous shot context (from DB or in-flight edits for synthetic shots)
    has_prev = prev_db is not None or prev_idx in edits_map
    if has_prev:
        parts.append(f"\nPREVIOUS SHOT {prev_idx + 1}:")
        parts.append(f"  Description: {shot_field(prev_db, 'shot_description', prev_idx)}")
        parts.append(f"  Start Frame: {shot_field(prev_db, 'start_frame_prompt', prev_idx)}")
        parts.append(f"  End Frame: {shot_field(prev_db, 'end_frame_prompt', prev_idx)}")
        parts.append(f"  Motion: {shot_field(prev_db, 'video_motion_prompt', prev_idx)}")

    # Next shot context
    has_next = next_db is not None or next_idx in edits_map
    if has_next:
        parts.append(f"\nNEXT SHOT {next_idx + 1}:")
        parts.append(f"  Description: {shot_field(next_db, 'shot_description', next_idx)}")
        parts.append(f"  Start Frame: {shot_field(next_db, 'start_frame_prompt', next_idx)}")
        parts.append(f"  End Frame: {shot_field(next_db, 'end_frame_prompt', next_idx)}")
        parts.append(f"  Motion: {shot_field(next_db, 'video_motion_prompt', next_idx)}")

    # Any existing edits the user already typed for this shot
    current_edits = edits_map.get(body.shot_index, {})
    if current_edits:
        parts.append(f"\nUSER'S PARTIAL INPUT FOR THIS SHOT (incorporate and expand on these):")
        for k, v in current_edits.items():
            if v.strip():
                parts.append(f"  {k}: {v}")

    user_prompt = "\n".join(parts)

    # Call LLM
    from vidpipe.services.llm import get_adapter
    model_id = body.text_model or scene.text_model or "gemini-2.5-flash"
    adapter = get_adapter(model_id, user_settings=user_settings)

    try:
        result = await adapter.generate_text(
            prompt=user_prompt,
            schema=GenerateShotFieldsResponse,
            temperature=0.7,
            system_prompt=system_prompt,
        )
        return result.model_dump()
    except Exception as e:
        logger.error("Generate shot fields failed for scene %s shot %d: %s", scene_id, body.shot_index, e)
        raise HTTPException(status_code=500, detail=f"Shot field generation failed: {e}")


class GenerateNewShotRequest(BaseModel):
    shot_index: int
    all_shot_edits: Optional[dict[int, dict[str, str]]] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    prompt: Optional[str] = None  # override scene prompt (use current edit-mode value)


class GenerateNewShotResponse(BaseModel):
    shot_index: int
    shot_description: str
    start_frame_prompt: str
    end_frame_prompt: str
    video_motion_prompt: str
    transition_notes: str
    head_sha: Optional[str] = None


@router.post("/scenes/{scene_id}/generate-new-shot")
async def generate_new_shot(
    scene_id: uuid.UUID,
    body: GenerateNewShotRequest,
    background_tasks: BackgroundTasks,
):
    """Generate a complete new shot: text fields (sync) + keyframes & clip (background).

    Phase 1 (synchronous): Create Shot DB row with LLM-generated text fields.
    Phase 2 (background): Generate start KF, end KF, and video clip via existing regen infra.

    Returns 202 with the generated text fields and head_sha for revert-on-cancel.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Check for existing shot at requested index — reuse if empty, reject if populated
        existing_result = await session.execute(
            select(Shot).where(
                Shot.scene_id == scene.id,
                Shot.shot_index == body.shot_index,
            )
        )
        existing_shot = existing_result.scalar_one_or_none()
        if existing_shot and (existing_shot.start_frame_prompt or existing_shot.end_frame_prompt
                or existing_shot.video_motion_prompt):
            raise HTTPException(status_code=409, detail=f"Shot already exists at index {body.shot_index}")

        # Load all shots for neighbor context
        shot_result = await session.execute(
            select(Shot)
            .where(Shot.scene_id == scene.id)
            .order_by(Shot.shot_index)
        )
        all_shots = list(shot_result.scalars().all())

        # Load user settings for LLM adapter
        us_result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        user_settings = us_result.scalar_one_or_none()

        # --- Phase 1: Generate text fields via LLM (same logic as generate_shot_fields) ---
        edits_map = body.all_shot_edits or {}

        def shot_field(shot_obj, field_db: str, shot_idx: int) -> str:
            idx_edits = edits_map.get(shot_idx, {})
            if field_db in idx_edits:
                return idx_edits[field_db]
            if shot_obj is None:
                return ""
            return getattr(shot_obj, field_db, "") or ""

        prev_db = next((s for s in all_shots if s.shot_index == body.shot_index - 1), None)
        next_db = next((s for s in all_shots if s.shot_index == body.shot_index + 1), None)
        prev_idx = body.shot_index - 1
        next_idx = body.shot_index + 1

        style = scene.style or "cinematic"

        field_blocks = []
        for field_name, instructions in _REGEN_FIELD_INSTRUCTIONS.items():
            field_blocks.append(f"  {field_name}: {instructions.replace('{style}', style)}")
        all_field_instructions = "\n".join(field_blocks)

        system_prompt = (
            f"You are a storyboard director specializing in short-form video content.\n"
            f"Visual style: {style}\n"
            f"Aspect ratio: {scene.aspect_ratio or '16:9'}\n\n"
            f"TASK: Generate ALL 5 text fields for a NEW shot at position {body.shot_index + 1}.\n\n"
            f"FIELD INSTRUCTIONS:\n{all_field_instructions}\n\n"
            f"Return a JSON object with exactly these 5 fields:\n"
            f"  shot_description, start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes"
        )

        # Prefer in-flight prompt from edit form over saved DB value
        effective_prompt = body.prompt or scene.prompt
        parts = []
        parts.append(f"SCENE CONCEPT: {effective_prompt}")
        if scene.style_guide:
            sg = scene.style_guide
            if isinstance(sg, dict):
                sg_text = sg.get("description") or json.dumps(sg)
            else:
                sg_text = str(sg)
            parts.append(f"STYLE GUIDE: {sg_text}")

        if scene.storyboard_raw and isinstance(scene.storyboard_raw, dict):
            characters = scene.storyboard_raw.get("characters")
            if characters:
                parts.append(f"CHARACTERS: {json.dumps(characters)}")

        has_prev = prev_db is not None or prev_idx in edits_map
        if has_prev:
            parts.append(f"\nPREVIOUS SHOT {prev_idx + 1}:")
            parts.append(f"  Description: {shot_field(prev_db, 'shot_description', prev_idx)}")
            parts.append(f"  Start Frame: {shot_field(prev_db, 'start_frame_prompt', prev_idx)}")
            parts.append(f"  End Frame: {shot_field(prev_db, 'end_frame_prompt', prev_idx)}")
            parts.append(f"  Motion: {shot_field(prev_db, 'video_motion_prompt', prev_idx)}")

        has_next = next_db is not None or next_idx in edits_map
        if has_next:
            parts.append(f"\nNEXT SHOT {next_idx + 1}:")
            parts.append(f"  Description: {shot_field(next_db, 'shot_description', next_idx)}")
            parts.append(f"  Start Frame: {shot_field(next_db, 'start_frame_prompt', next_idx)}")
            parts.append(f"  End Frame: {shot_field(next_db, 'end_frame_prompt', next_idx)}")
            parts.append(f"  Motion: {shot_field(next_db, 'video_motion_prompt', next_idx)}")

        current_edits = edits_map.get(body.shot_index, {})
        if current_edits:
            parts.append(f"\nUSER'S PARTIAL INPUT FOR THIS SHOT (incorporate and expand on these):")
            for k, v in current_edits.items():
                if v.strip():
                    parts.append(f"  {k}: {v}")

        user_prompt = "\n".join(parts)

        from vidpipe.services.llm import get_adapter
        model_id = body.text_model or scene.text_model or "gemini-2.5-flash"
        adapter = get_adapter(model_id, user_settings=user_settings)

        try:
            text_result = await adapter.generate_text(
                prompt=user_prompt,
                schema=GenerateShotFieldsResponse,
                temperature=0.7,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error("Generate new shot text failed for scene %s shot %d: %s", scene_id, body.shot_index, e)
            raise HTTPException(status_code=500, detail=f"Shot text generation failed: {e}")

        # --- Create or update Shot DB row ---
        if existing_shot:
            # Reuse the empty placeholder shot
            new_shot = existing_shot
            new_shot.shot_description = text_result.shot_description
            new_shot.start_frame_prompt = text_result.start_frame_prompt
            new_shot.end_frame_prompt = text_result.end_frame_prompt
            new_shot.video_motion_prompt = text_result.video_motion_prompt
            new_shot.transition_notes = text_result.transition_notes
            new_shot.status = "pending"
        else:
            new_shot = Shot(
                scene_id=scene.id,
                shot_index=body.shot_index,
                shot_description=text_result.shot_description,
                start_frame_prompt=text_result.start_frame_prompt,
                end_frame_prompt=text_result.end_frame_prompt,
                video_motion_prompt=text_result.video_motion_prompt,
                transition_notes=text_result.transition_notes,
                status="pending",
            )
            session.add(new_shot)

        # Update target_shot_count if needed
        scene.target_shot_count = max(scene.target_shot_count, body.shot_index + 1)

        # Ensure baseline checkpoint exists
        if not scene.head_sha:
            from vidpipe.services.checkpoint_service import create_checkpoint
            await create_checkpoint(
                session, scene,
                "Auto-save: edit baseline",
                metadata={"auto_baseline": True},
            )
            await session.commit()

        # Create checkpoint for the new shot
        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, scene,
            f"Generated new shot {body.shot_index + 1}",
            metadata={"generated_shot": body.shot_index},
        )
        await session.commit()

        head_sha = scene.head_sha

    # --- Phase 2: Queue background asset generation ---
    background_tasks.add_task(
        _run_shot_regeneration,
        scene_id,
        body.shot_index,
        ["start_keyframe", "end_keyframe", "video_clip"],
        None,  # no prompt overrides
        True,  # skip_checkpoint (we already created one above)
        video_model_override=body.video_model,
        image_model_override=body.image_model,
    )

    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content=GenerateNewShotResponse(
            shot_index=body.shot_index,
            shot_description=text_result.shot_description,
            start_frame_prompt=text_result.start_frame_prompt,
            end_frame_prompt=text_result.end_frame_prompt,
            video_motion_prompt=text_result.video_motion_prompt,
            transition_notes=text_result.transition_notes,
            head_sha=head_sha,
        ).model_dump(),
    )


async def _run_shot_regeneration(
    scene_id: uuid.UUID,
    shot_idx: int,
    targets: list[str],
    prompt_overrides: Optional[dict[str, str]],
    skip_checkpoint: bool = False,
    video_model_override: Optional[str] = None,
    image_model_override: Optional[str] = None,
    shot_edits: Optional[dict[str, str]] = None,
):
    """Background task for shot regeneration."""
    from vidpipe.services.event_bus import event_bus
    event_bus.emit(scene_id, "shot_regen_started", shot_index=shot_idx, targets=targets)
    logger.info("Regenerating shot %d for scene %s: %s", shot_idx, scene_id, targets)

    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            return

        shot_result = await session.execute(
            select(Shot).where(Shot.scene_id == scene.id, Shot.shot_index == shot_idx)
        )
        shot = shot_result.scalar_one_or_none()
        if not shot:
            return

        # When skip_checkpoint is set (edit mode), ensure a baseline checkpoint
        # exists so the user can revert on cancel.
        if skip_checkpoint and not scene.head_sha:
            from vidpipe.services.checkpoint_service import create_checkpoint
            await create_checkpoint(
                session, scene,
                "Auto-save: edit baseline",
                metadata={"auto_baseline": True},
            )
            await session.commit()

        from vidpipe.services.file_manager import FileManager
        file_mgr = FileManager()

        regenerated = []

        # Regenerate keyframes
        for target in targets:
            if target in ("start_keyframe", "end_keyframe"):
                position = "start" if target == "start_keyframe" else "end"
                try:
                    await _regenerate_keyframe(
                        session, scene, shot, position, file_mgr,
                        prompt_override=prompt_overrides.get(target) if prompt_overrides else None,
                        image_model_override=image_model_override,
                        shot_edits=shot_edits,
                    )
                    regenerated.append(target)

                    # KEYF-03 cascade: end keyframe change propagates to next shot's start
                    if position == "end":
                        next_shot_result = await session.execute(
                            select(Shot).where(
                                Shot.scene_id == scene.id,
                                Shot.shot_index == shot_idx + 1,
                            )
                        )
                        next_shot = next_shot_result.scalar_one_or_none()
                        if next_shot:
                            # Read the newly regenerated end keyframe bytes
                            new_end_kf_result = await session.execute(
                                select(Keyframe).where(
                                    Keyframe.shot_id == shot.id,
                                    Keyframe.position == "end",
                                )
                            )
                            new_end_kf = new_end_kf_result.scalar_one_or_none()
                            if new_end_kf:
                                end_bytes = await file_mgr.read_bytes(new_end_kf.file_path)
                                storage = get_storage_backend()
                                if isinstance(storage, LocalStorageBackend):
                                    inherited_path = file_mgr.save_keyframe_versioned(
                                        scene.id, shot_idx + 1, "start", end_bytes,
                                    )
                                    stored_path = str(inherited_path)
                                else:
                                    stored_path = await file_mgr.save_keyframe_versioned_async(
                                        scene.id, shot_idx + 1, "start", end_bytes,
                                    )
                                # Replace next shot's start keyframe
                                old_next_start_result = await session.execute(
                                    select(Keyframe).where(
                                        Keyframe.shot_id == next_shot.id,
                                        Keyframe.position == "start",
                                    )
                                )
                                old_next_start = old_next_start_result.scalar_one_or_none()
                                if old_next_start:
                                    await session.delete(old_next_start)
                                    await session.flush()
                                inherited_kf = Keyframe(
                                    shot_id=next_shot.id,
                                    position="start",
                                    prompt_used=next_shot.start_frame_prompt,
                                    file_path=stored_path,
                                    mime_type="image/png",
                                    source="inherited",
                                )
                                session.add(inherited_kf)
                                await session.flush()
                                logger.info(
                                    "Cascaded end keyframe from shot %d to start of shot %d",
                                    shot_idx, shot_idx + 1,
                                )
                except Exception as e:
                    logger.error("Failed to regenerate %s for shot %d: %s", target, shot_idx, e)

            elif target == "video_clip":
                try:
                    await _regenerate_clip(
                        session, scene, shot, file_mgr,
                        prompt_override=prompt_overrides.get("video_clip") if prompt_overrides else None,
                        video_model_override=video_model_override,
                        shot_edits=shot_edits,
                    )
                    regenerated.append(target)
                except Exception as e:
                    logger.error("Failed to regenerate clip for shot %d: %s", shot_idx, e)

        # Persist changes and optionally create checkpoint
        if regenerated:
            if not skip_checkpoint:
                from vidpipe.services.checkpoint_service import create_checkpoint
                await create_checkpoint(
                    session, scene,
                    f"Regenerated {', '.join(regenerated)} for shot {shot_idx + 1}",
                    metadata={"regenerated": regenerated, "shot_index": shot_idx},
                )
            await session.commit()
            event_bus.emit(scene_id, "shot_regen_done", shot_index=shot_idx)
            event_bus.emit(scene_id, "refresh")


async def _regenerate_keyframe(session, scene, shot, position, file_mgr, prompt_override=None, image_model_override=None, shot_edits=None):
    """Regenerate a single keyframe following the same methodology as the pipeline.

    - Start KF shot 0: text-to-image with style/character enrichment + ref images
    - Start KF shot N>0: image-conditioned from previous shot's end keyframe
    - End KF: image-conditioned from this shot's start keyframe with progression prompt
    """
    from pathlib import Path
    from vidpipe.config import settings as app_settings
    from vidpipe.db.models import ShotManifest as ShotManifestModel

    image_model = image_model_override or scene.image_model or app_settings.models.keyframe_image
    # Guard: Imagen models no longer supported
    if image_model.startswith("imagen-"):
        image_model = app_settings.models.image_gen

    # Build character bible prefix from storyboard data
    character_prefix = ""
    if scene.storyboard_raw and "characters" in scene.storyboard_raw:
        char_lines = []
        for ch in scene.storyboard_raw["characters"]:
            char_lines.append(
                f"{ch.get('name', 'Character')}: {ch.get('physical_description', '')}. "
                f"Wearing {ch.get('clothing_description', '')}."
            )
        if char_lines:
            character_prefix = "Characters: " + " ".join(char_lines) + " "

    # Build style prefix from style guide
    style_guide = scene.style_guide or {}
    style_prefix = ""
    if style_guide:
        parts = []
        if style_guide.get("visual_style"):
            parts.append(f"Style: {style_guide['visual_style']}")
        if style_guide.get("color_palette"):
            parts.append(f"Palette: {style_guide['color_palette']}")
        if parts:
            style_prefix = ". ".join(parts) + ". "

    # Load shot manifest for rewritten prompt + reference images
    sm_result = await session.execute(
        select(ShotManifestModel).where(
            ShotManifestModel.scene_id == scene.id,
            ShotManifestModel.shot_index == shot.shot_index,
        )
    )
    sm = sm_result.scalar_one_or_none()
    rewritten_prompt = sm.rewritten_keyframe_prompt if sm else None

    # Resolve asset reference images (for production-bible scenes)
    ref_image_bytes_list: list[bytes] = []
    if scene.production_bible_id and sm and sm.selected_reference_tags:
        try:
            from vidpipe.services import manifest_service
            from vidpipe.services.reference_selection import resolve_asset_image_bytes
            all_assets = await manifest_service.load_manifest_assets(session, scene.production_bible_id)
            asset_map = {a.manifest_tag: a for a in all_assets}
            for tag in sm.selected_reference_tags:
                asset = asset_map.get(tag)
                if asset:
                    ref_bytes = await resolve_asset_image_bytes(session, asset)
                    if ref_bytes:
                        ref_image_bytes_list.append(ref_bytes)
        except Exception as e:
            logger.warning("Failed to resolve reference images for regen: %s", e)

    # Route to ComfyUI or Vertex AI
    from vidpipe.pipeline.keyframes import (
        _generate_image_from_text, _generate_image_conditioned,
        COMFYUI_IMAGE_MODELS, _generate_image_comfyui, _generate_image_comfyui_flux,
    )
    is_comfyui = image_model in COMFYUI_IMAGE_MODELS
    comfy_client = None
    image_client = None
    if is_comfyui:
        from vidpipe.services.comfyui_client import get_comfyui_client
        comfy_client = await get_comfyui_client()
    else:
        from vidpipe.services.vertex_client import get_vertex_client, location_for_model
        image_client = get_vertex_client(location=location_for_model(image_model))

    # Extract edited field values (user edits take priority over stale rewrites)
    edited_start = (shot_edits or {}).get("start_frame_prompt")
    edited_end = (shot_edits or {}).get("end_frame_prompt")

    if position == "start":
        # --- START KEYFRAME ---
        # Determine the base prompt
        if prompt_override:
            base_prompt = prompt_override
        elif edited_start:
            base_prompt = edited_start
        elif rewritten_prompt:
            base_prompt = rewritten_prompt
        else:
            base_prompt = shot.start_frame_prompt

        if shot.shot_index == 0:
            # Shot 0: text-to-image with style/character enrichment
            enriched_prompt = f"{style_prefix}{character_prefix}{base_prompt}" if not prompt_override else base_prompt
            if is_comfyui and image_model.startswith("flux-"):
                from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS
                fw, fh = _FLUX_RESOLUTIONS.get(scene.aspect_ratio, (1024, 1024))
                image_bytes = await _generate_image_comfyui_flux(
                    comfy_client, enriched_prompt, seed=scene.seed,
                    width=fw, height=fh,
                )
            elif is_comfyui:
                image_bytes = await _generate_image_comfyui(comfy_client, enriched_prompt, seed=scene.seed)
            else:
                image_bytes = await _generate_image_from_text(
                    image_client, enriched_prompt, scene.aspect_ratio, image_model,
                    seed=scene.seed,
                    reference_images=ref_image_bytes_list or None,
                )
        else:
            # Shot N>0: inherit from previous shot's end keyframe (KEYF-03)
            # Pipeline copies end→start verbatim for continuity.
            # With prompt_override (extra direction), use conditioned generation instead.
            prev_shot_result = await session.execute(
                select(Shot).where(
                    Shot.scene_id == scene.id,
                    Shot.shot_index == shot.shot_index - 1,
                )
            )
            prev_shot = prev_shot_result.scalar_one_or_none()
            prev_end_kf = None
            if prev_shot:
                prev_kf_result = await session.execute(
                    select(Keyframe).where(
                        Keyframe.shot_id == prev_shot.id,
                        Keyframe.position == "end",
                    )
                )
                prev_end_kf = prev_kf_result.scalar_one_or_none()

            prev_end_bytes = None
            if prev_end_kf:
                try:
                    prev_end_bytes = await file_mgr.read_bytes(prev_end_kf.file_path)
                except FileNotFoundError:
                    pass

            if prev_end_bytes is not None:
                if prompt_override:
                    # Extra direction provided — conditioned generation from prev end frame
                    style_label = scene.style.replace("_", " ")
                    conditioning_prompt = (
                        f"Generate the NEXT keyframe continuing from the previous shot into this new shot. "
                        f"Style: {style_label}.\n\n"
                        f"TARGET START STATE (what the new image must depict):\n"
                        f"{prompt_override}\n\n"
                        f"CONSISTENCY CONSTRAINTS:\n"
                        f"- Same character appearance (face, hair, clothing, proportions)\n"
                        f"- Same {style_label} rendering style\n"
                        f"{character_prefix}"
                    )
                    if is_comfyui and image_model.startswith("flux-"):
                        from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS as _FR2
                        fw2, fh2 = _FR2.get(scene.aspect_ratio, (1024, 1024))
                        image_bytes = await _generate_image_comfyui_flux(
                            comfy_client, conditioning_prompt, seed=scene.seed,
                            width=fw2, height=fh2,
                        )
                    elif is_comfyui:
                        image_bytes = await _generate_image_comfyui(comfy_client, conditioning_prompt, seed=scene.seed)
                    else:
                        image_bytes = await _generate_image_conditioned(
                            image_client, prev_end_bytes, conditioning_prompt,
                            scene.aspect_ratio, image_model,
                            reference_images=ref_image_bytes_list or None,
                        )
                else:
                    # No extra direction — inherit verbatim (same as pipeline KEYF-03)
                    image_bytes = prev_end_bytes
            else:
                # Fallback: no previous end keyframe available, use text-to-image
                enriched_prompt = f"{style_prefix}{character_prefix}{base_prompt}" if not prompt_override else base_prompt
                if is_comfyui and image_model.startswith("flux-"):
                    from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS as _FR3
                    fw3, fh3 = _FR3.get(scene.aspect_ratio, (1024, 1024))
                    image_bytes = await _generate_image_comfyui_flux(
                        comfy_client, enriched_prompt, seed=scene.seed,
                        width=fw3, height=fh3,
                    )
                elif is_comfyui:
                    image_bytes = await _generate_image_comfyui(comfy_client, enriched_prompt, seed=scene.seed)
                else:
                    image_bytes = await _generate_image_from_text(
                        image_client, enriched_prompt, scene.aspect_ratio, image_model,
                        seed=scene.seed,
                        reference_images=ref_image_bytes_list or None,
                    )

        prompt_used = prompt_override or edited_start or rewritten_prompt or shot.start_frame_prompt
    else:
        # --- END KEYFRAME ---
        # Always image-conditioned from this shot's start keyframe
        start_kf_result = await session.execute(
            select(Keyframe).where(
                Keyframe.shot_id == shot.id,
                Keyframe.position == "start",
            )
        )
        start_kf = start_kf_result.scalar_one_or_none()

        if not start_kf:
            raise ValueError(f"No start keyframe available for shot {shot.shot_index} — generate start keyframe first")

        try:
            conditioning_bytes = await file_mgr.read_bytes(start_kf.file_path)
        except FileNotFoundError:
            raise ValueError(f"Start keyframe file not found for shot {shot.shot_index}")

        if prompt_override:
            conditioning_prompt = prompt_override
        else:
            end_prompt = edited_end or shot.end_frame_prompt
            style_label = scene.style.replace("_", " ")
            conditioning_prompt = (
                f"Generate the NEXT keyframe for this {style_label} shot, "
                f"showing clear visual progression {scene.target_clip_duration} seconds later.\n\n"
                f"TARGET END STATE (this is what the new image must depict):\n"
                f"{end_prompt}\n\n"
                f"The new image MUST show VISIBLE CHANGES from the reference image — "
                f"different pose, expression, body position, or camera framing. "
                f"If the reference is a close-up, the new image should show "
                f"a noticeably different expression, head angle, or gesture.\n\n"
                f"CONSISTENCY CONSTRAINTS:\n"
                f"- Same character appearance (face, hair, clothing, proportions)\n"
                f"- Same {style_label} rendering style\n"
                f"{character_prefix}"
            )

        if is_comfyui and image_model.startswith("flux-"):
            from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS as _FR4
            fw4, fh4 = _FR4.get(scene.aspect_ratio, (1024, 1024))
            image_bytes = await _generate_image_comfyui_flux(
                comfy_client, conditioning_prompt,
                seed=scene.seed + shot.shot_index + 1000,
                width=fw4, height=fh4,
            )
        elif is_comfyui:
            image_bytes = await _generate_image_comfyui(
                comfy_client, conditioning_prompt,
                seed=scene.seed + shot.shot_index + 1000,
            )
        else:
            image_bytes = await _generate_image_conditioned(
                image_client, conditioning_bytes, conditioning_prompt,
                scene.aspect_ratio, image_model,
                reference_images=ref_image_bytes_list or None,
            )

        prompt_used = prompt_override or edited_end or shot.end_frame_prompt

    # Save with versioned path
    storage = get_storage_backend()
    if isinstance(storage, LocalStorageBackend):
        filepath = file_mgr.save_keyframe_versioned(scene.id, shot.shot_index, position, image_bytes)
        stored_path = str(filepath)
    else:
        stored_path = await file_mgr.save_keyframe_versioned_async(scene.id, shot.shot_index, position, image_bytes)

    # Delete old keyframe for this position
    old_kf_result = await session.execute(
        select(Keyframe).where(Keyframe.shot_id == shot.id, Keyframe.position == position)
    )
    old_kf = old_kf_result.scalar_one_or_none()
    old_kf_path = old_kf.file_path if old_kf else None
    if old_kf:
        await session.delete(old_kf)
        await session.flush()

    # Clean up old S3 file (different key than the new versioned one)
    if old_kf_path and old_kf_path != stored_path and not storage.is_local():
        try:
            await storage.delete(old_kf_path)
        except Exception:
            logger.warning(f"Failed to delete old S3 keyframe {old_kf_path}")

    # Create new keyframe row
    kf = Keyframe(
        shot_id=shot.id,
        position=position,
        prompt_used=prompt_used,
        file_path=stored_path,
        mime_type="image/png",
        source="generated",
    )
    session.add(kf)
    await session.flush()


async def _regenerate_clip(session, scene, shot, file_mgr, prompt_override=None, video_model_override=None, shot_edits=None):
    """Regenerate video clip for a shot."""
    from vidpipe.config import settings as app_settings

    # Determine prompt — priority: prompt_override > edited value > rewritten > committed
    edited_motion = (shot_edits or {}).get("video_motion_prompt")
    if prompt_override:
        video_prompt = prompt_override
    elif edited_motion:
        video_prompt = edited_motion
    else:
        video_prompt = shot.video_motion_prompt
        from vidpipe.db.models import ShotManifest as ShotManifestModel
        sm_result = await session.execute(
            select(ShotManifestModel).where(
                ShotManifestModel.scene_id == scene.id,
                ShotManifestModel.shot_index == shot.shot_index,
            )
        )
        sm = sm_result.scalar_one_or_none()
        if sm and sm.rewritten_video_prompt:
            video_prompt = sm.rewritten_video_prompt

    # Load keyframes
    kf_result = await session.execute(
        select(Keyframe).where(Keyframe.shot_id == shot.id)
    )
    keyframes = kf_result.scalars().all()
    start_kf = next((k for k in keyframes if k.position == "start"), None)
    end_kf = next((k for k in keyframes if k.position == "end"), None)

    if not start_kf:
        raise ValueError(f"No start keyframe for shot {shot.shot_index}")

    start_bytes = await file_mgr.read_bytes(start_kf.file_path)
    end_bytes = await file_mgr.read_bytes(end_kf.file_path) if end_kf else start_bytes

    # Submit video generation
    video_model = video_model_override or scene.video_model or app_settings.models.video_generator

    from vidpipe.pipeline.video_gen import COMFYUI_VIDEO_MODELS
    from vidpipe.db.models import UserSettings, DEFAULT_USER_ID
    us_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    user_settings = us_result.scalar_one_or_none()

    import asyncio
    poll_interval = app_settings.pipeline.video_poll_interval
    max_polls = app_settings.pipeline.video_poll_max

    if video_model in COMFYUI_VIDEO_MODELS:
        # ---- ComfyUI path ----
        from vidpipe.services.comfyui_client import get_comfyui_client
        from vidpipe.services.comfyui_adapter import ComfyUIVideoAdapter

        comfy_host = user_settings.comfyui_host if user_settings else None
        comfy_key = user_settings.comfyui_api_key if user_settings else None
        comfy_client = await get_comfyui_client(host=comfy_host, api_key=comfy_key)
        adapter = ComfyUIVideoAdapter(comfy_client)

        # Load character reference images (if production-bible scene)
        char_ref_bytes: list[bytes] = []
        if scene.production_bible_id:
            from vidpipe.pipeline.video_gen import _load_char_ref_images
            char_ref_bytes = await _load_char_ref_images(session, scene)

        operation_id = await adapter.submit(
            video_prompt=video_prompt,
            start_frame_bytes=start_bytes,
            end_frame_bytes=end_bytes if end_kf else None,
            char_ref_bytes=char_ref_bytes,
            aspect_ratio=scene.aspect_ratio,
            seed=scene.seed or 0,
            shot_index=shot.shot_index,
            video_model=video_model,
        )

        # Poll ComfyUI
        for _ in range(max_polls):
            status, error_msg = await adapter.poll(operation_id)
            if status == "completed":
                video_bytes, _duration = await adapter.download(operation_id)
                break
            elif status == "failed":
                raise RuntimeError(f"ComfyUI job failed: {error_msg}")
            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"ComfyUI timed out for shot {shot.shot_index}")
    else:
        # ---- Veo / Vertex path ----
        from vidpipe.pipeline.video_gen import _submit_video_job

        from vidpipe.services.vertex_client import get_client
        client = get_client(user_settings=user_settings)

        operation = await _submit_video_job(
            client, video_model, video_prompt,
            start_bytes, end_bytes, scene,
        )

        for _ in range(max_polls):
            op = await client.aio.operations.get(operation=operation)
            if op.done:
                break
            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"Video generation timed out for shot {shot.shot_index}")

        result_op = await client.aio.operations.get(operation=operation)
        if not result_op.done:
            raise TimeoutError("Video generation did not complete")

        video = result_op.response
        if hasattr(video, "generated_videos") and video.generated_videos:
            video_data = video.generated_videos[0]
            if hasattr(video_data, "video") and video_data.video:
                video_bytes = video_data.video.video_bytes
            else:
                raise ValueError("No video data in response")
        else:
            raise ValueError("No generated videos in response")

    # Save with versioned path
    storage = get_storage_backend()
    if isinstance(storage, LocalStorageBackend):
        filepath = file_mgr.save_clip_versioned(scene.id, shot.shot_index, video_bytes)
        stored_path = str(filepath)
    else:
        stored_path = await file_mgr.save_clip_versioned_async(scene.id, shot.shot_index, video_bytes)

    # Delete old clip
    old_clip_result = await session.execute(
        select(VideoClip).where(VideoClip.shot_id == shot.id)
    )
    old_clip = old_clip_result.scalar_one_or_none()
    old_clip_path = old_clip.local_path if old_clip else None
    if old_clip:
        await session.delete(old_clip)
        await session.flush()

    # Clean up old S3 file (different key than the new versioned one)
    if old_clip_path and old_clip_path != stored_path and not storage.is_local():
        try:
            await storage.delete(old_clip_path)
        except Exception:
            logger.warning(f"Failed to delete old S3 clip {old_clip_path}")

    # Create new clip row
    clip = VideoClip(
        shot_id=shot.id,
        status="complete",
        local_path=stored_path,
        source="generated",
        prompt_used=video_prompt,
    )
    session.add(clip)
    await session.flush()


# ============================================================================
# PipeSVN: Scene-wide Regeneration
# ============================================================================

class RegenerateSceneRequest(BaseModel):
    scope: str  # "stale", "all", "stitch_only", "storyboard", "keyframes", "clips", "all_phases"
    shot_indices: Optional[list[int]] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    run_through: Optional[str] = None  # For "all_phases": how far to go


@router.post("/scenes/{scene_id}/regenerate")
async def regenerate_scene(
    scene_id: uuid.UUID,
    body: RegenerateSceneRequest,
    background_tasks: BackgroundTasks,
):
    """Scene-wide regeneration with different scopes.

    Scopes:
    - stale: regenerate only stale assets
    - all: regenerate everything from keyframing
    - stitch_only: re-run stitcher
    """
    async with async_session() as session:
        result = await session.execute(
            select(Scene).where(Scene.id == scene_id)
        )
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

    if body.scope == "stitch_only":
        background_tasks.add_task(_run_restitch, scene_id)
    elif body.scope == "storyboard":
        effective_text_model = body.text_model or scene.text_model
        if not effective_text_model:
            raise HTTPException(
                status_code=422,
                detail="Cannot regenerate storyboard: missing text_model",
            )
        background_tasks.add_task(
            _run_storyboard_regeneration, scene_id,
            text_model_override=body.text_model,
        )
    elif body.scope == "keyframes":
        effective_text_model = body.text_model or scene.text_model
        effective_image_model = body.image_model or scene.image_model
        if not effective_text_model or not effective_image_model:
            missing = []
            if not effective_text_model:
                missing.append("text_model")
            if not effective_image_model:
                missing.append("image_model")
            raise HTTPException(
                status_code=422,
                detail=f"Cannot regenerate keyframes: missing {', '.join(missing)}",
            )
        background_tasks.add_task(
            _run_keyframes_regeneration, scene_id,
            image_model_override=body.image_model,
            text_model_override=body.text_model,
        )
    elif body.scope == "clips":
        effective_text_model = body.text_model or scene.text_model
        effective_image_model = body.image_model or scene.image_model
        effective_video_model = body.video_model or scene.video_model
        missing = []
        if not effective_text_model:
            missing.append("text_model")
        if not effective_image_model:
            missing.append("image_model")
        if not effective_video_model:
            missing.append("video_model")
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot regenerate clips: missing {', '.join(missing)}",
            )
        background_tasks.add_task(
            _run_clips_regeneration, scene_id,
            video_model_override=body.video_model,
            text_model_override=body.text_model,
        )
    elif body.scope == "all_phases":
        # Validate models based on run_through
        effective_text_model = body.text_model or scene.text_model
        if not effective_text_model:
            raise HTTPException(
                status_code=422,
                detail="Cannot regenerate all phases: missing text_model",
            )
        run_through = body.run_through
        if run_through != "storyboard":
            effective_image_model = body.image_model or scene.image_model
            if not effective_image_model:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot regenerate all phases: missing image_model",
                )
        if run_through not in ("storyboard", "keyframes"):
            effective_video_model = body.video_model or scene.video_model
            if not effective_video_model:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot regenerate all phases: missing video_model",
                )
        background_tasks.add_task(
            _run_all_phases_regeneration, scene_id,
            run_through=run_through,
            text_model_override=body.text_model,
            image_model_override=body.image_model,
            video_model_override=body.video_model,
        )
    else:
        # "stale" or "all" — legacy full regen
        missing = []
        if not scene.text_model:
            missing.append("text_model")
        if not scene.image_model:
            missing.append("image_model")
        if not scene.video_model:
            missing.append("video_model")
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot regenerate: missing required model settings: {', '.join(missing)}",
            )
        background_tasks.add_task(_run_scene_regeneration, scene_id, body.scope, body.shot_indices)

    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "scope": body.scope},
    )


# ---------------------------------------------------------------------------
# WebSocket: real-time pipeline progress
# ---------------------------------------------------------------------------

@router.websocket("/scenes/{scene_id}/ws")
async def scene_websocket(websocket: WebSocket, scene_id: str):
    """Push pipeline progress events to the frontend in real time."""
    from vidpipe.services.event_bus import event_bus

    await websocket.accept()
    queue = event_bus.subscribe(scene_id)

    async def _send_events():
        while True:
            event = await queue.get()
            await websocket.send_json(event)

    async def _heartbeat():
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat"})

    async def _receive():
        # Drain incoming messages (ping/pong) to detect disconnect
        while True:
            await websocket.receive_text()

    try:
        await asyncio.gather(_send_events(), _heartbeat(), _receive())
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        event_bus.unsubscribe(scene_id, queue)


async def _run_restitch(scene_id: uuid.UUID, *, _emit_complete: bool = True):
    """Background task to re-stitch current clips."""
    from vidpipe.services.event_bus import event_bus
    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            return

        from vidpipe.pipeline.stitcher import stitch_videos
        await stitch_videos(session, scene)
        await session.refresh(scene)

        from vidpipe.services.checkpoint_service import create_checkpoint
        cp = await create_checkpoint(session, scene, "Re-stitched video")
        await session.commit()
        event_bus.emit(scene_id, "checkpoint_created", sha=cp.sha if cp else None, message="Re-stitched video")
        if _emit_complete:
            event_bus.emit(scene_id, "regen_complete", scope="stitch")


async def _run_storyboard_regeneration(
    scene_id: uuid.UUID,
    text_model_override: Optional[str] = None,
    *,
    _emit_complete: bool = True,
):
    """Background task to regenerate storyboard text only."""
    from vidpipe.services.event_bus import event_bus
    try:
        async with async_session() as session:
            result = await session.execute(select(Scene).where(Scene.id == scene_id))
            scene = result.scalar_one_or_none()
            if not scene:
                return

            # Persist text_model override so generate_storyboard picks it up
            if text_model_override and text_model_override != scene.text_model:
                scene.text_model = text_model_override
                await session.flush()

            saved_status = scene.status

            # Load user settings so Ollama cloud credentials are available
            us_result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
            )
            user_settings = us_result.scalar_one_or_none()

            from vidpipe.services.llm import get_adapter
            text_adapter = get_adapter(scene.text_model, user_settings=user_settings)

            from vidpipe.pipeline.storyboard import generate_storyboard
            await generate_storyboard(session, scene, text_adapter=text_adapter)

            # generate_storyboard sets status to "keyframing" — restore original
            scene.status = saved_status

            from vidpipe.services.checkpoint_service import create_checkpoint
            cp = await create_checkpoint(session, scene, "Regenerated storyboard text")
            await session.commit()
            event_bus.emit(scene_id, "checkpoint_created", sha=cp.sha if cp else None, message="Regenerated storyboard text")
            if _emit_complete:
                event_bus.emit(scene_id, "regen_complete", scope="storyboard")
    except Exception as e:
        logger.error("Storyboard regeneration failed for %s: %s", scene_id, e, exc_info=True)
        # Clear stuck generation_status so the UI doesn't spin forever
        try:
            async with async_session() as cleanup_session:
                from sqlalchemy import update
                await cleanup_session.execute(
                    update(Shot)
                    .where(Shot.scene_id == scene_id, Shot.generation_status == "generating_text")
                    .values(generation_status=None)
                )
                await cleanup_session.commit()
        except Exception:
            logger.error("Failed to clear generation_status for %s", scene_id, exc_info=True)
        event_bus.emit(scene_id, "error", phase="storyboard", message=str(e))
        if not _emit_complete:
            raise  # propagate to chained caller so subsequent phases are skipped


async def _run_keyframes_regeneration(
    scene_id: uuid.UUID,
    image_model_override: Optional[str] = None,
    text_model_override: Optional[str] = None,
    *,
    _emit_complete: bool = True,
):
    """Background task to regenerate stale/missing keyframes only."""
    try:
        async with async_session() as session:
            result = await session.execute(select(Scene).where(Scene.id == scene_id))
            scene = result.scalar_one_or_none()
            if not scene:
                return

            # Persist model overrides
            if image_model_override and image_model_override != scene.image_model:
                scene.image_model = image_model_override
                await session.flush()
            if text_model_override and text_model_override != scene.text_model:
                scene.text_model = text_model_override
                await session.flush()

            saved_status = scene.status

            # Load user settings for adapter creation
            us_result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
            )
            user_settings = us_result.scalar_one_or_none()

            from vidpipe.services.llm import get_adapter
            text_adapter = get_adapter(scene.text_model, user_settings=user_settings)

            # Gap-filling: generate_keyframes already skips shots with existing keyframes
            from vidpipe.pipeline.keyframes import generate_keyframes
            await generate_keyframes(session, scene, text_adapter=text_adapter)

            # generate_keyframes sets status — restore original
            scene.status = saved_status

            from vidpipe.services.checkpoint_service import create_checkpoint
            cp = await create_checkpoint(session, scene, "Regenerated stale keyframes")
            await session.commit()
            from vidpipe.services.event_bus import event_bus
            event_bus.emit(scene_id, "checkpoint_created", sha=cp.sha if cp else None, message="Regenerated stale keyframes")
            if _emit_complete:
                event_bus.emit(scene_id, "regen_complete", scope="keyframes")
    except Exception as e:
        logger.error("Keyframes regeneration failed for %s: %s", scene_id, e, exc_info=True)
        from vidpipe.services.event_bus import event_bus
        event_bus.emit(scene_id, "error", phase="keyframes", message=str(e))
        if not _emit_complete:
            raise  # propagate to chained caller so subsequent phases are skipped


async def _run_clips_regeneration(
    scene_id: uuid.UUID,
    video_model_override: Optional[str] = None,
    text_model_override: Optional[str] = None,
    *,
    _emit_complete: bool = True,
):
    """Background task to regenerate stale/missing video clips only."""
    try:
        async with async_session() as session:
            result = await session.execute(select(Scene).where(Scene.id == scene_id))
            scene = result.scalar_one_or_none()
            if not scene:
                return

            # Persist model overrides
            if video_model_override and video_model_override != scene.video_model:
                scene.video_model = video_model_override
                await session.flush()
            if text_model_override and text_model_override != scene.text_model:
                scene.text_model = text_model_override
                await session.flush()

            saved_status = scene.status

            # Load user settings for adapter creation
            us_result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
            )
            user_settings = us_result.scalar_one_or_none()

            from vidpipe.services.llm import get_adapter
            text_adapter = get_adapter(scene.text_model, user_settings=user_settings)
            vision_adapter = None
            if scene.vision_model:
                vision_adapter = get_adapter(scene.vision_model, user_settings=user_settings)

            # Gap-filling: ensure shots without clips have the right status
            # for generate_videos to pick them up
            shots_result = await session.execute(
                select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
            )
            shots = shots_result.scalars().all()

            for shot in shots:
                clip_result = await session.execute(
                    select(VideoClip).where(VideoClip.shot_id == shot.id)
                )
                clip = clip_result.scalar_one_or_none()
                if not clip:
                    # No clip exists — mark as keyframes_done so generate_videos picks it up
                    shot.status = "keyframes_done"
            await session.flush()

            from vidpipe.pipeline.video_gen import generate_videos
            await generate_videos(
                session, scene,
                text_adapter=text_adapter,
                vision_adapter=vision_adapter,
            )

            # Restore original status
            scene.status = saved_status

            from vidpipe.services.checkpoint_service import create_checkpoint
            cp = await create_checkpoint(session, scene, "Regenerated stale clips")
            await session.commit()
            from vidpipe.services.event_bus import event_bus
            event_bus.emit(scene_id, "checkpoint_created", sha=cp.sha if cp else None, message="Regenerated stale clips")
            if _emit_complete:
                event_bus.emit(scene_id, "regen_complete", scope="clips")
    except Exception as e:
        logger.error("Clips regeneration failed for %s: %s", scene_id, e, exc_info=True)
        from vidpipe.services.event_bus import event_bus
        event_bus.emit(scene_id, "error", phase="clips", message=str(e))
        if not _emit_complete:
            raise  # propagate to chained caller so subsequent phases are skipped


async def _run_all_phases_regeneration(
    scene_id: uuid.UUID,
    run_through: Optional[str] = None,
    text_model_override: Optional[str] = None,
    image_model_override: Optional[str] = None,
    video_model_override: Optional[str] = None,
):
    """Background task to chain per-phase regeneration up to run_through."""
    from vidpipe.services.event_bus import event_bus
    try:
        # Always run storyboard (suppress per-phase regen_complete)
        await _run_storyboard_regeneration(
            scene_id, text_model_override=text_model_override,
            _emit_complete=False,
        )

        if run_through == "storyboard":
            event_bus.emit(scene_id, "regen_complete", scope="all_phases")
            return

        # Keyframes
        await _run_keyframes_regeneration(
            scene_id,
            image_model_override=image_model_override,
            text_model_override=text_model_override,
            _emit_complete=False,
        )

        if run_through == "keyframes":
            event_bus.emit(scene_id, "regen_complete", scope="all_phases")
            return

        # Clips
        await _run_clips_regeneration(
            scene_id,
            video_model_override=video_model_override,
            text_model_override=text_model_override,
            _emit_complete=False,
        )

        if run_through == "video":
            event_bus.emit(scene_id, "regen_complete", scope="all_phases")
            return

        # Full pipeline — also restitch
        await _run_restitch(scene_id, _emit_complete=False)
        event_bus.emit(scene_id, "regen_complete", scope="all_phases")
    except Exception as e:
        logger.error("All-phases regeneration failed for %s: %s", scene_id, e, exc_info=True)
        event_bus.emit(scene_id, "error", message=str(e))


async def _run_scene_regeneration(
    scene_id: uuid.UUID,
    scope: str,
    shot_indices: Optional[list[int]],
):
    """Background task for scene-wide regeneration."""
    logger.info("Scene regeneration %s for %s", scope, scene_id)

    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            return

        shots_result = await session.execute(
            select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
        )
        shots = shots_result.scalars().all()

        from vidpipe.services.file_manager import FileManager
        from vidpipe.services.checkpoint_service import (
            compute_keyframe_staleness, compute_clip_staleness, create_checkpoint,
        )
        from vidpipe.db.models import ShotManifest as ShotManifestModel
        file_mgr = FileManager()

        regenerated_count = 0
        for shot in shots:
            if shot_indices and shot.shot_index not in shot_indices:
                continue

            # Load shot manifest
            sm_result = await session.execute(
                select(ShotManifestModel).where(
                    ShotManifestModel.scene_id == scene.id,
                    ShotManifestModel.shot_index == shot.shot_index,
                )
            )
            sm = sm_result.scalar_one_or_none()

            # Load keyframes
            kf_result = await session.execute(select(Keyframe).where(Keyframe.shot_id == shot.id))
            kfs = kf_result.scalars().all()
            start_kf = next((k for k in kfs if k.position == "start"), None)
            end_kf = next((k for k in kfs if k.position == "end"), None)

            # Load clip
            clip_result = await session.execute(select(VideoClip).where(VideoClip.shot_id == shot.id))
            clip = clip_result.scalar_one_or_none()

            targets = []
            if scope == "all":
                targets = ["start_keyframe", "end_keyframe", "video_clip"]
            elif scope == "stale":
                if compute_keyframe_staleness(shot, start_kf, sm) == "stale":
                    targets.append("start_keyframe")
                if compute_keyframe_staleness(shot, end_kf, sm) == "stale":
                    targets.append("end_keyframe")
                if compute_clip_staleness(shot, clip, sm) == "stale":
                    targets.append("video_clip")

            for target in targets:
                try:
                    if target in ("start_keyframe", "end_keyframe"):
                        position = "start" if target == "start_keyframe" else "end"
                        await _regenerate_keyframe(session, scene, shot, position, file_mgr)
                    elif target == "video_clip":
                        await _regenerate_clip(session, scene, shot, file_mgr)
                    regenerated_count += 1
                except Exception as e:
                    logger.error("Regeneration failed for shot %d %s: %s", shot.shot_index, target, e)

        if regenerated_count > 0:
            # Auto-stitch after regeneration so the final video stays current
            try:
                from vidpipe.pipeline.stitcher import stitch_videos
                await stitch_videos(session, scene)
                await session.refresh(scene)
            except Exception as e:
                logger.error("Auto-stitch after regeneration failed: %s", e)

            await create_checkpoint(
                session, scene,
                f"Regenerated {regenerated_count} asset(s) ({scope})",
            )
            await session.commit()


# ============================================================================
# PipeSVN: Asset Upload/Replace
# ============================================================================

@router.put("/scenes/{scene_id}/shots/{shot_idx}/keyframes/{position}")
async def upload_keyframe(
    scene_id: uuid.UUID,
    shot_idx: int,
    position: str,
    file: UploadFile = File(...),
):
    """Upload a keyframe image to replace the generated one."""
    if position not in ("start", "end"):
        raise HTTPException(status_code=400, detail="Position must be 'start' or 'end'")

    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot).where(Shot.scene_id == scene.id, Shot.shot_index == shot_idx)
        )
        shot = shot_result.scalar_one_or_none()
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")

        # Read upload data
        data = await file.read()

        from vidpipe.services.file_manager import FileManager
        file_mgr = FileManager()
        storage = get_storage_backend()

        if isinstance(storage, LocalStorageBackend):
            filepath = file_mgr.save_keyframe_versioned(scene.id, shot_idx, position, data)
            stored_path = str(filepath)
        else:
            stored_path = await file_mgr.save_keyframe_versioned_async(scene.id, shot_idx, position, data)

        # Delete old keyframe
        old_kf_result = await session.execute(
            select(Keyframe).where(Keyframe.shot_id == shot.id, Keyframe.position == position)
        )
        old_kf = old_kf_result.scalar_one_or_none()
        old_kf_path = old_kf.file_path if old_kf else None
        if old_kf:
            await session.delete(old_kf)
            await session.flush()

        # Create new keyframe row
        kf = Keyframe(
            shot_id=shot.id,
            position=position,
            prompt_used="uploaded",
            file_path=stored_path,
            mime_type=file.content_type or "image/png",
            source="uploaded",
        )
        session.add(kf)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, scene,
            f"Uploaded {position} keyframe for shot {shot_idx + 1}",
        )
        await session.commit()

        # Clean up old S3 file after commit
        if old_kf_path and old_kf_path != stored_path and not storage.is_local():
            try:
                await storage.delete(old_kf_path)
            except Exception:
                logger.warning(f"Failed to delete old S3 keyframe {old_kf_path}")

        return {"status": "uploaded", "file_path": stored_path, "keyframe_id": str(kf.id)}


@router.put("/scenes/{scene_id}/shots/{shot_idx}/clip")
async def upload_clip(
    scene_id: uuid.UUID,
    shot_idx: int,
    file: UploadFile = File(...),
):
    """Upload a video clip to replace the generated one."""
    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot).where(Shot.scene_id == scene.id, Shot.shot_index == shot_idx)
        )
        shot = shot_result.scalar_one_or_none()
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")

        data = await file.read()

        from vidpipe.services.file_manager import FileManager
        file_mgr = FileManager()
        storage = get_storage_backend()

        if isinstance(storage, LocalStorageBackend):
            filepath = file_mgr.save_clip_versioned(scene.id, shot_idx, data)
            stored_path = str(filepath)
        else:
            stored_path = await file_mgr.save_clip_versioned_async(scene.id, shot_idx, data)

        # Delete old clip
        old_clip_result = await session.execute(
            select(VideoClip).where(VideoClip.shot_id == shot.id)
        )
        old_clip = old_clip_result.scalar_one_or_none()
        old_clip_path = old_clip.local_path if old_clip else None
        if old_clip:
            await session.delete(old_clip)
            await session.flush()

        clip = VideoClip(
            shot_id=shot.id,
            status="complete",
            local_path=stored_path,
            source="uploaded",
            prompt_used="uploaded",
        )
        session.add(clip)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, scene,
            f"Uploaded clip for shot {shot_idx + 1}",
        )
        await session.commit()

        # Clean up old S3 file after commit
        if old_clip_path and old_clip_path != stored_path and not storage.is_local():
            try:
                await storage.delete(old_clip_path)
            except Exception:
                logger.warning(f"Failed to delete old S3 clip {old_clip_path}")

        return {"status": "uploaded", "file_path": stored_path, "clip_id": str(clip.id)}


@router.delete("/scenes/{scene_id}/shots/{shot_idx}/clip")
async def delete_clip(scene_id: uuid.UUID, shot_idx: int):
    """Delete a shot's video clip and its S3/local file."""
    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot).where(Shot.scene_id == scene.id, Shot.shot_index == shot_idx)
        )
        shot = shot_result.scalar_one_or_none()
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")

        clip_result = await session.execute(
            select(VideoClip).where(VideoClip.shot_id == shot.id)
        )
        clip = clip_result.scalar_one_or_none()
        if not clip:
            raise HTTPException(status_code=404, detail="No clip found")

        old_path = clip.local_path
        await session.delete(clip)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, scene,
            f"Removed clip for shot {shot_idx + 1}",
        )
        await session.commit()

    # Clean up file from S3 after commit
    if old_path:
        storage = get_storage_backend()
        if not storage.is_local():
            try:
                await storage.delete(old_path)
            except Exception:
                logger.warning(f"Failed to delete S3 clip {old_path}")

    return {"status": "deleted", "shot_index": shot_idx}


@router.delete("/scenes/{scene_id}/shots/{shot_idx}/keyframes/{position}")
async def delete_keyframe(scene_id: uuid.UUID, shot_idx: int, position: str):
    """Delete a shot's keyframe and its S3/local file."""
    if position not in ("start", "end"):
        raise HTTPException(status_code=400, detail="Position must be 'start' or 'end'")

    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_id))
        scene = result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        shot_result = await session.execute(
            select(Shot).where(Shot.scene_id == scene.id, Shot.shot_index == shot_idx)
        )
        shot = shot_result.scalar_one_or_none()
        if not shot:
            raise HTTPException(status_code=404, detail="Shot not found")

        kf_result = await session.execute(
            select(Keyframe).where(Keyframe.shot_id == shot.id, Keyframe.position == position)
        )
        kf = kf_result.scalar_one_or_none()
        if not kf:
            raise HTTPException(status_code=404, detail="Keyframe not found")

        old_path = kf.file_path
        await session.delete(kf)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, scene,
            f"Removed {position} keyframe for shot {shot_idx + 1}",
        )
        await session.commit()

    # Clean up file from S3 after commit
    if old_path:
        storage = get_storage_backend()
        if not storage.is_local():
            try:
                await storage.delete(old_path)
            except Exception:
                logger.warning(f"Failed to delete S3 keyframe {old_path}")

    return {"status": "deleted", "shot_index": shot_idx, "position": position}


# ---------------------------------------------------------------------------
# Productions
# ---------------------------------------------------------------------------

class ProductionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[list] = None

class ProductionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list] = None
    production_bible_id: Optional[str] = None

class ProductionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[list] = None
    production_bible_id: Optional[str] = None
    scene_count: int = 0
    created_at: str
    updated_at: str

@router.post("/productions", response_model=ProductionResponse)
async def create_production(body: ProductionCreate):
    from vidpipe.db.models import Production
    async with async_session() as session:
        prod = Production(name=body.name, description=body.description, tags=body.tags)
        session.add(prod)
        await session.commit()
        await session.refresh(prod)
        return ProductionResponse(
            id=str(prod.id),
            name=prod.name,
            description=prod.description,
            tags=prod.tags,
            production_bible_id=str(prod.production_bible_id) if prod.production_bible_id else None,
            scene_count=0,
            created_at=prod.created_at.isoformat(),
            updated_at=prod.updated_at.isoformat(),
        )

@router.get("/productions")
async def list_productions():
    from vidpipe.db.models import Production
    async with async_session() as session:
        result = await session.execute(select(Production).order_by(Production.updated_at.desc()))
        prods = result.scalars().all()
        items = []
        for p in prods:
            count_result = await session.execute(
                select(sa_func.count()).select_from(Scene).where(Scene.production_id == p.id, Scene.deleted_at.is_(None))
            )
            count = count_result.scalar() or 0
            items.append(ProductionResponse(
                id=str(p.id),
                name=p.name,
                description=p.description,
                tags=p.tags,
                production_bible_id=str(p.production_bible_id) if p.production_bible_id else None,
                scene_count=count,
                created_at=p.created_at.isoformat(),
                updated_at=p.updated_at.isoformat(),
            ))
        return items

@router.get("/productions/{production_id}", response_model=ProductionResponse)
async def get_production(production_id: uuid.UUID):
    from vidpipe.db.models import Production
    async with async_session() as session:
        result = await session.execute(select(Production).where(Production.id == production_id))
        prod = result.scalar_one_or_none()
        if not prod:
            raise HTTPException(status_code=404, detail="Production not found")
        count_result = await session.execute(
            select(sa_func.count()).select_from(Scene).where(Scene.production_id == prod.id, Scene.deleted_at.is_(None))
        )
        count = count_result.scalar() or 0
        return ProductionResponse(
            id=str(prod.id),
            name=prod.name,
            description=prod.description,
            tags=prod.tags,
            production_bible_id=str(prod.production_bible_id) if prod.production_bible_id else None,
            scene_count=count,
            created_at=prod.created_at.isoformat(),
            updated_at=prod.updated_at.isoformat(),
        )

@router.put("/productions/{production_id}", response_model=ProductionResponse)
async def update_production(production_id: uuid.UUID, body: ProductionUpdate):
    from vidpipe.db.models import Production
    async with async_session() as session:
        result = await session.execute(select(Production).where(Production.id == production_id))
        prod = result.scalar_one_or_none()
        if not prod:
            raise HTTPException(status_code=404, detail="Production not found")
        if body.name is not None:
            prod.name = body.name
        if body.description is not None:
            prod.description = body.description
        if body.tags is not None:
            prod.tags = body.tags
        if body.production_bible_id is not None:
            prod.production_bible_id = uuid.UUID(body.production_bible_id)
        await session.commit()
        await session.refresh(prod)
        count_result = await session.execute(
            select(sa_func.count()).select_from(Scene).where(Scene.production_id == prod.id, Scene.deleted_at.is_(None))
        )
        count = count_result.scalar() or 0
        return ProductionResponse(
            id=str(prod.id),
            name=prod.name,
            description=prod.description,
            tags=prod.tags,
            production_bible_id=str(prod.production_bible_id) if prod.production_bible_id else None,
            scene_count=count,
            created_at=prod.created_at.isoformat(),
            updated_at=prod.updated_at.isoformat(),
        )

@router.delete("/productions/{production_id}")
async def delete_production(production_id: uuid.UUID):
    from vidpipe.db.models import Production
    async with async_session() as session:
        result = await session.execute(select(Production).where(Production.id == production_id))
        prod = result.scalar_one_or_none()
        if not prod:
            raise HTTPException(status_code=404, detail="Production not found")
        # Unlink scenes from this production
        await session.execute(
            update(Scene).where(Scene.production_id == production_id).values(production_id=None)
        )
        await session.delete(prod)
        await session.commit()
        return {"status": "deleted", "production_id": str(production_id)}

@router.post("/productions/{production_id}/scenes")
async def batch_add_scenes_to_production(production_id: uuid.UUID, body: BatchAddScenesRequest):
    """Batch-add multiple scenes to a production in one atomic commit."""
    from vidpipe.db.models import Production
    if not body.scene_ids:
        raise HTTPException(status_code=400, detail="scene_ids must not be empty")
    async with async_session() as session:
        prod = (await session.execute(select(Production).where(Production.id == production_id))).scalar_one_or_none()
        if not prod:
            raise HTTPException(status_code=404, detail="Production not found")
        parsed_ids = []
        for sid in body.scene_ids:
            try:
                parsed_ids.append(uuid.UUID(sid))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid scene ID: {sid}")
        scenes = (await session.execute(select(Scene).where(Scene.id.in_(parsed_ids)))).scalars().all()
        if len(scenes) != len(parsed_ids):
            found = {s.id for s in scenes}
            missing = [str(sid) for sid in parsed_ids if sid not in found]
            raise HTTPException(status_code=404, detail=f"Scenes not found: {', '.join(missing)}")
        for scene in scenes:
            scene.production_id = prod.id
        await session.commit()
        return {
            "status": "added",
            "production_id": str(production_id),
            "scene_ids": [str(sid) for sid in parsed_ids],
            "count": len(parsed_ids),
        }

@router.post("/productions/{production_id}/scenes/{scene_id}")
async def add_scene_to_production(production_id: uuid.UUID, scene_id: uuid.UUID):
    from vidpipe.db.models import Production
    async with async_session() as session:
        prod = (await session.execute(select(Production).where(Production.id == production_id))).scalar_one_or_none()
        if not prod:
            raise HTTPException(status_code=404, detail="Production not found")
        scene = (await session.execute(select(Scene).where(Scene.id == scene_id))).scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        scene.production_id = prod.id
        await session.commit()
        return {"status": "added", "production_id": str(production_id), "scene_id": str(scene_id)}

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

@router.post("/maintenance/cleanup-cache")
async def cleanup_local_cache():
    """Remove local file cache. S3 remains the source of truth.

    Safe to call — stitcher/pipeline will re-download from S3 as needed.
    """
    file_mgr = FileManager()
    tmp_dir = file_mgr.base_dir
    count = 0
    for child in tmp_dir.iterdir():
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            count += 1
        except Exception:
            logger.warning(f"Failed to remove {child}", exc_info=True)
    return {"cleared": count, "tmp_dir": str(tmp_dir)}


@router.delete("/productions/{production_id}/scenes/{scene_id}")
async def remove_scene_from_production(production_id: uuid.UUID, scene_id: uuid.UUID):
    async with async_session() as session:
        scene = (await session.execute(
            select(Scene).where(Scene.id == scene_id, Scene.production_id == production_id)
        )).scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found in this production")
        scene.production_id = None
        await session.commit()
        return {"status": "removed", "production_id": str(production_id), "scene_id": str(scene_id)}


# ============================================================================
# Phase 16: Legacy /api/manifests/* → /api/production-bibles/* (301 Redirects)
# ============================================================================

def _legacy_manifest_redirect(request: Request) -> RedirectResponse:
    """Redirect legacy /api/manifests/* to /api/production-bibles/*.

    Returns HTTP 301 Moved Permanently so clients and search engines update
    their bookmarks. Query string is preserved.
    """
    new_path = request.url.path.replace("/manifests", "/production-bibles", 1)
    if request.url.query:
        new_path += f"?{request.url.query}"
    return RedirectResponse(url=new_path, status_code=301)


@router.api_route("/manifests/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def legacy_manifest_path_redirect(request: Request, path: str):
    """Redirect /api/manifests/{anything} to /api/production-bibles/{anything}."""
    return _legacy_manifest_redirect(request)


@router.api_route("/manifests", methods=["GET", "POST"])
async def legacy_manifest_list_redirect(request: Request):
    """Redirect /api/manifests to /api/production-bibles."""
    return _legacy_manifest_redirect(request)
