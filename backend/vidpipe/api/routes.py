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

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import and_ as sa_and, case, func as sa_func, select
from sqlalchemy.orm import selectinload

from vidpipe.db import async_session
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip, Manifest, Asset, SceneManifest as SceneManifestModel, SceneAudioManifest as SceneAudioManifestModel, GenerationCandidate, UserSettings, DEFAULT_USER_ID, ProjectCheckpoint
from vidpipe.orchestrator.pipeline import run_pipeline
from vidpipe.orchestrator.state import can_resume
from vidpipe.schemas.storyboard import SceneSchema
from vidpipe.schemas.storyboard_enhanced import EnhancedSceneSchema
from vidpipe.services.file_manager import FileManager
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
}
ALLOWED_VIDEO_MODELS = {
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-fast-generate-001",
    "wan-2.2-ref-i2v",
    "wan-2.2-i2v",
}

# Video models that support audio generation
AUDIO_CAPABLE_MODELS = ALLOWED_VIDEO_MODELS - {"veo-2.0-generate-001", "wan-2.2-ref-i2v", "wan-2.2-i2v"}

# Allowed clip durations per video model
ALLOWED_DURATIONS: dict[str, list[int]] = {
    "veo-2.0-generate-001": [5, 6, 7, 8],
    "veo-3.0-generate-001": [4, 6, 8],
    "veo-3.0-fast-generate-001": [4, 6, 8],
    "veo-3.1-generate-preview": [4, 6, 8],
    "veo-3.1-generate-001": [4, 6, 8],
    "veo-3.1-fast-generate-preview": [4, 6, 8],
    "veo-3.1-fast-generate-001": [4, 6, 8],
    "wan-2.2-ref-i2v": [5],
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
}

VIDEO_MODEL_COST_SILENT: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-preview": 0.40,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-preview": 0.10,
    "veo-3.1-fast-generate-001": 0.10,
    "wan-2.2-ref-i2v": 0.0,
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
    "wan-2.2-ref-i2v": 0.0,
    "wan-2.2-i2v": 0.0,
}


def _estimate_project_cost(
    project: "Project",
    generated_keyframes: int = 0,
    completed_clips: int = 0,
    generated_clips: int = 0,
    has_storyboard: bool = False,
    billed_veo_submissions: int = 0,
    extra_image_regens: int = 0,
) -> float:
    """Estimate cost based on actual artifacts generated.

    For complete projects, uses theoretical cost minus inherited assets,
    plus extra costs from escalation retries and safety regens.
    For incomplete projects, uses billed_veo_submissions (all Veo ops
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
    clip_dur = project.target_clip_duration or 6
    audio = project.audio_enabled and project.video_model in AUDIO_CAPABLE_MODELS
    vid_rate = (
        VIDEO_MODEL_COST_AUDIO.get(project.video_model or "", 0.40)
        if audio
        else VIDEO_MODEL_COST_SILENT.get(project.video_model or "", 0.40)
    )
    img_rate = IMAGE_MODEL_COST.get(project.image_model or "", 0.04)
    text_cost_rate = TEXT_MODEL_COST.get(project.text_model or "", 0.01)

    if project.status == "complete":
        # Full theoretical cost minus inherited assets
        scene_count = project.target_scene_count or math.ceil(
            (project.total_duration or 15) / clip_dur
        )
        full_cost = (
            text_cost_rate
            + (scene_count + 1) * img_rate
            + scene_count * clip_dur * vid_rate
        )
        # Subtract inherited keyframe and clip costs
        inherited_kf = max(0, (scene_count + 1) - generated_keyframes) if generated_keyframes > 0 else 0
        inherited_clips = max(0, scene_count - generated_clips) if generated_clips > 0 else 0
        cost = full_cost - (inherited_kf * img_rate) - (inherited_clips * clip_dur * vid_rate)
        # Add extra costs from escalation retries beyond the base count
        extra_submissions = max(0, billed_veo_submissions - generated_clips)
        cost += extra_submissions * clip_dur * vid_rate
        # Add safety regen image costs
        cost += extra_image_regens * img_rate
        return cost

    if project.status == "pending":
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
    manifest_id: Optional[str] = None
    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: bool = False
    candidate_count: int = 1
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None
    # Selective stage execution
    run_through: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response schema for POST /api/generate."""
    project_id: str
    status: str
    status_url: str


class StatusResponse(BaseModel):
    """Response schema for GET /api/projects/{id}/status."""
    project_id: str
    status: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class SceneReference(BaseModel):
    """Asset reference selected for a scene's Veo generation."""
    asset_id: str
    manifest_tag: str
    name: str
    asset_type: str
    thumbnail_url: Optional[str] = None
    reference_image_url: Optional[str] = None
    quality_score: Optional[float] = None
    is_face_crop: bool = False


class SceneDetail(BaseModel):
    """Scene detail within ProjectDetail response."""
    scene_index: int
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
    selected_references: list[SceneReference] = []
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


class ProjectDetail(BaseModel):
    """Response schema for GET /api/projects/{id}."""
    project_id: str
    title: Optional[str] = None
    prompt: str
    style: str
    aspect_ratio: str
    status: str
    created_at: str
    updated_at: str
    scene_count: int
    scenes: list[SceneDetail]
    error_message: Optional[str] = None
    total_duration: Optional[int] = None
    clip_duration: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    forked_from: Optional[str] = None
    manifest_id: Optional[str] = None
    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: bool = False
    candidate_count: int = 1
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None
    # Selective stage execution
    run_through: Optional[str] = None
    # PipeSVN
    head_sha: Optional[str] = None


class SceneEditPayload(BaseModel):
    """Per-scene edits within an edit request."""
    scene_description: Optional[str] = None
    start_frame_prompt: Optional[str] = None
    end_frame_prompt: Optional[str] = None
    video_motion_prompt: Optional[str] = None
    transition_notes: Optional[str] = None


class EditProjectRequest(BaseModel):
    """Request body for PATCH /api/projects/{id}/edit."""
    prompt: Optional[str] = None
    title: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    clip_duration: Optional[int] = None
    target_scene_count: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    vision_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    scene_edits: Optional[dict[int, SceneEditPayload]] = None
    removed_scenes: Optional[list[int]] = None
    commit_message: Optional[str] = None
    expected_sha: Optional[str] = None


class EditProjectResponse(BaseModel):
    """Response from PATCH /api/projects/{id}/edit."""
    project_id: str
    head_sha: str
    message: str
    changes_count: int


class ProjectListItem(BaseModel):
    """Item in list response for GET /api/projects."""
    project_id: str
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
    style: str = ""
    aspect_ratio: str = ""
    thumbnail_url: Optional[str] = None


class PaginatedProjects(BaseModel):
    """Paginated response envelope for GET /api/projects."""
    items: list[ProjectListItem]
    total: int
    page: int
    per_page: int


class ResumeResponse(BaseModel):
    """Response schema for POST /api/projects/{id}/resume."""
    project_id: str
    status: str
    status_url: str


class ContinueRequest(BaseModel):
    """Optional body for POST /api/projects/{id}/resume to advance run_through."""
    run_through: Optional[str] = None  # "keyframes", "video", or "all" (= run everything)
    # Model overrides — applied to the project before resuming
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
    """Request schema for POST /api/projects/{id}/fork."""
    prompt: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None
    clip_duration: Optional[int] = None
    total_duration: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None
    scene_edits: Optional[dict[int, dict[str, str]]] = None
    deleted_scenes: Optional[list[int]] = None
    clear_keyframes: Optional[list[int]] = None
    asset_changes: Optional[AssetChanges] = None
    # Phase 13: LLM Provider Abstraction
    vision_model: Optional[str] = None


class ForkResponse(BaseModel):
    """Response schema for POST /api/projects/{id}/fork."""
    project_id: str
    forked_from: str
    status: str
    status_url: str
    copied_scenes: int
    resume_from: str


class StopResponse(BaseModel):
    """Response schema for POST /api/projects/{id}/stop."""
    project_id: str
    status: str


class MetricsResponse(BaseModel):
    """Response schema for GET /api/metrics."""
    total_projects: int
    status_counts: dict[str, int]
    style_counts: dict[str, int]
    aspect_ratio_counts: dict[str, int]
    text_model_counts: dict[str, int]
    image_model_counts: dict[str, int]
    video_model_counts: dict[str, int]
    audio_counts: dict[str, int]
    scene_count_counts: dict[str, int]
    total_estimated_cost: float
    total_video_seconds: int
    avg_clip_duration: Optional[float] = None


# Manifest System Schemas

class CreateManifestRequest(BaseModel):
    """Request schema for POST /api/manifests."""
    name: str
    description: Optional[str] = None
    category: str = "CUSTOM"
    tags: Optional[list[str]] = Field(default=None)


class UpdateManifestRequest(BaseModel):
    """Request schema for PUT /api/manifests/{id}."""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None


class CreateManifestFromProjectRequest(BaseModel):
    """Request schema for POST /api/manifests/from-project."""
    project_id: str
    name: Optional[str] = None


class ManifestListItem(BaseModel):
    """Item in list response for GET /api/manifests."""
    manifest_id: str
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


class ManifestDetailResponse(BaseModel):
    """Response schema for GET /api/manifests/{id}."""
    manifest_id: str
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
    parent_manifest_id: Optional[str]
    source_video_duration: Optional[float] = None
    created_at: str
    updated_at: str
    assets: list["AssetResponse"]


class VideoUploadResponse(BaseModel):
    """Response schema for POST /api/manifests/{id}/upload-video."""
    task_id: str
    status: str
    manifest_id: str


class CreateAssetRequest(BaseModel):
    """Request schema for POST /api/manifests/{id}/assets."""
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
    manifest_id: str
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


# ============================================================================
# Background Task Wrapper
# ============================================================================

async def run_pipeline_background(project_id: uuid.UUID):
    """Run pipeline in background with fresh session.

    CRITICAL: Never share session across async boundaries.
    Create fresh session inside background task.
    """
    async with async_session() as session:
        try:
            await run_pipeline(session, project_id)
        except Exception as e:
            # Error already persisted to database by orchestrator
            logger.error(f"Background pipeline failed for {project_id}: {type(e).__name__}: {str(e)}")


# ============================================================================
# Endpoint Handlers
# ============================================================================

@router.post("/generate", status_code=202, response_model=GenerateResponse)
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start video generation pipeline in background.

    Creates project record with pending status and adds pipeline execution
    to background tasks. Returns 202 Accepted with project_id and status URL.
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

    # Derive scene count from total duration and clip duration
    scene_count = math.ceil(request.total_duration / request.clip_duration)

    async with async_session() as session:
        # Create project record
        project = Project(
            title=request.title,
            prompt=request.prompt,
            style=request.style,
            aspect_ratio=request.aspect_ratio,
            target_clip_duration=request.clip_duration,
            target_scene_count=scene_count,
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
        session.add(project)
        await session.flush()  # Get project.id before snapshot creation

        # Handle manifest_id if provided
        if request.manifest_id:
            manifest_uuid = uuid.UUID(request.manifest_id)

            # Validate manifest exists
            manifest = await manifest_service.get_manifest(session, manifest_uuid)
            if not manifest:
                raise HTTPException(status_code=404, detail=f"Manifest {request.manifest_id} not found")

            # Set project manifest fields
            project.manifest_id = manifest_uuid
            project.manifest_version = manifest.version

            # Create snapshot
            await manifest_service.create_snapshot(session, manifest_uuid, project.id)

            # Increment usage tracking
            await manifest_service.increment_usage(session, manifest_uuid)

            logger.info(f"Project {project.id} using manifest {request.manifest_id}, snapshot created")

            # Note: Conditional manifesting skip (Phase 6 success criteria #5) is achieved
            # by the presence of manifest_id on the project. When manifest_id is set, the
            # pipeline knows assets are pre-processed (snapshot exists). The pipeline's
            # manifesting step (to be added in Phase 7+) will check project.manifest_id
            # and skip if present. For now, the pipeline has no manifesting step, so the
            # skip is implicit — pre-built manifests just bypass the need for one.

        await session.commit()
        await session.refresh(project)

        project_id = project.id
        logger.info(f"Created project {project_id} for prompt: {request.prompt[:50]}...")

    # Add background task AFTER committing project
    background_tasks.add_task(run_pipeline_background, project_id)

    return GenerateResponse(
        project_id=str(project_id),
        status="pending",
        status_url=f"/api/projects/{project_id}/status",
    )


@router.get("/projects/{project_id}/status", response_model=StatusResponse)
async def get_project_status(project_id: uuid.UUID):
    """Get lightweight project status for polling.

    Returns project-level status only (no scene details).
    Use GET /api/projects/{id} for full detail.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return StatusResponse(
            project_id=str(project.id),
            status=project.status,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            error_message=project.error_message,
        )


@router.get("/projects/{project_id}", response_model=ProjectDetail)
async def get_project_detail(project_id: uuid.UUID):
    """Get full project detail with scene breakdown.

    Includes scene-level status, keyframe existence, and clip status.
    """
    async with async_session() as session:
        # Load project
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Load scenes with their keyframes and clips
        scenes_result = await session.execute(
            select(Scene)
            .where(Scene.project_id == project_id)
            .order_by(Scene.scene_index)
        )
        scenes = scenes_result.scalars().all()

        # Build scene details
        scene_details = []
        for scene in scenes:
            # Get keyframes for this scene
            keyframes_result = await session.execute(
                select(Keyframe).where(Keyframe.scene_id == scene.id)
            )
            keyframes = keyframes_result.scalars().all()

            has_start_keyframe = any(kf.position == "start" for kf in keyframes)
            has_end_keyframe = any(kf.position == "end" for kf in keyframes)

            # Get clip for this scene
            clip_result = await session.execute(
                select(VideoClip).where(VideoClip.scene_id == scene.id)
            )
            clip = clip_result.scalar_one_or_none()

            start_kf = next((kf for kf in keyframes if kf.position == "start"), None)
            end_kf = next((kf for kf in keyframes if kf.position == "end"), None)

            scene_details.append(SceneDetail(
                scene_index=scene.scene_index,
                description=scene.scene_description,
                status=scene.status,
                has_start_keyframe=has_start_keyframe,
                has_end_keyframe=has_end_keyframe,
                has_clip=clip is not None,
                clip_status=clip.status if clip else None,
                start_frame_prompt=scene.start_frame_prompt,
                end_frame_prompt=scene.end_frame_prompt,
                video_motion_prompt=scene.video_motion_prompt,
                transition_notes=scene.transition_notes,
                start_keyframe_url=f"/api/keyframes/{start_kf.id}" if start_kf else None,
                end_keyframe_url=f"/api/keyframes/{end_kf.id}" if end_kf else None,
                clip_url=f"/api/clips/{clip.id}" if clip and clip.status == "complete" and clip.local_path else None,
            ))

        # Load selected reference tags for all scenes (Phase 8)
        sm_result = await session.execute(
            select(SceneManifestModel).where(
                SceneManifestModel.project_id == project.id
            )
        )
        scene_manifests_by_index = {
            sm.scene_index: sm for sm in sm_result.scalars().all()
        }

        # If project has manifest, load assets for reference resolution
        ref_assets_by_tag = {}
        if project.manifest_id:
            assets_result = await session.execute(
                select(Asset).where(Asset.manifest_id == project.manifest_id)
            )
            ref_assets_by_tag = {
                a.manifest_tag: a for a in assets_result.scalars().all()
            }

        # Enrich scene_details with selected references
        for sd in scene_details:
            sm = scene_manifests_by_index.get(sd.scene_index)
            if sm and sm.selected_reference_tags:
                refs = []
                for tag in sm.selected_reference_tags:
                    asset = ref_assets_by_tag.get(tag)
                    if asset:
                        refs.append(SceneReference(
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

        # Enrich scene_details with staleness (PipeSVN)
        from vidpipe.services.checkpoint_service import compute_keyframe_staleness, compute_clip_staleness
        for sd in scene_details:
            sm = scene_manifests_by_index.get(sd.scene_index)
            # Find the matching scene object
            scene_obj = next((s for s in scenes if s.scene_index == sd.scene_index), None)
            if not scene_obj:
                continue

            # Get keyframes and clip for this scene
            kf_result2 = await session.execute(
                select(Keyframe).where(Keyframe.scene_id == scene_obj.id)
            )
            kfs = kf_result2.scalars().all()
            start_kf = next((k for k in kfs if k.position == "start"), None)
            end_kf = next((k for k in kfs if k.position == "end"), None)

            clip_result2 = await session.execute(
                select(VideoClip).where(VideoClip.scene_id == scene_obj.id)
            )
            clip_obj = clip_result2.scalar_one_or_none()

            sd.start_keyframe_staleness = compute_keyframe_staleness(scene_obj, start_kf, sm)
            sd.end_keyframe_staleness = compute_keyframe_staleness(scene_obj, end_kf, sm)
            sd.clip_staleness = compute_clip_staleness(scene_obj, clip_obj, sm)
            sd.start_keyframe_prompt_used = start_kf.prompt_used if start_kf else None
            sd.end_keyframe_prompt_used = end_kf.prompt_used if end_kf else None
            sd.clip_prompt_used = clip_obj.prompt_used if clip_obj else None
            if sm:
                sd.rewritten_keyframe_prompt = sm.rewritten_keyframe_prompt
                sd.rewritten_video_prompt = sm.rewritten_video_prompt

        return ProjectDetail(
            project_id=str(project.id),
            title=project.title,
            prompt=project.prompt,
            style=project.style,
            aspect_ratio=project.aspect_ratio,
            status=project.status,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            scene_count=len(scenes),
            scenes=scene_details,
            error_message=project.error_message,
            total_duration=project.total_duration,
            clip_duration=project.target_clip_duration,
            text_model=project.text_model,
            image_model=project.image_model,
            video_model=project.video_model,
            audio_enabled=project.audio_enabled,
            forked_from=str(project.forked_from_id) if project.forked_from_id else None,
            manifest_id=str(project.manifest_id) if project.manifest_id else None,
            quality_mode=project.quality_mode,
            candidate_count=project.candidate_count,
            vision_model=project.vision_model,
            run_through=project.run_through,
            head_sha=project.head_sha,
        )


@router.get("/projects", response_model=PaginatedProjects)
async def list_projects(
    page: int = 1,
    per_page: int = 10,
    view: Optional[str] = None,
    status: Optional[str] = None,
):
    """List projects with server-side pagination (newest first).

    Query params:
      - page: 1-based page number (default 1)
      - per_page: items per page, must be 10, 50, or 100 (default 10)
      - view: "cards" to include thumbnail_url (avoids extra query for list view)
      - status: filter by project status (e.g. "complete", "failed", "stopped")
    """
    if per_page not in (10, 50, 100):
        per_page = 10
    if page < 1:
        page = 1

    VALID_STATUSES = {"pending", "storyboarding", "keyframing", "video_gen", "stitching", "complete", "failed", "stopped", "staged"}

    async with async_session() as session:
        filters = [Project.deleted_at.is_(None)]
        if status and status in VALID_STATUSES:
            filters.append(Project.status == status)
        base_filter = filters[0] if len(filters) == 1 else sa_and(*filters)

        # Total count
        count_result = await session.execute(
            select(sa_func.count(Project.id)).where(base_filter)
        )
        total = count_result.scalar() or 0

        # Paginated projects
        result = await session.execute(
            select(Project)
            .where(base_filter)
            .order_by(Project.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
        projects = result.scalars().all()

        # Build thumbnail map when cards view requested
        thumbnail_map: dict[str, str] = {}
        if view == "cards" and projects:
            project_ids = [p.id for p in projects]
            thumb_q = await session.execute(
                select(Scene.project_id, Keyframe.id)
                .join(Scene, Keyframe.scene_id == Scene.id)
                .where(Scene.project_id.in_(project_ids))
                .where(Scene.scene_index == 0)
                .where(Keyframe.position == "start")
            )
            for row in thumb_q:
                pid_str = str(row[0])
                if pid_str not in thumbnail_map:
                    thumbnail_map[pid_str] = f"/api/keyframes/{row[1]}"

        return PaginatedProjects(
            items=[
                ProjectListItem(
                    project_id=str(p.id),
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
                )
                for p in projects
            ],
            total=total,
            page=page,
            per_page=per_page,
        )


@router.post("/projects/{project_id}/resume", status_code=202, response_model=ResumeResponse)
async def resume_project(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    body: Optional[ContinueRequest] = None,
):
    """Resume failed or interrupted pipeline in background.

    Accepts optional ContinueRequest body to advance run_through when
    continuing from a "staged" state.

    Returns 409 if project is not in a resumable state (including complete).
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Check if project can be resumed
        if not can_resume(project.status):
            raise HTTPException(
                status_code=409,
                detail=f"Project cannot be resumed from status '{project.status}'"
            )

        # Apply updates from ContinueRequest body
        if body:
            # Update run_through when continuing from staged state
            if project.status == "staged" and body.run_through is not None:
                new_val = None if body.run_through == "all" else body.run_through
                if new_val is not None and new_val not in ("storyboard", "keyframes", "video"):
                    raise HTTPException(
                        status_code=422,
                        detail=f"run_through must be 'storyboard', 'keyframes', 'video', or 'all'; got '{body.run_through}'",
                    )
                project.run_through = new_val

            # Apply model overrides
            if body.image_model is not None:
                if body.image_model not in ALLOWED_IMAGE_MODELS:
                    raise HTTPException(status_code=422, detail=f"Invalid image_model: {body.image_model}")
                project.image_model = body.image_model
            if body.vision_model is not None:
                if not (body.vision_model in ALLOWED_TEXT_MODELS or body.vision_model.startswith("ollama/")):
                    raise HTTPException(status_code=422, detail=f"Invalid vision_model: {body.vision_model}")
                project.vision_model = body.vision_model if body.vision_model else None
            if body.video_model is not None:
                if body.video_model not in ALLOWED_VIDEO_MODELS:
                    raise HTTPException(status_code=422, detail=f"Invalid video_model: {body.video_model}")
                project.video_model = body.video_model
            if body.audio_enabled is not None:
                if body.audio_enabled and project.video_model not in AUDIO_CAPABLE_MODELS:
                    raise HTTPException(status_code=422, detail=f"Audio not supported for {project.video_model}")
                project.audio_enabled = body.audio_enabled
            if body.clip_duration is not None:
                allowed = ALLOWED_DURATIONS.get(project.video_model or "", [5, 6, 7, 8])
                if body.clip_duration not in allowed:
                    raise HTTPException(
                        status_code=422,
                        detail=f"clip_duration {body.clip_duration} not supported for {project.video_model}. Allowed: {allowed}",
                    )
                project.target_clip_duration = body.clip_duration

            await session.commit()

        logger.info(f"Resuming project {project_id} from status {project.status}")

    # Add background task
    background_tasks.add_task(run_pipeline_background, project_id)

    return ResumeResponse(
        project_id=str(project_id),
        status=project.status,
        status_url=f"/api/projects/{project_id}/status",
    )


@router.post("/projects/{project_id}/stop", response_model=StopResponse)
async def stop_project(project_id: uuid.UUID):
    """Stop a running pipeline.

    Sets the project status to 'stopped'. The background pipeline checks this
    flag between steps and inside long-running loops, then exits gracefully.
    The project can be resumed later with POST /resume.

    Returns 409 if project is already in a terminal state (complete/failed/stopped).
    """
    ACTIVE_STATUSES = {"pending", "storyboarding", "keyframing", "video_gen", "stitching"}

    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.status not in ACTIVE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Project cannot be stopped from status '{project.status}'"
            )

        project.status = "stopped"
        await session.commit()

        logger.info(f"Project {project_id} marked as stopped")

    return StopResponse(project_id=str(project_id), status="stopped")


class UpdateProjectRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@router.patch("/projects/{project_id}", response_model=dict)
async def update_project(project_id: uuid.UUID, body: UpdateProjectRequest):
    """Update mutable project fields (currently just title)."""
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project or project.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Project not found")

        project.title = body.title
        await session.commit()

    return {"project_id": str(project_id), "title": body.title}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: uuid.UUID):
    """Soft-delete a project: sets deleted_at, removes disk assets.

    Only terminal-status projects (complete, failed, stopped) can be deleted.
    DB records are preserved for cost tracking.
    """
    DELETABLE_STATUSES = {"complete", "failed", "stopped"}

    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project or project.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.status not in DELETABLE_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete project with status '{project.status}'. Stop it first.",
            )

        project.deleted_at = sa_func.now()
        await session.commit()

    # Remove disk assets (keyframes, clips, output) after commit
    try:
        project_dir = FileManager().base_dir / str(project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)
            logger.info(f"Removed disk assets for project {project_id}")
    except Exception:
        logger.warning(f"Failed to remove disk assets for project {project_id}", exc_info=True)

    return {"status": "deleted", "project_id": str(project_id)}


class _ExpansionScenes(BaseModel):
    """Wrapper schema for Gemini structured output of expansion scenes."""
    scenes: list[SceneSchema] = Field(description="New scenes to append")


class _EnhancedExpansionScenes(BaseModel):
    """Wrapper schema for Gemini structured output of manifest-aware expansion scenes."""
    scenes: list[EnhancedSceneSchema] = Field(description="New scenes to append with manifest and audio data")


async def _generate_expansion_scenes(
    project: Project,
    kept_scenes: list[dict],
    num_new_scenes: int,
    start_index: int,
    asset_registry_block: Optional[str] = None,
    text_adapter=None,
) -> list[dict]:
    """Generate storyboard entries for new scenes extending an existing storyboard.

    Uses the LLM adapter system (Vertex AI or Ollama) to create scene entries
    that continue the narrative from the kept scenes.

    Args:
        project: The new forked project (has prompt, style, etc.)
        kept_scenes: storyboard_raw["scenes"] entries for kept scenes
        num_new_scenes: How many new scenes to generate
        start_index: scene_index for the first new scene
        asset_registry_block: When provided, enables manifest-aware generation
            with scene_manifest and audio_manifest in the output schema.
        text_adapter: Optional LLMAdapter instance. If None, falls back to
            Vertex AI with the project's text_model or gemini-2.5-flash.

    Returns:
        List of scene dicts ready for Scene record creation
    """
    from vidpipe.services.llm import get_adapter

    if text_adapter is None:
        model_id = project.text_model or "gemini-2.5-flash"
        text_adapter = get_adapter(model_id)

    style_label = (project.style or "cinematic").replace("_", " ")

    # Build context from kept scenes
    kept_summary = json.dumps(kept_scenes, indent=2) if kept_scenes else "[]"

    # Choose schema based on whether asset registry is available
    use_enhanced = asset_registry_block is not None
    schema = _EnhancedExpansionScenes if use_enhanced else _ExpansionScenes

    manifest_instructions = ""
    if use_enhanced:
        manifest_instructions = f"""
AVAILABLE ASSETS (from Asset Registry):
{asset_registry_block}

SCENE MANIFEST INSTRUCTIONS:
- Reference registered assets by their [TAG] (e.g., [CHAR_01], [ENV_02])
- Use the asset's reverse_prompt for visual detail
- Assign roles: subject | background | prop | interaction_target | environment
- Specify spatial positions and actions for each placed asset
- Include composition metadata: shot_type, camera_movement, focal_point
- Add continuity_notes describing visual continuity with previous scenes
- You MAY declare new_asset_declarations for assets NOT in the registry

AUDIO MANIFEST INSTRUCTIONS:
- dialogue_lines: Map speech to character tags (speaker_tag must be a registered [TAG])
- sfx: Sound effects with trigger descriptions and relative timing
- ambient: Base layer soundscape + environmental context
- music: Style, mood, tempo, instruments, and transition cues
- audio_continuity: What carries from previous scene, what's new, what cuts
"""

    prompt = f"""You are a storyboard director. You have an existing partial storyboard and need to generate {num_new_scenes} NEW continuation scene(s).

VISUAL STYLE: {style_label}
ASPECT RATIO: {project.aspect_ratio}
{manifest_instructions}
ORIGINAL SCRIPT:
{project.prompt}

EXISTING SCENES (already in the storyboard — do NOT repeat these):
{kept_summary}

Generate exactly {num_new_scenes} new scene(s) that continue the narrative from where the existing scenes leave off.
- Scene indices must start at {start_index} and increment by 1.
- Maintain visual consistency with the existing scenes (same characters, style, color palette).
- Follow the same keyframe prompt format: "A {style_label} rendering of..." with subject, action, setting, lighting, camera, style cues, color palette.
- Motion prompts should describe ONLY motion/camera, not re-describe visuals.
- Transition notes should connect smoothly from the last existing scene to the first new scene, and between new scenes.
- Include key_details for each scene (3-6 specific terms from the original script).
"""

    result = await text_adapter.generate_text(
        prompt=prompt,
        schema=schema,
        temperature=0.7,
        max_retries=3,
    )

    # Normalize scene indices to start at start_index
    scenes_out = []
    for i, scene in enumerate(result.scenes[:num_new_scenes]):
        sd = scene.model_dump()
        sd["scene_index"] = start_index + i
        scenes_out.append(sd)

    return scenes_out


async def _copy_assets_for_fork(
    session,
    source_manifest_id,
    source_project_id,
    new_project_id,
    asset_changes,  # Optional[AssetChanges]
) -> tuple[list, list[str]]:
    """Copy parent manifest assets to forked project with inheritance tracking.

    Returns:
        (list_of_new_assets, list_of_modified_asset_tags)
    """
    # Load canonical (non-inherited) assets for this manifest to avoid
    # cascading duplication when forking a project that was itself forked
    result = await session.execute(
        select(Asset).where(
            Asset.manifest_id == source_manifest_id,
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
            inherited_from_project=source_project_id,
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
        select(Asset).where(Asset.inherited_from_project == new_project_id)
    )

    return new_assets, modified_asset_tags


async def _copy_scene_manifests(
    session,
    source_project_id,
    new_project_id,
    scene_boundary: int,
    deleted_set: set,
):
    """Copy scene manifests for all non-deleted source scenes.

    Scenes below the invalidation boundary get a full copy.
    Scenes at/above the boundary get structural fields only (regen fields cleared).
    Uses post-deletion scene index mapping.
    """
    # Load source scene manifests
    result = await session.execute(
        select(SceneManifestModel).where(
            SceneManifestModel.project_id == source_project_id
        )
    )
    source_sms = list(result.scalars().all())

    def _old_to_new(old_idx: int) -> int:
        """Map original scene index to post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    for sm in source_sms:
        # Skip scenes that were deleted
        if sm.scene_index in deleted_set:
            continue

        new_idx = _old_to_new(sm.scene_index)

        if new_idx < scene_boundary:
            # Full copy for scenes below boundary
            new_sm = SceneManifestModel(
                project_id=new_project_id,
                scene_index=new_idx,
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
            # Structural copy for scenes at/above boundary — clear regen fields
            new_sm = SceneManifestModel(
                project_id=new_project_id,
                scene_index=new_idx,
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


async def _copy_scene_audio_manifests(
    session,
    source_project_id,
    new_project_id,
    deleted_set: set,
):
    """Copy scene audio manifests for all non-deleted source scenes.

    Audio manifests have no regen-specific fields, so all columns are copied
    as-is with scene index remapping for deleted scenes.
    """
    result = await session.execute(
        select(SceneAudioManifestModel).where(
            SceneAudioManifestModel.project_id == source_project_id
        )
    )
    source_sams = list(result.scalars().all())

    def _old_to_new(old_idx: int) -> int:
        """Map original scene index to post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    for sam in source_sams:
        if sam.scene_index in deleted_set:
            continue

        new_idx = _old_to_new(sam.scene_index)

        new_sam = SceneAudioManifestModel(
            project_id=new_project_id,
            scene_index=new_idx,
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
    scene_manifests: list,  # SceneManifest rows
    modified_asset_tags: set,
    deleted_set: set,
) -> int:
    """Find earliest scene using any modified asset.

    Returns:
        Scene index of first affected scene, or -1 if no match.
    """
    earliest = float("inf")
    for sm in scene_manifests:
        if sm.scene_index in deleted_set:
            continue
        if sm.asset_tags:
            for tag in sm.asset_tags:
                if tag in modified_asset_tags:
                    earliest = min(earliest, sm.scene_index)
                    break
    return int(earliest) if earliest != float("inf") else -1


def _compute_invalidation(
    source: Project,
    overrides: dict,
    scene_edits: Optional[dict[int, dict[str, str]]],
    deleted_scenes: Optional[list[int]] = None,
    clear_keyframes: Optional[list[int]] = None,
) -> tuple[str, int]:
    """Compute the pipeline resume point and scene copy boundary for a fork.

    Returns (resume_stage, scene_copy_boundary) where:
    - resume_stage: the pipeline status to start from
    - scene_copy_boundary: scenes with index < boundary are fully copied,
      scenes at/after boundary get partial or no copying depending on stage

    Scene boundary uses *new* (post-deletion) numbering when deleted_scenes
    are present.

    Inherited scenes are always preserved — global setting changes (prompt,
    style, models, aspect_ratio, etc.) never trigger re-generation of
    inherited content.  Only per-scene edits and explicit keyframe clears
    move the boundary.
    """
    scene_count = source.target_scene_count or 3

    # Compute deletion info
    deleted_set = set(deleted_scenes) if deleted_scenes else set()
    deleted_count = len(deleted_set)
    kept_count = scene_count - deleted_count

    # Compute new scene count from duration settings
    new_total = overrides.get("total_duration", source.total_duration) or source.total_duration or 15
    new_clip = overrides.get("clip_duration", source.target_clip_duration) or source.target_clip_duration or 6
    new_scene_count = math.ceil(new_total / new_clip)

    def _old_to_new(old_idx: int) -> int:
        """Map an original scene index to its post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    # Start with all scenes preserved (boundaries at new_scene_count = no invalidation)
    min_keyframe_boundary = new_scene_count
    min_video_boundary = new_scene_count

    # Cleared keyframes → keyframe invalidation
    if clear_keyframes:
        for idx in clear_keyframes:
            if 0 <= idx < scene_count and idx not in deleted_set:
                new_idx = _old_to_new(idx)
                min_keyframe_boundary = min(min_keyframe_boundary, new_idx)

    # Scene-level edits
    if scene_edits:
        for idx, edits in scene_edits.items():
            if idx in deleted_set:
                continue
            for field in edits:
                if field in ("start_frame_prompt", "end_frame_prompt"):
                    new_idx = _old_to_new(idx)
                    min_keyframe_boundary = min(min_keyframe_boundary, new_idx)
                elif field == "video_motion_prompt":
                    new_idx = _old_to_new(idx)
                    min_video_boundary = min(min_video_boundary, new_idx)

    # Expansion: new scenes need keyframes + video
    if new_scene_count > kept_count:
        min_keyframe_boundary = min(min_keyframe_boundary, kept_count)

    if min_keyframe_boundary < new_scene_count:
        return "keyframing", min_keyframe_boundary
    if min_video_boundary < new_scene_count:
        return "video_gen", min_video_boundary

    # No invalidation — just re-stitch
    return "stitching", new_scene_count


@router.post("/projects/{project_id}/fork", status_code=202, response_model=ForkResponse)
async def fork_project(project_id: uuid.UUID, request: ForkRequest, background_tasks: BackgroundTasks):
    """Fork a project with optional edits and resume from the appropriate pipeline stage.

    Creates a new project that copies existing assets up to the edit point, then
    resumes the pipeline from there. Copied assets are marked as 'inherited'.
    """
    async with async_session() as session:
        # Load source project
        result = await session.execute(select(Project).where(Project.id == project_id))
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Project not found")

        # Must be terminal
        if source.status not in ("complete", "failed", "stopped"):
            raise HTTPException(status_code=409, detail=f"Can only fork terminal projects, got '{source.status}'")

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

        # Validate deleted_scenes — must leave at least 1 scene
        deleted_set = set(request.deleted_scenes) if request.deleted_scenes else set()
        src_scene_count = source.target_scene_count or 3
        if deleted_set and len(deleted_set) >= src_scene_count:
            raise HTTPException(status_code=422, detail="Cannot delete all scenes; at least 1 must remain")

        # Validate asset_changes requires a manifest on the source project
        if request.asset_changes is not None and source.manifest_id is None:
            raise HTTPException(status_code=422, detail="Cannot apply asset_changes to a project without a manifest")

        # Compute invalidation point
        resume_from, scene_boundary = _compute_invalidation(
            source, overrides, request.scene_edits,
            deleted_scenes=request.deleted_scenes,
            clear_keyframes=request.clear_keyframes,
        )

        # Phase 12: Asset modification invalidation
        # If asset_changes has modified assets, check which scenes use them
        # and potentially tighten the invalidation boundary.
        if request.asset_changes is not None and source.manifest_id is not None:
            modified_asset_ids = list((request.asset_changes.modified_assets or {}).keys())
            if modified_asset_ids:
                # Load parent assets to find their manifest_tags
                parent_assets_result = await session.execute(
                    select(Asset).where(Asset.manifest_id == source.manifest_id)
                )
                parent_assets_map = {
                    str(a.id): a.manifest_tag for a in parent_assets_result.scalars().all()
                }
                mod_tags = {
                    parent_assets_map[aid] for aid in modified_asset_ids if aid in parent_assets_map
                }

                if mod_tags:
                    # Load source scene manifests for asset invalidation check
                    src_sm_result = await session.execute(
                        select(SceneManifestModel).where(
                            SceneManifestModel.project_id == source.id
                        )
                    )
                    src_scene_manifests = list(src_sm_result.scalars().all())

                    asset_inv_point = _compute_asset_invalidation_point(
                        src_scene_manifests, mod_tags, deleted_set
                    )

                    if asset_inv_point >= 0 and asset_inv_point < scene_boundary:
                        # Asset modification affects scenes before current boundary —
                        # tighten to asset invalidation point (keyframe re-run needed)
                        scene_boundary = asset_inv_point
                        if resume_from == "stitching" or resume_from == "video_gen":
                            resume_from = "keyframing"

        # Merge settings
        new_total = overrides.get("total_duration", source.total_duration)
        new_clip = overrides.get("clip_duration", source.target_clip_duration)
        new_scene_count = math.ceil((new_total or 15) / (new_clip or 6))

        # Adjust scene count and total duration for deletions — but only when
        # the frontend did NOT explicitly send total_duration. When total_duration
        # is explicit, new_scene_count already reflects the user's desired count
        # (including any expansion after deletion).
        if deleted_set and "total_duration" not in overrides:
            new_scene_count = max(1, new_scene_count - len(deleted_set))
            new_total = new_scene_count * (new_clip or 6)

        # Create new project
        new_project = Project(
            prompt=overrides.get("prompt", source.prompt),
            style=overrides.get("style", source.style),
            aspect_ratio=ar,
            target_clip_duration=new_clip,
            target_scene_count=new_scene_count,
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

        # Always copy storyboard data — inherited scenes are always preserved
        new_project.storyboard_raw = source.storyboard_raw
        new_project.style_guide = source.style_guide

        # Phase 12: Inherit manifest_id and manifest_version from source
        if source.manifest_id is not None:
            new_project.manifest_id = source.manifest_id
            new_project.manifest_version = source.manifest_version

        session.add(new_project)
        await session.commit()
        await session.refresh(new_project)

        new_id = new_project.id
        file_mgr = FileManager()
        file_mgr.get_project_dir(new_id)  # ensure dirs exist

        # Build set of scenes whose keyframes should be cleared
        clear_kf_set = set(request.clear_keyframes) if request.clear_keyframes else set()

        # Load source scenes
        src_scenes_result = await session.execute(
            select(Scene).where(Scene.project_id == source.id).order_by(Scene.scene_index)
        )
        src_scenes = src_scenes_result.scalars().all()

        copied_scenes = 0
        new_idx = 0  # renumbered index for non-deleted scenes

        if resume_from != "pending":
            for src_scene in src_scenes:
                idx = src_scene.scene_index

                # Skip deleted scenes
                if idx in deleted_set:
                    continue

                # Apply scene-level edits
                scene_desc = src_scene.scene_description
                start_fp = src_scene.start_frame_prompt
                end_fp = src_scene.end_frame_prompt
                motion = src_scene.video_motion_prompt
                transition = src_scene.transition_notes

                if request.scene_edits and idx in request.scene_edits:
                    edits = request.scene_edits[idx]
                    scene_desc = edits.get("scene_description", scene_desc)
                    start_fp = edits.get("start_frame_prompt", start_fp)
                    end_fp = edits.get("end_frame_prompt", end_fp)
                    motion = edits.get("video_motion_prompt", motion)
                    transition = edits.get("transition_notes", transition)

                # Determine what to copy for this scene (using new_idx for boundary checks)
                if resume_from == "stitching":
                    new_scene_status = "video_done"
                    copy_keyframes = True
                    copy_clip = True
                elif resume_from == "video_gen":
                    if new_idx < scene_boundary:
                        new_scene_status = "video_done"
                        copy_keyframes = True
                        copy_clip = True
                    else:
                        new_scene_status = "keyframes_done"
                        copy_keyframes = True
                        copy_clip = False
                elif resume_from == "keyframing":
                    if new_idx < scene_boundary:
                        new_scene_status = "video_done"
                        copy_keyframes = True
                        copy_clip = True
                    else:
                        new_scene_status = "pending"
                        copy_keyframes = False
                        copy_clip = False
                else:
                    new_scene_status = "pending"
                    copy_keyframes = False
                    copy_clip = False

                # If keyframes are explicitly cleared for this scene, don't copy them
                if idx in clear_kf_set:
                    copy_keyframes = False
                    copy_clip = False
                    if new_scene_status not in ("pending",):
                        new_scene_status = "pending"

                new_scene = Scene(
                    project_id=new_id,
                    scene_index=new_idx,
                    scene_description=scene_desc,
                    start_frame_prompt=start_fp,
                    end_frame_prompt=end_fp,
                    video_motion_prompt=motion,
                    transition_notes=transition,
                    status=new_scene_status,
                )
                session.add(new_scene)
                await session.flush()  # get new_scene.id

                # Copy keyframes
                if copy_keyframes:
                    kf_result = await session.execute(
                        select(Keyframe).where(Keyframe.scene_id == src_scene.id)
                    )
                    for kf in kf_result.scalars().all():
                        src_path = Path(kf.file_path)
                        if src_path.exists():
                            dst_path = file_mgr.get_project_dir(new_id) / "keyframes" / f"scene_{new_idx}_{kf.position}.png"
                            shutil.copy2(str(src_path), str(dst_path))
                            new_kf = Keyframe(
                                scene_id=new_scene.id,
                                position=kf.position,
                                prompt_used=kf.prompt_used,
                                file_path=str(dst_path),
                                mime_type=kf.mime_type,
                                source="inherited",
                            )
                            session.add(new_kf)

                # Copy clip
                if copy_clip:
                    clip_result = await session.execute(
                        select(VideoClip).where(VideoClip.scene_id == src_scene.id)
                    )
                    clip = clip_result.scalar_one_or_none()
                    if clip and clip.local_path:
                        src_clip_path = Path(clip.local_path)
                        if src_clip_path.exists():
                            dst_clip_path = file_mgr.get_project_dir(new_id) / "clips" / f"scene_{new_idx}.mp4"
                            shutil.copy2(str(src_clip_path), str(dst_clip_path))
                            new_clip = VideoClip(
                                scene_id=new_scene.id,
                                source="inherited",
                                status="complete",
                                local_path=str(dst_clip_path),
                                gcs_uri=clip.gcs_uri,
                                duration_seconds=clip.duration_seconds,
                            )
                            session.add(new_clip)

                if new_scene_status != "pending":
                    copied_scenes += 1

                new_idx += 1

            # Update storyboard_raw: apply edits, prune deleted scenes, renumber
            if new_project.storyboard_raw:
                sb = dict(new_project.storyboard_raw)
                if "scenes" in sb:
                    # Apply scene edits first (on original indices)
                    if request.scene_edits:
                        for idx_str, edits in request.scene_edits.items():
                            edit_idx = int(idx_str)
                            if edit_idx < len(sb["scenes"]):
                                for field, value in edits.items():
                                    sb["scenes"][edit_idx][field] = value
                    # Remove deleted scenes (iterate in reverse to preserve indices)
                    if deleted_set:
                        sb["scenes"] = [
                            s for i, s in enumerate(sb["scenes"]) if i not in deleted_set
                        ]
                        # Renumber scene_index to match post-deletion DB indices
                        for new_idx_sb, scene_data in enumerate(sb["scenes"]):
                            scene_data["scene_index"] = new_idx_sb
                    new_project.storyboard_raw = sb

        # Phase 12: Copy manifest assets and scene manifests for forked project
        if source.manifest_id is not None:
            # Copy assets with inheritance tracking
            _, modified_asset_tags = await _copy_assets_for_fork(
                session,
                source.manifest_id,
                source.id,
                new_id,
                request.asset_changes,
            )

            # Always copy scene manifests — inherited scenes are always preserved
            await _copy_scene_manifests(
                session,
                source.id,
                new_id,
                scene_boundary,
                deleted_set,
            )

            await _copy_scene_audio_manifests(
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
                        Asset.inherited_from_project == new_id,
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
                    manifest_id=source.manifest_id,
                    new_uploads=request.asset_changes.new_uploads,
                    existing_face_embeddings=existing_face_embeddings,
                )

        await session.commit()

    # Start pipeline in background
    background_tasks.add_task(run_pipeline_background, new_id)

    return ForkResponse(
        project_id=str(new_id),
        forked_from=str(project_id),
        status=resume_from,
        status_url=f"/api/projects/{new_id}/status",
        copied_scenes=copied_scenes,
        resume_from=resume_from,
    )


@router.patch("/projects/{project_id}/edit", response_model=EditProjectResponse)
async def edit_project_in_place(project_id: uuid.UUID, body: EditProjectRequest):
    """Edit project in-place with checkpoint (PipeSVN).

    Only allowed on terminal-status projects. Uses optimistic concurrency
    via expected_sha to prevent lost updates.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Must be terminal
        if project.status not in ("complete", "failed", "stopped", "staged"):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot edit project in status '{project.status}'"
            )

        # Optimistic concurrency check
        if body.expected_sha is not None and body.expected_sha != project.head_sha:
            raise HTTPException(
                status_code=409,
                detail=f"Conflict: expected head_sha={body.expected_sha}, "
                       f"actual={project.head_sha}. Another edit was committed."
            )

        changes = []

        # Apply project-level field changes
        field_map = {
            "prompt": body.prompt,
            "title": body.title,
            "style": body.style,
            "aspect_ratio": body.aspect_ratio,
            "target_clip_duration": body.clip_duration,
            "target_scene_count": body.target_scene_count,
            "text_model": body.text_model,
            "image_model": body.image_model,
            "video_model": body.video_model,
            "vision_model": body.vision_model,
            "audio_enabled": body.audio_enabled,
        }
        for attr, value in field_map.items():
            if value is not None:
                old_val = getattr(project, attr)
                if old_val != value:
                    setattr(project, attr, value)
                    changes.append({"type": "project_field", "field": attr, "old": str(old_val), "new": str(value)})

        # Apply scene edits
        if body.scene_edits:
            for scene_idx, edits in body.scene_edits.items():
                scene_result = await session.execute(
                    select(Scene).where(
                        Scene.project_id == project.id,
                        Scene.scene_index == int(scene_idx),
                    )
                )
                scene = scene_result.scalar_one_or_none()
                if not scene:
                    continue

                scene_field_map = {
                    "scene_description": edits.scene_description,
                    "start_frame_prompt": edits.start_frame_prompt,
                    "end_frame_prompt": edits.end_frame_prompt,
                    "video_motion_prompt": edits.video_motion_prompt,
                    "transition_notes": edits.transition_notes,
                }
                for attr, value in scene_field_map.items():
                    if value is not None:
                        old_val = getattr(scene, attr)
                        if old_val != value:
                            setattr(scene, attr, value)
                            changes.append({
                                "type": "scene_field",
                                "scene_index": int(scene_idx),
                                "field": attr,
                            })

        # Handle removed scenes (set status to "removed")
        if body.removed_scenes:
            for scene_idx in body.removed_scenes:
                scene_result = await session.execute(
                    select(Scene).where(
                        Scene.project_id == project.id,
                        Scene.scene_index == scene_idx,
                    )
                )
                scene = scene_result.scalar_one_or_none()
                if scene:
                    scene.status = "removed"
                    changes.append({"type": "scene_removed", "scene_index": scene_idx})

        # Handle scene expansion (target_scene_count increase)
        if body.target_scene_count is not None:
            existing_result = await session.execute(
                select(sa_func.count(Scene.id)).where(
                    Scene.project_id == project.id,
                    Scene.status != "removed",
                )
            )
            existing_count = existing_result.scalar() or 0
            if body.target_scene_count > existing_count:
                # Get max scene_index
                max_idx_result = await session.execute(
                    select(sa_func.max(Scene.scene_index)).where(
                        Scene.project_id == project.id
                    )
                )
                max_idx = max_idx_result.scalar() or 0
                for i in range(body.target_scene_count - existing_count):
                    new_idx = max_idx + 1 + i
                    new_scene = Scene(
                        project_id=project.id,
                        scene_index=new_idx,
                        scene_description="",
                        start_frame_prompt="",
                        end_frame_prompt="",
                        video_motion_prompt="",
                        status="pending",
                    )
                    session.add(new_scene)
                    changes.append({"type": "scene_added", "scene_index": new_idx})

        if not changes:
            raise HTTPException(status_code=400, detail="No changes to commit")

        # Create checkpoint
        from vidpipe.services.checkpoint_service import create_checkpoint
        message = body.commit_message or f"Edit: {len(changes)} change(s)"
        checkpoint = await create_checkpoint(
            session, project, message, metadata={"changes": changes}
        )

        await session.commit()

        return EditProjectResponse(
            project_id=str(project.id),
            head_sha=checkpoint.sha,
            message=message,
            changes_count=len(changes),
        )


@router.get("/projects/{project_id}/download")
async def download_video(project_id: uuid.UUID, dl: int = 0):
    """Download or stream final MP4 video file.

    Query params:
    - dl=1: force Content-Disposition: attachment (browser download).
      Default serves inline for <video> element streaming.
    - v=<sha>: ignored, used as client-side cache buster.

    Returns 409 if project is not complete.
    Returns 404 if output file does not exist.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Check if project is complete
        if project.status != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Project not ready for download (status: {project.status})"
            )

        # Check if output file exists
        if not project.output_path:
            raise HTTPException(status_code=404, detail="Output file path not set")

        output_path = Path(project.output_path)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found on disk")

        filename = f"video_{project_id}.mp4"
        disposition = "attachment" if dl else "inline"
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=filename,
            headers={
                "Content-Disposition": f'{disposition}; filename="{filename}"'
            }
        )


@router.get("/keyframes/{keyframe_id}")
async def get_keyframe_image(keyframe_id: uuid.UUID):
    """Serve a keyframe image by its database ID.

    Returns the PNG image file from disk.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Keyframe).where(Keyframe.id == keyframe_id)
        )
        keyframe = result.scalar_one_or_none()

        if not keyframe:
            raise HTTPException(status_code=404, detail="Keyframe not found")

        file_path = Path(keyframe.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Keyframe file not found on disk")

        return FileResponse(
            path=str(file_path),
            media_type=keyframe.mime_type,
        )


@router.get("/clips/{clip_id}")
async def get_clip_video(clip_id: uuid.UUID):
    """Serve a video clip by its database ID.

    Returns the MP4 video file from disk.
    """
    async with async_session() as session:
        result = await session.execute(
            select(VideoClip).where(VideoClip.id == clip_id)
        )
        clip = result.scalar_one_or_none()

        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")

        if not clip.local_path:
            raise HTTPException(status_code=404, detail="Clip file path not set")

        file_path = Path(clip.local_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Clip file not found on disk")

        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Aggregate metrics across all projects.

    Cost estimation is artifact-based: complete projects use theoretical full
    cost; incomplete projects count only actually-generated keyframes and
    completed video clips.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.deleted_at.is_(None))
        )
        projects = result.scalars().all()

        # Build lookup of actual artifacts per project for cost accuracy.
        # generated keyframes per project (only source='generated', not 'inherited')
        kf_q = await session.execute(
            select(
                Scene.project_id,
                sa_func.count(Keyframe.id),
            )
            .join(Scene, Keyframe.scene_id == Scene.id)
            .where(Keyframe.source == "generated")
            .group_by(Scene.project_id)
        )
        kf_counts: dict[uuid.UUID, int] = {row[0]: row[1] for row in kf_q}

        # completed video clips per project (only source='generated')
        vc_q = await session.execute(
            select(
                Scene.project_id,
                sa_func.count(VideoClip.id),
            )
            .join(Scene, VideoClip.scene_id == Scene.id)
            .where(VideoClip.status == "complete")
            .where(VideoClip.source == "generated")
            .group_by(Scene.project_id)
        )
        vc_counts: dict[uuid.UUID, int] = {row[0]: row[1] for row in vc_q}

        # Billed Veo submissions per project: all clips with operation_name
        # (Google charges even for failed/filtered operations)
        submission_expr = case(
            (VideoClip.veo_submission_count > 0, VideoClip.veo_submission_count),
            else_=1,  # backward compat: old rows with operation_name but count=0
        )
        billed_q = await session.execute(
            select(
                Scene.project_id,
                sa_func.sum(submission_expr),
            )
            .join(Scene, VideoClip.scene_id == Scene.id)
            .where(VideoClip.operation_name.isnot(None))
            .where(VideoClip.source == "generated")
            .group_by(Scene.project_id)
        )
        billed_counts: dict[uuid.UUID, int] = {row[0]: int(row[1]) for row in billed_q}

        # Safety regen image calls per project
        regen_q = await session.execute(
            select(
                Scene.project_id,
                sa_func.sum(VideoClip.safety_regen_count),
            )
            .join(Scene, VideoClip.scene_id == Scene.id)
            .where(VideoClip.safety_regen_count > 0)
            .group_by(Scene.project_id)
        )
        regen_counts: dict[uuid.UUID, int] = {row[0]: int(row[1]) for row in regen_q}

        # actual scene counts per project
        sc_q = await session.execute(
            select(
                Scene.project_id,
                sa_func.count(Scene.id),
            )
            .group_by(Scene.project_id)
        )
        scene_counts_per_project: dict[uuid.UUID, int] = {row[0]: row[1] for row in sc_q}

    status_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    aspect_ratio_counts: Counter[str] = Counter()
    text_model_counts: Counter[str] = Counter()
    image_model_counts: Counter[str] = Counter()
    video_model_counts: Counter[str] = Counter()
    audio_counts: Counter[str] = Counter()
    scene_count_counts: Counter[str] = Counter()
    total_cost = 0.0
    total_seconds = 0
    clip_durations: list[int] = []

    for p in projects:
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
        sc = scene_counts_per_project.get(p.id, 0)
        if sc > 0:
            scene_count_counts[str(sc)] += 1
        if p.total_duration:
            total_seconds += p.total_duration
        if p.target_clip_duration:
            clip_durations.append(p.target_clip_duration)
        total_cost += _estimate_project_cost(
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
        total_projects=len(projects),
        status_counts=dict(status_counts),
        style_counts=dict(style_counts),
        aspect_ratio_counts=dict(aspect_ratio_counts),
        text_model_counts=dict(text_model_counts),
        image_model_counts=dict(image_model_counts),
        video_model_counts=dict(video_model_counts),
        audio_counts=dict(audio_counts),
        scene_count_counts=dict(scene_count_counts),
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
# Manifest API Endpoints
# ============================================================================

def _manifest_to_list_item(m: Manifest) -> ManifestListItem:
    """Convert Manifest ORM model to ManifestListItem response."""
    return ManifestListItem(
        manifest_id=str(m.id),
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
        manifest_id=str(a.manifest_id),
        asset_type=a.asset_type,
        name=a.name,
        manifest_tag=a.manifest_tag,
        user_tags=a.user_tags,
        reference_image_url=a.reference_image_url,
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


@router.post("/manifests", status_code=201, response_model=ManifestListItem)
async def create_manifest(request: CreateManifestRequest):
    """Create a new manifest in DRAFT status."""
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


@router.post("/manifests/from-project", status_code=201, response_model=ManifestDetailResponse)
async def create_manifest_from_project(request: CreateManifestFromProjectRequest):
    """Create a manifest pre-populated from a project's storyboard data."""
    try:
        project_id = uuid.UUID(request.project_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid project_id format")

    async with async_session() as session:
        try:
            manifest, assets = await manifest_service.create_manifest_from_project(
                session,
                project_id=project_id,
                name=request.name,
            )
            await session.commit()
            await session.refresh(manifest)
            for a in assets:
                await session.refresh(a)

            return ManifestDetailResponse(
                manifest_id=str(manifest.id),
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
                parent_manifest_id=str(manifest.parent_manifest_id) if manifest.parent_manifest_id else None,
                source_video_duration=manifest.source_video_duration,
                created_at=manifest.created_at.isoformat(),
                updated_at=manifest.updated_at.isoformat(),
                assets=[_asset_to_response(a) for a in assets],
            )
        except ValueError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@router.get("/manifests", response_model=list[ManifestListItem])
async def list_manifests(
    category: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "updated_at",
    sort_order: str = "desc",
):
    """List manifests with optional filters and sorting."""
    async with async_session() as session:
        manifests = await manifest_service.list_manifests(
            session,
            category=category,
            status=status,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return [_manifest_to_list_item(m) for m in manifests]


@router.get("/manifests/{manifest_id}", response_model=ManifestDetailResponse)
async def get_manifest_detail(manifest_id: uuid.UUID):
    """Get manifest detail with assets."""
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found")

        assets = await manifest_service.list_assets(session, manifest_id)

        return ManifestDetailResponse(
            manifest_id=str(manifest.id),
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
            parent_manifest_id=str(manifest.parent_manifest_id) if manifest.parent_manifest_id else None,
            source_video_duration=manifest.source_video_duration,
            created_at=manifest.created_at.isoformat(),
            updated_at=manifest.updated_at.isoformat(),
            assets=[_asset_to_response(a) for a in assets],
        )


@router.put("/manifests/{manifest_id}", response_model=ManifestListItem)
async def update_manifest(manifest_id: uuid.UUID, request: UpdateManifestRequest):
    """Update manifest fields."""
    async with async_session() as session:
        try:
            # Build kwargs from non-None fields
            kwargs = {k: v for k, v in request.model_dump().items() if v is not None}
            manifest = await manifest_service.update_manifest(
                session,
                manifest_id,
                **kwargs,
            )
            await session.commit()
            await session.refresh(manifest)
            return _manifest_to_list_item(manifest)
        except ValueError as e:
            if "not found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@router.delete("/manifests/{manifest_id}")
async def delete_manifest(manifest_id: uuid.UUID):
    """Soft delete manifest. Returns 409 if referenced by projects."""
    async with async_session() as session:
        try:
            await manifest_service.delete_manifest(session, manifest_id)
            await session.commit()
            return {"status": "deleted", "manifest_id": str(manifest_id)}
        except ValueError as e:
            error_msg = str(e)
            if "not found" in error_msg:
                raise HTTPException(status_code=404, detail=error_msg)
            if "referenced by" in error_msg:
                raise HTTPException(status_code=409, detail=error_msg)
            raise HTTPException(status_code=422, detail=error_msg)


@router.post("/manifests/{manifest_id}/duplicate", status_code=201, response_model=ManifestListItem)
async def duplicate_manifest(manifest_id: uuid.UUID, name: Optional[str] = None):
    """Duplicate manifest with all assets."""
    async with async_session() as session:
        try:
            new_manifest = await manifest_service.duplicate_manifest(
                session,
                manifest_id,
                new_name=name,
            )
            await session.commit()
            await session.refresh(new_manifest)
            return _manifest_to_list_item(new_manifest)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("/manifests/{manifest_id}/assets", status_code=201, response_model=AssetResponse)
async def create_asset(manifest_id: uuid.UUID, request: CreateAssetRequest):
    """Create asset within manifest."""
    async with async_session() as session:
        try:
            asset = await manifest_service.create_asset(
                session,
                manifest_id=manifest_id,
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


@router.get("/manifests/{manifest_id}/assets", response_model=list[AssetResponse])
async def list_assets(manifest_id: uuid.UUID):
    """List assets for manifest."""
    async with async_session() as session:
        assets = await manifest_service.list_assets(session, manifest_id)
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

        # Save file (wrap in asyncio.to_thread since save_asset_image is sync)
        file_path = await asyncio.to_thread(
            manifest_service.save_asset_image,
            asset.manifest_id,
            asset_id,
            content,
            file.filename or "upload.png",
        )

        # Update asset reference_image_url to HTTP-serveable path
        asset.reference_image_url = f"/api/assets/{asset_id}/image"
        await session.commit()
        await session.refresh(asset)

        return _asset_to_response(asset)


@router.get("/assets/{asset_id}/image")
async def get_asset_image(asset_id: uuid.UUID):
    """Serve an asset's uploaded image by asset ID."""
    async with async_session() as session:
        asset = await manifest_service.get_asset(session, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        # Find file in uploads or crops directory
        manifest_dir = Path("tmp/manifests") / str(asset.manifest_id)
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
        suffix = file_path.suffix.lower()
        media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        media_type = media_types.get(suffix, "image/png")

        return FileResponse(path=str(file_path), media_type=media_type)


# ============================================================================
# Phase 5: Manifesting Engine Endpoints
# ============================================================================

@router.post("/manifests/{manifest_id}/process", status_code=202)
async def process_manifest(manifest_id: uuid.UUID):
    """Trigger manifesting pipeline for a manifest.

    Returns 202 Accepted immediately and runs processing in background.
    """
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found")

        # Verify status allows processing (DRAFT, READY, or ERROR for retry)
        if manifest.status not in ("DRAFT", "READY", "ERROR"):
            raise HTTPException(
                status_code=422,
                detail=f"Cannot process manifest in status {manifest.status}. Must be DRAFT, READY, or ERROR."
            )

        # Set status to PROCESSING
        manifest.status = "PROCESSING"
        await session.commit()

    # Start background task
    asyncio.create_task(process_manifest_task(str(manifest_id)))

    return {
        "task_id": f"manifest_{manifest_id}",
        "status": "started",
        "manifest_id": str(manifest_id),
    }


@router.get("/manifests/{manifest_id}/progress", response_model=ProcessingProgressResponse)
async def get_manifest_progress(manifest_id: uuid.UUID):
    """Get processing progress for a manifest."""
    task_id = f"manifest_{manifest_id}"

    # Check in-memory task status
    if task_id in TASK_STATUS:
        return ProcessingProgressResponse(**TASK_STATUS[task_id])

    # If not in memory, check database status
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found")

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


@router.post("/manifests/{manifest_id}/upload-video", status_code=202, response_model=VideoUploadResponse)
async def upload_video_for_manifest(
    manifest_id: uuid.UUID,
    file: UploadFile = File(...),
):
    """Upload a video file to extract frames for manifest creation.

    Accepts MP4, MOV, WebM up to 200MB. Returns 202 and runs extraction
    in background.
    """
    from vidpipe.config import settings

    max_size = settings.cv_analysis.max_video_file_size_mb * 1024 * 1024

    # Validate manifest exists and is DRAFT
    async with async_session() as session:
        manifest = await manifest_service.get_manifest(session, manifest_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found")
        if manifest.status not in ("DRAFT", "READY", "ERROR"):
            raise HTTPException(
                status_code=422,
                detail=f"Cannot upload video to manifest in status {manifest.status}. Must be DRAFT, READY, or ERROR."
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

    # Update manifest status
    async with async_session() as session:
        manifest = await session.get(Manifest, manifest_id)
        if manifest:
            manifest.status = "EXTRACTING"
            manifest.source_video_path = str(video_path)
            await session.commit()

    # Spawn background task
    asyncio.create_task(extract_video_frames_task(str(manifest_id), str(video_path)))

    return VideoUploadResponse(
        task_id=f"extract_{manifest_id}",
        status="started",
        manifest_id=str(manifest_id),
    )


@router.get("/manifests/{manifest_id}/extraction-progress", response_model=ProcessingProgressResponse)
async def get_extraction_progress(manifest_id: uuid.UUID):
    """Get video frame extraction progress for a manifest."""
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
            raise HTTPException(status_code=404, detail="Manifest not found")

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

@router.get("/projects/{project_id}/scenes/{scene_idx}/candidates")
async def list_candidates(project_id: str, scene_idx: int):
    """List all generation candidates for a specific scene with scores."""
    async with async_session() as session:
        result = await session.execute(
            select(GenerationCandidate)
            .where(
                GenerationCandidate.project_id == uuid.UUID(project_id),
                GenerationCandidate.scene_index == scene_idx,
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


@router.put("/projects/{project_id}/scenes/{scene_idx}/candidates/{candidate_id}/select")
async def select_candidate(project_id: str, scene_idx: int, candidate_id: str):
    """Manually override auto-selection for a scene's candidate.

    CRITICAL: Updates BOTH GenerationCandidate.is_selected AND VideoClip.local_path.
    The stitcher reads VideoClip.local_path, so both must be consistent.
    """
    async with async_session() as session:
        # Load all candidates for this scene
        all_result = await session.execute(
            select(GenerationCandidate).where(
                GenerationCandidate.project_id == uuid.UUID(project_id),
                GenerationCandidate.scene_index == scene_idx,
            )
        )
        all_candidates = all_result.scalars().all()
        if not all_candidates:
            raise HTTPException(404, "No candidates found for this scene")

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
        scene_result = await session.execute(
            select(Scene).where(
                Scene.project_id == uuid.UUID(project_id),
                Scene.scene_index == scene_idx,
            )
        )
        scene = scene_result.scalar_one_or_none()
        if scene:
            clip_result = await session.execute(
                select(VideoClip).where(VideoClip.scene_id == scene.id)
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
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    has_api_key: bool = False
    comfyui_host: Optional[str] = None
    has_comfyui_key: bool = False
    comfyui_cost_per_second: Optional[float] = None
    # Phase 13: Ollama configuration
    ollama_use_cloud: bool = False
    has_ollama_key: bool = False
    ollama_endpoint: Optional[str] = None
    ollama_models: Optional[list] = None


class UserSettingsUpdate(BaseModel):
    """Request schema for PUT /api/settings."""
    enabled_text_models: Optional[list[str]] = None
    enabled_image_models: Optional[list[str]] = None
    enabled_video_models: Optional[list[str]] = None
    default_text_model: Optional[str] = None
    default_image_model: Optional[str] = None
    default_video_model: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None
    vertex_api_key: Optional[str] = None
    clear_api_key: bool = False
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


class EnabledModelsResponse(BaseModel):
    """Lightweight response for GenerateForm model filtering."""
    enabled_text_models: Optional[list[str]] = None
    enabled_image_models: Optional[list[str]] = None
    enabled_video_models: Optional[list[str]] = None
    default_text_model: Optional[str] = None
    default_image_model: Optional[str] = None
    default_video_model: Optional[str] = None
    comfyui_cost_per_second: Optional[float] = None
    # Phase 13: Ollama models for dynamic model list
    ollama_models: Optional[list] = None


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
            gcp_project_id=settings.gcp_project_id,
            gcp_location=settings.gcp_location,
            has_api_key=bool(settings.vertex_api_key),
            comfyui_host=settings.comfyui_host,
            has_comfyui_key=bool(settings.comfyui_api_key),
            comfyui_cost_per_second=settings.comfyui_cost_per_second,
            ollama_use_cloud=bool(settings.ollama_use_cloud),
            has_ollama_key=bool(settings.ollama_api_key),
            ollama_endpoint=settings.ollama_endpoint,
            ollama_models=settings.ollama_models,
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
        raise HTTPException(400, f"Invalid default text model: {body.default_text_model}")
    if body.default_image_model and body.default_image_model not in ALLOWED_IMAGE_MODELS:
        raise HTTPException(400, f"Invalid default image model: {body.default_image_model}")
    if body.default_video_model and body.default_video_model not in ALLOWED_VIDEO_MODELS:
        raise HTTPException(400, f"Invalid default video model: {body.default_video_model}")

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
        settings.gcp_project_id = body.gcp_project_id
        settings.gcp_location = body.gcp_location

        # Only overwrite API key if explicitly provided (non-empty string)
        if body.clear_api_key:
            settings.vertex_api_key = None
        elif body.vertex_api_key:
            settings.vertex_api_key = body.vertex_api_key

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

        await session.commit()
        await session.refresh(settings)

        return UserSettingsResponse(
            enabled_text_models=settings.enabled_text_models,
            enabled_image_models=settings.enabled_image_models,
            enabled_video_models=settings.enabled_video_models,
            default_text_model=settings.default_text_model,
            default_image_model=settings.default_image_model,
            default_video_model=settings.default_video_model,
            gcp_project_id=settings.gcp_project_id,
            gcp_location=settings.gcp_location,
            has_api_key=bool(settings.vertex_api_key),
            comfyui_host=settings.comfyui_host,
            has_comfyui_key=bool(settings.comfyui_api_key),
            comfyui_cost_per_second=settings.comfyui_cost_per_second,
            ollama_use_cloud=bool(settings.ollama_use_cloud),
            has_ollama_key=bool(settings.ollama_api_key),
            ollama_endpoint=settings.ollama_endpoint,
            ollama_models=settings.ollama_models,
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
            comfyui_cost_per_second=settings.comfyui_cost_per_second,
            ollama_models=settings.ollama_models,
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


@router.get("/projects/{project_id}/checkpoints", response_model=list[CheckpointListItem])
async def list_checkpoints(project_id: uuid.UUID):
    """List all checkpoints for a project, newest first."""
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Project not found")

        cp_result = await session.execute(
            select(ProjectCheckpoint)
            .where(ProjectCheckpoint.project_id == project_id)
            .order_by(ProjectCheckpoint.created_at.desc())
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


@router.get("/projects/{project_id}/checkpoints/{sha}", response_model=CheckpointDetail)
async def get_checkpoint(project_id: uuid.UUID, sha: str):
    """Get checkpoint detail including snapshot."""
    async with async_session() as session:
        result = await session.execute(
            select(ProjectCheckpoint).where(
                ProjectCheckpoint.project_id == project_id,
                ProjectCheckpoint.sha == sha,
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


@router.get("/projects/{project_id}/checkpoints/{sha}/diff", response_model=CheckpointDiff)
async def get_checkpoint_diff(project_id: uuid.UUID, sha: str):
    """Get structured diff for a checkpoint."""
    async with async_session() as session:
        result = await session.execute(
            select(ProjectCheckpoint).where(
                ProjectCheckpoint.project_id == project_id,
                ProjectCheckpoint.sha == sha,
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
                select(ProjectCheckpoint).where(
                    ProjectCheckpoint.project_id == project_id,
                    ProjectCheckpoint.sha == cp.parent_sha,
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


@router.post("/projects/{project_id}/checkpoints")
async def create_manual_checkpoint(project_id: uuid.UUID):
    """Create a manual checkpoint of current state."""
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        from vidpipe.services.checkpoint_service import create_checkpoint
        cp = await create_checkpoint(session, project, "Manual checkpoint")
        await session.commit()

        return {"sha": cp.sha, "message": cp.message}


@router.delete("/projects/{project_id}/checkpoints/{sha}")
async def delete_checkpoint(project_id: uuid.UUID, sha: str):
    """Delete a checkpoint. Splices the chain (updates child's parent_sha)."""
    async with async_session() as session:
        result = await session.execute(
            select(ProjectCheckpoint).where(
                ProjectCheckpoint.project_id == project_id,
                ProjectCheckpoint.sha == sha,
            )
        )
        cp = result.scalar_one_or_none()
        if not cp:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # Don't allow deleting the head checkpoint
        proj_result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = proj_result.scalar_one_or_none()
        if project and project.head_sha == sha:
            raise HTTPException(status_code=400, detail="Cannot delete the current head checkpoint")

        # Splice: find child that points to this checkpoint, update its parent_sha
        child_result = await session.execute(
            select(ProjectCheckpoint).where(
                ProjectCheckpoint.project_id == project_id,
                ProjectCheckpoint.parent_sha == sha,
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

@router.post("/projects/{project_id}/revert")
async def revert_to_checkpoint(project_id: uuid.UUID, body: dict):
    """Revert project state to a specific checkpoint.

    Body: { "sha": "abc123..." }
    Creates a forward-commit checkpoint "Revert to {sha[:8]}".
    """
    target_sha = body.get("sha")
    if not target_sha:
        raise HTTPException(status_code=400, detail="Missing 'sha' in request body")

    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        cp_result = await session.execute(
            select(ProjectCheckpoint).where(
                ProjectCheckpoint.project_id == project_id,
                ProjectCheckpoint.sha == target_sha,
            )
        )
        target_cp = cp_result.scalar_one_or_none()
        if not target_cp:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        from vidpipe.services.checkpoint_service import restore_from_snapshot, create_checkpoint
        await restore_from_snapshot(session, project, target_cp.snapshot_data)

        revert_cp = await create_checkpoint(
            session, project,
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
# PipeSVN: Scene Regeneration
# ============================================================================

class RegenerateSceneRequest(BaseModel):
    targets: list[str]  # ["start_keyframe", "end_keyframe", "video_clip"]
    prompt_overrides: Optional[dict[str, str]] = None
    skip_checkpoint: bool = False
    video_model: Optional[str] = None
    image_model: Optional[str] = None
    scene_edits: Optional[dict[str, str]] = None


class RegenerateTextRequest(BaseModel):
    field: str  # "scene_description" | "start_frame_prompt" | "end_frame_prompt" | "video_motion_prompt" | "transition_notes"
    extra_context: str = ""
    text_model: Optional[str] = None  # override project's text_model (use current edit-mode selection)
    scene_edits: Optional[dict[str, str]] = None


class _RegenTextResult(BaseModel):
    text: str


@router.post("/projects/{project_id}/scenes/{scene_idx}/regenerate")
async def regenerate_scene_assets(
    project_id: uuid.UUID,
    scene_idx: int,
    body: RegenerateSceneRequest,
    background_tasks: BackgroundTasks,
):
    """Regenerate specific assets for a scene (keyframes and/or clip).

    Returns 202 --- regeneration runs in background.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene).where(
                Scene.project_id == project.id,
                Scene.scene_index == scene_idx,
            )
        )
        scene = scene_result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

    # Queue background task
    background_tasks.add_task(
        _run_scene_regeneration, project_id, scene_idx, body.targets, body.prompt_overrides, body.skip_checkpoint,
        video_model_override=body.video_model, image_model_override=body.image_model,
        scene_edits=body.scene_edits,
    )

    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "targets": body.targets,
            "scene_index": scene_idx,
            "head_sha": project.head_sha,
        },
    )


_REGEN_TEXT_VALID_FIELDS = {
    "scene_description",
    "start_frame_prompt",
    "end_frame_prompt",
    "video_motion_prompt",
    "transition_notes",
}

_REGEN_FIELD_INSTRUCTIONS = {
    "scene_description": (
        "Write a vivid narrative scene description capturing the key visual story beat. "
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
        "Write brief transition/continuity notes describing how this scene connects "
        "visually to its neighbors. Focus on visual elements that carry over: "
        "character position, lighting direction, color temperature, camera angle."
    ),
}


@router.post("/projects/{project_id}/scenes/{scene_idx}/regenerate-text")
async def regenerate_scene_text(
    project_id: uuid.UUID,
    scene_idx: int,
    body: RegenerateTextRequest,
):
    """Regenerate a single text field for a scene via LLM.

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
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene)
            .where(Scene.project_id == project.id)
            .order_by(Scene.scene_index)
        )
        all_scenes = list(scene_result.scalars().all())
        scene = next((s for s in all_scenes if s.scene_index == scene_idx), None)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Build neighbor context
        prev_scene = next((s for s in all_scenes if s.scene_index == scene_idx - 1), None)
        next_scene = next((s for s in all_scenes if s.scene_index == scene_idx + 1), None)

        # Load user settings for LLM adapter (needed for Ollama config)
        us_result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        user_settings = us_result.scalar_one_or_none()

    # Build system prompt
    style = project.style or "cinematic"
    field_instructions = _REGEN_FIELD_INSTRUCTIONS[body.field].replace("{style}", style)
    system_prompt = (
        f"You are a storyboard director specializing in short-form video content.\n"
        f"Visual style: {style}\n"
        f"Aspect ratio: {project.aspect_ratio or '16:9'}\n\n"
        f"TASK: Regenerate the '{body.field}' field for scene {scene_idx + 1}.\n\n"
        f"FORMAT INSTRUCTIONS:\n{field_instructions}\n\n"
        f"Return a JSON object with a single 'text' field containing the regenerated content."
    )

    # Build user prompt with context
    parts = []
    parts.append(f"PROJECT CONCEPT: {project.prompt}")
    if project.style_guide:
        sg = project.style_guide
        if isinstance(sg, dict):
            sg_text = sg.get("description") or json.dumps(sg)
        else:
            sg_text = str(sg)
        parts.append(f"STYLE GUIDE: {sg_text}")

    # Character bible from storyboard_raw
    if project.storyboard_raw and isinstance(project.storyboard_raw, dict):
        characters = project.storyboard_raw.get("characters")
        if characters:
            parts.append(f"CHARACTERS: {json.dumps(characters)}")

    # Current scene context (sibling fields) — prefer in-flight edits over committed DB values
    se = body.scene_edits or {}
    parts.append(f"\nCURRENT SCENE {scene_idx + 1}:")
    parts.append(f"  Description: {se.get('scene_description', scene.scene_description)}")
    parts.append(f"  Start Frame Prompt: {se.get('start_frame_prompt', scene.start_frame_prompt)}")
    parts.append(f"  End Frame Prompt: {se.get('end_frame_prompt', scene.end_frame_prompt)}")
    parts.append(f"  Motion Prompt: {se.get('video_motion_prompt', scene.video_motion_prompt)}")
    tn = se.get('transition_notes', scene.transition_notes)
    if tn:
        parts.append(f"  Transition Notes: {tn}")

    # Neighbor context
    if prev_scene:
        parts.append(f"\nPREVIOUS SCENE {prev_scene.scene_index + 1}: {prev_scene.scene_description}")
    if next_scene:
        parts.append(f"\nNEXT SCENE {next_scene.scene_index + 1}: {next_scene.scene_description}")

    # Extra context from user
    if body.extra_context and body.extra_context.strip():
        parts.append(f"\n[Additional direction: {body.extra_context.strip()}]")

    user_prompt = "\n".join(parts)

    # Call LLM — prefer request override (edit-mode selection) over saved project model
    from vidpipe.services.llm import get_adapter
    model_id = body.text_model or project.text_model or "gemini-2.5-flash"
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
        logger.error("Text regen failed for scene %d field %s: %s", scene_idx, body.field, e)
        raise HTTPException(status_code=500, detail=f"Text regeneration failed: {e}")


class GenerateSceneFieldsRequest(BaseModel):
    scene_index: int
    all_scene_edits: Optional[dict[int, dict[str, str]]] = None
    text_model: Optional[str] = None


class GenerateSceneFieldsResponse(BaseModel):
    scene_description: str
    start_frame_prompt: str
    end_frame_prompt: str
    video_motion_prompt: str
    transition_notes: str


@router.post("/projects/{project_id}/generate-scene-fields")
async def generate_scene_fields(
    project_id: uuid.UUID,
    body: GenerateSceneFieldsRequest,
):
    """Generate all 5 text fields for a new/empty scene via a single LLM call.

    Returns the generated fields directly (synchronous — no background task).
    The text is NOT saved to DB; the frontend applies fields as edits.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene)
            .where(Scene.project_id == project.id)
            .order_by(Scene.scene_index)
        )
        all_scenes = list(scene_result.scalars().all())

        # Load user settings for LLM adapter (needed for Ollama config)
        us_result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        user_settings = us_result.scalar_one_or_none()

    # Helper: get effective field value considering in-flight edits
    edits_map = body.all_scene_edits or {}

    def scene_field(scene_obj, field_db: str, scene_idx: int) -> str:
        """Return in-flight edit if present, else DB value."""
        idx_edits = edits_map.get(scene_idx, {})
        if field_db in idx_edits:
            return idx_edits[field_db]
        if scene_obj is None:
            return ""
        return getattr(scene_obj, field_db, "") or ""

    # Build neighbor context
    prev_db = next((s for s in all_scenes if s.scene_index == body.scene_index - 1), None)
    next_db = next((s for s in all_scenes if s.scene_index == body.scene_index + 1), None)
    prev_idx = body.scene_index - 1
    next_idx = body.scene_index + 1

    style = project.style or "cinematic"

    # Build system prompt
    field_blocks = []
    for field_name, instructions in _REGEN_FIELD_INSTRUCTIONS.items():
        field_blocks.append(f"  {field_name}: {instructions.replace('{style}', style)}")
    all_field_instructions = "\n".join(field_blocks)

    system_prompt = (
        f"You are a storyboard director specializing in short-form video content.\n"
        f"Visual style: {style}\n"
        f"Aspect ratio: {project.aspect_ratio or '16:9'}\n\n"
        f"TASK: Generate ALL 5 text fields for a NEW scene at position {body.scene_index + 1}.\n\n"
        f"FIELD INSTRUCTIONS:\n{all_field_instructions}\n\n"
        f"Return a JSON object with exactly these 5 fields:\n"
        f"  scene_description, start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes"
    )

    # Build user prompt with context
    parts = []
    parts.append(f"PROJECT CONCEPT: {project.prompt}")
    if project.style_guide:
        sg = project.style_guide
        if isinstance(sg, dict):
            sg_text = sg.get("description") or json.dumps(sg)
        else:
            sg_text = str(sg)
        parts.append(f"STYLE GUIDE: {sg_text}")

    # Character bible from storyboard_raw
    if project.storyboard_raw and isinstance(project.storyboard_raw, dict):
        characters = project.storyboard_raw.get("characters")
        if characters:
            parts.append(f"CHARACTERS: {json.dumps(characters)}")

    # Previous scene context (from DB or in-flight edits for synthetic scenes)
    has_prev = prev_db is not None or prev_idx in edits_map
    if has_prev:
        parts.append(f"\nPREVIOUS SCENE {prev_idx + 1}:")
        parts.append(f"  Description: {scene_field(prev_db, 'scene_description', prev_idx)}")
        parts.append(f"  Start Frame: {scene_field(prev_db, 'start_frame_prompt', prev_idx)}")
        parts.append(f"  End Frame: {scene_field(prev_db, 'end_frame_prompt', prev_idx)}")
        parts.append(f"  Motion: {scene_field(prev_db, 'video_motion_prompt', prev_idx)}")

    # Next scene context
    has_next = next_db is not None or next_idx in edits_map
    if has_next:
        parts.append(f"\nNEXT SCENE {next_idx + 1}:")
        parts.append(f"  Description: {scene_field(next_db, 'scene_description', next_idx)}")
        parts.append(f"  Start Frame: {scene_field(next_db, 'start_frame_prompt', next_idx)}")
        parts.append(f"  End Frame: {scene_field(next_db, 'end_frame_prompt', next_idx)}")
        parts.append(f"  Motion: {scene_field(next_db, 'video_motion_prompt', next_idx)}")

    # Any existing edits the user already typed for this scene
    current_edits = edits_map.get(body.scene_index, {})
    if current_edits:
        parts.append(f"\nUSER'S PARTIAL INPUT FOR THIS SCENE (incorporate and expand on these):")
        for k, v in current_edits.items():
            if v.strip():
                parts.append(f"  {k}: {v}")

    user_prompt = "\n".join(parts)

    # Call LLM
    from vidpipe.services.llm import get_adapter
    model_id = body.text_model or project.text_model or "gemini-2.5-flash"
    adapter = get_adapter(model_id, user_settings=user_settings)

    try:
        result = await adapter.generate_text(
            prompt=user_prompt,
            schema=GenerateSceneFieldsResponse,
            temperature=0.7,
            system_prompt=system_prompt,
        )
        return result.model_dump()
    except Exception as e:
        logger.error("Generate scene fields failed for project %s scene %d: %s", project_id, body.scene_index, e)
        raise HTTPException(status_code=500, detail=f"Scene field generation failed: {e}")


class GenerateNewSceneRequest(BaseModel):
    scene_index: int
    all_scene_edits: Optional[dict[int, dict[str, str]]] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None


class GenerateNewSceneResponse(BaseModel):
    scene_index: int
    scene_description: str
    start_frame_prompt: str
    end_frame_prompt: str
    video_motion_prompt: str
    transition_notes: str
    head_sha: Optional[str] = None


@router.post("/projects/{project_id}/generate-new-scene")
async def generate_new_scene(
    project_id: uuid.UUID,
    body: GenerateNewSceneRequest,
    background_tasks: BackgroundTasks,
):
    """Generate a complete new scene: text fields (sync) + keyframes & clip (background).

    Phase 1 (synchronous): Create Scene DB row with LLM-generated text fields.
    Phase 2 (background): Generate start KF, end KF, and video clip via existing regen infra.

    Returns 202 with the generated text fields and head_sha for revert-on-cancel.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Check no existing scene at requested index
        existing_result = await session.execute(
            select(Scene).where(
                Scene.project_id == project.id,
                Scene.scene_index == body.scene_index,
            )
        )
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"Scene already exists at index {body.scene_index}")

        # Load all scenes for neighbor context
        scene_result = await session.execute(
            select(Scene)
            .where(Scene.project_id == project.id)
            .order_by(Scene.scene_index)
        )
        all_scenes = list(scene_result.scalars().all())

        # Load user settings for LLM adapter
        us_result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        user_settings = us_result.scalar_one_or_none()

        # --- Phase 1: Generate text fields via LLM (same logic as generate_scene_fields) ---
        edits_map = body.all_scene_edits or {}

        def scene_field(scene_obj, field_db: str, scene_idx: int) -> str:
            idx_edits = edits_map.get(scene_idx, {})
            if field_db in idx_edits:
                return idx_edits[field_db]
            if scene_obj is None:
                return ""
            return getattr(scene_obj, field_db, "") or ""

        prev_db = next((s for s in all_scenes if s.scene_index == body.scene_index - 1), None)
        next_db = next((s for s in all_scenes if s.scene_index == body.scene_index + 1), None)
        prev_idx = body.scene_index - 1
        next_idx = body.scene_index + 1

        style = project.style or "cinematic"

        field_blocks = []
        for field_name, instructions in _REGEN_FIELD_INSTRUCTIONS.items():
            field_blocks.append(f"  {field_name}: {instructions.replace('{style}', style)}")
        all_field_instructions = "\n".join(field_blocks)

        system_prompt = (
            f"You are a storyboard director specializing in short-form video content.\n"
            f"Visual style: {style}\n"
            f"Aspect ratio: {project.aspect_ratio or '16:9'}\n\n"
            f"TASK: Generate ALL 5 text fields for a NEW scene at position {body.scene_index + 1}.\n\n"
            f"FIELD INSTRUCTIONS:\n{all_field_instructions}\n\n"
            f"Return a JSON object with exactly these 5 fields:\n"
            f"  scene_description, start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes"
        )

        parts = []
        parts.append(f"PROJECT CONCEPT: {project.prompt}")
        if project.style_guide:
            sg = project.style_guide
            if isinstance(sg, dict):
                sg_text = sg.get("description") or json.dumps(sg)
            else:
                sg_text = str(sg)
            parts.append(f"STYLE GUIDE: {sg_text}")

        if project.storyboard_raw and isinstance(project.storyboard_raw, dict):
            characters = project.storyboard_raw.get("characters")
            if characters:
                parts.append(f"CHARACTERS: {json.dumps(characters)}")

        has_prev = prev_db is not None or prev_idx in edits_map
        if has_prev:
            parts.append(f"\nPREVIOUS SCENE {prev_idx + 1}:")
            parts.append(f"  Description: {scene_field(prev_db, 'scene_description', prev_idx)}")
            parts.append(f"  Start Frame: {scene_field(prev_db, 'start_frame_prompt', prev_idx)}")
            parts.append(f"  End Frame: {scene_field(prev_db, 'end_frame_prompt', prev_idx)}")
            parts.append(f"  Motion: {scene_field(prev_db, 'video_motion_prompt', prev_idx)}")

        has_next = next_db is not None or next_idx in edits_map
        if has_next:
            parts.append(f"\nNEXT SCENE {next_idx + 1}:")
            parts.append(f"  Description: {scene_field(next_db, 'scene_description', next_idx)}")
            parts.append(f"  Start Frame: {scene_field(next_db, 'start_frame_prompt', next_idx)}")
            parts.append(f"  End Frame: {scene_field(next_db, 'end_frame_prompt', next_idx)}")
            parts.append(f"  Motion: {scene_field(next_db, 'video_motion_prompt', next_idx)}")

        current_edits = edits_map.get(body.scene_index, {})
        if current_edits:
            parts.append(f"\nUSER'S PARTIAL INPUT FOR THIS SCENE (incorporate and expand on these):")
            for k, v in current_edits.items():
                if v.strip():
                    parts.append(f"  {k}: {v}")

        user_prompt = "\n".join(parts)

        from vidpipe.services.llm import get_adapter
        model_id = body.text_model or project.text_model or "gemini-2.5-flash"
        adapter = get_adapter(model_id, user_settings=user_settings)

        try:
            text_result = await adapter.generate_text(
                prompt=user_prompt,
                schema=GenerateSceneFieldsResponse,
                temperature=0.7,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error("Generate new scene text failed for project %s scene %d: %s", project_id, body.scene_index, e)
            raise HTTPException(status_code=500, detail=f"Scene text generation failed: {e}")

        # --- Create Scene DB row ---
        new_scene = Scene(
            project_id=project.id,
            scene_index=body.scene_index,
            scene_description=text_result.scene_description,
            start_frame_prompt=text_result.start_frame_prompt,
            end_frame_prompt=text_result.end_frame_prompt,
            video_motion_prompt=text_result.video_motion_prompt,
            transition_notes=text_result.transition_notes,
            status="pending",
        )
        session.add(new_scene)

        # Update target_scene_count if needed
        project.target_scene_count = max(project.target_scene_count, body.scene_index + 1)

        # Ensure baseline checkpoint exists
        if not project.head_sha:
            from vidpipe.services.checkpoint_service import create_checkpoint
            await create_checkpoint(
                session, project,
                "Auto-save: edit baseline",
                metadata={"auto_baseline": True},
            )
            await session.commit()

        # Create checkpoint for the new scene
        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, project,
            f"Generated new scene {body.scene_index + 1}",
            metadata={"generated_scene": body.scene_index},
        )
        await session.commit()

        head_sha = project.head_sha

    # --- Phase 2: Queue background asset generation ---
    background_tasks.add_task(
        _run_scene_regeneration,
        project_id,
        body.scene_index,
        ["start_keyframe", "end_keyframe", "video_clip"],
        None,  # no prompt overrides
        True,  # skip_checkpoint (we already created one above)
        video_model_override=body.video_model,
        image_model_override=body.image_model,
    )

    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content=GenerateNewSceneResponse(
            scene_index=body.scene_index,
            scene_description=text_result.scene_description,
            start_frame_prompt=text_result.start_frame_prompt,
            end_frame_prompt=text_result.end_frame_prompt,
            video_motion_prompt=text_result.video_motion_prompt,
            transition_notes=text_result.transition_notes,
            head_sha=head_sha,
        ).model_dump(),
    )


async def _run_scene_regeneration(
    project_id: uuid.UUID,
    scene_idx: int,
    targets: list[str],
    prompt_overrides: Optional[dict[str, str]],
    skip_checkpoint: bool = False,
    video_model_override: Optional[str] = None,
    image_model_override: Optional[str] = None,
    scene_edits: Optional[dict[str, str]] = None,
):
    """Background task for scene regeneration."""
    logger.info("Regenerating scene %d for project %s: %s", scene_idx, project_id, targets)

    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return

        scene_result = await session.execute(
            select(Scene).where(Scene.project_id == project.id, Scene.scene_index == scene_idx)
        )
        scene = scene_result.scalar_one_or_none()
        if not scene:
            return

        # When skip_checkpoint is set (edit mode), ensure a baseline checkpoint
        # exists so the user can revert on cancel.
        if skip_checkpoint and not project.head_sha:
            from vidpipe.services.checkpoint_service import create_checkpoint
            await create_checkpoint(
                session, project,
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
                        session, project, scene, position, file_mgr,
                        prompt_override=prompt_overrides.get(target) if prompt_overrides else None,
                        image_model_override=image_model_override,
                        scene_edits=scene_edits,
                    )
                    regenerated.append(target)

                    # KEYF-03 cascade: end keyframe change propagates to next scene's start
                    if position == "end":
                        next_scene_result = await session.execute(
                            select(Scene).where(
                                Scene.project_id == project.id,
                                Scene.scene_index == scene_idx + 1,
                            )
                        )
                        next_scene = next_scene_result.scalar_one_or_none()
                        if next_scene:
                            # Read the newly regenerated end keyframe bytes
                            new_end_kf_result = await session.execute(
                                select(Keyframe).where(
                                    Keyframe.scene_id == scene.id,
                                    Keyframe.position == "end",
                                )
                            )
                            new_end_kf = new_end_kf_result.scalar_one_or_none()
                            if new_end_kf:
                                from pathlib import Path
                                end_bytes = Path(new_end_kf.file_path).read_bytes()
                                inherited_path = file_mgr.save_keyframe_versioned(
                                    project.id, scene_idx + 1, "start", end_bytes,
                                )
                                # Replace next scene's start keyframe
                                old_next_start_result = await session.execute(
                                    select(Keyframe).where(
                                        Keyframe.scene_id == next_scene.id,
                                        Keyframe.position == "start",
                                    )
                                )
                                old_next_start = old_next_start_result.scalar_one_or_none()
                                if old_next_start:
                                    await session.delete(old_next_start)
                                    await session.flush()
                                inherited_kf = Keyframe(
                                    scene_id=next_scene.id,
                                    position="start",
                                    prompt_used=next_scene.start_frame_prompt,
                                    file_path=str(inherited_path),
                                    mime_type="image/png",
                                    source="inherited",
                                )
                                session.add(inherited_kf)
                                await session.flush()
                                logger.info(
                                    "Cascaded end keyframe from scene %d to start of scene %d",
                                    scene_idx, scene_idx + 1,
                                )
                except Exception as e:
                    logger.error("Failed to regenerate %s for scene %d: %s", target, scene_idx, e)

            elif target == "video_clip":
                try:
                    await _regenerate_clip(
                        session, project, scene, file_mgr,
                        prompt_override=prompt_overrides.get("video_clip") if prompt_overrides else None,
                        video_model_override=video_model_override,
                        scene_edits=scene_edits,
                    )
                    regenerated.append(target)
                except Exception as e:
                    logger.error("Failed to regenerate clip for scene %d: %s", scene_idx, e)

        # Persist changes and optionally create checkpoint
        if regenerated:
            if not skip_checkpoint:
                from vidpipe.services.checkpoint_service import create_checkpoint
                await create_checkpoint(
                    session, project,
                    f"Regenerated {', '.join(regenerated)} for scene {scene_idx + 1}",
                    metadata={"regenerated": regenerated, "scene_index": scene_idx},
                )
            await session.commit()


async def _regenerate_keyframe(session, project, scene, position, file_mgr, prompt_override=None, image_model_override=None, scene_edits=None):
    """Regenerate a single keyframe following the same methodology as the pipeline.

    - Start KF scene 0: text-to-image with style/character enrichment + ref images
    - Start KF scene N>0: image-conditioned from previous scene's end keyframe
    - End KF: image-conditioned from this scene's start keyframe with progression prompt
    """
    from pathlib import Path
    from vidpipe.config import settings as app_settings
    from vidpipe.db.models import SceneManifest as SceneManifestModel

    image_model = image_model_override or project.image_model or app_settings.models.keyframe_image
    # Guard: Imagen models no longer supported
    if image_model.startswith("imagen-"):
        image_model = app_settings.models.image_gen

    # Build character bible prefix from storyboard data
    character_prefix = ""
    if project.storyboard_raw and "characters" in project.storyboard_raw:
        char_lines = []
        for ch in project.storyboard_raw["characters"]:
            char_lines.append(
                f"{ch.get('name', 'Character')}: {ch.get('physical_description', '')}. "
                f"Wearing {ch.get('clothing_description', '')}."
            )
        if char_lines:
            character_prefix = "Characters: " + " ".join(char_lines) + " "

    # Build style prefix from style guide
    style_guide = project.style_guide or {}
    style_prefix = ""
    if style_guide:
        parts = []
        if style_guide.get("visual_style"):
            parts.append(f"Style: {style_guide['visual_style']}")
        if style_guide.get("color_palette"):
            parts.append(f"Palette: {style_guide['color_palette']}")
        if parts:
            style_prefix = ". ".join(parts) + ". "

    # Load scene manifest for rewritten prompt + reference images
    sm_result = await session.execute(
        select(SceneManifestModel).where(
            SceneManifestModel.project_id == project.id,
            SceneManifestModel.scene_index == scene.scene_index,
        )
    )
    sm = sm_result.scalar_one_or_none()
    rewritten_prompt = sm.rewritten_keyframe_prompt if sm else None

    # Resolve asset reference images (for manifest projects)
    ref_image_bytes_list: list[bytes] = []
    if project.manifest_id and sm and sm.selected_reference_tags:
        try:
            from vidpipe.services import manifest_service
            from vidpipe.services.reference_selection import resolve_asset_image_bytes
            all_assets = await manifest_service.load_manifest_assets(session, project.manifest_id)
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
        COMFYUI_IMAGE_MODELS, _generate_image_comfyui,
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
    edited_start = (scene_edits or {}).get("start_frame_prompt")
    edited_end = (scene_edits or {}).get("end_frame_prompt")

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
            base_prompt = scene.start_frame_prompt

        if scene.scene_index == 0:
            # Scene 0: text-to-image with style/character enrichment
            enriched_prompt = f"{style_prefix}{character_prefix}{base_prompt}" if not prompt_override else base_prompt
            if is_comfyui:
                image_bytes = await _generate_image_comfyui(comfy_client, enriched_prompt, seed=project.seed)
            else:
                image_bytes = await _generate_image_from_text(
                    image_client, enriched_prompt, project.aspect_ratio, image_model,
                    seed=project.seed,
                    reference_images=ref_image_bytes_list or None,
                )
        else:
            # Scene N>0: inherit from previous scene's end keyframe (KEYF-03)
            # Pipeline copies end→start verbatim for continuity.
            # With prompt_override (extra direction), use conditioned generation instead.
            prev_scene_result = await session.execute(
                select(Scene).where(
                    Scene.project_id == project.id,
                    Scene.scene_index == scene.scene_index - 1,
                )
            )
            prev_scene = prev_scene_result.scalar_one_or_none()
            prev_end_kf = None
            if prev_scene:
                prev_kf_result = await session.execute(
                    select(Keyframe).where(
                        Keyframe.scene_id == prev_scene.id,
                        Keyframe.position == "end",
                    )
                )
                prev_end_kf = prev_kf_result.scalar_one_or_none()

            if prev_end_kf and Path(prev_end_kf.file_path).exists():
                prev_end_bytes = Path(prev_end_kf.file_path).read_bytes()

                if prompt_override:
                    # Extra direction provided — conditioned generation from prev end frame
                    style_label = project.style.replace("_", " ")
                    conditioning_prompt = (
                        f"Generate the NEXT keyframe continuing from the previous scene into this new scene. "
                        f"Style: {style_label}.\n\n"
                        f"TARGET START STATE (what the new image must depict):\n"
                        f"{prompt_override}\n\n"
                        f"CONSISTENCY CONSTRAINTS:\n"
                        f"- Same character appearance (face, hair, clothing, proportions)\n"
                        f"- Same {style_label} rendering style\n"
                        f"{character_prefix}"
                    )
                    if is_comfyui:
                        image_bytes = await _generate_image_comfyui(comfy_client, conditioning_prompt, seed=project.seed)
                    else:
                        image_bytes = await _generate_image_conditioned(
                            image_client, prev_end_bytes, conditioning_prompt,
                            project.aspect_ratio, image_model,
                            reference_images=ref_image_bytes_list or None,
                        )
                else:
                    # No extra direction — inherit verbatim (same as pipeline KEYF-03)
                    image_bytes = prev_end_bytes
            else:
                # Fallback: no previous end keyframe available, use text-to-image
                enriched_prompt = f"{style_prefix}{character_prefix}{base_prompt}" if not prompt_override else base_prompt
                if is_comfyui:
                    image_bytes = await _generate_image_comfyui(comfy_client, enriched_prompt, seed=project.seed)
                else:
                    image_bytes = await _generate_image_from_text(
                        image_client, enriched_prompt, project.aspect_ratio, image_model,
                        seed=project.seed,
                        reference_images=ref_image_bytes_list or None,
                    )

        prompt_used = prompt_override or edited_start or rewritten_prompt or scene.start_frame_prompt
    else:
        # --- END KEYFRAME ---
        # Always image-conditioned from this scene's start keyframe
        start_kf_result = await session.execute(
            select(Keyframe).where(
                Keyframe.scene_id == scene.id,
                Keyframe.position == "start",
            )
        )
        start_kf = start_kf_result.scalar_one_or_none()

        if not start_kf or not Path(start_kf.file_path).exists():
            raise ValueError(f"No start keyframe available for scene {scene.scene_index} — generate start keyframe first")

        conditioning_bytes = Path(start_kf.file_path).read_bytes()

        if prompt_override:
            conditioning_prompt = prompt_override
        else:
            end_prompt = edited_end or scene.end_frame_prompt
            style_label = project.style.replace("_", " ")
            conditioning_prompt = (
                f"Generate the NEXT keyframe for this {style_label} scene, "
                f"showing clear visual progression {project.target_clip_duration} seconds later.\n\n"
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

        if is_comfyui:
            image_bytes = await _generate_image_comfyui(
                comfy_client, conditioning_prompt,
                seed=project.seed + scene.scene_index + 1000,
            )
        else:
            image_bytes = await _generate_image_conditioned(
                image_client, conditioning_bytes, conditioning_prompt,
                project.aspect_ratio, image_model,
                reference_images=ref_image_bytes_list or None,
            )

        prompt_used = prompt_override or edited_end or scene.end_frame_prompt

    # Save with versioned path
    filepath = file_mgr.save_keyframe_versioned(project.id, scene.scene_index, position, image_bytes)

    # Delete old keyframe for this position
    old_kf_result = await session.execute(
        select(Keyframe).where(Keyframe.scene_id == scene.id, Keyframe.position == position)
    )
    old_kf = old_kf_result.scalar_one_or_none()
    if old_kf:
        await session.delete(old_kf)
        await session.flush()

    # Create new keyframe row
    kf = Keyframe(
        scene_id=scene.id,
        position=position,
        prompt_used=prompt_used,
        file_path=str(filepath),
        mime_type="image/png",
        source="generated",
    )
    session.add(kf)
    await session.flush()


async def _regenerate_clip(session, project, scene, file_mgr, prompt_override=None, video_model_override=None, scene_edits=None):
    """Regenerate video clip for a scene."""
    from vidpipe.config import settings as app_settings

    # Determine prompt — priority: prompt_override > edited value > rewritten > committed
    edited_motion = (scene_edits or {}).get("video_motion_prompt")
    if prompt_override:
        video_prompt = prompt_override
    elif edited_motion:
        video_prompt = edited_motion
    else:
        video_prompt = scene.video_motion_prompt
        from vidpipe.db.models import SceneManifest as SceneManifestModel
        sm_result = await session.execute(
            select(SceneManifestModel).where(
                SceneManifestModel.project_id == project.id,
                SceneManifestModel.scene_index == scene.scene_index,
            )
        )
        sm = sm_result.scalar_one_or_none()
        if sm and sm.rewritten_video_prompt:
            video_prompt = sm.rewritten_video_prompt

    # Load keyframes
    kf_result = await session.execute(
        select(Keyframe).where(Keyframe.scene_id == scene.id)
    )
    keyframes = kf_result.scalars().all()
    start_kf = next((k for k in keyframes if k.position == "start"), None)
    end_kf = next((k for k in keyframes if k.position == "end"), None)

    if not start_kf:
        raise ValueError(f"No start keyframe for scene {scene.scene_index}")

    start_bytes = Path(start_kf.file_path).read_bytes()
    end_bytes = Path(end_kf.file_path).read_bytes() if end_kf else start_bytes

    # Submit video generation
    video_model = video_model_override or project.video_model or app_settings.models.video_generator

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

        # Load character reference images (if manifest project)
        char_ref_bytes: list[bytes] = []
        if project.manifest_id:
            from vidpipe.pipeline.video_gen import _load_char_ref_images
            char_ref_bytes = await _load_char_ref_images(session, project)

        operation_id = await adapter.submit(
            video_prompt=video_prompt,
            start_frame_bytes=start_bytes,
            end_frame_bytes=end_bytes if end_kf else None,
            char_ref_bytes=char_ref_bytes,
            aspect_ratio=project.aspect_ratio,
            seed=project.seed or 0,
            scene_index=scene.scene_index,
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
            raise TimeoutError(f"ComfyUI timed out for scene {scene.scene_index}")
    else:
        # ---- Veo / Vertex path ----
        from vidpipe.pipeline.video_gen import _submit_video_job

        from vidpipe.services.vertex_client import get_client
        client = get_client(user_settings=user_settings)

        operation = await _submit_video_job(
            client, video_model, video_prompt,
            start_bytes, end_bytes, project,
        )

        for _ in range(max_polls):
            op = await client.aio.operations.get(operation=operation)
            if op.done:
                break
            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"Video generation timed out for scene {scene.scene_index}")

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
    filepath = file_mgr.save_clip_versioned(project.id, scene.scene_index, video_bytes)

    # Delete old clip
    old_clip_result = await session.execute(
        select(VideoClip).where(VideoClip.scene_id == scene.id)
    )
    old_clip = old_clip_result.scalar_one_or_none()
    if old_clip:
        await session.delete(old_clip)
        await session.flush()

    # Create new clip row
    clip = VideoClip(
        scene_id=scene.id,
        status="complete",
        local_path=str(filepath),
        source="generated",
        prompt_used=video_prompt,
    )
    session.add(clip)
    await session.flush()


# ============================================================================
# PipeSVN: Project-wide Regeneration
# ============================================================================

class RegenerateProjectRequest(BaseModel):
    scope: str  # "stale", "all", "stitch_only"
    scene_indices: Optional[list[int]] = None


@router.post("/projects/{project_id}/regenerate")
async def regenerate_project(
    project_id: uuid.UUID,
    body: RegenerateProjectRequest,
    background_tasks: BackgroundTasks,
):
    """Project-wide regeneration with different scopes.

    Scopes:
    - stale: regenerate only stale assets
    - all: regenerate everything from keyframing
    - stitch_only: re-run stitcher
    """
    async with async_session() as session:
        result = await session.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    if body.scope == "stitch_only":
        background_tasks.add_task(_run_restitch, project_id)
    else:
        background_tasks.add_task(_run_project_regeneration, project_id, body.scope, body.scene_indices)

    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "scope": body.scope},
    )


async def _run_restitch(project_id: uuid.UUID):
    """Background task to re-stitch current clips."""
    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return

        from vidpipe.pipeline.stitcher import stitch_videos
        await stitch_videos(session, project)
        await session.refresh(project)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(session, project, "Re-stitched video")
        await session.commit()


async def _run_project_regeneration(
    project_id: uuid.UUID,
    scope: str,
    scene_indices: Optional[list[int]],
):
    """Background task for project-wide regeneration."""
    logger.info("Project regeneration %s for %s", scope, project_id)

    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return

        scenes_result = await session.execute(
            select(Scene).where(Scene.project_id == project.id).order_by(Scene.scene_index)
        )
        scenes = scenes_result.scalars().all()

        from vidpipe.services.file_manager import FileManager
        from vidpipe.services.checkpoint_service import (
            compute_keyframe_staleness, compute_clip_staleness, create_checkpoint,
        )
        from vidpipe.db.models import SceneManifest as SceneManifestModel
        file_mgr = FileManager()

        regenerated_count = 0
        for scene in scenes:
            if scene_indices and scene.scene_index not in scene_indices:
                continue

            # Load scene manifest
            sm_result = await session.execute(
                select(SceneManifestModel).where(
                    SceneManifestModel.project_id == project.id,
                    SceneManifestModel.scene_index == scene.scene_index,
                )
            )
            sm = sm_result.scalar_one_or_none()

            # Load keyframes
            kf_result = await session.execute(select(Keyframe).where(Keyframe.scene_id == scene.id))
            kfs = kf_result.scalars().all()
            start_kf = next((k for k in kfs if k.position == "start"), None)
            end_kf = next((k for k in kfs if k.position == "end"), None)

            # Load clip
            clip_result = await session.execute(select(VideoClip).where(VideoClip.scene_id == scene.id))
            clip = clip_result.scalar_one_or_none()

            targets = []
            if scope == "all":
                targets = ["start_keyframe", "end_keyframe", "video_clip"]
            elif scope == "stale":
                if compute_keyframe_staleness(scene, start_kf, sm) == "stale":
                    targets.append("start_keyframe")
                if compute_keyframe_staleness(scene, end_kf, sm) == "stale":
                    targets.append("end_keyframe")
                if compute_clip_staleness(scene, clip, sm) == "stale":
                    targets.append("video_clip")

            for target in targets:
                try:
                    if target in ("start_keyframe", "end_keyframe"):
                        position = "start" if target == "start_keyframe" else "end"
                        await _regenerate_keyframe(session, project, scene, position, file_mgr)
                    elif target == "video_clip":
                        await _regenerate_clip(session, project, scene, file_mgr)
                    regenerated_count += 1
                except Exception as e:
                    logger.error("Regeneration failed for scene %d %s: %s", scene.scene_index, target, e)

        if regenerated_count > 0:
            # Auto-stitch after regeneration so the final video stays current
            try:
                from vidpipe.pipeline.stitcher import stitch_videos
                await stitch_videos(session, project)
                await session.refresh(project)
            except Exception as e:
                logger.error("Auto-stitch after regeneration failed: %s", e)

            await create_checkpoint(
                session, project,
                f"Regenerated {regenerated_count} asset(s) ({scope})",
            )
            await session.commit()


# ============================================================================
# PipeSVN: Asset Upload/Replace
# ============================================================================

@router.put("/projects/{project_id}/scenes/{scene_idx}/keyframes/{position}")
async def upload_keyframe(
    project_id: uuid.UUID,
    scene_idx: int,
    position: str,
    file: UploadFile = File(...),
):
    """Upload a keyframe image to replace the generated one."""
    if position not in ("start", "end"):
        raise HTTPException(status_code=400, detail="Position must be 'start' or 'end'")

    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene).where(Scene.project_id == project.id, Scene.scene_index == scene_idx)
        )
        scene = scene_result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        # Read upload data
        data = await file.read()

        from vidpipe.services.file_manager import FileManager
        file_mgr = FileManager()
        filepath = file_mgr.save_keyframe_versioned(project.id, scene_idx, position, data)

        # Delete old keyframe
        old_kf_result = await session.execute(
            select(Keyframe).where(Keyframe.scene_id == scene.id, Keyframe.position == position)
        )
        old_kf = old_kf_result.scalar_one_or_none()
        if old_kf:
            await session.delete(old_kf)
            await session.flush()

        # Create new keyframe row
        kf = Keyframe(
            scene_id=scene.id,
            position=position,
            prompt_used="uploaded",
            file_path=str(filepath),
            mime_type=file.content_type or "image/png",
            source="uploaded",
        )
        session.add(kf)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, project,
            f"Uploaded {position} keyframe for scene {scene_idx + 1}",
        )
        await session.commit()

        return {"status": "uploaded", "file_path": str(filepath), "keyframe_id": str(kf.id)}


@router.put("/projects/{project_id}/scenes/{scene_idx}/clip")
async def upload_clip(
    project_id: uuid.UUID,
    scene_idx: int,
    file: UploadFile = File(...),
):
    """Upload a video clip to replace the generated one."""
    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene).where(Scene.project_id == project.id, Scene.scene_index == scene_idx)
        )
        scene = scene_result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        data = await file.read()

        from vidpipe.services.file_manager import FileManager
        file_mgr = FileManager()
        filepath = file_mgr.save_clip_versioned(project.id, scene_idx, data)

        # Delete old clip
        old_clip_result = await session.execute(
            select(VideoClip).where(VideoClip.scene_id == scene.id)
        )
        old_clip = old_clip_result.scalar_one_or_none()
        if old_clip:
            await session.delete(old_clip)
            await session.flush()

        clip = VideoClip(
            scene_id=scene.id,
            status="complete",
            local_path=str(filepath),
            source="uploaded",
            prompt_used="uploaded",
        )
        session.add(clip)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, project,
            f"Uploaded clip for scene {scene_idx + 1}",
        )
        await session.commit()

        return {"status": "uploaded", "file_path": str(filepath), "clip_id": str(clip.id)}


@router.delete("/projects/{project_id}/scenes/{scene_idx}/clip")
async def delete_clip(project_id: uuid.UUID, scene_idx: int):
    """Delete a scene's video clip (file remains on disk)."""
    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene).where(Scene.project_id == project.id, Scene.scene_index == scene_idx)
        )
        scene = scene_result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        clip_result = await session.execute(
            select(VideoClip).where(VideoClip.scene_id == scene.id)
        )
        clip = clip_result.scalar_one_or_none()
        if not clip:
            raise HTTPException(status_code=404, detail="No clip found")

        await session.delete(clip)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, project,
            f"Removed clip for scene {scene_idx + 1}",
        )
        await session.commit()

        return {"status": "deleted", "scene_index": scene_idx}


@router.delete("/projects/{project_id}/scenes/{scene_idx}/keyframes/{position}")
async def delete_keyframe(project_id: uuid.UUID, scene_idx: int, position: str):
    """Delete a scene's keyframe (file remains on disk)."""
    if position not in ("start", "end"):
        raise HTTPException(status_code=400, detail="Position must be 'start' or 'end'")

    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        scene_result = await session.execute(
            select(Scene).where(Scene.project_id == project.id, Scene.scene_index == scene_idx)
        )
        scene = scene_result.scalar_one_or_none()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")

        kf_result = await session.execute(
            select(Keyframe).where(Keyframe.scene_id == scene.id, Keyframe.position == position)
        )
        kf = kf_result.scalar_one_or_none()
        if not kf:
            raise HTTPException(status_code=404, detail="Keyframe not found")

        await session.delete(kf)

        from vidpipe.services.checkpoint_service import create_checkpoint
        await create_checkpoint(
            session, project,
            f"Removed {position} keyframe for scene {scene_idx + 1}",
        )
        await session.commit()

        return {"status": "deleted", "scene_index": scene_idx, "position": position}
