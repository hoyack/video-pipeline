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
from sqlalchemy import case, func as sa_func, select
from sqlalchemy.orm import selectinload

from google.genai import types
from vidpipe.db import async_session
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip, Manifest, Asset
from vidpipe.orchestrator.pipeline import run_pipeline
from vidpipe.orchestrator.state import can_resume
from vidpipe.schemas.storyboard import SceneSchema
from vidpipe.services.file_manager import FileManager
from vidpipe.services.vertex_client import get_vertex_client, location_for_model
from vidpipe.services import manifest_service

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
    "imagen-3.0-generate-002",
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
}
ALLOWED_VIDEO_MODELS = {
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-fast-generate-001",
}

# Video models that support audio generation
AUDIO_CAPABLE_MODELS = ALLOWED_VIDEO_MODELS - {"veo-2.0-generate-001"}

# Auto-pairing: Imagen models → gemini-2.5-flash-image for conditioned generation.
# Gemini image models → use the same model for conditioning.
IMAGE_CONDITIONED_MAP: dict[str, str] = {
    "imagen-3.0-generate-002": "gemini-2.5-flash-image",
    "imagen-4.0-generate-001": "gemini-2.5-flash-image",
    "imagen-4.0-fast-generate-001": "gemini-2.5-flash-image",
    "imagen-4.0-ultra-generate-001": "gemini-2.5-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
}

# Allowed clip durations per video model
ALLOWED_DURATIONS: dict[str, list[int]] = {
    "veo-2.0-generate-001": [5, 6, 7, 8],
    "veo-3.0-generate-001": [4, 6, 8],
    "veo-3.0-fast-generate-001": [4, 6, 8],
    "veo-3.1-generate-preview": [4, 6, 8],
    "veo-3.1-generate-001": [4, 6, 8],
    "veo-3.1-fast-generate-preview": [4, 6, 8],
    "veo-3.1-fast-generate-001": [4, 6, 8],
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
    "imagen-3.0-generate-002": 0.03,
    "imagen-4.0-generate-001": 0.04,
    "imagen-4.0-fast-generate-001": 0.02,
    "imagen-4.0-ultra-generate-001": 0.06,
    "gemini-2.5-flash-image": 0.04,
    "gemini-3-pro-image-preview": 0.13,
}

VIDEO_MODEL_COST_SILENT: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-preview": 0.40,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-preview": 0.10,
    "veo-3.1-fast-generate-001": 0.10,
}

VIDEO_MODEL_COST_AUDIO: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-preview": 0.40,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-preview": 0.15,
    "veo-3.1-fast-generate-001": 0.15,
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
    prompt: str
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    clip_duration: int = 6
    total_duration: int = 15
    text_model: str = "gemini-2.5-flash"
    image_model: str = "imagen-4.0-fast-generate-001"
    video_model: str = "veo-3.1-fast-generate-001"
    enable_audio: bool = True


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


class ProjectDetail(BaseModel):
    """Response schema for GET /api/projects/{id}."""
    project_id: str
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


class ProjectListItem(BaseModel):
    """Item in list response for GET /api/projects."""
    project_id: str
    prompt: str
    status: str
    created_at: str
    total_duration: Optional[int] = None
    clip_duration: Optional[int] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None


class ResumeResponse(BaseModel):
    """Response schema for POST /api/projects/{id}/resume."""
    project_id: str
    status: str
    status_url: str


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
    created_at: str
    updated_at: str
    assets: list["AssetResponse"]


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

    # Validate model IDs
    if request.text_model not in ALLOWED_TEXT_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid text_model: {request.text_model}")
    if request.image_model not in ALLOWED_IMAGE_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid image_model: {request.image_model}")
    if request.video_model not in ALLOWED_VIDEO_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid video_model: {request.video_model}")

    # Validate audio: reject enable_audio=True for models without audio support
    if request.enable_audio and request.video_model not in AUDIO_CAPABLE_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Audio generation not supported for {request.video_model}",
        )

    # Derive scene count from total duration and clip duration
    scene_count = math.ceil(request.total_duration / request.clip_duration)

    async with async_session() as session:
        # Create project record
        project = Project(
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
            seed=random.randint(0, 2**32 - 1),
            status="pending",
        )
        session.add(project)
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

        return ProjectDetail(
            project_id=str(project.id),
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
        )


@router.get("/projects", response_model=list[ProjectListItem])
async def list_projects():
    """List all projects ordered by creation date (newest first)."""
    async with async_session() as session:
        result = await session.execute(
            select(Project).order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()

        return [
            ProjectListItem(
                project_id=str(p.id),
                prompt=p.prompt,
                status=p.status,
                created_at=p.created_at.isoformat(),
                total_duration=p.total_duration,
                clip_duration=p.target_clip_duration,
                text_model=p.text_model,
                image_model=p.image_model,
                video_model=p.video_model,
                audio_enabled=p.audio_enabled,
            )
            for p in projects
        ]


@router.post("/projects/{project_id}/resume", status_code=202, response_model=ResumeResponse)
async def resume_project(project_id: uuid.UUID, background_tasks: BackgroundTasks):
    """Resume failed or interrupted pipeline in background.

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


class _ExpansionScenes(BaseModel):
    """Wrapper schema for Gemini structured output of expansion scenes."""
    scenes: list[SceneSchema] = Field(description="New scenes to append")


async def _generate_expansion_scenes(
    project: Project,
    kept_scenes: list[dict],
    num_new_scenes: int,
    start_index: int,
) -> list[dict]:
    """Generate storyboard entries for new scenes extending an existing storyboard.

    Uses Gemini structured output to create scene entries that continue the
    narrative from the kept scenes, maintaining style and character consistency.

    Args:
        project: The new forked project (has prompt, style, etc.)
        kept_scenes: storyboard_raw["scenes"] entries for kept scenes
        num_new_scenes: How many new scenes to generate
        start_index: scene_index for the first new scene

    Returns:
        List of scene dicts ready for Scene record creation
    """
    model_id = project.text_model or "gemini-2.5-flash"
    client = get_vertex_client(location=location_for_model(model_id))
    style_label = (project.style or "cinematic").replace("_", " ")

    # Build context from kept scenes
    kept_summary = json.dumps(kept_scenes, indent=2) if kept_scenes else "[]"

    prompt = f"""You are a storyboard director. You have an existing partial storyboard and need to generate {num_new_scenes} NEW continuation scene(s).

VISUAL STYLE: {style_label}
ASPECT RATIO: {project.aspect_ratio}

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

    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_ExpansionScenes,
            temperature=0.7,
        ),
    )

    result = _ExpansionScenes.model_validate_json(response.text)

    # Normalize scene indices to start at start_index
    scenes_out = []
    for i, scene in enumerate(result.scenes[:num_new_scenes]):
        sd = scene.model_dump()
        sd["scene_index"] = start_index + i
        scenes_out.append(sd)

    return scenes_out


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
    """
    scene_count = source.target_scene_count or 3

    # Compute deletion info early — needed for scene count logic
    deleted_set = set(deleted_scenes) if deleted_scenes else set()
    deleted_count = len(deleted_set)
    kept_count = scene_count - deleted_count

    # Check if total_duration or clip_duration changes cause scene count change
    new_total = overrides.get("total_duration", source.total_duration) or source.total_duration or 15
    new_clip = overrides.get("clip_duration", source.target_clip_duration) or source.target_clip_duration or 6
    new_scene_count = math.ceil(new_total / new_clip)

    # Scene count change with no deletions
    if new_scene_count != scene_count and deleted_count == 0:
        if new_scene_count > scene_count:
            # Pure expansion: keep all existing scenes, only generate new ones
            return "keyframing", scene_count
        else:
            # Scene count decreased: full re-run
            return "pending", 0

    # Edits that force full re-run
    if any(k in overrides for k in ("prompt", "style", "text_model")):
        return "pending", 0

    # Edits that force keyframing restart
    if any(k in overrides for k in ("image_model", "aspect_ratio")):
        return "keyframing", 0

    # Edits that force video_gen restart (keep all keyframes)
    if any(k in overrides for k in ("video_model", "audio_enabled", "clip_duration")):
        if new_scene_count > kept_count:
            # Expansion needs keyframes for new scenes
            return "keyframing", kept_count
        return "video_gen", new_scene_count

    # Expansion after deletion: new scenes need storyboard+keyframes+video
    if new_scene_count > kept_count:
        return "keyframing", kept_count

    def _old_to_new(old_idx: int) -> int:
        """Map an original scene index to its post-deletion index."""
        return old_idx - sum(1 for d in deleted_set if d < old_idx)

    # Collect all keyframe-level and video-level invalidation candidates
    min_keyframe_boundary = new_scene_count
    min_video_boundary = new_scene_count

    # Scene deletions → keyframe invalidation at the deletion point (new numbering)
    if deleted_scenes:
        for d in deleted_scenes:
            if 0 <= d < scene_count:
                new_idx = _old_to_new(d)
                min_keyframe_boundary = min(min_keyframe_boundary, new_idx)

    # Cleared keyframes → keyframe invalidation (using original indices)
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
                       "text_model", "image_model", "video_model", "audio_enabled"):
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
        if tm and tm not in ALLOWED_TEXT_MODELS:
            raise HTTPException(status_code=422, detail=f"Invalid text_model: {tm}")

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

        # Compute invalidation point
        resume_from, scene_boundary = _compute_invalidation(
            source, overrides, request.scene_edits,
            deleted_scenes=request.deleted_scenes,
            clear_keyframes=request.clear_keyframes,
        )

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
            seed=random.randint(0, 2**32 - 1),
            forked_from_id=source.id,
            status=resume_from,
        )

        # Copy storyboard data if not starting from pending
        if resume_from != "pending":
            new_project.storyboard_raw = source.storyboard_raw
            new_project.style_guide = source.style_guide

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
                            dst_path = file_mgr.get_project_dir(new_id) / "keyframes" / src_path.name
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
                            dst_clip_path = file_mgr.get_project_dir(new_id) / "clips" / src_clip_path.name
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
                    new_project.storyboard_raw = sb

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


@router.get("/projects/{project_id}/download")
async def download_video(project_id: uuid.UUID):
    """Download final MP4 video file.

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

        # Return file as download
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"video_{project_id}.mp4",
            headers={
                "Content-Disposition": f'attachment; filename="video_{project_id}.mp4"'
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
        result = await session.execute(select(Project))
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

        # Find uploaded file in the uploads directory
        upload_dir = Path("tmp/manifests") / str(asset.manifest_id) / "uploads"
        matches = list(upload_dir.glob(f"{asset_id}_*")) if upload_dir.exists() else []
        if not matches:
            raise HTTPException(status_code=404, detail="Asset image not found on disk")

        file_path = matches[0]
        suffix = file_path.suffix.lower()
        media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        media_type = media_types.get(suffix, "image/png")

        return FileResponse(path=str(file_path), media_type=media_type)
