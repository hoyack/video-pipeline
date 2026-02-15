"""API route handlers and Pydantic response schemas."""

import logging
import math
import random
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from vidpipe.db import async_session
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip
from vidpipe.orchestrator.pipeline import run_pipeline
from vidpipe.orchestrator.state import can_resume

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
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    audio_enabled: Optional[bool] = None


class ProjectListItem(BaseModel):
    """Item in list response for GET /api/projects."""
    project_id: str
    prompt: str
    status: str
    created_at: str


class ResumeResponse(BaseModel):
    """Response schema for POST /api/projects/{id}/resume."""
    project_id: str
    status: str
    status_url: str


class StopResponse(BaseModel):
    """Response schema for POST /api/projects/{id}/stop."""
    project_id: str
    status: str


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

            scene_details.append(SceneDetail(
                scene_index=scene.scene_index,
                description=scene.scene_description,
                status=scene.status,
                has_start_keyframe=has_start_keyframe,
                has_end_keyframe=has_end_keyframe,
                has_clip=clip is not None,
                clip_status=clip.status if clip else None,
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
            text_model=project.text_model,
            image_model=project.image_model,
            video_model=project.video_model,
            audio_enabled=project.audio_enabled,
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


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0"
    }
