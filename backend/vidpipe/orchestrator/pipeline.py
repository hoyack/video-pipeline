"""Main pipeline orchestrator with idempotent step execution and metadata tracking.

Coordinates the full video generation pipeline with:
- State machine transitions
- Idempotent resume from any failed step
- Per-step timing and logging
- Error handling with failure state persistence
- Progress callback interface for CLI/API integration
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Project, PipelineRun, Scene, Keyframe, VideoClip
from vidpipe.orchestrator.state import get_resume_step
from vidpipe.pipeline.storyboard import generate_storyboard
from vidpipe.pipeline.keyframes import generate_keyframes
from vidpipe.pipeline.video_gen import generate_videos
from vidpipe.pipeline.stitcher import stitch_videos

logger = logging.getLogger(__name__)


async def run_pipeline(
    session: AsyncSession,
    project_id: uuid.UUID,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Execute full video generation pipeline with idempotent resume capability.

    Runs all 4 pipeline steps (storyboard, keyframes, video_gen, stitcher) with
    state machine transitions, failure recovery, and PipelineRun metadata tracking.

    Args:
        session: Async database session for all operations
        project_id: UUID of project to execute
        progress_callback: Optional callback for status updates (e.g., CLI progress display)

    Raises:
        ValueError: If project not found
        Exception: Re-raises any unhandled pipeline step failure after persisting error state

    Side effects:
        - Creates PipelineRun record with timing metadata
        - Updates project.status through state machine transitions
        - Persists project.error_message on failure
        - Calls progress_callback with step descriptions if provided
    """
    # Load project from database
    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    logger.info(f"Starting pipeline for project {project_id}, current status: {project.status}")

    # Create PipelineRun record
    run = PipelineRun(project_id=project_id)
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Track step timing
    step_log: Dict[str, float] = {}
    pipeline_start = time.monotonic()

    # Determine resume point using database state
    completed_steps = await _check_completed_steps(session, project)
    resume_step = get_resume_step(project.status, completed_steps)
    logger.info(f"Resume point: {resume_step}, completed_steps: {completed_steps}")

    # Reset failed status to resume step
    if project.status == "failed":
        project.status = resume_step
        await session.commit()

    try:
        # Step 1: Storyboard generation
        if project.status == "pending":
            step_start = time.monotonic()
            logger.info("Starting storyboard step")
            if progress_callback:
                progress_callback("Generating storyboard...")

            project.status = "storyboarding"
            await session.commit()

            await generate_storyboard(session, project)
            await session.refresh(project)

            # Transition handled by generate_storyboard (sets status to "keyframing")
            step_duration = time.monotonic() - step_start
            step_log["storyboard"] = step_duration
            logger.info(f"Storyboard step completed in {step_duration:.2f}s")

        # Step 2: Keyframe generation
        if project.status == "keyframing":
            step_start = time.monotonic()
            logger.info("Starting keyframes step")
            if progress_callback:
                progress_callback("Generating keyframes...")

            await generate_keyframes(session, project)
            await session.refresh(project)

            # Transition handled by generate_keyframes (sets status to "generating_video")
            # Note: The function sets "generating_video" but state machine expects "video_gen"
            # This is a deviation - need to fix status after keyframes
            if project.status == "generating_video":
                project.status = "video_gen"
                await session.commit()

            step_duration = time.monotonic() - step_start
            step_log["keyframes"] = step_duration
            logger.info(f"Keyframes step completed in {step_duration:.2f}s")

        # Step 3: Video generation
        if project.status == "video_gen":
            step_start = time.monotonic()
            logger.info("Starting video generation step")
            if progress_callback:
                progress_callback("Generating video clips...")

            await generate_videos(session, project)
            await session.refresh(project)

            project.status = "stitching"
            await session.commit()

            step_duration = time.monotonic() - step_start
            step_log["video_gen"] = step_duration
            logger.info(f"Video generation step completed in {step_duration:.2f}s")

        # Step 4: Stitching
        if project.status == "stitching":
            step_start = time.monotonic()
            logger.info("Starting stitching step")
            if progress_callback:
                progress_callback("Stitching final video...")

            await stitch_videos(session, project)
            await session.refresh(project)

            # Transition handled by stitch_videos (sets status to "complete")
            step_duration = time.monotonic() - step_start
            step_log["stitching"] = step_duration
            logger.info(f"Stitching step completed in {step_duration:.2f}s")

        # Finalize PipelineRun on success
        run.completed_at = datetime.utcnow()
        run.total_duration_seconds = time.monotonic() - pipeline_start
        run.log = step_log
        await session.commit()

        logger.info(f"Pipeline completed successfully in {run.total_duration_seconds:.2f}s")

    except Exception as e:
        # Handle pipeline failure
        logger.error(f"Pipeline failed at step {project.status}: {type(e).__name__}: {str(e)}")

        # Determine which step failed based on current status
        if project.status in ["pending", "storyboarding"]:
            step_name = "storyboard"
        elif project.status == "keyframing":
            step_name = "keyframes"
        elif project.status == "video_gen":
            step_name = "video_gen"
        elif project.status == "stitching":
            step_name = "stitching"
        else:
            step_name = "unknown"

        # Persist failure state
        project.status = "failed"
        project.error_message = f"{step_name} failed: {type(e).__name__}: {str(e)}"
        await session.commit()

        # Update PipelineRun with partial data
        run.completed_at = datetime.utcnow()
        run.total_duration_seconds = time.monotonic() - pipeline_start
        run.log = step_log
        await session.commit()

        # Re-raise exception for caller to handle
        raise


async def _check_completed_steps(session: AsyncSession, project: Project) -> Dict[str, bool]:
    """Query database to determine which pipeline steps are complete.

    Args:
        session: Database session
        project: Project to check

    Returns:
        Dict with keys:
            - has_storyboard: True if project has scenes
            - has_keyframes: True if all scenes have both start and end keyframes
            - has_clips: True if all scenes have completed video clips
    """
    # Check for storyboard (has scenes)
    scene_count_result = await session.execute(
        select(func.count(Scene.id)).where(Scene.project_id == project.id)
    )
    scene_count = scene_count_result.scalar()
    has_storyboard = scene_count > 0

    # Check for keyframes (all scenes have both start and end keyframes)
    if has_storyboard:
        # Get total scenes
        total_scenes = scene_count

        # Count scenes with both start and end keyframes
        scenes_with_keyframes_result = await session.execute(
            select(func.count(func.distinct(Scene.id)))
            .select_from(Scene)
            .join(Keyframe, Keyframe.scene_id == Scene.id)
            .where(Scene.project_id == project.id)
            .group_by(Scene.id)
            .having(func.count(Keyframe.id) >= 2)
        )
        scenes_with_keyframes = len(scenes_with_keyframes_result.all())

        has_keyframes = scenes_with_keyframes == total_scenes
    else:
        has_keyframes = False

    # Check for completed clips (all scenes have non-pending clips)
    if has_storyboard:
        # Count scenes with completed clips
        scenes_with_clips_result = await session.execute(
            select(func.count(func.distinct(Scene.id)))
            .select_from(Scene)
            .join(VideoClip, VideoClip.scene_id == Scene.id)
            .where(Scene.project_id == project.id)
            .where(VideoClip.status.in_(["completed", "rai_filtered"]))
        )
        scenes_with_clips = scenes_with_clips_result.scalar()

        has_clips = scenes_with_clips == scene_count
    else:
        has_clips = False

    return {
        "has_storyboard": has_storyboard,
        "has_keyframes": has_keyframes,
        "has_clips": has_clips,
    }
