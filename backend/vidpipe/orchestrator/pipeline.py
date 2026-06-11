"""Main pipeline orchestrator with idempotent step execution and metadata tracking.

Coordinates the full video generation pipeline with:
- State machine transitions
- Idempotent resume from any failed step
- Per-step timing and logging
- Error handling with failure state persistence
- Progress callback interface for CLI/API integration
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Scene, PipelineRun, Shot, Keyframe, VideoClip, AssetAppearance, ShotManifest as ShotManifestModel, ShotAudioManifest as ShotAudioManifestModel
from vidpipe.db.models import UserSettings, DEFAULT_USER_ID
from vidpipe.orchestrator.state import get_resume_step
from vidpipe.pipeline.storyboard import generate_storyboard
from vidpipe.pipeline.keyframes import generate_keyframes
from vidpipe.pipeline.video_gen import generate_videos
from vidpipe.pipeline.stitcher import stitch_videos
from vidpipe.services.llm import get_adapter

logger = logging.getLogger(__name__)

# Maps run_through value to the pipeline status that marks completion of that stage
_STAGE_BOUNDARY = {
    "storyboard": "keyframing",   # after storyboard completes, status becomes "keyframing"
    "keyframes": "video_gen",     # after keyframes complete, status becomes "video_gen"
    "video": "stitching",         # after video_gen completes, status becomes "stitching"
}


class PipelineStopped(Exception):
    """Raised when the user requests a pipeline stop."""


class PipelineStaged(Exception):
    """Raised when the pipeline reaches a user-requested stage boundary."""


async def _generate_expansion_if_needed(
    session: AsyncSession, scene: Scene, text_adapter=None,
) -> None:
    """Generate storyboard entries for expansion shots if needed.

    After a fork with delete-then-expand, the scene may have fewer Shot
    records than target_shot_count. This generates the missing shots using
    the existing storyboard as context, creating Shot records and updating
    storyboard_raw before keyframing begins.

    For manifest-aware scenes (manifest_id set), generates enhanced shots
    with shot_manifest and audio_manifest, creating ShotManifest and
    ShotAudioManifest rows for the new shots.
    """
    target = scene.target_shot_count or 0
    if target <= 0:
        return

    # Count existing shots
    result = await session.execute(
        select(func.count(Shot.id)).where(Shot.scene_id == scene.id)
    )
    existing_count = result.scalar() or 0

    if existing_count >= target:
        return

    num_new = target - existing_count
    logger.info(
        f"Scene {scene.id}: expanding storyboard — "
        f"{existing_count} shots exist, {target} needed, generating {num_new}"
    )

    from vidpipe.api.routes import _generate_expansion_shots

    kept_sb_shots = []
    if scene.storyboard_raw and "shots" in scene.storyboard_raw:
        kept_sb_shots = scene.storyboard_raw["shots"]

    # For production-bible-aware scenes, load assets and use enhanced schema
    asset_registry_block = None
    if scene.production_bible_id:
        from vidpipe.services.manifest_service import load_manifest_assets, format_asset_registry
        assets = await load_manifest_assets(session, scene.production_bible_id)
        if assets:
            asset_registry_block = format_asset_registry(assets)

    new_shot_data = await _generate_expansion_shots(
        scene, kept_sb_shots, num_new, start_index=existing_count,
        asset_registry_block=asset_registry_block,
        text_adapter=text_adapter,
    )

    for sd in new_shot_data:
        shot = Shot(
            scene_id=scene.id,
            shot_index=sd.get("shot_index", existing_count),
            shot_description=sd.get("shot_description", ""),
            start_frame_prompt=sd.get("start_frame_prompt", ""),
            end_frame_prompt=sd.get("end_frame_prompt", ""),
            video_motion_prompt=sd.get("video_motion_prompt", ""),
            transition_notes=sd.get("transition_notes", ""),
            status="pending",
        )
        session.add(shot)

        # Create ShotManifest row for manifest-aware expansion shots
        sm_data = sd.get("shot_manifest")
        if sm_data:
            placements = sm_data.get("placements", [])
            composition = sm_data.get("composition", {})
            shot_manifest = ShotManifestModel(
                scene_id=scene.id,
                shot_index=sd.get("shot_index", existing_count),
                manifest_json=sm_data,
                composition_shot_type=composition.get("shot_type"),
                composition_camera_movement=composition.get("camera_movement"),
                asset_tags=[p.get("asset_tag") for p in placements if p.get("asset_tag")],
                new_asset_count=len(sm_data.get("new_asset_declarations") or []),
            )
            session.add(shot_manifest)

        # Create ShotAudioManifest row for manifest-aware expansion shots
        am_data = sd.get("audio_manifest")
        if am_data:
            dialogue_lines = am_data.get("dialogue_lines", [])
            audio_manifest = ShotAudioManifestModel(
                scene_id=scene.id,
                shot_index=sd.get("shot_index", existing_count),
                dialogue_json=dialogue_lines,
                sfx_json=am_data.get("sfx", []),
                ambient_json=am_data.get("ambient"),
                music_json=am_data.get("music"),
                audio_continuity_json=am_data.get("audio_continuity"),
                speaker_tags=[d.get("speaker_tag") for d in dialogue_lines if d.get("speaker_tag")],
                has_dialogue=len(dialogue_lines) > 0,
                has_music=am_data.get("music") is not None,
            )
            session.add(audio_manifest)

    # Update storyboard_raw with the new shots
    if scene.storyboard_raw:
        sb = dict(scene.storyboard_raw)
        sb.setdefault("shots", []).extend(new_shot_data)
        scene.storyboard_raw = sb

    await session.commit()
    logger.info(f"Scene {scene.id}: expansion complete, {num_new} shots added")


async def _check_stage_boundary(
    session: AsyncSession, scene: Scene,
) -> bool:
    """Check if pipeline should pause at the current stage boundary.

    Returns True (and sets status to 'staged') if run_through is set
    and the pipeline has reached that boundary.
    """
    if scene.run_through is None:
        return False
    boundary_status = _STAGE_BOUNDARY.get(scene.run_through)
    if boundary_status and scene.status == boundary_status:
        scene.status = "staged"
        await session.commit()
        logger.info(
            f"Scene {scene.id}: staged at {scene.run_through} boundary"
        )
        return True
    return False


async def _check_stopped(session: AsyncSession, scene_id: uuid.UUID) -> None:
    """Re-read scene status from DB; raise PipelineStopped if stopped."""
    result = await session.execute(select(Scene.status).where(Scene.id == scene_id))
    status = result.scalar_one()
    if status == "stopped":
        raise PipelineStopped("Pipeline stopped by user")


async def run_pipeline(
    session: AsyncSession,
    scene_id: uuid.UUID,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Execute full video generation pipeline with idempotent resume capability.

    Runs all 4 pipeline steps (storyboard, keyframes, video_gen, stitcher) with
    state machine transitions, failure recovery, and PipelineRun metadata tracking.

    Phase 6: If scene.production_bible_id is set, manifesting is skipped
    (assets already processed via ManifestSnapshot). When a manifesting
    pipeline step is added (Phase 7+), check scene.production_bible_id here
    and skip the manifesting step if present.

    Args:
        session: Async database session for all operations
        scene_id: UUID of scene to execute
        progress_callback: Optional callback for status updates (e.g., CLI progress display)

    Raises:
        ValueError: If scene not found
        Exception: Re-raises any unhandled pipeline step failure after persisting error state

    Side effects:
        - Creates PipelineRun record with timing metadata
        - Updates scene.status through state machine transitions
        - Persists scene.error_message on failure
        - Calls progress_callback with step descriptions if provided
    """
    # Load scene from database
    result = await session.execute(select(Scene).where(Scene.id == scene_id))
    scene = result.scalar_one_or_none()
    if not scene:
        raise ValueError(f"Scene {scene_id} not found")

    logger.info(f"Starting pipeline for scene {scene_id}, current status: {scene.status}")

    # Load UserSettings for adapter creation
    from vidpipe.config import settings as app_settings
    from vidpipe.services.model_catalog import canonical_model_id
    user_settings_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    user_settings = user_settings_result.scalar_one_or_none()

    scene.text_model = canonical_model_id(scene.text_model) or scene.text_model
    scene.image_model = canonical_model_id(scene.image_model) or scene.image_model
    scene.video_model = canonical_model_id(scene.video_model) or scene.video_model
    scene.vision_model = canonical_model_id(scene.vision_model) or scene.vision_model
    await session.commit()

    # Create adapters from scene config + user settings
    text_model_id = scene.text_model or app_settings.models.storyboard_llm
    vision_model_id = scene.vision_model or scene.text_model or app_settings.models.storyboard_llm
    text_adapter = get_adapter(text_model_id, user_settings=user_settings)
    vision_adapter = get_adapter(vision_model_id, user_settings=user_settings)

    # Create PipelineRun record
    run = PipelineRun(scene_id=scene_id)
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Track step timing
    step_log: Dict[str, float] = {}
    pipeline_start = time.monotonic()

    # Determine resume point using database state
    completed_steps = await _check_completed_steps(session, scene)
    resume_step = get_resume_step(scene.status, completed_steps)
    logger.info(f"Resume point: {resume_step}, completed_steps: {completed_steps}")

    # Reset draft/failed/stopped/staged status to resume step (VGED-02 safety check)
    if scene.status in ("draft", "failed", "stopped", "staged"):
        scene.status = resume_step
        await session.commit()
        logger.info(f"Scene {scene.id}: transitioned from previous status to {resume_step}")

    try:
        # Step 1: Storyboard generation
        if scene.status == "pending":
            step_start = time.monotonic()
            logger.info("Starting storyboard step")
            if progress_callback:
                progress_callback("Generating storyboard...")

            scene.status = "storyboarding"
            await session.commit()

            await generate_storyboard(session, scene, text_adapter=text_adapter)
            await session.refresh(scene)

            # Transition handled by generate_storyboard (sets status to "keyframing")
            step_duration = time.monotonic() - step_start
            step_log["storyboard"] = step_duration
            logger.info(f"Storyboard step completed in {step_duration:.2f}s")

            if await _check_stage_boundary(session, scene):
                raise PipelineStaged()

        await _check_stopped(session, scene_id)

        # Brief pause between phases so the frontend can fetch and display
        # each phase's results before the next phase overwrites generation_status.
        await asyncio.sleep(0.3)

        # Step 2: Keyframe generation
        if scene.status == "keyframing":
            # Check if expansion shots are needed (fork delete-then-expand)
            await _generate_expansion_if_needed(session, scene, text_adapter=text_adapter)

            step_start = time.monotonic()
            logger.info("Starting keyframes step")
            if progress_callback:
                progress_callback("Generating keyframes...")

            await generate_keyframes(session, scene, text_adapter=text_adapter)
            await session.refresh(scene)

            # Transition handled by generate_keyframes (sets status to "generating_video")
            # Note: The function sets "generating_video" but state machine expects "video_gen"
            # This is a deviation - need to fix status after keyframes
            if scene.status == "generating_video":
                scene.status = "video_gen"
                await session.commit()

            step_duration = time.monotonic() - step_start
            step_log["keyframes"] = step_duration
            logger.info(f"Keyframes step completed in {step_duration:.2f}s")

            if await _check_stage_boundary(session, scene):
                raise PipelineStaged()

        await _check_stopped(session, scene_id)
        await asyncio.sleep(0.3)

        # Step 3: Video generation
        if scene.status == "video_gen":
            step_start = time.monotonic()
            logger.info("Starting video generation step")
            if progress_callback:
                progress_callback("Generating video clips (with CV analysis)...")

            await generate_videos(session, scene, text_adapter=text_adapter, vision_adapter=vision_adapter)
            await session.refresh(scene)

            scene.status = "stitching"
            await session.commit()

            step_duration = time.monotonic() - step_start
            # Note: video_gen duration includes per-shot CV analysis (Phase 9)
            step_log["video_gen"] = step_duration
            logger.info(f"Video generation step completed in {step_duration:.2f}s")

            if await _check_stage_boundary(session, scene):
                raise PipelineStaged()

        await _check_stopped(session, scene_id)
        await asyncio.sleep(0.3)

        # Step 4: Stitching
        if scene.status == "stitching":
            step_start = time.monotonic()
            logger.info("Starting stitching step")
            if progress_callback:
                progress_callback("Stitching final video...")

            await stitch_videos(session, scene)
            await session.refresh(scene)

            # Transition handled by stitch_videos (sets status to "complete")
            step_duration = time.monotonic() - step_start
            step_log["stitching"] = step_duration
            logger.info(f"Stitching step completed in {step_duration:.2f}s")

        # Auto-create initial checkpoint (PipeSVN)
        if scene.status == "complete" and scene.head_sha is None:
            try:
                from vidpipe.services.checkpoint_service import create_checkpoint
                await create_checkpoint(session, scene, "Initial checkpoint")
                logger.info(f"Auto-checkpoint created for scene {scene.id}")
            except Exception as cp_err:
                logger.warning(f"Failed to create auto-checkpoint: {cp_err}")

        # Clear any stale error from a previous failed run that this
        # (re)run has now recovered from
        if scene.status == "complete" and scene.error_message:
            scene.error_message = None

        # Finalize PipelineRun on success
        run.completed_at = datetime.utcnow()
        run.total_duration_seconds = time.monotonic() - pipeline_start
        run.log = step_log
        await session.commit()

        logger.info(f"Pipeline completed successfully in {run.total_duration_seconds:.2f}s")

    except PipelineStaged:
        logger.info(f"Pipeline staged at {scene.run_through} boundary")
        run.completed_at = datetime.utcnow()
        run.total_duration_seconds = time.monotonic() - pipeline_start
        run.log = step_log
        await session.commit()
        return

    except PipelineStopped:
        logger.info(f"Pipeline stopped by user at step {scene.status}")
        run.completed_at = datetime.utcnow()
        run.total_duration_seconds = time.monotonic() - pipeline_start
        run.log = step_log
        await session.commit()
        return

    except Exception as e:
        # Handle pipeline failure
        logger.error(f"Pipeline failed at step {scene.status}: {type(e).__name__}: {str(e)}")

        # Determine which step failed based on current status
        if scene.status in ["pending", "storyboarding"]:
            step_name = "storyboard"
        elif scene.status == "keyframing":
            step_name = "keyframes"
        elif scene.status == "video_gen":
            step_name = "video_gen"
        elif scene.status == "stitching":
            step_name = "stitching"
        else:
            step_name = "unknown"

        # Persist failure state
        scene.status = "failed"
        scene.error_message = f"{step_name} failed: {type(e).__name__}: {str(e)}"
        await session.commit()

        # Update PipelineRun with partial data
        run.completed_at = datetime.utcnow()
        run.total_duration_seconds = time.monotonic() - pipeline_start
        run.log = step_log
        await session.commit()

        # Re-raise exception for caller to handle
        raise


async def _check_completed_steps(session: AsyncSession, scene: Scene) -> Dict[str, bool]:
    """Query database to determine which pipeline steps are complete.

    Note: Video generation (Phase 9+) includes per-shot CV analysis
    for progressive asset enrichment. Analysis results are stored in
    shot_manifests.cv_analysis_json.

    Args:
        session: Database session
        scene: Scene to check

    Returns:
        Dict with keys:
            - has_storyboard: True if scene has shots
            - has_keyframes: True if all shots have both start and end keyframes
            - has_clips: True if all shots have completed video clips
    """
    # Check for storyboard (has shots WITH content)
    # Draft scenes have Shot rows but empty descriptions — those need storyboarding
    shot_count_result = await session.execute(
        select(func.count(Shot.id)).where(Shot.scene_id == scene.id)
    )
    shot_count = shot_count_result.scalar()

    # A storyboard is "done" only if shots exist AND at least one has non-empty text
    # (draft scenes create empty Shot rows that still need LLM generation)
    has_storyboard = False
    if shot_count > 0:
        filled_count_result = await session.execute(
            select(func.count(Shot.id)).where(
                Shot.scene_id == scene.id,
                Shot.shot_description != "",
                Shot.start_frame_prompt != "",
            )
        )
        filled_count = filled_count_result.scalar() or 0
        has_storyboard = filled_count > 0

    # Check for keyframes (all shots have both start and end keyframes)
    if has_storyboard:
        # Get total shots
        total_shots = shot_count

        # Count shots with both start and end keyframes
        shots_with_keyframes_result = await session.execute(
            select(func.count(func.distinct(Shot.id)))
            .select_from(Shot)
            .join(Keyframe, Keyframe.shot_id == Shot.id)
            .where(Shot.scene_id == scene.id)
            .group_by(Shot.id)
            .having(func.count(Keyframe.id) >= 2)
        )
        shots_with_keyframes = len(shots_with_keyframes_result.all())

        has_keyframes = shots_with_keyframes == total_shots
    else:
        has_keyframes = False

    # Check for completed clips (all shots have non-pending clips)
    if has_storyboard:
        # Count shots with completed clips
        shots_with_clips_result = await session.execute(
            select(func.count(func.distinct(Shot.id)))
            .select_from(Shot)
            .join(VideoClip, VideoClip.shot_id == Shot.id)
            .where(Shot.scene_id == scene.id)
            .where(VideoClip.status.in_(["completed", "rai_filtered"]))
        )
        shots_with_clips = shots_with_clips_result.scalar()

        has_clips = shots_with_clips == shot_count
    else:
        has_clips = False

    return {
        "has_storyboard": has_storyboard,
        "has_keyframes": has_keyframes,
        "has_clips": has_clips,
    }
