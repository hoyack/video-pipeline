"""Video clip generation using Veo 3.1 with first/last frame control.

This module implements video generation (VGEN-01 to VGEN-06):
- Submit Veo jobs with first and last frame interpolation
- Poll long-running operations with exponential backoff
- Handle RAI filtering gracefully without failing pipeline
- Detect and mark timeouts after max polls exceeded
- Persist operation ID before polling for idempotent resume
- Save MP4 clips to structured filesystem

Usage:
    from vidpipe.pipeline.video_gen import generate_videos

    async with async_session() as session:
        await generate_videos(session, project)
"""

import asyncio
from pathlib import Path
from typing import Optional

from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.config import settings
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip
from vidpipe.services.file_manager import FileManager
from vidpipe.services.vertex_client import get_vertex_client, location_for_model


async def generate_videos(session: AsyncSession, project: Project) -> None:
    """Generate video clips for all scenes using Veo 3.1.

    Implements VGEN-01 through VGEN-06:
    - First/last frame interpolation for smooth motion
    - Long-running operation polling with configurable interval
    - Operation ID persistence before polling for crash recovery
    - RAI filter detection and graceful handling
    - Timeout detection after max polls
    - MP4 file saving to structured directories

    Args:
        session: Async database session
        project: Project to generate videos for

    Side effects:
        - Creates VideoClip records for each scene
        - Saves MP4 files to tmp/{project_id}/clips/
        - Updates scene statuses to video_done, rai_filtered, or timed_out
        - Updates project status to "stitching"
    """
    video_model = project.video_model or settings.models.video_gen
    client = get_vertex_client(location=location_for_model(video_model))
    file_mgr = FileManager()

    # Query scenes ready for video generation
    result = await session.execute(
        select(Scene)
        .where(Scene.project_id == project.id)
        .where(Scene.status == "keyframes_done")
        .order_by(Scene.scene_index)
    )
    scenes = result.scalars().all()

    # Generate video for each scene
    for scene in scenes:
        # Check for user-requested stop
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

        await _generate_video_for_scene(session, scene, file_mgr, client, project, video_model)

    # Update project status
    project.status = "stitching"
    await session.commit()


async def _generate_video_for_scene(
    session: AsyncSession,
    scene: Scene,
    file_mgr: FileManager,
    client,
    project: Project,
    video_model: str,
) -> None:
    """Generate video clip for a single scene with polling and error handling.

    Implements idempotent resume (VGEN-03): if VideoClip already exists with
    operation_name, resumes polling rather than submitting new job.

    Args:
        session: Async database session
        scene: Scene to generate video for
        file_mgr: FileManager instance for saving clips
        client: Vertex AI client
        project: Parent project for configuration

    Side effects:
        - Creates/updates VideoClip record
        - Saves MP4 file to filesystem
        - Updates scene status
        - Commits after each poll iteration
    """
    # Load keyframes from database
    result = await session.execute(
        select(Keyframe)
        .where(Keyframe.scene_id == scene.id)
        .order_by(Keyframe.position)
    )
    keyframes = result.scalars().all()
    start_kf = next(k for k in keyframes if k.position == "start")
    end_kf = next(k for k in keyframes if k.position == "end")

    # Load image bytes from files
    start_frame_bytes = Path(start_kf.file_path).read_bytes()
    end_frame_bytes = Path(end_kf.file_path).read_bytes()

    # Check if VideoClip already exists (idempotent resume per VGEN-03)
    result = await session.execute(
        select(VideoClip).where(VideoClip.scene_id == scene.id)
    )
    clip = result.scalar_one_or_none()

    # If clip is None, submit new Veo job
    if clip is None:
        # For image-to-video, the style is already baked into the keyframe
        # images. The prompt should focus only on motion/camera/action.
        # Adding style text can conflict with the source frames.
        video_prompt = (
            f"{scene.video_motion_prompt}. "
            f"Maintain the visual style shown in the source frames."
        )

        video_config = types.GenerateVideosConfig(
            aspect_ratio=project.aspect_ratio,
            duration_seconds=project.target_clip_duration,
            last_frame=types.Image(image_bytes=end_frame_bytes, mime_type="image/png"),
            negative_prompt=(
                "photorealistic, photo, photograph, hyperrealistic, "
                "text overlay, watermark, logo, blurry, deformed"
            ),
        )

        # Set audio generation for Veo 3+ models
        if video_model != "veo-2.0-generate-001":
            video_config.generate_audio = bool(project.audio_enabled)

        # Use consistent seed for visual coherence across scenes
        if project.seed is not None:
            video_config.seed = project.seed

        # Disable prompt rewriter on Veo 2 (prevents cinematic/photorealistic drift)
        if video_model == "veo-2.0-generate-001":
            video_config.enhance_prompt = False

        operation = await client.aio.models.generate_videos(
            model=video_model,
            prompt=video_prompt,
            image=types.Image(image_bytes=start_frame_bytes, mime_type="image/png"),
            config=video_config,
        )

        # CRITICAL: Persist operation ID BEFORE polling (VGEN-03)
        clip = VideoClip(
            scene_id=scene.id,
            operation_name=operation.name,
            status="polling",
            poll_count=0
        )
        session.add(clip)
        await session.commit()

    # Poll operation with backoff (VGEN-02)
    poll_interval = settings.pipeline.video_poll_interval  # default 15s
    max_polls = settings.pipeline.video_poll_max  # default 40 (~10 min)

    for poll_attempt in range(clip.poll_count, max_polls):
        op_obj = types.GenerateVideosOperation(name=clip.operation_name)
        operation = await client.aio.operations.get(operation=op_obj)
        clip.poll_count = poll_attempt + 1

        if operation.done:
            if operation.response:
                # Check for RAI filtering (VGEN-04)
                if operation.response.rai_media_filtered_count and operation.response.rai_media_filtered_count > 0:
                    clip.status = "rai_filtered"
                    clip.error_message = "Content filtered by responsible AI"
                    scene.status = "rai_filtered"
                    await session.commit()
                    return  # Continue with other scenes

                # Success: download video
                gen_video = operation.response.generated_videos[0]
                if gen_video.video and gen_video.video.video_bytes:
                    video_bytes = gen_video.video.video_bytes
                elif gen_video.video and gen_video.video.gcs_uri:
                    video_bytes = await _download_from_gcs(gen_video.video.gcs_uri)
                else:
                    raise ValueError("No video data in response")

                # Save video clip (VGEN-06)
                file_path = file_mgr.save_clip(project.id, scene.scene_index, video_bytes)
                clip.local_path = str(file_path)
                clip.status = "complete"
                clip.duration_seconds = project.target_clip_duration
                scene.status = "video_done"
                await session.commit()
                return

            else:
                # Operation failed
                clip.status = "failed"
                clip.error_message = str(operation.error) if hasattr(operation, 'error') else "Unknown error"
                scene.status = "failed"
                await session.commit()
                return

        # Not done yet, commit poll progress and sleep
        await session.commit()
        await asyncio.sleep(poll_interval)

        # Check for user-requested stop between polls
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

    # Timeout (VGEN-05)
    clip.status = "timed_out"
    clip.error_message = f"Operation did not complete after {max_polls * poll_interval} seconds"
    scene.status = "timed_out"
    await session.commit()


async def _download_from_gcs(gcs_uri: str) -> bytes:
    """Download video bytes from Google Cloud Storage URI.

    Args:
        gcs_uri: GCS URI (gs://bucket/path)

    Returns:
        Video bytes

    Raises:
        httpx.HTTPError: If download fails
    """
    import httpx

    # Convert gs://bucket/path to https://storage.googleapis.com/bucket/path
    if gcs_uri.startswith("gs://"):
        http_url = gcs_uri.replace("gs://", "https://storage.googleapis.com/")
    else:
        http_url = gcs_uri

    async with httpx.AsyncClient() as client:
        response = await client.get(http_url)
        response.raise_for_status()
        return response.content
