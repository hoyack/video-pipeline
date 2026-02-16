"""Video clip generation using Veo with first/last frame control.

This module implements video generation (VGEN-01 to VGEN-06):
- Submit Veo jobs with first and last frame interpolation
- Poll long-running operations with exponential backoff
- Handle RAI filtering gracefully without failing pipeline
- Detect and mark timeouts after max polls exceeded
- Persist operation ID before polling for idempotent resume
- Save MP4 clips to structured filesystem
- Escalating content-policy remediation (VGEN-07):
  Level 0: original prompt
  Level 1: prepend safety language to video prompt
  Level 2: regenerate end keyframe with safety prompt + retry video

Usage:
    from vidpipe.pipeline.video_gen import generate_videos

    async with async_session() as session:
        await generate_videos(session, project)
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from google.genai import types
from google.genai.errors import ClientError, ServerError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

from vidpipe.config import settings
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip
from vidpipe.services.file_manager import FileManager
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-policy safety prefixes (escalating strength)
# ---------------------------------------------------------------------------
_VIDEO_SAFETY_PREFIXES = [
    # Level 0: no modification
    "",
    # Level 1: gentle safety reminder
    (
        "Ensure this content is safe and appropriate for all audiences. "
        "Avoid any depiction of violence, weapons, nudity, or controversial themes. "
    ),
    # Level 2: strong safety directive
    (
        "CRITICAL: Create ONLY family-friendly, non-controversial content. "
        "Absolutely no violence, weapons, blood, nudity, suggestive content, drugs, "
        "or any material that could violate content policies. "
        "Focus purely on artistic visual storytelling. "
    ),
]


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------
def _is_retriable(exc: BaseException) -> bool:
    """Return True only for transient errors worth retrying (429, 5xx)."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", 0) == 429
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _is_content_policy_exception(exc: BaseException) -> bool:
    """Check if exception is a content-policy rejection (not transient)."""
    if isinstance(exc, ClientError):
        msg = str(exc).lower()
        return any(kw in msg for kw in (
            "violat", "usage guidelines",
            "safety", "content polic", "responsible ai",
        ))
    return False


def _is_content_policy_operation(operation) -> bool:
    """Check if a completed Veo operation failed due to content policy."""
    # Case 1: RAI media filtering
    if hasattr(operation, "response") and operation.response:
        count = getattr(operation.response, "rai_media_filtered_count", None)
        if count and count > 0:
            return True

    # Case 2: operation error with policy-related message
    if not (hasattr(operation, "response") and operation.response):
        error = getattr(operation, "error", None)
        if error:
            error_str = str(error).lower()
            return any(kw in error_str for kw in (
                "violat", "usage guidelines",
                "safety", "content polic", "responsible ai",
            ))

    return False


# ---------------------------------------------------------------------------
# Retry-decorated video submission
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _submit_video_job(
    client,
    video_model: str,
    video_prompt: str,
    start_frame_bytes: bytes,
    end_frame_bytes: bytes,
    project: Project,
):
    """Submit a Veo video generation job with retry on transient 429/5xx errors.

    Content-policy errors (400/INVALID_ARGUMENT) are NOT retried here;
    they propagate to the caller for escalation.
    """
    video_config = types.GenerateVideosConfig(
        aspect_ratio=project.aspect_ratio,
        duration_seconds=project.target_clip_duration,
        last_frame=types.Image(image_bytes=end_frame_bytes, mime_type="image/png"),
        negative_prompt=(
            "photorealistic, photo, photograph, hyperrealistic, "
            "text overlay, watermark, logo, blurry, deformed"
        ),
    )

    # Audio generation for Veo 3+ models
    if video_model != "veo-2.0-generate-001":
        video_config.generate_audio = bool(project.audio_enabled)

    # Consistent seed for visual coherence across scenes
    if project.seed is not None:
        video_config.seed = project.seed

    # Disable prompt rewriter on Veo 2
    if video_model == "veo-2.0-generate-001":
        video_config.enhance_prompt = False

    return await client.aio.models.generate_videos(
        model=video_model,
        prompt=video_prompt,
        image=types.Image(image_bytes=start_frame_bytes, mime_type="image/png"),
        config=video_config,
    )


# ---------------------------------------------------------------------------
# End-keyframe regeneration for content-policy remediation
# ---------------------------------------------------------------------------
async def _regenerate_end_keyframe_safe(
    session: AsyncSession,
    scene: Scene,
    project: Project,
    start_frame_bytes: bytes,
    end_kf: Keyframe,
    file_mgr: FileManager,
) -> Optional[bytes]:
    """Regenerate the end keyframe with a safety-focused prompt.

    Returns the new image bytes on success, None on failure.
    Updates the Keyframe record and file on disk in-place.
    """
    from vidpipe.api.routes import IMAGE_CONDITIONED_MAP
    from vidpipe.pipeline.keyframes import _generate_image_conditioned

    image_model = project.image_model or settings.models.image_gen
    conditioned_model = IMAGE_CONDITIONED_MAP.get(
        image_model, settings.models.image_conditioned,
    )
    conditioned_client = get_vertex_client(
        location=location_for_model(conditioned_model),
    )

    style_label = project.style.replace("_", " ")
    conditioning_prompt = (
        f"Generate a safe, family-friendly keyframe for a {style_label} production. "
        f"Do NOT include any violence, weapons, nudity, blood, or controversial content.\n\n"
        f"IMPORTANT: The new image must look CLEARLY DIFFERENT from the reference — "
        f"use a DIFFERENT camera angle, wider framing, or noticeably different pose. "
        f"If the reference is a close-up, pull back to a medium or wide shot.\n\n"
        f"TARGET END STATE:\n{scene.end_frame_prompt}\n\n"
        f"Maintain {style_label} style and character appearance consistency."
    )

    try:
        end_frame_bytes = await _generate_image_conditioned(
            conditioned_client,
            start_frame_bytes,
            conditioning_prompt,
            project.aspect_ratio,
            conditioned_model,
        )

        # Save to disk (overwrites existing file)
        end_file_path = file_mgr.save_keyframe(
            project.id, scene.scene_index, "end", end_frame_bytes,
        )
        end_kf.file_path = str(end_file_path)
        end_kf.prompt_used = conditioning_prompt
        end_kf.source = "generated"
        await session.commit()

        logger.info(
            f"Scene {scene.scene_index}: regenerated end keyframe with safety prompt"
        )
        return end_frame_bytes

    except Exception as e:
        logger.error(
            f"Scene {scene.scene_index}: failed to regenerate end keyframe: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Poll loop (extracted for reuse across escalation levels)
# ---------------------------------------------------------------------------
async def _poll_video_operation(
    session: AsyncSession,
    clip: VideoClip,
    client,
    project: Project,
    scene: Scene,
    file_mgr: FileManager,
) -> str:
    """Poll a Veo operation until completion.

    Returns one of:
      "complete"        — video saved successfully
      "content_policy"  — failed due to content policy (caller should escalate)
      "timed_out"       — max polls exceeded
      "failed"          — non-policy failure
    """
    poll_interval = settings.pipeline.video_poll_interval
    max_polls = settings.pipeline.video_poll_max

    for poll_attempt in range(clip.poll_count, max_polls):
        op_obj = types.GenerateVideosOperation(name=clip.operation_name)
        operation = await client.aio.operations.get(operation=op_obj)
        clip.poll_count = poll_attempt + 1

        if operation.done:
            # --- Content-policy check (both RAI filter and error) ---
            if _is_content_policy_operation(operation):
                error_msg = "Content filtered by responsible AI"
                if not (hasattr(operation, "response") and operation.response):
                    error_msg = str(getattr(operation, "error", error_msg))
                clip.status = "failed"
                clip.error_message = error_msg
                await session.commit()
                return "content_policy"

            if operation.response:
                # Success: download video
                gen_video = operation.response.generated_videos[0]
                if gen_video.video and gen_video.video.video_bytes:
                    video_bytes = gen_video.video.video_bytes
                elif gen_video.video and gen_video.video.gcs_uri:
                    video_bytes = await _download_from_gcs(gen_video.video.gcs_uri)
                else:
                    clip.status = "failed"
                    clip.error_message = "No video data in response"
                    scene.status = "failed"
                    await session.commit()
                    return "failed"

                # Save video clip (VGEN-06)
                file_path = file_mgr.save_clip(
                    project.id, scene.scene_index, video_bytes,
                )
                clip.local_path = str(file_path)
                clip.status = "complete"
                clip.duration_seconds = project.target_clip_duration
                clip.source = "generated"
                scene.status = "video_done"
                await session.commit()
                return "complete"

            else:
                # Non-policy operation failure
                clip.status = "failed"
                clip.error_message = (
                    str(operation.error)
                    if hasattr(operation, "error")
                    else "Unknown error"
                )
                scene.status = "failed"
                await session.commit()
                return "failed"

        # Not done yet — commit poll progress and sleep
        await session.commit()
        await asyncio.sleep(poll_interval)

        # Check for user-requested stop between polls
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

    # Timeout (VGEN-05)
    clip.status = "timed_out"
    clip.error_message = (
        f"Operation did not complete after {max_polls * poll_interval} seconds"
    )
    scene.status = "timed_out"
    await session.commit()
    return "timed_out"


# ---------------------------------------------------------------------------
# Main per-scene video generation with escalating remediation
# ---------------------------------------------------------------------------
async def generate_videos(session: AsyncSession, project: Project) -> None:
    """Generate video clips for all scenes using Veo.

    Implements VGEN-01 through VGEN-07.
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

    for scene in scenes:
        # Check for user-requested stop
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

        await _generate_video_for_scene(
            session, scene, file_mgr, client, project, video_model,
        )

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
    """Generate video clip for a single scene with escalating content-policy
    remediation and transient-error retry.

    Escalation levels:
      0 — original prompt
      1 — prepend safety language to video prompt, retry submission
      2 — regenerate end keyframe with safety prompt, retry with strong safety prefix

    Idempotent resume (VGEN-03): if VideoClip already exists with
    operation_name, resumes polling rather than submitting a new job.
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

    start_frame_bytes = Path(start_kf.file_path).read_bytes()
    end_frame_bytes = Path(end_kf.file_path).read_bytes()

    # Check if VideoClip already exists (idempotent resume per VGEN-03)
    result = await session.execute(
        select(VideoClip).where(VideoClip.scene_id == scene.id)
    )
    clip = result.scalar_one_or_none()

    # If clip exists and is still polling, resume the poll (crash recovery)
    if clip and clip.status == "polling" and clip.operation_name:
        poll_result = await _poll_video_operation(
            session, clip, client, project, scene, file_mgr,
        )
        if poll_result != "content_policy":
            return  # complete, failed, or timed_out
        # Content policy → fall through to escalation loop
        logger.warning(
            f"Scene {scene.scene_index}: resumed poll hit content policy, "
            "starting remediation"
        )

    # ---- Escalating content-policy remediation loop ----
    max_levels = len(_VIDEO_SAFETY_PREFIXES)

    for safety_level in range(max_levels):
        if safety_level > 0:
            logger.warning(
                f"Scene {scene.scene_index}: content policy remediation "
                f"level {safety_level}/{max_levels - 1}"
            )

        # Level 2+: regenerate end keyframe with safety-focused prompt
        if safety_level >= 2:
            new_bytes = await _regenerate_end_keyframe_safe(
                session, scene, project, start_frame_bytes, end_kf, file_mgr,
            )
            if new_bytes:
                end_frame_bytes = new_bytes

        # Build video prompt with escalating safety prefix
        video_prompt = (
            f"{_VIDEO_SAFETY_PREFIXES[safety_level]}"
            f"{scene.video_motion_prompt}. "
            f"Maintain the visual style shown in the source frames."
        )

        # Submit job (retries transient 429/5xx automatically)
        try:
            operation = await _submit_video_job(
                client, video_model, video_prompt,
                start_frame_bytes, end_frame_bytes, project,
            )
        except Exception as e:
            if (
                _is_content_policy_exception(e)
                and safety_level < max_levels - 1
            ):
                logger.warning(
                    f"Scene {scene.scene_index}: submission rejected "
                    f"(content policy) at level {safety_level}, escalating"
                )
                continue  # try next safety level
            # Fatal: transient retries exhausted or last safety level
            logger.error(
                f"Scene {scene.scene_index}: video submission failed: {e}"
            )
            if clip is None:
                clip = VideoClip(
                    scene_id=scene.id,
                    status="failed",
                    source="generated",
                    error_message=str(e),
                )
                session.add(clip)
            else:
                clip.status = "failed"
                clip.error_message = str(e)
            scene.status = "failed"
            await session.commit()
            return

        # Create / update clip record (VGEN-03: persist before polling)
        if clip is None:
            clip = VideoClip(
                scene_id=scene.id,
                operation_name=operation.name,
                status="polling",
                poll_count=0,
                source="generated",
            )
            session.add(clip)
        else:
            clip.operation_name = operation.name
            clip.status = "polling"
            clip.poll_count = 0
            clip.error_message = None
        await session.commit()

        # Poll operation
        poll_result = await _poll_video_operation(
            session, clip, client, project, scene, file_mgr,
        )

        if poll_result == "complete":
            if safety_level > 0:
                logger.info(
                    f"Scene {scene.scene_index}: succeeded at safety level "
                    f"{safety_level}"
                )
            return
        elif poll_result == "content_policy":
            # Reset scene status for next attempt
            scene.status = "keyframes_done"
            await session.commit()
            continue  # try next safety level
        else:
            # timed_out or non-policy failure — don't escalate
            return

    # All safety levels exhausted
    logger.error(
        f"Scene {scene.scene_index}: content policy remediation exhausted "
        f"after {max_levels} levels"
    )
    if clip:
        clip.status = "failed"
        clip.error_message = (
            "Content policy violation persisted after all remediation attempts"
        )
    scene.status = "failed"
    await session.commit()


async def _download_from_gcs(gcs_uri: str) -> bytes:
    """Download video bytes from Google Cloud Storage URI."""
    import httpx

    if gcs_uri.startswith("gs://"):
        http_url = gcs_uri.replace("gs://", "https://storage.googleapis.com/")
    else:
        http_url = gcs_uri

    async with httpx.AsyncClient() as client:
        response = await client.get(http_url)
        response.raise_for_status()
        return response.content
