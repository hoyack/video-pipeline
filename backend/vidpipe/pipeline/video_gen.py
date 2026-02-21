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
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip, GenerationCandidate
from vidpipe.services.candidate_scoring import CandidateScoringService
from vidpipe.services.cv_analysis_service import CVAnalysisService
from vidpipe.services.entity_extraction import identify_new_entities, extract_and_register_new_entities
from vidpipe.services.file_manager import FileManager
from vidpipe.services.llm import get_adapter, LLMAdapter
from vidpipe.services.vertex_client import get_vertex_client, location_for_model
from vidpipe.services import manifest_service

# ---------------------------------------------------------------------------
# ComfyUI model IDs (routed to ComfyUI instead of Veo)
# ---------------------------------------------------------------------------
COMFYUI_VIDEO_MODELS = {"wan-2.2-ref-i2v", "wan-2.2-i2v"}


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy CV analysis service (Phase 9: progressive enrichment)
# ---------------------------------------------------------------------------
_cv_analysis_service: CVAnalysisService | None = None


def _get_cv_analysis_service() -> CVAnalysisService:
    """Return singleton CVAnalysisService, creating it on first call."""
    global _cv_analysis_service
    if _cv_analysis_service is None:
        _cv_analysis_service = CVAnalysisService()
    return _cv_analysis_service


# ---------------------------------------------------------------------------
# Lazy candidate scoring service (Phase 11: multi-candidate quality mode)
# ---------------------------------------------------------------------------
_candidate_scoring_service: CandidateScoringService | None = None


def _get_candidate_scoring_service() -> CandidateScoringService:
    """Return singleton CandidateScoringService, creating it on first call."""
    global _candidate_scoring_service
    if _candidate_scoring_service is None:
        _candidate_scoring_service = CandidateScoringService()
    return _candidate_scoring_service

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
# Transient gRPC operation error classification
# ---------------------------------------------------------------------------
_TRANSIENT_GRPC_CODES = {4, 8, 13, 14}  # DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, INTERNAL, UNAVAILABLE


def _is_transient_operation(operation) -> bool:
    """Return True if a completed Veo operation failed with a transient gRPC error.

    These errors (DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, INTERNAL, UNAVAILABLE)
    are server-side transient failures that warrant resubmission of the job.
    """
    error = getattr(operation, "error", None)
    if error is None:
        return False
    code = getattr(error, "code", None)
    if code is None:
        return False
    return code in _TRANSIENT_GRPC_CODES


# ---------------------------------------------------------------------------
# Retry-decorated polling RPC (guards operations.get against 429/5xx)
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _poll_operation_get(client, operation_name: str):
    """Fetch operation status with retry on transient HTTP errors (429/5xx)."""
    op_obj = types.GenerateVideosOperation(name=operation_name)
    return await client.aio.operations.get(operation=op_obj)


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
    reference_images: Optional[list] = None,
    candidate_count: int = 1,
):
    """Submit a Veo video generation job with retry on transient 429/5xx errors.

    Content-policy errors (400/INVALID_ARGUMENT) are NOT retried here;
    they propagate to the caller for escalation.
    """
    video_config = types.GenerateVideosConfig(
        aspect_ratio=project.aspect_ratio,
        duration_seconds=project.target_clip_duration,
        number_of_videos=candidate_count,
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

    # Veo API constraint: `image` + `last_frame` (frame interpolation) and
    # `reference_images` are mutually exclusive.  Keyframe-based frame
    # interpolation is essential for scene composition control, so we always
    # prefer it.  Reference images (identity preservation) are logged as
    # skipped — the rewritten prompt already describes the characters/assets.
    if reference_images:
        logger.info(
            f"Dropping {len(reference_images)} reference image(s) — "
            "frame interpolation (keyframes) takes priority over identity refs"
        )

    video_config.last_frame = types.Image(
        image_bytes=end_frame_bytes, mime_type="image/png"
    )
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
    from vidpipe.pipeline.keyframes import _generate_image_conditioned, COMFYUI_IMAGE_MODELS, _generate_image_comfyui

    image_model = project.image_model or settings.models.image_gen
    # Guard: Imagen models no longer supported — fall back to config default
    if image_model.startswith("imagen-"):
        image_model = settings.models.image_gen

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
        if image_model in COMFYUI_IMAGE_MODELS:
            from vidpipe.services.comfyui_client import get_comfyui_client
            comfy_client = await get_comfyui_client()
            end_frame_bytes = await _generate_image_comfyui(
                comfy_client, conditioning_prompt,
                seed=project.seed + scene.scene_index + 2000,
            )
        else:
            conditioned_client = get_vertex_client(
                location=location_for_model(image_model),
            )
            end_frame_bytes = await _generate_image_conditioned(
                conditioned_client,
                start_frame_bytes,
                conditioning_prompt,
                project.aspect_ratio,
                image_model,
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
    selected_refs: Optional[list] = None,
) -> str:
    """Poll a Veo operation until completion.

    Returns one of:
      "complete"        — video saved successfully
      "content_policy"  — failed due to content policy (caller should escalate)
      "transient"       — transient server error (caller should resubmit)
      "timed_out"       — max polls exceeded
      "failed"          — non-policy failure
    """
    poll_interval = settings.pipeline.video_poll_interval
    max_polls = settings.pipeline.video_poll_max

    for poll_attempt in range(clip.poll_count, max_polls):
        operation = await _poll_operation_get(client, clip.operation_name)
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

            # --- Transient server error check ---
            if _is_transient_operation(operation):
                error_msg = str(getattr(operation, "error", "Transient server error"))
                logger.warning(
                    f"Scene {scene.scene_index}: transient operation error "
                    f"(code {operation.error.code}): {error_msg}"
                )
                clip.status = "failed"
                clip.error_message = error_msg
                await session.commit()
                return "transient"

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
# Phase 11: Multi-candidate poll (collects ALL video bytes for quality mode)
# ---------------------------------------------------------------------------
async def _poll_and_collect_candidates(
    session: AsyncSession,
    clip: VideoClip,
    client,
    project: Project,
    scene: Scene,
    file_mgr: FileManager,
    selected_refs: Optional[list] = None,
) -> tuple[str, list[bytes]]:
    """Poll Veo operation and collect ALL candidate video bytes.

    Returns (status, video_bytes_list) where status is one of:
      "complete"        — all surviving candidates downloaded
      "content_policy"  — zero candidates survived RAI filtering
      "transient"       — transient server error (caller should resubmit)
      "timed_out"       — max polls exceeded
      "failed"          — non-policy failure

    For content_policy: in Quality Mode, only escalate if ZERO candidates survive.
    Some candidates may be RAI filtered while others succeed (partial success treated
    as "complete" with available survivors).
    """
    poll_interval = settings.pipeline.video_poll_interval
    max_polls = settings.pipeline.video_poll_max

    for poll_attempt in range(clip.poll_count, max_polls):
        operation = await _poll_operation_get(client, clip.operation_name)
        clip.poll_count = poll_attempt + 1

        if operation.done:
            # Check how many candidates survived
            generated_videos = []
            if hasattr(operation, "response") and operation.response:
                generated_videos = list(operation.response.generated_videos or [])

            rai_filtered = 0
            if hasattr(operation, "response") and operation.response:
                rai_filtered = getattr(operation.response, "rai_media_filtered_count", 0) or 0

            if len(generated_videos) == 0:
                # All candidates filtered or operation error
                if _is_content_policy_operation(operation):
                    error_msg = "Content filtered by responsible AI (all candidates)"
                    if not (hasattr(operation, "response") and operation.response):
                        error_msg = str(getattr(operation, "error", error_msg))
                    clip.status = "failed"
                    clip.error_message = error_msg
                    await session.commit()
                    return "content_policy", []

                # Transient server error check
                if _is_transient_operation(operation):
                    error_msg = str(getattr(operation, "error", "Transient server error"))
                    logger.warning(
                        f"Scene {scene.scene_index}: transient operation error "
                        f"(code {operation.error.code}): {error_msg}"
                    )
                    clip.status = "failed"
                    clip.error_message = error_msg
                    await session.commit()
                    return "transient", []

                # Non-policy, non-transient failure
                clip.status = "failed"
                clip.error_message = (
                    str(operation.error)
                    if hasattr(operation, "error")
                    else "Unknown error — no generated videos"
                )
                scene.status = "failed"
                await session.commit()
                return "failed", []

            # At least one candidate survived — partial RAI filter is OK
            if rai_filtered > 0:
                logger.warning(
                    f"Scene {scene.scene_index}: {rai_filtered} candidate(s) filtered by RAI, "
                    f"{len(generated_videos)} candidate(s) survived — treating as success"
                )

            # Download all surviving candidate bytes
            video_bytes_list: list[bytes] = []
            for gen_video in generated_videos:
                if gen_video.video and gen_video.video.video_bytes:
                    video_bytes_list.append(gen_video.video.video_bytes)
                elif gen_video.video and gen_video.video.gcs_uri:
                    video_bytes_list.append(
                        await _download_from_gcs(gen_video.video.gcs_uri)
                    )
                else:
                    logger.warning(
                        f"Scene {scene.scene_index}: candidate missing video data, skipping"
                    )

            if not video_bytes_list:
                clip.status = "failed"
                clip.error_message = "No video data in any candidate response"
                scene.status = "failed"
                await session.commit()
                return "failed", []

            # Mark clip as polling-complete (final path set later in _handle_quality_mode_candidates)
            clip.status = "complete"
            await session.commit()
            return "complete", video_bytes_list

        # Not done yet — commit poll progress and sleep
        await session.commit()
        await asyncio.sleep(poll_interval)

        # Check for user-requested stop between polls
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

    # Timeout
    clip.status = "timed_out"
    clip.error_message = (
        f"Operation did not complete after {max_polls * poll_interval} seconds"
    )
    scene.status = "timed_out"
    await session.commit()
    return "timed_out", []


# ---------------------------------------------------------------------------
# Phase 11: Save, score, and select best candidate (quality mode helper)
# ---------------------------------------------------------------------------
async def _handle_quality_mode_candidates(
    session: AsyncSession,
    scene: Scene,
    project: Project,
    clip: VideoClip,
    file_mgr: FileManager,
    video_bytes_list: list[bytes],
    scene_manifest_row,
    video_prompt: str,
    all_assets: list,
    has_refs: bool = False,
    scoring_service: Optional[CandidateScoringService] = None,
    cv_service: Optional[CVAnalysisService] = None,
) -> None:
    """Save all candidate videos, score them, and auto-select the best one.

    Steps:
    1. Save each candidate video to disk
    2. Create GenerationCandidate records
    3. Score all candidates via CandidateScoringService
    4. Update records with scores, mark winner as is_selected=True
    5. Update VideoClip.local_path to point to selected candidate
    6. Run CV analysis on selected candidate only
    """
    clips_dir = file_mgr.base_dir / str(project.id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    candidate_records: list[GenerationCandidate] = []

    # Step 1 & 2: Save each candidate and create DB records
    for i, video_bytes in enumerate(video_bytes_list):
        candidate_path = clips_dir / f"scene_{scene.scene_index}_candidate_{i}.mp4"
        candidate_path.write_bytes(video_bytes)

        candidate = GenerationCandidate(
            project_id=project.id,
            scene_index=scene.scene_index,
            candidate_number=i,
            local_path=str(candidate_path),
        )
        session.add(candidate)
        candidate_records.append(candidate)

    await session.flush()  # Assign IDs without full commit

    # Step 3: Find previous scene's selected clip for continuity scoring
    previous_clip_path = None
    if scene.scene_index > 0:
        prev_scene_result = await session.execute(
            select(Scene).where(
                Scene.project_id == project.id,
                Scene.scene_index == scene.scene_index - 1,
            )
        )
        prev_scene = prev_scene_result.scalar_one_or_none()
        if prev_scene:
            prev_clip_result = await session.execute(
                select(VideoClip).where(VideoClip.scene_id == prev_scene.id)
            )
            prev_clip = prev_clip_result.scalar_one_or_none()
            if prev_clip:
                previous_clip_path = prev_clip.local_path

    # Step 4: Score all candidates
    effective_scoring_service = scoring_service or _get_candidate_scoring_service()
    # Build candidates_info list (score_all_candidates expects dicts with "local_path")
    candidates_info = [{"local_path": cand.local_path} for cand in candidate_records]

    score_results = await effective_scoring_service.score_all_candidates(
        candidates_info=candidates_info,
        scene_index=scene.scene_index,
        scene_manifest_json=scene_manifest_row.manifest_json if scene_manifest_row else None,
        rewritten_video_prompt=video_prompt,
        existing_assets=all_assets,
        previous_scene_clip_path=previous_clip_path,
    )

    # Step 5: Update GenerationCandidate records with scores
    for cand, scores in zip(candidate_records, score_results):
        cand.manifest_adherence_score = scores.get("manifest_adherence_score")
        cand.visual_quality_score = scores.get("visual_quality_score")
        cand.continuity_score = scores.get("continuity_score")
        cand.prompt_adherence_score = scores.get("prompt_adherence_score")
        cand.composite_score = scores.get("composite_score")
        cand.scoring_details = scores.get("scoring_details")
        cand.scoring_cost = scores.get("scoring_cost", 0.0)

    # Step 6: Select winner (highest composite_score)
    winner_idx = max(
        range(len(score_results)),
        key=lambda i: score_results[i].get("composite_score", 0.0),
    )
    candidate_records[winner_idx].is_selected = True
    candidate_records[winner_idx].selected_by = "auto"

    logger.info(
        f"Scene {scene.scene_index}: selected candidate {winner_idx} "
        f"(composite={score_results[winner_idx].get('composite_score', 0):.2f}) "
        f"from {len(candidate_records)} candidates"
    )

    # Step 7: Update VideoClip.local_path to point to selected candidate
    clip.local_path = candidate_records[winner_idx].local_path
    clip.duration_seconds = project.target_clip_duration
    clip.source = "generated"
    scene.status = "video_done"

    await session.commit()

    # Step 8: Run CV analysis on selected candidate only
    await _run_post_generation_analysis(session, scene, clip, project, scene_manifest_row, cv_service=cv_service)


# ---------------------------------------------------------------------------
# Phase 9: Post-generation CV analysis for progressive enrichment
# ---------------------------------------------------------------------------
async def _run_post_generation_analysis(
    session: AsyncSession,
    scene: Scene,
    clip: VideoClip,
    project: Project,
    scene_manifest_row,  # SceneManifest | None (imported inline to avoid circular)
    cv_service: Optional[CVAnalysisService] = None,
) -> None:
    """Run CV analysis on completed video clip for progressive enrichment.

    This runs AFTER each scene's video clip completes, BEFORE the next scene
    starts. Enables progressive enrichment: assets from scene N feed into
    scene N+1's reference selection.

    Gracefully degrades — if analysis fails, logs warning and pipeline continues.
    Non-manifest projects skip CV analysis entirely (backward compatible).
    """
    try:
        # Guard 1: non-manifest projects skip CV analysis entirely
        if not project.manifest_id:
            return

        # Guard 2: no video clip path to analyze
        if not clip.local_path:
            return

        # Use provided service or fall back to adapter-unaware singleton for backward compat
        effective_cv_service = cv_service or _get_cv_analysis_service()

        # Load all manifest assets for face matching and entity extraction
        all_assets = await manifest_service.load_manifest_assets(
            session, project.manifest_id
        )

        # Collect keyframe paths (start and end frames for this scene)
        kf_result = await session.execute(
            select(Keyframe)
            .where(Keyframe.scene_id == scene.id)
            .order_by(Keyframe.position)
        )
        keyframes = kf_result.scalars().all()
        keyframe_paths = [
            kf.file_path
            for kf in keyframes
            if kf.file_path and Path(kf.file_path).exists()
        ]

        # Use a savepoint so that if CV analysis fails partway through
        # (after adding appearances/entities to the session), only the
        # savepoint is rolled back — not the outer transaction.  A full
        # session.rollback() would expire every ORM object in the session
        # (regardless of expire_on_commit) and break the caller's scene loop.
        async with session.begin_nested():
            # Step 5: Run full CV analysis (frame sampling, YOLO, face match, CLIP, vision LLM)
            analysis_result = await effective_cv_service.analyze_generated_content(
                scene_index=scene.scene_index,
                keyframe_paths=keyframe_paths,
                clip_path=clip.local_path,
                scene_manifest_json=(
                    scene_manifest_row.manifest_json if scene_manifest_row else None
                ),
                existing_assets=all_assets,
            )

            # Step 6: Track appearances — persist AssetAppearance records
            await effective_cv_service.track_appearances(
                session, project.id, scene.scene_index, analysis_result
            )

            # Step 7: Extract and register new entities into Asset Registry
            new_entities = identify_new_entities(analysis_result, all_assets)
            if new_entities:
                await extract_and_register_new_entities(
                    session,
                    project.id,
                    project.manifest_id,
                    scene.scene_index,
                    new_entities,
                    source="CLIP_EXTRACT",
                )

            # Step 8: Persist analysis results to SceneManifest (exclude raw embeddings)
            if scene_manifest_row is not None:
                scene_manifest_row.cv_analysis_json = analysis_result.model_dump(
                    exclude={"clip_embeddings"}
                )
                scene_manifest_row.continuity_score = analysis_result.continuity_score

        # Savepoint released — now commit to persist the analysis results
        await session.commit()

        logger.info(
            f"Scene {scene.scene_index}: CV analysis complete — "
            f"{len(analysis_result.face_matches)} face matches, "
            f"{analysis_result.new_entity_count} new entities, "
            f"continuity: {analysis_result.continuity_score:.1f}"
        )

    except Exception as e:
        logger.warning(
            f"Scene {scene.scene_index}: CV analysis failed (non-fatal): {e}"
        )
        # Pipeline continues — CV analysis failure is NOT a pipeline failure


# ---------------------------------------------------------------------------
# Main per-scene video generation with escalating remediation
# ---------------------------------------------------------------------------
async def generate_videos(
    session: AsyncSession,
    project: Project,
    text_adapter: Optional[LLMAdapter] = None,
    vision_adapter: Optional[LLMAdapter] = None,
) -> None:
    """Generate video clips for all scenes using Veo or ComfyUI.

    Implements VGEN-01 through VGEN-07.
    Routes to ComfyUI when video_model is in COMFYUI_VIDEO_MODELS.

    Args:
        session: Database session for persisting clips and candidates
        project: Project containing scenes to generate videos for
        text_adapter: Optional LLMAdapter for prompt rewriting. If None,
            PromptRewriterService falls back to get_adapter("gemini-2.5-flash").
        vision_adapter: Optional LLMAdapter for CV analysis and candidate scoring.
            If None, services fall back to get_adapter("gemini-2.5-flash").
    """
    video_model = project.video_model or settings.models.video_gen
    is_comfyui = video_model in COMFYUI_VIDEO_MODELS
    client = None if is_comfyui else get_vertex_client(location=location_for_model(video_model))
    file_mgr = FileManager()

    # Instantiate per-call services with vision_adapter instead of adapter-unaware singletons
    cv_service = CVAnalysisService(vision_adapter=vision_adapter)
    scoring_service = CandidateScoringService(vision_adapter=vision_adapter)

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

        if is_comfyui:
            await _generate_video_comfyui(
                session, scene, file_mgr, project, video_model,
            )
        else:
            await _generate_video_for_scene(
                session, scene, file_mgr, client, project, video_model,
                cv_service=cv_service,
                scoring_service=scoring_service,
                text_adapter=text_adapter,
            )

    # Update project status
    project.status = "stitching"
    await session.commit()


# ---------------------------------------------------------------------------
# ComfyUI video generation path
# ---------------------------------------------------------------------------
async def _generate_video_comfyui(
    session: AsyncSession,
    scene: Scene,
    file_mgr: FileManager,
    project: Project,
    video_model: str,
) -> None:
    """Generate video clip for a single scene via ComfyUI Cloud.

    Uses ComfyUIVideoAdapter which handles image upload, workflow building,
    status normalization, and result download.

    Simplified path compared to Veo:
    - No RAI escalation (Wan 2.2 has no content-policy filter)
    - No multi-candidate quality mode
    - Single output per run
    - Idempotent resume via operation_name prefix "comfyui:"
    """
    from vidpipe.db.models import SceneManifest as SceneManifestModel
    from vidpipe.services.comfyui_client import get_comfyui_client
    from vidpipe.services.comfyui_adapter import ComfyUIVideoAdapter

    is_i2v = video_model == "wan-2.2-i2v"

    # Load keyframes
    kf_result = await session.execute(
        select(Keyframe)
        .where(Keyframe.scene_id == scene.id)
        .order_by(Keyframe.position)
    )
    keyframes = kf_result.scalars().all()
    start_kf = next((k for k in keyframes if k.position == "start"), None)
    end_kf = next((k for k in keyframes if k.position == "end"), None)
    if is_i2v:
        if start_kf is None:
            raise ValueError(
                f"Scene {scene.scene_index} missing start keyframe — cannot generate video"
            )
    else:
        if start_kf is None or end_kf is None:
            missing = [p for p, kf in [("start", start_kf), ("end", end_kf)] if kf is None]
            raise ValueError(
                f"Scene {scene.scene_index} missing {' and '.join(missing)} "
                f"keyframe(s) — cannot generate video"
            )

    start_frame_bytes = Path(start_kf.file_path).read_bytes()
    end_frame_bytes = Path(end_kf.file_path).read_bytes() if end_kf else None

    # Load scene manifest for prompt rewriting and char refs
    scene_manifest_row = None
    if project.manifest_id:
        sm_result = await session.execute(
            select(SceneManifestModel).where(
                SceneManifestModel.project_id == project.id,
                SceneManifestModel.scene_index == scene.scene_index,
            )
        )
        scene_manifest_row = sm_result.scalar_one_or_none()

    # Build video prompt
    video_prompt = scene.video_motion_prompt
    if scene_manifest_row and scene_manifest_row.rewritten_video_prompt:
        video_prompt = scene_manifest_row.rewritten_video_prompt

    # Check for existing VideoClip (idempotent resume)
    clip_result = await session.execute(
        select(VideoClip).where(VideoClip.scene_id == scene.id)
    )
    clip = clip_result.scalar_one_or_none()

    # Build adapter from DB settings
    from vidpipe.db.models import UserSettings, DEFAULT_USER_ID
    us_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    user_settings = us_result.scalar_one_or_none()
    comfy_host = user_settings.comfyui_host if user_settings else None
    comfy_key = user_settings.comfyui_api_key if user_settings else None
    comfy_client = await get_comfyui_client(host=comfy_host, api_key=comfy_key)
    adapter = ComfyUIVideoAdapter(comfy_client)

    # If clip exists with comfyui: prefix and is polling, resume poll
    if clip and clip.status == "polling" and clip.operation_name and clip.operation_name.startswith("comfyui:"):
        logger.info(
            "Scene %d: resuming ComfyUI poll for %s",
            scene.scene_index, clip.operation_name,
        )
    else:
        # Fresh submission via adapter
        logger.info("Scene %d: submitting to ComfyUI", scene.scene_index)

        # Load character reference images (if manifest project)
        char_ref_bytes: list[bytes] = []
        if project.manifest_id:
            char_ref_bytes = await _load_char_ref_images(session, project)

        operation_id = await adapter.submit(
            video_prompt=video_prompt,
            start_frame_bytes=start_frame_bytes,
            end_frame_bytes=end_frame_bytes,
            char_ref_bytes=char_ref_bytes,
            aspect_ratio=project.aspect_ratio,
            seed=project.seed or 0,
            scene_index=scene.scene_index,
            video_model=video_model,
        )

        # Create/update clip record (persist before polling for crash recovery)
        if clip is None:
            clip = VideoClip(
                scene_id=scene.id,
                operation_name=operation_id,
                status="polling",
                poll_count=0,
                source="generated",
                veo_submission_count=1,
            )
            session.add(clip)
        else:
            clip.operation_name = operation_id
            clip.status = "polling"
            clip.poll_count = 0
            clip.error_message = None
        clip.prompt_used = video_prompt
        await session.commit()

    # --- Poll loop (adapter normalizes status to "completed"/"failed"/"running") ---
    poll_interval = settings.pipeline.video_poll_interval
    max_polls = settings.pipeline.video_poll_max

    for poll_attempt in range(clip.poll_count, max_polls):
        status, error_msg = await adapter.poll(clip.operation_name)
        clip.poll_count = poll_attempt + 1

        if status == "completed":
            # Download via adapter (handles history parsing + download)
            try:
                video_bytes, duration = await adapter.download(clip.operation_name)
            except Exception as e:
                clip.status = "failed"
                clip.error_message = f"Video download failed: {e}"
                scene.status = "failed"
                await session.commit()
                logger.error("Scene %d: ComfyUI download failed: %s", scene.scene_index, e)
                return

            # Save clip to disk
            file_path = file_mgr.save_clip(
                project.id, scene.scene_index, video_bytes,
            )
            clip.local_path = str(file_path)
            clip.status = "complete"
            clip.duration_seconds = duration
            clip.source = "generated"
            scene.status = "video_done"
            await session.commit()

            # Post-generation CV analysis (reuse existing)
            await _run_post_generation_analysis(
                session, scene, clip, project, scene_manifest_row,
            )

            logger.info("Scene %d: ComfyUI video complete", scene.scene_index)
            return

        elif status == "failed":
            clip.status = "failed"
            clip.error_message = error_msg or "ComfyUI job failed"
            scene.status = "failed"
            await session.commit()
            logger.error("Scene %d: ComfyUI job failed: %s", scene.scene_index, error_msg)
            return

        # Still running — sleep and continue
        await session.commit()
        await asyncio.sleep(poll_interval)

        # Check for user-requested stop
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

    # Timed out
    clip.status = "timed_out"
    clip.error_message = (
        f"ComfyUI operation did not complete after {max_polls * poll_interval} seconds"
    )
    scene.status = "timed_out"
    await session.commit()
    logger.error("Scene %d: ComfyUI poll timed out", scene.scene_index)


async def _load_char_ref_images(
    session: AsyncSession, project: Project
) -> list[bytes]:
    """Load up to 2 CHARACTER asset reference images from the manifest.

    Returns a list of image bytes (0-2 items).
    """
    if not project.manifest_id:
        return []

    from vidpipe.db.models import Asset
    result = await session.execute(
        select(Asset).where(
            Asset.manifest_id == project.manifest_id,
            Asset.asset_type == "CHARACTER",
            Asset.reference_image_url != None,
            Asset.is_inherited == False,
        ).order_by(Asset.sort_order).limit(2)
    )
    assets = result.scalars().all()

    char_refs: list[bytes] = []
    for asset in assets:
        image_path = _resolve_asset_image_path(asset)
        if image_path and image_path.exists():
            char_refs.append(image_path.read_bytes())
        else:
            logger.warning(
                f"Character reference image not found for asset {asset.id}: "
                f"url={asset.reference_image_url}"
            )
    return char_refs


def _resolve_asset_image_path(asset) -> Optional[Path]:
    """Resolve an asset's reference image to its on-disk path.

    Asset images live at tmp/manifests/{manifest_id}/{uploads|crops}/{asset_id}_*.
    The reference_image_url is an API route (/api/assets/{id}/image), not a
    filesystem path, so we locate the file using the same logic as the API route.
    """
    manifest_dir = Path("tmp/manifests") / str(asset.manifest_id)
    for subdir in ("uploads", "crops"):
        d = manifest_dir / subdir
        if d.exists():
            matches = list(d.glob(f"{asset.id}_*"))
            if matches:
                return matches[0]
    return None


async def _generate_video_for_scene(
    session: AsyncSession,
    scene: Scene,
    file_mgr: FileManager,
    client,
    project: Project,
    video_model: str,
    cv_service: Optional[CVAnalysisService] = None,
    scoring_service: Optional[CandidateScoringService] = None,
    text_adapter: Optional[LLMAdapter] = None,
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
    start_kf = next((k for k in keyframes if k.position == "start"), None)
    end_kf = next((k for k in keyframes if k.position == "end"), None)
    if start_kf is None or end_kf is None:
        missing = [p for p, kf in [("start", start_kf), ("end", end_kf)] if kf is None]
        raise ValueError(
            f"Scene {scene.scene_index} missing {' and '.join(missing)} "
            f"keyframe(s) — cannot generate video"
        )

    start_frame_bytes = Path(start_kf.file_path).read_bytes()
    end_frame_bytes = Path(end_kf.file_path).read_bytes()

    # Load scene manifest and select references (Phase 8)
    from vidpipe.db.models import SceneManifest as SceneManifestModel
    from vidpipe.services.reference_selection import select_references_for_scene

    # Initialize scene_manifest_row to None so it's accessible from both
    # completion paths (crash recovery resume and escalation loop).
    # Phase 11: Initialize all_assets to [] so quality mode without manifest
    # does not raise NameError when referencing it in _handle_quality_mode_candidates.
    scene_manifest_row = None
    selected_refs = []
    all_assets: list = []
    if project.manifest_id:
        # Query scene manifest
        sm_result = await session.execute(
            select(SceneManifestModel).where(
                SceneManifestModel.project_id == project.id,
                SceneManifestModel.scene_index == scene.scene_index
            )
        )
        scene_manifest_row = sm_result.scalar_one_or_none()

        if scene_manifest_row and scene_manifest_row.manifest_json:
            # Load all manifest assets
            all_assets = await manifest_service.load_manifest_assets(session, project.manifest_id)
            selected_refs = select_references_for_scene(
                scene_manifest_row.manifest_json,
                all_assets
            )

            # Persist selected tags for debugging and UI display
            if selected_refs:
                scene_manifest_row.selected_reference_tags = [r.manifest_tag for r in selected_refs]
                await session.commit()

        logger.info(
            f"Scene {scene.scene_index}: selected {len(selected_refs)} reference(s): "
            f"{[r.manifest_tag for r in selected_refs]}"
        )

    # Phase 10: Adaptive Prompt Rewriting for manifest projects
    base_video_prompt = None  # Will hold rewritten or original prompt
    if project.manifest_id and scene_manifest_row and scene_manifest_row.manifest_json:
        try:
            from vidpipe.services.prompt_rewriter import PromptRewriterService
            from vidpipe.db.models import SceneAudioManifest as SceneAudioManifestModel

            # Load audio manifest
            audio_result = await session.execute(
                select(SceneAudioManifestModel).where(
                    SceneAudioManifestModel.project_id == project.id,
                    SceneAudioManifestModel.scene_index == scene.scene_index
                )
            )
            audio_manifest_row = audio_result.scalar_one_or_none()
            audio_manifest_json = None
            if audio_manifest_row:
                audio_manifest_json = {
                    "dialogue_lines": audio_manifest_row.dialogue_json,
                    "sfx": audio_manifest_row.sfx_json,
                    "ambient": audio_manifest_row.ambient_json,
                    "music": audio_manifest_row.music_json,
                }

            # Load previous scene CV analysis for continuity
            previous_cv = None
            if scene.scene_index > 0:
                prev_sm_result = await session.execute(
                    select(SceneManifestModel).where(
                        SceneManifestModel.project_id == project.id,
                        SceneManifestModel.scene_index == scene.scene_index - 1
                    )
                )
                prev_sm = prev_sm_result.scalar_one_or_none()
                if prev_sm:
                    previous_cv = prev_sm.cv_analysis_json

            # all_assets already loaded above in Phase 8 block
            rewriter = PromptRewriterService(text_adapter=text_adapter)
            result = await rewriter.rewrite_video_prompt(
                scene=scene,
                scene_manifest_json=scene_manifest_row.manifest_json,
                audio_manifest_json=audio_manifest_json,
                placed_assets=all_assets,
                previous_cv_analysis=previous_cv,
                all_assets=all_assets,
            )

            base_video_prompt = result.rewritten_prompt

            # Persist rewritten video prompt
            scene_manifest_row.rewritten_video_prompt = result.rewritten_prompt

            # LLM reference selection overrides Phase 8's deterministic selection
            if result.selected_reference_tags:
                asset_map = {a.manifest_tag: a for a in all_assets}

                # Post-LLM enforcement: ensure placed CHARACTER assets are in refs
                placed_char_tags = {
                    p["asset_tag"]
                    for p in scene_manifest_row.manifest_json.get("placements", [])
                    if "asset_tag" in p
                    and asset_map.get(p["asset_tag"])
                    and asset_map[p["asset_tag"]].asset_type == "CHARACTER"
                    and asset_map[p["asset_tag"]].reference_image_url
                }
                current_tags = list(result.selected_reference_tags)
                missing_chars = placed_char_tags - set(current_tags)
                if missing_chars:
                    enforced = list(missing_chars) + current_tags
                    result.selected_reference_tags = enforced[:3]
                    logger.info(
                        f"Scene {scene.scene_index}: enforced placed CHARACTER refs "
                        f"{missing_chars} → {result.selected_reference_tags}"
                    )

                llm_selected = [
                    asset_map[tag]
                    for tag in result.selected_reference_tags
                    if tag in asset_map
                ]
                if llm_selected:
                    selected_refs = llm_selected
                    scene_manifest_row.selected_reference_tags = result.selected_reference_tags
                    logger.info(
                        f"Scene {scene.scene_index}: LLM override refs: "
                        f"{result.selected_reference_tags} "
                        f"(reason: {result.reference_reasoning})"
                    )

            await session.commit()

            logger.info(
                f"Scene {scene.scene_index}: video prompt rewritten "
                f"({len(result.rewritten_prompt)} chars)"
            )
        except Exception as e:
            # Re-raise PipelineStopped — it must propagate (inherits from Exception)
            from vidpipe.orchestrator.pipeline import PipelineStopped
            if isinstance(e, PipelineStopped):
                raise
            logger.warning(
                f"Scene {scene.scene_index}: video rewriter failed (non-fatal): {e}"
            )
            base_video_prompt = None  # Fall back to original

    # Check if VideoClip already exists (idempotent resume per VGEN-03)
    result = await session.execute(
        select(VideoClip).where(VideoClip.scene_id == scene.id)
    )
    clip = result.scalar_one_or_none()

    # If clip exists and is still polling, resume the poll (crash recovery)
    if clip and clip.status == "polling" and clip.operation_name:
        if project.quality_mode and project.candidate_count > 1:
            # Phase 11: Quality mode crash recovery — use multi-candidate poll
            poll_result, video_bytes_list = await _poll_and_collect_candidates(
                session, clip, client, project, scene, file_mgr, selected_refs,
            )
            if poll_result == "complete":
                await _handle_quality_mode_candidates(
                    session, scene, project, clip, file_mgr,
                    video_bytes_list,
                    scene_manifest_row,
                    base_video_prompt or scene.video_motion_prompt,
                    all_assets,
                    has_refs=bool(selected_refs),
                    scoring_service=scoring_service,
                    cv_service=cv_service,
                )
                return
            elif poll_result != "content_policy":
                return  # failed or timed_out
            # Content policy with zero survivors → fall through to escalation
            logger.warning(
                f"Scene {scene.scene_index}: quality-mode resumed poll hit content policy, "
                "starting remediation"
            )
        else:
            # Standard mode crash recovery: existing behavior
            poll_result = await _poll_video_operation(
                session, clip, client, project, scene, file_mgr, selected_refs,
            )
            if poll_result != "content_policy":
                if poll_result == "complete":
                    # Phase 9: Post-generation CV analysis for progressive enrichment
                    await _run_post_generation_analysis(
                        session, scene, clip, project, scene_manifest_row, cv_service=cv_service,
                    )
                return  # complete, failed, or timed_out
            # Content policy → fall through to escalation loop
            logger.warning(
                f"Scene {scene.scene_index}: resumed poll hit content policy, "
                "starting remediation"
            )

    # Build Veo reference images list (Phase 8)
    from vidpipe.services.reference_selection import resolve_asset_image_bytes
    veo_ref_images = None
    if selected_refs:
        veo_ref_images = []
        for asset in selected_refs:
            ref_bytes = await resolve_asset_image_bytes(session, asset)
            if ref_bytes:
                veo_ref_images.append(
                    types.VideoGenerationReferenceImage(
                        image=types.Image(
                            image_bytes=ref_bytes,
                            mime_type="image/png"
                        ),
                        reference_type=types.VideoGenerationReferenceType.ASSET
                    )
                )

        if not veo_ref_images:
            veo_ref_images = None  # No valid images found

    # ---- Escalating content-policy remediation loop ----
    max_levels = len(_VIDEO_SAFETY_PREFIXES)
    max_transient_retries = settings.pipeline.video_transient_retries
    _pending_safety_regens = 0

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
                _pending_safety_regens += 1
                if clip is not None:
                    clip.safety_regen_count = (clip.safety_regen_count or 0) + 1

        # Build video prompt with escalating safety prefix
        # Phase 10: Use rewritten prompt as base if available (manifest projects)
        # Safety prefix stacks on top; rewritten prompt already includes full formula + audio
        if base_video_prompt:
            video_prompt = (
                f"{_VIDEO_SAFETY_PREFIXES[safety_level]}"
                f"{base_video_prompt}"
            )
        else:
            video_prompt = (
                f"{_VIDEO_SAFETY_PREFIXES[safety_level]}"
                f"{scene.video_motion_prompt}. "
                f"Maintain the visual style shown in the source frames."
            )

        # Phase 11: Determine candidate count for this submission
        candidate_count = project.candidate_count if project.quality_mode else 1

        # ---- Inner transient-error retry loop ----
        for transient_attempt in range(max_transient_retries):
            if transient_attempt > 0:
                # Exponential backoff between transient resubmissions: 15s, 30s, 60s, ...
                backoff_seconds = 15 * (2 ** (transient_attempt - 1))
                logger.info(
                    f"Scene {scene.scene_index}: transient retry "
                    f"{transient_attempt}/{max_transient_retries - 1}, "
                    f"backing off {backoff_seconds}s"
                )
                await asyncio.sleep(backoff_seconds)

                # Check for user-requested stop during backoff
                await session.refresh(project)
                if project.status == "stopped":
                    from vidpipe.orchestrator.pipeline import PipelineStopped
                    raise PipelineStopped("Pipeline stopped by user")

            # Submit job (retries transient 429/5xx automatically)
            try:
                operation = await _submit_video_job(
                    client, video_model, video_prompt,
                    start_frame_bytes, end_frame_bytes, project,
                    reference_images=veo_ref_images,
                    candidate_count=candidate_count,
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
                    break  # break inner, continue outer (escalate)
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
                    veo_submission_count=1,
                    safety_regen_count=_pending_safety_regens,
                )
                session.add(clip)
            else:
                clip.operation_name = operation.name
                clip.status = "polling"
                clip.poll_count = 0
                clip.error_message = None
                clip.veo_submission_count = (clip.veo_submission_count or 0) + 1
            clip.prompt_used = video_prompt
            await session.commit()

            # Phase 11: Choose poll function based on mode
            if project.quality_mode and project.candidate_count > 1:
                poll_result, video_bytes_list = await _poll_and_collect_candidates(
                    session, clip, client, project, scene, file_mgr, selected_refs,
                )
            else:
                poll_result = await _poll_video_operation(
                    session, clip, client, project, scene, file_mgr, selected_refs,
                )
                video_bytes_list = []  # Not used in standard mode

            if poll_result == "complete":
                if safety_level > 0 or transient_attempt > 0:
                    logger.info(
                        f"Scene {scene.scene_index}: succeeded at safety level "
                        f"{safety_level}, transient attempt {transient_attempt}"
                    )
                # Phase 11: Quality mode — save/score/select all candidates
                if project.quality_mode and project.candidate_count > 1:
                    await _handle_quality_mode_candidates(
                        session, scene, project, clip, file_mgr,
                        video_bytes_list,
                        scene_manifest_row,
                        base_video_prompt or scene.video_motion_prompt,
                        all_assets,
                        has_refs=bool(veo_ref_images),
                        scoring_service=scoring_service,
                        cv_service=cv_service,
                    )
                else:
                    # Standard mode: Phase 9 post-generation CV analysis
                    await _run_post_generation_analysis(
                        session, scene, clip, project, scene_manifest_row, cv_service=cv_service,
                    )
                return
            elif poll_result == "content_policy":
                # Reset scene status for next attempt — break to escalate
                scene.status = "keyframes_done"
                await session.commit()
                break  # break inner, continue outer (escalate safety level)
            elif poll_result == "transient":
                # Resubmit at same safety level (inner loop continues)
                scene.status = "keyframes_done"
                await session.commit()
                continue
            else:
                # timed_out or permanent failure — don't escalate or retry
                return
        else:
            # Transient retries exhausted at this safety level — escalate
            # to next level (prompt modification may help avoid the issue)
            logger.warning(
                f"Scene {scene.scene_index}: transient retries exhausted "
                f"({max_transient_retries}) at safety level {safety_level}, escalating"
            )
            scene.status = "keyframes_done"
            await session.commit()
            continue  # next safety level

    # All safety levels exhausted
    logger.error(
        f"Scene {scene.scene_index}: remediation exhausted "
        f"after {max_levels} safety levels"
    )
    if clip:
        clip.status = "failed"
        clip.error_message = (
            "Remediation exhausted after all safety levels and transient retries"
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
