"""Sequential keyframe generation with visual continuity.

This module implements KEYF-01 through KEYF-06 requirements:
- Shot 0 start frame generated from text prompt (KEYF-01)
- End frames use image-conditioned generation (KEYF-02)
- Visual continuity via end-to-start frame inheritance (KEYF-03)
- Sequential processing, no parallelization (KEYF-04)
- Rate limiting with configurable delays (KEYF-05)
- Keyframe images saved as PNG files (KEYF-06)
"""

import asyncio
import logging
from typing import Optional

import numpy as np
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
from vidpipe.db.models import Scene, Shot, Keyframe
from vidpipe.services.file_manager import FileManager
from vidpipe.services.llm import LLMAdapter
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

logger = logging.getLogger(__name__)


def _detect_image_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    return "image/png"  # safe fallback for Gemini


# ---------------------------------------------------------------------------
# ComfyUI image models (routed to ComfyUI instead of Vertex AI)
# ---------------------------------------------------------------------------
COMFYUI_IMAGE_MODELS = {"qwen-fast"}

# ---------------------------------------------------------------------------
# Identity emphasis escalation prefixes for face verification retry
# ---------------------------------------------------------------------------
_IDENTITY_EMPHASIS_PREFIXES = [
    # Level 0: normal generation (no prefix)
    "",
    # Level 1: strong identity-matching instruction
    (
        "CRITICAL: The character's FACE must EXACTLY match the reference photo(s). "
        "Pay close attention to facial bone structure, eye shape, nose bridge, "
        "jawline, and skin tone. The generated face must be recognizable as the "
        "SAME PERSON shown in the reference images. "
    ),
]


def _is_retriable(exc: BaseException) -> bool:
    """Return True only for transient errors worth retrying."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", 0) == 429
    # Retry on connection/timeout errors
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _generate_image_from_text(
    client, prompt: str, aspect_ratio: str, image_model: str,
    seed: int | None = None,
    reference_images: list[bytes] | None = None,
) -> bytes:
    """Generate image from text prompt using Gemini generate_content().

    When reference_images are provided, they are prepended as image parts
    so Gemini can use them for visual identity grounding (face, clothing, etc.).

    Args:
        client: Vertex AI client instance
        prompt: Text description for image generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")
        image_model: Model ID to use for generation
        seed: Optional seed for reproducibility
        reference_images: Optional list of PNG bytes for identity grounding

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    # Build contents: [ref_image_1, ref_image_2, ..., text_prompt]
    # When reference images are present, prepend an identity-matching instruction
    # so Gemini knows these images define the characters' visual appearance.
    contents: list = []
    if reference_images:
        ref_prefix = (
            "The following reference photo(s) show the EXACT person(s) who must appear "
            "in the generated image. Match their face, skin tone, head shape, and "
            "distinguishing features as closely as possible. "
            "These are real reference photos — the generated character MUST look like "
            "the same person, not just a similar description.\n\n"
        )
        contents.append(ref_prefix)
        for ref_bytes in reference_images:
            contents.append(
                types.Part.from_bytes(data=ref_bytes, mime_type=_detect_image_mime(ref_bytes))
            )
    contents.append(prompt)

    response = await client.aio.models.generate_content(
        model=image_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise ValueError("No image generated in response")


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _generate_image_conditioned(
    client,
    reference_image_bytes: bytes,
    prompt: str,
    aspect_ratio: str,
    conditioned_model: str,
    reference_images: list[bytes] | None = None,
) -> bytes:
    """Generate image using conditioning frame + optional asset reference images.

    Contents order: [conditioning_frame, ref_image_1, ..., text_prompt]
    The conditioning frame comes first (strongest weight for visual continuity),
    followed by asset reference images for identity grounding.

    Args:
        client: Vertex AI client instance
        reference_image_bytes: PNG image data from previous frame (conditioning)
        prompt: Text description for conditioned generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")
        conditioned_model: Model ID to use for conditioned generation
        reference_images: Optional list of PNG bytes for identity grounding

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    # Build contents: [conditioning_frame, identity_instruction, ref_images..., text_prompt]
    contents: list = [
        types.Part.from_bytes(data=reference_image_bytes, mime_type=_detect_image_mime(reference_image_bytes)),
    ]
    if reference_images:
        contents.append(types.Part.from_text(text=(
            "The following reference photo(s) show the EXACT person(s) who must appear "
            "in the generated image. Match their face, skin tone, head shape, and "
            "distinguishing features as closely as possible."
        )))
        for ref_bytes in reference_images:
            contents.append(
                types.Part.from_bytes(data=ref_bytes, mime_type=_detect_image_mime(ref_bytes))
            )
    contents.append(types.Part.from_text(text=prompt))

    response = await client.aio.models.generate_content(
        model=conditioned_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # Extract image bytes from response
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise ValueError("No image generated in response")


async def _verify_keyframe_faces(
    keyframe_bytes: bytes,
    placed_char_assets: list,
    threshold: float | None = None,
) -> tuple[bool, float, str]:
    """Verify generated keyframe contains faces matching placed CHARACTER assets.

    Uses YOLO face detection + ArcFace embedding comparison.

    Soft degradation — returns (True, ...) when:
    - No placed chars have face_embedding → "no_embeddings_available"
    - No faces detected in keyframe → "no_faces_detected"
    - CV services fail → "verification_error"

    Args:
        keyframe_bytes: Generated keyframe image bytes
        placed_char_assets: Asset objects for placed CHARACTERs (must have face_embedding)
        threshold: Cosine similarity threshold (default from config)

    Returns:
        (passed, best_similarity, detail_string)
    """
    if threshold is None:
        threshold = settings.cv_analysis.keyframe_face_match_threshold

    # Filter to assets that actually have face embeddings
    assets_with_emb = [
        a for a in placed_char_assets
        if a.face_embedding is not None
    ]
    if not assets_with_emb:
        return True, 0.0, "no_embeddings_available"

    try:
        from vidpipe.services.cv_detection import CVDetectionService
        from vidpipe.services.face_matching import FaceMatchingService

        cv_detector = CVDetectionService()
        face_matcher = FaceMatchingService()

        # Detect faces in generated keyframe
        faces = await asyncio.to_thread(
            cv_detector.detect_faces_from_bytes, keyframe_bytes
        )
        if not faces:
            return True, 0.0, "no_faces_detected"

        # Crop the best face from the keyframe and get its embedding
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(keyframe_bytes)).convert("RGB")

        best_similarity = 0.0

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            # Add 10% padding
            fw, fh = x2 - x1, y2 - y1
            px, py = fw * 0.1, fh * 0.1
            cx1 = max(0, x1 - px)
            cy1 = max(0, y1 - py)
            cx2 = min(img.width, x2 + px)
            cy2 = min(img.height, y2 + py)

            face_crop = img.crop((cx1, cy1, cx2, cy2))

            # Convert crop to bytes for embedding
            buf = io.BytesIO()
            face_crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            try:
                gen_embedding = await asyncio.to_thread(
                    face_matcher.generate_embedding_from_bytes, crop_bytes
                )
            except ValueError:
                continue

            # Compare against each placed CHARACTER's stored embedding
            for asset in assets_with_emb:
                ref_embedding = np.frombuffer(
                    asset.face_embedding, dtype=np.float32
                ).copy()
                sim = FaceMatchingService.cosine_similarity(gen_embedding, ref_embedding)
                best_similarity = max(best_similarity, sim)

        passed = best_similarity >= threshold
        detail = (
            f"best_sim={best_similarity:.3f} threshold={threshold:.3f} "
            f"faces_detected={len(faces)} chars_checked={len(assets_with_emb)}"
        )
        return passed, best_similarity, detail

    except Exception as e:
        logger.warning(f"Face verification error (non-fatal): {e}")
        return True, 0.0, f"verification_error: {e}"


async def _generate_image_comfyui(
    comfy_client,
    prompt: str,
    seed: int,
    width: int = 1328,
    height: int = 1328,
) -> bytes:
    """Generate an image via ComfyUI Cloud using the Qwen txt2img workflow.

    Builds the workflow, queues it, polls until success, then downloads
    the output image.

    Args:
        comfy_client: ComfyUIClient instance
        prompt: Text description for image generation
        seed: Random seed for reproducibility
        width: Image width (default 1328, Qwen native)
        height: Image height (default 1328, Qwen native)

    Returns:
        PNG image data as bytes

    Raises:
        RuntimeError: On ComfyUI job failure or timeout
        ValueError: If no image output found in history
    """
    from vidpipe.services.comfyui_client import (
        build_qwen_txt2img_workflow,
        find_comfyui_image_output,
    )

    workflow = build_qwen_txt2img_workflow(
        prompt=prompt, width=width, height=height, seed=seed,
    )
    prompt_id = await comfy_client.queue_prompt(workflow)
    logger.info(f"ComfyUI Qwen txt2img queued: prompt_id={prompt_id}")

    # Poll until completion (check for "success", not "completed")
    max_polls = 120
    poll_interval = 3
    for attempt in range(max_polls):
        await asyncio.sleep(poll_interval)
        status, error_msg = await comfy_client.poll_status(prompt_id)
        if status == "success":
            break
        if status in ("error", "failed", "cancelled"):
            raise RuntimeError(
                f"ComfyUI job {prompt_id} failed: status={status}, error={error_msg}"
            )
        # Still pending/in_progress — keep polling
    else:
        raise RuntimeError(
            f"ComfyUI job {prompt_id} timed out after {max_polls * poll_interval}s"
        )

    # Fetch history and extract output image
    history = await comfy_client.get_history(prompt_id)
    filename, subfolder = find_comfyui_image_output(history, prompt_id)
    image_bytes = await comfy_client.download_output(filename, subfolder)
    logger.info(
        f"ComfyUI Qwen txt2img complete: {filename} ({len(image_bytes)} bytes)"
    )
    return image_bytes


async def generate_keyframes(
    session: AsyncSession,
    scene: Scene,
    text_adapter: Optional[LLMAdapter] = None,
) -> None:
    """Generate keyframes sequentially with visual continuity across shots.

    Implements sequential keyframe generation where:
    - Shot 0 start frame is generated from text prompt alone
    - Shot N start frame inherits shot N-1 end frame
    - All end frames use image-conditioned generation for continuity

    Args:
        session: Database session for persisting keyframes
        scene: Scene containing shots to generate keyframes for
        text_adapter: Optional LLMAdapter for prompt rewriting. If None,
            PromptRewriterService falls back to get_adapter("gemini-2.5-flash").

    Process:
        1. Query shots ordered by shot_index
        2. For each shot sequentially:
           a. Generate or inherit start frame
           b. Save start keyframe to filesystem and database
           c. Generate end frame using image-conditioned generation
           d. Save end keyframe to filesystem and database
           e. Update shot status and commit
           f. Rate limit delay before next shot
        3. Update scene status to "generating_video"

    Note:
        - Commits after each shot for crash recovery
        - Uses rate limiting to prevent 429 errors
        - Sequential processing ensures visual continuity (KEYF-04)
    """
    # Resolve image model from scene (with fallback to settings)
    image_model = scene.image_model or settings.models.image_gen

    # Guard: Imagen models no longer supported — fall back to config default
    if image_model.startswith("imagen-"):
        logger.warning(
            f"Scene uses unsupported Imagen model '{image_model}', "
            f"falling back to '{settings.models.image_gen}'"
        )
        image_model = settings.models.image_gen

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

    # Route to ComfyUI or Vertex AI based on model
    is_comfyui = image_model in COMFYUI_IMAGE_MODELS
    comfy_client = None
    image_client = None
    if is_comfyui:
        from vidpipe.services.comfyui_client import get_comfyui_client
        comfy_client = await get_comfyui_client()
    else:
        image_client = get_vertex_client(location=location_for_model(image_model))
    file_mgr = FileManager()

    # Query shots ordered by shot_index for sequential processing
    result = await session.execute(
        select(Shot)
        .where(Shot.scene_id == scene.id)
        .order_by(Shot.shot_index)
    )
    shots = result.scalars().all()

    # Track previous shot's end frame for inheritance
    previous_end_frame_bytes = None

    from vidpipe.services.event_bus import event_bus
    event_bus.emit(scene.id, "phase_started", phase="keyframes", total_shots=len(shots))

    # Process each shot sequentially (no parallelization)
    for shot in shots:
        # Per-shot stop flag check (VGED-11)
        await session.refresh(scene)
        if scene.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            logger.info(f"Pipeline stopped by user at shot {shot.shot_index}")
            raise PipelineStopped("Stopped by user")

        # Gap-filling: check for existing keyframes per position (VGED-06)
        existing_kfs_result = await session.execute(
            select(Keyframe).where(Keyframe.shot_id == shot.id)
        )
        existing_kfs = existing_kfs_result.scalars().all()
        existing_start_kf = next((k for k in existing_kfs if k.position == "start"), None)
        existing_end_kf = next((k for k in existing_kfs if k.position == "end"), None)

        # If both keyframes exist, skip entire shot (fork or user upload)
        if existing_start_kf and existing_end_kf:
            previous_end_frame_bytes = await file_mgr.read_bytes(existing_end_kf.file_path)
            # Don't downgrade shots that already have completed clips
            if shot.status != "video_done":
                shot.status = "keyframes_done"
            shot.generation_status = None
            await session.commit()
            logger.info(
                f"Shot {shot.shot_index}: both keyframes exist, skipping"
            )
            continue

        # KEYF-03 continuity: if shot has an uploaded start keyframe,
        # use it as previous_end for daisy-chain conditioning
        if existing_start_kf:
            # The uploaded start keyframe serves as "previous end" for conditioning
            _existing_start_bytes = await file_mgr.read_bytes(existing_start_kf.file_path)

        try:
            # Set generation_status for the shot (VGED-05)
            if not existing_start_kf:
                shot.generation_status = "generating_start_kf"
                await session.commit()
                event_bus.emit(scene.id, "shot_status", shot_index=shot.shot_index, status="generating_start_kf", phase="keyframes")

            # Phase 10: Adaptive Prompt Rewriting for manifest scenes
            # Also resolves asset reference images for multimodal keyframe generation
            rewritten_start_prompt = None
            selected_ref_assets: list = []
            ref_image_bytes_list: list[bytes] = []
            placed_char_assets: list = []  # CHARACTER assets placed in shot (for face verification)
            if scene.production_bible_id:
                try:
                    from vidpipe.services.prompt_rewriter import PromptRewriterService
                    from vidpipe.services.reference_selection import resolve_asset_image_bytes
                    from vidpipe.db.models import ShotManifest as ShotManifestModel

                    # Load shot manifest
                    sm_result = await session.execute(
                        select(ShotManifestModel).where(
                            ShotManifestModel.scene_id == scene.id,
                            ShotManifestModel.shot_index == shot.shot_index
                        )
                    )
                    shot_manifest_row = sm_result.scalar_one_or_none()

                    if shot_manifest_row and shot_manifest_row.manifest_json:
                        # Load assets
                        from vidpipe.services import manifest_service
                        all_assets = await manifest_service.load_manifest_assets(session, scene.production_bible_id)

                        # Load previous shot CV analysis for continuity
                        previous_cv = None
                        if shot.shot_index > 0:
                            prev_sm_result = await session.execute(
                                select(ShotManifestModel).where(
                                    ShotManifestModel.scene_id == scene.id,
                                    ShotManifestModel.shot_index == shot.shot_index - 1
                                )
                            )
                            prev_sm = prev_sm_result.scalar_one_or_none()
                            if prev_sm:
                                previous_cv = prev_sm.cv_analysis_json

                        rewriter = PromptRewriterService(text_adapter=text_adapter)
                        result = await rewriter.rewrite_keyframe_prompt(
                            shot=shot,
                            shot_manifest_json=shot_manifest_row.manifest_json,
                            placed_assets=all_assets,  # rewriter filters to placed internally
                            previous_cv_analysis=previous_cv,
                            all_assets=all_assets,
                        )

                        rewritten_start_prompt = result.rewritten_prompt

                        # Persist rewritten prompt and selected reference tags
                        shot_manifest_row.rewritten_keyframe_prompt = result.rewritten_prompt
                        shot_manifest_row.selected_reference_tags = result.selected_reference_tags
                        await session.commit()

                        logger.info(
                            f"Shot {shot.shot_index}: keyframe prompt rewritten "
                            f"(refs: {result.selected_reference_tags})"
                        )

                        # Post-LLM enforcement: ensure placed CHARACTER assets are in refs
                        asset_map = {a.manifest_tag: a for a in all_assets}
                        asset_map_by_id = {str(a.id): a for a in all_assets}
                        placed_char_tags = {
                            p["asset_tag"]
                            for p in shot_manifest_row.manifest_json.get("placements", [])
                            if "asset_tag" in p
                            and asset_map.get(p["asset_tag"])
                            and asset_map[p["asset_tag"]].asset_type == "CHARACTER"
                            and asset_map[p["asset_tag"]].reference_image_url
                        }

                        # Fix 4: Fallback — if shot has placements but none resolved
                        # to manifest characters, use ALL manifest CHARACTER assets
                        # with reference images. This guarantees reference images reach
                        # the image adapter even when storyboard tags were wrong.
                        if not placed_char_tags and scene.production_bible_id:
                            placed_char_tags = {
                                a.manifest_tag
                                for a in all_assets
                                if a.asset_type == "CHARACTER" and a.reference_image_url
                            }
                            if placed_char_tags:
                                logger.warning(
                                    f"Shot {shot.shot_index}: no placed chars resolved "
                                    f"from shot manifest, falling back to all manifest "
                                    f"CHARACTER assets: {placed_char_tags}"
                                )
                        current_tags = list(result.selected_reference_tags or [])
                        missing_chars = placed_char_tags - set(current_tags)
                        if missing_chars:
                            enforced = list(missing_chars) + current_tags
                            result.selected_reference_tags = enforced[:3]
                            logger.info(
                                f"Shot {shot.shot_index}: enforced placed CHARACTER refs "
                                f"{missing_chars} → {result.selected_reference_tags}"
                            )

                        # Update with enforced tags
                        shot_manifest_row.selected_reference_tags = result.selected_reference_tags
                        await session.commit()

                        # Collect placed CHARACTER assets for face verification
                        placed_char_assets = [
                            asset_map[tag]
                            for tag in placed_char_tags
                            if tag in asset_map
                        ]

                        # Fallback: if a placed CHARACTER has no face_embedding,
                        # try to borrow from its source (parent) asset
                        for asset in placed_char_assets:
                            if asset.face_embedding is None and asset.source_asset_id:
                                parent = asset_map_by_id.get(str(asset.source_asset_id))
                                if parent and parent.face_embedding:
                                    asset.face_embedding = parent.face_embedding
                                    logger.info(
                                        f"Shot {shot.shot_index}: borrowed face embedding "
                                        f"from parent {parent.manifest_tag} for {asset.manifest_tag}"
                                    )

                        # Resolve selected reference tags → asset image bytes
                        if result.selected_reference_tags:
                            for tag in result.selected_reference_tags:
                                asset = asset_map.get(tag)
                                if asset:
                                    ref_bytes = await resolve_asset_image_bytes(session, asset)
                                    if ref_bytes:
                                        ref_image_bytes_list.append(ref_bytes)
                                        selected_ref_assets.append(asset)
                            if ref_image_bytes_list:
                                logger.info(
                                    f"Shot {shot.shot_index}: resolved "
                                    f"{len(ref_image_bytes_list)} reference image(s) "
                                    f"for keyframe generation"
                                )
                except Exception as e:
                    logger.warning(
                        f"Shot {shot.shot_index}: keyframe rewriter failed (non-fatal): {e}"
                    )
                    rewritten_start_prompt = None  # Fall back to original
                    ref_image_bytes_list = []  # Reset on failure

            # Face verification retry config (max 2 retries = 3 total attempts)
            _max_identity_retries = 2

            # ---- START FRAME: Generate or inherit (skip if existing) ----
            if existing_start_kf:
                # Gap-filling: start keyframe already exists (user upload or fork)
                start_frame_bytes = await file_mgr.read_bytes(existing_start_kf.file_path)
                start_source = existing_start_kf.source
                logger.info(
                    f"Shot {shot.shot_index}: start keyframe exists, skipping generation"
                )
            elif shot.shot_index == 0:
                # Shot 0: Generate from text prompt (KEYF-01)
                # Prepend style guide + character bible for maximum fidelity
                # Phase 10: Use rewritten prompt when available (already includes asset details)
                if rewritten_start_prompt:
                    enriched_prompt = f"{style_prefix}{rewritten_start_prompt}"
                else:
                    enriched_prompt = f"{style_prefix}{character_prefix}{shot.start_frame_prompt}"

                # Face verification retry loop
                start_frame_bytes = None
                for identity_level in range(_max_identity_retries + 1):
                    prompt_with_emphasis = (
                        _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                        + enriched_prompt
                    )
                    if is_comfyui:
                        start_frame_bytes = await _generate_image_comfyui(
                            comfy_client, prompt_with_emphasis, seed=scene.seed,
                        )
                    else:
                        start_frame_bytes = await _generate_image_from_text(
                            image_client, prompt_with_emphasis, scene.aspect_ratio, image_model,
                            seed=scene.seed,
                            reference_images=ref_image_bytes_list or None,
                        )
                    # Verify face match if placed chars exist and not final attempt
                    if placed_char_assets and identity_level < _max_identity_retries:
                        passed, sim, detail = await _verify_keyframe_faces(
                            start_frame_bytes, placed_char_assets,
                        )
                        if passed:
                            logger.info(
                                f"Shot {shot.shot_index} start: face verification passed "
                                f"(level={identity_level}, {detail})"
                            )
                            break
                        else:
                            logger.warning(
                                f"Shot {shot.shot_index} start: face verification failed "
                                f"(level={identity_level}, {detail}), retrying"
                            )
                            continue
                    else:
                        break  # No verification needed or final attempt
                start_source = "generated"

                # Save start keyframe
                start_stored_path = await file_mgr.save_keyframe_async(
                    scene.id, shot.shot_index, "start", start_frame_bytes
                )

                # Create start keyframe database record
                start_keyframe = Keyframe(
                    shot_id=shot.id,
                    position="start",
                    file_path=start_stored_path,
                    mime_type="image/png",
                    source=start_source,
                    prompt_used=shot.start_frame_prompt,
                )
                session.add(start_keyframe)
            else:
                # Shot N: Inherit from previous shot's end frame (KEYF-03)
                start_frame_bytes = previous_end_frame_bytes
                start_source = "inherited"

                # Save start keyframe
                start_stored_path = await file_mgr.save_keyframe_async(
                    scene.id, shot.shot_index, "start", start_frame_bytes
                )

                # Create start keyframe database record
                start_keyframe = Keyframe(
                    shot_id=shot.id,
                    position="start",
                    file_path=start_stored_path,
                    mime_type="image/png",
                    source=start_source,
                    prompt_used=shot.start_frame_prompt,
                )
                session.add(start_keyframe)

            # ---- END FRAME: Generate with conditioning (skip if existing) ----
            if existing_end_kf:
                # Gap-filling: end keyframe already exists (user upload or fork)
                end_frame_bytes = await file_mgr.read_bytes(existing_end_kf.file_path)
                logger.info(
                    f"Shot {shot.shot_index}: end keyframe exists, skipping generation"
                )
            else:
                # Update generation_status for end frame (VGED-05)
                shot.generation_status = "generating_end_kf"
                await session.commit()
                event_bus.emit(scene.id, "shot_status", shot_index=shot.shot_index, status="generating_end_kf", phase="keyframes")

                style_label = scene.style.replace("_", " ")
                conditioning_prompt = (
                    f"Generate the NEXT keyframe for this {style_label} shot, "
                    f"showing clear visual progression {scene.target_clip_duration} seconds later.\n\n"
                    f"TARGET END STATE (this is what the new image must depict):\n"
                    f"{shot.end_frame_prompt}\n\n"
                    f"The new image MUST show VISIBLE CHANGES from the reference image — "
                    f"different pose, expression, body position, or camera framing. "
                    f"If the reference is a close-up, the new image should show "
                    f"a noticeably different expression, head angle, or gesture.\n\n"
                    f"CONSISTENCY CONSTRAINTS:\n"
                    f"- Same character appearance (face, hair, clothing, proportions)\n"
                    f"- Same {style_label} rendering style\n"
                    f"{character_prefix}"
                )

                # Face verification retry loop for end frame
                end_frame_bytes = None
                for identity_level in range(_max_identity_retries + 1):
                    prompt_with_emphasis = (
                        _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                        + conditioning_prompt
                    )
                    if is_comfyui:
                        # ComfyUI text-only: no image conditioning, use offset seed
                        end_frame_bytes = await _generate_image_comfyui(
                            comfy_client, prompt_with_emphasis,
                            seed=scene.seed + shot.shot_index + 1000,
                        )
                    else:
                        end_frame_bytes = await _generate_image_conditioned(
                            image_client, start_frame_bytes, prompt_with_emphasis,
                            scene.aspect_ratio, image_model,
                            reference_images=ref_image_bytes_list or None,
                        )
                    if placed_char_assets and identity_level < _max_identity_retries:
                        passed, sim, detail = await _verify_keyframe_faces(
                            end_frame_bytes, placed_char_assets,
                        )
                        if passed:
                            logger.info(
                                f"Shot {shot.shot_index} end: face verification passed "
                                f"(level={identity_level}, {detail})"
                            )
                            break
                        else:
                            logger.warning(
                                f"Shot {shot.shot_index} end: face verification failed "
                                f"(level={identity_level}, {detail}), retrying"
                            )
                            continue
                    else:
                        break

                # Save end keyframe
                end_stored_path = await file_mgr.save_keyframe_async(
                    scene.id, shot.shot_index, "end", end_frame_bytes
                )

                # Create end keyframe database record
                end_keyframe = Keyframe(
                    shot_id=shot.id,
                    position="end",
                    file_path=end_stored_path,
                    mime_type="image/png",
                    source="generated",
                    prompt_used=shot.end_frame_prompt,
                )
                session.add(end_keyframe)

            # Update shot status and prepare for next iteration
            shot.status = "keyframes_done"
            shot.generation_status = None  # Clear generation_status (VGED-05)
            previous_end_frame_bytes = end_frame_bytes

            # Commit after each shot for crash recovery
            await session.commit()
            event_bus.emit(scene.id, "shot_keyframe_ready", shot_index=shot.shot_index, position="end")
            event_bus.emit(scene.id, "refresh")

            # Rate limiting delay (KEYF-05)
            await asyncio.sleep(settings.pipeline.image_gen_delay)

        except Exception as e:
            # On exception: set generation_status to "failed" (VGED-05)
            shot.generation_status = "failed"
            await session.commit()
            raise

    # Update scene status after all keyframes generated
    scene.status = "generating_video"
    await session.commit()
    event_bus.emit(scene.id, "phase_completed", phase="keyframes")
