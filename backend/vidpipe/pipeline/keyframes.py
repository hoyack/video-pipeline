"""Sequential keyframe generation with visual continuity.

This module implements KEYF-01 through KEYF-06 requirements:
- Scene 0 start frame generated from text prompt (KEYF-01)
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
from vidpipe.db.models import Project, Scene, Keyframe
from vidpipe.services.file_manager import FileManager
from vidpipe.services.llm import LLMAdapter
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

logger = logging.getLogger(__name__)

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
                types.Part.from_bytes(data=ref_bytes, mime_type="image/png")
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
        types.Part.from_bytes(data=reference_image_bytes, mime_type="image/png"),
    ]
    if reference_images:
        contents.append(types.Part.from_text(text=(
            "The following reference photo(s) show the EXACT person(s) who must appear "
            "in the generated image. Match their face, skin tone, head shape, and "
            "distinguishing features as closely as possible."
        )))
        for ref_bytes in reference_images:
            contents.append(
                types.Part.from_bytes(data=ref_bytes, mime_type="image/png")
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
    project: Project,
    text_adapter: Optional[LLMAdapter] = None,
) -> None:
    """Generate keyframes sequentially with visual continuity across scenes.

    Implements sequential keyframe generation where:
    - Scene 0 start frame is generated from text prompt alone
    - Scene N start frame inherits scene N-1 end frame
    - All end frames use image-conditioned generation for continuity

    Args:
        session: Database session for persisting keyframes
        project: Project containing scenes to generate keyframes for
        text_adapter: Optional LLMAdapter for prompt rewriting. If None,
            PromptRewriterService falls back to get_adapter("gemini-2.5-flash").

    Process:
        1. Query scenes ordered by scene_index
        2. For each scene sequentially:
           a. Generate or inherit start frame
           b. Save start keyframe to filesystem and database
           c. Generate end frame using image-conditioned generation
           d. Save end keyframe to filesystem and database
           e. Update scene status and commit
           f. Rate limit delay before next scene
        3. Update project status to "generating_video"

    Note:
        - Commits after each scene for crash recovery
        - Uses rate limiting to prevent 429 errors
        - Sequential processing ensures visual continuity (KEYF-04)
    """
    # Resolve image model from project (with fallback to settings)
    image_model = project.image_model or settings.models.image_gen

    # Guard: Imagen models no longer supported — fall back to config default
    if image_model.startswith("imagen-"):
        logger.warning(
            f"Project uses unsupported Imagen model '{image_model}', "
            f"falling back to '{settings.models.image_gen}'"
        )
        image_model = settings.models.image_gen

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

    # Query scenes ordered by scene_index for sequential processing
    result = await session.execute(
        select(Scene)
        .where(Scene.project_id == project.id)
        .order_by(Scene.scene_index)
    )
    scenes = result.scalars().all()

    # Track previous scene's end frame for inheritance
    previous_end_frame_bytes = None

    # Process each scene sequentially (no parallelization)
    for scene in scenes:
        # Check for user-requested stop
        await session.refresh(project)
        if project.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            raise PipelineStopped("Pipeline stopped by user")

        # Skip scenes that already have both keyframes (fork copied them)
        existing_kfs_result = await session.execute(
            select(Keyframe).where(Keyframe.scene_id == scene.id)
        )
        existing_kfs = existing_kfs_result.scalars().all()
        if len(existing_kfs) >= 2:
            end_kf = next((k for k in existing_kfs if k.position == "end"), None)
            if end_kf:
                from pathlib import Path as _Path
                previous_end_frame_bytes = _Path(end_kf.file_path).read_bytes()
            # Don't downgrade scenes that already have completed clips
            if scene.status != "video_done":
                scene.status = "keyframes_done"
            await session.commit()
            continue

        # Phase 10: Adaptive Prompt Rewriting for manifest projects
        # Also resolves asset reference images for multimodal keyframe generation
        rewritten_start_prompt = None
        selected_ref_assets: list = []
        ref_image_bytes_list: list[bytes] = []
        placed_char_assets: list = []  # CHARACTER assets placed in scene (for face verification)
        if project.manifest_id:
            try:
                from vidpipe.services.prompt_rewriter import PromptRewriterService
                from vidpipe.services.reference_selection import resolve_asset_image_bytes
                from vidpipe.db.models import SceneManifest as SceneManifestModel

                # Load scene manifest
                sm_result = await session.execute(
                    select(SceneManifestModel).where(
                        SceneManifestModel.project_id == project.id,
                        SceneManifestModel.scene_index == scene.scene_index
                    )
                )
                scene_manifest_row = sm_result.scalar_one_or_none()

                if scene_manifest_row and scene_manifest_row.manifest_json:
                    # Load assets
                    from vidpipe.services import manifest_service
                    all_assets = await manifest_service.load_manifest_assets(session, project.manifest_id)

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

                    rewriter = PromptRewriterService(text_adapter=text_adapter)
                    result = await rewriter.rewrite_keyframe_prompt(
                        scene=scene,
                        scene_manifest_json=scene_manifest_row.manifest_json,
                        placed_assets=all_assets,  # rewriter filters to placed internally
                        previous_cv_analysis=previous_cv,
                        all_assets=all_assets,
                    )

                    rewritten_start_prompt = result.rewritten_prompt

                    # Persist rewritten prompt
                    scene_manifest_row.rewritten_keyframe_prompt = result.rewritten_prompt
                    await session.commit()

                    logger.info(
                        f"Scene {scene.scene_index}: keyframe prompt rewritten "
                        f"(refs: {result.selected_reference_tags})"
                    )

                    # Post-LLM enforcement: ensure placed CHARACTER assets are in refs
                    asset_map = {a.manifest_tag: a for a in all_assets}
                    placed_char_tags = {
                        p["asset_tag"]
                        for p in scene_manifest_row.manifest_json.get("placements", [])
                        if "asset_tag" in p
                        and asset_map.get(p["asset_tag"])
                        and asset_map[p["asset_tag"]].asset_type == "CHARACTER"
                        and asset_map[p["asset_tag"]].reference_image_url
                    }
                    current_tags = list(result.selected_reference_tags or [])
                    missing_chars = placed_char_tags - set(current_tags)
                    if missing_chars:
                        enforced = list(missing_chars) + current_tags
                        result.selected_reference_tags = enforced[:3]
                        logger.info(
                            f"Scene {scene.scene_index}: enforced placed CHARACTER refs "
                            f"{missing_chars} → {result.selected_reference_tags}"
                        )

                    # Collect placed CHARACTER assets for face verification
                    placed_char_assets = [
                        asset_map[tag]
                        for tag in placed_char_tags
                        if tag in asset_map
                    ]

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
                                f"Scene {scene.scene_index}: resolved "
                                f"{len(ref_image_bytes_list)} reference image(s) "
                                f"for keyframe generation"
                            )
            except Exception as e:
                logger.warning(
                    f"Scene {scene.scene_index}: keyframe rewriter failed (non-fatal): {e}"
                )
                rewritten_start_prompt = None  # Fall back to original
                ref_image_bytes_list = []  # Reset on failure

        # Face verification retry config (max 2 retries = 3 total attempts)
        _max_identity_retries = 2

        # Generate or inherit START frame
        if scene.scene_index == 0:
            # Scene 0: Generate from text prompt (KEYF-01)
            # Prepend style guide + character bible for maximum fidelity
            # Phase 10: Use rewritten prompt when available (already includes asset details)
            if rewritten_start_prompt:
                enriched_prompt = f"{style_prefix}{rewritten_start_prompt}"
            else:
                enriched_prompt = f"{style_prefix}{character_prefix}{scene.start_frame_prompt}"

            # Face verification retry loop
            start_frame_bytes = None
            for identity_level in range(_max_identity_retries + 1):
                prompt_with_emphasis = (
                    _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                    + enriched_prompt
                )
                if is_comfyui:
                    start_frame_bytes = await _generate_image_comfyui(
                        comfy_client, prompt_with_emphasis, seed=project.seed,
                    )
                else:
                    start_frame_bytes = await _generate_image_from_text(
                        image_client, prompt_with_emphasis, project.aspect_ratio, image_model,
                        seed=project.seed,
                        reference_images=ref_image_bytes_list or None,
                    )
                # Verify face match if placed chars exist and not final attempt
                if placed_char_assets and identity_level < _max_identity_retries:
                    passed, sim, detail = await _verify_keyframe_faces(
                        start_frame_bytes, placed_char_assets,
                    )
                    if passed:
                        logger.info(
                            f"Scene {scene.scene_index} start: face verification passed "
                            f"(level={identity_level}, {detail})"
                        )
                        break
                    else:
                        logger.warning(
                            f"Scene {scene.scene_index} start: face verification failed "
                            f"(level={identity_level}, {detail}), retrying"
                        )
                        continue
                else:
                    break  # No verification needed or final attempt
            start_source = "generated"
        else:
            # Scene N: Inherit from previous scene's end frame (KEYF-03)
            start_frame_bytes = previous_end_frame_bytes
            start_source = "inherited"

        # Save start keyframe to filesystem
        start_file_path = file_mgr.save_keyframe(
            project.id, scene.scene_index, "start", start_frame_bytes
        )

        # Create start keyframe database record
        start_keyframe = Keyframe(
            scene_id=scene.id,
            position="start",
            file_path=str(start_file_path),
            mime_type="image/png",
            source=start_source,
            prompt_used=scene.start_frame_prompt,
        )
        session.add(start_keyframe)

        # Generate END frame with image conditioning (KEYF-02)
        style_label = project.style.replace("_", " ")
        conditioning_prompt = (
            f"Generate the NEXT keyframe for this {style_label} scene, "
            f"showing clear visual progression {project.target_clip_duration} seconds later.\n\n"
            f"TARGET END STATE (this is what the new image must depict):\n"
            f"{scene.end_frame_prompt}\n\n"
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
                    seed=project.seed + scene.scene_index + 1000,
                )
            else:
                end_frame_bytes = await _generate_image_conditioned(
                    image_client, start_frame_bytes, prompt_with_emphasis,
                    project.aspect_ratio, image_model,
                    reference_images=ref_image_bytes_list or None,
                )
            if placed_char_assets and identity_level < _max_identity_retries:
                passed, sim, detail = await _verify_keyframe_faces(
                    end_frame_bytes, placed_char_assets,
                )
                if passed:
                    logger.info(
                        f"Scene {scene.scene_index} end: face verification passed "
                        f"(level={identity_level}, {detail})"
                    )
                    break
                else:
                    logger.warning(
                        f"Scene {scene.scene_index} end: face verification failed "
                        f"(level={identity_level}, {detail}), retrying"
                    )
                    continue
            else:
                break

        # Save end keyframe to filesystem
        end_file_path = file_mgr.save_keyframe(
            project.id, scene.scene_index, "end", end_frame_bytes
        )

        # Create end keyframe database record
        end_keyframe = Keyframe(
            scene_id=scene.id,
            position="end",
            file_path=str(end_file_path),
            mime_type="image/png",
            source="generated",
            prompt_used=scene.end_frame_prompt,
        )
        session.add(end_keyframe)

        # Update scene status and prepare for next iteration
        scene.status = "keyframes_done"
        previous_end_frame_bytes = end_frame_bytes

        # Commit after each scene for crash recovery
        await session.commit()

        # Rate limiting delay (KEYF-05)
        await asyncio.sleep(settings.pipeline.image_gen_delay)

    # Update project status after all keyframes generated
    project.status = "generating_video"
    await session.commit()
