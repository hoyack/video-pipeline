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
COMFYUI_IMAGE_MODELS = {"qwen-fast", "qwen-image-edit", "flux-dev", "flux-dev-lora", "flux-dev-redux", "flux-dev-full"}

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
    # Level 2: maximum emphasis — sacrifice composition for face identity
    (
        "ABSOLUTE PRIORITY — FACE IDENTITY: Every facial feature in the generated "
        "image must match the reference photos with photographic accuracy. "
        "The face is the single most important element of this image. Sacrifice "
        "background detail or composition fidelity if needed to preserve exact "
        "face identity. "
    ),
]


def _build_identity_instruction(
    text_description: str | None = None,
    emphasis_level: int = 0,
) -> str:
    """Build feature-anchored identity instruction for Gemini.

    Injects specific facial features from Actor.base_appearance_prompt
    into the identity grounding instruction, improving retention ~41%
    over generic "same person" instructions.

    Args:
        text_description: Actor.base_appearance_prompt facial features text.
        emphasis_level: 0=standard, 1=escalated, 2=maximum.

    Returns:
        Identity instruction string to prepend before reference images.
    """
    features = ""
    if text_description:
        # Truncate very long descriptions to keep prompt focused
        desc = text_description[:300].strip()
        features = f" This person's key features: {desc}"

    if emphasis_level <= 0:
        return (
            "The following reference photo(s) show the EXACT person who must appear "
            "in the generated image." + features + " "
            "Match their face structure, skin tone, and distinguishing features precisely. "
            "The generated character MUST be recognizable as the SAME PERSON.\n\n"
        )
    elif emphasis_level == 1:
        return (
            "CRITICAL IDENTITY REQUIREMENT: The character's face must EXACTLY match "
            "the reference photo(s)." + features + " "
            "Pay extreme attention to facial bone structure, eye shape and spacing, "
            "nose bridge, jawline contour, and skin tone. "
            "The face must be the SAME PERSON, not merely similar.\n\n"
        )
    else:  # Level 2+
        return (
            "ABSOLUTE PRIORITY — FACE IDENTITY: Match the reference photos with "
            "photographic accuracy." + features + " "
            "The face is the most important element. Sacrifice background detail or "
            "composition fidelity if needed to preserve exact face identity.\n\n"
        )


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
    identity_instruction: str | None = None,
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
        identity_instruction: Optional feature-anchored identity prompt
            (from _build_identity_instruction). Falls back to generic prefix.

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    # Build contents: [identity_instruction, ref_image_1, ..., text_prompt]
    # When reference images are present, prepend an identity-matching instruction
    # so Gemini knows these images define the characters' visual appearance.
    contents: list = []
    if reference_images:
        ref_prefix = identity_instruction or (
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
    identity_instruction: str | None = None,
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
        identity_instruction: Optional feature-anchored identity prompt
            (from _build_identity_instruction). Falls back to generic prefix.

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
        ref_prefix = identity_instruction or (
            "The following reference photo(s) show the EXACT person(s) who must appear "
            "in the generated image. Match their face, skin tone, head shape, and "
            "distinguishing features as closely as possible."
        )
        contents.append(types.Part.from_text(text=ref_prefix))
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
    ref_embeddings: list | None = None,
) -> tuple[bool, float, str]:
    """Verify generated keyframe contains faces matching placed CHARACTER assets.

    Uses YOLO face detection + ArcFace embedding comparison.

    Supports two embedding sources:
    - Legacy: Asset.face_embedding bytes from manifest pathway
    - CastBinding: ref_embeddings (numpy arrays) from prequalification service

    Soft degradation — returns (True, ...) when:
    - No embeddings available from either source → "no_embeddings_available"
    - No faces detected in keyframe → "no_faces_detected"
    - CV services fail → "verification_error"

    Args:
        keyframe_bytes: Generated keyframe image bytes
        placed_char_assets: Asset objects for placed CHARACTERs (must have face_embedding)
        threshold: Cosine similarity threshold (default from config)
        ref_embeddings: Pre-computed numpy embeddings from prequalification
            (bridges CastBinding flow where no Asset.face_embedding exists)

    Returns:
        (passed, best_similarity, detail_string)
    """
    if threshold is None:
        threshold = settings.cv_analysis.keyframe_face_match_threshold

    # Collect reference embeddings from both sources
    all_ref_embeddings: list = []

    # Source 1: Legacy Asset.face_embedding bytes
    for a in placed_char_assets:
        if a.face_embedding is not None:
            emb = np.frombuffer(a.face_embedding, dtype=np.float32).copy()
            all_ref_embeddings.append(emb)

    # Source 2: Prequalified ActorRef embeddings (numpy arrays)
    if ref_embeddings:
        all_ref_embeddings.extend(ref_embeddings)

    if not all_ref_embeddings:
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

            # Compare against all reference embeddings
            for ref_emb in all_ref_embeddings:
                sim = FaceMatchingService.cosine_similarity(gen_embedding, ref_emb)
                best_similarity = max(best_similarity, sim)

        passed = best_similarity >= threshold
        detail = (
            f"best_sim={best_similarity:.3f} threshold={threshold:.3f} "
            f"faces_detected={len(faces)} refs_checked={len(all_ref_embeddings)}"
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


async def _generate_image_comfyui_edit(
    comfy_client,
    prompt: str,
    input_image_bytes: bytes,
    seed: int,
) -> bytes:
    """Generate an edited image via ComfyUI Cloud using the Qwen Image Edit workflow.

    Uploads the input image, builds the edit workflow, queues it, polls until
    success, then downloads the output image.

    Args:
        comfy_client: ComfyUIClient instance
        prompt: Edit instruction describing what to change
        input_image_bytes: PNG/JPEG bytes of the image to edit
        seed: Random seed for reproducibility

    Returns:
        PNG image data as bytes

    Raises:
        RuntimeError: On ComfyUI job failure or timeout
        ValueError: If no image output found in history
    """
    from vidpipe.services.comfyui_client import (
        build_qwen_image_edit_workflow,
        find_comfyui_image_output,
    )

    # Upload the input image to ComfyUI
    input_filename = await comfy_client.upload_image(
        input_image_bytes, "image_qwen_image_edit_input_image.png"
    )

    workflow = build_qwen_image_edit_workflow(
        prompt=prompt, input_image_filename=input_filename, seed=seed,
    )
    prompt_id = await comfy_client.queue_prompt(workflow)
    logger.info(f"ComfyUI Qwen Image Edit queued: prompt_id={prompt_id}")

    # Poll until completion
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
    else:
        raise RuntimeError(
            f"ComfyUI job {prompt_id} timed out after {max_polls * poll_interval}s"
        )

    # Fetch history and extract output image
    history = await comfy_client.get_history(prompt_id)
    filename, subfolder = find_comfyui_image_output(history, prompt_id)
    image_bytes = await comfy_client.download_output(filename, subfolder)
    logger.info(
        f"ComfyUI Qwen Image Edit complete: {filename} ({len(image_bytes)} bytes)"
    )
    return image_bytes


async def _generate_image_comfyui_flux(
    comfy_client,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    lora_filename: Optional[str] = None,
    lora_strength: float = 0.8,
    reference_image_filenames: Optional[list[str]] = None,
    reference_strengths: Optional[list[float]] = None,
) -> bytes:
    """Generate an image via ComfyUI Cloud using the Flux.1 Dev txt2img workflow.

    Builds the workflow with optional LoRA and reference images, queues it,
    polls until success, then downloads the output image.

    Args:
        comfy_client: ComfyUIClient instance
        prompt: Text description for image generation
        seed: Random seed for reproducibility
        width: Image width (default 1024)
        height: Image height (default 1024)
        lora_filename: Optional LoRA .safetensors filename on ComfyUI server
        lora_strength: LoRA strength (default 0.8)
        reference_image_filenames: Optional list of uploaded reference image
            filenames on the ComfyUI server (max 3)
        reference_strengths: Optional per-reference conditioning strengths

    Returns:
        PNG image data as bytes

    Raises:
        RuntimeError: On ComfyUI job failure or timeout
        ValueError: If no image output found in history
    """
    from vidpipe.services.comfyui_client import (
        build_flux_txt2img_workflow,
        find_comfyui_image_output,
    )

    workflow = build_flux_txt2img_workflow(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        lora_filename=lora_filename,
        lora_strength=lora_strength,
        reference_image_filenames=reference_image_filenames,
        reference_strengths=reference_strengths,
    )
    prompt_id = await comfy_client.queue_prompt(workflow)
    logger.info(f"ComfyUI Flux txt2img queued: prompt_id={prompt_id}")

    # Poll until completion
    max_polls = 120
    poll_interval = 3
    for attempt in range(max_polls):
        await asyncio.sleep(poll_interval)
        status, error_msg = await comfy_client.poll_status(prompt_id)
        if status == "success":
            break
        if status in ("error", "failed", "cancelled"):
            raise RuntimeError(
                f"ComfyUI Flux job {prompt_id} failed: status={status}, error={error_msg}"
            )
        # Still pending/in_progress — keep polling
    else:
        raise RuntimeError(
            f"ComfyUI Flux job {prompt_id} timed out after {max_polls * poll_interval}s"
        )

    # Fetch history and extract output image
    history = await comfy_client.get_history(prompt_id)
    filename, subfolder = find_comfyui_image_output(history, prompt_id)
    image_bytes = await comfy_client.download_output(filename, subfolder)
    logger.info(
        f"ComfyUI Flux txt2img complete: {filename} ({len(image_bytes)} bytes)"
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
    event_bus.emit(
        scene.id,
        "phase_started",
        phase="keyframes",
        total_shots=len(shots),
        message=f"Starting keyframe generation for {len(shots)} shot(s)",
    )

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
                event_bus.emit(
                    scene.id,
                    "shot_status",
                    shot_index=shot.shot_index,
                    status="generating_start_kf",
                    phase="keyframes",
                    message=f"Shot {shot.shot_index + 1}: generating start keyframe",
                )

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

            # ── CastBinding fallback ────────────────────────────────────
            # When the Production Bible has CastBindings (new flow) but no
            # legacy Assets, load reference images via the tag resolver.
            # This bridges the gap where load_manifest_assets() returns []
            # but ActorRefs exist with images.
            _cast_char_text_description: str | None = None
            _cast_resolved = None  # Tag resolver result (used by prequalification)
            if not ref_image_bytes_list and scene.production_bible_id and not is_comfyui:
                try:
                    from vidpipe.services.tag_resolver import resolve_tags_with_assets

                    # Shot-aware: extract CHARACTER tags from this shot's manifest placements
                    # instead of resolving ALL @tags from scene.prompt for every shot
                    shot_char_tags: set[str] = set()
                    if shot_manifest_row and shot_manifest_row.manifest_json:
                        for p in shot_manifest_row.manifest_json.get("placements", []):
                            tag = p.get("asset_tag")
                            if tag:
                                clean_tag = tag.lstrip("@")
                                if clean_tag and clean_tag not in asset_map:
                                    shot_char_tags.add(clean_tag)

                    if shot_char_tags:
                        # Resolve only THIS shot's CastBinding tags
                        tag_prompt = " ".join(f"@{t}" for t in shot_char_tags)
                    elif hasattr(shot, 'characters_present') and shot.characters_present is not None:
                        # Screenwriter agent populated characters_present —
                        # use it as authoritative source (empty = scenery shot)
                        if shot.characters_present:
                            tag_prompt = " ".join(f"@{t.lstrip('@')}" for t in shot.characters_present)
                        else:
                            tag_prompt = ""  # Scenery shot: no character refs
                    elif scene.prompt:
                        # Fallback: no manifest AND no agent data — use scene.prompt
                        # (pre-agent backward compat)
                        tag_prompt = scene.prompt
                    else:
                        tag_prompt = ""

                    _tag_source = "shot manifest" if shot_char_tags else "scene prompt"

                    if tag_prompt:
                        _cast_resolved = await resolve_tags_with_assets(
                            tag_prompt, scene.production_bible_id, session,
                        )
                        for aref in _cast_resolved.asset_refs:
                            if aref.asset_type == "CHARACTER" and aref.reference_image_urls:
                                _cast_char_text_description = aref.text_description
                                for ref_url in aref.reference_image_urls:
                                    try:
                                        ref_data = await file_mgr.read_bytes(ref_url)
                                        ref_image_bytes_list.append(ref_data)
                                    except Exception:
                                        pass

                    # Fallback: if shot manifest tags yielded no CHARACTER refs
                    # (e.g. LLM typo in tag), retry with scene.prompt which has
                    # the original user-authored tags
                    if not ref_image_bytes_list and shot_char_tags and scene.prompt:
                        logger.warning(
                            f"Shot {shot.shot_index}: shot manifest tags yielded no "
                            f"CHARACTER refs, falling back to scene.prompt"
                        )
                        _cast_resolved = await resolve_tags_with_assets(
                            scene.prompt, scene.production_bible_id, session,
                        )
                        _tag_source = "scene prompt (fallback)"
                        for aref in _cast_resolved.asset_refs:
                            if aref.asset_type == "CHARACTER" and aref.reference_image_urls:
                                _cast_char_text_description = aref.text_description
                                for ref_url in aref.reference_image_urls:
                                    try:
                                        ref_data = await file_mgr.read_bytes(ref_url)
                                        ref_image_bytes_list.append(ref_data)
                                    except Exception:
                                        pass

                    if ref_image_bytes_list:
                        logger.info(
                            f"Shot {shot.shot_index}: CastBinding resolved "
                            f"{len(ref_image_bytes_list)} ref(s) from "
                            f"{_tag_source}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Shot {shot.shot_index}: CastBinding ref resolution failed "
                        f"(non-fatal): {e}"
                    )

            # Face verification retry config (max 3 retries = 4 total attempts)
            # Level 0: standard, Level 1: escalated, Level 2+: iterative refinement
            _max_identity_retries = 3

            # Collect prequalified ref embeddings for face verification
            # (bridges CastBinding flow where Asset.face_embedding is absent)
            _prequalified_ref_embeddings: list = []

            # Extract character appearance description for identity prompt anchoring
            # Prefer legacy Asset.reverse_prompt, fall back to CastBinding text
            _char_text_description: str | None = None
            for ca in placed_char_assets:
                if getattr(ca, "reverse_prompt", None):
                    _char_text_description = ca.reverse_prompt
                    break
            if not _char_text_description:
                _char_text_description = _cast_char_text_description

            if ref_image_bytes_list and not any(
                getattr(a, "face_embedding", None) for a in placed_char_assets
            ):
                # No legacy Asset embeddings — prequalify refs on-the-fly
                try:
                    from vidpipe.services.ref_prequalification import (
                        prequalify_refs, QualifiedRef,
                    )
                    _ref_urls = [
                        u for a in selected_ref_assets
                        for u in ([a.reference_image_url] if getattr(a, "reference_image_url", None) else [])
                    ]
                    # If no selected_ref_assets (CastBinding path), prequalify
                    # directly from ref_image_bytes_list via the resolved URLs
                    if not _ref_urls and _cast_resolved and _cast_resolved.asset_refs:
                        _ref_urls = [
                            url for aref in _cast_resolved.asset_refs
                            if aref.asset_type == "CHARACTER"
                            for url in aref.reference_image_urls
                        ]
                    if _ref_urls:
                        _qualified = await prequalify_refs(_ref_urls, file_mgr)
                        _prequalified_ref_embeddings = [q.face_embedding for q in _qualified]
                        # Use only qualified ref bytes for generation
                        if _qualified:
                            ref_image_bytes_list = [q.image_bytes for q in _qualified]
                            logger.info(
                                f"Shot {shot.shot_index}: prequalified "
                                f"{len(_qualified)} / {len(_ref_urls)} refs"
                            )
                except Exception as e:
                    logger.warning(f"Ref prequalification failed (non-fatal): {e}")

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

                # Face verification retry loop with best-so-far tracking
                start_frame_bytes = None
                best_frame_bytes: bytes | None = None
                best_similarity: float = -1.0
                for identity_level in range(_max_identity_retries + 1):
                    # Build identity instruction with feature anchoring
                    identity_instr = _build_identity_instruction(
                        _char_text_description, emphasis_level=identity_level,
                    ) if ref_image_bytes_list else None

                    prompt_with_emphasis = (
                        _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                        + enriched_prompt
                    )
                    if is_comfyui and image_model.startswith("flux-"):
                        # Flux model: use binding-based reference resolution
                        flux_lora = None
                        flux_ref_filenames: list[str] = []
                        flux_ref_strengths: list[float] = []
                        if scene.production_bible_id:
                            try:
                                from vidpipe.services.tag_resolver import resolve_tags_with_assets
                                resolved = await resolve_tags_with_assets(
                                    prompt_with_emphasis, scene.production_bible_id, session,
                                )
                                # Categorize by asset_type
                                for aref in resolved.asset_refs:
                                    if aref.asset_type == "CHARACTER" and aref.lora_url and not flux_lora:
                                        flux_lora = aref.lora_url
                                    elif aref.reference_image_urls:
                                        # CHARACTER without LoRA, PROP, SET -> reference image
                                        ref_url = aref.reference_image_urls[0]
                                        ref_data = await file_mgr.read_bytes(ref_url)
                                        uploaded_name = await comfy_client.upload_image(
                                            ref_data, f"flux_ref_{aref.tag}.png"
                                        )
                                        flux_ref_filenames.append(uploaded_name)
                                        flux_ref_strengths.append(0.65)
                                if flux_lora or flux_ref_filenames:
                                    logger.info(
                                        f"Shot {shot.shot_index} start: Flux bindings resolved "
                                        f"lora={flux_lora is not None}, refs={len(flux_ref_filenames)}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Shot {shot.shot_index}: Flux binding resolution failed "
                                    f"(non-fatal): {e}"
                                )
                                flux_lora = None
                                flux_ref_filenames = []
                                flux_ref_strengths = []
                        from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS
                        fw, fh = _FLUX_RESOLUTIONS.get(scene.aspect_ratio, (1024, 1024))
                        start_frame_bytes = await _generate_image_comfyui_flux(
                            comfy_client, prompt_with_emphasis, seed=scene.seed,
                            width=fw, height=fh,
                            lora_filename=flux_lora,
                            reference_image_filenames=flux_ref_filenames or None,
                            reference_strengths=flux_ref_strengths or None,
                        )
                    elif is_comfyui:
                        # qwen-image-edit needs an input image; for shot 0 start
                        # there's none, so fall back to qwen-fast txt2img
                        start_frame_bytes = await _generate_image_comfyui(
                            comfy_client, prompt_with_emphasis, seed=scene.seed,
                        )
                    elif identity_level >= 2 and best_frame_bytes is not None:
                        # Level 2+: iterative refinement — feed best attempt as
                        # conditioning frame alongside original refs
                        start_frame_bytes = await _generate_image_conditioned(
                            image_client, best_frame_bytes, prompt_with_emphasis,
                            scene.aspect_ratio, image_model,
                            reference_images=ref_image_bytes_list or None,
                            identity_instruction=identity_instr,
                        )
                    else:
                        start_frame_bytes = await _generate_image_from_text(
                            image_client, prompt_with_emphasis, scene.aspect_ratio, image_model,
                            seed=scene.seed,
                            reference_images=ref_image_bytes_list or None,
                            identity_instruction=identity_instr,
                        )
                    # Verify face match if placed chars exist and not final attempt
                    if placed_char_assets and identity_level < _max_identity_retries:
                        passed, sim, detail = await _verify_keyframe_faces(
                            start_frame_bytes, placed_char_assets,
                            ref_embeddings=_prequalified_ref_embeddings or None,
                        )
                        # Track best attempt
                        if sim > best_similarity:
                            best_similarity = sim
                            best_frame_bytes = start_frame_bytes
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
                # Use best attempt if we tracked one and it's better
                if best_frame_bytes is not None and best_similarity > 0:
                    start_frame_bytes = best_frame_bytes
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
                event_bus.emit(
                    scene.id,
                    "shot_status",
                    shot_index=shot.shot_index,
                    status="generating_end_kf",
                    phase="keyframes",
                    message=f"Shot {shot.shot_index + 1}: generating end keyframe",
                )

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

                # Face verification retry loop for end frame with best-so-far tracking
                end_frame_bytes = None
                best_end_bytes: bytes | None = None
                best_end_sim: float = -1.0
                for identity_level in range(_max_identity_retries + 1):
                    # Build identity instruction with feature anchoring
                    end_identity_instr = _build_identity_instruction(
                        _char_text_description, emphasis_level=identity_level,
                    ) if ref_image_bytes_list else None

                    prompt_with_emphasis = (
                        _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                        + conditioning_prompt
                    )
                    if is_comfyui and image_model.startswith("flux-"):
                        # Flux model: text-only (no image conditioning), offset seed
                        # Reuse binding-resolved LoRA/refs if available from start frame
                        from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS as _FR
                        efw, efh = _FR.get(scene.aspect_ratio, (1024, 1024))
                        end_frame_bytes = await _generate_image_comfyui_flux(
                            comfy_client, prompt_with_emphasis,
                            seed=scene.seed + shot.shot_index + 1000,
                            width=efw, height=efh,
                        )
                    elif is_comfyui:
                        if image_model == "qwen-image-edit":
                            # Use image-edit workflow with start frame as input
                            end_frame_bytes = await _generate_image_comfyui_edit(
                                comfy_client, prompt_with_emphasis,
                                input_image_bytes=start_frame_bytes,
                                seed=scene.seed + shot.shot_index + 1000,
                            )
                        else:
                            # ComfyUI text-only: no image conditioning, use offset seed
                            end_frame_bytes = await _generate_image_comfyui(
                                comfy_client, prompt_with_emphasis,
                                seed=scene.seed + shot.shot_index + 1000,
                            )
                    elif identity_level >= 2 and best_end_bytes is not None:
                        # Level 2+: iterative refinement — feed best end attempt
                        # as conditioning alongside original refs
                        end_frame_bytes = await _generate_image_conditioned(
                            image_client, best_end_bytes, prompt_with_emphasis,
                            scene.aspect_ratio, image_model,
                            reference_images=ref_image_bytes_list or None,
                            identity_instruction=end_identity_instr,
                        )
                    else:
                        end_frame_bytes = await _generate_image_conditioned(
                            image_client, start_frame_bytes, prompt_with_emphasis,
                            scene.aspect_ratio, image_model,
                            reference_images=ref_image_bytes_list or None,
                            identity_instruction=end_identity_instr,
                        )
                    if placed_char_assets and identity_level < _max_identity_retries:
                        passed, sim, detail = await _verify_keyframe_faces(
                            end_frame_bytes, placed_char_assets,
                            ref_embeddings=_prequalified_ref_embeddings or None,
                        )
                        # Track best attempt
                        if sim > best_end_sim:
                            best_end_sim = sim
                            best_end_bytes = end_frame_bytes
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
                # Use best attempt if we tracked one
                if best_end_bytes is not None and best_end_sim > 0:
                    end_frame_bytes = best_end_bytes

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
            event_bus.emit(
                scene.id,
                "shot_keyframe_ready",
                shot_index=shot.shot_index,
                position="end",
                message=f"Shot {shot.shot_index + 1} keyframes ready",
            )
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
    event_bus.emit(
        scene.id,
        "phase_completed",
        phase="keyframes",
        message="Keyframe generation complete",
    )
