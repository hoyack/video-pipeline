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
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

logger = logging.getLogger(__name__)


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
) -> bytes:
    """Generate image from text prompt using Imagen or Gemini.

    Imagen models use the generate_images() API.
    Gemini image models use generate_content() with response_modalities=["IMAGE"].

    Args:
        client: Vertex AI client instance
        prompt: Text description for image generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")
        image_model: Model ID to use for generation

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    if image_model.startswith("imagen-"):
        # Imagen models → generate_images() API
        imagen_config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            output_mime_type="image/png",
        )
        if seed is not None:
            imagen_config.seed = seed
            imagen_config.add_watermark = False
        response = await client.aio.models.generate_images(
            model=image_model,
            prompt=prompt,
            config=imagen_config,
        )

        if response.generated_images:
            return response.generated_images[0].image.image_bytes

        raise ValueError("No image generated in response")
    else:
        # Gemini image models → generate_content() with image output
        response = await client.aio.models.generate_content(
            model=image_model,
            contents=[prompt],
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
) -> bytes:
    """Generate image using reference image for conditioning.

    Uses Gemini model with multimodal input (reference image + text) and
    image output via response_modalities. Imagen's generate_images API
    doesn't support reference image conditioning, so we use Gemini which
    natively handles multimodal understanding + image generation.

    Args:
        client: Vertex AI client instance
        reference_image_bytes: PNG image data to use as reference
        prompt: Text description for conditioned generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")
        conditioned_model: Model ID to use for conditioned generation

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    response = await client.aio.models.generate_content(
        model=conditioned_model,
        contents=[
            types.Part.from_bytes(data=reference_image_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # Extract image bytes from response
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise ValueError("No image generated in response")


async def generate_keyframes(session: AsyncSession, project: Project) -> None:
    """Generate keyframes sequentially with visual continuity across scenes.

    Implements sequential keyframe generation where:
    - Scene 0 start frame is generated from text prompt alone
    - Scene N start frame inherits scene N-1 end frame
    - All end frames use image-conditioned generation for continuity

    Args:
        session: Database session for persisting keyframes
        project: Project containing scenes to generate keyframes for

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
    # Resolve image models from project (with fallback to settings)
    image_model = project.image_model or settings.models.image_gen

    # Auto-pair the conditioned model
    from vidpipe.api.routes import IMAGE_CONDITIONED_MAP
    conditioned_model = IMAGE_CONDITIONED_MAP.get(
        image_model, settings.models.image_conditioned
    )

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

    # Get location-aware clients for each model
    image_client = get_vertex_client(location=location_for_model(image_model))
    conditioned_client = get_vertex_client(location=location_for_model(conditioned_model))
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
        rewritten_start_prompt = None
        if project.manifest_id:
            try:
                from vidpipe.services.prompt_rewriter import PromptRewriterService
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

                    rewriter = PromptRewriterService()
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
            except Exception as e:
                logger.warning(
                    f"Scene {scene.scene_index}: keyframe rewriter failed (non-fatal): {e}"
                )
                rewritten_start_prompt = None  # Fall back to original

        # Generate or inherit START frame
        if scene.scene_index == 0:
            # Scene 0: Generate from text prompt (KEYF-01)
            # Prepend style guide + character bible for maximum fidelity
            # Phase 10: Use rewritten prompt when available (already includes asset details)
            if rewritten_start_prompt:
                # Rewriter already injected asset reverse_prompts; omit character_prefix
                # to avoid double-injection. Keep style_prefix for model-level consistency.
                enriched_prompt = f"{style_prefix}{rewritten_start_prompt}"
            else:
                enriched_prompt = f"{style_prefix}{character_prefix}{scene.start_frame_prompt}"
            start_frame_bytes = await _generate_image_from_text(
                image_client, enriched_prompt, project.aspect_ratio, image_model,
                seed=project.seed,
            )
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
        end_frame_bytes = await _generate_image_conditioned(
            conditioned_client, start_frame_bytes, conditioning_prompt, project.aspect_ratio,
            conditioned_model,
        )

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
