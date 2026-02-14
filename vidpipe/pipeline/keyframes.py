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
from google.genai import types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
)

from vidpipe.config import settings
from vidpipe.db.models import Project, Scene, Keyframe
from vidpipe.services.file_manager import FileManager
from vidpipe.services.vertex_client import get_vertex_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 2),
    retry=retry_if_exception_type(Exception),
)
async def _generate_image_from_text(client, prompt: str, aspect_ratio: str) -> bytes:
    """Generate image from text prompt using Nano Banana Pro.

    Args:
        client: Vertex AI client instance
        prompt: Text description for image generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response

    Note:
        Includes exponential backoff retry with jitter (max 5 attempts).
        Retries on all exceptions to handle transient API failures.
    """
    response = await client.aio.models.generate_content(
        model=settings.models.image_gen,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    # Extract image bytes from response
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise ValueError("No image generated in response")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 2),
    retry=retry_if_exception_type(Exception),
)
async def _generate_image_conditioned(
    client, reference_image_bytes: bytes, prompt: str, aspect_ratio: str
) -> bytes:
    """Generate image using reference image for conditioning.

    Args:
        client: Vertex AI client instance
        reference_image_bytes: PNG image data to use as reference
        prompt: Text description for conditioned generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response

    Note:
        Includes exponential backoff retry with jitter (max 5 attempts).
        Image conditioning maintains visual style and composition.
    """
    response = await client.aio.models.generate_content(
        model=settings.models.image_gen,
        contents=[
            types.Part.from_bytes(data=reference_image_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
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
    # Get client and file manager
    client = get_vertex_client()
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
        # Generate or inherit START frame
        if scene.scene_index == 0:
            # Scene 0: Generate from text prompt (KEYF-01)
            start_frame_bytes = await _generate_image_from_text(
                client, scene.start_frame_prompt, project.aspect_ratio
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
        conditioning_prompt = (
            f"Using this image as reference, show the same scene "
            f"{project.target_clip_duration} seconds later. "
            f"{scene.end_frame_prompt}. "
            f"Maintain visual style, lighting, composition, and character appearance."
        )
        end_frame_bytes = await _generate_image_conditioned(
            client, start_frame_bytes, conditioning_prompt, project.aspect_ratio
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
