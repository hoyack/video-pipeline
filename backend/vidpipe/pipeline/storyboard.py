"""Storyboard generation using Gemini structured output.

Transforms user text prompts into scene-by-scene breakdowns with:
- Keyframe image prompts (start/end)
- Motion descriptions for video interpolation
- Cross-scene style guide for visual consistency

Spec reference: STOR-01 through STOR-05
"""

import json
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from google.genai import types
from vidpipe.config import settings
from vidpipe.db.models import Project, Scene
from vidpipe.schemas.storyboard import StoryboardOutput
from vidpipe.services.vertex_client import get_vertex_client, location_for_model


# System prompt template for Gemini storyboard generation.
# {style} and {aspect_ratio} are filled at runtime.
STORYBOARD_SYSTEM_PROMPT = """You are a storyboard director specializing in short-form video content.

Your task is to transform the user's script into a visual storyboard optimized for AI-powered video generation.

VISUAL STYLE: {style}
This is the MANDATORY visual style. Do NOT fall back to photorealistic or cinematic unless that is the explicitly requested style.

ASPECT RATIO: {aspect_ratio}
Compose all keyframe image prompts for this aspect ratio. For 16:9, favor wide establishing shots and horizontal compositions. For 9:16, favor close-ups, vertical framing, and portrait-oriented compositions.

REQUIREMENTS:
- Break the script into 3-5 distinct visual scenes
- Each scene should have clear narrative progression
- Describe all characters that appear in the video with consistent physical details (see characters field)
- Provide detailed image prompts for start and end keyframes
- Include motion descriptions for video interpolation between keyframes
- Add transition notes to ensure visual continuity between scenes
- Create a comprehensive style guide aligned with {style}

KEYFRAME PROMPT FORMAT (start_frame_prompt and end_frame_prompt):
Each prompt MUST follow this exact structure:
1. MEDIUM DECLARATION (first words): "A {style} rendering of..."
2. SUBJECT: Detailed character description matching the character bible — same face, hair, clothing, proportions every time
3. ACTION/POSE: What the character is doing, body position
4. SETTING: Environment, background elements
5. LIGHTING: Light source direction, quality, mood
6. CAMERA: Shot type (wide/medium/close-up), angle, lens
7. STYLE CUES: Rendering technique details specific to {style} (e.g., line weight, color fills, shading approach, texture)
8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic

VIDEO MOTION PROMPT FORMAT (video_motion_prompt):
Describe ONLY motion and camera movement. Do NOT re-describe characters, setting, or style.
The keyframe images already provide the visual context — the motion prompt controls what moves.
Focus on: camera movement (pan, dolly, track, crane), subject animation, environmental animation.
Good example: "Slow dolly forward as the subject turns to face the camera, hair gently blowing in the breeze"
Bad example: "A blonde woman in anime style turns around in a congressional hearing room" (re-describes visuals)

GOAL: Ensure all scenes maintain visual coherence in {style} style while telling a compelling story."""


async def generate_storyboard(session: AsyncSession, project: Project) -> None:
    """Generate storyboard from project prompt using Gemini structured output.

    Transforms project.prompt into structured storyboard with:
    - StyleGuide stored in project.style_guide
    - Scene records created in database
    - Project status updated to "keyframing"

    Implements retry logic per STOR-05: up to 3 attempts with temperature
    reduction on JSON parse failures.

    Args:
        session: AsyncSession for database operations
        project: Project instance with prompt to transform

    Raises:
        json.JSONDecodeError: If JSON parsing fails after retries
        ValidationError: If Pydantic validation fails after retries
    """
    model_id = project.text_model or settings.models.storyboard_llm
    client = get_vertex_client(location=location_for_model(model_id))

    style_label = project.style.replace("_", " ")

    # Build system prompt with style, aspect ratio, and scene count
    system_prompt = STORYBOARD_SYSTEM_PROMPT.format(
        style=style_label,
        aspect_ratio=project.aspect_ratio,
    ).replace(
        "- Break the script into 3-5 distinct visual scenes",
        f"- Break the script into exactly {project.target_scene_count} distinct visual scenes",
    )

    full_prompt = f"{system_prompt}\n\nScript: {project.prompt}"

    # Retry strategy: up to 3 attempts, reduce temperature on each retry
    attempt = 0
    max_attempts = 3
    base_temperature = 0.7

    @retry(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type((json.JSONDecodeError, ValidationError))
    )
    async def generate_with_retry():
        nonlocal attempt
        # Reduce temperature by 0.15 on each retry
        temperature = base_temperature - (attempt * 0.15)
        attempt += 1

        # Call Gemini with structured output constraint
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StoryboardOutput,
                temperature=max(0.0, temperature)  # Ensure non-negative
            )
        )

        # Parse and validate response
        storyboard = StoryboardOutput.model_validate_json(response.text)
        return storyboard

    # Execute with retry logic
    storyboard = await generate_with_retry()

    # Update project with storyboard data
    project.style_guide = storyboard.style_guide.model_dump()
    project.storyboard_raw = storyboard.model_dump()

    # Create Scene records from storyboard
    for scene_data in storyboard.scenes:
        scene = Scene(
            project_id=project.id,
            scene_index=scene_data.scene_index,
            scene_description=scene_data.scene_description,
            start_frame_prompt=scene_data.start_frame_prompt,
            end_frame_prompt=scene_data.end_frame_prompt,
            video_motion_prompt=scene_data.video_motion_prompt,
            transition_notes=scene_data.transition_notes,
            status="pending"
        )
        session.add(scene)

    # Update project status to indicate storyboard completion
    project.status = "keyframing"

    # Commit all changes
    await session.commit()
