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
from vidpipe.services.vertex_client import get_vertex_client


# System prompt for Gemini storyboard generation
STORYBOARD_SYSTEM_PROMPT = """You are a cinematic storyboard director specializing in short-form video content.

Your task is to transform the user's script into a visual storyboard optimized for AI-powered video generation.

REQUIREMENTS:
- Break the script into 3-5 distinct visual scenes
- Each scene should have clear narrative progression
- Provide detailed image prompts for start and end keyframes with:
  * Subject composition and positioning
  * Lighting and atmosphere details
  * Camera angle and framing
  * Visual style elements
- Include motion descriptions that work for video interpolation between keyframes
- Add transition notes to ensure visual continuity between scenes
- Create a comprehensive style guide with:
  * Overall visual aesthetic
  * Color palette and mood
  * Camera movement approach

GOAL: Ensure all scenes maintain visual coherence while telling a compelling story."""


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
    client = get_vertex_client()

    # Build full prompt with system instructions and user script
    full_prompt = f"{STORYBOARD_SYSTEM_PROMPT}\n\nScript: {project.prompt}"

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
            model=settings.models.storyboard_llm,
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
