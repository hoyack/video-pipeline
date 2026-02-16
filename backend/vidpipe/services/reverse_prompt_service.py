"""Reverse-prompting service using Gemini vision API.

This module provides AI-powered analysis of asset images to generate
recreation prompts and visual descriptions suitable for video generation.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai.types import GenerateContentConfig, Part
from tenacity import retry, stop_after_attempt, wait_exponential

from vidpipe.services.vertex_client import get_vertex_client

logger = logging.getLogger(__name__)


class ReversePromptService:
    """Gemini vision-based reverse-prompting service."""

    def __init__(self, client: Optional[genai.Client] = None):
        """Initialize service with optional client.

        Args:
            client: Optional genai.Client. If None, gets default via get_vertex_client()
        """
        self._client = client

    @property
    def client(self) -> genai.Client:
        """Lazy-load client on first use."""
        if self._client is None:
            self._client = get_vertex_client()
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Reverse-prompt retry {retry_state.attempt_number}/3: {retry_state.outcome.exception()}"
        ),
    )
    async def reverse_prompt_asset(
        self, image_path: str, asset_type: str, user_name: str = ""
    ) -> dict:
        """Generate reverse_prompt and visual_description for asset crop.

        Args:
            image_path: Path to asset image
            asset_type: Asset type (CHARACTER, OBJECT, ENVIRONMENT, etc.)
            user_name: Optional user-provided name

        Returns:
            {
                "reverse_prompt": str,  # Prompt-style recreation description
                "visual_description": str,  # Production bible entry
                "quality_score": float,  # 1-10
                "suggested_name": str  # If user_name empty
            }
        """
        # Read image and detect mime type
        img_path = Path(image_path)
        image_bytes = img_path.read_bytes()
        suffix = img_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        # Get type-specific system prompt
        system_prompt = self._get_system_prompt(asset_type)

        # User context
        user_context = (
            f"User-provided name: {user_name}" if user_name else "No name provided."
        )

        # Call Gemini vision API
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Part.from_bytes(data=image_bytes, mime_type=mime_type),
                f"{system_prompt}\n\n{user_context}",
            ],
            config=GenerateContentConfig(
                temperature=0.4,  # Lower temp for consistency
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "reverse_prompt": {"type": "string"},
                        "visual_description": {"type": "string"},
                        "quality_score": {"type": "number"},
                        "suggested_name": {"type": "string"},
                    },
                    "required": [
                        "reverse_prompt",
                        "visual_description",
                        "quality_score",
                    ],
                },
            ),
        )

        return json.loads(response.text)

    def _get_system_prompt(self, asset_type: str) -> str:
        """Return type-specific system prompt for reverse-prompting.

        Args:
            asset_type: Asset type (CHARACTER, OBJECT, ENVIRONMENT, etc.)

        Returns:
            System prompt tailored to the asset type
        """
        CHARACTER_PROMPT = """You are a visual prompt engineer for AI video generation.
Analyze this CHARACTER image and produce a JSON response with:

1. "reverse_prompt": Write a detailed prompt that would recreate this character in an AI image/video generator. Include: physical build, skin tone, hair (color, style, length), facial features (eye color/shape, nose, jaw, facial hair), expression, clothing (every garment with color and material), accessories, pose, lighting on the subject, and camera angle. Be specific enough that the generated result would be recognizable as this person. Write in prompt style, not prose. ~100-150 words.

2. "visual_description": Narrative description for a production bible. What is distinctive/signature about this character? What must stay consistent across scenes? What is variable (removable accessories, changeable expressions)? ~50-80 words.

3. "quality_score": Rate 1-10 how suitable this image is as a reference for AI generation (clear, well-lit, good angle, unoccluded = higher score).

4. "suggested_name": If no user name provided, suggest one based on appearance."""

        OBJECT_PROMPT = """You are a visual prompt engineer for AI video generation.
Analyze this OBJECT/PROP image and produce a JSON response with:

1. "reverse_prompt": Detailed prompt to recreate this object. Include: shape, material, color, texture, condition (new/worn), scale/size indicators, any text/branding, distinguishing features, lighting, background context. ~80-120 words.

2. "visual_description": What makes this object notable? Key features for consistency? ~40-60 words.

3. "quality_score": 1-10 suitability as AI generation reference.

4. "suggested_name": Descriptive name if none provided."""

        ENVIRONMENT_PROMPT = """You are a visual prompt engineer for AI video generation.
Analyze this ENVIRONMENT image and produce a JSON response with:

1. "reverse_prompt": Detailed prompt to recreate this setting. Include: location type, architecture/layout, lighting (time of day, sources, mood), weather, depth/perspective, key landmarks, color palette, atmosphere, any people/objects present, camera framing. ~120-180 words.

2. "visual_description": Setting characteristics. What defines this space? Mood/atmosphere? ~60-100 words.

3. "quality_score": 1-10 suitability as AI generation reference.

4. "suggested_name": Location name if none provided."""

        prompts = {
            "CHARACTER": CHARACTER_PROMPT,
            "OBJECT": OBJECT_PROMPT,
            "PROP": OBJECT_PROMPT,
            "ENVIRONMENT": ENVIRONMENT_PROMPT,
            "VEHICLE": OBJECT_PROMPT.replace("OBJECT/PROP", "VEHICLE"),
            "STYLE": ENVIRONMENT_PROMPT.replace("ENVIRONMENT", "STYLE REFERENCE"),
            "OTHER": OBJECT_PROMPT,
        }

        return prompts.get(asset_type, OBJECT_PROMPT)
