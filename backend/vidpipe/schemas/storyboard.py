"""Pydantic schemas for storyboard structured output from Gemini.

These schemas define the expected structure for LLM-generated storyboards,
enabling structured output constraints via response_schema parameter.

Spec reference: Section 5.1 (Storyboard Output Schema)
"""

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_to_str(v: Any) -> str:
    """Coerce list/non-str values to comma-separated string.

    Some LLM providers (e.g. Ollama) return arrays for fields declared as
    string in the JSON schema.  This validator normalises them so Pydantic
    validation succeeds regardless of provider quirks.
    """
    if isinstance(v, list):
        return ", ".join(str(item) for item in v)
    return v


CoercedStr = Annotated[str, BeforeValidator(_coerce_to_str)]


class StyleGuide(BaseModel):
    """Cross-scene visual consistency guide.

    Defines the overall aesthetic, color scheme, and camera approach
    to ensure visual coherence across all generated scenes.
    """

    visual_style: CoercedStr = Field(
        description="Overall aesthetic approach (e.g., 'cinematic realism', 'animated', 'vintage film')"
    )
    color_palette: CoercedStr = Field(
        description="Dominant colors and lighting mood (e.g., 'warm golden tones', 'cool blue shadows')"
    )
    camera_style: CoercedStr = Field(
        description="Camera movement and framing approach (e.g., 'handheld dynamic', 'smooth tracking shots')"
    )


class CharacterDescription(BaseModel):
    """Consistent character description for cross-scene identity.

    Ensures every appearance of a character uses identical visual details
    so downstream image models render them consistently.
    """

    name: str = Field(
        description="Character name or short identifier (e.g., 'the woman', 'the detective')"
    )
    physical_description: str = Field(
        description="Detailed physical appearance: build, height, hair color and style, "
        "eye color, skin tone, facial features, age range, distinguishing marks"
    )
    clothing_description: str = Field(
        description="Detailed clothing: garments, colors, textures, accessories, shoes"
    )


class SceneSchema(BaseModel):
    """Individual scene definition with keyframe and motion prompts.

    Each scene includes detailed image prompts for start/end keyframes,
    a motion description for video interpolation, and transition notes.
    """

    scene_index: int = Field(
        description="Sequential scene number starting from 0"
    )
    scene_description: str = Field(
        description="Narrative description of what happens in this scene. "
        "Reference specific names, organizations, terms, and concepts from the original script. "
        "Note any text that should appear on-screen (titles, labels, signage)."
    )
    key_details: list[str] = Field(
        description="3-6 specific names, terms, or concepts from the original script "
        "that THIS scene must visually convey (e.g., organization names, technical terms, statistics)"
    )
    start_frame_prompt: str = Field(
        description="Detailed image prompt beginning with 'A {style} rendering of...' "
        "followed by subject, action, setting, lighting, camera, style cues, and color palette. "
        "Character descriptions must match the character bible exactly."
    )
    end_frame_prompt: str = Field(
        description="Detailed image prompt beginning with 'A {style} rendering of...' "
        "showing scene progression. Character descriptions must match the character bible exactly."
    )
    video_motion_prompt: str = Field(
        description="Motion and camera movement ONLY. Do not re-describe characters, "
        "setting, or style. Example: 'Slow dolly forward as the subject turns to face the camera'"
    )
    transition_notes: str = Field(
        description="How this scene visually connects to the next scene"
    )


class StoryboardOutput(BaseModel):
    """Complete storyboard output from Gemini structured generation.

    Contains a style guide for visual consistency and a list of scenes
    with detailed prompts for downstream keyframe and video generation.
    """

    style_guide: StyleGuide = Field(
        description="Visual consistency guide applied across all scenes"
    )
    characters: list[CharacterDescription] = Field(
        description="All characters appearing in the video with consistent physical and "
        "clothing descriptions. These descriptions must be referenced identically in every "
        "keyframe prompt where the character appears."
    )
    scenes: list[SceneSchema] = Field(
        description="List of scenes with detailed prompts and motion descriptions"
    )
