"""Pydantic schemas for storyboard structured output from Gemini.

These schemas define the expected structure for LLM-generated storyboards,
enabling structured output constraints via response_schema parameter.

Spec reference: Section 5.1 (Storyboard Output Schema)
"""

from pydantic import BaseModel, Field


class StyleGuide(BaseModel):
    """Cross-scene visual consistency guide.

    Defines the overall aesthetic, color scheme, and camera approach
    to ensure visual coherence across all generated scenes.
    """

    visual_style: str = Field(
        description="Overall aesthetic approach (e.g., 'cinematic realism', 'animated', 'vintage film')"
    )
    color_palette: str = Field(
        description="Dominant colors and lighting mood (e.g., 'warm golden tones', 'cool blue shadows')"
    )
    camera_style: str = Field(
        description="Camera movement and framing approach (e.g., 'handheld dynamic', 'smooth tracking shots')"
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
        description="Brief narrative description of what happens in this scene"
    )
    start_frame_prompt: str = Field(
        description="Detailed image prompt for opening keyframe with composition, lighting, and style details"
    )
    end_frame_prompt: str = Field(
        description="Detailed image prompt for closing keyframe showing scene progression"
    )
    video_motion_prompt: str = Field(
        description="Motion/action description for video interpolation between keyframes"
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
    scenes: list[SceneSchema] = Field(
        description="List of 3-5 scenes with detailed prompts and motion descriptions"
    )
