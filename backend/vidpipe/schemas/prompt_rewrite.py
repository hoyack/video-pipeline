"""Pydantic output schemas for adaptive prompt rewriting (Phase 10).

These models define the structured output returned by PromptRewriterService
when it calls Gemini 2.5 Flash to rewrite storyboard prompts with manifest
enrichment, asset reverse_prompts, continuity patches, and audio direction.

Spec reference: Phase 10 - Adaptive Prompt Rewriting
"""

from typing import Optional
from pydantic import BaseModel, Field


class RewrittenKeyframePromptOutput(BaseModel):
    """Structured output from the keyframe prompt rewriter.

    Returned by PromptRewriterService.rewrite_keyframe_prompt().
    Contains a cinematography-formula static image prompt for Imagen,
    with LLM-reasoned reference asset selection.
    """

    rewritten_prompt: str = Field(
        description=(
            "Final keyframe generation prompt, under 400 words, "
            "following [Cinematography]+[Subject]+[Action]+[Context]+[Style] formula"
        )
    )
    selected_reference_tags: list[str] = Field(
        description=(
            "Exactly 3 manifest_tags of assets selected as references, "
            "ordered by priority (most important first)"
        )
    )
    reference_reasoning: str = Field(
        description="One sentence explaining why these 3 references were chosen"
    )
    continuity_applied: Optional[str] = Field(
        default=None,
        description="Summary of continuity corrections applied (None if scene 0)"
    )


class RewrittenVideoPromptOutput(BaseModel):
    """Structured output from the video prompt rewriter.

    Returned by PromptRewriterService.rewrite_video_prompt().
    Contains a motion-focused prompt for Veo 3.1 with audio direction
    embedded inline (dialogue in quotes, SFX:, Ambient:, Music: notation).
    """

    rewritten_prompt: str = Field(
        description=(
            "Final video generation prompt, under 500 words. Motion-focused. "
            "Audio direction embedded: dialogue in quotes, SFX:, Ambient:, Music:"
        )
    )
    selected_reference_tags: list[str] = Field(
        description=(
            "Exactly 3 manifest_tags of assets selected as references, "
            "ordered by priority (most important first)"
        )
    )
    reference_reasoning: str = Field(
        description="One sentence explaining why these 3 references were chosen"
    )
    continuity_applied: Optional[str] = Field(
        default=None,
        description="Summary of continuity corrections applied (None if scene 0)"
    )
