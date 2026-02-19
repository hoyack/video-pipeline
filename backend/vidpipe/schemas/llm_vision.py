"""Pydantic models for vision LLM call site responses.

These schemas replace inline dict-based response_schema definitions in
reverse_prompt_service.py, cv_analysis_service.py, and candidate_scoring.py.
They provide type safety and consistent validation across all vision call sites.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ReversePromptOutput(BaseModel):
    """Structured output for reverse-prompting an asset image.

    Used by reverse_prompt_service.py to extract a visual description
    and generation-ready prompt from a reference image crop.
    """
    reverse_prompt: str = Field(
        description="Detailed text-to-image prompt describing the visual content"
    )
    visual_description: str = Field(
        description="Human-readable description of what is visible in the image"
    )
    quality_score: float = Field(
        description="Quality score 0.0-10.0 estimating how suitable this image is as a reference"
    )
    suggested_name: Optional[str] = Field(
        default=None,
        description="Optional suggested short name or label for this asset (e.g. 'Red sports car')",
    )


class SemanticAnalysisOutput(BaseModel):
    """Structured output for per-scene semantic analysis against a manifest.

    Used by cv_analysis_service.py to evaluate how well a generated clip
    matches the intended scene manifest and continuity expectations.
    """
    manifest_adherence: float = Field(
        description="Score 0.0-10.0 for how well the scene content matches the manifest assets"
    )
    visual_quality: float = Field(
        description="Score 0.0-10.0 for overall visual quality and composition"
    )
    continuity_issues: list[str] = Field(
        default=[],
        description="List of specific continuity issues found (empty list if none)",
    )
    new_entities_description: list[dict] = Field(
        default=[],
        description="List of new entities detected not present in the manifest (dicts with name, description)",
    )
    overall_scene_description: str = Field(
        default="",
        description="Brief overall description of what is happening in the scene",
    )


class VisualPromptScoreOutput(BaseModel):
    """Structured output for scoring a candidate clip against visual criteria.

    Used by candidate_scoring.py to evaluate Gemini-assessed quality
    dimensions for multi-candidate quality mode selection.
    """
    visual_quality: float = Field(
        description="Score 0.0-10.0 for overall visual quality of the clip"
    )
    prompt_adherence: float = Field(
        description="Score 0.0-10.0 for how well the clip follows the generation prompt"
    )
