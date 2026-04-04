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
    """Structured output for per-shot semantic analysis against a manifest.

    Used by cv_analysis_service.py to evaluate how well a generated clip
    matches the intended shot manifest and continuity expectations.
    """
    manifest_adherence: float = Field(
        description="Score 0.0-10.0 for how well the shot content matches the manifest assets"
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
    overall_shot_description: str = Field(
        default="",
        description="Brief overall description of what is happening in the shot",
    )


class CharacterKeyframeVerificationOutput(BaseModel):
    """Structured output for character-level keyframe verification.

    Used by keyframes.py to verify a generated character crop against
    expected identity and wardrobe references using the scene vision model.
    """

    passed: bool = Field(
        description="True if the generated crop is acceptable for this character"
    )
    character_visible: bool = Field(
        description="True if the expected character is visible in the generated crop"
    )
    identity_match: bool = Field(
        description="True if identity/species/markings sufficiently match the references"
    )
    wardrobe_match: bool = Field(
        description="True if wardrobe/look sufficiently matches the references"
    )
    identity_score: float = Field(
        description="Identity match score from 0.0-10.0"
    )
    wardrobe_score: float = Field(
        description="Wardrobe/look match score from 0.0-10.0"
    )
    issues: list[str] = Field(
        default=[],
        description="Short list of concrete mismatch issues, if any",
    )
    summary: str = Field(
        default="",
        description="Brief summary of the verification result",
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


class SetMetadataOutput(BaseModel):
    """Structured output for auto-generating library set metadata from a reference image.

    Used by the asset library generate-metadata endpoint to produce
    description, reverse_prompt, and lighting_notes in one vision call.
    """
    description: str = Field(
        description="Visual description of the set/environment for a production bible. "
        "Describe the location type, architecture, atmosphere, color palette, and notable features. ~60-100 words."
    )
    reverse_prompt: str = Field(
        description="Detailed text-to-image prompt to recreate this set/environment. "
        "Include: location type, architecture/layout, lighting, weather, depth/perspective, "
        "key landmarks, color palette, atmosphere, camera framing. ~120-180 words."
    )
    lighting_notes: str = Field(
        description="Concise lighting description for the set. Include: time of day, "
        "light sources (natural/artificial), direction, color temperature, shadows, "
        "mood/atmosphere created by the lighting. ~30-60 words."
    )


class ActorMetadataOutput(BaseModel):
    """Structured output for auto-generating actor metadata from a reference image."""
    description: str = Field(
        description="Physical appearance description of the character/person for a production bible. "
        "Describe body type, face shape, hair, skin tone, distinguishing features, "
        "typical expression, and overall presence. ~60-100 words."
    )
    base_appearance_prompt: str = Field(
        description="Detailed text-to-image prompt to recreate this character's appearance. "
        "Include: gender, age range, ethnicity, body type, hair (style/color/length), "
        "face details (eyes, nose, jawline), skin tone, distinguishing marks, "
        "default clothing style, posture, expression. Be specific enough for consistent "
        "identity across generations. ~120-180 words."
    )


class PropMetadataOutput(BaseModel):
    """Structured output for auto-generating prop metadata from a reference image."""
    description: str = Field(
        description="Physical description of the prop/object for a production bible. "
        "Describe shape, size, materials, color, condition, and notable features. ~60-100 words."
    )
    appearance_prompt: str = Field(
        description="Detailed text-to-image prompt to recreate this prop/object. "
        "Include: object type, dimensions/proportions, material/texture, color palette, "
        "wear/condition, distinctive details, lighting interaction (reflective, matte, etc.), "
        "viewing angle. ~80-120 words."
    )
