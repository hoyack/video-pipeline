"""Pydantic schemas for Screenplay LLM structured output and API responses.

Defines the expected structure for each step in the ScreenwriterService
generation chain, enabling structured output constraints via response_schema.

Spec reference: Phase 18 Screenplay System (SCRN-01, SCRN-02, SCRN-05)
"""

from typing import Optional

from pydantic import BaseModel, Field


class LoglineOutput(BaseModel):
    """LLM output for logline generation."""

    logline: str = Field(description="A compelling one-sentence logline summarizing the story")
    genre: str = Field(description="Primary genre (e.g. Action, Drama, Comedy, Sci-Fi, Thriller)")


class TreatmentOutput(BaseModel):
    """LLM output for treatment generation."""

    treatment: str = Field(description="2-5 paragraph narrative treatment expanding on the logline")


class CharacterBreakdownEntry(BaseModel):
    """Single character breakdown within the screenplay."""

    name: str = Field(description="Character name")
    role: str = Field(description="Character role: PROTAGONIST, ANTAGONIST, SUPPORTING, or EXTRA")
    arc: str = Field(description="Brief character arc description")
    description: str = Field(description="Physical and personality description")
    manifest_tag: Optional[str] = Field(
        default=None,
        description="Production Bible manifest tag (e.g. CHAR_01) if character exists in the bible",
    )


class CharacterBreakdownsOutput(BaseModel):
    """LLM output for character breakdowns."""

    characters: list[CharacterBreakdownEntry] = Field(
        description="List of all characters appearing in the production"
    )


class SceneBreakdownEntry(BaseModel):
    """Single scene breakdown entry."""

    scene_number: int = Field(description="Sequential scene number starting from 1")
    slugline: str = Field(description="Scene heading (e.g. 'INT. OFFICE - DAY')")
    intent: str = Field(description="Narrative intent for this scene; becomes the scene prompt")
    emotional_beat: str = Field(description="The emotional tone or shift in this scene")
    story_state_in: str = Field(description="Story state entering this scene")
    story_state_out: str = Field(description="Story state exiting this scene")
    characters_present: list[str] = Field(
        description="Characters in this scene (manifest_tags like CHAR_01 or character names)"
    )
    set_ref: Optional[str] = Field(
        default=None,
        description="Set/location reference (manifest_tag or name from Production Bible)",
    )
    props_required: list[str] = Field(
        default_factory=list,
        description="Props needed for this scene (manifest_tags or names)",
    )


class SceneBreakdownOutput(BaseModel):
    """LLM output for scene breakdown."""

    scene_count: int = Field(description="Total number of scenes")
    scenes: list[SceneBreakdownEntry] = Field(description="Ordered list of scene breakdowns")


class ShotListEntry(BaseModel):
    """Single shot within a scene's shot list."""

    scene_number: int = Field(description="Scene this shot belongs to")
    shot_number: int = Field(description="Sequential shot number within the scene")
    shot_type: str = Field(description="Shot type (e.g. WIDE, CLOSE-UP, MEDIUM, OTS, POV)")
    camera_movement: str = Field(
        description="Camera movement (e.g. STATIC, PAN LEFT, DOLLY IN, TRACKING)"
    )
    action_notes: str = Field(description="What happens in this shot")
    duration_hint: str = Field(description="Suggested duration (e.g. '3-5s', '2s')")


class ShotListOutput(BaseModel):
    """LLM output for shot list generation."""

    shots: list[ShotListEntry] = Field(description="Ordered list of shots across all scenes")


class ScriptOutput(BaseModel):
    """LLM output for full script."""

    script: str = Field(description="Full screenplay script text in standard screenplay format")
