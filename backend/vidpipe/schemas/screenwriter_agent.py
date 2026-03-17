"""Pydantic schemas for the Screenwriter Agent's structured LLM output.

The Screenwriter Agent runs two LLM calls before the main storyboard generation:
1. Script Analysis — identifies characters, story beats, tone, and emotional arc
2. Shot Breakdown — assigns characters_present[] per shot as authoritative constraint

These schemas constrain Gemini structured output to produce validated, typed results.

Spec reference: Screenwriter Agent Plan (Phases A-E)
"""

from typing import Optional
from pydantic import BaseModel, Field


class CharacterReference(BaseModel):
    """A character identified in the scene prompt."""

    tag: str = Field(
        description="@tag from Available Assets list, or descriptive name if no tag"
    )
    role: str = Field(
        description="protagonist | supporting | background | mentioned_only"
    )
    screen_time_hint: str = Field(
        description="heavy | moderate | brief"
    )
    first_appearance_beat: int = Field(
        description="Index of story_beat where character first appears"
    )


class StoryBeat(BaseModel):
    """A key narrative moment in the scene."""

    index: int
    description: str = Field(
        description="What happens in this beat (1-2 sentences)"
    )
    characters_involved: list[str] = Field(
        description="@tags of characters in this beat"
    )
    emotional_tone: str = Field(
        description="tense | joyful | melancholy | awe | neutral | triumphant | somber"
    )
    is_climax: bool = False


class ScriptAnalysis(BaseModel):
    """Step 1 output: narrative structure analysis of the scene prompt."""

    narrative_summary: str = Field(
        description="1-2 sentence summary of the story"
    )
    tone: str = Field(
        description="cinematic | documentary | commercial | narrative | experimental"
    )
    genre: str = Field(
        description="drama | action | comedy | sci-fi | commercial | music_video | fantasy | thriller | romance | horror"
    )
    pacing: str = Field(
        description="slow_burn | steady | fast_cut | montage"
    )
    characters: list[CharacterReference]
    settings: list[str] = Field(
        description="Locations/environments mentioned or implied"
    )
    story_beats: list[StoryBeat]
    emotional_arc: str = Field(
        description="Description of tension/emotion curve across the scene"
    )


class ShotAssignment(BaseModel):
    """Per-shot character and narrative assignment."""

    shot_index: int
    beat_index: int = Field(
        description="Which story_beat this shot primarily serves"
    )
    narrative_intent: str = Field(
        description="What this shot communicates (1 sentence)"
    )
    characters_present: list[str] = Field(
        description=(
            "@tags of characters VISIBLE in this shot. "
            "EMPTY for scenery/establishing shots."
        )
    )
    setting: str = Field(
        description="Location/environment for this shot"
    )
    time_of_day: str = Field(
        description="dawn | morning | midday | afternoon | golden_hour | dusk | night"
    )
    emotional_weight: float = Field(
        ge=0, le=10,
        description="0=establishing, 10=climax"
    )
    duration_hint: float = Field(
        description="Suggested duration in seconds"
    )
    transition_from_previous: Optional[str] = Field(
        default=None,
        description="How this shot connects from the previous"
    )


class ShotBreakdown(BaseModel):
    """Step 2 output: shot-level assignments with explicit characters_present."""

    shots: list[ShotAssignment]
    arc_coverage: str = Field(
        description="Brief note on how shots map to the emotional arc"
    )
    uncovered_beats: list[int] = Field(
        default_factory=list,
        description="Beat indices not covered by any shot"
    )
