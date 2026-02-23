"""Pydantic schemas for manifest-aware storyboard output with asset placement and audio direction.

These schemas extend the base storyboard schemas with ShotManifest and ShotAudioManifest,
enabling Gemini to produce structured output that references manifest assets and includes
detailed audio direction. Used when project.manifest_id is set.

Spec reference: Phase 7 - Manifest-Aware Storyboarding and Audio Manifest
"""

from typing import Optional
from pydantic import BaseModel, Field

from vidpipe.schemas.storyboard import StyleGuide, CharacterDescription, ShotSchema


class AssetPlacement(BaseModel):
    """Asset placement within a shot with spatial, action, and continuity metadata."""

    asset_tag: str = Field(
        description="Manifest tag e.g. CHAR_01, ENV_02"
    )
    role: str = Field(
        description="subject | background | prop | interaction_target | environment"
    )
    position: str = Field(
        description="Spatial hint: center, left, right, foreground, background"
    )
    action: Optional[str] = Field(
        default=None,
        description="What asset does in shot"
    )
    expression: Optional[str] = Field(
        default=None,
        description="For characters: facial expression, body language"
    )
    wardrobe_note: Optional[str] = Field(
        default=None,
        description="Clothing/appearance notes for continuity"
    )


class ShotComposition(BaseModel):
    """Camera and framing composition for a shot."""

    shot_type: str = Field(
        description="wide_shot | medium_shot | close_up | two_shot | establishing"
    )
    camera_movement: str = Field(
        description="static | slow_pan_left | dolly_forward | crane_up | tracking"
    )
    focal_point: str = Field(
        description="What camera focuses on (asset tag or description)"
    )


class DialogueLine(BaseModel):
    """Dialogue line with speaker, timing, and delivery metadata."""

    speaker_tag: str = Field(
        description="Character asset tag e.g. CHAR_01"
    )
    speaker_name: str = Field(
        description="Character name for readability"
    )
    line: str = Field(
        description="Exact dialogue text"
    )
    delivery: Optional[str] = Field(
        default=None,
        description="How said: muttered, shouted, whispered"
    )
    timing: str = Field(
        description="When in shot: start | mid-shot | end"
    )
    emphasis: Optional[list[str]] = Field(
        default=None,
        description="Words to emphasize"
    )


class SFXEntry(BaseModel):
    """Sound effect with trigger, timing, and volume metadata."""

    effect: str = Field(
        description="Sound effect description"
    )
    trigger: str = Field(
        description="What causes the sound"
    )
    timing: str = Field(
        description="Relative timing e.g. mid-shot, throughout, 0:02-0:04"
    )
    volume: str = Field(
        description="subtle | prominent | background"
    )


class AmbientAudio(BaseModel):
    """Ambient audio layers for environmental soundscape."""

    base_layer: str = Field(
        description="Primary ambient sound"
    )
    environmental: Optional[str] = Field(
        default=None,
        description="Environmental audio layer"
    )
    weather: Optional[str] = Field(
        default=None,
        description="Weather-related audio"
    )
    time_cues: Optional[str] = Field(
        default=None,
        description="Time-of-day audio cues"
    )


class MusicDirection(BaseModel):
    """Music direction with style, mood, tempo, and transition metadata."""

    style: str = Field(
        description="Music style"
    )
    mood: str = Field(
        description="Music mood"
    )
    tempo: str = Field(
        description="slow | moderate | fast | accelerating"
    )
    instruments: Optional[list[str]] = Field(
        default=None,
        description="Instruments used"
    )
    transition: str = Field(
        description="How music enters/exits: fade in, cut, swell"
    )


class AudioContinuity(BaseModel):
    """Audio continuity tracking across shot boundaries."""

    carries_from_previous: list[str] = Field(
        default_factory=list,
        description="Audio elements that carry over from previous shot"
    )
    new_in_this_shot: list[str] = Field(
        default_factory=list,
        description="New audio elements introduced in this shot"
    )
    cuts_from_previous: list[str] = Field(
        default_factory=list,
        description="Audio elements that cut from previous shot"
    )


class ShotManifestSchema(BaseModel):
    """Per-shot asset placement manifest with composition and continuity metadata."""

    shot_index: int = Field(
        description="Shot number this manifest applies to"
    )
    composition: ShotComposition = Field(
        description="Camera and framing composition"
    )
    placements: list[AssetPlacement] = Field(
        description="All assets in this shot"
    )
    continuity_notes: Optional[str] = Field(
        default=None,
        description="Continuity notes for this shot"
    )
    new_asset_declarations: Optional[list[dict]] = Field(
        default=None,
        description="Assets not in registry: [{name, type, description}]"
    )


class ShotAudioManifestSchema(BaseModel):
    """Per-shot audio direction manifest with dialogue, SFX, ambient, and music."""

    shot_index: int = Field(
        description="Shot number this audio manifest applies to"
    )
    dialogue_lines: list[DialogueLine] = Field(
        default_factory=list,
        description="Dialogue lines in this shot"
    )
    sfx: list[SFXEntry] = Field(
        default_factory=list,
        description="Sound effects in this shot"
    )
    ambient: Optional[AmbientAudio] = Field(
        default=None,
        description="Ambient audio layers"
    )
    music: Optional[MusicDirection] = Field(
        default=None,
        description="Music direction"
    )
    audio_continuity: Optional[AudioContinuity] = Field(
        default=None,
        description="Audio continuity tracking"
    )


class EnhancedShotSchema(ShotSchema):
    """Shot schema enhanced with manifest and audio manifest metadata.

    Inherits all fields from ShotSchema and adds shot_manifest and audio_manifest.
    """

    shot_manifest: ShotManifestSchema = Field(
        description="Asset placement manifest for this shot"
    )
    audio_manifest: ShotAudioManifestSchema = Field(
        description="Audio direction manifest for this shot"
    )


class EnhancedStoryboardOutput(BaseModel):
    """Complete storyboard output with manifest-aware shots.

    This is a separate model (not a subclass of StoryboardOutput) because
    the shots field type differs (EnhancedShotSchema vs ShotSchema).
    """

    style_guide: StyleGuide = Field(
        description="Visual consistency guide applied across all shots"
    )
    characters: list[CharacterDescription] = Field(
        description="All characters appearing in the video with consistent physical and "
        "clothing descriptions. These descriptions must be referenced identically in every "
        "keyframe prompt where the character appears."
    )
    shots: list[EnhancedShotSchema] = Field(
        description="List of shots with detailed prompts, manifest placements, and audio direction"
    )
