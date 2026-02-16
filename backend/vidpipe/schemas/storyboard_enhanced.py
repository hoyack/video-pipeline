"""Pydantic schemas for manifest-aware storyboard output with asset placement and audio direction.

These schemas extend the base storyboard schemas with SceneManifest and SceneAudioManifest,
enabling Gemini to produce structured output that references manifest assets and includes
detailed audio direction. Used when project.manifest_id is set.

Spec reference: Phase 7 - Manifest-Aware Storyboarding and Audio Manifest
"""

from typing import Optional
from pydantic import BaseModel, Field

from vidpipe.schemas.storyboard import StyleGuide, CharacterDescription, SceneSchema


class AssetPlacement(BaseModel):
    """Asset placement within a scene with spatial, action, and continuity metadata."""

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
        description="What asset does in scene"
    )
    expression: Optional[str] = Field(
        default=None,
        description="For characters: facial expression, body language"
    )
    wardrobe_note: Optional[str] = Field(
        default=None,
        description="Clothing/appearance notes for continuity"
    )


class SceneComposition(BaseModel):
    """Camera and framing composition for a scene."""

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
        description="When in scene: start | mid-scene | end"
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
        description="Relative timing e.g. mid-scene, throughout, 0:02-0:04"
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
    """Audio continuity tracking across scene boundaries."""

    carries_from_previous: list[str] = Field(
        default_factory=list,
        description="Audio elements that carry over from previous scene"
    )
    new_in_this_scene: list[str] = Field(
        default_factory=list,
        description="New audio elements introduced in this scene"
    )
    cuts_from_previous: list[str] = Field(
        default_factory=list,
        description="Audio elements that cut from previous scene"
    )


class SceneManifestSchema(BaseModel):
    """Per-scene asset placement manifest with composition and continuity metadata."""

    scene_index: int = Field(
        description="Scene number this manifest applies to"
    )
    composition: SceneComposition = Field(
        description="Camera and framing composition"
    )
    placements: list[AssetPlacement] = Field(
        description="All assets in this scene"
    )
    continuity_notes: Optional[str] = Field(
        default=None,
        description="Continuity notes for this scene"
    )
    new_asset_declarations: Optional[list[dict]] = Field(
        default=None,
        description="Assets not in registry: [{name, type, description}]"
    )


class SceneAudioManifestSchema(BaseModel):
    """Per-scene audio direction manifest with dialogue, SFX, ambient, and music."""

    scene_index: int = Field(
        description="Scene number this audio manifest applies to"
    )
    dialogue_lines: list[DialogueLine] = Field(
        default_factory=list,
        description="Dialogue lines in this scene"
    )
    sfx: list[SFXEntry] = Field(
        default_factory=list,
        description="Sound effects in this scene"
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


class EnhancedSceneSchema(SceneSchema):
    """Scene schema enhanced with manifest and audio manifest metadata.

    Inherits all fields from SceneSchema and adds scene_manifest and audio_manifest.
    """

    scene_manifest: SceneManifestSchema = Field(
        description="Asset placement manifest for this scene"
    )
    audio_manifest: SceneAudioManifestSchema = Field(
        description="Audio direction manifest for this scene"
    )


class EnhancedStoryboardOutput(BaseModel):
    """Complete storyboard output with manifest-aware scenes.

    This is a separate model (not a subclass of StoryboardOutput) because
    the scenes field type differs (EnhancedSceneSchema vs SceneSchema).
    """

    style_guide: StyleGuide = Field(
        description="Visual consistency guide applied across all scenes"
    )
    characters: list[CharacterDescription] = Field(
        description="All characters appearing in the video with consistent physical and "
        "clothing descriptions. These descriptions must be referenced identically in every "
        "keyframe prompt where the character appears."
    )
    scenes: list[EnhancedSceneSchema] = Field(
        description="List of scenes with detailed prompts, manifest placements, and audio direction"
    )
