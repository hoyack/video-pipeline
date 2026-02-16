# Phase 7: Manifest-Aware Storyboarding and Audio Manifest - Research

**Researched:** 2026-02-16
**Domain:** LLM-based storyboarding with asset registry integration, structured audio manifest generation
**Confidence:** HIGH

## Summary

Phase 7 enhances the storyboarding pipeline to consume the Asset Registry (from Phase 5) and produce structured scene manifests with asset-tagged placements and comprehensive audio direction. This phase bridges the gap between asset preparation and generation by enabling the LLM to reference specific, pre-processed visual assets rather than generating generic scene descriptions.

The research confirms that Gemini 2.5 supports robust structured output with full JSON schema compliance (as of 2026), making it ideal for generating complex nested manifests. The V2 architecture documents define complete schemas for SceneManifest and SceneAudioManifest. Veo 3.1 provides native audio generation with structured prompt formats for dialogue, SFX, ambient, and music layers. Professional VFX pipelines rely on similar manifest-based workflows for shot breakdowns and asset tracking.

**Primary recommendation:** Extend the existing storyboard pipeline with enhanced system prompts that include full Asset Registry context, then add structured scene manifest and audio manifest schemas to capture LLM output. Use Gemini's native JSON schema support for validation.

## Standard Stack

### Core Dependencies (Already in Project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| google-genai | ^0.12.0+ | Gemini API client with structured output | Official Google SDK, supports Vertex AI mode with ADC auth |
| Pydantic | 2.x | Schema definition and validation | De facto standard for Python data validation, works out-of-box with Gemini |
| SQLAlchemy | 2.0 | Database ORM for manifest tables | Already used for all models, async support |
| tenacity | 8.x+ | Retry logic with backoff | Already used in storyboard.py for JSON failures |

### New Database Tables Required

| Table | Purpose | When Created |
|-------|---------|--------------|
| `scene_manifests` | Structured per-scene asset placements and composition metadata | Phase 7 |
| `scene_audio_manifests` | Per-scene audio direction with dialogue mapping, SFX, ambient, music | Phase 7 |

### Schema Design Pattern

**From v2-pipe-optimization.md lines 524-579:**
```python
# SceneManifest captures asset-to-scene mapping
scene_manifests:
  - scene_index: int
  - composition: {shot_type, camera_movement, focal_point}
  - placements: [{asset_tag, role, position, action, expression, wardrobe_note}]
  - audio: {dialogue, sfx, ambient, music}
  - continuity: {inherits_from, lighting_match, wardrobe_changes, new_elements}
  - selected_references: [tag1, tag2, tag3]  # Max 3 for Veo
```

**From v2-manifest.md lines 412-456:**
```python
# SceneAudioManifest for comprehensive audio direction
scene_audio_manifests:
  - scene_index: int
  - dialogue_lines: [{speaker_tag, speaker_name, line, delivery, timing, emphasis}]
  - sfx: [{effect, trigger, timing, volume}]
  - ambient: {base_layer, environmental, weather, time_cues}
  - music: {style, mood, tempo, instruments, transition}
  - audio_continuity: {carries_from_previous, new_in_this_scene, cuts_from_previous}
```

## Architecture Patterns

### Recommended Implementation Flow

```
PHASE 1 ENHANCEMENT (Storyboarding with Asset Context)
  ↓
1. Load Asset Registry from database
   - Query all assets for manifest_id (from project.manifest_id)
   - Retrieve: manifest_tag, name, reverse_prompt, visual_description, quality_score
   ↓
2. Build enhanced system prompt
   - Existing: style, aspect_ratio, scene count requirements
   - NEW: Inject asset registry block (formatted manifest tags + descriptions)
   - NEW: Instruct LLM to reference assets by [TAG] in placements
   - NEW: Allow declaring NEW assets not yet in registry
   ↓
3. Define extended Pydantic schemas
   - Extend StoryboardOutput with scene_manifests field
   - Add SceneManifestSchema with placements array
   - Add SceneAudioManifestSchema with dialogue/sfx/ambient/music
   ↓
4. Gemini call with structured output
   - response_schema = EnhancedStoryboardOutput
   - response_mime_type = "application/json"
   - Same retry logic with temperature reduction (existing pattern)
   ↓
5. Persist structured manifests
   - Insert scene_manifests rows (one per scene)
   - Insert scene_audio_manifests rows (one per scene)
   - Store manifest_json as JSONB for flexibility
   ↓
6. Existing behavior continues unchanged
   - Scene records created with prompts
   - Project status → "keyframing"
```

### System Prompt Enhancement Pattern

**Current pattern (storyboard.py lines 25-77):**
```python
STORYBOARD_SYSTEM_PROMPT = """You are a storyboard director...
VISUAL STYLE: {style}
ASPECT RATIO: {aspect_ratio}
REQUIREMENTS:
- Break the script into {scene_count} distinct visual scenes
- Describe all characters with consistent physical details
..."""
```

**Enhanced pattern for Phase 7:**
```python
ENHANCED_STORYBOARD_PROMPT = """You are a storyboard director...
VISUAL STYLE: {style}
ASPECT RATIO: {aspect_ratio}

AVAILABLE ASSETS (from Asset Registry):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{asset_registry_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When creating scene manifests:
- Reference assets by their [TAG] (e.g., [CHAR_01], [ENV_02])
- Use reverse_prompts for visual details (already optimized)
- Assign roles: subject, background, prop, interaction_target
- Specify spatial positions and actions
- You may declare NEW assets not in registry — describe them textually

AUDIO DIRECTION:
For each scene, generate a comprehensive audio manifest:
- Dialogue: Map speech to character tags with delivery notes
- SFX: Describe effects with timing (use format "SFX: description")
- Ambient: Base layer + environmental context
- Music: Style, mood, instruments, transition cues

REQUIREMENTS:
- Break script into {scene_count} scenes
- Each scene MUST include scene_manifest and audio_manifest
..."""
```

**Asset registry block formatting:**
```python
def format_asset_registry(assets: list[Asset]) -> str:
    """Format assets for LLM context injection."""
    lines = []
    for asset in assets:
        lines.append(f"[{asset.manifest_tag}] \"{asset.name}\" ({asset.asset_type}, quality: {asset.quality_score}/10)")
        lines.append(f"  Reverse prompt: {asset.reverse_prompt[:200]}...")
        if asset.visual_description:
            lines.append(f"  Notes: {asset.visual_description[:150]}...")
        lines.append("")
    return "\n".join(lines)
```

### Pydantic Schema Extension Pattern

**Existing schemas (schemas/storyboard.py lines 49-84):**
```python
class SceneSchema(BaseModel):
    scene_index: int
    scene_description: str
    key_details: list[str]
    start_frame_prompt: str
    end_frame_prompt: str
    video_motion_prompt: str
    transition_notes: str
```

**New schemas for Phase 7:**
```python
class AssetPlacement(BaseModel):
    """Asset placement within a scene."""
    asset_tag: str = Field(description="Manifest tag reference (e.g., CHAR_01, ENV_02)")
    role: str = Field(description="subject | background | prop | interaction_target")
    position: str = Field(description="Spatial hint: center, left, right, foreground, background")
    action: Optional[str] = Field(description="What this asset is doing in the scene", default=None)
    expression: Optional[str] = Field(description="For characters: facial expression, body language", default=None)
    wardrobe_note: Optional[str] = Field(description="Clothing/appearance notes for continuity", default=None)

class SceneComposition(BaseModel):
    """Visual composition metadata."""
    shot_type: str = Field(description="wide_shot | medium_shot | close_up | two_shot | establishing")
    camera_movement: str = Field(description="static | slow_pan_left | dolly_forward | crane_up | tracking")
    focal_point: str = Field(description="What the camera focuses on (asset tag or description)")

class DialogueLine(BaseModel):
    """Single line of dialogue mapped to character."""
    speaker_tag: str = Field(description="Character asset tag (e.g., CHAR_01)")
    speaker_name: str = Field(description="Character name for readability")
    line: str = Field(description="Exact dialogue text")
    delivery: Optional[str] = Field(description="How it's said: muttered, shouted, whispered", default=None)
    timing: str = Field(description="When in scene: start | mid-scene | end")
    emphasis: Optional[list[str]] = Field(description="Words to emphasize", default=None)

class SFXEntry(BaseModel):
    """Sound effect entry with timing."""
    effect: str = Field(description="Description of the sound effect")
    trigger: str = Field(description="What causes this sound (action/event)")
    timing: str = Field(description="Timestamp or relative timing (e.g., '0:02-0:04', 'throughout')")
    volume: str = Field(description="subtle | prominent | background")

class AmbientAudio(BaseModel):
    """Ambient sound layers."""
    base_layer: str = Field(description="Primary ambient sound (e.g., 'office hum', 'forest birds')")
    environmental: Optional[str] = Field(description="Environmental context sounds", default=None)
    weather: Optional[str] = Field(description="Weather-related sounds if applicable", default=None)
    time_cues: Optional[str] = Field(description="Time-of-day audio cues", default=None)

class MusicDirection(BaseModel):
    """Music cues and style."""
    style: str = Field(description="Musical style (e.g., 'tense jazz piano', 'orchestral swell')")
    mood: str = Field(description="Emotional tone of the music")
    tempo: str = Field(description="slow | moderate | fast | accelerating")
    instruments: Optional[list[str]] = Field(description="Specific instruments featured", default=None)
    transition: str = Field(description="How music enters/exits (fade in, cut, swell)")

class AudioContinuity(BaseModel):
    """Audio continuity tracking between scenes."""
    carries_from_previous: list[str] = Field(description="Audio elements that continue from previous scene")
    new_in_this_scene: list[str] = Field(description="Audio elements introduced in this scene")
    cuts_from_previous: list[str] = Field(description="Audio elements that stop from previous scene")

class SceneManifestSchema(BaseModel):
    """Structured scene manifest with asset placements."""
    scene_index: int
    composition: SceneComposition
    placements: list[AssetPlacement] = Field(description="All assets appearing in this scene")
    continuity_notes: Optional[str] = Field(description="Visual continuity with previous scenes", default=None)
    new_asset_declarations: Optional[list[dict]] = Field(
        description="Assets declared but not in registry: [{name, type, description}]",
        default=None
    )

class SceneAudioManifestSchema(BaseModel):
    """Structured audio manifest for a scene."""
    scene_index: int
    dialogue_lines: list[DialogueLine] = Field(default_factory=list)
    sfx: list[SFXEntry] = Field(default_factory=list)
    ambient: Optional[AmbientAudio] = Field(default=None)
    music: Optional[MusicDirection] = Field(default=None)
    audio_continuity: Optional[AudioContinuity] = Field(default=None)

class EnhancedSceneSchema(SceneSchema):
    """Extended scene schema with manifests."""
    scene_manifest: SceneManifestSchema
    audio_manifest: SceneAudioManifestSchema

class EnhancedStoryboardOutput(BaseModel):
    """Storyboard output with manifests."""
    style_guide: StyleGuide
    characters: list[CharacterDescription]
    scenes: list[EnhancedSceneSchema]
```

### Database Schema Pattern

```sql
-- Scene manifests table
CREATE TABLE scene_manifests (
    project_id UUID REFERENCES projects(id),
    scene_index INTEGER NOT NULL,
    manifest_json JSONB NOT NULL,           -- Full SceneManifest structure
    composition_shot_type VARCHAR(50),      -- Denormalized for queries
    composition_camera_movement VARCHAR(50),
    asset_tags TEXT[],                      -- Array of referenced tags
    new_asset_count INTEGER DEFAULT 0,      -- Count of declared new assets
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, scene_index)
);

CREATE INDEX idx_scene_manifests_asset_tags ON scene_manifests USING GIN(asset_tags);

-- Scene audio manifests table
CREATE TABLE scene_audio_manifests (
    project_id UUID REFERENCES projects(id),
    scene_index INTEGER NOT NULL,
    dialogue_json JSONB,                    -- Array of DialogueLine
    sfx_json JSONB,                         -- Array of SFXEntry
    ambient_json JSONB,                     -- AmbientAudio object
    music_json JSONB,                       -- MusicDirection object
    audio_continuity_json JSONB,            -- AudioContinuity object
    speaker_tags TEXT[],                    -- Denormalized dialogue speakers
    has_dialogue BOOLEAN DEFAULT FALSE,
    has_music BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, scene_index)
);

CREATE INDEX idx_scene_audio_speaker_tags ON scene_audio_manifests USING GIN(speaker_tags);
```

### Integration with Existing Pipeline

**Current storyboard.py flow (lines 80-171):**
```python
async def generate_storyboard(session, project):
    # 1. Build prompt
    # 2. Call Gemini with retry
    # 3. Parse StoryboardOutput
    # 4. Update project.style_guide
    # 5. Create Scene records
    # 6. Set status = "keyframing"
    # 7. Commit
```

**Enhanced flow for Phase 7:**
```python
async def generate_storyboard(session, project):
    # 1. NEW: Load Asset Registry if manifest_id exists
    if project.manifest_id:
        assets = await load_manifest_assets(session, project.manifest_id)
        asset_registry_block = format_asset_registry(assets)
    else:
        asset_registry_block = "No assets registered. Describe all elements."

    # 2. Build enhanced prompt with asset context
    system_prompt = ENHANCED_STORYBOARD_PROMPT.format(
        style=project.style,
        aspect_ratio=project.aspect_ratio,
        scene_count=project.target_scene_count,
        asset_registry_block=asset_registry_block
    )

    # 3. Call Gemini with extended schema
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[system_prompt + "\n\nScript: " + project.prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EnhancedStoryboardOutput,  # NEW SCHEMA
            temperature=temperature
        )
    )

    # 4. Parse and validate
    storyboard = EnhancedStoryboardOutput.model_validate_json(response.text)

    # 5. Existing: Update project.style_guide, create Scene records
    # ... (unchanged)

    # 6. NEW: Persist scene manifests
    for scene_data in storyboard.scenes:
        # Create scene_manifests entry
        await session.execute(
            insert(SceneManifest).values(
                project_id=project.id,
                scene_index=scene_data.scene_index,
                manifest_json=scene_data.scene_manifest.model_dump(),
                composition_shot_type=scene_data.scene_manifest.composition.shot_type,
                composition_camera_movement=scene_data.scene_manifest.composition.camera_movement,
                asset_tags=[p.asset_tag for p in scene_data.scene_manifest.placements],
                new_asset_count=len(scene_data.scene_manifest.new_asset_declarations or [])
            )
        )

        # Create scene_audio_manifests entry
        audio = scene_data.audio_manifest
        await session.execute(
            insert(SceneAudioManifest).values(
                project_id=project.id,
                scene_index=scene_data.scene_index,
                dialogue_json=[d.model_dump() for d in audio.dialogue_lines],
                sfx_json=[s.model_dump() for s in audio.sfx],
                ambient_json=audio.ambient.model_dump() if audio.ambient else None,
                music_json=audio.music.model_dump() if audio.music else None,
                audio_continuity_json=audio.audio_continuity.model_dump() if audio.audio_continuity else None,
                speaker_tags=[d.speaker_tag for d in audio.dialogue_lines],
                has_dialogue=len(audio.dialogue_lines) > 0,
                has_music=audio.music is not None
            )
        )

    # 7. Commit
    await session.commit()
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON schema validation | Custom validators, regex parsing | Pydantic with Gemini response_schema | Gemini natively enforces schema, Pydantic validates on receipt. Auto-fails with clear errors. |
| Asset registry formatting | String templates, f-strings | Dedicated format function with consistent structure | LLM context windows are large but not infinite. Structured formatting ensures all assets fit while maintaining readability. |
| Dialogue-to-character mapping | Regex extraction from prompts | Structured DialogueLine schema with speaker_tag | LLM can directly map dialogue to asset tags. No post-processing or ambiguity. |
| Audio layer composition | Parsing unstructured text | Separate schemas for dialogue/SFX/ambient/music | Professional audio pipelines use discrete layers. Structured data enables downstream audio synthesis (future). |
| Retry logic for JSON failures | Custom retry loops | tenacity library (already in use) | Existing pattern in storyboard.py works. Temperature reduction on retry improves schema compliance. |

**Key insight:** Gemini 2.5's JSON schema support makes structured manifests trivial. The hard part is NOT parsing — it's designing schemas that capture the right semantic information for downstream phases. The V2 architecture documents already define these schemas based on professional VFX workflows.

## Common Pitfalls

### Pitfall 1: Asset Registry Context Overload

**What goes wrong:** Including full reverse_prompts for all assets exceeds token limits or degrades LLM performance.

**Why it happens:** reverse_prompt fields can be 100-200 words each. 20 assets × 150 words = 3,000 words just for asset context.

**How to avoid:**
- Truncate reverse_prompts to first 200 characters in system prompt
- Include full visual_description only for CHARACTER types
- Use summary format: `[TAG] "Name" (type, quality) + 1-sentence description`
- Consider filtering: only include assets with quality_score ≥ 6.0

**Warning signs:**
- Gemini response times exceed 15 seconds
- LLM starts hallucinating asset tags not in registry
- Schema validation errors due to incomplete responses

### Pitfall 2: Treating Manifests as Prompts

**What goes wrong:** Directly using manifest JSON as generation prompts leads to verbose, poorly formatted prompts.

**Why it happens:** Manifests are structured data for tracking and reference selection. They're not optimized for LLM prompt consumption.

**How to avoid:**
- Manifests are FOR planning and asset selection
- Phase 8 (Adaptive Prompt Rewriting) will CONSUME manifests to BUILD generation prompts
- Manifest placements → injected into rewritten prompts with asset reverse_prompts
- Audio manifests → formatted into Veo 3.1 audio prompt syntax

**Example of correct flow:**
```
SceneManifest.placements = [
  {asset_tag: "CHAR_01", role: "subject", action: "examining letter"}
]
      ↓ (Phase 8: Prompt Rewriter)
Generation prompt = "A weathered middle-aged man {CHAR_01.reverse_prompt}
  examining a letter under a desk lamp..."
```

### Pitfall 3: Assuming Perfect Asset Tag Matching

**What goes wrong:** LLM invents asset tags not in registry or misspells existing tags.

**Why it happens:** LLM may generalize from patterns or forget exact tags from long context.

**How to avoid:**
- System prompt: "ONLY use tags from the Available Assets list. NEVER invent new tags."
- Post-validation: Check all placement.asset_tag values against registry
- Allow `new_asset_declarations` field for truly new assets (described but not tagged)
- Log warnings for unrecognized tags, flag scenes for manual review

**Warning signs:**
- Tags like "CHAR_03" when only CHAR_01 and CHAR_02 exist
- Inconsistent formatting: "char_01" vs "CHAR_01"
- Generic tags: "CHARACTER_1" instead of registry format

### Pitfall 4: Audio Manifest Over-Specification

**What goes wrong:** Generating frame-by-frame SFX timing that Veo can't honor.

**Why it happens:** LLM can generate hyper-detailed audio cues, but Veo 3.1 uses prompt-guided generation, not precise timeline control.

**How to avoid:**
- Audio manifests should describe WHAT sounds occur, not exact timing
- SFX timing: use "start", "mid-scene", "end", or ranges like "0:02-0:04"
- Veo interprets audio prompts holistically — it's not frame-accurate
- Think "audio direction" not "audio sequencing"

**Correct level of detail:**
```json
// GOOD: Descriptive, timing hints
{
  "effect": "paper rustling",
  "trigger": "character examines letter",
  "timing": "mid-scene",
  "volume": "subtle"
}

// TOO DETAILED: Frame-level precision Veo can't achieve
{
  "effect": "paper rustling",
  "start_frame": 48,
  "end_frame": 72,
  "volume_envelope": [0.2, 0.5, 0.3, 0.1]
}
```

### Pitfall 5: Ignoring Character Continuity from Phase 2

**What goes wrong:** Manifest placements specify wardrobe/expression details that contradict the existing StoryboardOutput.characters field.

**Why it happens:** Two sources of character description (Phase 2 character bible + Phase 7 manifests) can diverge.

**How to avoid:**
- Enhanced system prompt: "Wardrobe notes in placements MUST match character descriptions exactly."
- Phase 2's `characters` field remains the source of truth for consistent appearance
- Manifest placements add SCENE-SPECIFIC details: actions, expressions, prop interactions
- If manifest declares wardrobe change, it should note: "different from standard — see scene context"

**Example:**
```python
# Phase 2: characters field
{
  "name": "Detective Marcus",
  "clothing_description": "Rumpled brown trench coat, fedora hat, burgundy tie"
}

# Phase 7: placement with continuity
{
  "asset_tag": "CHAR_01",
  "wardrobe_note": "same trench coat as previous scenes, hat removed, tie loosened"
}
```

## Code Examples

Verified patterns from existing codebase and design docs:

### Loading Manifest Assets
```python
# Source: Adapted from manifest_service.py pattern
async def load_manifest_assets(
    session: AsyncSession,
    manifest_id: uuid.UUID
) -> list[Asset]:
    """Load all assets for a manifest, ordered by quality."""
    from vidpipe.db.models import Asset
    from sqlalchemy import select

    result = await session.execute(
        select(Asset)
        .where(Asset.manifest_id == manifest_id)
        .order_by(Asset.quality_score.desc().nullslast())
    )
    return list(result.scalars().all())
```

### Formatting Asset Registry for LLM Context
```python
# Source: Pattern from v2-pipe-optimization.md lines 481-512
def format_asset_registry(assets: list[Asset]) -> str:
    """Format asset registry for LLM system prompt injection.

    Truncates prompts to fit context window while preserving key details.
    """
    if not assets:
        return "No assets registered. Describe all visual elements in scenes."

    lines = ["AVAILABLE ASSETS FOR THIS PROJECT:", "━" * 40, ""]

    for asset in assets:
        # Header line with key metadata
        quality = f"{asset.quality_score:.1f}/10" if asset.quality_score else "N/A"
        lines.append(f"[{asset.manifest_tag}] \"{asset.name}\" ({asset.asset_type}, quality: {quality})")

        # Truncated reverse prompt (most important for generation)
        if asset.reverse_prompt:
            truncated = asset.reverse_prompt[:200] + "..." if len(asset.reverse_prompt) > 200 else asset.reverse_prompt
            lines.append(f"  Reverse prompt: {truncated}")

        # Visual description for key details (only for high-quality assets)
        if asset.visual_description and (asset.quality_score or 0) >= 7.0:
            truncated_desc = asset.visual_description[:150] + "..." if len(asset.visual_description) > 150 else asset.visual_description
            lines.append(f"  Production notes: {truncated_desc}")

        lines.append("")  # Blank line between assets

    lines.append("━" * 40)
    lines.append("Reference assets by [TAG]. You may declare NEW assets not in the registry.")

    return "\n".join(lines)
```

### Enhanced Storyboard Generation with Manifests
```python
# Source: Adapted from storyboard.py with Phase 7 enhancements
async def generate_storyboard_with_manifests(
    session: AsyncSession,
    project: Project
) -> None:
    """Generate storyboard with scene and audio manifests."""
    from vidpipe.schemas.storyboard_enhanced import EnhancedStoryboardOutput
    from vidpipe.services.vertex_client import get_vertex_client, location_for_model
    from google.genai import types

    model_id = project.text_model or settings.models.storyboard_llm
    client = get_vertex_client(location=location_for_model(model_id))

    # Load asset registry if manifest attached
    asset_registry_block = ""
    if project.manifest_id:
        assets = await load_manifest_assets(session, project.manifest_id)
        asset_registry_block = format_asset_registry(assets)
    else:
        asset_registry_block = "No manifest selected. Describe all visual elements."

    # Build enhanced system prompt
    system_prompt = ENHANCED_STORYBOARD_PROMPT.format(
        style=project.style.replace("_", " "),
        aspect_ratio=project.aspect_ratio,
        scene_count=project.target_scene_count,
        asset_registry_block=asset_registry_block
    )

    full_prompt = f"{system_prompt}\n\nScript: {project.prompt}"

    # Retry with temperature reduction (existing pattern)
    attempt = 0
    max_attempts = 3
    base_temperature = 0.7

    @retry(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type((json.JSONDecodeError, ValidationError))
    )
    async def generate_with_retry():
        nonlocal attempt
        temperature = base_temperature - (attempt * 0.15)
        attempt += 1

        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EnhancedStoryboardOutput,
                temperature=max(0.0, temperature)
            )
        )

        return EnhancedStoryboardOutput.model_validate_json(response.text)

    storyboard = await generate_with_retry()

    # Update project (existing pattern)
    project.style_guide = storyboard.style_guide.model_dump()
    project.storyboard_raw = storyboard.model_dump()

    # Create Scene records (existing)
    for scene_data in storyboard.scenes:
        scene = Scene(
            project_id=project.id,
            scene_index=scene_data.scene_index,
            scene_description=scene_data.scene_description,
            start_frame_prompt=scene_data.start_frame_prompt,
            end_frame_prompt=scene_data.end_frame_prompt,
            video_motion_prompt=scene_data.video_motion_prompt,
            transition_notes=scene_data.transition_notes,
            status="pending"
        )
        session.add(scene)

    # NEW: Persist scene manifests
    from vidpipe.db.models import SceneManifest, SceneAudioManifest

    for scene_data in storyboard.scenes:
        # Scene visual manifest
        scene_manifest = SceneManifest(
            project_id=project.id,
            scene_index=scene_data.scene_index,
            manifest_json=scene_data.scene_manifest.model_dump(),
            composition_shot_type=scene_data.scene_manifest.composition.shot_type,
            composition_camera_movement=scene_data.scene_manifest.composition.camera_movement,
            asset_tags=[p.asset_tag for p in scene_data.scene_manifest.placements],
            new_asset_count=len(scene_data.scene_manifest.new_asset_declarations or [])
        )
        session.add(scene_manifest)

        # Scene audio manifest
        audio = scene_data.audio_manifest
        audio_manifest = SceneAudioManifest(
            project_id=project.id,
            scene_index=scene_data.scene_index,
            dialogue_json=[d.model_dump() for d in audio.dialogue_lines],
            sfx_json=[s.model_dump() for s in audio.sfx],
            ambient_json=audio.ambient.model_dump() if audio.ambient else None,
            music_json=audio.music.model_dump() if audio.music else None,
            audio_continuity_json=audio.audio_continuity.model_dump() if audio.audio_continuity else None,
            speaker_tags=[d.speaker_tag for d in audio.dialogue_lines],
            has_dialogue=len(audio.dialogue_lines) > 0,
            has_music=audio.music is not None
        )
        session.add(audio_manifest)

    project.status = "keyframing"
    await session.commit()
```

### Veo 3.1 Audio Prompt Formatting (for future Phase 8)
```python
# Source: Veo 3.1 prompting guide + v2-manifest.md
def format_audio_for_veo_prompt(audio_manifest: SceneAudioManifest) -> str:
    """Convert structured audio manifest to Veo 3.1 prompt format.

    This is NOT used in Phase 7 — it's for Phase 8 (Adaptive Prompt Rewriting).
    Included here to show how manifests will be consumed.

    Reference: https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1
    """
    audio_parts = []

    # Dialogue (use quotation marks)
    for dialogue in audio_manifest.dialogue_json:
        speaker = dialogue["speaker_name"]
        line = dialogue["line"]
        delivery = dialogue.get("delivery", "")

        if delivery:
            audio_parts.append(f'{speaker} {delivery}: "{line}"')
        else:
            audio_parts.append(f'{speaker} says, "{line}"')

    # Sound effects (use "SFX:" prefix)
    for sfx in audio_manifest.sfx_json:
        effect = sfx["effect"]
        timing = sfx.get("timing", "")
        if timing:
            audio_parts.append(f"SFX: {effect} ({timing})")
        else:
            audio_parts.append(f"SFX: {effect}")

    # Ambient (describe background soundscape)
    if audio_manifest.ambient_json:
        ambient = audio_manifest.ambient_json
        base = ambient.get("base_layer", "")
        env = ambient.get("environmental", "")

        ambient_desc = f"Ambient noise: {base}"
        if env:
            ambient_desc += f", {env}"
        audio_parts.append(ambient_desc)

    # Music (separate sentence for musical direction)
    if audio_manifest.music_json:
        music = audio_manifest.music_json
        style = music.get("style", "")
        mood = music.get("mood", "")
        transition = music.get("transition", "")

        music_desc = f"Music: {style}, {mood} mood"
        if transition:
            music_desc += f" ({transition})"
        audio_parts.append(music_desc)

    return " ".join(audio_parts)

# Example output:
# Marcus mutters: "This doesn't add up..." SFX: paper rustling (mid-scene)
# Ambient noise: quiet office hum, distant traffic through window
# Music: low tense jazz piano, suspenseful mood (fade in from silence)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Generic scene descriptions | Asset-tagged manifests with reverse_prompts | V2 architecture (Phase 5-7) | Storyboard references actual processed assets, not LLM hallucinations |
| Unstructured audio in prompts | Separate audio manifest with dialogue mapping | V2 audio manifest addition | Enables character-voice consistency, layered audio composition |
| Hardcoded JSON parsing | Gemini native schema validation | Gemini 2.5 (early 2026) | No more regex extraction, automatic validation, temperature-based retry |
| All-in-one storyboard schema | Separate scene/audio manifests | V2 manifest system | Cleaner separation, enables independent audio/visual iteration |

**Deprecated/outdated:**
- OpenAPI 3.0 subset for schema: Gemini 2.5 now uses full JSON Schema
- Manual character consistency prompts: Asset Registry provides canonical descriptions via reverse_prompts
- Generic "add music" instructions: Structured MusicDirection schema with instruments, tempo, transition

**Emerging patterns (professional VFX to AI video):**
- Shot breakdown manifests: Traditional VFX uses tools like ShotGrid. V2 uses LLM-generated SceneManifest with similar structure.
- Audio cue sheets: Film sound departments use spotting sheets. SceneAudioManifest captures same information.
- Asset libraries: Studios maintain tagged asset databases. Phase 5 Asset Registry is the AI-native equivalent.

## Open Questions

1. **LLM consistency with 20+ assets in context**
   - What we know: Gemini 2.5 Pro/Flash handle long contexts well. V2 docs show 11-asset example.
   - What's unclear: Performance with 30-40 assets, especially quality of asset tag references.
   - Recommendation: Implement asset filtering (quality threshold ≥6.0) and monitor tag accuracy. Consider splitting very large manifests into category subsets (characters-only pass, then environments pass).

2. **New asset declarations vs. manifested assets**
   - What we know: LLM can declare new assets not in registry via `new_asset_declarations` field.
   - What's unclear: How these integrate with Phase 6 keyframe generation. Should they trigger inline manifesting?
   - Recommendation: Phase 7 stores declarations as structured data. Phase 8 (keyframe) can optionally generate these as one-off assets or flag for user review. Start with logging/flagging, add auto-generation in Phase 9.

3. **Audio manifest usage by Veo**
   - What we know: Veo 3.1 accepts dialogue/SFX/ambient in prompts. Format is known.
   - What's unclear: Quality of multi-speaker dialogue with character tag mapping. Does Veo preserve speaker identity?
   - Recommendation: Phase 7 captures the manifest structure. Phase 8 formats for Veo. Expect iteration on prompt formatting based on actual Veo output quality. Audio is "best effort" guided generation, not guaranteed precision.

4. **Manifest validation post-generation**
   - What we know: Schema validation ensures structure. But does content match intent?
   - What's unclear: Should we add semantic validation (all asset_tags exist in registry, composition.focal_point references valid tag)?
   - Recommendation: Add post-validation step after storyboard generation. Log warnings for unrecognized tags. Don't block — let user review. Phase 6 (CV analysis) will later validate what was actually generated vs. manifest intent.

## Sources

### Primary (HIGH confidence)

**Gemini Structured Output:**
- [Structured outputs | Gemini API | Google AI for Developers](https://ai.google.dev/gemini-api/docs/structured-output) - JSON schema support, Pydantic integration
- [Google announces support for JSON Schema in Gemini API](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/) - 2026 announcement, full schema support

**Veo Audio Prompting:**
- [Ultimate prompting guide for Veo 3.1 | Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1) - Official audio prompt format (dialogue quotes, SFX prefix, ambient description)
- [Veo on Vertex AI video generation prompt guide](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide) - Current Veo 3.1 capabilities

**V2 Architecture (Internal):**
- `/home/ubuntu/work/video-pipeline/docs/v2-manifest.md` - SceneManifest and SceneAudioManifest schemas (lines 412-579)
- `/home/ubuntu/work/video-pipeline/docs/v2-pipe-optimization.md` - Full V2 pipeline flow with manifest integration (lines 481-1097)

### Secondary (MEDIUM confidence)

**VFX Pipeline Practices:**
- [VFX Pipeline: A Complete Guide | MASV](https://massive.io/workflow/vfx-pipeline/) - Shot breakdowns, asset management, layout phase
- [Visual Effects Pipeline: Complete Guide | CADA](https://cada-edu.com/guides/visual-effects-pipeline-guide-to-vfx-process) - Previs, asset tracking, shot manifests

**Audio Production Workflows:**
- [Spotting for Sound Design | SOUNDCLASS](https://soundclass.weebly.com/6-spotting-for-sound-design.html) - Audio cue sheets, SFX timing, ambient layers
- [The Ultimate Guide To Audio Post & Sound Design | 344 Audio](https://www.344audio.com/post/the-ultimate-guide-to-audio-post-production-sound-design) - Multi-layered ambient sound (8+ tracks), dialogue mapping

### Tertiary (LOW confidence)

- AI video generation research (arXiv papers) - Emerging patterns for structured scene representation. Not yet standardized.

## Metadata

**Confidence breakdown:**
- Gemini structured output: HIGH - Official docs, working implementation in Phase 2
- Scene manifest schemas: HIGH - Fully defined in V2 docs with examples
- Audio manifest schemas: HIGH - Defined in v2-manifest.md with Veo 3.1 prompt format
- VFX workflow mapping: MEDIUM - Analogies to professional pipelines, not exact matches
- Audio quality expectations: MEDIUM - Veo audio is prompt-guided, not precision-controlled

**Research date:** 2026-02-16
**Valid until:** 30 days (stable technology stack, well-documented APIs)

**Key dependencies validated:**
- Gemini 2.5 JSON schema support: ✅ Confirmed in official docs
- Veo 3.1 audio prompt format: ✅ Confirmed in prompting guide
- Existing storyboard.py pattern: ✅ Verified in codebase (tenacity retry, Pydantic validation)
- Asset Registry structure: ✅ Implemented in Phase 5 (reverse_prompt, visual_description, quality_score fields exist)
- SQLAlchemy async support: ✅ Already used throughout project
