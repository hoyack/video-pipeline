# Phase 10: Adaptive Prompt Rewriting - Research

**Researched:** 2026-02-16
**Domain:** LLM-driven prompt assembly, Gemini structured output, pipeline integration
**Confidence:** HIGH

## Summary

Phase 10 adds a dedicated Gemini rewriter call that fires **before each keyframe and video generation**, replacing the static storyboard prompts with dynamically assembled prompts. The rewriter synthesizes five inputs: (1) the original storyboard prompt, (2) manifest metadata (shot type, camera movement, placements), (3) asset reverse_prompts from the Asset Registry, (4) continuity patches from the previous scene's CV analysis, and (5) audio direction from the scene's audio manifest.

The core challenge is **integration positioning** — the rewriter must hook into both `keyframes.py` (before Imagen calls) and `video_gen.py` (before Veo calls), using different prompt construction logic for each. For keyframes, the rewriter produces a detailed static image prompt (cinematography formula for Imagen). For video, it produces a motion-focused prompt with audio direction embedded (dialogue in quotes, SFX:, Ambient: notation for Veo). Both rewritten prompts are stored in new `scene_manifests` columns (`rewritten_keyframe_prompt`, `rewritten_video_prompt`) separate from the originals, which are preserved verbatim.

The rewriter also performs reference selection with reasoning — it examines the full asset registry and explains *why* each of the 3 reference images was chosen. This duplicates some of Phase 8's deterministic selection logic, but the LLM selection can incorporate manifest context that the rule-based system cannot. The design decision here is whether to run both (LLM validates/overrides rule-based) or use the LLM as the sole selector.

**Primary recommendation:** Implement as a new `PromptRewriterService` that wraps Gemini 2.5 Flash with `response_mime_type="application/json"`, called from both `keyframes.py` and `video_gen.py` after scene manifest loading. Store rewritten prompts in two new `SceneManifest` columns. Use the LLM rewriter for keyframes and video independently (they have different formula requirements). The Phase 8 rule-based reference selector remains as the fallback when no manifest exists.

## Standard Stack

### Core (Already Installed — No New Dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| google-genai | Current | Gemini 2.5 Flash for rewriting | Already used throughout — `get_vertex_client()`, structured output via response_schema |
| pydantic | v2 | Response schema validation, structured LLM output | Already used throughout — all schemas are pydantic BaseModel |
| SQLAlchemy async | 2.0+ | Persist rewritten prompts to scene_manifests | Already used throughout |
| tenacity | Current | Retry logic on LLM failures | Already used in storyboard.py, keyframes.py, video_gen.py |

### No New Packages Needed

Phase 10 is pure orchestration — assembling existing data (manifest JSON, asset reverse_prompts, CV analysis results) and calling Gemini. All libraries are already installed.

**Installation:**
```bash
# No new packages — uses existing stack
```

## Architecture Patterns

### Recommended Project Structure

```
backend/vidpipe/
├── services/
│   └── prompt_rewriter.py        # NEW: PromptRewriterService
├── schemas/
│   └── prompt_rewrite.py         # NEW: RewrittenPromptOutput pydantic schema
├── pipeline/
│   ├── keyframes.py              # MODIFIED: call rewriter before generation
│   └── video_gen.py              # MODIFIED: call rewriter before Veo submission
└── db/
    └── models.py                 # MODIFIED: add 2 columns to SceneManifest
```

Migration:
```
backend/
└── migrate_phase10.sql           # ALTER TABLE scene_manifests ADD COLUMN ...
```

### Pattern 1: Prompt Assembly Inputs

The rewriter receives five structured inputs per scene:

```
INPUT 1: Original storyboard prompt
  scene.start_frame_prompt   (for keyframe rewrite)
  scene.video_motion_prompt  (for video rewrite)

INPUT 2: Manifest metadata (from SceneManifest.manifest_json)
  composition.shot_type, camera_movement, focal_point
  placements: [{asset_tag, role, position, action, expression, wardrobe_note}]
  continuity_notes (storyboard-level)

INPUT 3: Asset reverse_prompts (from Asset Registry)
  For each asset_tag in placements → lookup Asset.reverse_prompt
  Also Asset.visual_description if quality >= 7.0

INPUT 4: Continuity patch (from PREVIOUS scene's SceneManifest)
  scene_manifests[N-1].cv_analysis_json.continuity_issues
  scene_manifests[N-1].cv_analysis_json.overall_scene_description
  scene_manifests[N-1].continuity_score

INPUT 5: Audio direction (from SceneAudioManifest)
  dialogue_lines: [{speaker_tag, line, delivery, timing}]
  sfx: [{effect, trigger, timing, volume}]
  ambient: {base_layer, environmental}
  music: {style, mood, tempo, transition}
```

**Source:** Project codebase — `storyboard_enhanced.py` schemas, `models.py` SceneManifest/SceneAudioManifest, `cv_analysis_service.py` CVAnalysisResult.

### Pattern 2: Rewriter Service Structure

```python
# Source: project codebase patterns (storyboard.py, reverse_prompt_service.py)
from pydantic import BaseModel, Field
from typing import Optional
from google.genai import types

class RewrittenKeyframePromptOutput(BaseModel):
    """Structured output from keyframe prompt rewriter."""
    rewritten_prompt: str = Field(
        description="Final keyframe generation prompt, under 400 words, "
        "following [Cinematography]+[Subject]+[Action]+[Context]+[Style] formula"
    )
    selected_reference_tags: list[str] = Field(
        description="Exactly the manifest_tags of the 3 assets selected as references, "
        "ordered by priority (most important first)"
    )
    reference_reasoning: str = Field(
        description="One sentence explaining why these 3 references were chosen"
    )
    continuity_applied: Optional[str] = Field(
        default=None,
        description="Summary of continuity corrections applied (empty if scene 0)"
    )


class RewrittenVideoPromptOutput(BaseModel):
    """Structured output from video prompt rewriter."""
    rewritten_prompt: str = Field(
        description="Final video generation prompt, under 500 words. Motion-focused. "
        "Audio direction embedded: dialogue in quotes, SFX:, Ambient:, Music:"
    )
    selected_reference_tags: list[str] = Field(
        description="Exactly the manifest_tags of the 3 assets selected as references"
    )
    reference_reasoning: str = Field(
        description="One sentence explaining why these 3 references were chosen"
    )
    continuity_applied: Optional[str] = Field(
        default=None,
        description="Summary of continuity corrections applied"
    )


class PromptRewriterService:
    """Assembles final generation prompts by injecting manifest metadata,
    asset reverse_prompts, continuity corrections, and audio direction.

    Called once per scene for keyframe generation, and once per scene
    for video generation (separate calls with different formula).
    """

    # Rate limiting: reuse Phase 5 pattern (5 concurrent requests)
    _semaphore = asyncio.Semaphore(5)

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_vertex_client()
        return self._client

    async def rewrite_keyframe_prompt(
        self,
        scene: Scene,
        scene_manifest_json: dict,
        placed_assets: list[Asset],
        previous_cv_analysis: dict | None,
        all_assets: list[Asset],
    ) -> RewrittenKeyframePromptOutput:
        """Rewrite keyframe prompt with manifest enrichment."""
        async with self._semaphore:
            system_prompt = self._build_keyframe_system_prompt()
            user_context = self._assemble_keyframe_context(
                scene, scene_manifest_json, placed_assets, previous_cv_analysis, all_assets
            )
            return await self._call_rewriter(
                system_prompt, user_context, RewrittenKeyframePromptOutput
            )

    async def rewrite_video_prompt(
        self,
        scene: Scene,
        scene_manifest_json: dict,
        audio_manifest_json: dict | None,
        placed_assets: list[Asset],
        previous_cv_analysis: dict | None,
        all_assets: list[Asset],
    ) -> RewrittenVideoPromptOutput:
        """Rewrite video prompt with manifest enrichment + audio direction."""
        async with self._semaphore:
            system_prompt = self._build_video_system_prompt()
            user_context = self._assemble_video_context(
                scene, scene_manifest_json, audio_manifest_json,
                placed_assets, previous_cv_analysis, all_assets
            )
            return await self._call_rewriter(
                system_prompt, user_context, RewrittenVideoPromptOutput
            )

    async def _call_rewriter(self, system_prompt: str, user_context: str, schema):
        """Call Gemini 2.5 Flash with structured output schema."""
        from tenacity import retry, stop_after_attempt, retry_if_exception_type
        import json

        @retry(stop=stop_after_attempt(3), retry=retry_if_exception_type((json.JSONDecodeError, Exception)))
        async def _attempt():
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[f"{system_prompt}\n\n{user_context}"],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.4,  # Lower than storyboard (0.7) — less creative, more precise
                )
            )
            return schema.model_validate_json(response.text)

        return await _attempt()
```

**Source:** Pattern adapted from `storyboard.py` Gemini structured output approach and `reverse_prompt_service.py` retry pattern.

### Pattern 3: Keyframe System Prompt

```python
KEYFRAME_REWRITER_SYSTEM_PROMPT = """You are a professional cinematographer and VFX director.
You are assembling a static image generation prompt for a keyframe.

Your output must follow this EXACT formula:
[Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]

RULES:
1. Start with shot type and camera details from the scene manifest
2. Describe subjects using their EXACT reverse_prompt details (not what you imagine)
3. Include spatial positions from manifest placements (left/right/center/foreground/background)
4. Include wardrobe_note details for continuity
5. Apply any continuity corrections from the previous scene's CV analysis
6. Preserve the original prompt's narrative intent, but upgrade its visual specificity
7. Keep under 400 words (Imagen sweet spot for keyframes)
8. Select exactly 3 reference asset tags — explain why

WHAT NOT TO DO:
- Do not re-invent character descriptions (use reverse_prompt verbatim)
- Do not ignore continuity corrections
- Do not exceed 400 words
- Do not reference audio (keyframes are static images)
"""

VIDEO_REWRITER_SYSTEM_PROMPT = """You are a professional cinematographer and VFX director.
You are assembling a video generation prompt for Veo 3.1.

Your output must follow this EXACT formula:
[Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance] + [Audio Direction]

RULES:
1. Describe MOTION only — the reference images provide visual context
2. Camera movement from manifest (dolly, pan, static, etc.)
3. Character actions and expressions from manifest placements
4. Embed audio direction at the end:
   - Dialogue: Character name says "exact words" (delivery note)
   - SFX: brief description at timing
   - Ambient: base soundscape
   - Music: style, mood, transition
5. Apply continuity corrections from CV analysis of previous scene
6. Keep under 500 words (Veo 3.1 prompt sweet spot)
7. Select exactly 3 reference asset tags — explain why

CRITICAL:
- Include ALL audio direction from the manifest (Veo generates audio from this)
- Do not re-describe visual appearance (reference images handle that)
- Motion prompt should tell Veo what MOVES, not how things look
"""
```

**Source:** v2-pipe-optimization.md section 6 "The LLM Rewriter" for prompt formula and requirements.

### Pattern 4: Continuity Context Assembly

```python
def _build_continuity_patch(previous_cv_analysis: dict | None, scene_index: int) -> str:
    """Build continuity correction block from previous scene's CV analysis."""
    if scene_index == 0 or previous_cv_analysis is None:
        return "CONTINUITY: This is the first scene — no previous scene continuity needed."

    semantic = previous_cv_analysis.get("semantic_analysis", {})
    issues = semantic.get("continuity_issues", [])
    scene_desc = semantic.get("overall_scene_description", "")
    score = previous_cv_analysis.get("continuity_score", 0.0)

    lines = [
        f"CONTINUITY PATCH (from Scene {scene_index - 1} CV Analysis):",
        f"Previous scene continuity score: {score:.1f}/10",
    ]

    if scene_desc:
        lines.append(f"What scene {scene_index - 1} actually showed: {scene_desc}")

    if issues:
        lines.append("Issues flagged (MUST address in this scene):")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("No continuity issues flagged — maintain current appearance.")

    return "\n".join(lines)
```

### Pattern 5: Integration Points in Pipeline

**Keyframes pipeline integration** (modify `keyframes.py`):

```python
# In generate_keyframes() / _generate_keyframe_for_scene(), BEFORE the Imagen call:
# After loading scene_manifest_row and before building prompt:

if project.manifest_id and scene_manifest_row and scene_manifest_row.manifest_json:
    from vidpipe.services.prompt_rewriter import PromptRewriterService

    # Load placed assets
    all_assets = await manifest_service.load_manifest_assets(session, project.manifest_id)
    placed_assets = _resolve_placed_assets(scene_manifest_row.manifest_json, all_assets)

    # Load previous scene's CV analysis for continuity
    previous_cv = None
    if scene.scene_index > 0:
        prev_sm = await _load_scene_manifest(session, project.id, scene.scene_index - 1)
        if prev_sm:
            previous_cv = prev_sm.cv_analysis_json

    rewriter = PromptRewriterService()
    result = await rewriter.rewrite_keyframe_prompt(
        scene=scene,
        scene_manifest_json=scene_manifest_row.manifest_json,
        placed_assets=placed_assets,
        previous_cv_analysis=previous_cv,
        all_assets=all_assets,
    )

    # Store rewritten prompt in scene manifest
    scene_manifest_row.rewritten_keyframe_prompt = result.rewritten_prompt
    await session.commit()

    # Use rewritten prompt for generation (override storyboard prompt)
    start_prompt = result.rewritten_prompt
    # Note: end_frame inherits or is derived from start via image conditioning
else:
    start_prompt = scene.start_frame_prompt  # Fallback: no manifest
```

**Video gen pipeline integration** (modify `video_gen.py`):

```python
# In _generate_video_for_scene(), AFTER loading scene_manifest_row and selected_refs,
# BEFORE building video_prompt string:

if project.manifest_id and scene_manifest_row and scene_manifest_row.manifest_json:
    # Load audio manifest
    from vidpipe.db.models import SceneAudioManifest as SceneAudioManifestModel
    audio_result = await session.execute(
        select(SceneAudioManifestModel).where(
            SceneAudioManifestModel.project_id == project.id,
            SceneAudioManifestModel.scene_index == scene.scene_index
        )
    )
    audio_manifest_row = audio_result.scalar_one_or_none()

    # Load previous CV analysis
    previous_cv = None
    if scene.scene_index > 0:
        prev_sm = await _load_scene_manifest(session, project.id, scene.scene_index - 1)
        if prev_sm:
            previous_cv = prev_sm.cv_analysis_json

    all_assets = await manifest_service.load_manifest_assets(session, project.manifest_id)
    placed_assets = _resolve_placed_assets(scene_manifest_row.manifest_json, all_assets)

    rewriter = PromptRewriterService()
    result = await rewriter.rewrite_video_prompt(
        scene=scene,
        scene_manifest_json=scene_manifest_row.manifest_json,
        audio_manifest_json=(
            {
                "dialogue_lines": audio_manifest_row.dialogue_json,
                "sfx": audio_manifest_row.sfx_json,
                "ambient": audio_manifest_row.ambient_json,
                "music": audio_manifest_row.music_json,
            }
            if audio_manifest_row else None
        ),
        placed_assets=placed_assets,
        previous_cv_analysis=previous_cv,
        all_assets=all_assets,
    )

    # Store rewritten video prompt
    scene_manifest_row.rewritten_video_prompt = result.rewritten_prompt
    await session.commit()

    video_prompt = result.rewritten_prompt
else:
    # Fallback: original pipeline behavior (no manifest)
    video_prompt = (
        f"{scene.video_motion_prompt}. "
        f"Maintain the visual style shown in the source frames."
    )
```

### Pattern 6: Database Schema Addition

Two new columns needed in `scene_manifests`:

```sql
-- migrate_phase10.sql
ALTER TABLE scene_manifests
    ADD COLUMN rewritten_keyframe_prompt TEXT,
    ADD COLUMN rewritten_video_prompt TEXT;
```

SQLAlchemy model update (`models.py`):

```python
class SceneManifest(Base):
    # ... existing columns ...

    # Phase 10: Adaptive Prompt Rewriting
    rewritten_keyframe_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rewritten_video_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
```

**Source:** v2-pipe-optimization.md line 1262-1263 documents these exact column names (`rewritten_keyframe_prompt`, `rewritten_video_prompt`) as part of the V2 schema design.

### Anti-Patterns to Avoid

- **Rewriting in the storyboard phase:** Storyboard runs once before any generation. Rewriting must happen per-scene, immediately before each generation call, so it incorporates CV analysis from previous scenes.
- **Storing rewritten prompts in the `Scene` model:** The `Scene` model stores original storyboard prompts. Rewritten prompts go in `SceneManifest` to keep them separate and allow comparison. This preserves the original for debugging.
- **Single rewriter call for both keyframe and video:** Keyframe prompts follow an image composition formula; video prompts are motion-focused with audio direction. They are different enough to warrant separate system prompts and schemas.
- **LLM selecting references AND rule-based selecting references:** Phase 8's `select_references_for_scene()` is deterministic and fast. Phase 10 adds LLM reasoning on top. Use Phase 8 as the fallback (no manifest path). For manifest projects, the LLM's `selected_reference_tags` can override Phase 8's selection for the 3 reference images loaded in `video_gen.py`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Structured LLM output | Manual JSON parsing from raw text | Gemini `response_schema=` (pydantic) | Already used in storyboard.py; handles malformed JSON, retries automatically |
| Prompt length enforcement | Manual truncation | LLM instruction + word limit in schema description | LLM follows constraints when explicitly stated; truncation breaks sentence structure |
| Continuity state machine | Complex state tracking | `scene_manifests.cv_analysis_json` (Phase 9 output) | Phase 9 already produces structured continuity data per scene — read it |
| Rate limiting | Custom queue | `asyncio.Semaphore(5)` | Already established pattern from Phase 5 reverse-prompting |
| Asset format block | Custom formatter | Extend `format_asset_registry()` from `manifest_service.py` | Existing function already formats assets for LLM — add reverse_prompt verbosity for rewriter |

**Key insight:** The value of Phase 10 is the **assembly logic** (what to inject and in what order), not the underlying mechanics. All mechanics (Gemini calls, structured output, asset loading, semaphore limiting) are already proven patterns in this codebase.

## Common Pitfalls

### Pitfall 1: Rewritten Prompt Overrides Safety Prefixes

**What goes wrong:** `video_gen.py` currently builds `video_prompt` as `f"{_VIDEO_SAFETY_PREFIXES[safety_level]}{scene.video_motion_prompt}..."`. When Phase 10 replaces `scene.video_motion_prompt` with the rewritten prompt, the safety prefix concatenation logic is still needed.

**Why it happens:** The rewritten prompt replaces `scene.video_motion_prompt` but the safety level escalation loop in `_generate_video_for_scene()` prepends safety prefixes on each retry. If the rewritten prompt variable is set at the start, the safety prefix still needs to be prepended on escalation attempts.

**How to avoid:** Store the rewritten prompt in a local variable (`base_video_prompt`). In the escalation loop, compute `video_prompt = f"{_VIDEO_SAFETY_PREFIXES[safety_level]}{base_video_prompt}"`. The rewritten prompt is the base; safety prefixes still stack on top.

**Warning signs:** Content policy rejections succeed at level 0 but not at level 1 (prompt doesn't actually get safety prefix because the base variable was overwritten).

### Pitfall 2: Continuity Patch for Scene 0

**What goes wrong:** Attempting to load `scene_manifests[scene_index - 1]` when `scene_index == 0` causes a query for `scene_index == -1`, which returns `None` or crashes.

**Why it happens:** Continuity checks compare N-1 with N. Scene 0 has no N-1.

**How to avoid:** Always guard with `if scene.scene_index == 0: previous_cv = None`. The system prompt instructs the LLM "this is the first scene — no continuity needed" when `previous_cv` is None.

**Warning signs:** SQLAlchemy warnings about invalid scene_index, or LLM receiving empty continuity block and hallucinating constraints.

### Pitfall 3: Rewriter Called on Non-Manifest Projects

**What goes wrong:** `PromptRewriterService.rewrite_video_prompt()` called for projects without a manifest. No manifest means no `scene_manifest_row`, no asset tags, no continuity data. The rewriter would produce a prompt with no asset enrichment — worse than the original static prompt because it loses the storyboard's carefully crafted visual language.

**Why it happens:** Forgetting to guard with `if project.manifest_id:` before calling the rewriter.

**How to avoid:** The rewriter **only activates for manifest projects**. Non-manifest projects continue using `scene.video_motion_prompt` directly (existing behavior). This is the same guard pattern already in `video_gen.py` for Phase 8's reference selection.

**Warning signs:** Rewriter produces generic prompts like "a scene with characters" because no asset data was injected.

### Pitfall 4: Reverse Prompt Token Overflow

**What goes wrong:** Including full `reverse_prompt` for all placed assets plus continuity patch plus audio direction pushes the context over Gemini's token limit. For 6 placed assets with 500-char reverse_prompts each = 3000 chars of asset data before any other context.

**Why it happens:** `reverse_prompt` fields can be 300-500 words each. Multiple assets quickly overflow reasonable context.

**How to avoid:** Limit reverse_prompt injection to **placed assets only** (not all registry assets). Truncate reverse_prompts to 200 characters in the rewriter context (consistent with `format_asset_registry()` which already does 200-char truncation). Only include `visual_description` for quality_score >= 7.0 and truncate to 150 chars (same rule as `format_asset_registry()`).

**Warning signs:** Gemini returning 429 "context too long" or structured output validation failures due to truncated responses.

### Pitfall 5: LLM Reference Selection Conflicts with Phase 8 Selection

**What goes wrong:** Phase 8 (`reference_selection.py`) deterministically selects 3 assets and stores their URLs in `veo_ref_images`. Phase 10's LLM rewriter also selects 3 assets. If the LLM selects different assets than Phase 8, the `veo_ref_images` list and the `rewritten_video_prompt` reference different assets — the prompt says "see the attached CHAR_01 face crop" but the actual reference images are ENV_01, OBJ_01, CHAR_02.

**Why it happens:** Two independent selection systems producing different outputs without coordination.

**How to avoid:** Phase 10 LLM selection output (`selected_reference_tags`) is used to **override Phase 8's selection** when building `veo_ref_images`. After the LLM rewriter runs, rebuild `veo_ref_images` from `result.selected_reference_tags` instead of Phase 8's deterministic output. Store LLM's selection in `scene_manifest_row.selected_reference_tags` (overwriting Phase 8's stored tags).

**Warning signs:** Reference images and prompt mentioning different assets. Debugging confusion about which assets were actually passed to Veo.

### Pitfall 6: Missing `rewritten_keyframe_prompt` for End Frame

**What goes wrong:** `keyframes.py` generates both `start_frame_prompt` and `end_frame_prompt`. The rewriter is called once. The end frame is generated via image-conditioned generation (`_generate_image_conditioned`), not text-to-image. The rewritten prompt applies to the start frame only.

**Why it happens:** Phase 10 naturally maps to the start frame. End frame generation is conditioned on the start frame image, not directly on a text prompt.

**How to avoid:** The rewritten keyframe prompt applies to **start frame generation only**. End frame continues to use `scene.end_frame_prompt` as the conditioning prompt (passed to `_generate_image_conditioned`). The end frame prompt could also be rewritten (separate rewriter call targeting end state), but this is not required by Phase 10's success criteria — defer if needed.

## Code Examples

Verified patterns from project codebase:

### Gemini Structured Output (from storyboard.py)

```python
# Source: backend/vidpipe/pipeline/storyboard.py — generate_with_retry()
response = await client.aio.models.generate_content(
    model=model_id,  # "gemini-2.5-flash" for rewriter
    contents=[full_prompt],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RewrittenVideoPromptOutput,  # Pydantic model
        temperature=0.4,  # Lower than storyboard's 0.7
    )
)
result = RewrittenVideoPromptOutput.model_validate_json(response.text)
```

### Asset Block Assembly (from manifest_service.py)

```python
# Source: backend/vidpipe/services/manifest_service.py — format_asset_registry()
# Reuse the same pattern but only for PLACED assets, not all registry assets:

def format_placed_assets_for_rewriter(
    placed_asset_tags: list[str],
    all_assets: list[Asset],
    scene_manifest_json: dict,
) -> str:
    """Format placed asset descriptions for LLM rewriter context."""
    asset_map = {a.manifest_tag: a for a in all_assets}
    placements = scene_manifest_json.get("placements", [])

    lines = ["PLACED ASSETS IN THIS SCENE:", "━" * 40]
    for placement in placements:
        tag = placement.get("asset_tag")
        asset = asset_map.get(tag)
        if not asset:
            lines.append(f"[{tag}] — asset not found in registry")
            continue

        quality_str = f"{asset.quality_score:.1f}/10" if asset.quality_score else "N/A"
        role = placement.get("role", "unknown")
        position = placement.get("position", "")
        action = placement.get("action", "")
        wardrobe = placement.get("wardrobe_note", "")
        expression = placement.get("expression", "")

        lines.append(f"[{tag}] \"{asset.name}\" — {role} at {position} (quality: {quality_str})")
        if action:
            lines.append(f"  Action: {action}")
        if expression:
            lines.append(f"  Expression: {expression}")
        if wardrobe:
            lines.append(f"  Wardrobe: {wardrobe}")
        if asset.reverse_prompt:
            rp = asset.reverse_prompt[:200] + ("..." if len(asset.reverse_prompt) > 200 else "")
            lines.append(f"  Visual description: {rp}")
        lines.append("")

    return "\n".join(lines)
```

### Continuity Load Pattern

```python
# Source: adapted from video_gen.py SceneManifest query pattern
async def _load_previous_cv_analysis(
    session: AsyncSession,
    project_id: uuid.UUID,
    current_scene_index: int,
) -> dict | None:
    """Load CV analysis from previous scene for continuity patching."""
    if current_scene_index == 0:
        return None

    from vidpipe.db.models import SceneManifest as SceneManifestModel
    from sqlalchemy import select

    result = await session.execute(
        select(SceneManifestModel).where(
            SceneManifestModel.project_id == project_id,
            SceneManifestModel.scene_index == current_scene_index - 1
        )
    )
    prev_manifest = result.scalar_one_or_none()

    if prev_manifest and prev_manifest.cv_analysis_json:
        return prev_manifest.cv_analysis_json
    return None
```

### Audio Direction Formatting

```python
def format_audio_direction(audio_manifest_json: dict | None) -> str:
    """Format audio manifest into prompt-injectable audio direction block."""
    if not audio_manifest_json:
        return "AUDIO: No audio direction specified."

    lines = ["AUDIO DIRECTION:"]

    dialogue = audio_manifest_json.get("dialogue_lines") or []
    for d in dialogue:
        speaker = d.get("speaker_name", d.get("speaker_tag", "Character"))
        line = d.get("line", "")
        delivery = d.get("delivery", "")
        timing = d.get("timing", "")
        delivery_note = f" ({delivery})" if delivery else ""
        lines.append(f'  Dialogue ({timing}): {speaker} says{delivery_note}: "{line}"')

    sfx = audio_manifest_json.get("sfx") or []
    for s in sfx:
        effect = s.get("effect", "")
        timing = s.get("timing", "")
        volume = s.get("volume", "subtle")
        lines.append(f"  SFX ({timing}, {volume}): {effect}")

    ambient = audio_manifest_json.get("ambient") or {}
    if ambient:
        base = ambient.get("base_layer", "")
        env = ambient.get("environmental", "")
        if base:
            lines.append(f"  Ambient: {base}" + (f", {env}" if env else ""))

    music = audio_manifest_json.get("music") or {}
    if music:
        style = music.get("style", "")
        mood = music.get("mood", "")
        tempo = music.get("tempo", "")
        transition = music.get("transition", "")
        lines.append(f"  Music: {style}, {mood}, {tempo} tempo. {transition}")

    return "\n".join(lines)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static storyboard prompts used verbatim | Dynamic per-scene rewriting with asset injection | Phase 10 | Character descriptions become precise (reverse_prompt verbatim) rather than storyboard's approximation |
| Phase 8 deterministic reference selection | LLM-reasoned reference selection | Phase 10 | References chosen with scene context, not just shot-type rules |
| Audio direction in separate manifest (unused at generation) | Audio direction embedded in video prompt | Phase 10 | Veo 3+ actually generates dialogue/SFX from the prompt text |
| Continuity enforced by storyboard structure only | Continuity enforced by CV analysis patch per scene | Phase 10 | "Marcus's coat is unbuttoned (was unbuttoned in scene 2)" — actual visual state tracking |

**Not deprecated:**
- Phase 8 `select_references_for_scene()` — still used as fallback for non-manifest projects, and LLM selection overrides it for manifest projects
- `scene.video_motion_prompt` — still stored, still used as fallback, and fed into the rewriter as "original intent"

## Open Questions

### 1. Should the keyframe end_frame also be rewritten?

**What we know:** Start frame uses text-to-image generation (Imagen). End frame uses image-conditioned generation from start frame. The current `scene.end_frame_prompt` is passed as the conditioning prompt.

**What's unclear:** Phase 10 success criteria only mention `rewritten_keyframe_prompt` (singular). Does this cover just start frame, or both start and end?

**Recommendation:** Implement start frame rewriting only for Phase 10. The rewritten start frame already embeds asset descriptions and continuity. End frame conditioning inherits the visual state from the generated start frame. If end frame prompt quality is a problem, defer to a future enhancement.

### 2. What is the token budget for the rewriter context?

**What we know:** Gemini 2.5 Flash has a 1M token context window. The rewriter input (manifest JSON + asset blocks + audio direction + continuity patch) will be 2,000-5,000 tokens. This is well within limits.

**What's unclear:** How verbose should the rewriter context be? More context means better prompts but higher cost per call.

**Recommendation:** Conservative starting point — truncate reverse_prompts to 200 chars (consistent with `format_asset_registry()`), only include `visual_description` for quality >= 7.0. This keeps context under 3,000 tokens. Monitor Gemini response quality and expand if needed.

### 3. Failure mode: rewriter service unavailable or times out

**What we know:** If the Gemini rewriter call fails after retries, the keyframe or video generation should not block.

**What's unclear:** Should a rewriter failure cause the scene to fail, or gracefully fall back to the original storyboard prompt?

**Recommendation:** Graceful fallback — if `PromptRewriterService.rewrite_video_prompt()` raises an exception after retries, log a warning and fall back to the original `scene.video_motion_prompt`. This matches the Phase 9 CV analysis pattern (`_run_post_generation_analysis()` wraps everything in try/except and logs warning on failure). The rewriter is an enhancement, not a hard dependency.

### 4. Should reverse_prompts be updated from CV analysis?

**What we know:** Success criterion 5 states "Reverse prompts refined based on what models actually produce (not just initial descriptions)." Phase 9 CV analysis produces `overall_scene_description` and `continuity_issues` per scene. But `Asset.reverse_prompt` is set during the manifesting phase (Phase 5) and reflects the uploaded reference image, not what Veo generated.

**What's unclear:** Does "refine reverse_prompts" mean: (a) update `Asset.reverse_prompt` in the database based on what CV analysis saw, or (b) inject CV analysis findings as additional context in the rewriter prompt?

**Recommendation:** Approach (b) — inject CV analysis as the continuity patch (already implemented above), not as updates to `Asset.reverse_prompt`. Updating `Asset.reverse_prompt` would change the ground truth of the asset permanently and could cause drift across multiple generations. The continuity patch is ephemeral context for this specific project.

### 5. Cost per scene

**What we know:** Gemini 2.5 Flash pricing: $0.10/1M input tokens, $0.40/1M output tokens (as of early 2026). Each rewriter call: ~3,000 input tokens + ~500 output tokens. That's $0.0003 input + $0.0002 output = ~$0.0005 per call.

**What's unclear:** Is this 1 call per scene (video only) or 2 calls per scene (keyframe + video)?

**Recommendation:** 2 calls per scene (one for keyframe, one for video). Total cost: ~$0.001 per scene, or ~$0.01 for a 10-scene project. The v2-pipe-optimization.md estimated "$0.01-0.03 per scene" which aligns (their estimate probably used Gemini 2.5 Pro pricing).

## Sources

### Primary (HIGH confidence)

- Project codebase: `backend/vidpipe/pipeline/video_gen.py` — Integration points, safety prefix pattern, SceneManifest loading pattern
- Project codebase: `backend/vidpipe/pipeline/keyframes.py` — Keyframe generation flow
- Project codebase: `backend/vidpipe/pipeline/storyboard.py` — Gemini structured output pattern, temperature settings, retry logic
- Project codebase: `backend/vidpipe/services/manifest_service.py` — `format_asset_registry()`, `load_manifest_assets()`, `format_asset_registry()` truncation rules
- Project codebase: `backend/vidpipe/schemas/storyboard_enhanced.py` — SceneManifestSchema, AssetPlacement, AudioManifest schemas
- Project codebase: `backend/vidpipe/db/models.py` — SceneManifest existing columns, Asset fields
- Project docs: `docs/v2-pipe-optimization.md` section 6 (lines 690-780) — LLM rewriter spec, prompt formula, cost estimates
- Project docs: `docs/v2-pipe-optimization.md` lines 1262-1263 — `rewritten_keyframe_prompt`, `rewritten_video_prompt` column names
- Phase 9 RESEARCH.md — CV analysis outputs available to Phase 10 (CVAnalysisResult, SemanticAnalysis schema)

### Secondary (MEDIUM confidence)

- `docs/v2-pipe-optimization.md` lines 1040-1096 — Pipeline flow diagram showing rewrite positioned before each generation step
- Phase 8 plans — reference selection already happens; Phase 10 adds LLM selection that can override

### Tertiary (LOW confidence)

- Gemini 2.5 Flash pricing ($0.10/$0.40 per 1M tokens) — verified in v2-pipe-optimization.md cost tracking section, but API pricing changes frequently; confirm at time of implementation

## Metadata

**Confidence breakdown:**
- Standard stack (Gemini Flash, pydantic, SQLAlchemy): **HIGH** — exact same stack used throughout existing codebase
- Integration points (keyframes.py, video_gen.py hooks): **HIGH** — read actual source code, integration is clear
- Database schema additions (2 new columns): **HIGH** — column names specified in v2-pipe-optimization.md, pattern clear from models.py
- Prompt formula (cinematography + subject + action + context + style): **HIGH** — documented explicitly in v2-pipe-optimization.md section 6
- Audio direction embedding format: **HIGH** — SceneAudioManifestModel schema clear, Veo prompt format documented
- Continuity patching: **HIGH** — CVAnalysisResult schema from Phase 9 research, cv_analysis_json column established
- LLM vs rule-based reference selection coordination: **MEDIUM** — design decision not fully specified in docs; recommendation is LLM overrides Phase 8 for manifest projects

**Research date:** 2026-02-16
**Valid until:** ~60 days (stable domain — Gemini API and this codebase's patterns are stable; Gemini 2.5 Flash pricing may shift)
