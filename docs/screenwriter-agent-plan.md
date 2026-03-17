# Screenwriter Agent System

A pipeline-integrated agent that transforms scene-level prompts into shot-level screenplays with explicit character assignment, narrative arc management, and dynamic shot expansion.

## The Problem

When a user writes a scene prompt like:

```
Open on @BRANDON_ONE at dawn, standing at the top of a ridge overlooking the city.
As they descend into the city, cut between iconic, stylized shots: the glint of
sunlight on the Capitol Records building, a movie premiere's flashing cameras...
```

The current system:
1. Storyboard LLM generates 3-5 shots with `start_frame_prompt` and `end_frame_prompt`
2. Shot manifests have `placements[]` with `asset_tag` references
3. Keyframe stage looks up those tags to load reference images

**The gap:** The storyboard LLM is a single-shot call that simultaneously writes the narrative, breaks it into shots, composes keyframe prompts, assigns assets to shots, designs audio, and maintains continuity. It's doing too many jobs at once, and does none of them deeply enough. Specifically:

- **Character assignment is implicit.** The LLM may or may not mention @BRANDON_ONE in each shot. There's no verification that the right characters appear in the right shots.
- **Shot expansion is destructive.** Adding a shot requires re-running the entire storyboard, losing prior work.
- **Narrative arc is unstructured.** The LLM generates shots sequentially but has no explicit model of story beats, tension curves, or emotional progression.
- **CastBinding tags don't flow to keyframes.** Shot manifest placements reference tags that only exist in the (empty) Assets table, not the CastBinding table.

## Architecture Decision: No Framework

### Research Findings

| Framework | Verdict | Why |
|-----------|---------|-----|
| **CrewAI** | Skip | ~56% token overhead vs raw calls, async conflicts with FastAPI event loop, solves a different problem (agent negotiation, not data transformation) |
| **LangGraph** | Overkill | Replaces infrastructure we already have. Good for 10+ step branching, but our pipeline is sequential |
| **AutoGen** | Wrong paradigm | Conversation-driven; we need structured pipeline steps |
| **Claude Agent SDK** | Wrong tool | Designed for autonomous exploration, not deterministic transformation |
| **DSPy** | Irrelevant | Prompt optimization framework; creative tasks lack optimization metrics |
| **PydanticAI** | Watch list | Natural upgrade path if we ever need dynamic tool routing |
| **Raw LLM chain** | **Use this** | Zero overhead, already integrated, perfect async/FastAPI/SQLAlchemy fit |

**Core insight from research:** Every production system solving script-to-shot automation (ViMax, Filmustage, Katalist, LTX Studio) uses **sequential structured pipelines** with Pydantic-validated LLM outputs — exactly what we already have. No framework adds value over `async def step(input) -> PydanticModel`.

**Decision:** Extend the existing `ScreenwriterService` + `generate_storyboard()` pattern. Each new capability is an async function that calls `LLMAdapter.generate_text(prompt, schema)`, validates output, and persists to SQLAlchemy.

## System Design

### Pipeline Flow

```
Scene.prompt (with @tags)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 1: SCRIPT ANALYSIS                              │
│ "What is this story about?"                          │
│                                                       │
│ Input:  scene.prompt, production_bible bindings       │
│ Output: ScriptAnalysis                                │
│   - narrative_summary (1-2 sentences)                │
│   - tone, genre, pacing                              │
│   - characters_referenced[] (from @tags + inferred)  │
│   - settings[] (locations mentioned or implied)      │
│   - story_beats[] (key moments in the narrative)     │
│   - emotional_arc (tension curve description)        │
│                                                       │
│ LLM: Single call, structured output                  │
│ Persists to: Scene.screenplay_context                │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 2: SHOT BREAKDOWN                               │
│ "How should this story be told visually?"            │
│                                                       │
│ Input:  ScriptAnalysis, target_shot_count,           │
│         available assets (from binding registry)     │
│ Output: ShotBreakdown                                │
│   - shots[]:                                         │
│     - shot_index                                     │
│     - beat (which story_beat this shot serves)       │
│     - narrative_intent (what this shot communicates) │
│     - characters_present[] (explicit @tag list)      │
│     - setting (location/environment)                 │
│     - time_of_day                                    │
│     - emotional_weight (0-10, for arc shaping)       │
│     - duration_hint (seconds)                        │
│                                                       │
│ LLM: Single call, structured output                  │
│ Persists to: Scene.storyboard_raw (or new table)     │
│                                                       │
│ KEY: characters_present is the AUTHORITATIVE list    │
│ of which @tags appear in each shot. This drives      │
│ downstream reference image loading.                  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 3: SHOT MANIFEST GENERATION                     │
│ "What exactly is in each frame?"                     │
│                                                       │
│ Input:  ShotBreakdown, available assets,             │
│         style guide                                  │
│ Output: Per-shot ShotManifest (existing schema)      │
│   - composition (shot_type, camera_movement)         │
│   - placements[] (asset_tag, role, position, etc.)   │
│   - audio_manifest                                   │
│                                                       │
│ This is the EXISTING storyboard manifest generation  │
│ but now it receives characters_present[] as a hard   │
│ constraint rather than leaving it to LLM discretion. │
│                                                       │
│ LLM: One call per shot (parallelizable)              │
│ Persists to: ShotManifest                            │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 4: KEYFRAME PROMPT WRITING                      │
│ "What does the camera see?"                          │
│                                                       │
│ Input:  ShotManifest, character descriptions,        │
│         previous shot continuity, style guide        │
│ Output: start_frame_prompt, end_frame_prompt,        │
│         video_motion_prompt                          │
│                                                       │
│ This is the EXISTING PromptRewriterService           │
│ but now it receives guaranteed character placements   │
│ from Step 2-3 instead of hoping the storyboard LLM  │
│ mentioned them.                                      │
│                                                       │
│ LLM: One call per shot (parallelizable)              │
│ Persists to: Shot, ShotManifest                      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ STEP 5: VALIDATION & RECONCILIATION                  │
│ "Did we miss anything?"                              │
│                                                       │
│ Input:  All shots, ScriptAnalysis                    │
│ Checks:                                              │
│   - Every @tag from scene.prompt appears in at       │
│     least one shot's characters_present              │
│   - Story beats are covered (no gaps in arc)         │
│   - Character screen time is proportional to role    │
│   - No shot has zero placements                      │
│                                                       │
│ Deterministic (no LLM needed)                        │
│ Logs warnings, does NOT block pipeline               │
└─────────────────────────────────────────────────────┘
```

### Dynamic Shot Expansion

When a user adds a shot after initial generation:

```
Existing:  [Shot 0] [Shot 1] [Shot 2] [Shot 3]
User adds: [Shot 0] [Shot 1] [Shot 2] [NEW] [Shot 3]
```

**Current behavior:** Must re-run entire storyboard (destructive).

**Proposed behavior:**

```
┌─────────────────────────────────────────────────────┐
│ SHOT EXPANSION                                       │
│                                                       │
│ Input:                                               │
│   - ScriptAnalysis (from Step 1, already persisted)  │
│   - Existing shots[] with their narrative_intent     │
│   - Insert position (between shot_index 2 and 3)    │
│   - Optional user hint ("add a shot of Brandon      │
│     walking through the backlot")                    │
│                                                       │
│ LLM determines:                                      │
│   - What story beat this new shot serves             │
│   - Which characters should appear                   │
│   - Narrative intent that bridges the surrounding    │
│     shots                                            │
│   - Adjustments to surrounding shots' transition     │
│     notes (if needed)                                │
│                                                       │
│ Output: New ShotBreakdown entry + updated            │
│         transition_notes for adjacent shots           │
│                                                       │
│ Then runs Steps 3-4 for the new shot only            │
│ (manifest generation + prompt writing)               │
└─────────────────────────────────────────────────────┘
```

This is non-destructive — existing shots keep their prompts, manifests, and any already-generated keyframes/clips. Only the new shot and its neighbors' transitions are touched.

### Shot Deletion / Reordering

Similarly, removing or reordering shots only requires:
1. Re-indexing `shot_index` values
2. Regenerating `transition_notes` for adjacent shots (single LLM call)
3. Existing keyframes/clips remain valid (they don't depend on shot_index)

## Character Assignment: The Core Innovation

The key data structure is `characters_present[]` in the Shot Breakdown (Step 2). This is an **explicit, authoritative list** of which @tags appear in each shot.

### How It Flows Downstream

```
Step 2 output:
  Shot 0: characters_present: ["BRANDON_ONE"]
  Shot 1: characters_present: []              ← scenery only
  Shot 2: characters_present: []              ← scenery only
  Shot 3: characters_present: ["BRANDON_ONE"]

Step 3 (manifest generation):
  Shot 0 placements: [{asset_tag: "BRANDON_ONE", role: "subject", ...}]
  Shot 1 placements: [{asset_tag: "CAPITOL_RECORDS", role: "environment", ...}]
  Shot 2 placements: [{asset_tag: "MOVIE_PREMIERE", role: "environment", ...}]
  Shot 3 placements: [{asset_tag: "BRANDON_ONE", role: "subject", ...}]

Keyframe generation:
  Shot 0: CastBinding fallback → loads BRANDON_ONE ActorRef images ✓
  Shot 1: No CHARACTER placements → no ref images (correct!) ✓
  Shot 2: No CHARACTER placements → no ref images (correct!) ✓
  Shot 3: CastBinding fallback → loads BRANDON_ONE ActorRef images ✓
```

### Shot-Aware CastBinding Resolution

The CastBinding fallback in `keyframes.py` should be updated to check the shot manifest's placements rather than resolving from `scene.prompt`:

```python
# Instead of:
if not ref_image_bytes_list and scene.production_bible_id:
    _cast_resolved = await resolve_tags_with_assets(
        scene.prompt,  # ← resolves ALL @tags for EVERY shot
        ...
    )

# Use:
if not ref_image_bytes_list and scene.production_bible_id:
    # Get CHARACTER tags from this shot's manifest placements
    shot_char_tags = set()
    if shot_manifest_row and shot_manifest_row.manifest_json:
        for p in shot_manifest_row.manifest_json.get("placements", []):
            # Check if this placement tag is a CastBinding
            if p.get("asset_tag") and p["asset_tag"] not in asset_map:
                shot_char_tags.add(p["asset_tag"])

    if shot_char_tags:
        _cast_resolved = await resolve_tags_with_assets(
            " ".join(f"@{t}" for t in shot_char_tags),  # ← only THIS shot's tags
            scene.production_bible_id, session,
        )
```

This is a small change that can be made immediately (before the full screenwriter agent is built) to make the existing pipeline shot-aware.

## Pydantic Schemas

### ScriptAnalysis (Step 1 output)

```python
class CharacterReference(BaseModel):
    tag: str = Field(description="@tag from scene prompt or inferred from text")
    role: str = Field(description="protagonist | supporting | background | mentioned_only")
    screen_time_hint: str = Field(description="heavy | moderate | brief | absent")
    first_appearance_beat: int = Field(description="Index into story_beats where character first appears")

class StoryBeat(BaseModel):
    index: int
    description: str = Field(description="What happens in this beat (1-2 sentences)")
    characters_involved: list[str] = Field(description="@tags of characters in this beat")
    emotional_tone: str = Field(description="tense | joyful | melancholy | awe | neutral | ...")
    is_climax: bool = Field(default=False)

class ScriptAnalysis(BaseModel):
    narrative_summary: str = Field(description="1-2 sentence summary of the story")
    tone: str = Field(description="Overall tone: cinematic | documentary | commercial | narrative | ...")
    genre: str = Field(description="Genre: drama | action | comedy | sci-fi | commercial | ...")
    pacing: str = Field(description="slow_burn | steady | fast_cut | montage")
    characters: list[CharacterReference]
    settings: list[str] = Field(description="Locations/environments mentioned or implied")
    story_beats: list[StoryBeat]
    emotional_arc: str = Field(description="Description of tension/emotion curve across the scene")
```

### ShotBreakdown (Step 2 output)

```python
class ShotAssignment(BaseModel):
    shot_index: int
    beat_index: int = Field(description="Which story_beat this shot primarily serves")
    narrative_intent: str = Field(description="What this shot communicates to the viewer (1 sentence)")
    characters_present: list[str] = Field(description="@tags of characters visible in this shot")
    setting: str = Field(description="Location/environment for this shot")
    time_of_day: str = Field(description="dawn | morning | midday | afternoon | golden_hour | dusk | night")
    emotional_weight: float = Field(ge=0, le=10, description="How emotionally important (0=establishing, 10=climax)")
    duration_hint: float = Field(description="Suggested duration in seconds")
    transition_from_previous: str | None = Field(default=None, description="How this shot connects from the previous")

class ShotBreakdown(BaseModel):
    shots: list[ShotAssignment]
    arc_coverage: str = Field(description="Brief note on how shots map to the emotional arc")
    uncovered_beats: list[int] = Field(default_factory=list, description="Beat indices not covered by any shot (validation)")
```

### ShotExpansion (for dynamic shot addition)

```python
class ShotExpansionRequest(BaseModel):
    insert_after_index: int
    user_hint: str | None = None

class ShotExpansionResult(BaseModel):
    new_shot: ShotAssignment
    updated_transitions: list[dict] = Field(
        description="[{shot_index: int, new_transition_notes: str}] for adjacent shots"
    )
```

## Implementation Phases

### Phase A: Shot-Aware CastBinding Resolution (Immediate)

**Scope:** Fix the CastBinding fallback to use shot manifest placements instead of scene.prompt.

**Files:** `backend/vidpipe/pipeline/keyframes.py` (~15 lines changed)

**Impact:** Shots without characters stop getting unwanted reference images. Shots with characters get the right refs.

**Prerequisite:** The storyboard LLM must place CastBinding @tags in shot manifests. The binding-aware storyboard prompt (`format_binding_registry`) already instructs the LLM to do this, and `_remap_unrecognized_tags()` catches mistakes. Verify this works with the "Test" production bible.

### Phase B: Script Analysis + Shot Breakdown (Core Agent)

**Scope:** Add Steps 1-2 as new pipeline functions. Replace the "single-shot storyboard" with a two-step process: analyze script → assign characters to shots.

**Files:**
- `backend/vidpipe/schemas/screenwriter_agent.py` (new — ScriptAnalysis, ShotBreakdown schemas)
- `backend/vidpipe/services/screenwriter_agent.py` (new — analyze_script(), break_into_shots())
- `backend/vidpipe/pipeline/storyboard.py` (modify — call agent before manifest generation)
- `backend/vidpipe/db/models.py` (add Scene.script_analysis column)

**LLM calls:** 2 per scene (analysis + breakdown), replacing 0 (these are new steps before the existing storyboard call).

**Impact:** Characters are explicitly assigned to shots. The storyboard LLM receives `characters_present[]` as hard constraints instead of inferring them.

### Phase C: Per-Shot Manifest + Prompt Generation (Parallelization)

**Scope:** Instead of one massive storyboard call that generates all shots at once, generate manifests and prompts per-shot in parallel. Each shot receives its ShotAssignment as context.

**Files:**
- `backend/vidpipe/pipeline/storyboard.py` (restructure — parallel per-shot generation)
- `backend/vidpipe/services/prompt_rewriter.py` (may merge with per-shot generation)

**LLM calls:** N parallel calls (one per shot) instead of 1 large call. Total tokens similar, but latency drops to ~1 shot's generation time.

**Impact:** Better quality per-shot (LLM focuses on one shot at a time), faster generation (parallel), and unlocks Phase D.

### Phase D: Dynamic Shot Expansion (Non-Destructive Editing)

**Scope:** Add/remove/reorder shots without regenerating existing ones.

**Files:**
- `backend/vidpipe/services/screenwriter_agent.py` (add expand_shot(), remove_shot())
- `backend/vidpipe/api/routes.py` (new endpoints: POST /scenes/{id}/shots/insert, DELETE /scenes/{id}/shots/{index})

**LLM calls:** 1 per expansion (generate new shot context + update transitions).

**Impact:** Users can iteratively refine their storyboard. The "add a shot" workflow becomes: click add → LLM generates shot that fits the arc → generate keyframe for just that shot.

### Phase E: Validation & Arc Analysis (Quality Gate)

**Scope:** Post-generation validation that all characters are placed, all beats are covered, screen time is proportional, and the emotional arc is maintained.

**Files:**
- `backend/vidpipe/services/screenwriter_agent.py` (add validate_screenplay())

**LLM calls:** 0 (deterministic validation) or 1 (optional LLM arc review).

**Impact:** Catches missing characters or story gaps before expensive image/video generation.

## Data Model Changes

```sql
-- Scene gets script analysis storage
ALTER TABLE scenes ADD COLUMN script_analysis JSON;

-- Shot gets explicit character assignment from screenwriter
ALTER TABLE shots ADD COLUMN characters_present JSON;  -- ["BRANDON_ONE", ...]
ALTER TABLE shots ADD COLUMN beat_index INTEGER;
ALTER TABLE shots ADD COLUMN narrative_intent TEXT;
ALTER TABLE shots ADD COLUMN emotional_weight REAL;
```

## Cost Analysis

Current storyboard: **1 LLM call** per scene (large prompt, ~2000 input tokens, ~3000 output tokens).

With screenwriter agent: **2 + N LLM calls** per scene:
- Script analysis: ~500 input, ~500 output
- Shot breakdown: ~800 input, ~800 output
- Per-shot manifest+prompt: ~600 input, ~800 output each (parallelized)

For a 5-shot scene: 2 + 5 = 7 calls vs 1. But:
- Total tokens: ~12,000 vs ~5,000 (2.4x increase)
- Quality: dramatically better (focused LLM attention per shot)
- Latency: similar (parallel per-shot calls vs one sequential mega-call)
- Cost at Gemini Flash pricing ($0.075/1M input): ~$0.001 per scene (negligible)

## Why Not CrewAI / LangGraph / etc.

The research conclusively showed that:

1. **This is a sequential transformation pipeline**, not a multi-agent conversation. Each step has well-defined input/output schemas. Agents don't negotiate or decide tools at runtime.

2. **The codebase already has the perfect pattern.** `ScreenwriterService` does exactly this: sequential LLM calls with Pydantic schemas and SQLAlchemy persistence. The storyboard pipeline does the same. We're just adding 2 steps to the sequence.

3. **Framework overhead is real.** CrewAI adds ~56% token overhead. LangGraph adds state management complexity. Neither solves a problem we have.

4. **Every comparable system uses raw pipelines.** ViMax (academic), Filmustage (commercial), Katalist (commercial) — all use sequential structured LLM calls with validation steps between them.

The right tool is the one we already have: `async def step(input: PydanticModel) -> PydanticModel` with `LLMAdapter.generate_text(prompt, schema)`.

## Key Files Reference

| File | Current Role | Screenwriter Agent Role |
|------|-------------|------------------------|
| `services/screenwriter.py` | 6-step screenplay generation (Production-level) | Unchanged — this is the macro screenplay |
| `pipeline/storyboard.py` | Single-shot scene → shots | Modified — calls screenwriter agent before manifest generation |
| `services/screenwriter_agent.py` | **Does not exist** | **New** — analyze_script(), break_into_shots(), expand_shot() |
| `schemas/screenwriter_agent.py` | **Does not exist** | **New** — ScriptAnalysis, ShotBreakdown, ShotExpansion |
| `services/prompt_rewriter.py` | Enriches shot prompts post-storyboard | Unchanged — receives better input from screenwriter agent |
| `services/manifest_service.py` | format_binding_registry() | Unchanged — provides asset context to screenwriter agent |
| `pipeline/keyframes.py` | CastBinding fallback (scene-level) | Fixed — shot-aware CastBinding resolution |
| `db/models.py` | Scene, Shot, ShotManifest | Extended — script_analysis, characters_present columns |
