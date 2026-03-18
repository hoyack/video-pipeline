# Phase 27: Storyboard Performance Optimization - Research

**Researched:** 2026-03-17
**Domain:** Python async concurrency, LLM call decomposition, real-time event streaming
**Confidence:** HIGH

## Summary

This phase transforms the monolithic storyboard Call #3 (which generates all shots in a single LLM invocation) into N parallel per-shot LLM calls, adds real-time per-shot progress events, skips unnecessary screenwriter agent calls for simple scenes, and removes redundant schema fields. The current codebase already has all infrastructure needed: `asyncio.gather` patterns (used in `manifesting_engine.py`, `candidate_scoring.py`), WebSocket event bus with `shot_text_ready` event type (already defined in `wsTypes.ts` and handled in `EditModeOverlay.tsx`), and the `ShotAssignment` schema from the screenwriter agent that provides per-shot context.

The core change is in `backend/vidpipe/pipeline/storyboard.py` where the single `generate_with_retry()` call (line 543-560) that produces an `EnhancedStoryboardOutput` for ALL shots must be replaced with N parallel calls, each producing output for a single shot. A new per-shot Pydantic schema is needed (subset of `EnhancedStoryboardOutput` for one shot), and the results must be assembled back into the existing shape. The screenwriter bypass and schema field elimination are straightforward conditional logic changes.

**Primary recommendation:** Decompose the work into three plans: (1) Create per-shot schema + parallel generation core with screenwriter bypass, (2) Wire up per-shot progress events and result assembly, (3) Frontend display of incremental progress and cleanup of redundant fields.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **P0: Split Call #3 into Parallel Per-Shot Calls** -- Monolithic `generate_storyboard()` inner call decomposed into N parallel per-shot LLM calls via `asyncio.gather(*[generate_shot_manifest(shot) for shot in breakdown.shots])`. Each per-shot call receives ShotAssignment + style context + bound asset data. Each produces shot manifest, keyframe prompts, and audio manifest. Results assembled back into full `EnhancedStoryboardOutput` structure. New per-shot Pydantic schema needed. Shared style_guide generated once and passed as context.
- **P1: Per-Shot Progress Events** -- Emit `shot_text_ready` SSE event as each parallel per-shot call completes. Progress format includes shot index and total count. Uses existing SSE event infrastructure. Frontend must display per-shot progress during storyboard generation.
- **P2: Skip Screenwriter Agent for Simple Scenes** -- When `dynamic_shot_count=False` AND single-shot scene: skip Call #1 (script analysis) and Call #2 (shot breakdown). Inject minimal `shot_constraints` from `target_shot_count` alone. Saves 30-80 seconds on slow models. Bypass check lives in storyboard pipeline entry point.
- **P3: Eliminate Redundant Schema Fields** -- Remove `characters[]` from storyboard output schema (already in Actor `base_appearance_prompt` and CastBinding data). Simplify `style_guide` generation (derive from scene `style` field + fixed template).

### Claude's Discretion
- Internal implementation of the per-shot schema (field names, nesting)
- Error handling strategy when individual per-shot calls fail (fail-fast vs. partial results)
- How to merge per-shot results back into existing `EnhancedStoryboardOutput` shape (adapter layer vs. restructured schema)
- Whether to add a concurrency limit on parallel per-shot calls (e.g., max 5 concurrent)
- SSE event payload structure for `shot_text_ready`

### Deferred Ideas (OUT OF SCOPE)
- Streaming structured output -- Process and persist shots as JSON tokens arrive (Gap 2 partial)
- Model-aware timeouts -- Different timeout configs for fast vs slow models (Gap 2 partial)
- Per-shot re-generation / caching -- Only regenerate shots whose inputs changed, cache by prompt hash (Gap 5 / Phase D from plan)
- Progress heartbeats -- Emit "still generating..." events during individual long calls
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SBPERF-01 | Storyboard Call #3 decomposed into N parallel per-shot LLM calls; each generates manifest + keyframe prompts + audio for one shot via asyncio.gather | Per-shot schema design, `asyncio.gather` pattern, result assembly strategy, error handling |
| SBPERF-02 | Per-shot `shot_text_ready` SSE progress events emitted as each parallel call completes, visible in the frontend during generation | Event bus emit pattern, `ShotTextReadyEvent` already exists in wsTypes.ts, EditModeOverlay already handles it |
| SBPERF-03 | Screenwriter agent (Calls #1-2) skipped when `dynamic_shot_count=False` and single-shot scene; minimal shot_constraints injected from target_shot_count | Bypass condition, minimal ShotAssignment construction |
| SBPERF-04 | Redundant `characters[]` and `style_guide` fields removed from per-shot output schema; uses existing actor binding data and scene style field instead | EnhancedStoryboardOutput schema restructuring, style_guide template pattern |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| asyncio | stdlib | Parallel per-shot LLM calls via `asyncio.gather` | Already used throughout codebase for async patterns |
| pydantic | 2.x | Per-shot output schema definition | Already used for all LLM structured output schemas |
| sqlalchemy | 2.0 async | Database persistence of shot/manifest rows | Existing ORM layer |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tenacity | existing | Retry logic for individual per-shot calls | Wrap each per-shot LLM call |

No new dependencies needed. All work uses existing stack.

## Architecture Patterns

### Current Flow (Monolithic Call #3)
```
generate_storyboard()
  ├── 1. Load context (bindings, assets, tags)
  ├── 2. Screenwriter Agent (Call #1: analyze_script, Call #2: break_into_shots)
  ├── 3. Build full system prompt with ALL shot constraints
  ├── 4. Single LLM call → EnhancedStoryboardOutput (ALL shots at once)  ← BOTTLENECK
  ├── 5. Persist ALL shots to DB
  ├── 6. Persist ALL shot manifests and audio manifests
  └── 7. Emit events (shot_text_ready for each, phase_completed)
```

### Target Flow (Parallel Per-Shot Calls)
```
generate_storyboard()
  ├── 1. Load context (bindings, assets, tags) [UNCHANGED]
  ├── 2. Screenwriter Agent (Calls #1-2) [CONDITIONAL: skip for simple scenes]
  ├── 3. Generate style_guide once (from scene.style + template OR short LLM call)
  ├── 4. For each ShotAssignment:
  │     └── Parallel: generate_single_shot(shot_assignment, style_context, asset_data)
  │         ├── LLM call → PerShotOutput (one shot only)
  │         ├── Emit shot_text_ready event
  │         └── Return result
  ├── 5. Assemble PerShotOutput[] → EnhancedStoryboardOutput shape
  ├── 6. Persist shots, manifests, audio manifests [UNCHANGED logic]
  └── 7. Emit phase_completed
```

### Pattern 1: Per-Shot Schema Design

**Recommendation:** Create a `PerShotOutput` schema that is essentially `EnhancedShotSchema` minus the `characters[]` and with per-shot audio included inline. The assembly step wraps these back into `EnhancedStoryboardOutput` for downstream compatibility.

```python
class PerShotOutput(BaseModel):
    """Output from a single per-shot LLM call.

    Contains everything the monolithic call produced for ONE shot:
    shot description, keyframe prompts, manifest, and audio.
    Does NOT include characters[] or style_guide (those come from context).
    """
    shot_index: int
    shot_description: str
    key_details: list[str]
    start_frame_prompt: str
    end_frame_prompt: str
    video_motion_prompt: str
    transition_notes: str
    shot_manifest: ShotManifestSchema  # reuse existing
    audio_manifest: ShotAudioManifestSchema  # reuse existing
```

This avoids duplicating field definitions by reusing existing schema components from `storyboard_enhanced.py`.

### Pattern 2: Parallel Execution with Progress Events

```python
async def _generate_single_shot(
    adapter: LLMAdapter,
    shot_assignment: ShotAssignment,
    system_prompt: str,
    scene_id: uuid.UUID,
    total_shots: int,
) -> PerShotOutput:
    """Generate manifest + prompts + audio for a single shot."""
    # Build per-shot prompt with assignment context
    shot_prompt = _build_per_shot_prompt(system_prompt, shot_assignment)

    result = await adapter.generate_text(
        prompt=shot_prompt,
        schema=PerShotOutput,
        temperature=0.7,
        max_retries=2,
    )

    # Emit progress immediately on completion
    from vidpipe.services.event_bus import event_bus
    event_bus.emit(scene_id, "shot_text_ready",
                   shot_index=result.shot_index,
                   total_shots=total_shots)

    return result

# Parallel execution
results = await asyncio.gather(
    *[_generate_single_shot(adapter, sa, system_prompt, scene.id, len(shots))
      for sa in breakdown.shots],
    return_exceptions=True,
)
```

### Pattern 3: Screenwriter Bypass for Simple Scenes

```python
# In generate_storyboard(), BEFORE the screenwriter agent block:
skip_screenwriter = (
    not getattr(scene, "dynamic_shot_count", False)
    and scene.target_shot_count == 1
)

if skip_screenwriter:
    # Inject minimal shot assignment without LLM calls
    screenwriter_breakdown = ShotBreakdown(
        shots=[ShotAssignment(
            shot_index=0,
            beat_index=0,
            narrative_intent=scene.prompt[:200],
            characters_present=[],  # Will be filled by per-shot LLM
            setting="",
            time_of_day="midday",
            emotional_weight=5.0,
            duration_hint=float(scene.target_clip_duration),
        )],
        arc_coverage="single shot",
        uncovered_beats=[],
    )
else:
    # Existing screenwriter agent flow...
```

### Pattern 4: Style Guide Template (No LLM Call)

```python
def _derive_style_guide(scene_style: str) -> dict:
    """Build a style_guide dict from scene.style without an LLM call.

    For the per-shot approach, each shot gets the style as context,
    but we no longer ask the LLM to generate a fresh style_guide.
    """
    style_label = scene_style.replace("_", " ")
    return {
        "visual_style": style_label,
        "color_palette": f"Consistent with {style_label} aesthetic",
        "camera_style": "As directed per shot composition",
    }
```

### Pattern 5: Result Assembly

```python
def _assemble_storyboard_output(
    per_shot_results: list[PerShotOutput],
    style_guide: StyleGuide,
) -> EnhancedStoryboardOutput:
    """Assemble parallel per-shot results into the existing EnhancedStoryboardOutput shape."""
    enhanced_shots = []
    for result in sorted(per_shot_results, key=lambda r: r.shot_index):
        enhanced_shots.append(EnhancedShotSchema(
            shot_index=result.shot_index,
            shot_description=result.shot_description,
            key_details=result.key_details,
            start_frame_prompt=result.start_frame_prompt,
            end_frame_prompt=result.end_frame_prompt,
            video_motion_prompt=result.video_motion_prompt,
            transition_notes=result.transition_notes,
            shot_manifest=result.shot_manifest,
            audio_manifest=result.audio_manifest,
        ))

    return EnhancedStoryboardOutput(
        style_guide=style_guide,
        characters=[],  # SBPERF-04: removed, using binding data instead
        shots=enhanced_shots,
    )
```

### Anti-Patterns to Avoid
- **Shared mutable state across parallel calls:** Each per-shot coroutine must be independent. Do not share a mutable prompt builder or result accumulator across gather tasks.
- **Emitting events inside gather without scene_id:** Always pass scene_id explicitly to the per-shot function; do not rely on closure capture of session-scoped variables.
- **Committing inside parallel tasks:** Database commits must happen AFTER gather completes, in the main coroutine. Individual tasks should only return data, not interact with the DB session (SQLAlchemy async sessions are NOT safe for concurrent use from multiple coroutines).
- **Ignoring return_exceptions:** If one shot fails, `asyncio.gather` without `return_exceptions=True` cancels all other tasks. Use `return_exceptions=True` and handle failures after gather completes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parallel async execution | Custom task pool | `asyncio.gather` | Already proven in codebase (manifesting_engine.py, candidate_scoring.py) |
| Per-shot schema | Duplicate all fields | Compose from existing `ShotManifestSchema`, `ShotAudioManifestSchema` | Reuse avoids drift between schema definitions |
| WebSocket events | New event transport | Existing `event_bus.emit()` + `useSceneWebSocket` | Infrastructure already handles `shot_text_ready` end-to-end |
| Retry logic | Custom retry wrapper | `tenacity` or `adapter.generate_text(max_retries=N)` | LLM adapter already handles retries internally |

## Common Pitfalls

### Pitfall 1: SQLAlchemy Session Concurrency
**What goes wrong:** Passing the same `AsyncSession` to multiple concurrent `asyncio.gather` tasks causes "greenlet" errors or data corruption. SQLAlchemy async sessions are NOT thread-safe or coroutine-safe for concurrent writes.
**Why it happens:** `asyncio.gather` runs coroutines concurrently on the same event loop. If two tasks flush/commit through the same session, internal state corrupts.
**How to avoid:** Per-shot tasks must ONLY return data (the `PerShotOutput` model). All database writes (Shot creation, ShotManifest creation, etc.) happen sequentially AFTER `asyncio.gather` completes, in the calling coroutine which owns the session.
**Warning signs:** `MissingGreenlet` errors, `DetachedInstanceError`, or silently missing rows.

### Pitfall 2: Event Emission Timing
**What goes wrong:** Emitting `shot_text_ready` inside the gather task but the frontend refresh finds no committed data (because the session hasn't committed yet).
**Why it happens:** Events fire before DB commit. The frontend receives the event, polls the API, but the shot data isn't persisted yet.
**How to avoid:** Two options: (A) Emit events inside the gather task for immediate progress indication, but the frontend should use the event as a UI counter update (not a data fetch trigger for that specific shot), OR (B) Emit events after gather completes and shots are committed. **Recommendation: Option A** -- emit inside gather for real-time progress counter, then emit a `refresh` event after commit so the frontend fetches all shot data at once.
**Warning signs:** Frontend shows "Shot 3 of 5 ready" but clicking on it shows empty data.

### Pitfall 3: Non-Manifest Mode Regression
**What goes wrong:** The parallel per-shot path only works for manifest-aware scenes (`use_manifests=True`). Non-manifest scenes (no production_bible_id) still use the basic `StoryboardOutput` schema without manifests or audio. Breaking this path causes regressions.
**Why it happens:** The parallelization applies to Call #3 which uses `EnhancedStoryboardOutput` (manifest mode). The basic `StoryboardOutput` path is a different code branch.
**How to avoid:** Keep the non-manifest path using the existing monolithic call. Only parallelize the manifest-aware path (which is the one that's slow because of the larger schema). Or, apply parallelization to both but with different per-shot schemas.
**Warning signs:** Tests passing for manifest mode but basic generation failing.

### Pitfall 4: Shot Index Mismatch
**What goes wrong:** The per-shot LLM call produces a `shot_index` that doesn't match the `ShotAssignment.shot_index` it was given.
**Why it happens:** LLM structured output can produce any integer for shot_index.
**How to avoid:** Override the LLM-returned `shot_index` with the `ShotAssignment.shot_index` that was passed as input. Trust the input, not the LLM output for index values.
**Warning signs:** Duplicate shot indices in the assembled output, missing shots.

### Pitfall 5: Gap-Filling Mode Interaction
**What goes wrong:** The existing gap-filling logic (draft scenes with some shots already filled) interacts poorly with the parallel per-shot approach.
**Why it happens:** Gap-filling only generates content for empty shots, preserving user-provided text. The parallel approach must respect this -- only spawn per-shot tasks for empty shots.
**How to avoid:** Filter the `breakdown.shots` list to only include assignments for shots that need generation (matching `empty_shots` indices). Pass filled shot context to each per-shot call for narrative continuity.
**Warning signs:** User-provided shot text overwritten, or gap shots generated without narrative context.

### Pitfall 6: Characters[] Removal Downstream Impact
**What goes wrong:** Removing `characters[]` from `EnhancedStoryboardOutput` breaks downstream code that reads `storyboard.characters`.
**Why it happens:** `scene.storyboard_raw` stores the full model dump, and downstream code may access `storyboard_raw["characters"]`.
**How to avoid:** Keep `characters` as an empty list in the assembled output (not removed entirely). This preserves the JSON shape. Search for all references to `.characters` on storyboard objects.
**Warning signs:** KeyError on "characters" in downstream pipeline stages.

## Code Examples

### Current Monolithic Call (to be replaced)
```python
# storyboard.py lines 543-563 — the bottleneck
@retry(
    stop=stop_after_attempt(max_attempts),
    retry=retry_if_exception_type((json.JSONDecodeError, ValidationError))
)
async def generate_with_retry():
    nonlocal attempt
    temperature = base_temperature - (attempt * 0.15)
    attempt += 1
    response_schema = EnhancedStoryboardOutput if use_manifests else StoryboardOutput
    storyboard = await adapter.generate_text(
        prompt=full_prompt,
        schema=response_schema,
        temperature=max(0.0, temperature),
        max_retries=1,
    )
    return storyboard

storyboard = await generate_with_retry()
```

### Existing Event Emission (already works)
```python
# storyboard.py line 737 — current shot_text_ready emission pattern
for shot_data in storyboard.shots:
    event_bus.emit(scene.id, "shot_text_ready", shot_index=shot_data.shot_index)
```

### Existing Frontend Handler (already works)
```typescript
// EditModeOverlay.tsx line 187-194 — already handles shot_text_ready
case "shot_text_ready":
  setWsProgress(prev =>
    prev.phase === "storyboard"
      ? { ...prev, completedShots: prev.completedShots + 1 }
      : prev,
  );
  onRefresh?.();
  break;
```

### Existing asyncio.gather Pattern in Codebase
```python
# manifesting_engine.py line 515 — proven pattern
await asyncio.gather(*[process_asset_reverse_prompt(a) for a in assets_needing_prompts])
```

### Screenwriter Bypass Condition
```python
# Current code (storyboard.py line 396):
is_dynamic = getattr(scene, "dynamic_shot_count", False)

# Bypass condition to add:
skip_screenwriter = (
    not is_dynamic
    and (scene.target_shot_count or 3) == 1
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic Call #3 for all shots | Parallel per-shot calls | Phase 27 | 3-5x speedup on slow models |
| Characters[] in storyboard output | Character data from Actor/CastBinding | Phase 22 (binding system) | Redundant field can be removed |
| Style guide from LLM | Derived from scene.style field | Phase 27 | Eliminates tokens, slightly less creative |
| Screenwriter always runs | Conditional bypass for simple scenes | Phase 27 | 30-80s saved for single-shot scenes |

## Key Implementation Details

### Per-Shot Prompt Construction
Each per-shot call needs focused context. The prompt should include:
1. The shared system prompt (ENHANCED_STORYBOARD_PROMPT, already formatted with style + aspect_ratio + asset_registry)
2. The specific `ShotAssignment` from the screenwriter (narrative_intent, characters_present, setting, emotional_weight)
3. Style guide context (pre-derived, not generated)
4. Instruction to generate output for THIS shot only
5. Previous/next shot context for transition continuity (brief summary of adjacent shots)

### Concurrency Limit Recommendation
For Ollama cloud endpoints that may have rate limits, use `asyncio.Semaphore(5)` to cap concurrent per-shot calls. For Gemini, no limit needed (API handles rate limiting via retries). Implementation:

```python
semaphore = asyncio.Semaphore(5)  # or configurable

async def _generate_with_semaphore(shot_assignment, ...):
    async with semaphore:
        return await _generate_single_shot(shot_assignment, ...)
```

### Error Handling Recommendation: Fail-Fast
**Recommendation:** Use `return_exceptions=True` with `asyncio.gather`, then after all tasks complete, check for failures. If ANY shot failed, retry the failed shots once. If still failing, raise the error (pipeline fails and can be resumed). Rationale: a storyboard with missing shots is unusable for downstream keyframe/video generation.

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
failures = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
if failures:
    # Retry failed shots once
    retry_tasks = [tasks[i] for i, _ in failures]
    retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
    # If still failing, raise first error
    still_failed = [r for r in retry_results if isinstance(r, Exception)]
    if still_failed:
        raise still_failed[0]
    # Merge retry results back
    for (orig_idx, _), retry_result in zip(failures, retry_results):
        results[orig_idx] = retry_result
```

### ShotTextReadyEvent Payload Extension
The existing `ShotTextReadyEvent` only has `shot_index`. Add `total_shots` so the frontend can display "Shot 2 of 5":

```python
# Backend emit
event_bus.emit(scene.id, "shot_text_ready", shot_index=idx, total_shots=total)
```

```typescript
// Frontend type extension
export interface ShotTextReadyEvent extends WsEventBase {
  type: "shot_text_ready";
  shot_index: number;
  total_shots?: number;  // NEW: optional for backward compat
}
```

The frontend already gets `total_shots` from `phase_started` event, so the `total_shots` in `shot_text_ready` is redundant but useful for standalone display.

### Backward Compatibility
- `storyboard_raw` JSON must maintain the same shape (style_guide, characters, shots)
- `characters` can be `[]` but must exist as a key
- `EnhancedStoryboardOutput` model used for assembly must remain valid
- Non-manifest scenes (no production_bible_id) continue using monolithic path
- `StoryboardOutput` (non-enhanced) path is unchanged

## Open Questions

1. **Non-manifest parallel path**
   - What we know: Only manifest-aware scenes use EnhancedStoryboardOutput. Basic StoryboardOutput is simpler.
   - What's unclear: Should we parallelize non-manifest scenes too? They are generally faster (smaller schema), but slow models still suffer.
   - Recommendation: Start with manifest-aware parallelization only. If needed, extend to non-manifest in a follow-up. Keeps scope contained.

2. **Style guide generation method**
   - What we know: Currently the LLM generates style_guide as part of EnhancedStoryboardOutput. SBPERF-04 says to simplify it.
   - What's unclear: Should it be a pure template (no LLM) or a brief one-shot LLM call? Template is faster but less creative.
   - Recommendation: Use a pure template derived from scene.style. The style_guide fields (visual_style, color_palette, camera_style) are already encoded in the system prompt -- the LLM-generated version was redundant context.

3. **Transition notes continuity across parallel shots**
   - What we know: Each shot needs `transition_notes` describing how it connects to adjacent shots. In the monolithic call, the LLM sees all shots and can create coherent transitions.
   - What's unclear: In parallel mode, shot N doesn't know what shot N+1 looks like.
   - Recommendation: Pass the `ShotAssignment` for adjacent shots (N-1 and N+1) as brief context to each per-shot call. The assignment includes `narrative_intent` which is enough for transition planning.

## Sources

### Primary (HIGH confidence)
- `backend/vidpipe/pipeline/storyboard.py` -- Full current implementation examined (740 lines)
- `backend/vidpipe/schemas/storyboard_enhanced.py` -- EnhancedStoryboardOutput, ShotManifestSchema, ShotAudioManifestSchema
- `backend/vidpipe/schemas/storyboard.py` -- StyleGuide, CharacterDescription, StoryboardOutput, ShotSchema
- `backend/vidpipe/services/screenwriter_agent.py` -- ScreenwriterAgentService, ScriptAnalysis, ShotBreakdown
- `backend/vidpipe/schemas/screenwriter_agent.py` -- ShotAssignment schema (per-shot context for parallel calls)
- `backend/vidpipe/services/event_bus.py` -- EventBus singleton, emit/subscribe pattern
- `frontend/src/api/wsTypes.ts` -- ShotTextReadyEvent already defined
- `frontend/src/components/EditModeOverlay.tsx` -- shot_text_ready already handled (lines 187-194)
- `frontend/src/hooks/useSceneWebSocket.ts` -- WebSocket connection management
- `backend/vidpipe/orchestrator/pipeline.py` -- Pipeline orchestrator, storyboard invocation point
- `docs/storyboard-gap.md` -- PRD with timing data and architecture analysis

### Secondary (MEDIUM confidence)
- `backend/vidpipe/services/manifesting_engine.py` -- asyncio.gather pattern reference (line 515)
- `backend/vidpipe/services/candidate_scoring.py` -- asyncio.gather pattern reference (line 422)
- `backend/vidpipe/services/llm/base.py` -- LLMAdapter.generate_text() interface

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools already in the codebase, no new dependencies
- Architecture: HIGH - Direct examination of all source files involved, clear decomposition path
- Pitfalls: HIGH - Based on known SQLAlchemy async patterns and existing codebase conventions
- Frontend integration: HIGH - shot_text_ready event already fully wired end-to-end

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (stable -- no external dependency changes expected)
