# Phase 27: Storyboard Performance Optimization - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning
**Source:** PRD Express Path (docs/storyboard-gap.md)

<domain>
## Phase Boundary

This phase optimizes the storyboard generation pipeline for slow LLM models (Ollama cloud, large parameter models) by decomposing the monolithic Call #3 into parallel per-shot LLM calls, adding real-time per-shot progress events, skipping the screenwriter agent for simple scenes, and removing redundant schema fields. The result is 3-5x faster storyboard generation with incremental progress feedback.

**In scope:** Storyboard Call #3 decomposition, per-shot progress events, screenwriter agent bypass, schema field elimination.
**Out of scope:** Streaming structured output, model-aware timeouts, per-shot re-generation/caching (Gap 5 / Phase D from plan), ComfyUI/Veo changes.

</domain>

<decisions>
## Implementation Decisions

### P0: Split Call #3 into Parallel Per-Shot Calls
- The monolithic `generate_storyboard()` inner call that produces `EnhancedStoryboardOutput` for ALL shots must be decomposed into N parallel per-shot LLM calls
- Each per-shot call receives: `ShotAssignment` from the shot breakdown, style context, bound asset data
- Each per-shot call produces: shot manifest (composition, placements, new_asset_declarations), keyframe prompts (start_frame_prompt, end_frame_prompt, video_motion_prompt), audio manifest (dialogue_lines, sfx, ambient, music, audio_continuity)
- Parallelization via `asyncio.gather(*[generate_shot_manifest(shot) for shot in breakdown.shots])`
- Results are assembled back into the full `EnhancedStoryboardOutput` structure after all per-shot calls complete
- A new per-shot Pydantic schema is needed for the individual LLM call output (subset of EnhancedStoryboardOutput for one shot)
- The shared `style_guide` is generated once (either from a simpler pre-call or from scene style field + template) and passed as context to each per-shot call

### P1: Per-Shot Progress Events
- Emit `shot_text_ready` SSE event as each parallel per-shot call completes
- Progress format: `shot_text_ready` with shot index and total count (e.g., "Shot 2 of 5 ready")
- Uses existing SSE event infrastructure in the pipeline (same pattern as other progress events)
- Frontend must display per-shot progress during storyboard generation

### P2: Skip Screenwriter Agent for Simple Scenes
- When `dynamic_shot_count=False` AND scene is single-shot: skip Call #1 (script analysis) and Call #2 (shot breakdown)
- Inject a minimal `shot_constraints` block derived from `target_shot_count` alone
- This saves 30-80 seconds on slow models
- The bypass check lives in the storyboard pipeline entry point

### P3: Eliminate Redundant Schema Fields
- Remove `characters[]` from storyboard output schema — character descriptions already exist in Actor `base_appearance_prompt` and CastBinding data
- Simplify `style_guide` generation — derive from scene `style` field + a fixed template rather than asking the LLM to generate it each time
- This reduces output token count and speeds up generation

### Claude's Discretion
- Internal implementation of the per-shot schema (field names, nesting)
- Error handling strategy when individual per-shot calls fail (fail-fast vs. partial results)
- How to merge per-shot results back into the existing `EnhancedStoryboardOutput` shape (adapter layer vs. restructured schema)
- Whether to add a concurrency limit on parallel per-shot calls (e.g., max 5 concurrent)
- SSE event payload structure for `shot_text_ready`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Storyboard Pipeline
- `backend/vidpipe/pipeline/storyboard.py` — Current monolithic storyboard generation (Call #3 lives here)
- `backend/vidpipe/schemas/storyboard.py` — `EnhancedStoryboardOutput`, `ShotManifest`, `AudioManifest` schemas
- `backend/vidpipe/services/screenwriter_agent.py` — Screenwriter agent (Calls #1 and #2): `analyze_script()`, `break_into_shots()`

### Shot Breakdown & Schemas
- `backend/vidpipe/schemas/screenwriter.py` — `ScriptAnalysis`, `ShotBreakdown`, `ShotAssignment` schemas
- `backend/vidpipe/schemas/storyboard.py` — Full storyboard output schema including per-shot manifests

### Progress Events & SSE
- `backend/vidpipe/orchestrator/pipeline.py` — Pipeline orchestrator, event emission patterns
- `backend/vidpipe/api/routes.py` — SSE endpoint for progress streaming

### Asset Binding Context
- `backend/vidpipe/services/tag_resolver.py` — Tag resolution for bound assets
- `backend/vidpipe/services/checkpoint_service.py` — Checkpoint/binding data access

### PRD
- `docs/storyboard-gap.md` — Full gap analysis with timing data, architecture details, and prioritized recommendations

</canonical_refs>

<specifics>
## Specific Ideas

### Timing Data (from PRD measurements)
- Single-shot scene on slow model (Ollama qwen3.5:397b-cloud): Call #1=28s, Call #2=48s, Call #3=5-10+ min. Total: ~7-12 min
- 4-shot scene: Calls #1+#2=~5 min, Call #3=~6 min. Total: ~11 min
- Gemini Flash comparison: 10-30 seconds total for same pipeline
- Expected improvement: 5-shot scene from 10 min → 2 min (parallel per-shot)

### Architecture Pattern
```python
asyncio.gather(*[generate_shot_manifest(shot_assignment) for shot in breakdown.shots])
```

### Per-Shot Output Schema (Conceptual)
Each per-shot call generates for its assigned shot:
- `shot_description`, `key_details[]`
- `start_frame_prompt`, `end_frame_prompt`, `video_motion_prompt`
- `transition_notes`
- Manifest: `composition`, `placements[]`, `new_asset_declarations[]`
- Audio: `dialogue_lines[]`, `sfx[]`, `ambient`, `music`, `audio_continuity`

</specifics>

<deferred>
## Deferred Ideas

- **Streaming structured output** — Process and persist shots as JSON tokens arrive (Gap 2 partial)
- **Model-aware timeouts** — Different timeout configs for fast vs slow models (Gap 2 partial)
- **Per-shot re-generation / caching** — Only regenerate shots whose inputs changed, cache by prompt hash (Gap 5 / Phase D from plan)
- **Progress heartbeats** — Emit "still generating..." events during individual long calls

</deferred>

---

*Phase: 27-storyboard-performance-optimization*
*Context gathered: 2026-03-17 via PRD Express Path*
