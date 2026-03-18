# Storyboard Generation: Gap Analysis

## Current Configuration

### LLM Call Sequence (3 sequential calls)

| # | Step | Function | Schema | Purpose |
|---|------|----------|--------|---------|
| 1 | Script Analysis | `ScreenwriterAgentService.analyze_script()` | `ScriptAnalysis` | Extract narrative summary, tone, characters, beats, arc |
| 2 | Shot Breakdown | `ScreenwriterAgentService.break_into_shots()` | `ShotBreakdown` | Assign characters_present[] per shot, emotional weights |
| 3 | Main Storyboard | `generate_storyboard()` inner call | `EnhancedStoryboardOutput` | Full storyboard: style guide, characters, per-shot keyframe prompts, shot manifests (placements, composition), audio manifests (dialogue, SFX, ambient, music) |

All three calls are **sequential** — each waits for the previous to complete.

### Observed Timing (ollama/qwen3.5:397b-cloud)

Scene `c5bde292` (1-shot, dynamic):
- Call #1 (Script Analysis): **28 seconds**
- Call #2 (Shot Breakdown): **48 seconds**
- Call #3 (Main Storyboard): **5-10+ minutes** (ongoing at time of measurement)
- **Total: ~7-12 minutes for a single-shot storyboard**

Scene `283caaa4` (dynamic, 4 shots generated):
- Call #1 + #2: ~5 minutes combined
- Call #3: ~6 minutes
- **Total: ~11 minutes**

By comparison, Gemini Flash models complete the same pipeline in 10-30 seconds total.

### Why Call #3 Is So Slow

The `EnhancedStoryboardOutput` schema is massive. For each shot, the LLM must generate:
- `style_guide` (visual_style, color_palette, camera_style)
- `characters[]` (name, physical_description, clothing_description)
- Per-shot: `shot_description`, `key_details[]`, `start_frame_prompt`, `end_frame_prompt`, `video_motion_prompt`, `transition_notes`
- Per-shot manifest: `composition` (shot_type, camera_movement, camera_angle, lens, depth_of_field), `placements[]` (asset_tag, role, position, action, expression, wardrobe_note), `new_asset_declarations[]`
- Per-shot audio: `dialogue_lines[]`, `sfx[]`, `ambient`, `music`, `audio_continuity`

A 5-shot storyboard generates ~3000-5000 tokens of structured JSON. Slow models (Ollama cloud) have low tokens/second throughput on this output.

### No Intermediate Progress

During Call #3 (which takes minutes), the user sees no progress. The UI shows "generating storyboard" with no shot-level updates until the entire call completes and all shots are persisted at once.

---

## Comparison with Screenwriter Agent Plan

Reference: `docs/screenwriter-agent-plan.md`

### What the Plan Proposed (5 Steps)

1. **Script Analysis** — Implemented (Call #1)
2. **Shot Breakdown** — Implemented (Call #2)
3. **Per-Shot Manifest Generation** — NOT implemented (proposed: 1 call per shot, parallelizable)
4. **Per-Shot Keyframe Prompt Writing** — NOT implemented (proposed: 1 call per shot, parallelizable)
5. **Validation & Reconciliation** — Partially implemented (deterministic validation exists, no LLM arc review)

### What Actually Exists

Steps 1-2 are implemented. But Steps 3-4 were NOT split out — instead, the single monolithic Call #3 still generates everything at once (manifests + prompts + audio for ALL shots in a single structured output). The screenwriter agent adds 2 calls before the monolithic call but doesn't replace it.

---

## Identified Gaps

### Gap 1: Monolithic Call #3 (Biggest Performance Issue)

**Problem:** The main storyboard call generates the entire `EnhancedStoryboardOutput` in a single LLM invocation. For slow models, this takes 5-10 minutes with zero intermediate feedback.

**What the plan proposed:** Break Call #3 into N parallel per-shot calls. Each shot generates its own manifest + keyframe prompts + audio. This would:
- Reduce latency from O(N shots) to O(1 shot) (parallel execution)
- Enable per-shot progress streaming (emit `shot_text_ready` as each shot completes)
- Reduce failure blast radius (one shot failing doesn't lose all shots)

**Estimated impact:** For a 5-shot scene on a slow model:
- Current: 1 call x 10 min = **10 minutes**, no progress
- Proposed: 5 parallel calls x 2 min each = **2 minutes**, with per-shot progress

### Gap 2: No Model-Aware Timeout or Streaming

**Problem:** The LLM adapter has no model-specific timeout configuration. A fast Gemini Flash call and a slow Ollama 397B call use the same timeouts. There's no streaming — the entire response must complete before any processing begins.

**What could help:**
- Streaming structured output (process and persist shots as they arrive)
- Model-aware timeouts (Ollama cloud models need longer timeouts)
- Progress heartbeats (emit "still generating..." events during long calls)

### Gap 3: Redundant Character/Style Generation

**Problem:** Call #3 generates `characters[]` and `style_guide` every time, even though:
- Character descriptions already exist in the Actor's `base_appearance_prompt` and CastBinding data
- Style guide could be derived from the scene's `style` field + a simpler template

These redundant generations add tokens to the output and slow down the model.

### Gap 4: Screenwriter Agent Always Runs (Even When Unnecessary)

**Problem:** The screenwriter agent (Calls #1 and #2) runs for every new manifest-mode scene, adding 30-80 seconds even when the user has already specified exact shot count and doesn't need narrative analysis.

**What could help:**
- Skip screenwriter agent when `dynamic_shot_count=False` and single-shot scenes
- Cache script analysis per prompt hash (reuse across storyboard re-runs)
- Make screenwriter agent optional via a scene-level toggle

### Gap 5: No Caching or Incremental Re-generation

**Problem:** Re-running the storyboard (e.g., after a prompt edit) regenerates everything from scratch — all 3 LLM calls, all shots, all manifests. There's no way to regenerate just the changed shots.

**What the plan proposed (Phase D):** Dynamic shot expansion/deletion that only regenerates affected shots and updates transitions for neighbors.

---

## Recommendations (Prioritized)

### P0: Split Call #3 into Parallel Per-Shot Calls
- Biggest single improvement for slow models
- Each shot gets its own LLM call with focused context (ShotAssignment + style + assets)
- Can emit `shot_text_ready` per shot for real-time progress
- Architecture: `asyncio.gather(*[generate_shot_manifest(shot_assignment) for shot in breakdown.shots])`

### P1: Add Progress Events During Long Calls
- Emit heartbeat events ("generating shot 1 of 5...") even without streaming
- For parallel per-shot mode: emit per-shot completion as natural progress

### P2: Skip Screenwriter Agent for Simple Scenes
- When `dynamic_shot_count=False` and user specified exact count, skip Calls #1-2
- Inject a minimal `shot_constraints` block from `target_shot_count` alone
- Saves 30-80 seconds on slow models

### P3: Eliminate Redundant Schema Fields
- Remove `characters[]` from storyboard output (already in binding registry)
- Simplify `style_guide` to a fixed template + scene.style

### P4: Implement Per-Shot Re-generation (Phase D from Plan)
- Only regenerate shots whose inputs changed
- Use screenwriter agent's `characters_present[]` as cache key
