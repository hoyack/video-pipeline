# Video Generation Pipeline: Process Documentation

This document traces the two primary video generation workflows in the system — **"New" (immediate pipeline)** and **"New Draft" → "Generate All Phases" (deferred pipeline)** — and compares them to identify material differences in execution.

---

## Table of Contents

1. [Pipeline A: "New" (Immediate Full Pipeline)](#pipeline-a-new-immediate-full-pipeline)
2. [Pipeline B: "New Draft" → "Generate All Phases"](#pipeline-b-new-draft--generate-all-phases)
3. [Workflow Diagrams](#workflow-diagrams)
4. [Comparison Matrix](#comparison-matrix)
5. [Material Differences](#material-differences)

---

## Pipeline A: "New" (Immediate Full Pipeline)

**Entry Point:** User clicks **"+ New"** in SceneList → fills GenerateForm → submits.

### Step 1: Frontend Submission

The `GenerateForm` component collects all required fields upfront:
- `prompt` (required), `title`, `style`, `aspect_ratio`, `clip_duration`, `total_duration`
- `text_model`, `image_model`, `video_model` (all required)
- Optional: `manifest_id`, `vision_model`, `enable_audio`, `quality_mode`, `candidate_count`, `run_through`

Calls `POST /api/generate` with the full payload.

**Key:** All models and configuration are validated as **required** at submission time. The form cannot submit without them.

### Step 2: Scene Creation + Immediate Pipeline Launch

**Route:** `POST /api/generate` → `generate_video()` (`routes.py:678`)

1. **Strict validation** — aspect ratio, clip duration, all model IDs, audio capability, quality mode — all checked with hard 422 failures.
2. **Shot count derived** from `ceil(total_duration / clip_duration)`.
3. **Scene record created** with `status="pending"` — all fields populated.
4. If `manifest_id` provided: creates `ManifestSnapshot`, increments usage.
5. **No Shot rows created** — shots are created during storyboard generation.
6. `background_tasks.add_task(run_pipeline_background, scene_id)` — pipeline starts immediately.
7. Returns `202 Accepted` with `scene_id`.

### Step 3: Pipeline Orchestration

**Function:** `run_pipeline()` (`orchestrator/pipeline.py:187`)

The unified orchestrator runs all four stages sequentially in a single background task with a single database session. It creates a `PipelineRun` record for timing metadata and uses `get_resume_step()` for idempotent restart capability.

**Status state machine:**
```
pending → storyboarding → keyframing → video_gen → stitching → complete
```

Each transition is committed to the database. On failure at any point: `status="failed"`, `error_message` persisted.

#### Stage 1: Storyboard (`storyboard.py:276`)

1. Checks for existing shots (gap-filling for resumed scenes).
2. Builds system prompt with style, aspect ratio, shot count.
3. If manifest-aware: loads asset registry, uses `EnhancedStoryboardOutput` schema.
4. Calls LLM with structured output (retry up to 3x with temperature reduction).
5. **Creates Shot rows** from LLM output — `shot_description`, `start_frame_prompt`, `end_frame_prompt`, `video_motion_prompt`, `transition_notes`.
6. If manifest-aware: creates `ShotManifest` and `ShotAudioManifest` rows per shot.
7. Sets `scene.status = "keyframing"`.

**Critical detail:** In Pipeline A, **no Shot rows exist before this stage**. The storyboard stage creates them fresh from the LLM output.

#### Stage 2: Keyframes (`keyframes.py:377`)

For each shot sequentially:
1. Shot 0: generates start frame via text-to-image (Imagen/ComfyUI).
   - Prompt = style prefix + character prefix + `shot.start_frame_prompt`
   - If manifest: adaptive prompt rewriting, reference image resolution, face verification.
2. Shots 1+: **inherits** previous shot's end frame as start frame (KEYF-03 continuity).
3. Generates end frame with image conditioning (start frame → end frame).
4. Creates `Keyframe` records (position=start/end) with filesystem paths.
5. Sets `shot.status = "keyframes_done"` after each shot.
6. Rate-limiting delay between shots.
7. Sets `scene.status = "generating_video"`.

**The orchestrator then normalizes** `"generating_video"` → `"video_gen"` (status name mismatch between keyframe stage and state machine).

#### Stage 3: Video Generation (`video_gen.py:796`)

For each shot with `status="keyframes_done"`:
1. Loads keyframe images from filesystem.
2. Gets video prompt (`shot.video_motion_prompt` or rewritten version).
3. Routes to Veo or ComfyUI based on `video_model`.
4. Submits generation request with keyframes + prompt.
5. Polls for completion, downloads video.
6. If quality mode: generates N candidates, scores via CV, picks best.
7. If manifest-aware: runs CV analysis for asset appearance tracking.
8. Creates `VideoClip` record, saves to filesystem.
9. Sets `shot.status = "video_done"`.
10. Orchestrator sets `scene.status = "stitching"`.

#### Stage 4: Stitching (`stitcher.py:25`)

1. Queries all `VideoClip` records in shot order.
2. If `crossfade_seconds == 0`: ffmpeg concat (stream copy, no re-encoding).
3. If crossfade: ffmpeg xfade filter (re-encoding).
4. Saves `final.mp4` to `tmp/{scene_id}/output/`.
5. Sets `scene.status = "complete"`, `scene.output_path`.
6. Auto-creates initial PipeSVN checkpoint.

---

## Pipeline B: "New Draft" → "Generate All Phases"

**Entry Point:** User clicks **"+ New Draft"** in SceneList → configures in EditModeOverlay → clicks **"All Phases"**.

### Step 1: Draft Creation

The `handleNewDraft()` function calls `POST /api/scenes` with **no arguments** (empty body, all defaults).

**Route:** `POST /api/scenes` → `create_draft_scene()` (`routes.py:804`)

1. **Lenient validation** — all fields are optional. Aspect ratio, models, clip duration can all be `None`.
2. **Shot count** defaults to `1` (from `CreateSceneRequest.shot_count = 1`).
3. **Scene record created** with `status="draft"`:
   - `prompt = ""` (empty string)
   - `style = ""` (empty string)
   - `aspect_ratio = ""` (empty string)
   - `target_clip_duration = 0`
   - `text_model = None`, `image_model = None`, `video_model = None`
4. **Empty Shot rows created immediately** — 1 row (default) with all text fields as empty strings.
5. **No pipeline launched.** Returns `201 Created`.

### Step 2: User Configuration in Edit Mode

Frontend navigates to `/scenes/{id}`. `SceneDetail` detects `status="draft"` and auto-enters edit mode, rendering `EditModeOverlay`.

The user configures:
- Prompt, title, style
- Aspect ratio, clip duration, shot count
- Text model, image model, video model
- Manifest, audio, quality mode
- Individual shot descriptions (optional — inline editing)

Changes are held in local component state until the user triggers generation.

### Step 3: "All Phases" Trigger

User clicks the **"All Phases"** button (`EditModeOverlay.tsx:1510`).

**`handleRegenerate("all_phases")`** (`EditModeOverlay.tsx:724`):
1. **Auto-saves pending edits** by calling `PATCH /api/scenes/{id}/edit` (`editScene()`).
   - This persists prompt, style, models, shot count changes, shot text edits.
   - Creates a PipeSVN checkpoint for the edit.
   - Shot expansion: if user increased shot count, new empty Shot rows are appended.
2. Calls `POST /api/scenes/{id}/regenerate` with `scope="all_phases"`.
   - Passes current model selections and optional `run_through`.

### Step 4: Regeneration Chain

**Route:** `POST /api/scenes/{id}/regenerate` → `regenerate_scene()` (`routes.py:5038`)

Validates models based on `run_through` scope. Launches `_run_all_phases_regeneration()` as a background task.

**Function:** `_run_all_phases_regeneration()` (`routes.py:5405`)

This chains four **independent** per-phase regeneration functions sequentially. Each function:
- Opens its **own fresh database session** (not shared).
- Applies model overrides.
- Calls the same underlying pipeline function (e.g., `generate_storyboard`).
- **Saves and restores the scene status** — the phase function sets status internally (e.g., `"keyframing"`), but the wrapper restores the original status afterward.
- Creates a PipeSVN checkpoint after each phase.
- Emits WebSocket events for frontend progress tracking.

#### Phase 1: `_run_storyboard_regeneration()` (`routes.py:5220`)

1. Opens fresh session, loads scene.
2. Applies `text_model` override if provided.
3. Saves current `scene.status`.
4. Calls `generate_storyboard(session, scene, text_adapter)` — **same function as Pipeline A**.
5. `generate_storyboard` finds **existing empty Shot rows** (created during draft creation) → enters **gap-filling mode**: updates empty shots in-place rather than creating new rows.
6. Restores `scene.status` to saved value (undoes the `"keyframing"` transition).
7. Creates checkpoint: "Regenerated storyboard text".

#### Phase 2: `_run_keyframes_regeneration()` (`routes.py:5270`)

1. Opens fresh session, loads scene.
2. Applies `image_model` and `text_model` overrides.
3. Saves current `scene.status`.
4. Calls `generate_keyframes(session, scene, text_adapter)` — **same function as Pipeline A**.
5. Restores `scene.status`.
6. Creates checkpoint: "Regenerated stale keyframes".

#### Phase 3: `_run_clips_regeneration()` (`routes.py:5326`)

1. Opens fresh session, loads scene.
2. Applies `video_model` and `text_model` overrides.
3. Saves current `scene.status`.
4. **Gap-fix:** iterates shots, any shot without a `VideoClip` gets `status = "keyframes_done"` so `generate_videos` picks it up.
5. Calls `generate_videos(session, scene, text_adapter, vision_adapter)` — **same function as Pipeline A**.
6. Restores `scene.status`.
7. Creates checkpoint: "Regenerated stale clips".

#### Phase 4: `_run_restitch()` (`routes.py:5199`)

1. Opens fresh session, loads scene.
2. Calls `stitch_videos(session, scene)` — **same function as Pipeline A**.
3. Creates checkpoint: "Re-stitched video".
4. Emits `regen_complete` event.

---

## Workflow Diagrams

### Pipeline A: "New" (Immediate Full Pipeline)

```
USER                          FRONTEND                         BACKEND
 │                               │                                │
 ├─ Clicks "+ New" ─────────────►│                                │
 │                               ├─ Shows GenerateForm            │
 │                               │  (all fields required)         │
 ├─ Fills form, clicks ─────────►│                                │
 │  "Generate"                   │                                │
 │                               ├─ POST /api/generate ──────────►│
 │                               │  {prompt, style, models...}    │
 │                               │                                ├─ Validate ALL fields (strict)
 │                               │                                ├─ Derive shot_count = ceil(dur/clip)
 │                               │                                ├─ Create Scene (status="pending")
 │                               │                                │  (NO Shot rows yet)
 │                               │                                ├─ Create ManifestSnapshot (if manifest)
 │                               │                                ├─ Commit
 │                               │◄─ 202 {scene_id} ─────────────┤
 │                               │                                │
 │                               ├─ Navigate to ProgressView      ├─ BACKGROUND: run_pipeline()
 │                               │  (polls /status)               │  ┌────────────────────────────┐
 │                               │                                │  │ Single session, single task │
 │                               │                                │  │ PipelineRun record created  │
 │                               │                                │  └────────────────────────────┘
 │                               │                                │
 │                               │                                ├─ [1] STORYBOARD
 │                               │                                │  status: pending → storyboarding
 │                               │                                │  LLM generates structured output
 │                               │                                │  ► Creates Shot rows (NEW)
 │                               │                                │  ► Creates ShotManifest rows
 │                               │                                │  status → keyframing
 │                               │                                │
 │                               │                                ├─ [2] KEYFRAMES
 │                               │                                │  For each shot sequentially:
 │                               │                                │    Shot 0: text-to-image → start
 │                               │                                │    Shot N: inherit prev end → start
 │                               │                                │    Image-conditioned → end frame
 │                               │                                │  ► Creates Keyframe rows
 │                               │                                │  status → video_gen
 │                               │                                │
 │                               │                                ├─ [3] VIDEO GENERATION
 │                               │                                │  For each shot:
 │                               │                                │    Load keyframes + motion prompt
 │                               │                                │    Route to Veo or ComfyUI
 │                               │                                │    Poll → download clip
 │                               │                                │  ► Creates VideoClip rows
 │                               │                                │  status → stitching
 │                               │                                │
 │                               │                                ├─ [4] STITCHING
 │                               │                                │  ffmpeg concat/xfade
 │                               │                                │  ► final.mp4
 │                               │                                │  status → complete
 │                               │                                │  Auto-checkpoint created
 │                               │                                │
 │◄──────── Video ready ─────────┤◄─ status=complete ─────────────┤
```

### Pipeline B: "New Draft" → "Generate All Phases"

```
USER                          FRONTEND                         BACKEND
 │                               │                                │
 ├─ Clicks "+ New Draft" ───────►│                                │
 │                               ├─ POST /api/scenes ────────────►│
 │                               │  {empty body}                  │
 │                               │                                ├─ Create Scene (status="draft")
 │                               │                                │  prompt="", style="", models=None
 │                               │                                ├─ Create 1 EMPTY Shot row
 │                               │                                │  (all text fields = "")
 │                               │                                ├─ Commit
 │                               │◄─ 201 {scene_id} ─────────────┤
 │                               │                                │
 │                               ├─ Navigate to SceneDetail       │  *** No pipeline launched ***
 │                               ├─ Detect status="draft"         │
 │                               ├─ Auto-enter EditModeOverlay    │
 │                               │                                │
 ├─ Configures settings ────────►│                                │
 │  (prompt, models, style,      │  (held in component state)     │
 │   shots, audio, etc.)         │                                │
 │                               │                                │
 ├─ Clicks "All Phases" ───────►│                                │
 │                               │                                │
 │                               ├─ PATCH /api/scenes/{id}/edit ─►│
 │                               │  (auto-save pending edits)     ├─ Persist: prompt, style, models,
 │                               │                                │  shot count, shot text edits
 │                               │                                ├─ Expand Shot rows if count increased
 │                               │                                ├─ Create checkpoint
 │                               │◄─ 200 {head_sha} ─────────────┤
 │                               │                                │
 │                               ├─ POST /scenes/{id}/regenerate ►│
 │                               │  {scope: "all_phases"}         │
 │                               │                                ├─ Validate models for scope
 │                               │◄─ 202 Accepted ───────────────┤
 │                               │                                │
 │                               ├─ Show RegenProgressBar         ├─ BACKGROUND: _run_all_phases_regeneration()
 │                               │  (WebSocket events)            │  ┌─────────────────────────────────┐
 │                               │                                │  │ Chains 4 independent functions  │
 │                               │                                │  │ Each opens its OWN session      │
 │                               │                                │  │ No PipelineRun record           │
 │                               │                                │  │ Status saved/restored per phase │
 │                               │                                │  └─────────────────────────────────┘
 │                               │                                │
 │                               │                                ├─ [1] _run_storyboard_regeneration()
 │                               │                                │  New session
 │                               │                                │  generate_storyboard() ← same fn
 │                               │                                │  ► UPDATES existing empty Shot rows
 │                               │                                │    (gap-filling mode)
 │                               │                                │  Status restored (not advanced)
 │                               │                                │  Checkpoint: "Regenerated storyboard"
 │                               │                                │
 │                               │                                ├─ [2] _run_keyframes_regeneration()
 │                               │                                │  New session
 │                               │                                │  generate_keyframes() ← same fn
 │                               │                                │  ► Creates Keyframe rows
 │                               │                                │  Status restored
 │                               │                                │  Checkpoint: "Regenerated keyframes"
 │                               │                                │
 │                               │                                ├─ [3] _run_clips_regeneration()
 │                               │                                │  New session
 │                               │                                │  Gap-fix: set shot.status for pickup
 │                               │                                │  generate_videos() ← same fn
 │                               │                                │  ► Creates VideoClip rows
 │                               │                                │  Status restored
 │                               │                                │  Checkpoint: "Regenerated clips"
 │                               │                                │
 │                               │                                ├─ [4] _run_restitch()
 │                               │                                │  New session
 │                               │                                │  stitch_videos() ← same fn
 │                               │                                │  ► final.mp4
 │                               │                                │  Checkpoint: "Re-stitched video"
 │                               │                                │  Emit regen_complete
 │                               │                                │
 │◄──────── Video ready ─────────┤◄─ WS: regen_complete ──────────┤
```

---

## Comparison Matrix

| Aspect | Pipeline A: "New" | Pipeline B: "New Draft" → "All Phases" |
|--------|-------------------|----------------------------------------|
| **API Endpoint** | `POST /api/generate` | `POST /api/scenes` + `PATCH .../edit` + `POST .../regenerate` |
| **Initial Status** | `"pending"` | `"draft"` |
| **Models Required At Creation** | Yes — all three (text, image, video) | No — all nullable, can be `None` |
| **Prompt Required At Creation** | Yes | No — defaults to `""` |
| **Shot Rows At Creation** | None — created by storyboard LLM | Yes — pre-created as empty rows |
| **Shot Count Source** | `ceil(total_duration / clip_duration)` | `request.shot_count` (default: 1) |
| **Pipeline Trigger** | Automatic on scene creation | Manual — user clicks "All Phases" |
| **Edit Step Before Generation** | None | `PATCH /api/scenes/{id}/edit` auto-saves |
| **Background Task Function** | `run_pipeline()` (orchestrator) | `_run_all_phases_regeneration()` (routes.py) |
| **Session Strategy** | Single session for entire pipeline | Fresh session per phase (4 sessions total) |
| **PipelineRun Record** | Yes — timing metadata tracked | No — not created |
| **Status Machine** | Advances: pending→storyboarding→keyframing→video_gen→stitching→complete | Saved/restored per phase — **does not advance through state machine** |
| **Resume Capability** | `get_resume_step()` checks DB state | Each phase is independently re-runnable |
| **Storyboard: Shot Handling** | Creates new Shot rows from LLM output | Updates existing empty rows (gap-filling) |
| **Clips: Shot Status Prep** | Orchestrator naturally advances status | Gap-fix loop sets `shot.status = "keyframes_done"` |
| **Checkpoints** | 1 auto-checkpoint at completion | 4+ checkpoints (one per phase + edit) |
| **Error Handling** | Sets `scene.status = "failed"` + `error_message` | Emits WebSocket `error` event, **does not set scene status to failed** |
| **Stop/Cancel** | `_check_stopped()` polls between phases | Not implemented — no stop checks between chained phases |
| **run_through (Partial Execution)** | `PipelineStaged` exception at stage boundary | Early return within chain function |
| **Progress Communication** | `progress_callback` + event bus | WebSocket events only |
| **Expansion Shot Check** | `_generate_expansion_if_needed()` before keyframes | Not called — relies on edit step having expanded shots |

---

## Material Differences

### 1. Status Machine Bypass (HIGH IMPACT)

**Pipeline A** advances the scene through a well-defined status state machine:
```
pending → storyboarding → keyframing → video_gen → stitching → complete
```
Each transition is committed, enabling resume-from-failure at any point.

**Pipeline B** saves the scene's status before each phase and **restores it afterward**. The scene status does not advance through the state machine during generation. After all four phases complete, the scene's status reflects whatever it was before generation started (typically `"draft"` or its last-edited state). The final status depends on `stitch_videos()` setting `status="complete"` during the stitching phase — but the wrapper then **restores the saved status**, potentially overwriting "complete".

**Impact:** In Pipeline B, `scene.status` may not reflect the actual pipeline progress during or after execution. Frontend relies on WebSocket events (`regen_complete`) rather than database status polling.

### 2. No PipelineRun Metadata (MEDIUM IMPACT)

Pipeline A creates a `PipelineRun` record that tracks per-stage timing (`step_log`) and total duration. Pipeline B does not create this record, so there is no timing telemetry for draft-originated generations.

### 3. Session Isolation (MEDIUM IMPACT)

Pipeline A uses a **single database session** for the entire pipeline. This means all stages share the same ORM identity map, and objects loaded in one stage are available in the next without re-querying.

Pipeline B opens a **fresh session per phase**. Each phase must re-load the scene and related objects independently. While this is safer for crash isolation, it means:
- No shared ORM state between phases.
- Objects modified by one phase are only visible to the next after commit + re-query.
- The status save/restore pattern was necessary because of this session boundary.

### 4. Shot Creation Timing (MEDIUM IMPACT)

Pipeline A creates Shot rows **during storyboard generation** — the LLM decides the content and the rows are created fresh.

Pipeline B pre-creates **empty Shot rows at draft creation time** with a default count of 1. When the user later configures shot count (e.g., to 5), additional empty rows are appended via the edit endpoint. The storyboard stage then operates in **gap-filling mode**, updating these existing empty rows rather than creating new ones.

**Potential gap:** If the user changes `shot_count` in the edit step but the storyboard LLM generates a different number of shots than expected, the gap-filling logic must reconcile: it updates empty rows by index and creates new rows for any index the LLM produced that doesn't have an existing row.

### 5. Error Handling Divergence (MEDIUM IMPACT)

Pipeline A sets `scene.status = "failed"` and persists `scene.error_message` to the database on any exception. The scene's failure is durable — refreshing the page will show the error.

Pipeline B catches the exception in `_run_all_phases_regeneration()` and emits a WebSocket `error` event, but **does not update `scene.status` to "failed"**. If the WebSocket connection drops, the frontend may not learn about the failure. The scene remains in its pre-generation status indefinitely.

### 6. Stop/Cancel Handling (LOW-MEDIUM IMPACT)

Pipeline A calls `_check_stopped()` between each stage. If the user sets a stop flag, the pipeline halts gracefully with `scene.status = "stopped"`.

Pipeline B's chain function (`_run_all_phases_regeneration`) does **not** call `_check_stopped()` between phases. If a user requests cancellation, the currently-running phase will complete before the pipeline stops (if it stops at all — the stop mechanism may not propagate across the fresh-session boundaries).

### 7. Expansion Shot Check Missing (LOW IMPACT)

Pipeline A calls `_generate_expansion_if_needed()` before keyframes to handle fork/delete-then-expand scenarios.

Pipeline B does not call this function. It relies on the edit step having already expanded shot rows. This is likely fine for the happy path but could miss edge cases where shot expansion is needed after storyboard generation.

### 8. `total_duration` Derivation (LOW IMPACT)

Pipeline A computes `total_duration` from the user's explicit input and derives `shot_count = ceil(total_duration / clip_duration)`.

Pipeline B sets `total_duration = shot_count * (clip_duration or 6)` as a derived value at draft creation. If clip_duration is later changed via edit, `total_duration` may become stale unless the edit endpoint also recalculates it.

---

### Summary

Both pipelines ultimately invoke the **same core functions** (`generate_storyboard`, `generate_keyframes`, `generate_videos`, `stitch_videos`). The generative output — storyboard content, keyframe images, video clips, stitched output — is functionally identical.

The material differences are in the **orchestration layer**: Pipeline A uses a purpose-built orchestrator with proper state machine transitions, timing metadata, resume capability, and error persistence. Pipeline B chains the same functions through a simpler wrapper that trades orchestration rigor for flexibility (per-phase checkpoints, independent sessions, status restoration). The most impactful gaps are the status machine bypass (making database state unreliable for progress tracking) and the silent error handling (no durable failure state).
