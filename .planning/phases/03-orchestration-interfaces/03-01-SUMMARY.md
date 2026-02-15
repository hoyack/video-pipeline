---
phase: 03-orchestration-interfaces
plan: 01
subsystem: orchestrator
tags: [orchestration, state-machine, idempotency, resume-logic, metadata-tracking]

dependency_graph:
  requires:
    - vidpipe.pipeline.storyboard.generate_storyboard
    - vidpipe.pipeline.keyframes.generate_keyframes
    - vidpipe.pipeline.video_gen.generate_videos
    - vidpipe.pipeline.stitcher.stitch_videos
    - vidpipe.db.models (Project, Scene, Keyframe, VideoClip, PipelineRun)
  provides:
    - vidpipe.orchestrator.run_pipeline
    - vidpipe.orchestrator.state (state machine constants and helpers)
  affects:
    - CLI implementation (will use run_pipeline with progress_callback)
    - API implementation (will use run_pipeline without progress_callback)

tech_stack:
  added: []
  patterns:
    - State machine pattern for pipeline transitions
    - Idempotent step execution with database state checks
    - Progress callback interface for decoupled UI updates
    - Per-step timing with monotonic clock for accuracy

key_files:
  created:
    - vidpipe/orchestrator/__init__.py
    - vidpipe/orchestrator/state.py
    - vidpipe/orchestrator/pipeline.py
  modified: []

decisions:
  - Use completed_steps dict from database queries for failed state resume logic
  - Wrap each step in try/except with project.status and error_message persistence
  - Use time.monotonic() for step timing (not datetime) to avoid timezone issues
  - Detect and fix status mismatch (generate_keyframes sets "generating_video" vs state machine expects "video_gen")

metrics:
  duration_seconds: 129
  tasks_completed: 2
  files_created: 3
  commits: 2
  completed_at: "2026-02-15T02:03:36Z"
---

# Phase 03 Plan 01: Pipeline Orchestrator Summary

**One-liner:** Implemented state machine orchestrator with idempotent resume, PipelineRun metadata tracking, and progress callback interface.

## What Was Built

Created `vidpipe/orchestrator/` module with state machine coordination for the full video generation pipeline.

**State Machine (`state.py`):**
- 7 pipeline states: pending, storyboarding, keyframing, video_gen, stitching, complete, failed
- 5 state transitions mapping active steps to next steps
- RESUMABLE_STATES set excluding only "complete"
- `can_resume()` helper checking if status is resumable
- `get_resume_step()` logic using completed_steps dict to find re-entry point for failed pipelines

**Pipeline Orchestrator (`pipeline.py`):**
- `run_pipeline(session, project_id, progress_callback)` main entry point
- Sequential 4-step execution with state machine transitions
- Idempotent resume using `_check_completed_steps()` database queries:
  - `has_storyboard`: checks for Scene records
  - `has_keyframes`: checks for start/end Keyframe records on all scenes
  - `has_clips`: checks for completed/rai_filtered VideoClip records on all scenes
- Per-step error handling with `project.status = "failed"` and `project.error_message` persistence
- PipelineRun metadata tracking with `started_at`, `completed_at`, `total_duration_seconds`, and `log` (per-step timing)
- Progress callback invocations for CLI display ("Generating storyboard...", etc.)

**Module Exports (`__init__.py`):**
- Public API: `run_pipeline` exported for CLI and API use

## Implementation Details

**Idempotent Resume Logic:**
The orchestrator can resume from any interrupted or failed state by querying the database for completed work:

1. Failed project → query database for completed_steps
2. Determine resume step based on what's actually in database
3. Reset status from "failed" to resume step status
4. Execute remaining steps (skipping completed work)

**State Transition Flow:**
```
pending -> storyboarding (during storyboard step)
storyboarding -> keyframing (by generate_storyboard)
keyframing -> video_gen (by generate_keyframes + orchestrator fixup)
video_gen -> stitching (by orchestrator)
stitching -> complete (by stitch_videos)
```

**Metadata Tracking:**
PipelineRun records capture:
- Execution start/end timestamps
- Total duration in seconds
- Per-step timing in log dict: `{"storyboard": 12.5, "keyframes": 45.2, ...}`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Circular import in __init__.py**
- **Found during:** Task 1 verification
- **Issue:** __init__.py imported run_pipeline from pipeline.py before it existed, causing ModuleNotFoundError
- **Fix:** Temporarily removed import in Task 1, added it back in Task 2 after pipeline.py was created
- **Files modified:** vidpipe/orchestrator/__init__.py
- **Commit:** 144eb97

**2. [Rule 1 - Bug] Status mismatch between generate_keyframes and state machine**
- **Found during:** Task 2 implementation
- **Issue:** generate_keyframes sets `project.status = "generating_video"` but state machine expects "video_gen"
- **Fix:** Added status correction after keyframes step: `if project.status == "generating_video": project.status = "video_gen"`
- **Files modified:** vidpipe/orchestrator/pipeline.py
- **Commit:** d0d7a88

## Testing & Verification

All verification checks passed:
- `from vidpipe.orchestrator import run_pipeline` imports successfully
- `from vidpipe.orchestrator.state import PIPELINE_STATES, STEP_TRANSITIONS, can_resume, get_resume_step` imports successfully
- PIPELINE_STATES contains all 7 states
- STEP_TRANSITIONS maps 5 active states correctly
- `can_resume("failed")` returns True
- `can_resume("complete")` returns False
- run_pipeline signature accepts (session, project_id, progress_callback) parameters

## Dependencies Ready for Next Plans

The orchestrator module is now ready for:
- **Plan 03-02 (CLI):** Will import run_pipeline and pass Rich progress callback
- **Plan 03-03 (API):** Will import run_pipeline without progress_callback for background task execution

Both interfaces can leverage the same orchestration logic with crash recovery and resume capability.

## Files Created

1. **vidpipe/orchestrator/__init__.py** (11 lines)
   - Module docstring and public API exports

2. **vidpipe/orchestrator/state.py** (97 lines)
   - PIPELINE_STATES, STEP_TRANSITIONS, RESUMABLE_STATES constants
   - can_resume() and get_resume_step() helper functions

3. **vidpipe/orchestrator/pipeline.py** (257 lines)
   - run_pipeline() main orchestrator function
   - _check_completed_steps() database query helper
   - Error handling and PipelineRun metadata tracking

## Key Decisions

1. **Use completed_steps dict for resume logic:** Querying database state (scenes, keyframes, clips) ensures accurate resume even if status field is stale or corrupted.

2. **Fix status mismatch inline:** Rather than changing generate_keyframes (Phase 2 code), corrected the status in orchestrator to match state machine expectations.

3. **Separate step timing from PipelineRun creation:** Create PipelineRun record at start, update with timing on completion for accurate metadata.

4. **Progress callback as optional parameter:** Enables CLI progress display without coupling orchestrator to Rich library.

## Self-Check: PASSED

**Created files exist:**
```bash
FOUND: vidpipe/orchestrator/__init__.py
FOUND: vidpipe/orchestrator/state.py
FOUND: vidpipe/orchestrator/pipeline.py
```

**Commits exist:**
```bash
FOUND: 144eb97 (Task 1)
FOUND: d0d7a88 (Task 2)
```

**Import verification:**
All imports successful, function signatures correct.
