---
phase: 15-video-generation-editor
plan: 02
subsystem: pipeline
tags: [gap-filling, generation-status, stop-flag, storyboard, keyframes, video-gen, stitcher, orchestrator]

# Dependency graph
requires:
  - phase: 15-video-generation-editor
    provides: Plan 01 - Draft project creation, Scene.generation_status column, generate endpoint
  - phase: 02-generation-pipeline
    provides: Pipeline stages (storyboard, keyframes, video_gen, stitcher)
  - phase: 03-orchestration-interfaces
    provides: Pipeline orchestrator with state machine and resume logic
provides:
  - Gap-filling storyboard that preserves user-provided scene text
  - Gap-filling keyframes that skip existing start/end keyframes individually
  - Gap-filling video_gen that skips scenes with existing VideoClip
  - Gap-filling stitcher that skips if output file already exists
  - Per-scene generation_status tracking in storyboard, keyframes, and video_gen
  - Per-scene stop flag checking in keyframes and video_gen loops
  - Draft-to-pending transition in orchestrator
  - Draft-aware _check_completed_steps (verifies scene text, not just scene existence)
affects: [15-video-generation-editor, frontend-video-editor]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gap-filling pattern: check for existing content before generating, skip if present"
    - "generation_status lifecycle: set before operation, clear after, set 'failed' on exception"
    - "Per-scene stop flag: refresh project status and check 'stopped' at top of each scene loop"

key-files:
  created: []
  modified:
    - backend/vidpipe/pipeline/storyboard.py
    - backend/vidpipe/pipeline/keyframes.py
    - backend/vidpipe/pipeline/video_gen.py
    - backend/vidpipe/pipeline/stitcher.py
    - backend/vidpipe/orchestrator/pipeline.py

key-decisions:
  - "Draft storyboard: partition scenes into filled/empty, pass filled as LLM context, generate only for empty indices"
  - "Keyframe gap-filling checks start/end individually (not just both-or-nothing) to handle partial uploads"
  - "generation_status set in outer try/except to guarantee 'failed' state on any exception"
  - "_check_completed_steps now verifies scenes have non-empty text, not just scene count (draft awareness)"
  - "Stitcher gap-filling requires both output_path set AND file on disk (avoids stale path edge case)"

patterns-established:
  - "Gap-filling is additive: if existing content skip; else proceed as normal (no regression for old projects)"
  - "generation_status lifecycle: set before, clear after, failed on exception (3 states per operation)"

requirements-completed: [VGED-06, VGED-11]

# Metrics
duration: 4min
completed: 2026-02-21
---

# Phase 15 Plan 02: Pipeline Gap-Filling Summary

**All four pipeline stages (storyboard, keyframes, video_gen, stitcher) modified for gap-filling mode with per-scene generation_status tracking and per-scene stop flag checking**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-21T23:07:56Z
- **Completed:** 2026-02-21T23:12:32Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Storyboard stage detects existing scenes, preserves user-provided text, and generates only for empty scenes with filled scenes as LLM context
- Keyframes stage checks for existing start/end keyframes individually, generates only missing ones, sets generation_status per phase
- Video generation stage skips scenes with existing completed clips, wraps generation in generation_status lifecycle
- Stitcher skips if final video file already exists at project.output_path
- Orchestrator handles draft-to-pending transition and refined _check_completed_steps for draft awareness

## Task Commits

Each task was committed atomically:

1. **Task 1: Add gap-filling and per-scene status to storyboard and keyframes stages** - `01a2508` (feat)
2. **Task 2: Add gap-filling and per-scene status to video_gen, stitcher, and orchestrator** - `bc775c4` (feat)

## Files Created/Modified
- `backend/vidpipe/pipeline/storyboard.py` - Gap-filling: detect existing scenes, partition into filled/empty, update only empty rows, set generation_status
- `backend/vidpipe/pipeline/keyframes.py` - Gap-filling: check individual start/end keyframes, skip existing, generation_status lifecycle, per-scene stop flag
- `backend/vidpipe/pipeline/video_gen.py` - Gap-filling: skip scenes with existing VideoClip with local_path, generation_status lifecycle
- `backend/vidpipe/pipeline/stitcher.py` - Gap-filling: skip stitch if output file exists and output_path is set
- `backend/vidpipe/orchestrator/pipeline.py` - Draft-to-pending transition, _check_completed_steps verifies scene text content

## Decisions Made
- Draft storyboard partitions scenes into filled (user text) and empty (need generation), generates only for empty indices while passing filled scenes as context to the LLM
- Keyframe gap-filling checks start and end keyframes individually (not all-or-nothing) to handle partial user uploads
- generation_status lifecycle uses outer try/except to guarantee "failed" state on any exception
- _check_completed_steps now verifies that scenes have non-empty scene_description AND start_frame_prompt to confirm storyboard is actually done (draft projects have Scene rows but empty text)
- Stitcher gap-filling requires both project.output_path to be set AND the file to exist on disk to skip (avoids stale path from previous run)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed _check_completed_steps draft awareness**
- **Found during:** Task 2
- **Issue:** _check_completed_steps counted Scene rows as "has storyboard" regardless of content. Draft projects with empty scenes would be detected as having a storyboard, causing the pipeline to skip storyboarding and start at keyframing with empty scenes.
- **Fix:** Added check that at least one scene has non-empty scene_description AND start_frame_prompt
- **Files modified:** backend/vidpipe/orchestrator/pipeline.py
- **Committed in:** bc775c4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Essential fix for correctness of draft project pipeline resume. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All pipeline stages support gap-filling mode for VideoGenEditor
- generation_status tracking enables frontend per-scene spinners
- Per-scene stop flag enables responsive pause/resume at scene granularity
- Ready for Plan 03: Frontend VideoGenEditor integration

## Self-Check: PASSED

All files and commits verified:
- 15-02-SUMMARY.md: FOUND
- Commit 01a2508 (Task 1): FOUND
- Commit bc775c4 (Task 2): FOUND
- storyboard.py: FOUND
- keyframes.py: FOUND
- video_gen.py: FOUND
- stitcher.py: FOUND
- pipeline.py: FOUND

---
*Phase: 15-video-generation-editor*
*Completed: 2026-02-21*
