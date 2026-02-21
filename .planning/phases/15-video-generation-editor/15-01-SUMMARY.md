---
phase: 15-video-generation-editor
plan: 01
subsystem: api
tags: [fastapi, draft-projects, video-editor, endpoints, sqlalchemy]

# Dependency graph
requires:
  - phase: 03-orchestration-interfaces
    provides: API router, background pipeline execution, resume endpoint pattern
  - phase: 06-generateform-integration
    provides: Manifest snapshot creation and usage tracking
provides:
  - POST /api/projects endpoint for draft project creation without pipeline start
  - POST /api/projects/{id}/generate endpoint for gap-filling pipeline execution
  - PUT /api/projects/{id}/final-video endpoint for video upload
  - Scene.generation_status column for per-scene progress tracking
  - "draft" status in PIPELINE_STATES and RESUMABLE_STATES
affects: [15-video-generation-editor, frontend-video-editor]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Draft project pattern: create project with empty Scene rows, start pipeline separately"
    - "Gap-filling generation: start_generation endpoint accepts any resumable state including draft/complete"

key-files:
  created: []
  modified:
    - backend/vidpipe/orchestrator/state.py
    - backend/vidpipe/db/models.py
    - backend/vidpipe/api/routes.py

key-decisions:
  - "Empty string sentinel for draft scenes: gap-filling logic uses `if not s.scene_description.strip()` to detect empty scenes"
  - "Draft status added to both PIPELINE_STATES and RESUMABLE_STATES for full state machine support"
  - "POST /api/projects separate from POST /api/generate to maintain backward compatibility"
  - "scene_count 1-20 validation on draft creation for sensible limits"
  - "generate endpoint allows generation from complete state (re-run) in addition to draft/stopped/staged/failed"

patterns-established:
  - "Draft project creation: status=draft, empty Scene rows with all text fields as empty strings"
  - "generate endpoint model override pattern: optional body fields applied to project before pipeline start"

requirements-completed: [VGED-01, VGED-02, VGED-03, VGED-04, VGED-05]

# Metrics
duration: 4min
completed: 2026-02-21
---

# Phase 15 Plan 01: Video Generation Editor Backend Infrastructure Summary

**Draft project creation with empty scenes, gap-filling generate endpoint, final-video upload, and Scene.generation_status tracking column**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-21T23:01:45Z
- **Completed:** 2026-02-21T23:05:27Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added "draft" status to PIPELINE_STATES and RESUMABLE_STATES with proper get_resume_step handling
- Added Scene.generation_status column (nullable String(32)) for per-scene generation progress tracking
- Created POST /api/projects endpoint that creates draft projects with empty Scene rows without starting pipeline
- Created POST /api/projects/{id}/generate endpoint that starts pipeline from draft/stopped/staged/failed/complete states
- Created PUT /api/projects/{id}/final-video endpoint that accepts MP4 upload and sets project.output_path
- Added generation_status to SceneDetail API response and "draft" to project list filter

## Task Commits

Each task was committed atomically:

1. **Task 1: Add draft status to state machine and generation_status to Scene model** - `9930100` (feat)
2. **Task 2: Add POST /projects, POST /projects/{id}/generate, and PUT /projects/{id}/final-video endpoints** - `214cb12` (feat)

## Files Created/Modified
- `backend/vidpipe/orchestrator/state.py` - Added "draft" to PIPELINE_STATES, RESUMABLE_STATES, and get_resume_step
- `backend/vidpipe/db/models.py` - Added Scene.generation_status column (nullable String(32))
- `backend/vidpipe/api/routes.py` - Three new endpoints, four new Pydantic schemas, generation_status in SceneDetail, "draft" in VALID_STATUSES

## Decisions Made
- Empty string sentinel for draft scenes: gap-filling logic uses `if not s.scene_description.strip()` to detect empty scenes needing generation
- Draft status added to both PIPELINE_STATES and RESUMABLE_STATES for full state machine support
- POST /api/projects is a separate endpoint from POST /api/generate to maintain backward compatibility
- scene_count validated 1-20 on draft creation for sensible limits
- generate endpoint allows starting from "complete" state (re-generation) in addition to draft/stopped/staged/failed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend API surface complete for Video Generation Editor frontend (Plan 02)
- Scene.generation_status column ready for pipeline to update during execution
- Draft projects appear in project list with status "draft"
- Existing POST /api/generate endpoint unchanged (backward compatible)

## Self-Check: PASSED

All files and commits verified:
- 15-01-SUMMARY.md: FOUND
- Commit 9930100 (Task 1): FOUND
- Commit 214cb12 (Task 2): FOUND
- state.py: FOUND
- models.py: FOUND
- routes.py: FOUND

---
*Phase: 15-video-generation-editor*
*Completed: 2026-02-21*
