---
phase: 21-sequence-ui-polish
plan: 01
subsystem: api, ui
tags: [fastapi, typescript, pydantic, sequence, drag-and-drop]

# Dependency graph
requires:
  - phase: 16-production-bible-foundation
    provides: "Sequence model, sequence CRUD endpoints, scene_order column"
provides:
  - "PUT /api/sequences/{id}/scenes/reorder bulk scene reorder endpoint"
  - "SceneListItem.scene_order TypeScript field"
  - "SceneReorderInSequenceRequest TypeScript type"
  - "reorderScenesInSequence() client function"
affects: [21-02-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Bulk reorder endpoint pattern (matching existing reorder_sequences)"]

key-files:
  created: []
  modified:
    - backend/vidpipe/api/sequences.py
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts

key-decisions:
  - "SceneReorderRequest placed in sequences.py (not separate file) since it extends existing sequence endpoints"
  - "scene_order field added after sequence_id in SceneListItem for logical grouping"

patterns-established:
  - "Nested reorder pattern: /api/{parent}/{id}/{child}/reorder for bulk child ordering"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-03-01
---

# Phase 21 Plan 01: Sequence Scene Reorder API + Frontend Types Summary

**Bulk scene reorder endpoint (PUT /api/sequences/{id}/scenes/reorder) with SceneListItem.scene_order field and reorderScenesInSequence client function**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-01T23:07:32Z
- **Completed:** 2026-03-01T23:09:05Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added PUT /api/sequences/{id}/scenes/reorder endpoint with ownership validation (404 sequence, 422 scene membership)
- Added scene_order field to SceneListItem TypeScript interface for safe access in SortableSequenceSection
- Added reorderScenesInSequence() client function following existing reorderSequences pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Add bulk scene reorder endpoint to sequences.py** - `4b6d6db` (feat)
2. **Task 2: Add scene_order to SceneListItem type and reorderScenesInSequence client function** - `ab60b10` (feat)

## Files Created/Modified
- `backend/vidpipe/api/sequences.py` - Added SceneReorderRequest model and reorder_scenes_in_sequence endpoint
- `frontend/src/api/types.ts` - Added scene_order to SceneListItem, added SceneReorderInSequenceRequest type
- `frontend/src/api/client.ts` - Added SceneReorderInSequenceRequest import and reorderScenesInSequence function

## Decisions Made
- SceneReorderRequest placed in sequences.py alongside existing sequence schemas (not a separate file) since it extends the sequence domain
- scene_order field positioned after sequence_id in SceneListItem for logical grouping of sequence-related fields

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All backend and frontend foundations ready for Plan 02 (frontend UI wiring with drag-and-drop)
- reorderScenesInSequence() available for SortableSequenceSection to call on drag-end

## Self-Check: PASSED

All files exist, all commits verified.

---
*Phase: 21-sequence-ui-polish*
*Completed: 2026-03-01*
