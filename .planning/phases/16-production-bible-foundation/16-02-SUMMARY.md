---
phase: 16-production-bible-foundation
plan: 02
subsystem: database, api
tags: [sqlalchemy, fastapi, sequences, pydantic, crud]

# Dependency graph
requires:
  - phase: 16-01
    provides: Production model and productions table that Sequence belongs to

provides:
  - Sequence ORM model with title, description, order, act, color fields
  - Scene.sequence_id FK and Scene.scene_order for grouping membership
  - 7 CRUD API endpoints under /api/productions/{id}/sequences and /api/sequences/{id}
  - Bulk reorder endpoint for drag-and-drop ordering
  - Assign/unassign scene-to-sequence endpoint

affects: [16-04-drag-drop-sequencing-ui, sequences-frontend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Dedicated route file per domain (sequences.py) per CLAUDE.md convention
    - DELETE unsequences child rows rather than cascade-deleting them
    - Auto-increment order fields (max+1) for append semantics
    - Model fields_set check for nullable field updates (allows clearing act/color to None)

key-files:
  created:
    - backend/vidpipe/api/sequences.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/api/app.py

key-decisions:
  - "Sequence placed between Production and Scene in models.py declaration order — Production must exist before Sequence FK is resolved"
  - "DELETE sequence unsequences child scenes (sets sequence_id=NULL) not delete — prevents accidental data loss"
  - "Auto-increment scene_order within target sequence (max+1) when not provided — append semantics match drag-and-drop UX"
  - "model_fields_set check for act/color update — allows explicit None to clear optional fields without unset ambiguity"
  - "Act values validated against VALID_ACTS set (ACT_1, ACT_2, ACT_3) — rejects invalid values with 422"

patterns-established:
  - "New API domains get dedicated route files (sequences.py) — not appended to routes.py"
  - "Bulk reorder validates all IDs belong to the production before updating — prevents cross-production contamination"

requirements-completed: [SEQ-01, SEQ-02, SEQ-03, SEQ-04]

# Metrics
duration: 2min
completed: 2026-03-01
---

# Phase 16 Plan 02: Sequence Grouping Layer Summary

**SQLAlchemy Sequence model with production/scene FKs and 7-endpoint CRUD API for narrative chapter organization above the Scene layer**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-01T01:06:54Z
- **Completed:** 2026-03-01T01:09:02Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added `Sequence` ORM model (id, production_id, title, description, order, act, color, created_at, updated_at) with FK to productions
- Added `sequence_id` FK and `scene_order` column to Scene model with idempotent migration entries
- Created `backend/vidpipe/api/sequences.py` with 7 fully async CRUD endpoints
- Registered `sequence_router` in `app.py` immediately after main router

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Sequence model and Scene FK migration** - `8f0f785` (feat)
2. **Task 2: Create Sequence CRUD API routes** - `4d8a6de` (feat)
3. **Fix: restore Sequence import after Plan 01 concurrent edit** - `f049dbd` (fix)

**Plan metadata:** (see final commit below)

## Files Created/Modified
- `backend/vidpipe/db/models.py` - Added Sequence class and sequence_id/scene_order on Scene
- `backend/vidpipe/db/__init__.py` - Imported Sequence; added Phase 16 migration entries
- `backend/vidpipe/api/sequences.py` - New: 7 sequence CRUD endpoints
- `backend/vidpipe/api/app.py` - Registered sequence_router

## Decisions Made
- DELETE sequence unsequences child scenes rather than deleting them — prevents accidental scene loss
- Auto-increment scene_order (max+1) on assign when not explicitly provided — append semantics match drag-and-drop
- `model_fields_set` check for act/color to allow explicitly clearing optional fields to None
- VALID_ACTS validated server-side with 422 response — rejects ACT_4, etc.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Re-added Sequence import to db/__init__.py after Plan 01 external edit**
- **Found during:** Task 2 verification
- **Issue:** Plan 01 ran concurrently and rewrote the import line in db/__init__.py, dropping the `Sequence` import we added in Task 1
- **Fix:** Re-added `Sequence` to the import line to ensure `Base.metadata.create_all()` includes the sequences table
- **Files modified:** backend/vidpipe/db/__init__.py
- **Verification:** `from vidpipe.db import async_session, init_database; from vidpipe.db.models import Sequence` imports successfully
- **Committed in:** f049dbd

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking concurrent edit)
**Impact on plan:** Essential for correctness — without the import, the sequences table would not be created by create_all(). No scope creep.

## Issues Encountered
- Plan 01 ran concurrently and modified `db/__init__.py` and `db/models.py` (ProductionBible rename). Applied Rule 3 auto-fix to restore Sequence import. The Plan 01 changes were compatible (Manifest alias preserved, Sequence model placement correct).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Sequence model and CRUD API are complete and tested
- Frontend drag-and-drop grouping UI (Plan 04) can consume all 7 endpoints
- Scene assignment endpoint handles both assign and unassign via PUT /api/scenes/{id}/sequence
- Bulk reorder endpoint ready for @dnd-kit integration

---
*Phase: 16-production-bible-foundation*
*Completed: 2026-03-01*
