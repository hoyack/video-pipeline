---
phase: 19-bible-context-fix-cleanup
plan: 01
subsystem: api, database, services
tags: [sqlalchemy, fastapi, pydantic, production-bible, screenwriter]

# Dependency graph
requires:
  - phase: 16-production-bible-foundation
    provides: ProductionBible model and rename migrations
  - phase: 18-screenplay-system
    provides: ScreenwriterService with load_bible_context, screenplay endpoints
provides:
  - Production.production_bible_id FK column for direct bible lookup
  - Fixed load_bible_context that works before any Scenes exist
  - ProductionResponse API schema with production_bible_id field
  - Scene production_bible_id propagation from Production in screenplay flow
  - Clean sound_router import (no try/except guard)
affects: [19-02-PLAN, frontend-production-detail, storyboard-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Direct Production->ProductionBible FK for bible context lookup (not via Scene)"

key-files:
  created: []
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/services/screenwriter.py
    - backend/vidpipe/api/routes.py
    - backend/vidpipe/api/screenplay.py
    - backend/vidpipe/api/app.py

key-decisions:
  - "Production.production_bible_id FK enables bible context before scenes exist"
  - "Direct import for sound_router since module now exists (Phase 17-03 complete)"

patterns-established:
  - "Production-level bible association: query Production.production_bible_id instead of Scene.production_bible_id for cross-entity lookups"

requirements-completed: [SCRN-10]

# Metrics
duration: 3min
completed: 2026-03-01
---

# Phase 19 Plan 01: Bible Context Fix + Cleanup Summary

**Direct Production.production_bible_id FK fixes load_bible_context for bible-first screenplay flow, with API schema updates and sound_router cleanup**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-01T21:50:05Z
- **Completed:** 2026-03-01T21:53:05Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Fixed load_bible_context to query Production.production_bible_id directly, making bible context available even before any Scenes exist
- Fixed generate_scene_breakdown entity validation to use the same Production-based lookup
- Added production_bible_id to ProductionResponse and ProductionUpdate API schemas
- Propagated Production.production_bible_id to new Scenes in generate_scenes_from_screenplay
- Cleaned up sound_router import from try/except guard to direct import

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Production.production_bible_id FK, migration, and fix load_bible_context + entity validation** - `81fc020` (feat)
2. **Task 2: Update ProductionResponse, propagate bible_id to scenes, fix screenplay endpoint, clean sound_router** - `4d72708` (feat)

## Files Created/Modified
- `backend/vidpipe/db/models.py` - Added Production.production_bible_id nullable FK column
- `backend/vidpipe/db/__init__.py` - Added ALTER TABLE migration for production_bible_id
- `backend/vidpipe/services/screenwriter.py` - Rewrote load_bible_context and entity validation to use Production directly
- `backend/vidpipe/api/routes.py` - Updated ProductionResponse/ProductionUpdate schemas and all response constructors
- `backend/vidpipe/api/screenplay.py` - Propagates Production.production_bible_id to new Scene rows
- `backend/vidpipe/api/app.py` - Direct sound_router import replacing try/except guard

## Decisions Made
- Production.production_bible_id FK enables bible context before any scenes exist (fixes the core bug where "Bible -> Screenplay -> Generate Full" lost all bible context)
- Direct import for sound_router since the module has existed since Phase 17-03 (try/except was a forward-compat guard that's no longer needed)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Production bible context now available in screenplay flow before scene creation
- ProductionResponse includes production_bible_id for frontend display
- Ready for Phase 19 Plan 02 (remaining cleanup tasks)

## Self-Check: PASSED

All 6 modified files exist. Both task commits (81fc020, 4d72708) verified in git log. SUMMARY.md created.

---
*Phase: 19-bible-context-fix-cleanup*
*Completed: 2026-03-01*
