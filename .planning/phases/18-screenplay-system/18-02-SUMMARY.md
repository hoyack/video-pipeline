---
phase: 18-screenplay-system
plan: 02
subsystem: api
tags: [fastapi, screenplay, rest-api, storyboard-enrichment, background-tasks]

# Dependency graph
requires:
  - phase: 18-screenplay-system
    provides: Screenplay ORM model, ScreenwriterService, Pydantic schemas, Scene screenplay columns
  - phase: 13-llm-provider-abstraction-ollama
    provides: LLMAdapter ABC and get_adapter() registry for model routing
provides:
  - Screenplay REST API (11 endpoints) with CRUD, status transitions, and generation
  - Scene creation from LOCKED screenplay breakdown (generate-scenes)
  - Storyboard prompt enrichment with screenplay context (SCRN-13/SCRN-14)
  - SceneListItem.screenplay_breakdown_index for frontend badge (SCRN-15)
affects: [18-03 (frontend UI), storyboard pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Domain-split route file (screenplay.py) following sequences.py pattern"
    - "Background task with plain-value parameters (no ORM objects from request session)"
    - "Upsert-style GET endpoint (get or create DRAFT)"
    - "Additive-only storyboard enrichment gated on screenplay_context presence"

key-files:
  created:
    - backend/vidpipe/api/screenplay.py
  modified:
    - backend/vidpipe/api/app.py
    - backend/vidpipe/api/routes.py
    - backend/vidpipe/pipeline/storyboard.py

key-decisions:
  - "Tasks 1 and 2 committed together since all endpoints reside in the same screenplay.py file"
  - "Background task for full generation opens its own async_session with plain-value params to avoid DetachedInstanceError"
  - "Screenplay enrichment in storyboard.py uses hasattr guard for backward compatibility with pre-migration Scene objects"
  - "generate-scenes defaults to cinematic style, 16:9 aspect, 8s clip duration for screenplay-created scenes"
  - "Force query param on generate-scenes for idempotent re-generation (deletes existing screenplay-linked scenes)"

patterns-established:
  - "Screenplay API: inline generation for individual steps (~5-10s), background task for full chain"
  - "ScreenplayResponse schema serializes all Screenplay fields including generating_step for progress polling"

requirements-completed: [SCRN-03, SCRN-06, SCRN-09, SCRN-12, SCRN-13, SCRN-14]

# Metrics
duration: 3min
completed: 2026-03-01
---

# Phase 18 Plan 02: Screenplay REST API and Storyboard Enrichment Summary

**11-endpoint Screenplay API with CRUD, per-step generation, scene creation from breakdown, and conditional storyboard prompt enrichment from screenplay context**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-01T15:46:37Z
- **Completed:** 2026-03-01T15:50:07Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- 11 Screenplay API endpoints: GET (upsert), PUT (LOCKED guard), PATCH status, POST generate-scenes, POST generate (full chain background), and 6 individual generation endpoints
- Scene creation from LOCKED screenplay breakdown with idempotency guard (force query param)
- Storyboard prompt enrichment with screenplay direction (slugline, intent, emotional beat, characters, set, props) when Scene.screenplay_context is present
- SceneListItem updated with screenplay_breakdown_index for SCRN-15 frontend badge

## Task Commits

Each task was committed atomically:

1. **Tasks 1+2: CRUD, status, generation endpoints + app.py registration + routes.py SceneListItem** - `9f4110f` (feat)
2. **Task 3: Screenplay context enrichment in storyboard.py** - `38c8daa` (feat)

## Files Created/Modified
- `backend/vidpipe/api/screenplay.py` - Screenplay REST API: CRUD, status transitions, 7 generation endpoints, generate-scenes
- `backend/vidpipe/api/app.py` - screenplay_router import and registration
- `backend/vidpipe/api/routes.py` - SceneListItem.screenplay_breakdown_index field and population
- `backend/vidpipe/pipeline/storyboard.py` - Conditional screenplay context enrichment in generate_storyboard()

## Decisions Made
- Tasks 1 and 2 committed together since all endpoints reside in the same screenplay.py file -- splitting would require an artificial partial-file commit
- Background task for full generation extracts plain values (production_id string, screenplay_id string, text_model, bible_context, user_settings sentinel) before launching -- opens its own async_session inside the background function to avoid DetachedInstanceError
- Screenplay enrichment in storyboard.py uses hasattr guard for backward compatibility with Scene instances loaded before the migration ran
- generate-scenes defaults to cinematic style, 16:9 aspect ratio, 8-second clip duration for screenplay-created scenes
- Force query param on generate-scenes enables idempotent re-generation by deleting existing screenplay-linked scenes first

## Deviations from Plan

### Auto-fixed Issues

**1. [Plan Structure] Combined Tasks 1 and 2 into single commit**
- **Found during:** Task 1
- **Issue:** Plan specified separate commits for CRUD endpoints (Task 1) and generation endpoints (Task 2), but both target the same file (screenplay.py). Creating the file with only CRUD endpoints then adding generation endpoints is functionally a single file creation.
- **Fix:** Created screenplay.py with all 11 endpoints in one commit. Both task verifications pass independently.
- **Impact:** No functional impact. Both sets of endpoints verified separately.

---

**Total deviations:** 1 (structural commit grouping)
**Impact on plan:** No functional impact. All endpoints exist and verify correctly.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Screenplay REST API complete, ready for Plan 03 (frontend UI)
- All 11 endpoints importable and registered in app.py
- Storyboard enrichment hooks in place for screenplay-linked scenes
- SceneListItem updated for SCRN-15 badge rendering in frontend

## Self-Check: PASSED

- FOUND: backend/vidpipe/api/screenplay.py (created)
- FOUND: backend/vidpipe/api/app.py (modified)
- FOUND: backend/vidpipe/api/routes.py (modified)
- FOUND: backend/vidpipe/pipeline/storyboard.py (modified)
- FOUND: commit 9f4110f (Tasks 1+2)
- FOUND: commit 38c8daa (Task 3)

---
*Phase: 18-screenplay-system*
*Completed: 2026-03-01*
