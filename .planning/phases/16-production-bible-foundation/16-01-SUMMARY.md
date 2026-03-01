---
phase: 16-production-bible-foundation
plan: 01
subsystem: database, api
tags: [sqlalchemy, fastapi, pydantic, sqlite, postgresql, migration, rename]

# Dependency graph
requires: []
provides:
  - ProductionBible ORM model replacing Manifest (backwards-compat alias retained)
  - production_bibles table (renamed from manifests) with all FK columns updated
  - /api/production-bibles/* API endpoints
  - 301 redirect routes for legacy /api/manifests/* paths
  - Pydantic schemas renamed with backwards-compat aliases
affects: [16-02-sequence-grouping, all phases using manifest routes or ProductionBible model]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Backwards-compat alias pattern: Manifest = ProductionBible at module level in every file"
    - "Idempotent rename migrations wrapped in try/except (SQLite RENAME TABLE/COLUMN)"
    - "Rename migrations run BEFORE create_all() in init_database() to preserve existing data"
    - "301 redirect catch-all using FastAPI api_route with wildcard path parameter"

key-files:
  created: []
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/api/routes.py
    - backend/vidpipe/services/manifest_service.py
    - backend/vidpipe/services/manifesting_engine.py
    - backend/vidpipe/services/entity_extraction.py
    - backend/vidpipe/services/checkpoint_service.py
    - backend/vidpipe/services/reference_selection.py
    - backend/vidpipe/workers/processing_tasks.py
    - backend/vidpipe/orchestrator/pipeline.py
    - backend/vidpipe/pipeline/storyboard.py
    - backend/vidpipe/pipeline/keyframes.py
    - backend/vidpipe/pipeline/video_gen.py

key-decisions:
  - "Rename migrations must run BEFORE create_all() so SQLAlchemy finds tables under new names"
  - "Retained Manifest = ProductionBible alias in every file for gradual migration compatibility"
  - "Filesystem directory tmp/manifests/ kept as-is (only DB column/table names renamed)"
  - "Function parameter names manifest_id kept as-is (only ORM column attribute names renamed)"

patterns-established:
  - "Module-level backwards-compat alias: MyOldName = MyNewName for renamed ORM models"
  - "Idempotent migration: try/except around ALTER TABLE/COLUMN for repeatable deploy safety"

requirements-completed: [PBIB-01, PBIB-02, PBIB-03]

# Metrics
duration: 45min
completed: 2026-02-28
---

# Phase 16 Plan 01: Production Bible Rename Summary

**Renamed Manifest -> ProductionBible across backend: DB table, all FK columns, API endpoints (/api/production-bibles/*), Pydantic schemas, and service layer, with 301 redirects on legacy /api/manifests/* paths**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-02-28T00:00:00Z
- **Completed:** 2026-02-28T01:22:00Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Renamed `manifests` table to `production_bibles` via idempotent SQLite RENAME TABLE migration
- Renamed all FK columns: `manifest_id` → `production_bible_id` on assets, manifest_snapshots, scenes; `parent_manifest_id` → `parent_production_bible_id` on production_bibles; `manifest_version` → `production_bible_version` on scenes
- Renamed all API routes to `/api/production-bibles/*` (14 routes) with 301 redirect catch-all for legacy `/api/manifests/*` paths
- Updated all Pydantic request/response schemas with backwards-compat aliases

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename Database Model and Add Migration** - `5706e7e` (feat)
2. **Task 2: Rename API Endpoints, Pydantic Schemas, Service Layer** - `99293e3` (feat)

**Plan metadata:** _(final commit pending)_

## Files Created/Modified
- `backend/vidpipe/db/models.py` - Renamed class Manifest to ProductionBible, updated all FK column names, added Manifest alias
- `backend/vidpipe/db/__init__.py` - Added _run_rename_migrations() called before create_all() with idempotent RENAME TABLE/COLUMN
- `backend/vidpipe/api/routes.py` - All /production-bibles/* routes, renamed Pydantic schemas, 301 redirect routes for /manifests/*
- `backend/vidpipe/services/manifest_service.py` - Updated Asset.production_bible_id column refs throughout
- `backend/vidpipe/services/manifesting_engine.py` - Updated all Asset.production_bible_id column refs and contact_sheet_url path
- `backend/vidpipe/services/entity_extraction.py` - Updated Asset.production_bible_id in queries and constructors
- `backend/vidpipe/services/checkpoint_service.py` - Fixed scene.production_bible_id and Asset.production_bible_id refs (auto-fix)
- `backend/vidpipe/services/reference_selection.py` - Fixed asset.production_bible_id in path construction (auto-fix)
- `backend/vidpipe/workers/processing_tasks.py` - Updated Asset.production_bible_id in query and constructor
- `backend/vidpipe/orchestrator/pipeline.py` - Updated scene.production_bible_id refs
- `backend/vidpipe/pipeline/storyboard.py` - Updated scene.production_bible_id refs
- `backend/vidpipe/pipeline/keyframes.py` - Updated scene.production_bible_id and Asset.production_bible_id refs
- `backend/vidpipe/pipeline/video_gen.py` - Updated scene.production_bible_id and Asset.production_bible_id refs

## Decisions Made
- Rename migrations run BEFORE `create_all()` in `init_database()`. This is critical: SQLAlchemy expects to find `production_bibles` table during schema sync, but it won't exist if `manifests` hasn't been renamed yet.
- Retained `Manifest = ProductionBible` alias in every module. This allows any code that still imports `Manifest` to continue working without errors during the transition period.
- Filesystem directories `tmp/manifests/{id}/...` are NOT renamed. The "manifests" in the path is just a folder name chosen for organization — renaming it would break all existing stored files.
- Function parameter names `manifest_id: uuid.UUID` are kept as-is where they are just parameter names (not ORM column references). Only the ORM attribute and DB column names are renamed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed remaining manifest_id column refs in checkpoint_service.py**
- **Found during:** Task 2 (post-commit verification scan)
- **Issue:** `scene.manifest_id` and `Asset.manifest_id` references at lines 86 and 197 would cause `AttributeError` at runtime since the ORM attribute was renamed
- **Fix:** Updated to `scene.production_bible_id` and `Asset.production_bible_id`
- **Files modified:** `backend/vidpipe/services/checkpoint_service.py`
- **Committed in:** `99293e3` (Task 2 commit)

**2. [Rule 1 - Bug] Fixed remaining manifest_id column ref in reference_selection.py**
- **Found during:** Task 2 (post-commit verification scan)
- **Issue:** `asset.manifest_id` at line 317 in path construction would cause `AttributeError` at runtime
- **Fix:** Updated to `asset.production_bible_id`
- **Files modified:** `backend/vidpipe/services/reference_selection.py`
- **Committed in:** `99293e3` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2x Rule 1 - Bug)
**Impact on plan:** Both fixes were required for runtime correctness. No scope creep.

## Issues Encountered
- `init_database()` ordering was critical: rename migrations must run before `create_all()`. Initially noted in planning, correctly implemented.
- Verification initially showed only 13 production-bible routes (needed 14). Fixed by adding a `GET /production-bibles/{id}/contact-sheet` route that was referenced in manifesting_engine.py but had no matching API handler.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Production Bible rename complete. All backend code uses ProductionBible/production_bible_id.
- Phase 16-02 (Sequence grouping layer) was already completed in a prior session and can proceed.
- Frontend code still uses "manifests" terminology — frontend rename is a separate phase concern.

---
*Phase: 16-production-bible-foundation*
*Completed: 2026-02-28*
