---
phase: 12-fork-system-integration-with-manifests
plan: 01
subsystem: database, api
tags: [sqlalchemy, pydantic, sqlite, migration, fork, assets, manifest]

# Dependency graph
requires:
  - phase: 11-multi-candidate-quality-mode
    provides: GenerationCandidate model and quality_mode on Project
  - phase: 04-manifest-system-foundation
    provides: Asset and Manifest ORM models with manifest_id FK
  - phase: 03-orchestration-interfaces
    provides: ForkRequest schema and fork_project endpoint

provides:
  - Asset model with is_inherited, inherited_from_asset, inherited_from_project columns
  - SQL migration file (migrate_phase12.sql) for applying inheritance columns to existing DBs
  - AssetChanges, ModifiedAsset, NewUpload Pydantic models for fork asset editing
  - ForkRequest.asset_changes optional field for fork-based asset overrides
  - ProjectDetail.manifest_id for EditForkPanel manifest-aware fork UI
  - Fork endpoint 422 validation: asset_changes requires source project to have a manifest

affects:
  - 12-02 (fork endpoint asset inheritance logic — depends on these schema foundations)
  - frontend EditForkPanel (reads manifest_id from ProjectDetail to fetch manifest assets)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mapped[Optional[uuid.UUID]] with ForeignKey for self-referential and cross-table inheritance tracking"
    - "Pydantic nested models (AssetChanges wraps ModifiedAsset and NewUpload) for structured fork payloads"
    - "Early validation guard in fork endpoint before heavy compute (_compute_invalidation)"

key-files:
  created:
    - backend/migrate_phase12.sql
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/api/routes.py

key-decisions:
  - "Asset inheritance columns use TEXT (UUID stored as TEXT in SQLite) in SQL migration but Mapped[Optional[uuid.UUID]] with ForeignKey in ORM — consistent with existing source_asset_id pattern"
  - "AssetChanges placed before ForkRequest (not after) — follows logical declaration dependency order"
  - "manifest_id in ProjectDetail is Optional[str] (not UUID) for JSON API consistency with other ID fields"
  - "422 validation for asset_changes without manifest placed before _compute_invalidation to fail fast on invalid state"

patterns-established:
  - "Phase N inheritance tracking: add Phase N comment block after last field of extended class"
  - "Fork validation guards: input checks ordered from cheapest to most expensive before background task dispatch"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 12 Plan 01: Fork System Integration with Manifests — Foundation Summary

**Asset inheritance schema (3 new ORM columns + SQL migration) and extended fork API payload with AssetChanges, ModifiedAsset, NewUpload Pydantic models plus manifest_id in ProjectDetail**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-17T02:55:24Z
- **Completed:** 2026-02-17T02:57:30Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created `backend/migrate_phase12.sql` with 3 ALTER TABLE statements adding is_inherited, inherited_from_asset, inherited_from_project to assets table
- Extended Asset ORM model with Phase 12 fork inheritance tracking fields using existing ForeignKey pattern
- Added three new Pydantic models (ModifiedAsset, NewUpload, AssetChanges) and extended ForkRequest with asset_changes
- Added manifest_id to ProjectDetail response and wired it from project.manifest_id in get_project_detail
- Added 422 guard in fork_project: asset_changes requires source project to have manifest_id

## Task Commits

Each task was committed atomically:

1. **Task 1: DB migration and Asset ORM inheritance fields** - `d1267bf` (feat)
2. **Task 2: ForkRequest schema extension and ProjectDetail manifest_id** - `5ea3a4b` (feat)

## Files Created/Modified
- `backend/migrate_phase12.sql` - SQL migration adding is_inherited, inherited_from_asset, inherited_from_project columns to assets table
- `backend/vidpipe/db/models.py` - Asset class extended with 3 inheritance tracking fields (Phase 12 block)
- `backend/vidpipe/api/routes.py` - ModifiedAsset/NewUpload/AssetChanges models added; ForkRequest.asset_changes added; ProjectDetail.manifest_id added; get_project_detail return updated; fork_project 422 guard added

## Decisions Made
- Asset inheritance columns use `TEXT` type in SQL migration (SQLite stores UUIDs as TEXT) but `Mapped[Optional[uuid.UUID]]` with ForeignKey in the ORM — consistent with `source_asset_id` pattern already established in Phase 9
- `manifest_id` in `ProjectDetail` is `Optional[str]` (not `Optional[uuid.UUID]`) for JSON API consistency with all other ID fields in the response schema
- 422 validation for `asset_changes` without manifest is placed directly after deleted_scenes validation, before `_compute_invalidation` — fail fast on invalid input before expensive operations

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Importing routes.py in a test subprocess fails because db/__init__.py triggers Settings validation requiring config.yaml. Resolved by using AST-based verification for model field presence checks, and a standalone Pydantic test script for schema parsing behavior — both sufficient for plan's verification requirements.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Foundation layer complete: Asset inheritance columns in DB and ORM, AssetChanges schema in ForkRequest, manifest_id in ProjectDetail
- Plan 02 can now implement the actual asset inheritance logic in the fork endpoint (copy/modify/add assets using these new fields and the AssetChanges payload)
- SQL migration file ready to be applied to production databases with `sqlite3 vidpipe.db < migrate_phase12.sql`

---
*Phase: 12-fork-system-integration-with-manifests*
*Completed: 2026-02-17*

## Self-Check: PASSED

- FOUND: backend/migrate_phase12.sql
- FOUND: backend/vidpipe/db/models.py
- FOUND: backend/vidpipe/api/routes.py
- FOUND: .planning/phases/12-fork-system-integration-with-manifests/12-01-SUMMARY.md
- FOUND: commit d1267bf (Task 1)
- FOUND: commit 5ea3a4b (Task 2)
