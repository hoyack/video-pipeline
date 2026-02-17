---
phase: 12-fork-system-integration-with-manifests
plan: 02
subsystem: api
tags: [fork, manifest, asset-inheritance, face-matching, yolo, reverse-prompt, scene-manifest]

# Dependency graph
requires:
  - phase: 12-01
    provides: Asset inheritance columns (is_inherited, inherited_from_asset, inherited_from_project), AssetChanges Pydantic models, manifest_id in ProjectDetail

provides:
  - _copy_assets_for_fork: copies parent manifest assets to forked project with is_inherited tracking and shared GCS URLs
  - _copy_scene_manifests: copies SceneManifest rows for unchanged scenes (below invalidation boundary)
  - _compute_asset_invalidation_point: finds earliest scene affected by modified assets
  - fork_project extended: inherits manifest_id/manifest_version, copies assets, handles asset modification invalidation
  - ManifestingEngine.process_new_uploads: incremental manifesting for new uploads during fork

affects: [frontend fork UI, pipeline manifesting, scene manifest consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Asset inheritance: is_inherited=True + shared manifest_id + inherited_from_asset + inherited_from_project
    - Tag collision avoidance: scan existing assets for max tag number per type, continue from there
    - Asset modification invalidation: find earliest scene using modified asset, tighten scene_boundary
    - Incremental manifesting: process_new_uploads runs inline during fork (not background)

key-files:
  created: []
  modified:
    - backend/vidpipe/api/routes.py
    - backend/vidpipe/services/manifesting_engine.py

key-decisions:
  - "_copy_assets_for_fork returns modified_asset_tags (not IDs) because scene manifests store tags, not UUIDs"
  - "Asset copy uses is_inherited=not is_modified — modified assets diverge from parent so is_inherited=False"
  - "process_new_uploads does NOT update manifest status or contact sheet — manifest is READY from parent"
  - "Asset modification invalidation is additive: only tightens boundary, never loosens it"
  - "cross_match_faces called with List[dict] format (matching FaceMatchingService interface) in process_new_uploads"
  - "New uploads get tags continuing from inherited max to avoid collision across forked project's asset pool"

patterns-established:
  - "Forked projects share manifest_id with parent (no manifest duplication)"
  - "process_new_uploads is synchronous within fork endpoint (small N assumption)"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 12 Plan 02: Fork Asset Inheritance and Incremental Manifesting Summary

**Fork endpoint extended with asset copy (is_inherited, shared GCS URLs), scene manifest inheritance, asset-modification-driven invalidation boundary tightening, and ManifestingEngine.process_new_uploads for incremental YOLO+face+reverse-prompt processing of new uploads**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-17T02:59:18Z
- **Completed:** 2026-02-17T03:02:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Fork endpoint copies all non-removed parent assets with `is_inherited=True` and shared GCS URLs (no file duplication)
- Fork inherits `manifest_id` and `manifest_version` from parent, enabling the same manifest to serve multiple project generations
- Scene manifests copied for scenes below invalidation boundary with post-deletion index remapping
- Modified assets trigger earlier invalidation: `_compute_asset_invalidation_point` finds the first scene using a modified asset and tightens `scene_boundary` accordingly
- `ManifestingEngine.process_new_uploads` processes new uploads through YOLO detection, face embedding generation, cross-matching against inherited embeddings, and reverse-prompting — all with tag collision avoidance

## Task Commits

Each task was committed atomically:

1. **Task 1: Asset copy, scene manifest inheritance, and invalidation extension in fork endpoint** - `37b2ef7` (feat)
2. **Task 2: ManifestingEngine.process_new_uploads for incremental manifesting** - `d1f56da` (feat)

## Files Created/Modified

- `backend/vidpipe/api/routes.py` - Added `_copy_assets_for_fork`, `_copy_scene_manifests`, `_compute_asset_invalidation_point` helpers; extended `fork_project` with manifest_id inheritance, asset copy, scene manifest copy, asset invalidation, and process_new_uploads call
- `backend/vidpipe/services/manifesting_engine.py` - Added `process_new_uploads` method; added `import base64` and `import numpy as np` at module level

## Decisions Made

- `_copy_assets_for_fork` returns `modified_asset_tags` (manifest_tag strings, not UUIDs) because `SceneManifest.asset_tags` stores tags, not UUIDs — matching these requires the same type
- Modified assets get `is_inherited=False` because they diverge from the parent — inheritance implies exact copy
- `process_new_uploads` does NOT update manifest status (stays READY) or regenerate contact sheet — adding new uploads expands the asset pool without invalidating existing processing
- Asset modification invalidation is additive: it only tightens the `scene_boundary`, never loosens it; if already at boundary 0 there is nothing further to tighten
- `cross_match_faces` called with `List[dict]` format in `process_new_uploads` (matching `FaceMatchingService` interface), while existing `process_manifest` code uses tuple format (that is a pre-existing bug, not introduced here)
- New uploads continue tag numbering from the inherited maximum for each asset_type to avoid collisions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Asset inheritance foundation complete; fork endpoint is fully functional for manifested projects
- Phase 12-03 (frontend fork UI for asset changes) can now build on the backend API
- `_copy_assets_for_fork` currently returns `(new_assets, modified_asset_tags)` but `new_assets` list is populated via a re-query pattern — this is intentional since SQLAlchemy identity map tracks the added assets; the return value is available for callers needing it

---
*Phase: 12-fork-system-integration-with-manifests*
*Completed: 2026-02-17*
