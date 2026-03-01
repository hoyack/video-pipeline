---
phase: 16-production-bible-foundation
plan: 03
subsystem: frontend
tags: [typescript, react, routing, rename, production-bible]

# Dependency graph
requires:
  - 16-01 (backend Production Bible rename — /api/production-bibles/* endpoints)
provides:
  - ProductionBibleLibrary component (replaces ManifestLibrary)
  - ProductionBibleCreator component with department tabs (replaces ManifestCreator)
  - ProductionBibleCard component (replaces ManifestCard)
  - ProductionBibleSelector component (replaces ManifestSelector)
  - Routes at /production-bibles/* (replacing /manifests/*)
  - Nav label "Production Bibles" pointing to /production-bibles
  - API client functions using /api/production-bibles/ URLs
  - TypeScript types with production_bible_id fields
  - Department tabs (Casting, Art Department, Sound) in ProductionBibleCreator
affects: [all frontend components that used Manifest terminology]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Backward-compat type aliases: export type ManifestListItem = ProductionBibleListItem"
    - "Backward-compat function aliases: export const listManifests = listProductionBibles"
    - "Department tab filter pattern: filter assets by asset_type per tab config"
    - "Old Manifest*.tsx components retained (not deleted) — new Production Bible*.tsx are canonical"

key-files:
  created:
    - frontend/src/components/ProductionBibleCard.tsx
    - frontend/src/components/ProductionBibleLibrary.tsx
    - frontend/src/components/ProductionBibleSelector.tsx
    - frontend/src/components/ProductionBibleCreator.tsx
  modified:
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts
    - frontend/src/App.tsx
    - frontend/src/components/Layout.tsx
    - frontend/src/components/SceneDetail.tsx
    - frontend/src/components/EditModeOverlay.tsx
    - frontend/src/components/GenerateForm.tsx
    - frontend/src/components/EditForkPanel.tsx

key-decisions:
  - "Backward-compat aliases added for types and functions — old Manifest*.tsx files retained for gradual migration"
  - "EditForkPanel, EditModeOverlay, GenerateForm updated to use production_bible_id (auto-fixed as Rule 2 — missing correctness)"
  - "Department tabs defined with DEPARTMENT_TABS config array: Casting=CHARACTER, Art Dept=ENV/PROP/OBJECT/VEHICLE/STYLE, Sound=placeholder"
  - "Sound tab shows 'Audio direction coming soon' placeholder — no assets yet"

patterns-established:
  - "DEPARTMENT_TABS array config pattern for filtering asset lists by type"
  - "Backward-compat export alias pattern for smooth rename transitions"

requirements-completed: [PBIB-04, PBIB-05, PBIB-06]

# Metrics
duration: 25min
completed: 2026-02-28
---

# Phase 16 Plan 03: Production Bible Frontend Rename Summary

**Frontend rename from Manifest to Production Bible: TypeScript types, API client, four new components (Card, Library, Selector, Creator with department tabs), routes at /production-bibles/*, nav label update, and Production Bible terminology throughout.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-02-28
- **Tasks:** 2
- **Files modified:** 8
- **Files created:** 4

## Accomplishments

- Renamed TypeScript types: `ManifestListItem` -> `ProductionBibleListItem`, `ManifestDetail` -> `ProductionBibleDetail`, `CreateManifestRequest` -> `CreateProductionBibleRequest`, `UpdateManifestRequest` -> `UpdateProductionBibleRequest`
- Updated field names: `manifest_id` -> `production_bible_id`, `parent_manifest_id` -> `parent_production_bible_id` in all relevant types (`SceneDetail`, `GenerateRequest`, `EditSceneRequest`, `CreateDraftSceneRequest`)
- Renamed all API client functions: `listProductionBibles`, `createProductionBible`, `getProductionBibleDetail`, `updateProductionBible`, `deleteProductionBible`, `duplicateProductionBible`, `processProductionBible`, `uploadVideoForProductionBible`, `fetchProductionBibleAssets`, `importSceneToProductionBible`
- All API URLs updated to `/api/production-bibles/*`
- Created four new canonical components: `ProductionBibleCard`, `ProductionBibleLibrary`, `ProductionBibleSelector`, `ProductionBibleCreator`
- `ProductionBibleCreator` includes department tabs in Stage 3: Casting (CHARACTER assets), Art Department (ENVIRONMENT/PROP/OBJECT/VEHICLE/STYLE), Sound (placeholder with "Audio direction coming soon")
- Updated routes in `App.tsx`: `/manifests/*` -> `/production-bibles/*`
- Updated nav item in `Layout.tsx`: "Manifests" -> "Production Bibles"
- Updated `EditModeOverlay`, `GenerateForm`, `EditForkPanel`, `SceneDetail` to use `production_bible_id` and `ProductionBibleSelector`
- Added backward-compat type aliases and function aliases for gradual migration

## Task Commits

1. **Task 1: Rename TypeScript types, API client functions, and create new component files** - `6c74c9e`
2. **Task 2: Update routing, navigation, and component references** - `9faaa52`

## Files Created/Modified

**Created:**
- `frontend/src/components/ProductionBibleCard.tsx` - Card component with production_bible_id field, "Copy production bible ID" tooltip
- `frontend/src/components/ProductionBibleLibrary.tsx` - Library with "Production Bible Library" header, "+ New Production Bible" button, correct filter/sort
- `frontend/src/components/ProductionBibleSelector.tsx` - Selector with "Select a production bible to attach reference images" prompt
- `frontend/src/components/ProductionBibleCreator.tsx` - Full creator with department tabs (Casting/Art Dept/Sound) in Stage 3

**Modified:**
- `frontend/src/api/types.ts` - Renamed types + backward-compat aliases
- `frontend/src/api/client.ts` - Renamed functions + updated URLs + backward-compat aliases
- `frontend/src/App.tsx` - Routes updated to /production-bibles/*
- `frontend/src/components/Layout.tsx` - Nav item "Production Bibles" at /production-bibles
- `frontend/src/components/SceneDetail.tsx` - Uses detail.production_bible_id
- `frontend/src/components/EditModeOverlay.tsx` - Uses ProductionBibleSelector + production_bible_id
- `frontend/src/components/GenerateForm.tsx` - Uses ProductionBibleSelector + production_bible_id
- `frontend/src/components/EditForkPanel.tsx` - Uses detail.production_bible_id

## Decisions Made

- Backward-compat aliases added for all renamed types and functions to allow gradual migration of remaining code that uses the old `ManifestLibrary.tsx`, `ManifestCard.tsx`, etc.
- Old `Manifest*.tsx` component files NOT deleted — new `ProductionBible*.tsx` are canonical; deletion can happen in a follow-up cleanup.
- `EditForkPanel`, `EditModeOverlay`, `GenerateForm` were found to use `detail.manifest_id` — updated inline as part of Task 2 (Rule 2: missing correctness fix).
- Department tab configuration is defined as a const array `DEPARTMENT_TABS` for easy extension.
- Sound tab is a placeholder with "Audio direction coming soon" message.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Correctness] Updated EditModeOverlay.tsx manifest_id references**
- **Found during:** Task 2 (scanning for all manifest_id usages from SceneDetail type)
- **Issue:** `EditModeOverlay` initialized state from `detail.manifest_id` and sent `req.manifest_id`, both of which would be undefined since the type field was renamed
- **Fix:** Updated to `detail.production_bible_id` and `req.production_bible_id`
- **Files modified:** `frontend/src/components/EditModeOverlay.tsx`
- **Committed in:** `9faaa52`

**2. [Rule 2 - Missing Correctness] Updated EditForkPanel.tsx manifest_id references**
- **Found during:** Task 2 (scanning for all manifest_id usages from SceneDetail type)
- **Issue:** `EditForkPanel` checked `detail.manifest_id` in 3 places; would always be undefined after the type rename
- **Fix:** Updated all to `detail.production_bible_id`
- **Files modified:** `frontend/src/components/EditForkPanel.tsx`
- **Committed in:** `9faaa52`

**3. [Rule 2 - Missing Correctness] Updated GenerateForm.tsx manifest_id in GenerateRequest**
- **Found during:** Task 2 (scanning for all manifest_id usages)
- **Issue:** `GenerateForm` sent `manifest_id` in the generate request body; backend now expects `production_bible_id`
- **Fix:** Updated field name and switched ManifestSelector to ProductionBibleSelector
- **Files modified:** `frontend/src/components/GenerateForm.tsx`
- **Committed in:** `9faaa52`

---

**Total deviations:** 3 auto-fixed (3x Rule 2 - Missing Correctness)
**Impact on plan:** All critical for correct operation. No scope creep.

## Self-Check: PASSED

- ProductionBibleLibrary.tsx: FOUND
- ProductionBibleCreator.tsx: FOUND
- ProductionBibleCard.tsx: FOUND
- ProductionBibleSelector.tsx: FOUND
- TypeScript compilation: No errors
- Routes /production-bibles/* in App.tsx: CONFIRMED
- Nav "Production Bibles" in Layout.tsx: CONFIRMED
- API URLs /api/production-bibles/ in client.ts: CONFIRMED
- Department tabs in ProductionBibleCreator.tsx: CONFIRMED
