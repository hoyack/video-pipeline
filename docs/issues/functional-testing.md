# Frontend Functional Testing Findings

This document captures issues observed during manual functional testing via Playwright MCP on:

- URL: `http://localhost:5174`
- Context used: `docs/frontend-docs.md`
- Scope exercised: Projects list/detail, Generate form, Dashboard, Manifests library/creator, Settings, mobile viewport behavior
- Constraint: Findings only (no fixes applied)

## Test Summary

- Core pages load and are navigable.
- Most API-backed views returned `200` during stable runs.
- Several functional/UX inconsistencies were found.
- Intermittent dev-server runtime errors were also observed during one part of the session.

## Findings

## 1. Mobile Header Causes Horizontal Page Overflow

- Severity: Medium
- Area: Global layout/navigation (`Layout` behavior)
- Repro steps:
1. Open `http://localhost:5174`.
2. Set viewport to mobile width (tested at `375x812`).
3. Observe page width / horizontal scrolling.

- Observed:
  - Document has horizontal overflow on mobile.
  - Measured in browser runtime:
    - `clientWidth = 375`
    - `scrollWidth = 451`
    - `hasHorizontalOverflow = true`
  - Offending element detection showed the nav row exceeds viewport:
    - `nav.flex.gap-1` with text `ProjectsManifestsDashboardSettings`
    - Right edge exceeded viewport by ~`76px`.

- Expected:
  - No horizontal overflow at common mobile widths.
  - Navigation should wrap, collapse, scroll internally, or otherwise stay within viewport.

## 2. Generate Form Manifest Preview Conflicts with Displayed Asset Count

- Severity: Medium
- Area: Generate form -> selected manifest card
- Repro steps:
1. Go to `Projects` -> `+ New`.
2. In `Asset Manifest`, click `Add`.
3. Select manifest `Hoyack`.

- Observed:
  - Selected card header reports: `3 assets | CHARACTERS | v1`.
  - Thumbnail strip shows 5 thumbnail chips plus `+1` overflow indicator (implies >5 items).
  - Thumbnails also included repeated labels/tags.

- Expected:
  - Thumbnail summary should be consistent with reported asset count, or explicitly indicate why counts differ.

## 3. Manifest Category Values Are Inconsistent Between Library Filter and Creator

- Severity: Medium
- Area: Manifest filtering and creation
- Repro steps:
1. Open `Manifests` library.
2. Note filter pill values (includes `Environment`).
3. Click `+ New Manifest`.
4. Inspect creator category dropdown values (includes `ENVIRONMENTS`).

- Observed:
  - Library filter uses singular `Environment` (mapped category value `ENVIRONMENT`).
  - Creator uses plural `ENVIRONMENTS`.
  - Category taxonomies are not aligned across these two flows.

- Expected:
  - A single canonical category enum across create/edit/filter paths.

## 4. Quality Mode Is Available in Storyboard-Only Runs and Shows Misleading Cost Copy

- Severity: Low
- Area: Generate form state logic
- Repro steps:
1. Open `Projects` -> `+ New`.
2. Set `Generate Through` to `Storyboard`.
3. Toggle `Quality Mode` on.

- Observed:
  - UI still allows `Quality Mode` and candidate count controls.
  - Cost text reads like: `~2x video generation cost`, even though storyboard-only runs do not perform video generation in that run.

- Expected:
  - Either disable/hide quality mode for storyboard-only runs, or adjust wording/cost semantics to avoid referencing video-generation impact in this mode.

## 5. Intermittent Dev Runtime/HMR Errors Observed (Session-Transient)

- Severity: Medium (dev workflow reliability)
- Area: Vite HMR / module reload path
- Repro context:
  - During one test segment (while app was open and reloading modules), multiple runtime/HMR errors appeared in console.

- Observed errors included:
  - `SyntaxError: ... /src/api/client.ts ... does not provide an export named 'createDraftProject'`
  - `Failed to reload /src/components/VideoGenEditor.tsx`
  - `Failed to reload /src/components/GenerateThroughSlider.tsx`
  - `Failed to reload /src/components/ProjectConfigBar.tsx`
  - `404` fetches for missing module paths above during hot-reload attempts

- Notes:
  - These errors were not present after a clean reload later in the same session.
  - Still worth documenting because they indicate instability in active dev/hot-reload conditions.

## 6. Settings Page Emits Browser Password-Field Structural Warnings

- Severity: Low
- Area: Settings form markup
- Repro steps:
1. Navigate to `Settings`.
2. Observe browser console verbose output.

- Observed:
  - Browser warning: password field not contained in a form element (repeated for multiple password inputs).

- Expected:
  - Credential inputs wrapped in form semantics or intentionally structured to avoid repeated DOM warnings.

## Notes

- No code changes were made to address these issues.
- This file records findings only, per request.
