---
status: diagnosed
phase: 16-production-bible-foundation
source: 16-01-SUMMARY.md, 16-02-SUMMARY.md, 16-03-SUMMARY.md, 16-04-SUMMARY.md
started: 2026-03-01T02:00:00Z
updated: 2026-03-01T03:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Production Bibles Navigation Link
expected: The sidebar/nav shows "Production Bibles" as a menu item. Clicking it navigates to /production-bibles and loads the library page.
result: pass

### 2. Production Bible Library Page
expected: The /production-bibles page shows a list of existing production bibles (if any) with a "+ New Production Bible" button. Each card shows production bible name and metadata.
result: pass

### 3. Create Production Bible with Department Tabs
expected: Clicking "+ New Production Bible" opens the creator wizard. Stage 3 (asset review) shows three department tabs: Casting, Art Department, and Sound. Casting shows CHARACTER assets, Art Department shows ENVIRONMENT/PROP/OBJECT/VEHICLE/STYLE assets, Sound shows "Audio direction coming soon" placeholder.
result: pass

### 4. Production Bible Selector in Generate Form
expected: When creating a new scene via the Generate Form, the production bible selector dropdown shows available production bibles (not "manifests"). Selecting one attaches it to the generation request.
result: issue
reported: "It says 'Asset Manifest' on the container label in the Scene dropdown for Production Bible"
severity: minor

### 5. Legacy /manifests/ URL Redirect
expected: Navigating to /api/manifests/ (or any old manifest API path) returns a 301 redirect to the corresponding /api/production-bibles/ path.
result: pass

### 6. Create First Sequence
expected: In a production detail view, when no sequences exist, a "Create Sequence" button is visible. Clicking it creates a "Chapter 1" sequence and the view switches from flat scene list to grouped sequence view.
result: pass

### 7. Drag Scene Into Sequence
expected: In the grouped view, scenes can be dragged from the "Unsequenced" section into a sequence container. The scene moves to the target sequence with optimistic UI update (no page reload needed).
result: pass

### 8. Sequence Header Editing
expected: Double-clicking a sequence title enables inline editing (type new name, press Enter or blur to save). The sequence header shows a color dot, scene count badge, and a collapse chevron to hide/show scenes.
result: pass

### 9. Sequence Context Menu
expected: Clicking the "..." button on a sequence header opens a context menu with Edit, Change Color, and Delete options. Change Color shows 8 preset color circles.
result: pass

### 10. Delete Sequence
expected: Deleting a sequence via context menu removes the sequence but does NOT delete the scenes — they return to the Unsequenced section.
result: pass

## Summary

total: 10
passed: 9
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Production bible selector dropdown shows 'Production Bible' label (not 'manifests' or 'Asset Manifest')"
  status: failed
  reason: "User reported: It says 'Asset Manifest' on the container label in the Scene dropdown for Production Bible"
  severity: minor
  test: 4
  root_cause: "EditModeOverlay.tsx line 1077 still says 'Asset Manifest' instead of 'Production Bible'. Comment on line 1069 also uses old terminology."
  artifacts:
    - path: "frontend/src/components/EditModeOverlay.tsx"
      issue: "Label text 'Asset Manifest' not renamed to 'Production Bible'"
  missing:
    - "Change 'Asset Manifest' to 'Production Bible' on line 1077"
    - "Update comment on line 1069 from 'Asset Manifest' to 'Production Bible'"
  debug_session: ""
