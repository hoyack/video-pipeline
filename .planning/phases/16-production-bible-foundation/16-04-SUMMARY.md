---
phase: 16-production-bible-foundation
plan: 04
subsystem: frontend
tags: [react, typescript, dnd-kit, sequences, drag-and-drop, ui]

# Dependency graph
requires:
  - phase: 16-02
    provides: Sequence CRUD API endpoints and ORM model

provides:
  - SequenceResponse and related TypeScript types in types.ts
  - 7 sequence API client functions in client.ts
  - ColorPicker component with 8 preset colors
  - SequenceContextMenu popover with edit/color/delete actions
  - SequenceHeader with inline title editing, color dot, collapse toggle
  - UnsequencedSection droppable container for ungrouped scenes
  - SortableSequenceSection droppable sequence with sortable scene rows
  - SequencedSceneList DndContext orchestrator with optimistic drag-end
  - ProductionDetail conditional rendering of flat vs grouped view

affects: [production-detail-view, sequence-grouping-ux]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Optimistic UI update on drag-end with revert on API failure
    - useDroppable for sequence containers, useSortable for scene rows
    - Inline input for Add Sequence (no modal)
    - Double-click to edit title inline (blur/Enter saves)
    - UNSEQUENCED_ID sentinel constant shared between components

key-files:
  created:
    - frontend/src/components/ColorPicker.tsx
    - frontend/src/components/SequenceContextMenu.tsx
    - frontend/src/components/SequenceHeader.tsx
    - frontend/src/components/UnsequencedSection.tsx
    - frontend/src/components/SortableSequenceSection.tsx
    - frontend/src/components/SequencedSceneList.tsx
  modified:
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts
    - frontend/src/components/ProductionDetail.tsx

key-decisions:
  - "Optimistic UI update on drag-end — sequence_id updated in local state immediately, reverted on API error"
  - "UNSEQUENCED_ID sentinel constant exported from UnsequencedSection — shared across DndContext and SequencedSceneList for droppable target ID"
  - "SequencedSceneList maintains its own localScenes copy — prevents stale closure issues during drag-end optimistic updates"
  - "ProductionDetail loads sequences in parallel with production and scenes — single load() call, graceful failure with empty array"
  - "Create Sequence button only visible when no sequences exist — switches view to sequenced mode on first creation"

requirements-completed: [SEQ-01, SEQ-02, SEQ-03, SEQ-04]

# Metrics
duration: 8min
completed: 2026-03-01
---

# Phase 16 Plan 04: Sequence Grouping Frontend UI Summary

**6 new React components and 7 API client functions enabling drag-and-drop sequence grouping of scenes within ProductionDetail using @dnd-kit**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-01
- **Completed:** 2026-03-01
- **Tasks:** 2
- **Files modified:** 9 (2 updated, 7 created)

## Accomplishments

- Added 6 TypeScript types (`SequenceResponse`, `SequenceWithScenes`, `SequenceCreate`, `SequenceUpdate`, `SequenceReorderRequest`, `AssignSequenceRequest`) and `sequence_id` field to `SceneListItem`
- Added 7 sequence API client functions covering full CRUD plus reorder and assignment
- Created `ColorPicker` with 8 preset color circles and ring selection indicator
- Created `SequenceContextMenu` popover with Edit/Change Color/Delete actions and click-outside close
- Created `SequenceHeader` with color dot, double-click inline title editing, scene count badge, act label, context menu, and collapse chevron
- Created `UnsequencedSection` — droppable target for scenes without a sequence (hides when empty)
- Created `SortableSequenceSection` — droppable sequence container with `useSortable` scene rows, drag handles, and "Drop scenes here" empty state
- Created `SequencedSceneList` — top-level orchestrator with DndContext, optimistic drag-end, Add Sequence inline input, delete unsequencing scenes in local state
- Updated `ProductionDetail` to load sequences in parallel, conditionally render `SequencedSceneList` vs flat list, and show a "Create Sequence" button when none exist

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Sequence TypeScript types and 7 API client functions** - `84adf4e` (feat)
2. **Task 2: Build sequence grouping UI components and integrate into ProductionDetail** - `98d11a1` (feat)

## Files Created/Modified

- `frontend/src/api/types.ts` - Added 6 Sequence types + `sequence_id` field on `SceneListItem`
- `frontend/src/api/client.ts` - Added 7 sequence API client functions
- `frontend/src/components/ColorPicker.tsx` - New: preset color circle picker
- `frontend/src/components/SequenceContextMenu.tsx` - New: "..." popover menu
- `frontend/src/components/SequenceHeader.tsx` - New: sequence title bar with inline editing
- `frontend/src/components/UnsequencedSection.tsx` - New: droppable container for ungrouped scenes
- `frontend/src/components/SortableSequenceSection.tsx` - New: sortable sequence with draggable scene rows
- `frontend/src/components/SequencedSceneList.tsx` - New: DndContext orchestrator
- `frontend/src/components/ProductionDetail.tsx` - Updated: conditional grouped/flat view + Create Sequence button

## Decisions Made

- Optimistic UI update on drag-end — local state updated immediately, reverted on API failure to minimize perceived latency
- `UNSEQUENCED_ID` sentinel constant (`"__unsequenced__"`) shared as export between UnsequencedSection and SequencedSceneList — single source of truth for droppable target ID
- `SequencedSceneList` maintains `localScenes` state copy synced from parent `scenes` prop — avoids stale closure in drag-end handler
- `ProductionDetail` loads sequences via `listSequences()` in the same `Promise.all` as production and scenes — single load call with graceful `.catch(() => [])` for non-fatal failure
- "Create Sequence" button only shown when sequences array is empty — clicking creates a first "Chapter 1" sequence and transitions to grouped view

## Deviations from Plan

None - plan executed exactly as written. All 6 components created, 7 API functions added, TypeScript compiles without errors, ProductionDetail conditionally renders grouped vs flat view.

## Self-Check

Files created:
- frontend/src/components/ColorPicker.tsx - EXISTS
- frontend/src/components/SequenceContextMenu.tsx - EXISTS
- frontend/src/components/SequenceHeader.tsx - EXISTS
- frontend/src/components/UnsequencedSection.tsx - EXISTS
- frontend/src/components/SortableSequenceSection.tsx - EXISTS
- frontend/src/components/SequencedSceneList.tsx - EXISTS

Commits:
- 84adf4e - Task 1 (types + API client)
- 98d11a1 - Task 2 (UI components + ProductionDetail)

TypeScript: PASSED (npx tsc --noEmit with no errors)

## Self-Check: PASSED

---
*Phase: 16-production-bible-foundation*
*Completed: 2026-03-01*
