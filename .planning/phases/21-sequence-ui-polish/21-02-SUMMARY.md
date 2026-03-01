---
phase: 21-sequence-ui-polish
plan: 02
subsystem: ui
tags: [react, typescript, dnd-kit, drag-and-drop, sequence]

# Dependency graph
requires:
  - phase: 21-sequence-ui-polish
    provides: "Bulk scene reorder endpoint, reorderScenesInSequence client function, scene_order field"
  - phase: 16-production-bible-foundation
    provides: "Sequence model, sequence CRUD endpoints, SequencedSceneList component"
provides:
  - "Sequence drag-and-drop reorder via SortableContext"
  - "Within-sequence scene drag-and-drop reorder"
  - "Act field setter submenu in context menu"
  - "Duration badge display in sequence header"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: ["Type-discriminated DnD routing via active.data.current.type in single DndContext"]

key-files:
  created: []
  modified:
    - frontend/src/components/SequencedSceneList.tsx
    - frontend/src/components/SortableSequenceSection.tsx
    - frontend/src/components/SequenceHeader.tsx
    - frontend/src/components/SequenceContextMenu.tsx

key-decisions:
  - "Single DndContext with type-discriminated handleDragEnd avoids nested DndContext conflicts"
  - "Sequence drag handle placed outside SequenceHeader to keep header layout clean"
  - "Act picker uses same submenu pattern as color picker for UI consistency"

patterns-established:
  - "Type-discriminated DnD: useSortable data.type routes drag events in shared DndContext"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-03-01
---

# Phase 21 Plan 02: Sequence UI Feature Wiring Summary

**Sequence and scene drag-and-drop reorder, act field setter submenu, and duration badge display in SequenceHeader**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-01T23:11:11Z
- **Completed:** 2026-03-01T23:13:53Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Wired sequence-level drag reorder with SortableContext and optimistic API calls to reorderSequences
- Wired within-sequence scene reorder with nested SortableContext and reorderScenesInSequence API calls
- Added act picker submenu (ACT_1/ACT_2/ACT_3/None) to SequenceContextMenu with highlight for selected act
- Added formatDuration helper and conditional duration badge to SequenceHeader

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire sequence DnD reorder and within-sequence scene reorder** - `fca52a8` (feat)
2. **Task 2: Add act field setter and duration display** - `a603871` (feat)

## Files Created/Modified
- `frontend/src/components/SequencedSceneList.tsx` - Added SortableContext for sequences, refactored handleDragEnd with type discrimination, imported reorderSequences/reorderScenesInSequence
- `frontend/src/components/SortableSequenceSection.tsx` - Made section sortable for sequence reorder, wrapped scenes in SortableContext, tagged DraggableSceneRow with type/sequenceId data
- `frontend/src/components/SequenceHeader.tsx` - Added formatDuration helper, duration badge, passed act/onActChange to SequenceContextMenu
- `frontend/src/components/SequenceContextMenu.tsx` - Added act/onActChange props, showActPicker state, ACT_OPTIONS array, act picker submenu

## Decisions Made
- Used single DndContext with type-discriminated handleDragEnd (routing on active.data.current.type) to avoid nested DndContext conflicts per Research Pitfall 1
- Placed sequence drag handle outside SequenceHeader as a sibling element to keep header layout clean and avoid prop drilling drag listeners
- Act picker follows same submenu toggle pattern as existing color picker for UI consistency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 4 tech debt items from the v1.0 audit for Issue #24 are now fully wired in the frontend
- Phase 21 (Sequence UI Polish) is complete

## Self-Check: PASSED

All files exist, all commits verified.

---
*Phase: 21-sequence-ui-polish*
*Completed: 2026-03-01*
