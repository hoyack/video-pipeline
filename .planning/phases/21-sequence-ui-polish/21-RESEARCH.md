# Phase 21: Sequence UI Polish - Research

**Researched:** 2026-03-01
**Domain:** Frontend UI (React + @dnd-kit), minor backend endpoint addition
**Confidence:** HIGH

## Summary

Phase 21 closes 4 tech debt items from the v1.0 audit by wiring up Sequence frontend features that have backend support but lack UI integration. The backend API is already comprehensive (CRUD, reorder sequences, assign scenes) and the frontend has the component structure (SequencedSceneList, SortableSequenceSection, SequenceHeader, SequenceContextMenu) but four specific features are missing or incomplete:

1. **Sequence drag-and-drop reorder**: `reorderSequences()` client function exists but is never called from UI. No DnD on sequence headers.
2. **Act field UI**: Act displays as a read-only badge but has no setter. SequenceContextMenu has color/edit/delete but no "Set Act" option.
3. **Total duration**: `SequenceResponse.total_duration` is computed by the backend but not rendered in SequenceHeader.
4. **Within-sequence scene reorder**: Scenes can be dragged between sequences but not reordered within one. Backend lacks a bulk scene reorder endpoint (only individual `assign_scene_to_sequence` with `scene_order`).

**Primary recommendation:** Add the missing UI controls to existing components and add one new backend endpoint (`PUT /api/sequences/{id}/scenes/reorder`) for bulk scene reorder within a sequence.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| @dnd-kit/core | 6.3.1 | DnD context, sensors, events | Already in project, used for scene drag between sequences |
| @dnd-kit/sortable | 10.0.0 | SortableContext, useSortable, arrayMove | Already in project (EditModeOverlay uses it for shot reorder) |
| @dnd-kit/utilities | 3.2.2 | CSS.Transform helper | Already in project |
| React | 19.2.0 | UI framework | Project standard |
| TypeScript | strict | Type safety | Project standard |
| Tailwind CSS | 4 | Styling | Project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| FastAPI | (installed) | Backend endpoint | New scene reorder endpoint |
| SQLAlchemy | (installed) | ORM queries | Bulk scene_order update |

No new dependencies needed. Everything is already installed.

## Architecture Patterns

### Existing Component Structure
```
frontend/src/components/
  ProductionDetail.tsx      # Parent — renders SequencedSceneList when sequences exist
  SequencedSceneList.tsx    # DndContext wrapper, sequence CRUD, scene assignment
  SortableSequenceSection.tsx # Droppable sequence with DraggableSceneRow children
  SequenceHeader.tsx        # Sequence title, color dot, scene count badge, act badge, collapse
  SequenceContextMenu.tsx   # Three-dot menu: edit title, change color, delete
  UnsequencedSection.tsx    # Droppable zone for unsequenced scenes
```

### Backend Endpoint Structure
```
backend/vidpipe/api/sequences.py
  GET  /api/productions/{id}/sequences         — list (already computes total_duration)
  POST /api/productions/{id}/sequences         — create
  GET  /api/sequences/{id}                     — detail with scenes
  PUT  /api/sequences/{id}                     — update (title, description, act, color)
  DELETE /api/sequences/{id}                   — delete (unsequences children)
  PUT  /api/productions/{id}/sequences/reorder — bulk reorder sequences
  PUT  /api/scenes/{id}/sequence               — assign scene to sequence
  [MISSING] PUT /api/sequences/{id}/scenes/reorder — bulk reorder scenes within sequence
```

### Pattern 1: Optimistic UI Update (Established Pattern)
**What:** Update local state immediately, revert on API failure
**When to use:** All drag-and-drop operations
**Example (from SequencedSceneList.tsx):**
```typescript
// Optimistic update
const previousScenes = localScenes;
setLocalScenes((prev) => prev.map((s) => ...));
try {
  await apiCall();
  onRefresh?.();
} catch {
  setLocalScenes(previousScenes);  // Revert
  setError("Failed to move scene");
}
```

### Pattern 2: DnD Sortable List (Established Pattern in EditModeOverlay)
**What:** SortableContext wrapping useSortable items with arrayMove on drag end
**When to use:** Reordering items within a list (sequence reorder, scene reorder within sequence)
**Example (from EditModeOverlay.tsx lines 850-864):**
```typescript
import { SortableContext, verticalListSortingStrategy, arrayMove } from "@dnd-kit/sortable";

const sensors = useSensors(
  useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
);

function handleDragEnd(event: DragEndEvent) {
  const { active, over } = event;
  if (over && active.id !== over.id) {
    setOrder(prev => {
      const oldIndex = prev.indexOf(active.id as string);
      const newIndex = prev.indexOf(over.id as string);
      return arrayMove(prev, oldIndex, newIndex);
    });
  }
}
```

### Pattern 3: Context Menu Options (Established Pattern in SequenceContextMenu)
**What:** Three-dot button opening dropdown with action items
**When to use:** Adding Act selector to context menu
**Example (existing in SequenceContextMenu.tsx):**
```typescript
<button
  onClick={() => setShowColorPicker((v) => !v)}
  className="w-full text-left px-3 py-2 text-sm text-gray-200 hover:bg-gray-700"
>
  Change color
</button>
{showColorPicker && (
  <div className="border-t border-gray-700 px-1 py-1">
    <ColorPicker selectedColor={color} onColorSelect={handleColorSelect} />
  </div>
)}
```

### Anti-Patterns to Avoid
- **Nested DndContexts without collision detection strategy:** SequencedSceneList already has a DndContext for cross-sequence scene dragging. Adding sequence reorder and within-sequence scene reorder requires careful DnD architecture. Use separate DnD contexts or a single context with `closestCenter` collision detection and data attributes to distinguish drag types.
- **Missing `scene_order` in SceneListItem type:** Frontend TypeScript type is missing `scene_order` even though backend returns it. Fix the type before relying on it for sort.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sortable list DnD | Custom drag handlers | @dnd-kit/sortable SortableContext + arrayMove | Already used in EditModeOverlay for shot reorder |
| Dropdown selectors | Custom select widget | Simple button list in context menu submenu | Matches existing ColorPicker pattern in context menu |
| Duration formatting | Manual string concat | Simple `formatDuration(seconds)` helper | Edge cases (null, 0, large values) |

## Common Pitfalls

### Pitfall 1: Nested DndContext Conflicts
**What goes wrong:** Adding SortableContext for sequence reorder inside the existing DndContext (which handles scene cross-sequence drag) causes drag events to fire on the wrong context.
**Why it happens:** @dnd-kit's DndContext captures all drag events within its DOM subtree.
**How to avoid:** For sequence reorder, there are two options:
  - (A) Use a single DndContext with data-type discrimination in `handleDragEnd` (check `active.data.current.type === 'sequence'` vs `'scene'`).
  - (B) Keep sequence reorder separate (dedicated drag handle on SequenceHeader that calls `reorderSequences` API on reorder).
**Recommendation:** Option A is cleaner -- add a `type` field to the `data` property of both sequence and scene draggables, then route in `handleDragEnd`.

### Pitfall 2: SceneListItem Missing scene_order
**What goes wrong:** TypeScript type `SceneListItem` in `frontend/src/api/types.ts` does not include `scene_order` despite backend returning it. Sorting code in `SortableSequenceSection.tsx` uses an unsafe cast.
**Why it happens:** Type was defined before scene_order was added to the backend response.
**How to avoid:** Add `scene_order?: number | null;` to the `SceneListItem` interface in `types.ts`.
**Warning signs:** TypeScript casts like `(a as SceneListItem & { scene_order?: number })`.

### Pitfall 3: Optimistic Revert Complexity with Reorder
**What goes wrong:** Reverting a multi-item reorder requires saving the entire previous order array, not just one item.
**Why it happens:** arrayMove returns a new array; you need the full pre-drag state to revert.
**How to avoid:** Save `previousSequences` / `previousScenes` before optimistic update, restore on catch.

### Pitfall 4: Act Value Validation
**What goes wrong:** Backend validates `VALID_ACTS = {"ACT_1", "ACT_2", "ACT_3"}` and returns 422 on invalid values.
**Why it happens:** Frontend sends free-form text instead of constrained values.
**How to avoid:** Use a fixed button list (ACT 1 / ACT 2 / ACT 3 / None) in the UI, not a text input.

### Pitfall 5: Duration Display for Null/Zero
**What goes wrong:** Displaying "0s" or "null" when no scenes have completed.
**Why it happens:** `total_duration` is null when no scenes have durations, or 0 if all are zero.
**How to avoid:** Show duration badge only when `total_duration != null && total_duration > 0`. Format as `Xm Ys` for readability.

## Code Examples

### Sequence Reorder DnD Handler
```typescript
// In SequencedSceneList.tsx — extend handleDragEnd
import { arrayMove } from "@dnd-kit/sortable";
import { reorderSequences } from "../api/client.ts";

async function handleDragEnd(event: DragEndEvent) {
  const { active, over } = event;
  if (!over || active.id === over.id) return;

  const activeType = active.data.current?.type;

  if (activeType === "sequence") {
    // Sequence reorder
    const oldIndex = sequences.findIndex((s) => s.id === active.id);
    const newIndex = sequences.findIndex((s) => s.id === over.id);
    if (oldIndex === -1 || newIndex === -1) return;

    const previousSequences = sequences;
    const reordered = arrayMove(sequences, oldIndex, newIndex);
    setSequences(reordered);

    try {
      await reorderSequences(productionId, {
        sequence_ids: reordered.map((s) => s.id),
      });
    } catch {
      setSequences(previousSequences);
      setError("Failed to reorder sequences");
    }
    return;
  }

  // Scene cross-sequence drag (existing logic)
  // ...
}
```

### Act Selector in Context Menu
```typescript
// In SequenceContextMenu.tsx — add act submenu
const ACTS = [
  { value: "ACT_1", label: "Act 1" },
  { value: "ACT_2", label: "Act 2" },
  { value: "ACT_3", label: "Act 3" },
  { value: null, label: "None" },
];

{showActPicker && (
  <div className="border-t border-gray-700 py-1">
    {ACTS.map((a) => (
      <button
        key={a.value ?? "none"}
        onClick={() => handleActSelect(a.value)}
        className={`w-full text-left px-3 py-1.5 text-sm ${
          currentAct === a.value ? "text-blue-400" : "text-gray-200"
        } hover:bg-gray-700`}
      >
        {a.label}
      </button>
    ))}
  </div>
)}
```

### Duration Display in SequenceHeader
```typescript
// In SequenceHeader.tsx — add duration badge after scene count
function formatDuration(seconds: number): string {
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }
  return `${seconds}s`;
}

{sequence.total_duration != null && sequence.total_duration > 0 && (
  <span className="text-xs text-gray-400 flex-shrink-0">
    {formatDuration(sequence.total_duration)}
  </span>
)}
```

### Backend: Bulk Scene Reorder Within Sequence
```python
# In sequences.py — new endpoint
class SceneReorderRequest(BaseModel):
    scene_ids: list[str]  # ordered list of scene UUIDs

@sequence_router.put("/sequences/{sequence_id}/scenes/reorder")
async def reorder_scenes_in_sequence(sequence_id: str, body: SceneReorderRequest):
    """Bulk reorder scenes within a sequence."""
    async with async_session() as session:
        seq = await session.get(Sequence, uuid.UUID(sequence_id))
        if seq is None:
            raise HTTPException(status_code=404, detail="Sequence not found")

        # Validate all scenes belong to this sequence
        result = await session.execute(
            select(Scene.id).where(Scene.sequence_id == uuid.UUID(sequence_id))
        )
        existing_ids = {str(row[0]) for row in result.all()}

        for sid in body.scene_ids:
            if sid not in existing_ids:
                raise HTTPException(
                    status_code=422,
                    detail=f"Scene {sid} not in sequence {sequence_id}",
                )

        for idx, sid in enumerate(body.scene_ids):
            await session.execute(
                update(Scene)
                .where(Scene.id == uuid.UUID(sid))
                .values(scene_order=idx)
            )

        await session.commit()
        return {"status": "reordered"}
```

## Gap Analysis: What Exists vs What's Needed

| Feature | Backend | Frontend Client | Frontend UI | Gap |
|---------|---------|-----------------|-------------|-----|
| Sequence reorder | PUT reorder endpoint exists | `reorderSequences()` exists | No DnD on sequences | UI only |
| Act field | PUT update accepts `act` | `updateSequence()` handles it | Read-only badge, no setter | UI only |
| Total duration | Computed in list endpoint | `SequenceResponse.total_duration` typed | Not rendered | UI only |
| Scene reorder within sequence | No bulk endpoint | No client function | No DnD within sequence | Backend + Client + UI |
| `scene_order` in SceneListItem type | Returned by API | Missing from TypeScript type | Cast workaround | Type fix |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| @dnd-kit/core only (droppable zones) | @dnd-kit/sortable (SortableContext) | Already in project | Use SortableContext for within-list reorder |
| Individual scene_order updates | Bulk reorder endpoint | Needed for this phase | Single API call instead of N calls |

## Open Questions

1. **DnD Architecture: Single vs Multiple Contexts**
   - What we know: SequencedSceneList uses one DndContext for cross-sequence scene drag. Need to also support sequence-to-sequence reorder and within-sequence scene reorder.
   - What's unclear: Whether @dnd-kit handles nested sortable contexts well, or if a unified context with type discrimination is required.
   - Recommendation: Use a single DndContext with `active.data.current.type` discrimination to route between sequence reorder, scene cross-sequence drag, and scene within-sequence reorder. This avoids nested context issues.

2. **Act Clearing**
   - What we know: Backend update endpoint uses `model_fields_set` to detect explicit null (clearing act).
   - What's unclear: Whether the SequenceUpdate Pydantic model properly serializes null for `act` when sent from frontend.
   - Recommendation: Frontend should send `{ act: null }` explicitly to clear. Verify the `updateSequence` client includes null values in JSON body.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `backend/vidpipe/api/sequences.py` — full backend API reviewed
- Codebase inspection: `frontend/src/components/Sequence*.tsx` — all sequence components reviewed
- Codebase inspection: `frontend/src/api/client.ts` + `types.ts` — client API functions and types reviewed
- Codebase inspection: `frontend/src/components/EditModeOverlay.tsx` — existing @dnd-kit/sortable pattern
- Codebase inspection: `backend/vidpipe/db/models.py` — Sequence and Scene model fields
- `package.json` — @dnd-kit/core 6.3.1, @dnd-kit/sortable 10.0.0

### Secondary (MEDIUM confidence)
- @dnd-kit/sortable API: SortableContext, useSortable, arrayMove confirmed via dist/index.d.ts exports

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in project, versions confirmed from package.json
- Architecture: HIGH - all existing components inspected, patterns extracted from working code
- Pitfalls: HIGH - identified from actual code issues (missing type field, nested DnD contexts, act validation)

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (stable — no external dependencies changing)
