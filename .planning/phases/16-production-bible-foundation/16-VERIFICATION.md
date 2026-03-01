---
phase: 16-production-bible-foundation
verified: 2026-02-28T00:00:00Z
status: gaps_found
score: 7/8 success criteria verified
re_verification: false
gaps:
  - truth: "Frontend renders scenes grouped by sequence when sequences exist, with collapsible sections and drag between sequences"
    status: failed
    reason: "Backend SceneListItem Pydantic schema in routes.py does not include sequence_id or scene_order fields. The API serialization loop at line 1369 omits both fields. As a result, all scenes returned from GET /api/scenes have sequence_id=undefined in the frontend, so SequencedSceneList.scenesForSequence() always returns empty arrays for every sequence, and all scenes appear in UnsequencedSection regardless of their actual sequence assignment."
    artifacts:
      - path: "backend/vidpipe/api/routes.py"
        issue: "SceneListItem Pydantic model (line 352) is missing sequence_id: Optional[str] and scene_order: Optional[int] fields. The serialization block (line 1369) does not map p.sequence_id or p.scene_order."
    missing:
      - "Add sequence_id: Optional[str] = None and scene_order: Optional[int] = None to the SceneListItem Pydantic model in routes.py"
      - "Add sequence_id=str(p.sequence_id) if p.sequence_id else None and scene_order=p.scene_order to the SceneListItem() constructor call in list_scenes()"
human_verification:
  - test: "Navigate to a production detail page, create a sequence, assign a scene to it, reload — verify the scene appears in the sequence section (not Unsequenced)"
    expected: "Scene appears under the named sequence header, not in the Unsequenced section"
    why_human: "Requires a running backend + frontend with real data to confirm the fix is end-to-end correct"
  - test: "Verify department tabs in Production Bible creator (open a Production Bible with CHARACTER and ENVIRONMENT assets)"
    expected: "Casting tab shows CHARACTER assets; Art Department tab shows ENVIRONMENT/PROP/OBJECT/VEHICLE/STYLE assets; Sound tab shows 'Audio direction coming soon' placeholder"
    why_human: "Requires running frontend with real asset data — cannot verify tab filtering logic against real assets programmatically"
---

# Phase 16: Production Bible Foundation Verification Report

**Phase Goal:** Rename the Manifest concept to Production Bible across the entire stack (database, API, frontend), introduce department tab structure in the Production Bible detail view, and add an optional Sequence grouping layer above Scenes for narrative chapter organization
**Verified:** 2026-02-28
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `production_bibles` table exists with all data migrated from `manifests`; all FK columns renamed to `production_bible_id` | VERIFIED | `ProductionBible.__tablename__ == 'production_bibles'`; `Asset.production_bible_id`, `ManifestSnapshot.production_bible_id`, `Scene.production_bible_id`, `Scene.production_bible_version` all confirmed in models.py; `_run_rename_migrations()` handles table+column renames |
| 2 | All API endpoints respond at `/api/production-bibles/*` with 301 redirects from legacy `/api/manifests/*` paths | VERIFIED | 14 routes at `/api/production-bibles/*`; 2 legacy redirect routes at `/api/manifests` and `/api/manifests/{path:path}` with status_code=301 confirmed in routes.py |
| 3 | Frontend uses "Production Bible" terminology everywhere; routes updated to `/production-bibles/*` | VERIFIED | App.tsx routes at `/production-bibles`, `/production-bibles/new`, `/production-bibles/:id/edit`; Layout.tsx nav shows "Production Bibles"; all components use ProductionBible naming |
| 4 | Production Bible detail view has three department tabs: Casting, Art Department, Sound — with existing assets sorted into correct tabs | VERIFIED | `DEPARTMENT_TABS` constant with Casting (CHARACTER), Art Department (ENVIRONMENT/PROP/OBJECT/VEHICLE/STYLE), Sound (placeholder) defined in ProductionBibleCreator.tsx; `filteredAssets` filters by `activeTab`; Sound tab renders "Audio direction coming soon" |
| 5 | `sequences` table stores optional grouping layer with title, description, order, act, and color fields | VERIFIED | `Sequence` model confirmed: `__tablename__ = 'sequences'`, fields `title`, `description`, `order`, `act`, `color`, `production_id`, `created_at`, `updated_at` all present |
| 6 | Scene model has optional `sequence_id` FK; scenes with null sequence_id remain in flat list | VERIFIED | `Scene.sequence_id` mapped as `Optional[uuid.UUID]` with `ForeignKey("sequences.id"), nullable=True`; `Scene.scene_order` column present; DB migrations include `ALTER TABLE scenes ADD COLUMN sequence_id` and `scene_order` |
| 7 | Sequence CRUD API under `/api/productions/{id}/sequences` with drag-and-drop reorder support | VERIFIED | 7 routes confirmed: list, create, get, update, delete, reorder (PUT `/api/productions/{id}/sequences/reorder`), assign (`PUT /api/scenes/{id}/sequence`); delete unsequences children; bulk reorder validates ownership |
| 8 | Frontend renders scenes grouped by sequence when sequences exist, with collapsible sections and drag between sequences | FAILED | All UI components exist (SequencedSceneList, SortableSequenceSection, SequenceHeader, UnsequencedSection, ColorPicker, SequenceContextMenu) and ProductionDetail conditionally renders SequencedSceneList. However, the backend `SceneListItem` Pydantic model (routes.py line 352) does not include `sequence_id` or `scene_order` fields. The serialization loop (line 1369) does not map these fields. Every scene returned from `GET /api/scenes` will have `sequence_id=undefined` on the frontend, so `scenesForSequence()` returns empty arrays for every sequence — all scenes always appear in UnsequencedSection |

**Score:** 7/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/vidpipe/db/models.py` | ProductionBible model + Sequence model + Scene.sequence_id | VERIFIED | ProductionBible.__tablename__='production_bibles', Manifest alias; Sequence with all fields; Scene.sequence_id FK, Scene.scene_order |
| `backend/vidpipe/db/__init__.py` | Migration renaming manifests table, FK columns, adding sequence columns | VERIFIED | `_run_rename_migrations()` handles table + column renames; `_run_migrations()` adds sequence_id and scene_order to scenes |
| `backend/vidpipe/api/routes.py` | Renamed endpoints + 301 redirects + ProductionBible import | VERIFIED | 14 production-bibles routes; 2 legacy redirect routes; imports ProductionBible from models |
| `backend/vidpipe/api/routes.py` (SceneListItem) | SceneListItem includes sequence_id field | FAILED | SceneListItem Pydantic model at line 352 is missing sequence_id and scene_order fields; serialization at line 1369 does not include them |
| `backend/vidpipe/services/manifest_service.py` | Service using ProductionBible model | VERIFIED | Imports ProductionBible; uses local `Manifest = ProductionBible` alias; all .production_bible_id accesses confirmed |
| `backend/vidpipe/api/sequences.py` | 7 sequence CRUD endpoints | VERIFIED | 7 routes: GET/POST list, GET/PUT/DELETE single, PUT reorder, PUT assign; all with DB logic, validation, proper responses |
| `backend/vidpipe/api/app.py` | sequence_router included | VERIFIED | `from vidpipe.api.sequences import sequence_router` + `app.include_router(sequence_router)` at lines 16, 67 |
| `frontend/src/api/types.ts` | ProductionBible types + SequenceResponse + SceneListItem.sequence_id | VERIFIED | ProductionBibleListItem, ProductionBibleDetail, CreateProductionBibleRequest, UpdateProductionBibleRequest, deprecated aliases; SequenceResponse, SequenceWithScenes, SequenceCreate, SequenceUpdate, SequenceReorderRequest, AssignSequenceRequest; SceneListItem includes `sequence_id?: string | null` |
| `frontend/src/api/client.ts` | API client using /api/production-bibles/ + 7 sequence functions | VERIFIED | All production-bible endpoints use /api/production-bibles/ paths; listSequences, createSequence, getSequence, updateSequence, deleteSequence, reorderSequences, assignSceneToSequence all present |
| `frontend/src/components/ProductionBibleLibrary.tsx` | Library component with Production Bible terminology | VERIFIED | Uses listProductionBibles, ProductionBibleListItem, ProductionBibleCard; "Production Bible Library" header |
| `frontend/src/components/ProductionBibleCreator.tsx` | Creator with department tabs | VERIFIED | DEPARTMENT_TABS, activeTab state, tab rendering, filteredAssets logic, Sound placeholder |
| `frontend/src/components/ProductionBibleCard.tsx` | Card with production_bible_id | VERIFIED | Uses production_bible_id field throughout |
| `frontend/src/components/ProductionBibleSelector.tsx` | Selector with Production Bible terminology | VERIFIED | File exists, uses ProductionBible naming |
| `frontend/src/components/SequencedSceneList.tsx` | Main sequence grouping container | VERIFIED | DndContext, listSequences fetch, handleDragEnd, handleAddSequence, handleDeleteSequence, SortableSequenceSection, UnsequencedSection |
| `frontend/src/components/SortableSequenceSection.tsx` | Droppable sequence section | VERIFIED | useDroppable, useSortable on DraggableSceneRow, SequenceHeader, scene rendering by scene_order |
| `frontend/src/components/SequenceHeader.tsx` | Sequence header with color dot + title + collapse | VERIFIED | Color dot, editable title (double-click), scene_count badge, act label, SequenceContextMenu, collapse toggle |
| `frontend/src/components/UnsequencedSection.tsx` | Container for unsequenced scenes | VERIFIED | useDroppable with UNSEQUENCED_ID, renders if scenes.length > 0, exports UNSEQUENCED_ID |
| `frontend/src/components/SequenceContextMenu.tsx` | Context menu for sequence | VERIFIED | Edit, Change color (with ColorPicker), Delete with confirmation |
| `frontend/src/components/ColorPicker.tsx` | 8-preset color picker | VERIFIED | 8 PRESET_COLORS, selected ring, onColorSelect callback |
| `frontend/src/components/ProductionDetail.tsx` | Conditionally renders SequencedSceneList | VERIFIED | Fetches sequences, renders SequencedSceneList when sequences.length > 0, flat list when no sequences, "Create Sequence" button when empty |
| `frontend/src/App.tsx` | Routes at /production-bibles/* | VERIFIED | /production-bibles, /production-bibles/new, /production-bibles/:id/edit all route to correct components |
| `frontend/src/components/Layout.tsx` | Nav shows Production Bibles | VERIFIED | href: "/production-bibles", label: "Production Bibles" |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `backend/vidpipe/api/routes.py` | `backend/vidpipe/db/models.py` | ProductionBible import | VERIFIED | `from vidpipe.db.models import ... ProductionBible ...` present |
| `backend/vidpipe/api/routes.py` | `backend/vidpipe/services/manifest_service.py` | service function calls | VERIFIED | `manifest_service.` calls throughout routes.py |
| `backend/vidpipe/db/__init__.py` | `backend/vidpipe/db/models.py` | migration references new table name | VERIFIED | `production_bibles` referenced in _run_rename_migrations and _run_migrations |
| `backend/vidpipe/api/sequences.py` | `backend/vidpipe/db/models.py` | Sequence and Scene model imports | VERIFIED | `from vidpipe.db.models import Production, Scene, Sequence` |
| `backend/vidpipe/api/app.py` | `backend/vidpipe/api/sequences.py` | router inclusion | VERIFIED | `from vidpipe.api.sequences import sequence_router` + `app.include_router(sequence_router)` |
| `backend/vidpipe/db/models.py` | `backend/vidpipe/db/models.py` | Scene.sequence_id FK to Sequence.id | VERIFIED | `ForeignKey("sequences.id"), nullable=True` in Scene.sequence_id |
| `frontend/src/api/client.ts` | `backend/vidpipe/api/routes.py` | fetch calls to /api/production-bibles/* | VERIFIED | All client functions use /api/production-bibles/ paths |
| `frontend/src/App.tsx` | `frontend/src/components/ProductionBibleLibrary.tsx` | route rendering | VERIFIED | Route `/production-bibles` renders `<ProductionBibleLibrary>` |
| `frontend/src/components/Layout.tsx` | `frontend/src/App.tsx` | nav link to /production-bibles | VERIFIED | href: "/production-bibles" in NAV_ITEMS |
| `frontend/src/api/client.ts` | `backend/vidpipe/api/sequences.py` | fetch calls to /api/productions/{id}/sequences | VERIFIED | listSequences, createSequence etc. use correct paths |
| `frontend/src/components/ProductionDetail.tsx` | `frontend/src/components/SequencedSceneList.tsx` | conditional render on sequences.length > 0 | VERIFIED | `sequences.length > 0 ? <SequencedSceneList> : flat list` pattern |
| `frontend/src/components/SortableSequenceSection.tsx` | `frontend/src/api/client.ts` | assignSceneToSequence in handleDragEnd | VERIFIED | `assignSceneToSequence(sceneId, { sequence_id: targetSequenceId })` called in SequencedSceneList.handleDragEnd |
| `backend/vidpipe/api/routes.py` (SceneListItem) | `frontend/src/components/SequencedSceneList.tsx` | sequence_id field in scene data | NOT_WIRED | Backend SceneListItem schema does not include sequence_id; frontend component depends on it for grouping |

### Requirements Coverage

| Requirement | Source Plan | Description | Status |
|-------------|-------------|-------------|--------|
| PBIB-01 | 16-01-PLAN.md | DB rename: manifests → production_bibles table; FK columns renamed to production_bible_id | SATISFIED |
| PBIB-02 | 16-01-PLAN.md | API endpoints at /api/production-bibles/* | SATISFIED |
| PBIB-03 | 16-01-PLAN.md | 301 redirects from /api/manifests/* to /api/production-bibles/* | SATISFIED |
| PBIB-04 | 16-03-PLAN.md | Frontend uses Production Bible terminology; routes at /production-bibles/* | SATISFIED |
| PBIB-05 | 16-03-PLAN.md | API client calls /api/production-bibles/* endpoints | SATISFIED |
| PBIB-06 | 16-03-PLAN.md | Department tabs (Casting, Art Department, Sound) in Production Bible creator | SATISFIED |
| SEQ-01 | 16-02-PLAN.md + 16-04-PLAN.md | sequences table with title, description, order, act, color; Scene.sequence_id FK | SATISFIED |
| SEQ-02 | 16-02-PLAN.md | Sequence CRUD API under /api/productions/{id}/sequences | SATISFIED |
| SEQ-03 | 16-02-PLAN.md | DELETE sequence unsequences child scenes (does not delete them) | SATISFIED |
| SEQ-04 | 16-04-PLAN.md | Frontend renders scenes grouped by sequence with drag-and-drop | BLOCKED — backend SceneListItem does not return sequence_id field; frontend grouping is broken |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `frontend/src/components/ProductionBibleCreator.tsx` | 58 | Sound tab has `assetTypes: []` | Info | Intentional per spec — Sound tab is a planned placeholder |
| `frontend/src/components/ProductionBibleCreator.tsx` | 1003 | "Audio direction coming soon" | Info | Intentional per spec — Sound department is placeholder for future work |
| `backend/vidpipe/api/routes.py` | 352-376 | SceneListItem missing sequence_id/scene_order | Blocker | Prevents scene grouping from functioning — sequences UI cannot display scenes in correct groups |

### Human Verification Required

#### 1. Department Tabs Asset Filtering

**Test:** Open a Production Bible with a mix of CHARACTER, ENVIRONMENT, and PROP assets. Click each tab.
**Expected:** Casting tab shows only CHARACTER assets; Art Department tab shows ENVIRONMENT/PROP/OBJECT/VEHICLE/STYLE assets; Sound tab shows "Audio direction coming soon"
**Why human:** Requires running frontend with real production bible data

#### 2. 301 Redirect Behavior

**Test:** Make an HTTP request to `/api/manifests` and confirm it returns 301 to `/api/production-bibles`
**Expected:** HTTP 301 with Location header pointing to /api/production-bibles
**Why human:** Browser/curl behavior with redirect following vs. raw redirect inspection cannot be verified without a running server

### Gaps Summary

One gap blocks goal achievement for Success Criterion #8:

**The backend `SceneListItem` API response schema is missing `sequence_id` and `scene_order` fields.**

The entire sequence grouping UI in `SequencedSceneList` depends on each `SceneListItem` having a `sequence_id` field to group scenes under their sequence. The frontend `SceneListItem` TypeScript type correctly defines `sequence_id?: string | null` (added per the plan), but the backend Pydantic model at `routes.py:352` does not include these fields, and the serialization block at `routes.py:1369` does not map `p.sequence_id` or `p.scene_order` to the response.

As a result:
- `scenesForSequence(seqId)` in `SequencedSceneList` always returns `[]` for every sequence
- Every scene always appears in `UnsequencedSection`
- Drag-and-drop assignment via the API works correctly (the `PUT /api/scenes/{id}/sequence` endpoint is functional), but scenes will revert to appearing unsequenced on next page load

The fix requires two changes in `backend/vidpipe/api/routes.py`:
1. Add `sequence_id: Optional[str] = None` and `scene_order: Optional[int] = None` to `SceneListItem` (line 352)
2. Add `sequence_id=str(p.sequence_id) if p.sequence_id else None` and `scene_order=p.scene_order` to the `SceneListItem()` constructor (line 1369)

All other success criteria are fully achieved. The backend rename (PBIB-01 through PBIB-03), frontend rename (PBIB-04 through PBIB-06), sequence database model (SEQ-01), sequence CRUD API (SEQ-02, SEQ-03), and all sequence UI components (SEQ-04 partially) are substantive and wired correctly.

---

_Verified: 2026-02-28_
_Verifier: Claude (gsd-verifier)_
