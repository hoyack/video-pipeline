---
phase: 08-veo-reference-passthrough-and-clean-sheets
plan: 03
subsystem: ui
tags:
  - react
  - typescript
  - api
  - pydantic
  - scene-detail
dependency_graph:
  requires:
    - phase: 08
      plan: 01
      reason: "SceneManifest.selected_reference_tags column and reference selection service"
  provides:
    - name: "SceneReference API response model"
      for: "Frontend display of selected identity references per scene"
    - name: "SceneDetail.selected_references field"
      for: "Project detail API response with reference data"
    - name: "SceneCard reference badges UI"
      for: "Visual display of which assets were used for Veo generation"
  affects:
    - "backend/vidpipe/api/routes.py"
    - "frontend/src/api/types.ts"
    - "frontend/src/components/SceneCard.tsx"
tech_stack:
  added:
    - "SceneReference pydantic model"
    - "SceneReference TypeScript interface"
  patterns:
    - "Efficient reference resolution via in-memory join (no N+1 queries)"
    - "Backward-compatible empty array for non-manifest scenes"
key_files:
  created: []
  modified:
    - path: "backend/vidpipe/api/routes.py"
      changes: "Added SceneReference model, extended SceneDetail, added reference resolution in get_project_detail"
    - path: "frontend/src/api/types.ts"
      changes: "Added SceneReference interface, extended SceneDetail with selected_references field"
    - path: "frontend/src/components/SceneCard.tsx"
      changes: "Added Identity References badge section with thumbnail, manifest_tag, and quality score"
decisions:
  - id: "08-03-01"
    choice: "Efficient in-memory join pattern for reference resolution"
    rationale: "One query for all scene manifests, one query for all assets, then in-memory join prevents N+1 query problem"
    alternatives_rejected:
      - "Per-scene asset lookup (N+1 queries)"
      - "Eager loading with ORM relationships (adds complexity to models)"
  - id: "08-03-02"
    choice: "Reference badges only appear when scene has selected_references"
    rationale: "Backward compatible for non-manifest projects and scenes without references"
    alternatives_rejected:
      - "Always show section with 'No references' placeholder (clutters UI for majority of scenes)"
      - "Show badge count even when zero (confusing for pre-Phase 8 projects)"
metrics:
  duration: 1.8
  completed_date: "2026-02-17"
  task_count: 2
  file_count: 3
  commits:
    - hash: "4288582"
      message: "feat(08-03): add SceneReference API response with selected reference data"
    - hash: "80193be"
      message: "feat(08-03): add reference badges to SceneCard UI"
---

# Phase 08 Plan 03: Reference Display in SceneCard UI Summary

**One-liner:** SceneDetail API response includes selected_references array with asset metadata, SceneCard component displays reference badges showing thumbnail, manifest_tag, and quality score for each identity reference used in Veo generation.

## What Was Built

**Backend API:**
- **SceneReference pydantic model** with 8 fields: asset_id, manifest_tag, name, asset_type, thumbnail_url, reference_image_url, quality_score, is_face_crop
- **SceneDetail.selected_references** field (list[SceneReference], defaults to empty array for backward compatibility)
- **get_project_detail route enhancement** with efficient reference resolution:
  - Single query for all SceneManifest records by project_id
  - Single query for all Asset records by manifest_id
  - In-memory join via manifest_tag lookup (no N+1 queries)
  - Only queries assets if project.manifest_id exists
  - Populates selected_references for each scene from SceneManifest.selected_reference_tags

**Frontend UI:**
- **SceneReference TypeScript interface** matching backend schema
- **SceneDetail.selected_references** optional field
- **SceneCard Identity References section** (expanded state only):
  - Reference badges with thumbnail (6x6px rounded), manifest_tag (blue-400), quality score (gray-500)
  - Badge layout: horizontal flex with gap-2, gray-800/50 background, gray-700 border
  - Conditional render: only appears when `scene.selected_references && scene.selected_references.length > 0`
  - Positioned after keyframe images, before prompt sections

## Deviations from Plan

None - plan executed exactly as written. All specifications followed precisely.

## Key Decisions Made

**1. Efficient in-memory join for reference resolution (Decision 08-03-01)**
- **What:** Query all scene manifests and assets once, then join in memory by manifest_tag
- **Why:** Prevents N+1 query problem when resolving references for multiple scenes
- **Impact:** O(n+m) complexity instead of O(n*m), scales efficiently for projects with many scenes

**2. Conditional badge display (Decision 08-03-02)**
- **What:** Reference section only renders when selected_references exists and is non-empty
- **Why:** Backward compatible for non-manifest projects and scenes without references
- **Impact:** Existing UI unchanged for projects created before Phase 8, clean UX for scenes without references

## Implementation Notes

**API response structure:**
```typescript
{
  scene_index: 2,
  // ... existing SceneDetail fields ...
  selected_references: [
    {
      asset_id: "uuid",
      manifest_tag: "CHAR_01",
      name: "Character 1 Face Crop",
      asset_type: "CHARACTER",
      thumbnail_url: "/api/assets/uuid/image",
      reference_image_url: "/api/assets/uuid/image",
      quality_score: 8.5,
      is_face_crop: true
    },
    // ... up to 3 references total
  ]
}
```

**Badge visual hierarchy:**
- Thumbnail first (if available) for visual recognition
- manifest_tag in blue-400 for clear identity
- Quality score in gray-500 for context (rounded to 1 decimal)
- Consistent with existing SceneCard design language (10px uppercase labels, gray-800 backgrounds)

**Placement in SceneCard:**
1. Keyframe images (start/end)
2. **Identity References** (Phase 8 - new)
3. Prompt sections (start/end/motion/transition)
4. Clip video player

## Testing Performed

**Backend verification:**
- ✓ SceneReference model has 8 fields (asset_id, manifest_tag, name, asset_type, thumbnail_url, reference_image_url, quality_score, is_face_crop)
- ✓ SceneDetail.selected_references field exists
- ✓ SceneManifestModel imported in routes.py
- ✓ selected_references appears 2 times in routes.py (model field + population logic)

**Frontend verification:**
- ✓ TypeScript compilation passes with no type errors
- ✓ SceneReference interface exists in types.ts
- ✓ selected_references field added to SceneDetail interface
- ✓ SceneCard imports SceneReference type
- ✓ "Identity References" label renders in expanded state
- ✓ Reference badges iterate over scene.selected_references array

## Artifacts Created

**Files modified:**
- `backend/vidpipe/api/routes.py` (+54 lines)
  - SceneReference pydantic model (9 lines)
  - SceneDetail.selected_references field (1 line)
  - Reference resolution logic in get_project_detail (44 lines)
- `frontend/src/api/types.ts` (+11 lines)
  - SceneReference interface (9 lines)
  - SceneDetail.selected_references field (1 line)
- `frontend/src/components/SceneCard.tsx` (+37 lines)
  - Import SceneReference type (1 line)
  - Identity References section (36 lines)

**Database queries (per get_project_detail call):**
- 1 query: All SceneManifest records for project
- 1 query: All Asset records for project's manifest (if manifest_id exists)
- 0 queries: In-memory join by manifest_tag

## Next Steps

**Phase 8 Plan 02 (if not already complete):**
- Integrate reference selection service into video_gen pipeline
- Pass selected_references to Veo API as reference_images parameter
- Persist selected_reference_tags to SceneManifest after selection

**Phase 8 Clean Sheets (Plan 02 extension):**
- Display clean sheet tier (tier2_rembg, tier3_gemini) in reference badges
- Show clean sheet override indicator when clean_image_url used instead of reference_image_url
- Add face similarity score for tier3 clean sheets

## Self-Check: PASSED

**Files modified:**
```
FOUND: backend/vidpipe/api/routes.py
FOUND: frontend/src/api/types.ts
FOUND: frontend/src/components/SceneCard.tsx
```

**Commits exist:**
```
FOUND: 4288582 (feat: add SceneReference API response with selected reference data)
FOUND: 80193be (feat: add reference badges to SceneCard UI)
```

**Backend verification:**
```
SceneReference fields: ['asset_id', 'manifest_tag', 'name', 'asset_type', 'thumbnail_url', 'reference_image_url', 'quality_score', 'is_face_crop']
selected_references in SceneDetail: True
SceneManifestModel imported and used in routes.py
```

**Frontend verification:**
```
TypeScript compilation: PASSED (no errors)
SceneReference interface: EXISTS
selected_references in SceneDetail: EXISTS
Identity References section: RENDERS
```

All claimed artifacts verified and present in repository.
