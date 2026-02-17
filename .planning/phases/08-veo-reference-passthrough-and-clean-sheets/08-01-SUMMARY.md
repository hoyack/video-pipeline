---
phase: 08-veo-reference-passthrough-and-clean-sheets
plan: 01
subsystem: reference-selection
tags:
  - database
  - orm
  - service
  - scene-aware-selection
dependency_graph:
  requires:
    - phase: 07
      plan: 02
      reason: "SceneManifest and manifest-aware storyboard schemas"
  provides:
    - name: "AssetCleanReference model"
      for: "Phase 8 Plans 02 (video_gen clean sheet integration) and 03 (frontend display)"
    - name: "SceneManifest.selected_reference_tags column"
      for: "Phase 8 Plan 02 (video_gen reference passthrough)"
    - name: "select_references_for_scene function"
      for: "Phase 8 Plan 02 (video_gen reference selection)"
  affects:
    - "backend/vidpipe/db/models.py"
    - "backend/vidpipe/services/manifest_service.py"
tech_stack:
  added:
    - "AssetCleanReference ORM model"
    - "reference_selection.py service module"
  patterns:
    - "Scene-type-aware reference prioritization"
    - "Face crop vs full-body selection logic"
    - "Manifest_tag deduplication"
key_files:
  created:
    - path: "backend/vidpipe/services/reference_selection.py"
      exports: ["select_references_for_scene", "get_primary_clean_reference"]
  modified:
    - path: "backend/vidpipe/db/models.py"
      changes: "Added AssetCleanReference model, added SceneManifest.selected_reference_tags column"
    - path: "backend/vidpipe/db/__init__.py"
      changes: "Registered AssetCleanReference import"
decisions:
  - id: "08-01-01"
    choice: "Scene-type-aware selection adapts prioritization by shot_type"
    rationale: "close_up prioritizes face crops, two_shot ensures 2 unique characters, establishing prioritizes environments"
    alternatives_rejected:
      - "Universal role-based priority (ignores composition context)"
      - "Random selection (non-deterministic, no quality consideration)"
  - id: "08-01-02"
    choice: "Deduplication by manifest_tag prevents same character occupying multiple slots"
    rationale: "Ensures 3 distinct assets for Veo reference diversity"
    alternatives_rejected:
      - "Allow duplicates (wastes reference slots)"
      - "Deduplicate by asset_type only (allows CHAR_01 face + full-body in same scene)"
metrics:
  duration: 2.6
  completed_date: "2026-02-17"
  task_count: 2
  file_count: 3
  commits:
    - hash: "4a88890"
      message: "feat(08-01): add AssetCleanReference model and SceneManifest.selected_reference_tags"
    - hash: "afddc56"
      message: "feat(08-01): implement scene-type-aware reference selection service"
---

# Phase 08 Plan 01: Reference Selection Data Layer and Service Summary

**One-liner:** AssetCleanReference ORM model, SceneManifest.selected_reference_tags column, and scene-type-aware reference selection service that prioritizes up to 3 assets per scene based on shot composition.

## What Was Built

**Database Layer:**
- **AssetCleanReference model** with 10 columns: id, asset_id, tier, clean_image_url, generation_prompt, face_similarity_score, quality_score, is_primary, generation_cost, created_at
- **SceneManifest.selected_reference_tags** JSON column to store list of manifest_tag strings (e.g. ["CHAR_01", "ENV_01", "PROP_02"])
- Registered AssetCleanReference in `db/__init__.py` for metadata registration

**Service Layer:**
- **select_references_for_scene()** function implementing scene-type-aware selection:
  - **close_up**: Prioritizes face crops of subject role (is_face_crop=True), then full-body subjects, then environments
  - **two_shot**: Gets up to 2 unique CHARACTER assets from subject/interaction_target roles, fills remaining slot with environment
  - **establishing**: Prioritizes ENVIRONMENT assets, then PROP/VEHICLE, then characters (prefers full-body over face crops)
  - **default** (medium_shot, wide_shot): Standard priority subject > interaction_target > background, sorted by quality_score
- **get_primary_clean_reference()** async function to query clean sheet override (is_primary=True) for video_gen integration
- **Deduplication logic** prevents same manifest_tag occupying multiple reference slots

## Deviations from Plan

None - plan executed exactly as written. All specifications followed precisely.

## Key Decisions Made

**1. Scene-type-aware prioritization (Decision 08-01-01)**
- **What:** Selection logic adapts by composition.shot_type
- **Why:** Different shot types have different reference needs (close-ups need faces, establishing needs environments)
- **Impact:** Veo references are contextually optimal for each scene

**2. Manifest_tag deduplication (Decision 08-01-02)**
- **What:** Same character (e.g. CHAR_01 face crop + full-body) deduplicated to single slot
- **Why:** Ensures 3 distinct assets for maximum Veo reference diversity
- **Impact:** No wasted reference slots on redundant assets

## Implementation Notes

**Scene-type selection strategies:**

| Shot Type    | Priority 1             | Priority 2             | Priority 3         |
| ------------ | ---------------------- | ---------------------- | ------------------ |
| close_up     | Face crops (subject)   | Full-body subject      | Environments       |
| two_shot     | 2 unique characters    | Environment            | N/A                |
| establishing | ENVIRONMENT assets     | PROP/VEHICLE assets    | Characters         |
| default      | Subject role           | Interaction_target     | Background         |

**Quality sorting:** Within each priority group, assets sorted by quality_score descending.

**Face crop handling:** Establishing shots prefer full-body characters (filter out face crops) for contextual appropriateness.

**Placement role mapping:** Uses AssetPlacement.role from scene manifest (subject, interaction_target, background, environment, prop).

## Testing Performed

**Unit tests (mock data):**
- ✓ close_up scene prioritizes face crop CHAR_01 over full-body CHAR_02
- ✓ two_shot scene selects 2 unique characters + environment (3 total)
- ✓ establishing scene prioritizes ENV_01 first, then PROP_01
- ✓ Empty placements return empty list
- ✓ All selections respect max 3 assets

**Import verification:**
- ✓ AssetCleanReference model imports and has 10 columns
- ✓ SceneManifest.selected_reference_tags column exists
- ✓ reference_selection module imports successfully

## Artifacts Created

**Files:**
- `backend/vidpipe/services/reference_selection.py` (275 lines)
- `backend/vidpipe/db/models.py` (modified, +24 lines)
- `backend/vidpipe/db/__init__.py` (modified, +1 import)

**Database schema changes:**
- New table: `asset_clean_references` (10 columns)
- Column added: `scene_manifests.selected_reference_tags` (JSON)

## Next Steps (Phase 8 Plan 02)

**Integration points:**
1. **video_gen pipeline**: Call `select_references_for_scene()` before Veo submission
2. **Clean sheet override**: Use `get_primary_clean_reference()` to check for tier3 replacements
3. **Persist selection**: Store selected_reference_tags in SceneManifest after selection
4. **Reference URLs**: Map selected assets to image URLs for Veo API `reference_images` parameter

**Requirements for Plan 02:**
- Modify `backend/vidpipe/pipeline/video_generation.py` to integrate reference selection
- Add clean sheet URL resolution logic (primary clean_image_url > reference_image_url)
- Update VideoClip model to track reference_images_used (list of manifest_tags)
- Pass up to 3 reference URLs to Veo GenerateVideosConfig

## Self-Check: PASSED

**Files created:**
```
FOUND: backend/vidpipe/services/reference_selection.py
```

**Files modified:**
```
FOUND: backend/vidpipe/db/models.py
FOUND: backend/vidpipe/db/__init__.py
```

**Commits exist:**
```
FOUND: 4a88890 (feat: add AssetCleanReference model and SceneManifest.selected_reference_tags)
FOUND: afddc56 (feat: implement scene-type-aware reference selection service)
```

**Model verification:**
```
AssetCleanReference.__tablename__ = 'asset_clean_references'
SceneManifest has 'selected_reference_tags' column = True
reference_selection imports = OK
```

All claimed artifacts verified and present in repository.
