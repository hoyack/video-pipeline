---
phase: 08-veo-reference-passthrough-and-clean-sheets
plan: 02
subsystem: video-generation-pipeline
tags:
  - veo-api
  - reference-images
  - clean-sheets
  - identity-preservation
dependency_graph:
  requires:
    - phase: 08
      plan: 01
      reason: "AssetCleanReference model and reference_selection service"
    - phase: 07
      plan: 02
      reason: "SceneManifest and manifest-aware storyboard"
  provides:
    - name: "Reference image passthrough to Veo 3.1"
      for: "Phase 8 Plan 03 (frontend display of selected references)"
    - name: "Clean sheet generation (Tier 2 and 3)"
      for: "Manifest processing pipeline in Phase 5+"
  affects:
    - "backend/vidpipe/pipeline/video_gen.py"
    - "backend/vidpipe/services/clean_sheet_service.py"
tech_stack:
  added:
    - "VideoGenerationReferenceImage (Veo API)"
    - "rembg for background removal (optional)"
    - "insightface for face validation (optional)"
  patterns:
    - "Lazy loading for optional CV dependencies"
    - "CPU-bound work in thread pool (asyncio.to_thread)"
    - "Face similarity validation with retry and threshold loosening"
key_files:
  created:
    - path: "backend/vidpipe/services/clean_sheet_service.py"
      exports: ["generate_clean_sheet", "generate_tier2_clean_sheet", "generate_tier3_clean_sheet", "compute_face_similarity"]
  modified:
    - path: "backend/vidpipe/pipeline/video_gen.py"
      changes: "Added reference_images parameter, 8s duration override, scene manifest loading, reference selection, clean sheet override checking"
decisions:
  - id: "08-02-01"
    choice: "Duration forced to 8 seconds when reference_images attached (Veo 3.1 API constraint)"
    rationale: "Veo API requires 8s duration when using reference images - this is non-negotiable"
    alternatives_rejected:
      - "Allow variable duration (API would reject request)"
  - id: "08-02-02"
    choice: "Reference images passed on ALL safety escalation levels (0, 1, 2)"
    rationale: "Identity references are independent of content-policy safety prefixes"
    alternatives_rejected:
      - "Remove references on escalation (loses character consistency)"
  - id: "08-02-03"
    choice: "Tier 3 face validation with 3 attempts and threshold loosening (0.6 → 0.5)"
    rationale: "Balances quality control with success rate - strict threshold first, loosen on final attempt"
    alternatives_rejected:
      - "Single attempt with strict threshold (too many failures)"
      - "No validation (quality risk for character preservation)"
metrics:
  duration: 4.5
  completed_date: "2026-02-17"
  task_count: 2
  file_count: 2
  commits:
    - hash: "1bcbf11"
      message: "feat(08-02): enhance video_gen with reference image passthrough and 8s duration enforcement"
    - hash: "06cdb98"
      message: "feat(08-02): implement clean sheet generation service with Tier 2/3"
---

# Phase 08 Plan 02: Veo Reference Passthrough and Clean Sheet Generation Summary

**One-liner:** Video generation pipeline passes up to 3 reference images to Veo 3.1 for identity preservation, with clean sheet generation (Tier 2 rembg, Tier 3 Gemini Image) for reference quality optimization.

## What Was Built

**Video Generation Pipeline Enhancement:**
- **Reference selection integration**: `_generate_video_for_scene` loads scene manifest, calls `select_references_for_scene()`, gets up to 3 assets
- **Clean sheet override**: Uses `get_primary_clean_reference()` to check if Tier 2/3 clean sheet exists, prefers clean_image_url over reference_image_url
- **Veo API passthrough**: Builds `VideoGenerationReferenceImage` list with `reference_type=ASSET`, passes to `_submit_video_job`
- **8-second duration enforcement**: Overrides `project.target_clip_duration` to 8 when references attached (Veo API constraint)
- **Selected tags persistence**: Stores selected reference tags in `SceneManifest.selected_reference_tags` for debugging/UI
- **Backward compatibility**: Non-manifest projects (manifest_id=None) skip reference selection entirely

**Clean Sheet Service (Tier 2):**
- **Background removal**: Uses `rembg.remove()` in thread pool (CPU-bound)
- **Gray background**: Converts RGBA to RGB with #808080 neutral gray
- **Storage**: `tmp/manifests/{manifest_id}/clean_sheets/tier2_{asset_id}.png`
- **Database record**: Creates `AssetCleanReference` with tier="tier2_rembg", is_primary=True, generation_cost=0.0
- **Graceful degradation**: Returns None if rembg not installed (optional dependency)

**Clean Sheet Service (Tier 3):**
- **Gemini Image generation**: Uses reference_images parameter with conditioning prompt
- **Face similarity validation**: For CHARACTER assets with face_embedding
  - Extracts face from generated clean sheet using insightface
  - Computes cosine similarity against stored embedding
  - 3 retry attempts with threshold: 0.6 → 0.6 → 0.5 (loosens on final attempt)
  - Fails if all attempts below threshold
- **Storage**: `tmp/manifests/{manifest_id}/clean_sheets/tier3_{asset_id}.png`
- **Database record**: Creates `AssetCleanReference` with tier="tier3_gemini", face_similarity_score, generation_cost=0.03
- **Non-CHARACTER handling**: Skips validation for ENVIRONMENT/PROP/VEHICLE

## Deviations from Plan

None - plan executed exactly as written. All specifications followed precisely.

## Key Decisions Made

**1. Duration forced to 8 seconds with references (Decision 08-02-01)**
- **What:** Override `duration_seconds` to 8 when `reference_images` is non-empty
- **Why:** Veo 3.1 API constraint - must use 8s duration when passing references
- **Impact:** Videos with references are always 8s regardless of project.target_clip_duration

**2. References passed on all safety levels (Decision 08-02-02)**
- **What:** `veo_ref_images` built once before escalation loop, passed to all attempts
- **Why:** Identity references are independent of content-policy safety prefixes
- **Impact:** Character consistency maintained even when safety prompts escalate

**3. Tier 3 validation with retry and threshold loosening (Decision 08-02-03)**
- **What:** 3 attempts with thresholds 0.6, 0.6, 0.5 for face similarity
- **Why:** Balance quality (strict threshold) with success rate (loosen on final attempt)
- **Impact:** Higher success rate while maintaining quality standards

## Implementation Notes

**Reference image flow:**
```
Scene manifest → select_references_for_scene() → [Asset, Asset, Asset]
                                                      ↓
                                            get_primary_clean_reference()
                                                      ↓
                                   clean_sheet_url OR reference_image_url
                                                      ↓
                                            Path.read_bytes()
                                                      ↓
                                   VideoGenerationReferenceImage(
                                       reference_image=Image(bytes),
                                       reference_type=ASSET
                                   )
                                                      ↓
                                   _submit_video_job(reference_images=[...])
```

**Duration override logic:**
```python
duration_seconds = 8 if reference_images else project.target_clip_duration
logger.info(f"Duration overridden to 8s (reference images attached)")
```

**Tier 2 clean sheet process:**
1. Load original image → rembg.remove() (thread pool)
2. Convert RGBA to RGB with gray background
3. Save to tmp/manifests/{manifest_id}/clean_sheets/tier2_{asset_id}.png
4. Create AssetCleanReference record (is_primary=True, cost=0.0)

**Tier 3 clean sheet process:**
1. Build conditioning prompt with clean sheet directives
2. Call Gemini Image with reference_images=[original]
3. For CHARACTER with face_embedding: validate similarity (3 attempts)
4. Save best result to tier3_{asset_id}.png
5. Create AssetCleanReference record (similarity_score, cost=0.03)

## Testing Performed

**Module imports:**
- ✓ video_gen.py loads without error
- ✓ clean_sheet_service.py loads without error
- ✓ All 4 functions importable (generate_clean_sheet, tier2, tier3, compute_face_similarity)

**Code verification:**
- ✓ reference_images parameter exists in _submit_video_job signature
- ✓ 5 occurrences of "reference_images" in video_gen.py
- ✓ Duration override to 8 seconds when references attached
- ✓ selected_reference_tags persisted to SceneManifest
- ✓ reference_type=VideoGenerationReferenceType.ASSET used
- ✓ select_references_for_scene called in _generate_video_for_scene
- ✓ 7 occurrences of AssetCleanReference in clean_sheet_service
- ✓ asyncio.to_thread used for CPU-bound work (rembg, face similarity)
- ✓ is_primary=True set for all clean sheets

**Backward compatibility:**
- ✓ Non-manifest projects: `if project.manifest_id:` guard ensures no reference selection
- ✓ selected_refs=[] when no manifest → veo_ref_images=None → original behavior preserved

## Artifacts Created

**Files created:**
- `backend/vidpipe/services/clean_sheet_service.py` (383 lines)

**Files modified:**
- `backend/vidpipe/pipeline/video_gen.py` (+76 lines)

**New imports in video_gen.py:**
- `from vidpipe.services import manifest_service`
- `from vidpipe.db.models import SceneManifest as SceneManifestModel`
- `from vidpipe.services.reference_selection import select_references_for_scene, get_primary_clean_reference`

**New API usage:**
- `types.VideoGenerationReferenceImage`
- `types.VideoGenerationReferenceType.ASSET`
- `video_config.reference_images = reference_images`

## Next Steps (Phase 8 Plan 03)

**Frontend integration:**
1. Display selected_reference_tags in scene manifest UI
2. Show clean sheet preview in asset detail view
3. Allow manual tier selection (Tier 2 vs Tier 3)
4. Display face_similarity_score for validation transparency

**Operational tasks:**
1. Install rembg: `pip install rembg`
2. Install insightface: `pip install insightface`
3. Test Tier 2 generation on existing manifest
4. Test Tier 3 generation with face validation
5. Verify Veo 3.1 accepts reference_images parameter

## Self-Check: PASSED

**Files created:**
```
FOUND: backend/vidpipe/services/clean_sheet_service.py
```

**Files modified:**
```
FOUND: backend/vidpipe/pipeline/video_gen.py
```

**Commits exist:**
```
FOUND: 1bcbf11 (feat(08-02): enhance video_gen with reference image passthrough and 8s duration enforcement)
FOUND: 06cdb98 (feat(08-02): implement clean sheet generation service with Tier 2/3)
```

**Module verification:**
```
video_gen loads: OK
clean_sheet_service loads: OK
reference_images parameter: OK (5 occurrences)
duration_seconds = 8: OK (2 occurrences)
reference_type=ASSET: OK
select_references_for_scene: OK
AssetCleanReference: OK (7 occurrences)
asyncio.to_thread: OK (2 occurrences for rembg and face similarity)
is_primary=True: OK (2 occurrences)
```

All claimed artifacts verified and present in repository.
