---
phase: 05-manifesting-engine
plan: 01
subsystem: services
tags: [yolo, insightface, arcface, gemini-vision, cv, face-recognition, reverse-prompting]

# Dependency graph
requires:
  - phase: 04-manifest-system-foundation
    provides: Asset model, manifest CRUD operations, manifest_service
provides:
  - Extended Asset model with 9 Phase 5 fields for CV analysis
  - CVDetectionService for YOLO object/face detection
  - FaceMatchingService for ArcFace embeddings and cross-matching
  - ReversePromptService for Gemini vision reverse-prompting
affects: [05-02-manifesting-engine-orchestrator, 05-03-ui-processing-progress]

# Tech tracking
tech-stack:
  added: [ultralytics (YOLO), insightface (ArcFace), Pillow (cropping), numpy (embeddings)]
  patterns: [lazy model loading, RuntimeError with troubleshooting guidance, normalized embeddings, type-specific prompts]

key-files:
  created:
    - backend/vidpipe/services/cv_detection.py
    - backend/vidpipe/services/face_matching.py
    - backend/vidpipe/services/reverse_prompt_service.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/services/manifest_service.py

key-decisions:
  - "Store face embeddings as bytes (numpy.tobytes()) not JSON for 10x storage reduction"
  - "Use yolov8m.pt (medium model) for balance of speed and accuracy"
  - "Extract faces from person detections (upper 40% of bbox) until dedicated face model added"
  - "Use gemini-2.0-flash-exp for reverse-prompting (speed over accuracy for 20+ crops)"
  - "Lazy-load all CV models to avoid import-time overhead and allow graceful failure"
  - "Add VEHICLE asset type with VEH prefix for automotive content"

patterns-established:
  - "Lazy model loading pattern: Load expensive models (_load_models) on first use, not at import or __init__"
  - "Clear error messages: RuntimeError with troubleshooting guidance for model download failures (network, disk space, CUDA)"
  - "Normalized embeddings: Store embeddings as unit vectors so cosine similarity = dot product"
  - "Type-specific system prompts: Different reverse-prompting strategies for CHARACTER vs OBJECT vs ENVIRONMENT"

# Metrics
duration: 3.6min
completed: 2026-02-16
---

# Phase 05 Plan 01: Core Services Summary

**Extended Asset model with CV fields and created YOLO detection, ArcFace face matching, and Gemini reverse-prompting services with lazy loading**

## Performance

- **Duration:** 3.6 min (215 seconds)
- **Started:** 2026-02-16T20:59:16Z
- **Completed:** 2026-02-16T21:02:51Z
- **Tasks:** 2
- **Files modified:** 6
- **Commits:** 2

## Accomplishments

- Extended Asset model with 9 new Phase 5 fields for CV analysis (reverse_prompt, visual_description, detection_class, detection_confidence, is_face_crop, crop_bbox, face_embedding, quality_score, source_asset_id)
- Created CVDetectionService with YOLO wrapper for object/face detection with lazy model loading and error handling
- Created FaceMatchingService with ArcFace embedding generation and cosine similarity-based cross-matching
- Created ReversePromptService with Gemini vision API integration and type-specific system prompts
- Added VEHICLE asset type support with VEH prefix
- Implemented idempotent database migrations for all new columns

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Phase 5 fields to Asset model and migrate schema** - `89ced52` (feat)
2. **Task 2: Create CV detection, face matching, and reverse-prompt services** - `9ebb3b8` (feat)

## Files Created/Modified

**Created:**
- `backend/vidpipe/services/cv_detection.py` - YOLO-based object and face detection with lazy model loading, bbox extraction, and image cropping
- `backend/vidpipe/services/face_matching.py` - ArcFace embedding generation, cross-matching via cosine similarity, and face grouping
- `backend/vidpipe/services/reverse_prompt_service.py` - Gemini vision API wrapper with asset-type-specific system prompts for reverse-prompting

**Modified:**
- `backend/vidpipe/db/models.py` - Added 9 Phase 5 columns to Asset model
- `backend/vidpipe/db/__init__.py` - Added idempotent ALTER TABLE migrations for Phase 5 fields
- `backend/vidpipe/services/manifest_service.py` - Added VEHICLE to VALID_ASSET_TYPES and TAG_PREFIX_MAP

## Decisions Made

1. **Lazy model loading pattern**: All CV models (YOLO, ArcFace) load on first method call, not at import or __init__. This avoids import-time overhead (~5-10 seconds) and allows services to be imported even if dependencies aren't installed.

2. **Clear error handling**: Model loading failures raise RuntimeError with actionable troubleshooting guidance (network connectivity for downloads, disk space for models, CUDA availability for GPU inference).

3. **Normalized embeddings**: FaceMatchingService normalizes embeddings to unit vectors on generation, so cosine similarity computation simplifies to dot product. This follows insightface best practices and reduces storage precision requirements.

4. **Type-specific prompts**: ReversePromptService uses different system prompts for CHARACTER (focus on appearance, clothing), OBJECT (material, condition), and ENVIRONMENT (lighting, mood, atmosphere). This maximizes Gemini's context-aware output quality.

5. **Face extraction strategy**: Until dedicated face detection model is added, extract faces from YOLO "person" detections by cropping upper 40% of person bounding box. This is a practical heuristic for portrait-oriented uploads.

6. **VEHICLE asset type**: Added VEHICLE to supported types with VEH prefix to handle automotive content (cars, bikes, drones) distinct from generic OBJECT.

## Deviations from Plan

None - plan executed exactly as written. All service modules created with specified interfaces, lazy loading, and error handling patterns per research doc recommendations.

## Issues Encountered

None - all services imported successfully, cosine similarity tests passed with known vectors, and database migrations ran idempotently.

## User Setup Required

None - no external service configuration required. Services use existing Vertex AI client from Phase 2. CV dependencies (ultralytics, insightface, onnxruntime-gpu) will auto-download models on first use.

## Next Phase Readiness

Ready for Plan 02 (Manifesting Engine Orchestrator):
- Asset model has all required fields for storing CV analysis results
- Three core services are importable and tested (import verification, cosine similarity tests)
- Lazy loading ensures orchestrator can compose services without immediate model initialization
- VEHICLE type support enables automotive manifest processing

**Dependencies needed for runtime**: ultralytics, insightface, onnxruntime-gpu, opencv-python. These are optional for import but required for actual CV processing.

## Self-Check: PASSED

All files created and all commits exist in git history:
- ✓ backend/vidpipe/services/cv_detection.py
- ✓ backend/vidpipe/services/face_matching.py
- ✓ backend/vidpipe/services/reverse_prompt_service.py
- ✓ 89ced52 (Task 1 commit)
- ✓ 9ebb3b8 (Task 2 commit)

---
*Phase: 05-manifesting-engine*
*Completed: 2026-02-16*
