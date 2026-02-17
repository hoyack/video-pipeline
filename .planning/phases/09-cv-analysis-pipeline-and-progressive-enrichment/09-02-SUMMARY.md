---
phase: 09-cv-analysis-pipeline-and-progressive-enrichment
plan: 02
subsystem: cv-analysis
tags: [yolo, arcface, clip, gemini-vision, pydantic, async, entity-extraction]

# Dependency graph
requires:
  - phase: 09-01
    provides: CLIPEmbeddingService, FrameSampler, AssetAppearance model, CVAnalysisConfig
  - phase: 05-manifesting-engine
    provides: CVDetectionService, FaceMatchingService, ReversePromptService, Asset model

provides:
  - CVAnalysisService: post-generation orchestrator composing YOLO + ArcFace + CLIP + Gemini Vision
  - CVAnalysisResult: structured Pydantic model with all analysis data
  - entity_extraction module: identify_new_entities() + extract_and_register_new_entities()
  - AssetAppearance persistence via track_appearances()

affects:
  - 09-03 (pipeline orchestrator integration — consumes CVAnalysisService)
  - 10-adaptive-prompt-rewriting (uses continuity_score + semantic_analysis)
  - 12-fork-system-integration (uses AssetAppearance records for timeline)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Lazy service instantiation in orchestrator (no constructor args, _get_X() methods)
    - asyncio.to_thread() for ALL CPU-bound inference (YOLO, ArcFace, CLIP)
    - asyncio.Semaphore(3) for rate-limiting concurrent Gemini API calls
    - Optional Gemini Vision semantic analysis — fails gracefully, rest of pipeline continues
    - Quality gate pattern: quality_score < threshold → skip, log at INFO level
    - Caller manages DB transactions (no commit inside service functions)

key-files:
  created:
    - backend/vidpipe/services/cv_analysis_service.py
    - backend/vidpipe/services/entity_extraction.py
  modified: []

key-decisions:
  - "CVAnalysisService uses lazy _get_X() getters for child services to avoid import-time model loading"
  - "Semantic analysis is OPTIONAL — Gemini Vision failure does not fail the overall analysis"
  - "asyncio.to_thread() wraps all CPU-bound inference (YOLO, ArcFace, CLIP) to avoid event loop blocking"
  - "Face matching uses cosine similarity on numpy float32 embeddings deserialized via np.frombuffer"
  - "extract_and_register_new_entities uses asyncio.Semaphore(3) for Gemini rate-limiting (reduced from Phase 5's 5)"
  - "Extracted assets do NOT auto-add to scene manifests — Asset Registry only (intent vs validation)"
  - "IoU > 0.70 deduplication threshold for overlapping entity detections"

patterns-established:
  - "Orchestrator lazy-loads child services: _get_cv_service(), _get_face_service(), _get_clip_service()"
  - "Quality gate: skip registration below threshold, log with score and threshold values"
  - "CLIP embeddings stored as bytes (emb.tobytes()) matching face_embedding pattern for storage efficiency"

# Metrics
duration: 6min
completed: 2026-02-16
---

# Phase 9 Plan 02: CV Analysis Orchestrator and Entity Extraction Summary

**Post-generation CV analysis pipeline composing YOLO + ArcFace + CLIP + Gemini Vision into CVAnalysisService, with entity extraction and quality-gated asset registration.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-16T00:33:54Z
- **Completed:** 2026-02-16T00:39:54Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- CVAnalysisService.analyze_generated_content() orchestrates a 6-step analysis pipeline: frame extraction (from clip or keyframes), YOLO detection, ArcFace face matching against Asset Registry, CLIP embeddings, Gemini Vision semantic analysis, and continuity score computation
- entity_extraction module with identify_new_entities() (IoU deduplication, YOLO class mapping) and extract_and_register_new_entities() (quality gate, reverse-prompting, ArcFace + CLIP embeddings, manifest_tag auto-generation)
- track_appearances() persists matched face detections as AssetAppearance records for timeline tracking

## Task Commits

Each task was committed atomically:

1. **Task 1: CV analysis orchestrator service** - `28b59c4` (feat)
2. **Task 2: Entity extraction and registration service** - `dbbe2bc` (feat)

## Files Created/Modified

- `backend/vidpipe/services/cv_analysis_service.py` - CVAnalysisService orchestrator with all data models (FrameDetection, FaceMatchResult, SemanticAnalysis, CVAnalysisResult)
- `backend/vidpipe/services/entity_extraction.py` - Entity extraction with NewEntityDetection model, identify_new_entities(), extract_and_register_new_entities(), _yolo_class_to_asset_type(), _compute_iou()

## Decisions Made

- CVAnalysisService lazy-loads child services via `_get_X()` getter methods — avoids import-time model loading (60-200MB+ models) and allows graceful failure if a library isn't installed
- Semantic analysis is optional: Gemini Vision failure (network error, API error, missing frames) returns None and the rest of CVAnalysisResult remains valid with continuity_score=0.0
- All CPU-bound inference (YOLO predict, InsightFace get, CLIP get_image_features) wrapped in asyncio.to_thread() to avoid blocking the FastAPI event loop
- Face matching against Asset Registry: deserialize stored float32 bytes via np.frombuffer, compute cosine similarity with FaceMatchingService.cosine_similarity(), keep best match above face_match_threshold (default 0.6)
- Entity extraction uses Semaphore(3) rather than Phase 5's Semaphore(5) since extraction runs inline with generation (more conservative rate limiting)
- Extracted assets are added to Asset Registry only — NOT auto-added to scene manifests, preserving the design principle that manifests represent "intent" while CV analysis provides "validation"

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CVAnalysisService is ready for Plan 03 (pipeline orchestrator integration) to call after video_gen completes
- Both services import cleanly and all Pydantic models have the specified fields
- The analyze_generated_content() → track_appearances() → extract_and_register_new_entities() flow is ready to be wired into the pipeline orchestrator

---
*Phase: 09-cv-analysis-pipeline-and-progressive-enrichment*
*Completed: 2026-02-16*

## Self-Check: PASSED

- FOUND: backend/vidpipe/services/cv_analysis_service.py
- FOUND: backend/vidpipe/services/entity_extraction.py
- FOUND: .planning/phases/09-cv-analysis-pipeline-and-progressive-enrichment/09-02-SUMMARY.md
- FOUND: commit 28b59c4 (feat: CV analysis orchestrator)
- FOUND: commit dbbe2bc (feat: entity extraction service)
