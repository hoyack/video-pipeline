---
phase: 09-cv-analysis-pipeline-and-progressive-enrichment
plan: 03
subsystem: pipeline
tags: [cv-analysis, yolo, arcface, clip, progressive-enrichment, video-gen, orchestrator]

# Dependency graph
requires:
  - phase: 09-01
    provides: frame_sampler, CVAnalysisResult, CLIPEmbeddingService
  - phase: 09-02
    provides: CVAnalysisService.analyze_generated_content, track_appearances, identify_new_entities, extract_and_register_new_entities

provides:
  - Per-scene CV analysis hook wired into video generation loop (_run_post_generation_analysis)
  - Progressive enrichment: assets from scene N in registry before scene N+1
  - SceneManifest.cv_analysis_json populated after each scene
  - SceneManifest.continuity_score set from semantic analysis
  - AssetAppearance records created for each matched face in each scene
  - Graceful degradation: CV failure never fails the pipeline
  - Backward compatibility: non-manifest projects skip analysis entirely

affects: [10-adaptive-prompt-rewriting, 11-multi-candidate-quality-mode, 12-fork-system-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy singleton for heavyweight services (_get_cv_analysis_service)"
    - "try/except wrapping of optional pipeline steps for graceful degradation"
    - "scene_manifest_row initialized to None before conditional block for dual-path access"

key-files:
  created: []
  modified:
    - backend/vidpipe/pipeline/video_gen.py
    - backend/vidpipe/orchestrator/pipeline.py

key-decisions:
  - "CV analysis runs inline per-scene (not batch at end) to enable progressive enrichment before next scene"
  - "scene_manifest_row initialized to None before manifest_id guard so both crash-recovery and escalation paths can access it"
  - "clip_embeddings excluded from cv_analysis_json persistence (model_dump exclude={clip_embeddings}) to avoid large binary in JSON"
  - "CV analysis failure wrapped in try/except — never escalates to pipeline failure (graceful degradation)"

patterns-established:
  - "Post-generation hook pattern: _run_post_generation_analysis() called at EVERY successful poll_result == complete return"
  - "Non-manifest project guard: if not project.manifest_id: return immediately"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 9 Plan 03: CV Analysis Pipeline Integration Summary

**Per-scene CV analysis (YOLO + ArcFace + CLIP + Gemini) wired into video_gen loop with progressive asset enrichment — each scene's extracted assets feed into the next scene's reference selection**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-17T01:18:48Z
- **Completed:** 2026-02-17T01:20:58Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `_run_post_generation_analysis()` to video_gen.py — runs after EACH scene's clip completes
- CV analysis runs before next scene starts, enabling progressive enrichment (assets from scene N in registry for scene N+1)
- SceneManifest.cv_analysis_json and continuity_score populated after each successful scene
- AssetAppearance records created for face-matched assets per scene
- New entities (unmatched detections) extracted and registered into Asset Registry
- Non-manifest projects skip analysis entirely (backward compatible)
- CV failures wrap in try/except — never propagate to pipeline failure
- Orchestrator updated: progress message, step_log comment, docstring, AssetAppearance import

## Task Commits

Each task was committed atomically:

1. **Task 1: Per-scene CV analysis hook in video generation pipeline** - `942c66d` (feat)
2. **Task 2: Pipeline orchestrator analysis step logging** - `a42872f` (feat)

## Files Created/Modified

- `backend/vidpipe/pipeline/video_gen.py` - Added CVAnalysisService/entity_extraction imports, lazy singleton, _run_post_generation_analysis() function, CV analysis calls at both completion code paths
- `backend/vidpipe/orchestrator/pipeline.py` - Added AssetAppearance import, updated progress callback message, added step_log comment, updated _check_completed_steps docstring

## Decisions Made

- CV analysis runs inline per-scene (not batch at end) to enable progressive enrichment before next scene generates
- `scene_manifest_row` initialized to `None` before the `if project.manifest_id:` block so both the crash-recovery resume path and the escalation loop completion path can access it without re-querying
- `clip_embeddings` excluded from `cv_analysis_json` persistence via `model_dump(exclude={"clip_embeddings"})` to avoid storing large binary data as JSON
- CV analysis failure wrapped in `try/except Exception` — never escalates to pipeline failure (graceful degradation with warning log)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all changes applied cleanly to existing code.

## Next Phase Readiness

- CV analysis pipeline fully integrated; analysis results in scene_manifests.cv_analysis_json
- Asset Registry enriched progressively as pipeline runs each scene
- Phase 10 (adaptive prompt rewriting) can read cv_analysis_json continuity scores and new entity data to adapt prompts
- Phase 11 (multi-candidate quality mode) can use continuity_score for candidate ranking

---
*Phase: 09-cv-analysis-pipeline-and-progressive-enrichment*
*Completed: 2026-02-17*
