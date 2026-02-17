---
phase: 11-multi-candidate-quality-mode
plan: 01
subsystem: database
tags: [sqlalchemy, pydantic, fastapi, clip, cv-analysis, gemini-flash, scoring]

# Dependency graph
requires:
  - phase: 09-cv-analysis-pipeline
    provides: CVAnalysisService.analyze_generated_content and CLIPEmbeddingService.compute_similarity
  - phase: 10-adaptive-prompt-rewriting
    provides: rewritten_video_prompt stored on SceneManifest

provides:
  - GenerationCandidate ORM model with score columns, selection state, and cost tracking
  - Project.quality_mode and Project.candidate_count columns with safe defaults
  - CandidateScoringService with score_candidate and score_all_candidates methods
  - SCORE_WEIGHTS constant (manifest_adherence=0.35, visual_quality=0.25, continuity=0.25, prompt_adherence=0.15)

affects:
  - 11-02 (pipeline integration will use CandidateScoringService and GenerationCandidate)
  - 11-03 (API/UI will read quality_mode and candidate_count from ProjectDetail)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy-loaded child services via _get_X() getters (matches Phase 9 CVAnalysisService pattern)"
    - "cv2 imported inside frame extractor functions (Phase 9 convention, avoids ImportError)"
    - "Module-level helper functions for frame extraction (not static methods)"
    - "asyncio.Semaphore(3) for concurrent Gemini call rate limiting"
    - "Batched Gemini Flash call for visual_quality + prompt_adherence (single LLM call for two scores)"

key-files:
  created:
    - backend/vidpipe/services/candidate_scoring.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/api/routes.py

key-decisions:
  - "GenerationCandidate placed after VideoClip in models.py — logically related (video output artifacts)"
  - "Composite index idx_candidates_project_scene on (project_id, scene_index) for efficient per-scene queries"
  - "Both quality_mode and candidate_count have default values (False/1) so existing projects load correctly without migration"
  - "Gemini visual_quality and prompt_adherence batched into single Flash call to minimize cost and latency"
  - "Scene 0 continuity auto-scores 10.0 (no prior scene to compare against)"
  - "Scoring failures use neutral 5.0 fallback — never escalate to pipeline failure (graceful degradation matches Phase 9 pattern)"
  - "Candidate_count clamped to 1 when quality_mode=False to prevent accidental multi-generation"

patterns-established:
  - "Four-dimension weighted scoring: manifest_adherence=0.35, visual_quality=0.25, continuity=0.25, prompt_adherence=0.15"

# Metrics
duration: 3min
completed: 2026-02-16
---

# Phase 11 Plan 01: Multi-Candidate Quality Mode — Data Layer and Scoring Engine Summary

**GenerationCandidate ORM model with four-dimension composite scoring via CVAnalysisService, CLIP embeddings, and batched Gemini Flash call**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-16T09:59:50Z
- **Completed:** 2026-02-16T10:02:33Z
- **Tasks:** 2
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- GenerationCandidate model with all score columns (manifest_adherence, visual_quality, continuity, prompt_adherence, composite), selection state (is_selected, selected_by), cost tracking, and composite index on (project_id, scene_index)
- Project extended with quality_mode (bool, default False) and candidate_count (int, default 1) — safe defaults for all existing projects
- GenerateRequest / ProjectDetail / ProjectListItem schemas extended with quality mode fields and validation (candidate_count 1-4; quality_mode requires >=2)
- CandidateScoringService with four weighted dimensions, parallel scoring via asyncio.gather() + Semaphore(3), and graceful fallbacks

## Task Commits

Each task was committed atomically:

1. **Task 1: GenerationCandidate model and Project quality columns** - `1ea06db` (feat)
2. **Task 2: CandidateScoringService with four-dimension composite scoring** - `bc95301` (feat)

## Files Created/Modified

- `backend/vidpipe/db/models.py` — Added GenerationCandidate ORM model, Index import, quality_mode + candidate_count columns on Project
- `backend/vidpipe/api/routes.py` — Extended GenerateRequest/ProjectDetail/ProjectListItem schemas; added validation and Project creation fields
- `backend/vidpipe/services/candidate_scoring.py` — New: CandidateScoringService, SCORE_WEIGHTS, frame extraction helpers

## Decisions Made

- Batched Gemini Flash call for visual_quality + prompt_adherence avoids two separate API calls per candidate
- cv2 imported inside _extract_first_frame/_extract_last_frame per Phase 9 convention (avoids ImportError when opencv not installed)
- Scoring failures use neutral 5.0 fallback and WARNING log — never escalate to pipeline failure (graceful degradation)
- candidate_count forced to 1 when quality_mode=False to prevent accidental multi-generation even if caller passes candidate_count > 1

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - direct import verification required bypassing db/__init__.py (which loads full config), used importlib.util.spec_from_file_location for isolated model testing.

## Next Phase Readiness

- GenerationCandidate model is ready for Plan 02 to use when storing per-candidate clips and scores
- CandidateScoringService.score_all_candidates is ready for pipeline integration (Plan 02)
- ProjectDetail now exposes quality_mode and candidate_count for UI (Plan 03)
- No blockers

---
*Phase: 11-multi-candidate-quality-mode*
*Completed: 2026-02-16*

## Self-Check: PASSED

- FOUND: backend/vidpipe/db/models.py
- FOUND: backend/vidpipe/api/routes.py
- FOUND: backend/vidpipe/services/candidate_scoring.py
- FOUND: .planning/phases/11-multi-candidate-quality-mode/11-01-SUMMARY.md
- FOUND: commit 1ea06db (Task 1)
- FOUND: commit bc95301 (Task 2)
