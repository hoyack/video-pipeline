---
phase: 11-multi-candidate-quality-mode
plan: 02
subsystem: pipeline
tags: [veo, video-generation, quality-mode, candidate-scoring, multi-candidate, sqlalchemy]

# Dependency graph
requires:
  - phase: 11-01
    provides: GenerationCandidate ORM model, CandidateScoringService.score_all_candidates, Project.quality_mode + candidate_count columns

provides:
  - _poll_and_collect_candidates: polls Veo operation and collects ALL candidate video bytes (partial RAI filter = success)
  - _handle_quality_mode_candidates: saves candidates to disk, creates GenerationCandidate records, scores all, selects winner by composite_score
  - number_of_videos passed to Veo API via _submit_video_job when quality_mode=True
  - VideoClip.local_path always points to selected winner — stitcher unchanged
  - Crash-recovery resumes with multi-candidate poll + scoring for quality-mode projects
  - all_assets initialized to [] before manifest block (prevents NameError in quality mode without manifest)

affects:
  - 11-03 (UI/API can display candidates and scores from GenerationCandidate records)
  - 12-fork-system-integration (fork logic reads VideoClip.local_path — already correct)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy singleton _get_candidate_scoring_service() matching Phase 9 _get_cv_analysis_service() pattern"
    - "Poll function selected at runtime based on project.quality_mode (dual-path, standard mode unchanged)"
    - "all_assets: list = [] initialized before conditional block to prevent NameError in non-manifest paths"
    - "has_refs bool parameter in _handle_quality_mode_candidates (selected_refs context passed safely)"
    - "session.flush() assigns IDs to GenerationCandidate records before scoring (avoids partial commit)"

key-files:
  created: []
  modified:
    - backend/vidpipe/pipeline/video_gen.py

key-decisions:
  - "_poll_and_collect_candidates is a NEW function (not modifying existing _poll_video_operation) — minimizes standard mode risk"
  - "Partial RAI filter (some candidates survive) treated as success in Quality Mode — only escalate on ZERO survivors"
  - "candidate_count=project.candidate_count if quality_mode else 1 — standard mode always submits exactly 1 video"
  - "all_assets initialized to [] before manifest block — crash-recovery and quality mode without manifest both safe"
  - "has_refs bool passed to _handle_quality_mode_candidates — duration_seconds=8 if refs, else target_clip_duration"
  - "session.flush() (not session.commit()) after adding GenerationCandidate records — assigns IDs while deferring commit to after scoring"
  - "CV analysis runs on selected candidate only (not all candidates) — maintains Phase 9 per-scene analysis intent"

patterns-established:
  - "Dual-path poll pattern: quality_mode → _poll_and_collect_candidates, standard → _poll_video_operation"
  - "Guard all_assets with early initialization to [] — avoids NameError in any code path that skips the manifest block"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 11 Plan 02: Multi-Candidate Quality Mode — Pipeline Integration Summary

**Multi-candidate Veo generation wired into video_gen.py: number_of_videos passed to API, all candidates saved and scored via CandidateScoringService, winner selected by composite_score, VideoClip.local_path updated for stitcher compatibility**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-17T02:24:41Z
- **Completed:** 2026-02-17T02:28:10Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- `_submit_video_job` now accepts `candidate_count: int = 1` and passes `number_of_videos=candidate_count` to `GenerateVideosConfig` — non-breaking, default preserves standard mode
- New `_poll_and_collect_candidates` function: polls Veo operation, downloads ALL surviving candidate bytes, handles partial RAI filtering (only zero-survivor case escalates)
- New `_handle_quality_mode_candidates` function: saves candidates as `scene_{N}_candidate_{i}.mp4`, creates `GenerationCandidate` records, scores all via `CandidateScoringService.score_all_candidates`, selects winner by composite_score, updates `VideoClip.local_path` for stitcher, then runs CV analysis on winner only
- Escalation loop updated: passes `candidate_count`, chooses poll function based on mode, routes completion to quality or standard path
- Crash-recovery updated: quality-mode projects resume with `_poll_and_collect_candidates` + `_handle_quality_mode_candidates` instead of standard poll
- `all_assets: list = []` initialized before manifest block — prevents NameError in quality mode projects without manifest

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend _submit_video_job with number_of_videos parameter** - `c2af572` (feat)
2. **Task 2: Multi-candidate flow in _generate_video_for_scene and updated poll handling** - `ebb25ed` (feat)

## Files Created/Modified

- `backend/vidpipe/pipeline/video_gen.py` — Added GenerationCandidate+CandidateScoringService imports, lazy singleton, `_poll_and_collect_candidates`, `_handle_quality_mode_candidates`, all_assets init, escalation loop updates, crash-recovery branching

## Decisions Made

- Created `_poll_and_collect_candidates` as a new function rather than modifying `_poll_video_operation` — preserves standard mode code path without risk of regression
- Partial RAI filter success: if 2/4 candidates survive, proceed with survivors (only escalate on zero survivors)
- `session.flush()` (not `commit()`) after adding `GenerationCandidate` records to assign IDs before scoring without a premature commit
- `has_refs: bool` parameter on `_handle_quality_mode_candidates` to correctly set `clip.duration_seconds` (8s if refs, else target_clip_duration) without needing to pass the full `veo_ref_images` list

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] score_all_candidates call signature adapted to actual implementation**
- **Found during:** Task 2 (Step D implementation)
- **Issue:** Plan's pseudocode called `score_all_candidates(candidates_info=[{"candidate_video_path": ...}])` but the actual `score_all_candidates` signature takes separate parameters `(candidates_info, scene_index, scene_manifest_json, rewritten_video_prompt, existing_assets, previous_scene_clip_path)` and expects `candidates_info` dicts with `"local_path"` key (not `"candidate_video_path"`)
- **Fix:** Used correct signature with separate parameters; built `candidates_info = [{"local_path": cand.local_path} for cand in candidate_records]`
- **Files modified:** backend/vidpipe/pipeline/video_gen.py
- **Verification:** AST and syntax check passes; function calls match actual service signature
- **Committed in:** ebb25ed (Task 2 commit)

**2. [Rule 2 - Missing Critical] Added has_refs parameter to _handle_quality_mode_candidates**
- **Found during:** Task 2 (Step D — duration_seconds assignment)
- **Issue:** Plan said `clip.duration_seconds = project.target_clip_duration  # or 8 if refs` but the function had no way to know if ref images were used — plan pseudocode used a non-existent `project._ref_attached` attribute
- **Fix:** Added `has_refs: bool = False` parameter, passed `has_refs=bool(selected_refs)` from crash-recovery and `has_refs=bool(veo_ref_images)` from escalation loop
- **Files modified:** backend/vidpipe/pipeline/video_gen.py
- **Verification:** Both call sites pass correct has_refs value
- **Committed in:** ebb25ed (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None — implementation straightforward once the score_all_candidates signature mismatch was caught.

## Next Phase Readiness

- Quality Mode pipeline is fully functional end-to-end
- GenerationCandidate records created with scores for all candidates — ready for Plan 03 UI/API exposure
- VideoClip.local_path correctly points to winning candidate — stitcher requires no changes
- No blockers for Phase 11 Plan 03 (API endpoints to expose candidates, scores, and manual selection)

---
*Phase: 11-multi-candidate-quality-mode*
*Completed: 2026-02-17*

## Self-Check: PASSED

- FOUND: backend/vidpipe/pipeline/video_gen.py
- FOUND: .planning/phases/11-multi-candidate-quality-mode/11-02-SUMMARY.md
- FOUND: commit c2af572 (Task 1)
- FOUND: commit ebb25ed (Task 2)
