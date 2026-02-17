---
phase: 11-multi-candidate-quality-mode
verified: 2026-02-17T02:36:45Z
status: passed
score: 15/15 must-haves verified
---

# Phase 11: Multi-Candidate Quality Mode Verification Report

**Phase Goal:** Users can generate 2-4 candidate clips per scene with composite quality scoring (manifest adherence, visual quality, continuity, prompt adherence) and select the best take manually or automatically
**Verified:** 2026-02-17T02:36:45Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | GenerationCandidate model stores per-candidate video paths, scores, and selection state | VERIFIED | `backend/vidpipe/db/models.py` lines 290-325: complete ORM model with all required columns, composite index |
| 2  | Project model has quality_mode and candidate_count columns with safe defaults | VERIFIED | `models.py` lines 171-173: `quality_mode: Mapped[bool] = mapped_column(Boolean, default=False)`, `candidate_count: Mapped[int] = mapped_column(Integer, default=1)` |
| 3  | CandidateScoringService computes weighted composite from four dimensions with correct weights | VERIFIED | `backend/vidpipe/services/candidate_scoring.py` lines 23-28: `SCORE_WEIGHTS = {"manifest_adherence": 0.35, "visual_quality": 0.25, "continuity": 0.25, "prompt_adherence": 0.15}` — assertion enforces sum=1.0 |
| 4  | Scoring uses CVAnalysisService, CLIPEmbeddingService, and batched Gemini Flash call | VERIFIED | Lines 236-242 (CVAnalysisService.analyze_generated_content), lines 291-306 (CLIPEmbeddingService.compute_similarity), lines 359-380 (Gemini Flash batched call for visual+prompt) |
| 5  | When quality_mode=True, Veo receives number_of_videos=candidate_count | VERIFIED | `backend/vidpipe/pipeline/video_gen.py` line 178: `number_of_videos=candidate_count` in GenerateVideosConfig; line 1011: `candidate_count = project.candidate_count if project.quality_mode else 1` |
| 6  | All returned candidates are saved as MP4 files and GenerationCandidate records | VERIFIED | `_handle_quality_mode_candidates` lines 523-534: saves each candidate to `scene_{N}_candidate_{i}.mp4`, creates GenerationCandidate record per candidate |
| 7  | CandidateScoringService scores all candidates; highest composite_score auto-selected | VERIFIED | Lines 557-586: `score_all_candidates` called, winner found via `max(..., key=lambda i: score_results[i].get("composite_score", 0.0))`, `is_selected=True` set on winner |
| 8  | Selected candidate's local_path is written to VideoClip.local_path for stitcher | VERIFIED | Line 595: `clip.local_path = candidate_records[winner_idx].local_path` |
| 9  | Standard mode behavior is completely unchanged (quality_mode=False) | VERIFIED | Dual-path pattern: lines 1071-1079 route to either `_poll_and_collect_candidates` (quality) or `_poll_video_operation` (standard); default `candidate_count=1` |
| 10 | RAI filtering only escalates if ZERO candidates survive | VERIFIED | `_poll_and_collect_candidates` lines 417-437: returns "content_policy" only when `len(generated_videos) == 0`; partial filter treated as success |
| 11 | Crash-recovery for quality mode uses multi-candidate poll and scoring | VERIFIED | Lines 908-929: `if project.quality_mode and project.candidate_count > 1` branch uses `_poll_and_collect_candidates` + `_handle_quality_mode_candidates` |
| 12 | GET /api/projects/{id}/scenes/{idx}/candidates returns candidates with scores | VERIFIED | `backend/vidpipe/api/routes.py` lines 1946-1977: `list_candidates` endpoint returns all GenerationCandidate records with full score breakdown as CandidateResponse |
| 13 | PUT candidates/{cid}/select deselects current winner, selects new one, updates VideoClip.local_path | VERIFIED | Lines 1980-2030: `select_candidate` deselects all (`c.is_selected = False`), selects chosen, sets `selected_by="user"`, updates `clip.local_path = chosen.local_path` |
| 14 | GenerateForm has Quality Mode toggle with 2/3/4x candidate count selector and cost multiplier | VERIFIED | `frontend/src/components/GenerateForm.tsx` lines 328-384: amber toggle, pills for [2,3,4], cost impact display using `qualityModeCostMultiplier` |
| 15 | User can click a candidate to manually override auto-selection; SceneCard fetches candidates when quality mode | VERIFIED | `frontend/src/components/SceneCard.tsx` lines 53-62 (lazy fetch on expand+qualityMode), lines 195-210 (click sends PUT selectCandidate, optimistic UI update) |

**Score:** 15/15 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/vidpipe/db/models.py` | GenerationCandidate ORM model + Project quality columns | VERIFIED | Lines 290-325 (GenerationCandidate), lines 171-173 (Project quality_mode + candidate_count) |
| `backend/vidpipe/services/candidate_scoring.py` | CandidateScoringService, SCORE_WEIGHTS, score_candidate, score_all_candidates | VERIFIED | All present and substantive: 446 lines, real scoring logic using CVAnalysisService, CLIP, Gemini Flash |
| `backend/vidpipe/pipeline/video_gen.py` | Multi-candidate generation flow, number_of_videos passed | VERIFIED | `_poll_and_collect_candidates`, `_handle_quality_mode_candidates`, `_submit_video_job` with `candidate_count`, dual-path escalation loop |
| `backend/vidpipe/api/routes.py` | list_candidates GET and select_candidate PUT endpoints | VERIFIED | Lines 1946-2030: both endpoints substantive, CandidateResponse schema defined at line 475 |
| `frontend/src/api/types.ts` | CandidateScore TypeScript interface | VERIFIED | Lines 2-15: complete interface with all score fields |
| `frontend/src/components/GenerateForm.tsx` | Quality Mode toggle with cost impact | VERIFIED | Lines 328-384: amber toggle, 2/3/4x pills, cost estimate display, quality_mode included in generateVideo request |
| `frontend/src/components/SceneCard.tsx` | CandidateComparison panel, projectId/qualityMode props | VERIFIED | Lines 29-39 (props), 53-62 (lazy fetch), 187-265 (candidate comparison grid) |
| `frontend/src/components/ProjectDetail.tsx` | Passes projectId and qualityMode to SceneCard | VERIFIED | Lines 242-249: `projectId={detail.project_id}` and `qualityMode={detail.quality_mode}` passed to every SceneCard |
| `frontend/src/lib/constants.ts` | qualityModeCostMultiplier helper | VERIFIED | Lines 131-133: exported function `qualityModeCostMultiplier(candidateCount: number): number` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `candidate_scoring.py` | `cv_analysis_service.py` | CVAnalysisService.analyze_generated_content | WIRED | Lazy import + call at line 125-126, used in `_score_manifest_adherence` lines 235-263 |
| `candidate_scoring.py` | `clip_embedding_service.py` | CLIPEmbeddingService.compute_similarity | WIRED | Lazy import + call at line 132-133, used in `_score_continuity` lines 291-306 |
| `video_gen.py` | `candidate_scoring.py` | CandidateScoringService.score_all_candidates | WIRED | Import at line 42, singleton at lines 68-76, called at lines 561-568 |
| `video_gen.py` | `models.py` | GenerationCandidate records created per candidate | WIRED | Import at line 41, instantiated at lines 527-534, flushed at line 536 |
| `ProjectDetail.tsx` | `SceneCard.tsx` | projectId and qualityMode props | WIRED | Lines 242-249: both props passed at every SceneCard render |
| `SceneCard.tsx` | `/api/projects/{id}/scenes/{idx}/candidates` | listCandidates fetch in useEffect | WIRED | Lines 53-62: `listCandidates(projectId, scene.scene_index)` called when `qualityMode && expanded && !candidatesLoaded` |
| `routes.py` | `models.py` | GenerationCandidate query and VideoClip update | WIRED | Lines 1949-1957 (query), lines 2014-2027 (VideoClip update in select_candidate) |
| `client.ts` | `/api/projects/{id}/scenes/{idx}/candidates` | listCandidates + selectCandidate functions | WIRED | Lines 211-230: both API functions imported and used in SceneCard |

### Requirements Coverage

No explicit REQUIREMENTS.md entries mapped to phase 11.

### Anti-Patterns Found

None. No TODOs, stubs, placeholder returns, or empty handlers found in any phase 11 files.

### Human Verification Required

#### 1. Quality Mode Toggle Visual Appearance

**Test:** Navigate to the Generate Video form. Confirm that the Quality Mode toggle uses an amber color scheme and is visually distinct from other toggles on the page.
**Expected:** Toggle turns amber when enabled; 2x/3x/4x pills appear; cost multiplier updates live.
**Why human:** Visual distinction and color rendering cannot be verified programmatically.

#### 2. Candidate Comparison Grid Display

**Test:** On a quality-mode project with completed scenes, expand a SceneCard. Confirm the candidate comparison grid appears with individual sub-scores and a highlighted selected candidate.
**Expected:** Grid shows all candidates with composite score, sub-scores (manifest, quality, continuity, prompt), and amber border on selected candidate.
**Why human:** Actual rendering and layout of the candidate comparison grid requires visual inspection.

#### 3. Manual Selection Override End-to-End

**Test:** On a quality-mode project, click a non-selected candidate in the SceneCard comparison grid. Confirm the selection updates optimistically (amber border moves immediately) and the server confirms the change.
**Expected:** Clicking a candidate immediately shows it as selected (optimistic), backend confirms with `selected_by: "user"`, and re-stitching would use the new candidate.
**Why human:** Requires a running project with actual quality-mode candidates to test the interaction.

### Gaps Summary

No gaps found. All 15 observable truths are verified. The phase goal is fully achieved:

- Data layer: GenerationCandidate ORM model with all score dimensions, Project quality columns, SCORE_WEIGHTS constant — all present and substantive.
- Pipeline: Veo receives `number_of_videos`, all candidates saved and scored, winner auto-selected by composite score, VideoClip.local_path updated for stitcher compatibility.
- API: GET candidates and PUT select endpoints both substantive with dual consistency (GenerationCandidate.is_selected AND VideoClip.local_path kept in sync).
- UI: Quality Mode toggle with cost impact in GenerateForm, candidate comparison grid with click-to-select in SceneCard, ProjectDetail correctly propagates props.

**Minor observation (non-blocking):** The `list_projects` endpoint (routes.py lines 782-796) does not populate `quality_mode`/`candidate_count` in the ProjectListItem response objects, even though the schema defines these fields with defaults. This means the project list view cannot indicate quality-mode projects. This does not block the phase goal (generation, scoring, selection all work) but is a cosmetic completeness gap.

---

_Verified: 2026-02-17T02:36:45Z_
_Verifier: Claude (gsd-verifier)_
