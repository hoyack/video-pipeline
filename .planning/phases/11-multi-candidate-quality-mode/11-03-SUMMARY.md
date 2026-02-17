---
phase: 11-multi-candidate-quality-mode
plan: 03
subsystem: api, ui
tags: [fastapi, react, typescript, pydantic, quality-mode, candidates]

# Dependency graph
requires:
  - phase: 11-01
    provides: GenerationCandidate model with scoring fields and Project quality_mode columns
  - phase: 11-02
    provides: _poll_and_collect_candidates and _handle_quality_mode_candidates pipeline integration

provides:
  - GET /api/projects/{id}/scenes/{idx}/candidates — returns all scored candidates for a scene
  - PUT /api/projects/{id}/scenes/{idx}/candidates/{cid}/select — manual override with VideoClip.local_path sync
  - CandidateResponse Pydantic schema
  - CandidateScore TypeScript interface
  - Quality Mode toggle in GenerateForm with 2/3/4x pill selector and cost multiplier display
  - CandidateComparison panel in SceneCard with score breakdown and click-to-select
  - ProjectDetail passes projectId and qualityMode props to SceneCard
affects:
  - phase-12-fork-system-integration
  - future UI enhancements on candidate comparison

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Optimistic UI update on candidate selection (setCandidates local state update before server confirmation)
    - Lazy candidate loading (fetch only when expanded + qualityMode + not yet loaded)
    - Dual consistency update: GenerationCandidate.is_selected AND VideoClip.local_path kept in sync

key-files:
  created: []
  modified:
    - backend/vidpipe/api/routes.py
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts
    - frontend/src/components/GenerateForm.tsx
    - frontend/src/components/SceneCard.tsx
    - frontend/src/components/ProjectDetail.tsx
    - frontend/src/lib/constants.ts
    - .gitignore

key-decisions:
  - "Candidate comparison panel uses lazy loading (fetches on first expand) to avoid N API calls on project load"
  - "selectCandidate PUT updates both is_selected flag AND VideoClip.local_path — stitcher reads local_path so both must stay consistent"
  - "Optimistic UI update on selection: setCandidates fires immediately, PUT request fires async — avoids UI lag"
  - ".gitignore fixed: lib/ pattern changed to /lib/ (root-only) to allow frontend/src/lib/ to be committed"

patterns-established:
  - "Candidate score grid: 2-col mobile, 4-col lg, amber border for selected, scored with composite + sub-scores"
  - "Quality Mode UI: amber color scheme (not blue) to visually distinguish from standard mode controls"

# Metrics
duration: 8min
completed: 2026-02-16
---

# Phase 11 Plan 03: Candidate API Endpoints and Quality Mode UI Summary

**Two candidate REST endpoints, Quality Mode toggle in GenerateForm with cost multiplier display, and scored candidate comparison grid in SceneCard with click-to-override selection**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-02-16T00:00:00Z
- **Completed:** 2026-02-16T00:08:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- GET `/api/projects/{id}/scenes/{idx}/candidates` returns all candidates with full score breakdown
- PUT `…/candidates/{cid}/select` atomically deselects all, selects chosen, updates `VideoClip.local_path` for stitcher consistency
- GenerateForm Quality Mode toggle with 2x/3x/4x pill selector and live cost estimate (`~Nx = $X.XX est.`)
- SceneCard shows candidate comparison grid when expanded in a quality mode project; clicking a card sends PUT and updates UI optimistically
- ProjectDetail correctly propagates `projectId` and `qualityMode` to every SceneCard

## Task Commits

Each task was committed atomically:

1. **Task 1: Candidate API endpoints and response schemas** - `bbd9d55` (feat)
2. **Task 2: Frontend Quality Mode toggle and candidate comparison UI** - `b8f10f2` (feat)

## Files Created/Modified
- `backend/vidpipe/api/routes.py` - Added `CandidateResponse` schema, `list_candidates` GET and `select_candidate` PUT endpoints; imported `GenerationCandidate`
- `frontend/src/api/types.ts` - Added `CandidateScore` interface; extended `GenerateRequest` and `ProjectDetail` with `quality_mode`/`candidate_count`
- `frontend/src/api/client.ts` - Added `listCandidates()` and `selectCandidate()` API functions; added `CandidateScore` import
- `frontend/src/components/GenerateForm.tsx` - Quality Mode toggle (amber), 2/3/4x candidate count pills, cost impact display, request body update
- `frontend/src/components/SceneCard.tsx` - New `projectId`/`qualityMode` props, lazy candidate loading, candidate comparison grid with scores and click-to-select
- `frontend/src/components/ProjectDetail.tsx` - Passes `projectId` and `qualityMode` to each `SceneCard`
- `frontend/src/lib/constants.ts` - Added `qualityModeCostMultiplier()` helper
- `.gitignore` - Fixed `lib/` pattern to root-anchored `/lib/` so `frontend/src/lib/` is now trackable

## Decisions Made
- Lazy candidate loading: candidates are only fetched when the SceneCard is expanded and `qualityMode=true`, avoiding N API calls on project detail load
- Dual consistency: `selectCandidate` PUT updates both `GenerationCandidate.is_selected` and `VideoClip.local_path` — the stitcher reads `VideoClip.local_path`, so both must stay consistent
- Optimistic UI: `setCandidates` fires immediately on click (before server response) to avoid perceived lag

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed .gitignore blocking frontend/src/lib/ commits**
- **Found during:** Task 2 (committing frontend files)
- **Issue:** `.gitignore` had bare `lib/` pattern which matched `frontend/src/lib/` anywhere in tree, preventing `constants.ts` from being tracked
- **Fix:** Changed `lib/` to `/lib/` (root-anchored) so only the repo-root `lib/` Python virtual env directory is excluded; `frontend/src/lib/` is now committable
- **Files modified:** `.gitignore`
- **Verification:** `git check-ignore -v frontend/src/lib/constants.ts` returns exit code 1 (not ignored); TypeScript compiles cleanly with committed file
- **Committed in:** `b8f10f2` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** Necessary correctness fix — `constants.ts` was in the plan's `files_modified` list but could never have been committed under the old gitignore. No scope creep.

## Issues Encountered
None beyond the gitignore fix above.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- Phase 11 complete: data layer (11-01), pipeline integration (11-02), and UI/API (11-03) all done
- Multi-candidate Quality Mode is fully end-to-end: users can toggle it, pipeline generates N candidates, scoring auto-selects best, users can manually override in the UI
- Ready for Phase 12: Fork System Integration with Manifests

---
*Phase: 11-multi-candidate-quality-mode*
*Completed: 2026-02-16*
