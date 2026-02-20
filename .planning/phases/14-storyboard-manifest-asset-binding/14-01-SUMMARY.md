---
phase: 14-storyboard-manifest-asset-binding
plan: "01"
subsystem: backend
tags: [storyboard, manifest, character-tags, defense-in-depth, prompt-hardening, remapping]

# Dependency graph
requires:
  - phase: 07-manifest-aware-storyboarding-and-audio-manifest
    provides: manifest-aware storyboard pipeline with scene manifests and asset registry
  - phase: 10-adaptive-prompt-rewriting
    provides: prompt rewriter service with placed_tags resolution
provides:
  - Four-layer defense-in-depth preventing LLM CHARACTER tag invention
  - Deterministic post-LLM tag remapping function
  - Prompt rewriter fallback for unresolved placed_tags
  - Keyframe enforcement fallback for empty placed_char_tags
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [defense-in-depth, deterministic-remapping, fallback-chain]

key-files:
  created: [docs/storyboard-spec.md]
  modified:
    - backend/vidpipe/pipeline/storyboard.py
    - backend/vidpipe/services/manifest_service.py
    - backend/vidpipe/services/prompt_rewriter.py
    - backend/vidpipe/pipeline/keyframes.py

key-decisions:
  - "Positional order mapping for tag remapping (first unrecognized → first manifest char)"
  - "Four independent layers so any single failure still produces correct output"

patterns-established:
  - "Defense-in-depth: each pipeline stage independently validates manifest asset binding"

requirements-completed: [SBIND-01, SBIND-02, SBIND-03, SBIND-04]

# Metrics
duration: 15min
completed: 2026-02-19
---

# Phase 14: Storyboard Manifest Asset Binding Summary

**Four-layer defense-in-depth fix preventing storyboard LLM from inventing CHARACTER tags, with prompt hardening, deterministic remapping, rewriter fallback, and keyframe enforcement**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-02-19
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Hardened ENHANCED_STORYBOARD_PROMPT to mandate existing CHARACTER tags and restrict new_asset_declarations to non-CHARACTER types
- Added `_remap_unrecognized_tags()` deterministic backstop that catches LLM-invented CHARACTER tags post-generation
- Prompt rewriter fallback marks ALL CHARACTER assets as MUST SELECT when placed_tags don't resolve
- Keyframe enforcement fallback uses all manifest CHARACTER assets when placed_char_tags is empty
- Updated storyboard-spec.md with comprehensive documentation

## Task Commits

1. **Task 1: Storyboard prompt hardening + post-LLM tag remapping (SBIND-01, SBIND-02)** - `05b521f` (fix)
2. **Task 2: Prompt rewriter + keyframe enforcement fallbacks (SBIND-03, SBIND-04)** - `05b521f` (fix)

## Files Created/Modified
- `backend/vidpipe/pipeline/storyboard.py` - Hardened prompt + `_remap_unrecognized_tags()` function + call site in generate_storyboard()
- `backend/vidpipe/services/manifest_service.py` - Hardened format_asset_registry() footer
- `backend/vidpipe/services/prompt_rewriter.py` - SBIND-03 fallback in `_list_available_references()`
- `backend/vidpipe/pipeline/keyframes.py` - SBIND-04 fallback for empty placed_char_tags
- `docs/storyboard-spec.md` - Comprehensive storyboard pipeline documentation

## Decisions Made
- Positional order mapping for tag remapping (first unrecognized → first manifest char) — simple, deterministic, covers the common case
- All four layers are independent — any single layer succeeding produces correct output
- Non-manifest projects follow original code paths with zero changes

## Deviations from Plan
None - plan executed as written. Both tasks shipped in a single commit.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All SBIND requirements satisfied
- Reference image pipeline now robust against LLM tag invention
- No known blockers for future work

---
*Phase: 14-storyboard-manifest-asset-binding*
*Completed: 2026-02-19*
