---
phase: 07-manifest-aware-storyboarding-and-audio-manifest
plan: 02
subsystem: pipeline
tags: [storyboard, llm, manifest, audio, conditional-logic]

# Dependency graph
requires:
  - phase: 07-manifest-aware-storyboarding-and-audio-manifest
    plan: 01
    provides: EnhancedStoryboardOutput schema and SceneManifest/SceneAudioManifest ORM models
  - phase: 06-generateform-integration
    provides: Projects with optional manifest_id foreign key
provides:
  - Manifest-aware storyboard generation with asset registry context injection
  - Scene and audio manifest persistence to database
  - Backward-compatible storyboard pipeline for non-manifest projects
affects: [08-veo-reference-passthrough, 09-cv-analysis-pipeline, 10-adaptive-prompt-rewriting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conditional schema selection (EnhancedStoryboardOutput vs StoryboardOutput) based on project.manifest_id"
    - "LLM context injection via formatted asset registry text block in system prompt"
    - "Post-validation logging for asset tag references with warnings (not errors)"
    - "Separate prompts (ENHANCED vs STORYBOARD_SYSTEM_PROMPT) for different execution paths"

key-files:
  created: []
  modified:
    - backend/vidpipe/services/manifest_service.py
    - backend/vidpipe/pipeline/storyboard.py

key-decisions:
  - "load_manifest_assets orders by quality_score desc (not sort_order) for LLM attention prioritization"
  - "Production notes only included for quality >= 7.0 to manage context window size"
  - "Asset tags validated post-generation with warnings (not errors) to allow new asset declarations"
  - "use_manifests flag determines schema, prompt, and persistence path in single function"

patterns-established:
  - "Conditional execution paths based on project.manifest_id for backward compatibility"
  - "Asset registry formatted as structured text with headers, separators, and truncation rules"
  - "Manifest persistence happens after Scene creation, before final commit"

# Metrics
duration: 3.1min
completed: 2026-02-16
---

# Phase 07 Plan 02: Enhanced Storyboard Pipeline Summary

**Manifest-aware storyboard generation with asset registry injection and structured scene/audio manifest persistence**

## Performance

- **Duration:** 3min 4s
- **Started:** 2026-02-16T23:38:07Z
- **Completed:** 2026-02-16T23:41:11Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `load_manifest_assets()` to query assets ordered by quality_score for LLM attention prioritization
- Added `format_asset_registry()` to format assets as structured text with truncation and quality-based filtering
- Enhanced storyboard pipeline with conditional manifest-aware behavior based on `project.manifest_id`
- Implemented ENHANCED_STORYBOARD_PROMPT with asset registry context and manifest/audio instructions
- Conditional schema selection: EnhancedStoryboardOutput for manifest projects, StoryboardOutput for non-manifest projects
- Persist SceneManifest and SceneAudioManifest records with denormalized query fields
- Post-validate asset tag references with logger.warning for unrecognized tags
- Full backward compatibility: projects without manifest_id use original behavior unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Add asset loading and formatting functions to manifest_service** - `5c12a8f` (feat)
2. **Task 2: Enhance storyboard generation with manifest context and manifest persistence** - `78febc8` (feat)

## Files Created/Modified

- `backend/vidpipe/services/manifest_service.py` - Added load_manifest_assets() with quality-based ordering, format_asset_registry() with truncation rules and quality filtering
- `backend/vidpipe/pipeline/storyboard.py` - Added logging, EnhancedStoryboardOutput import, ENHANCED_STORYBOARD_PROMPT constant, conditional manifest-aware logic, SceneManifest/SceneAudioManifest persistence

## Decisions Made

**Quality-based ordering for LLM context:**
- `load_manifest_assets()` orders by quality_score desc (nulls last) instead of sort_order
- Rationale: Highest-quality assets appear first in system prompt for better LLM attention distribution
- This differs from `list_assets()` which orders by sort_order for UI display

**Production notes filtering:**
- visual_description only included for assets with quality_score >= 7.0
- Prevents context window bloat from lower-quality crops per Research pitfall 1
- Reverse prompts always included (truncated to 200 chars)

**Post-validation with warnings:**
- Asset tag references validated after LLM generation
- Unrecognized tags logged as warnings (not errors) to allow new_asset_declarations
- Enables LLM to declare assets not in the registry while tracking discrepancies

**Conditional execution in single function:**
- `use_manifests` flag controls schema, prompt, and persistence paths
- No function signature changes: `generate_storyboard(session, project)` unchanged
- Maintains single entry point for both manifest and non-manifest projects

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. All verification checks passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 08 (Veo Reference Passthrough and Clean Sheets):**
- Scene manifests persisted with asset_tags array for reference selection
- Audio manifests persisted with dialogue, SFX, ambient, music for Veo 3+ audio generation
- Backward compatibility ensures existing projects continue to work
- Asset tag post-validation provides quality control without blocking generation

**No blockers.**

## Self-Check: PASSED

All files exist:
- FOUND: backend/vidpipe/services/manifest_service.py (modified)
- FOUND: backend/vidpipe/pipeline/storyboard.py (modified)

All commits exist:
- FOUND: 5c12a8f (Task 1: asset loading and formatting)
- FOUND: 78febc8 (Task 2: enhanced storyboard pipeline)

Verification commands passed:
- `load_manifest_assets` and `format_asset_registry` import successfully
- `format_asset_registry([])` returns "No assets registered..." message
- `format_asset_registry([MockAsset()])` produces formatted output with [TAG], quality score, reverse prompt, and production notes
- `generate_storyboard` imports successfully
- All structure checks passed: EnhancedStoryboardOutput import, ENHANCED_STORYBOARD_PROMPT constant, use_manifests conditional, SceneManifest/SceneAudioManifest persistence, logger.warning for tag validation

---
*Phase: 07-manifest-aware-storyboarding-and-audio-manifest*
*Completed: 2026-02-16*
