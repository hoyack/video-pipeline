---
phase: 10-adaptive-prompt-rewriting
plan: 01
subsystem: api
tags: [gemini, pydantic, sqlalchemy, tenacity, prompt-rewriting, vertex-ai]

# Dependency graph
requires:
  - phase: 07-manifest-aware-storyboarding
    provides: SceneManifest model and manifest_json structure with placements/composition
  - phase: 08-veo-reference-passthrough
    provides: reference_image_url on Asset, selected_reference_tags on SceneManifest
  - phase: 09-cv-analysis-pipeline
    provides: cv_analysis_json and continuity_score on SceneManifest for continuity patches
provides:
  - PromptRewriterService with rewrite_keyframe_prompt and rewrite_video_prompt methods
  - RewrittenKeyframePromptOutput and RewrittenVideoPromptOutput pydantic schemas
  - SceneManifest with rewritten_keyframe_prompt and rewritten_video_prompt columns
  - migrate_phase10.sql for existing database migration
affects:
  - 10-02: pipeline integration — keyframes.py and video_gen.py will call PromptRewriterService

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PromptRewriterService uses asyncio.Semaphore(5) class-level rate limiting (Phase 5 pattern)"
    - "Gemini structured output via response_schema=pydantic_model at temperature=0.4 (precise, not creative)"
    - "tenacity @retry(stop_after_attempt(3), retry_if_exception_type(Exception)) on inner async function"
    - "Context assembly builds 5-section user_context block: original prompt, composition, placed assets, continuity, audio"
    - "reverse_prompt truncated to 200 chars, visual_description to 150 chars (quality >= 7.0 only)"

key-files:
  created:
    - backend/vidpipe/schemas/prompt_rewrite.py
    - backend/vidpipe/services/prompt_rewriter.py
    - backend/migrate_phase10.sql
  modified:
    - backend/vidpipe/db/models.py

key-decisions:
  - "PromptRewriterService has separate rewrite_keyframe_prompt and rewrite_video_prompt methods — keyframe formula is static-image-focused, video formula is motion+audio-focused"
  - "Module-level KEYFRAME_REWRITER_SYSTEM_PROMPT and VIDEO_REWRITER_SYSTEM_PROMPT constants (not instance methods) — pure instruction strings"
  - "Helper functions (_format_placed_assets, _build_continuity_patch, etc.) are module-level functions, not static methods — simplifies testing"
  - "_list_available_references shows only assets WITH reference_image_url as selectable — LLM cannot select what Veo cannot receive"
  - "Rewriter does not store result — caller (Plan 02: keyframes.py/video_gen.py) is responsible for persisting rewritten_keyframe_prompt and rewritten_video_prompt"

patterns-established:
  - "Graceful fallback pattern: rewriter raises after 3 retries; caller wraps in try/except and falls back to original storyboard prompt"
  - "Scene-0 guard: _build_continuity_patch returns first-scene message when scene_index==0 or previous_cv_analysis is None"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 10 Plan 01: Adaptive Prompt Rewriting — Schema and Service Summary

**PromptRewriterService with Gemini 2.5 Flash structured output, assembling 5 context inputs (original prompt, manifest composition, placed asset reverse_prompts, CV continuity patch, audio direction) into cinematography-formula prompts with LLM-reasoned reference selection**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-17T01:43:26Z
- **Completed:** 2026-02-17T01:45:54Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created RewrittenKeyframePromptOutput and RewrittenVideoPromptOutput pydantic schemas with all required fields including Field() descriptions for Gemini structured output
- Implemented full PromptRewriterService with separate keyframe and video rewriting paths, each assembling a 5-section context block
- Added rewritten_keyframe_prompt and rewritten_video_prompt nullable Text columns to SceneManifest model plus SQL migration for existing databases

## Task Commits

Each task was committed atomically:

1. **Task 1: Pydantic schemas, SceneManifest columns, and SQL migration** - `f0762e0` (feat)
2. **Task 2: PromptRewriterService with context assembly and Gemini calls** - `bcdba07` (feat)

**Plan metadata:** (docs commit following)

## Files Created/Modified

- `backend/vidpipe/schemas/prompt_rewrite.py` - RewrittenKeyframePromptOutput and RewrittenVideoPromptOutput pydantic BaseModels
- `backend/vidpipe/services/prompt_rewriter.py` - PromptRewriterService with full context assembly, Gemini calls, and helper functions
- `backend/vidpipe/db/models.py` - SceneManifest extended with 2 new nullable Text columns (Phase 10 section)
- `backend/migrate_phase10.sql` - ALTER TABLE statements for existing database migration

## Decisions Made

- Separate rewrite methods for keyframe vs video: static-image and motion-focused formulas require different system prompts and schemas. Same service, different entry points.
- Module-level helper functions (not static methods) for _format_placed_assets, _build_continuity_patch, _format_audio_direction, _list_available_references: simplifies unit testing and follows Python conventions.
- _list_available_references separates assets with and without reference_image_url: LLM knows exactly what is selectable as a Veo reference — avoids hallucinating a tag that has no image.
- Rewriter does not persist results itself: caller responsibility (Plan 02 pipeline integration) stores rewritten prompts in SceneManifest columns.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- SceneManifest column verification from backend/ directory failed because Settings requires config.yaml (not present in backend/). Resolved by running verification from project root where config.yaml exists. Not a code issue — just a test environment path.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PromptRewriterService is ready for Plan 02 pipeline integration
- Plan 02 will call rewrite_keyframe_prompt() in keyframes.py before Imagen and rewrite_video_prompt() in video_gen.py before Veo
- Graceful fallback (try/except wrapping rewriter calls) should be implemented in Plan 02 per research Pitfall 3 guidance
- Safety prefix stacking in video_gen.py must use base_video_prompt pattern (research Pitfall 1) when integrating

---
*Phase: 10-adaptive-prompt-rewriting*
*Completed: 2026-02-17*
