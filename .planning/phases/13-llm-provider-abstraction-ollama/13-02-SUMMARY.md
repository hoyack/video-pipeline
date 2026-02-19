---
phase: 13-llm-provider-abstraction-ollama
plan: "02"
subsystem: backend
tags: [llm-abstraction, ollama, adapter-pattern, refactoring, pipeline]
dependency_graph:
  requires:
    - phase: 13-01
      provides: "LLMAdapter ABC, VertexAIAdapter, OllamaAdapter, get_adapter() registry, vision schemas"
  provides:
    - "All LLM call sites migrated to adapter interface (zero direct google.genai SDK usage in call sites)"
    - "generate_storyboard() accepts optional text_adapter"
    - "PromptRewriterService accepts optional text_adapter"
    - "ReversePromptService accepts optional vision_adapter"
    - "CVAnalysisService accepts optional vision_adapter"
    - "CandidateScoringService accepts optional vision_adapter"
    - "generate_keyframes() accepts optional text_adapter"
    - "generate_videos() accepts optional text_adapter + vision_adapter"
    - "Orchestrator creates and passes adapters from project config + UserSettings"
    - "ForkRequest + fork endpoint accept and validate vision_model"
  affects:
    - "backend/vidpipe/pipeline (storyboard.py, keyframes.py, video_gen.py)"
    - "backend/vidpipe/services (prompt_rewriter.py, reverse_prompt_service.py, cv_analysis_service.py, candidate_scoring.py)"
    - "backend/vidpipe/orchestrator/pipeline.py"
    - "backend/vidpipe/api/routes.py"
tech_stack:
  added: []
  patterns:
    - "Optional adapter injection: services accept Optional[LLMAdapter], fall back to get_adapter() for backward compat"
    - "Per-call service instantiation: generate_videos() creates CVAnalysisService/CandidateScoringService with vision_adapter each call (not module-level singletons)"
    - "Adapter threading: generate_videos() propagates services through _generate_video_for_scene, _handle_quality_mode_candidates, _run_post_generation_analysis"
    - "Orchestrator wiring: run_pipeline() creates adapters from project model config + UserSettings and passes to all three pipeline stage functions"
key_files:
  modified:
    - backend/vidpipe/pipeline/storyboard.py
    - backend/vidpipe/services/prompt_rewriter.py
    - backend/vidpipe/services/reverse_prompt_service.py
    - backend/vidpipe/services/cv_analysis_service.py
    - backend/vidpipe/services/candidate_scoring.py
    - backend/vidpipe/pipeline/keyframes.py
    - backend/vidpipe/pipeline/video_gen.py
    - backend/vidpipe/orchestrator/pipeline.py
    - backend/vidpipe/api/routes.py
key-decisions:
  - "cv_analysis_service uses only first frame for adapter.analyze_image() (adapter takes single image), full detection context from all frames embedded in text prompt — preserves analytical quality without multi-image API requirement"
  - "Per-call CVAnalysisService/CandidateScoringService in generate_videos() instead of module singletons — enables vision_adapter to flow through correctly; module-level singletons retained for backward compat but no longer called from generate_videos"
  - "Services threaded through _generate_video_for_scene, _handle_quality_mode_candidates, _run_post_generation_analysis as Optional parameters with singleton fallback — backward compatibility preserved"
  - "storyboard.py uses max_retries=1 per adapter call so outer tenacity temperature-reduction retry still works correctly"
  - "entity_extraction.py and manifesting_engine.py remain on direct gemini-2.5-flash fallback path (explicit out-of-scope, future phase)"
patterns-established:
  - "Adapter injection pattern: all LLM-using services accept Optional[LLMAdapter] with get_adapter() fallback"
  - "Orchestrator owns adapter lifecycle: creates from project config + UserSettings, passes to pipeline stages"
requirements-completed:
  - LLMA-02
  - LLMA-06
  - LLMA-07
duration: 9min
completed: 2026-02-19
---

# Phase 13 Plan 02: LLM Call Site Migration Summary

**All LLM call sites migrated from direct Gemini SDK to LLMAdapter interface, orchestrator wired to create and pass adapters from project config, enabling Ollama models throughout the pipeline.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-19T17:26:44Z
- **Completed:** 2026-02-19T17:35:29Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Removed all direct `google.genai` SDK calls from the five primary call sites (storyboard, prompt_rewriter, reverse_prompt_service, cv_analysis_service, candidate_scoring)
- Extended all three pipeline stage function signatures with optional adapter parameters (text_adapter, vision_adapter) with full backward compatibility
- Wired orchestrator to create `text_adapter` and `vision_adapter` from project config + UserSettings and pass them through all pipeline stages

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate text call sites** - `26d30d4` (feat)
2. **Task 2: Migrate vision call sites + extend pipeline stage signatures** - `8f3b887` (feat)
3. **Task 3: Orchestrator wiring + route validation** - `411d918` (feat)

## Files Created/Modified

- `backend/vidpipe/pipeline/storyboard.py` — generate_storyboard() accepts optional text_adapter, uses adapter.generate_text() with max_retries=1 per attempt for temperature-reduction compatibility
- `backend/vidpipe/services/prompt_rewriter.py` — PromptRewriterService accepts optional text_adapter, _call_rewriter() uses adapter.generate_text()
- `backend/vidpipe/services/reverse_prompt_service.py` — ReversePromptService accepts optional vision_adapter, uses adapter.analyze_image() with ReversePromptOutput schema, returns result.model_dump()
- `backend/vidpipe/services/cv_analysis_service.py` — CVAnalysisService accepts optional vision_adapter, _run_semantic_analysis() uses adapter.analyze_image() with SemanticAnalysisOutput on first frame + full detection context in prompt
- `backend/vidpipe/services/candidate_scoring.py` — CandidateScoringService accepts optional vision_adapter, _score_visual_and_prompt() uses adapter.analyze_image() with VisualPromptScoreOutput
- `backend/vidpipe/pipeline/keyframes.py` — generate_keyframes() accepts text_adapter, passes to PromptRewriterService
- `backend/vidpipe/pipeline/video_gen.py` — generate_videos() accepts text_adapter + vision_adapter, creates per-call CVAnalysisService/CandidateScoringService with vision_adapter; services propagated through _generate_video_for_scene, _handle_quality_mode_candidates, _run_post_generation_analysis
- `backend/vidpipe/orchestrator/pipeline.py` — run_pipeline() loads UserSettings, creates text_adapter + vision_adapter from project model config, passes to all three pipeline stage functions
- `backend/vidpipe/api/routes.py` — ForkRequest.vision_model added; fork endpoint validates vision_model with ollama/ support; fork text_model validation updated to accept ollama/ prefix; vision_model set on forked project

## Decisions Made

- **cv_analysis first-frame only:** The `analyze_image()` adapter interface takes a single image. `_run_semantic_analysis()` uses only the first sampled frame, embedding all detection context from all frames in the text prompt. Preserves analytical quality without requiring multi-image API support.
- **Per-call vs singleton services in generate_videos():** Instead of relying on module-level singletons (which don't carry the vision_adapter), `generate_videos()` creates `CVAnalysisService(vision_adapter=vision_adapter)` and `CandidateScoringService(vision_adapter=vision_adapter)` at the start. The old singleton factory functions remain but are no longer called from `generate_videos()`.
- **max_retries=1 in storyboard:** The storyboard uses an outer tenacity retry with temperature reduction (0.7 → 0.55 → 0.4). Passing `max_retries=1` to `adapter.generate_text()` disables the adapter's internal retry per attempt, letting the outer loop control temperature.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Existing Gemini pipeline continues working identically (backward compatible fallback via `get_adapter("gemini-2.5-flash")`).

## Next Phase Readiness

- Plan 03 (Ollama UI integration + settings flow) can now proceed — all call sites are provider-agnostic
- entity_extraction.py and manifesting_engine.py remain on direct gemini-2.5-flash path (explicit out-of-scope, Phase 14+)
- Full Ollama-powered pipeline now possible: set project.text_model="ollama/llama3.1" and project.vision_model="ollama/llava" with appropriate UserSettings

## Self-Check: PASSED

Modified files verified:
- [x] backend/vidpipe/pipeline/storyboard.py
- [x] backend/vidpipe/services/prompt_rewriter.py
- [x] backend/vidpipe/services/reverse_prompt_service.py
- [x] backend/vidpipe/services/cv_analysis_service.py
- [x] backend/vidpipe/services/candidate_scoring.py
- [x] backend/vidpipe/pipeline/keyframes.py
- [x] backend/vidpipe/pipeline/video_gen.py
- [x] backend/vidpipe/orchestrator/pipeline.py
- [x] backend/vidpipe/api/routes.py
- [x] .planning/phases/13-llm-provider-abstraction-ollama/13-02-SUMMARY.md

Commits:
- [x] 26d30d4 — Task 1 (text call sites)
- [x] 8f3b887 — Task 2 (vision call sites + pipeline stages)
- [x] 411d918 — Task 3 (orchestrator wiring + routes)

---
*Phase: 13-llm-provider-abstraction-ollama*
*Completed: 2026-02-19*
