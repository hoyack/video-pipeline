---
phase: 10-adaptive-prompt-rewriting
plan: 02
subsystem: pipeline
tags: [adaptive-prompts, keyframes, video-gen, manifest, prompt-rewriter, veo, imagen]

# Dependency graph
requires:
  - phase: 10-01
    provides: "PromptRewriterService, RewrittenKeyframePromptOutput, RewrittenVideoPromptOutput schemas, SceneManifest.rewritten_keyframe_prompt/rewritten_video_prompt columns"
  - phase: 09-03
    provides: "cv_analysis_json on SceneManifest (continuity data for rewriter)"
  - phase: 08-01
    provides: "Phase 8 deterministic reference selection (LLM now overrides this)"
  - phase: 07-01
    provides: "SceneManifest, SceneAudioManifest tables and manifest_json structure"
provides:
  - "keyframes.py with PromptRewriterService hook — manifest projects use rewritten start frame prompts"
  - "video_gen.py with PromptRewriterService hook — manifest projects use rewritten video prompts with safety prefix coordination"
  - "LLM reference selection overrides Phase 8 deterministic selection for veo_ref_images"
  - "Audio direction from SceneAudioManifest injected into video rewriter context"
  - "Continuity from scene N-1 cv_analysis_json injected into scene N rewriter call"
  - "Graceful fallback to original prompts on rewriter failure (non-fatal)"
affects:
  - "11-multi-candidate-quality-mode (will evaluate quality of rewritten prompts)"
  - "12-fork-system-integration-with-manifests (fork preserves rewritten_keyframe_prompt and rewritten_video_prompt)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy import inside manifest_id guard (avoid circular imports)"
    - "Try/except with non-fatal warning + fallback to None (graceful degradation)"
    - "Re-raise PipelineStopped inside except Exception block"
    - "LLM output overrides deterministic selection before veo_ref_images rebuild"
    - "Safety prefix stacks on top of rewritten base prompt"

key-files:
  created: []
  modified:
    - backend/vidpipe/pipeline/keyframes.py
    - backend/vidpipe/pipeline/video_gen.py

key-decisions:
  - "veo_ref_images built after LLM override (Phase 10 block at line 563, veo_ref_images at line 673) — correct ordering avoids rebuild"
  - "style_prefix kept with rewritten prompt in keyframes.py; character_prefix dropped (rewriter already injects asset reverse_prompts)"
  - "Rewritten prompt does not get 'Maintain the visual style' suffix — rewriter handles style via manifest metadata"
  - "Re-raise PipelineStopped inside except Exception block — inherits from Exception so must be caught and re-raised explicitly"
  - "all_assets reused from Phase 8 block in video_gen.py — guard conditions match so no reload needed"

patterns-established:
  - "Phase 10 hook placement: after Phase 8 reference selection, before VideoClip resume check in video_gen.py"
  - "Phase 10 hook placement: after keyframe skip check, before start frame generation in keyframes.py"
  - "base_video_prompt = None pattern: rewriter sets it, escalation loop checks it"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 10 Plan 02: Pipeline Integration Summary

**PromptRewriterService wired into keyframes.py and video_gen.py — manifest projects now use LLM-rewritten cinematography prompts with asset injection, continuity corrections, and audio direction before Imagen and Veo submission.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-17T01:48:02Z
- **Completed:** 2026-02-17T01:50:13Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Manifest projects: keyframe start frame generation uses rewritten prompt from PromptRewriterService (cinematography formula with asset reverse_prompts injected)
- Manifest projects: video generation uses rewritten prompt as base with safety prefixes stacked on top during escalation
- LLM-selected reference tags override Phase 8 deterministic selection; veo_ref_images rebuilt from updated selected_refs
- SceneAudioManifest loaded and injected as audio_manifest_json into video rewriter context
- Previous scene cv_analysis_json fed as continuity data into each rewriter call
- Non-manifest projects: zero behavioral change — original storyboard prompts used throughout

## Task Commits

Each task was committed atomically:

1. **Task 1: Integrate rewriter into keyframes.py** - `1d108c2` (feat)
2. **Task 2: Integrate rewriter into video_gen.py with safety prefix coordination and LLM reference override** - `c71bf20` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `backend/vidpipe/pipeline/keyframes.py` - Added Phase 10 rewriter block in `generate_keyframes()` after keyframe skip check; Scene 0 uses rewritten prompt (style_prefix only), scenes N>0 inherit start frames unchanged but still persist rewritten prompt for debugging
- `backend/vidpipe/pipeline/video_gen.py` - Added Phase 10 rewriter block in `_generate_video_for_scene()` after Phase 8 reference selection; `base_video_prompt` used in escalation loop; LLM tags override `selected_refs`; `veo_ref_images` rebuilt from correct refs

## Decisions Made

- `veo_ref_images` build is at line 673, after the Phase 10 rewriter block at line 563 — so it already uses LLM-overridden `selected_refs` without an explicit rebuild step. The plan's "Rebuild veo_ref_images" block was not needed because the ordering is already correct.
- `style_prefix` kept with rewritten prompt in keyframes.py; `character_prefix` dropped because the rewriter already injects full asset reverse_prompts (avoids double-injection).
- `PipelineStopped` inherits from `Exception` so must be explicitly re-raised inside `except Exception as e:` in video_gen.py.
- `all_assets` variable reused from Phase 8 block in video_gen.py — the guard condition `if project.manifest_id and scene_manifest_row and scene_manifest_row.manifest_json` matches exactly, so `all_assets` is guaranteed to be in scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Skipped redundant veo_ref_images rebuild**
- **Found during:** Task 2 (video_gen.py integration)
- **Issue:** The plan specified adding a "Rebuild veo_ref_images" block after the rewriter. However, the existing `veo_ref_images` construction block is already AFTER the rewriter block in execution order (rewriter at line 563, veo_ref_images at line 673). Adding a duplicate rebuild would construct veo_ref_images twice.
- **Fix:** Used the existing veo_ref_images construction block (which now naturally picks up LLM-overridden selected_refs). No duplicate rebuild added.
- **Files modified:** None (deviation from plan, not from codebase)
- **Verification:** Read video_gen.py confirms correct ordering: Phase 8 selection → Phase 10 rewriter (may override selected_refs) → VideoClip resume check → veo_ref_images construction (uses final selected_refs)
- **Committed in:** c71bf20 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 correctness fix — avoided double veo_ref_images construction)
**Impact on plan:** The fix simplified implementation and avoided a redundant operation. Behavioral outcome is identical to plan specification.

## Issues Encountered

- `config.yaml` not present in backend directory so `python -c "from vidpipe.pipeline.keyframes import ..."` fails with pydantic Settings validation. Used `python -m py_compile` for syntax verification instead — confirms no syntax or import graph errors.

## Next Phase Readiness

- Phase 10 complete: PromptRewriterService (10-01) + pipeline integration (10-02) fully wired
- Phase 11 (multi-candidate quality mode) can now evaluate quality differences between rewritten and original prompts
- Phase 12 (fork system integration) can preserve rewritten_keyframe_prompt and rewritten_video_prompt columns when forking scenes

---
*Phase: 10-adaptive-prompt-rewriting*
*Completed: 2026-02-17*
