---
phase: 24-comfyui-flux-workflows
plan: 02
subsystem: pipeline
tags: [comfyui, flux, keyframes, routing, lora, reference-images, binding-resolution]

# Dependency graph
requires:
  - phase: 24-comfyui-flux-workflows
    plan: 01
    provides: "build_flux_txt2img_workflow() builder, _FLUX_RESOLUTIONS, Flux workflow templates"
  - phase: 23-tag-syntax-binding-pipeline
    provides: "resolve_tags_with_assets() with ResolvedAssetRef for binding-based resolution"
provides:
  - "Flux model IDs in COMFYUI_IMAGE_MODELS and ALLOWED_IMAGE_MODELS"
  - "_generate_image_comfyui_flux() function with LoRA and reference image support"
  - "Binding-based reference categorization (CHARACTER -> LoRA, PROP/SET -> reference images)"
  - "Flux routing in all 6 COMFYUI_IMAGE_MODELS check sites"
  - "Frontend Flux model options in IMAGE_MODELS catalog"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flux routing via image_model.startswith('flux-') sub-branch within is_comfyui blocks"
    - "Binding-based asset categorization: CHARACTER with lora_url -> LoRA, others -> reference images"
    - "Flux resolution mapping via _FLUX_RESOLUTIONS dict from comfyui_client"

key-files:
  created: []
  modified:
    - backend/vidpipe/pipeline/keyframes.py
    - backend/vidpipe/api/routes.py
    - backend/vidpipe/pipeline/video_gen.py
    - backend/vidpipe/api/asset_library.py
    - frontend/src/lib/constants.ts

key-decisions:
  - "Flux end-frame generation uses text-only (no image conditioning) since Flux txt2img doesn't support conditioning frames"
  - "Binding resolution wrapped in try/except for graceful fallback to basic Flux generation without LoRA/refs"
  - "Asset library generate-image endpoints use 1:1 aspect ratio (1024x1024) for standalone Flux image generation"

patterns-established:
  - "Flux sub-branch pattern: 'if is_comfyui and image_model.startswith(\"flux-\"):' before existing Qwen 'elif is_comfyui:'"
  - "LoRA filename from first CHARACTER binding with lora_url; remaining assets contribute reference images"

requirements-completed: [FLUX-06, FLUX-07, FLUX-08]

# Metrics
duration: 5min
completed: 2026-03-14
---

# Phase 24 Plan 02: Flux Pipeline Routing Summary

**Flux.1 Dev models wired into all keyframe generation paths with binding-based LoRA/reference categorization and frontend model selection**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T22:00:23Z
- **Completed:** 2026-03-14T22:05:33Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- All four Flux model IDs accepted by API validation and routed to ComfyUI across 6 check sites
- New _generate_image_comfyui_flux() function handles Flux-specific workflow building, polling, and image download
- Binding-based reference resolution categorizes CHARACTER assets for LoRA injection and PROP/SET for reference images
- Frontend IMAGE_MODELS catalog includes all four Flux options with REFERENCE_IMAGE_MODELS updated

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Flux model routing, _generate_image_comfyui_flux(), and binding-based reference resolution** - `4e5fe54` (feat)
2. **Task 2: Add Flux model options to frontend IMAGE_MODELS catalog** - `c58dd21` (feat)

## Files Created/Modified
- `backend/vidpipe/pipeline/keyframes.py` - Extended COMFYUI_IMAGE_MODELS, added _generate_image_comfyui_flux(), added Flux branches in start/end frame generation with binding resolution
- `backend/vidpipe/api/routes.py` - Extended ALLOWED_IMAGE_MODELS and IMAGE_MODEL_COST, added Flux branches in _regenerate_keyframe (4 call sites)
- `backend/vidpipe/pipeline/video_gen.py` - Added Flux branch in safety escalation keyframe regeneration
- `backend/vidpipe/api/asset_library.py` - Added Flux branches in actor, set, and prop generate-image endpoints
- `frontend/src/lib/constants.ts` - Added 4 Flux model entries to IMAGE_MODELS, added flux-dev-redux/flux-dev-full to REFERENCE_IMAGE_MODELS

## Decisions Made
- Flux end-frame generation uses text-only (no image conditioning) since Flux txt2img workflow doesn't support conditioning frames like Gemini does
- Binding resolution wrapped in try/except for graceful fallback -- binding failures never crash the pipeline
- Asset library generate-image endpoints use 1:1 aspect ratio (1024x1024) for standalone Flux generation since these are individual asset images, not scene keyframes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 24 (ComfyUI Flux.1 Workflows) is now complete
- Flux models are fully selectable in the UI, routed through ComfyUI for all generation paths
- Ready for next phase

---
*Phase: 24-comfyui-flux-workflows*
*Completed: 2026-03-14*

## Self-Check: PASSED

All files found, all commits verified.
