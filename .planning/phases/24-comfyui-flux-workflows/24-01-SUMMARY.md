---
phase: 24-comfyui-flux-workflows
plan: 01
subsystem: services
tags: [comfyui, flux, workflow-templates, image-generation, lora, reference-injection]

# Dependency graph
requires:
  - phase: 23-tag-syntax-binding-pipeline
    provides: "ResolvedAssetRef with lora_url and reference_image_urls for binding-based generation"
provides:
  - "Four Flux.1 Dev ComfyUI API-format workflow JSON templates (base, lora, refs, full)"
  - "build_flux_txt2img_workflow() builder with dynamic template selection"
  - "_FLUX_RESOLUTIONS dict and flux_resolution() helper"
  - "Cached template loaders following established Qwen/Wan pattern"
affects: [24-02-PLAN, keyframes, video_gen, routes, asset_library]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flux.1 Dev workflow template + builder pattern (extends existing ComfyUI template pattern)"
    - "unCLIPConditioning for reference image injection via CLIPVision encoding"
    - "Dynamic ImageBatch rewiring when fewer than 3 reference images provided"

key-files:
  created:
    - backend/vidpipe/services/comfyui_templates/flux-txt2img-base.json
    - backend/vidpipe/services/comfyui_templates/flux-txt2img-with-lora.json
    - backend/vidpipe/services/comfyui_templates/flux-txt2img-with-refs.json
    - backend/vidpipe/services/comfyui_templates/flux-txt2img-full.json
  modified:
    - backend/vidpipe/services/comfyui_client.py

key-decisions:
  - "Used unCLIPConditioning (built-in ComfyUI node) for reference injection instead of UNO custom nodes for broader server compatibility"
  - "ImageBatch inputs dynamically rewired based on actual reference count to avoid dangling node connections"
  - "Reference strength defaults to 0.65 per-reference with override support via reference_strengths parameter"

patterns-established:
  - "Flux template selection: has_lora AND has_refs -> full, has_lora -> lora, has_refs -> refs, else -> base"
  - "Unused LoadImage nodes removed from workflow when fewer than 3 references provided"

requirements-completed: [FLUX-01, FLUX-02, FLUX-03, FLUX-04, FLUX-05]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 24 Plan 01: Flux Workflow Templates Summary

**Four Flux.1 Dev ComfyUI workflow templates with dynamic builder function using unCLIPConditioning for reference injection and LoraLoaderModelOnly for identity LoRA**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-14T21:55:43Z
- **Completed:** 2026-03-14T21:58:15Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Four ComfyUI API-format JSON templates covering all Flux.1 Dev workflow variants (base, LoRA, reference injection, full hybrid)
- build_flux_txt2img_workflow() builder function with dynamic template selection based on LoRA/reference presence
- _FLUX_RESOLUTIONS dict mapping aspect ratios (16:9, 9:16, 1:1) to Flux-native pixel dimensions
- Reference image handling with unused LoadImage node cleanup and ImageBatch input rewiring for 1-3 refs

## Task Commits

Each task was committed atomically:

1. **Task 1: Create four Flux.1 Dev workflow JSON templates** - `784482d` (feat)
2. **Task 2: Add build_flux_txt2img_workflow() builder and template loaders** - `cadde08` (feat)

## Files Created/Modified
- `backend/vidpipe/services/comfyui_templates/flux-txt2img-base.json` - Standard Flux.1 Dev txt2img with UNETLoader, DualCLIPLoader, ModelSamplingFlux, KSampler, SaveImage
- `backend/vidpipe/services/comfyui_templates/flux-txt2img-with-lora.json` - Base + LoraLoaderModelOnly node (node 14) for character identity injection
- `backend/vidpipe/services/comfyui_templates/flux-txt2img-with-refs.json` - Base + LoadImage/ImageBatch/CLIPVisionLoader/CLIPVisionEncode/unCLIPConditioning for up to 3 reference images
- `backend/vidpipe/services/comfyui_templates/flux-txt2img-full.json` - Full hybrid combining LoRA (node 14) and reference injection (nodes 20-33) paths
- `backend/vidpipe/services/comfyui_client.py` - Added _FLUX_RESOLUTIONS, flux_resolution(), four template loaders, build_flux_txt2img_workflow()

## Decisions Made
- Used unCLIPConditioning (built-in ComfyUI node) for reference injection instead of UNO custom nodes -- more broadly compatible with ComfyUI servers, no custom extension required
- ImageBatch inputs dynamically rewired based on actual reference count (1, 2, or 3) to avoid dangling node connections that would cause ComfyUI errors
- Reference strength defaults to 0.65 per-reference with override support via reference_strengths parameter

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Templates and builder ready for Plan 24-02 to wire into keyframe pipeline routing
- _FLUX_RESOLUTIONS and flux_resolution() exported for keyframes.py aspect ratio mapping
- build_flux_txt2img_workflow() exported for _generate_image_comfyui_flux() adapter function

## Self-Check: PASSED

All 4 template files exist, both task commits verified, SUMMARY.md created.

---
*Phase: 24-comfyui-flux-workflows*
*Completed: 2026-03-14*
