# Phase 24: ComfyUI Flux.1 Workflows - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning
**Source:** PRD Express Path (docs/assets_mapping.md, Section B)

<domain>
## Phase Boundary

This phase introduces Flux.1 Dev as an image generation backend in ComfyUI. It covers workflow template creation, a builder function that dynamically selects templates, keyframe pipeline routing for binding-based asset references, and frontend model options. It does NOT cover LoRA training infrastructure (Phase 25) or frontend @tag autocomplete/preview (Phase 26).

Specifically, this phase delivers:
- Four ComfyUI workflow JSON templates: base, with_lora, with_references, full hybrid
- `build_flux_txt2img_workflow()` builder function in `comfyui_client.py`
- Flux model IDs in `COMFYUI_IMAGE_MODELS` with routing in keyframe pipeline
- Binding-based reference resolution path in `keyframes.py` using `ResolvedAssetRef` from Phase 23
- Frontend Flux model options in `IMAGE_MODELS` catalog

</domain>

<decisions>
## Implementation Decisions

### Workflow Templates
- Four templates stored in `backend/vidpipe/templates/comfyui/` (or `docs/comfyui/`)
- `flux_txt2img_base.json` — Standard Flux.1 Dev text-to-image, no references
- `flux_txt2img_with_lora.json` — Flux.1 Dev + dynamic LoRA loader node
- `flux_txt2img_with_references.json` — Flux.1 Dev + UNO/Redux conditioning for up to 3 reference images
- `flux_txt2img_full.json` — Full hybrid: LoRA + UNO reference conditioning
- Templates are ComfyUI API-format JSON (node graph), not frontend workflow format

### Builder Function
- `build_flux_txt2img_workflow()` in `comfyui_client.py` dynamically selects template based on:
  - No LoRA, no refs → base template
  - LoRA only → with_lora template
  - Refs only → with_references template
  - Both → full template
- Parameters: prompt, negative_prompt, width, height, seed, lora_filename (optional), lora_strength (default 0.8), reference_image_filenames (optional), reference_strengths (optional)
- Injects runtime values (LoRA filename, reference image filenames, prompt text, dimensions, seed) into template node fields

### Model Registry
- New Flux model IDs: `flux-dev`, `flux-dev-lora`, `flux-dev-redux`, `flux-dev-full`
- Added to `COMFYUI_IMAGE_MODELS` set in `keyframes.py` — this is the router (like `COMFYUI_VIDEO_MODELS` in `video_gen.py`)
- Both the main pipeline AND any regenerate paths must check this set

### Keyframe Pipeline Updates
- When scene has `production_bible_id` AND prompt contains tags, use binding-based path
- `resolve_tags_with_assets()` (from Phase 23) provides `ResolvedAssetRef[]`
- Categorize by type: CHARACTER refs → LoRA path, PROP/SET refs → UNO reference images
- If CHARACTER has `lora_url` → use LoRA; if not → use reference images as UNO input
- Falls back to existing keyframe generation path when no bindings or non-Flux model selected

### Frontend Model Options
- Add Flux model entries to `IMAGE_MODELS` in `constants.ts` (or equivalent)
- Users can select Flux models in the scene creation form per-scene

### Claude's Discretion
- Exact ComfyUI node IDs and wiring in workflow templates (depends on available custom nodes)
- UNO vs Redux choice for reference injection (UNO preferred per PRD)
- Reference image upload mechanism to ComfyUI server (may need upload endpoint call before workflow queue)
- LoRA file download/caching strategy on ComfyUI server
- Error handling for missing LoRA files or unavailable reference images

</decisions>

<specifics>
## Specific Ideas

- PRD recommends hybrid approach: LoRA for characters, UNO/Redux for props/sets
- Reference strengths: ~0.65 for props, ~0.60 for sets (need empirical tuning, use as defaults)
- LoRA strength default: 0.8 (from PRD Section 5.3)
- Multiple characters per shot: system can only load one LoRA at a time — use one LoRA + one UNO reference for two-character shots
- ComfyUI client already has `queue_prompt()` and workflow building patterns in existing `comfyui_client.py`

</specifics>

<deferred>
## Deferred Ideas

- LoRA training pipeline (dataset prep, training dispatch, status tracking) → Phase 25
- Frontend @tag autocomplete in scene editor → Phase 26
- Frontend tag preview panel → Phase 26
- Reference image strength tuning UI (per-binding overrides) → Future
- Multiple LoRA merging for multi-character shots → Future

</deferred>

---

*Phase: 24-comfyui-flux-workflows*
*Context gathered: 2026-03-14 via PRD Express Path*
