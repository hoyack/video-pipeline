---
phase: 24-comfyui-flux-workflows
verified: 2026-03-14T22:30:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 24: ComfyUI Flux.1 Dev Workflows Verification Report

**Phase Goal:** Introduce Flux.1 Dev as an image generation backend with workflow templates for base, LoRA, reference injection, and hybrid modes — with a builder function that dynamically selects the right template and routes binding-based asset references through the keyframe pipeline

**Verified:** 2026-03-14T22:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Four Flux.1 Dev workflow JSON templates exist in comfyui_templates directory | VERIFIED | `flux-txt2img-base.json`, `flux-txt2img-with-lora.json`, `flux-txt2img-with-refs.json`, `flux-txt2img-full.json` all present and parse as valid JSON |
| 2 | `build_flux_txt2img_workflow()` selects the correct template based on LoRA/reference presence | VERIFIED | Builder tested: base (no lora/refs), lora-only, refs-only, full (both) — all select correctly |
| 3 | Builder injects prompt, seed, dimensions, LoRA filename, and reference filenames into workflow | VERIFIED | Node injections confirmed: node 6 (prompt), node 7 (negative), node 3 (seed), node 5 (width/height), node 14 (lora_name, strength_model), nodes 20-22 (reference filenames) |
| 4 | Templates follow ComfyUI API format with correct node wiring for Flux.1 Dev | VERIFIED | All templates: UNETLoader on node 12, SaveImage on node 9, `_meta` on every node, correct node type hierarchy |
| 5 | Flux model IDs are accepted by the API validation layer | VERIFIED | All four Flux IDs present in `ALLOWED_IMAGE_MODELS` in routes.py (lines 54-57) |
| 6 | Selecting a Flux model routes keyframe generation to ComfyUI (not Vertex AI) | VERIFIED | `COMFYUI_IMAGE_MODELS` includes all four Flux IDs (keyframes.py line 53); `is_comfyui` check gates routing; `image_model.startswith("flux-")` sub-branch calls `_generate_image_comfyui_flux()` |
| 7 | Binding-based reference resolution categorizes CHARACTER refs for LoRA and PROP/SET refs for reference images | VERIFIED | keyframes.py lines 832-868: CHARACTER with `lora_url` -> `flux_lora`; all others with `reference_image_urls` -> `flux_ref_filenames`. Wrapped in try/except for non-fatal fallback. Note: `lora_url` is always `None` currently (tagged "Future Phase 25") but the routing code is correctly structured |
| 8 | Regeneration paths in routes.py, video_gen.py, and asset_library.py handle Flux models correctly | VERIFIED | routes.py: 4 call sites in `_regenerate_keyframe` (lines 5016-5019, 5073-5079, 5094-5100, 5150-5157). video_gen.py: safety escalation lines 291-299. asset_library.py: actor (918-927), set (1475-1484), prop (1968-1977) |
| 9 | Frontend IMAGE_MODELS catalog includes all four Flux model options | VERIFIED | constants.ts lines 89-92: flux-dev, flux-dev-lora, flux-dev-redux, flux-dev-full. REFERENCE_IMAGE_MODELS includes flux-dev-redux and flux-dev-full (lines 138-139) |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-base.json` | Base Flux.1 Dev txt2img workflow | VERIFIED | 10 nodes, UNETLoader on node 12, SaveImage on node 9, all `_meta` present |
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-with-lora.json` | Flux.1 Dev + LoRA workflow | VERIFIED | 11 nodes, LoraLoaderModelOnly on node 14 |
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-with-refs.json` | Flux.1 Dev + UNO reference injection workflow | VERIFIED | 17 nodes, LoadImage on node 20, CLIPVisionLoader/Encode, unCLIPConditioning |
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-full.json` | Full hybrid Flux.1 Dev + LoRA + UNO workflow | VERIFIED | 18 nodes, LoraLoaderModelOnly on node 14, LoadImage on node 20 |
| `backend/vidpipe/services/comfyui_client.py` | `build_flux_txt2img_workflow` builder function | VERIFIED | Exported at line 670, `_FLUX_RESOLUTIONS` at line 592, `flux_resolution()` helper at line 599, four cached template loaders |
| `backend/vidpipe/pipeline/keyframes.py` | Flux model routing and `_generate_image_comfyui_flux()` | VERIFIED | `_generate_image_comfyui_flux()` at line 444; COMFYUI_IMAGE_MODELS extended at line 53; binding-based resolution at lines 832-877 |
| `backend/vidpipe/api/routes.py` | Flux model IDs in ALLOWED_IMAGE_MODELS | VERIFIED | Lines 54-57; 4 Flux IDs in set; `_generate_image_comfyui_flux` imported at line 4985; 4 call sites handled |
| `frontend/src/lib/constants.ts` | Flux model entries in IMAGE_MODELS array | VERIFIED | Lines 89-92 add all four Flux entries; REFERENCE_IMAGE_MODELS updated at lines 138-139 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `comfyui_client.py` | `flux-txt2img-base.json` | template loading with cached path | WIRED | `_FLUX_BASE_TEMPLATE_PATH` at line 611, `_load_flux_base_template()` at line 630; pattern `comfyui_templates.*flux-txt2img` confirmed |
| `keyframes.py` | `comfyui_client.py` | import `build_flux_txt2img_workflow` | WIRED | Import at line 479 inside `_generate_image_comfyui_flux()`; called at line 484 |
| `keyframes.py` | `tag_resolver.py` | import `resolve_tags_with_assets` | WIRED | Dynamic import at line 839 inside binding resolution block; called at lines 840-842 |
| `routes.py` | `keyframes.py` | import `COMFYUI_IMAGE_MODELS` for routing | WIRED | Line 4985: imports `COMFYUI_IMAGE_MODELS, _generate_image_comfyui, _generate_image_comfyui_flux` |
| `asset_library.py` | `keyframes.py` | import `COMFYUI_IMAGE_MODELS` for routing | WIRED | Lines 861, 1436, 1924 import `COMFYUI_IMAGE_MODELS`; lines 922, 1479, 1972 import `_generate_image_comfyui_flux` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FLUX-01 | 24-01 | Flux.1 Dev base text-to-image ComfyUI workflow template | SATISFIED | `flux-txt2img-base.json` exists with UNETLoader, DualCLIPLoader, KSampler, VAEDecode, SaveImage |
| FLUX-02 | 24-01 | Flux.1 Dev + dynamic LoRA loader workflow template | SATISFIED | `flux-txt2img-with-lora.json` with `LoraLoaderModelOnly` on node 14 |
| FLUX-03 | 24-01 | Flux.1 Dev + UNO/Redux reference injection workflow for up to 3 reference images | SATISFIED | `flux-txt2img-with-refs.json` with LoadImage nodes 20-22, ImageBatch, CLIPVisionLoader/Encode, unCLIPConditioning; dynamic node cleanup for < 3 refs verified |
| FLUX-04 | 24-01 | Full hybrid Flux.1 Dev + LoRA + UNO workflow template | SATISFIED | `flux-txt2img-full.json` combines LoRA (node 14) and reference injection (nodes 20-33) |
| FLUX-05 | 24-01 | `build_flux_txt2img_workflow()` builder with dynamic template selection | SATISFIED | Builder at comfyui_client.py line 670; all 4 template selections tested programmatically and pass; `_FLUX_RESOLUTIONS` dict and `flux_resolution()` helper exported |
| FLUX-06 | 24-02 | Flux model IDs added to COMFYUI_IMAGE_MODELS with routing in keyframe pipeline | SATISFIED | All 4 IDs in COMFYUI_IMAGE_MODELS (keyframes.py line 53); `_generate_image_comfyui_flux()` called at both start-frame (line 871) and end-frame (line 988) generation points |
| FLUX-07 | 24-02 | Binding-based reference resolution categorizes ResolvedAssetRefs by type | SATISFIED | keyframes.py lines 832-877; CHARACTER+lora_url -> LoRA, others -> reference images; parallel upload via `comfy_client.upload_image()`; try/except fallback to basic generation |
| FLUX-08 | 24-02 | Frontend Flux model options added to IMAGE_MODELS catalog | SATISFIED | constants.ts lines 89-92: 4 Flux entries; flux-dev-redux and flux-dev-full added to REFERENCE_IMAGE_MODELS |

No orphaned requirements — all 8 FLUX IDs are claimed by plans and verified in the codebase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `keyframes.py` | 1061 | `except Exception as e` — `e` assigned but unused | Info | Pre-existing lint warning (F841), not introduced by this phase |
| `tag_resolver.py` | 54 | `lora_url: str \| None = None  # Future Phase 25 LoRA` | Info | The CHARACTER->LoRA branch in FLUX-07 is structurally correct but will never fire until Phase 25 populates `lora_url`. This is intentional forward-compatible design, not a stub. |
| `video_gen.py` | 759, 761 | `file_mgr` referenced but undefined (F821) | Info | Pre-existing bug in an unrelated S3 path, not introduced by this phase (commit 5bcbd77 pre-dates phase 24) |

No blocker anti-patterns introduced by this phase.

### Human Verification Required

#### 1. Flux End-to-End Image Generation

**Test:** Select "Flux.1 Dev" as the image model, create a project, and run the keyframe pipeline with a ComfyUI server that has `flux1-dev.safetensors`, `ae.safetensors`, `t5xxl_fp16.safetensors`, and `clip_l.safetensors` loaded.
**Expected:** Keyframes are generated via ComfyUI using the base Flux template; images appear in the UI.
**Why human:** Requires a running ComfyUI server with the Flux model weights installed; cannot be verified programmatically.

#### 2. LoRA Template Selection via Binding

**Test:** Create a production bible with a CHARACTER binding where the actor has a `lora_url` set (requires Phase 25 to populate this field). Use a Flux model and verify LoRA injection.
**Expected:** The `flux-txt2img-with-lora.json` template is selected and the LoRA filename is injected at node 14.
**Why human:** `lora_url` is always `None` in current `tag_resolver.py` (marked "Future Phase 25") so this code path cannot fire with current data.

#### 3. Reference Image Upload and Conditioning

**Test:** Create a production bible with PROP or SET bindings that have reference images. Use `flux-dev-redux` model and generate a keyframe.
**Expected:** Reference images are uploaded to ComfyUI, nodes 20-22 are populated with server filenames, and the unCLIPConditioning node applies visual conditioning.
**Why human:** Requires a running ComfyUI server with `clip_vision_g.safetensors` available; visual quality of conditioning cannot be assessed programmatically.

### Gaps Summary

No gaps. All 9 observable truths are verified. All 8 requirement IDs (FLUX-01 through FLUX-08) are satisfied with substantive, wired implementations. The only notable limitation is that the CHARACTER->LoRA sub-path of FLUX-07 is structurally correct but non-functional until Phase 25 populates the `lora_url` field on `ResolvedAssetRef`. This is an intentional design decision documented in both the SUMMARY and the code comment, not a gap in this phase's deliverables.

---

_Verified: 2026-03-14T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
