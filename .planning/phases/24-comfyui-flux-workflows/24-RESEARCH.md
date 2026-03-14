# Phase 24: ComfyUI Flux.1 Workflows - Research

**Researched:** 2026-03-14
**Domain:** ComfyUI Flux.1 Dev workflow templates, dynamic builder functions, keyframe pipeline routing
**Confidence:** HIGH

## Summary

This phase introduces Flux.1 Dev as an image generation backend running through ComfyUI, alongside the existing Qwen and Vertex AI (Gemini) image models. The codebase already has a well-established pattern for ComfyUI workflow templates (JSON API format in `backend/vidpipe/services/comfyui_templates/`), builder functions that deep-copy and inject runtime parameters (`build_qwen_txt2img_workflow`, `build_wan22_i2v_workflow`), and routing logic via model-set membership (`COMFYUI_IMAGE_MODELS` in `keyframes.py`). The Phase 23 tag resolver (`resolve_tags_with_assets`) already produces `ResolvedAssetRef` objects with `asset_type`, `reference_image_urls`, and `lora_url` fields -- all the metadata needed for this phase's categorization logic.

The primary work is: (1) create four ComfyUI API-format JSON workflow templates for Flux.1 Dev variants, (2) add a `build_flux_txt2img_workflow()` builder function to `comfyui_client.py`, (3) add Flux model IDs to `COMFYUI_IMAGE_MODELS` and `ALLOWED_IMAGE_MODELS`, (4) add binding-based reference categorization in `keyframes.py` that feeds CHARACTER refs to LoRA and PROP/SET refs to UNO reference images, and (5) add frontend model options.

**Primary recommendation:** Follow the existing template + builder pattern exactly (qwen-txt2img.json is the closest analog). Store four workflow JSON files in `backend/vidpipe/services/comfyui_templates/`. The builder function selects template based on LoRA/ref presence. Reference image upload to ComfyUI uses the existing `client.upload_image()` method before workflow submission.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Four templates stored in `backend/vidpipe/templates/comfyui/` (or `docs/comfyui/`): flux_txt2img_base.json, flux_txt2img_with_lora.json, flux_txt2img_with_references.json, flux_txt2img_full.json
- Templates are ComfyUI API-format JSON (node graph), not frontend workflow format
- `build_flux_txt2img_workflow()` in `comfyui_client.py` dynamically selects template based on: no LoRA/no refs -> base; LoRA only -> with_lora; refs only -> with_references; both -> full
- Parameters: prompt, negative_prompt, width, height, seed, lora_filename (optional), lora_strength (default 0.8), reference_image_filenames (optional), reference_strengths (optional)
- New Flux model IDs: flux-dev, flux-dev-lora, flux-dev-redux, flux-dev-full
- Added to `COMFYUI_IMAGE_MODELS` set in `keyframes.py` (the image model router)
- Both the main pipeline AND any regenerate paths must check this set
- When scene has `production_bible_id` AND prompt contains tags, use binding-based path
- `resolve_tags_with_assets()` (from Phase 23) provides `ResolvedAssetRef[]`
- Categorize by type: CHARACTER refs -> LoRA path, PROP/SET refs -> UNO reference images
- If CHARACTER has `lora_url` -> use LoRA; if not -> use reference images as UNO input
- Falls back to existing keyframe generation path when no bindings or non-Flux model selected
- Frontend Flux model entries in `IMAGE_MODELS` in `constants.ts`

### Claude's Discretion
- Exact ComfyUI node IDs and wiring in workflow templates (depends on available custom nodes)
- UNO vs Redux choice for reference injection (UNO preferred per PRD)
- Reference image upload mechanism to ComfyUI server (may need upload endpoint call before workflow queue)
- LoRA file download/caching strategy on ComfyUI server
- Error handling for missing LoRA files or unavailable reference images

### Deferred Ideas (OUT OF SCOPE)
- LoRA training pipeline (dataset prep, training dispatch, status tracking) -> Phase 25
- Frontend @tag autocomplete in scene editor -> Phase 26
- Frontend tag preview panel -> Phase 26
- Reference image strength tuning UI (per-binding overrides) -> Future
- Multiple LoRA merging for multi-character shots -> Future
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FLUX-01 | Flux.1 Dev base text-to-image workflow template | Workflow node structure documented; follows existing qwen-txt2img.json pattern |
| FLUX-02 | Flux.1 Dev + dynamic LoRA loader workflow template | LoraLoaderModelOnly node wiring documented; existing Qwen template already uses LoRA loader |
| FLUX-03 | Flux.1 Dev + UNO/Redux reference injection for up to 3 refs | UNO custom node (ComfyUI-UNO-Flux) structure documented; LoadImage + UNO conditioning path |
| FLUX-04 | Full hybrid Flux.1 Dev + LoRA + UNO workflow template | Combines FLUX-02 and FLUX-03 node chains |
| FLUX-05 | build_flux_txt2img_workflow() builder function | Follows build_qwen_txt2img_workflow() pattern exactly; template selection via LoRA/ref presence |
| FLUX-06 | Flux model IDs in COMFYUI_IMAGE_MODELS with routing | Identical to existing COMFYUI_VIDEO_MODELS routing pattern; 6 touchpoints identified |
| FLUX-07 | Binding-based reference resolution categorizing ResolvedAssetRefs | resolve_tags_with_assets() already returns typed ResolvedAssetRef[]; categorization is pure Python filtering |
| FLUX-08 | Frontend Flux model options in IMAGE_MODELS catalog | Follows existing IMAGE_MODELS array pattern in constants.ts |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ComfyUI API | cloud.comfy.org | Workflow execution engine | Already integrated via `comfyui_client.py` singleton |
| ComfyUI-UNO-Flux | latest | UNO reference conditioning nodes for Flux | Preferred reference injection per PRD; alternative to Redux |
| Flux.1 Dev | flux1-dev.safetensors | Base diffusion model | Open-source, high-quality, LoRA-compatible |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| httpx | existing | Image upload to ComfyUI | Already used in ComfyUIClient.upload_image() |
| copy (stdlib) | - | Deep-copy workflow templates | Existing pattern for all workflow builders |
| json (stdlib) | - | Load/parse workflow JSON templates | Existing pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| UNO reference injection | Flux Redux | Redux is simpler but UNO supports multiple refs and is PRD-recommended |
| UNO reference injection | IP-Adapter | Good for faces but less general; UNO handles props/sets better |
| Template JSON files | Programmatic node construction | Templates are maintainable, debuggable, exportable from ComfyUI GUI |

**Installation:** No new Python dependencies needed. The ComfyUI-UNO-Flux custom node must be installed on the ComfyUI server (not in this codebase).

## Architecture Patterns

### Recommended Template Location
```
backend/vidpipe/services/comfyui_templates/
    qwen-txt2img.json          # existing
    qwen-image-edit.json       # existing
    wan-i2v.json               # existing
    flux-txt2img-base.json     # NEW (FLUX-01)
    flux-txt2img-with-lora.json    # NEW (FLUX-02)
    flux-txt2img-with-refs.json    # NEW (FLUX-03)
    flux-txt2img-full.json     # NEW (FLUX-04)
```

Note: CONTEXT.md mentions `backend/vidpipe/templates/comfyui/` as a possible location, but the existing templates live in `backend/vidpipe/services/comfyui_templates/`. Use the existing location for consistency.

### Pattern 1: Workflow Template + Builder Function (Established Pattern)
**What:** JSON template defines the static node graph; builder function deep-copies and injects runtime parameters.
**When to use:** For every ComfyUI workflow variant.
**Example (existing pattern from codebase):**
```python
# From comfyui_client.py - this is the exact pattern to follow
_FLUX_BASE_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "flux-txt2img-base.json"
)
_cached_flux_base_template: Optional[dict] = None

def _load_flux_base_template() -> dict:
    global _cached_flux_base_template
    if _cached_flux_base_template is None:
        with open(_FLUX_BASE_TEMPLATE_PATH) as f:
            _cached_flux_base_template = json.load(f)
    return _cached_flux_base_template

def build_flux_txt2img_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    lora_filename: Optional[str] = None,
    lora_strength: float = 0.8,
    reference_image_filenames: Optional[list[str]] = None,
    reference_strengths: Optional[list[float]] = None,
) -> dict:
    # Select template based on what's provided
    has_lora = lora_filename is not None
    has_refs = reference_image_filenames and len(reference_image_filenames) > 0

    if has_lora and has_refs:
        template = _load_flux_full_template()
    elif has_lora:
        template = _load_flux_lora_template()
    elif has_refs:
        template = _load_flux_refs_template()
    else:
        template = _load_flux_base_template()

    workflow = copy.deepcopy(template)
    # Inject prompt, seed, dimensions into known node IDs
    # Inject lora_filename, lora_strength if applicable
    # Inject reference image filenames if applicable
    return workflow
```

### Pattern 2: Model Routing via Set Membership (Established Pattern)
**What:** A module-level `set` of model IDs determines routing to ComfyUI vs Vertex AI.
**When to use:** For every new ComfyUI-backed model.
**Example (existing pattern from codebase):**
```python
# In keyframes.py - extend the existing set
COMFYUI_IMAGE_MODELS = {
    "qwen-fast", "qwen-image-edit",
    "flux-dev", "flux-dev-lora", "flux-dev-redux", "flux-dev-full",
}

# In routes.py - extend the gatekeeper
ALLOWED_IMAGE_MODELS = {
    "gemini-2.5-flash-image", "gemini-3-pro-image-preview",
    "qwen-fast", "qwen-image-edit",
    "flux-dev", "flux-dev-lora", "flux-dev-redux", "flux-dev-full",
}
```

### Pattern 3: Binding-Based Reference Categorization
**What:** After resolving tags via `resolve_tags_with_assets()`, categorize `ResolvedAssetRef[]` by `asset_type` for the image generation adapter.
**When to use:** When scene has `production_bible_id` AND prompt contains tags AND Flux model is selected.
**Example:**
```python
# In the keyframe pipeline, after resolve_tags_with_assets()
char_refs = [r for r in resolved.asset_refs if r.asset_type == "CHARACTER"]
prop_refs = [r for r in resolved.asset_refs if r.asset_type == "PROP"]
set_refs  = [r for r in resolved.asset_refs if r.asset_type == "SET"]

# Determine LoRA (from first CHARACTER with lora_url)
lora_filename = None
lora_strength = 0.8
for ref in char_refs:
    if ref.lora_url:
        lora_filename = ref.lora_url  # Will need download/cache logic
        break

# Collect reference images (PROP + SET refs, plus CHARACTERs without LoRA)
reference_filenames = []
for ref in prop_refs + set_refs:
    for url in ref.reference_image_urls[:1]:  # First ref image per asset
        # Upload to ComfyUI, collect filename
        uploaded_name = await comfy_client.upload_image(image_bytes, f"ref_{ref.tag}.png")
        reference_filenames.append(uploaded_name)

# Characters without LoRA also go to UNO reference path
for ref in char_refs:
    if not ref.lora_url and ref.reference_image_urls:
        image_bytes = await _download_ref_image(ref.reference_image_urls[0])
        uploaded_name = await comfy_client.upload_image(image_bytes, f"ref_{ref.tag}.png")
        reference_filenames.append(uploaded_name)
```

### Pattern 4: ComfyUI Image Generation + Poll (Established Pattern)
**What:** Queue workflow, poll for completion, download output image.
**When to use:** For all ComfyUI image generation calls.
**Example (exact pattern from existing `_generate_image_comfyui`):**
```python
async def _generate_image_comfyui_flux(
    comfy_client,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    lora_filename: Optional[str] = None,
    lora_strength: float = 0.8,
    reference_image_filenames: Optional[list[str]] = None,
    reference_strengths: Optional[list[float]] = None,
) -> bytes:
    workflow = build_flux_txt2img_workflow(
        prompt=prompt, width=width, height=height, seed=seed,
        lora_filename=lora_filename, lora_strength=lora_strength,
        reference_image_filenames=reference_image_filenames,
        reference_strengths=reference_strengths,
    )
    prompt_id = await comfy_client.queue_prompt(workflow)
    # Same poll loop as _generate_image_comfyui
    max_polls, poll_interval = 120, 3
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval)
        status, error_msg = await comfy_client.poll_status(prompt_id)
        if status == "success":
            break
        if status in ("error", "failed", "cancelled"):
            raise RuntimeError(f"ComfyUI Flux job {prompt_id} failed: {error_msg}")
    else:
        raise RuntimeError(f"ComfyUI Flux job {prompt_id} timed out")
    history = await comfy_client.get_history(prompt_id)
    filename, subfolder = find_comfyui_image_output(history, prompt_id)
    return await comfy_client.download_output(filename, subfolder)
```

### Anti-Patterns to Avoid
- **Single model ID for all Flux variants:** The CONTEXT.md specifies four distinct model IDs (flux-dev, flux-dev-lora, flux-dev-redux, flux-dev-full). This allows users to explicitly choose their workflow variant per scene. Do NOT collapse them into a single "flux-dev" with auto-detection of what's available.
- **Storing templates outside the established directory:** The existing templates are in `backend/vidpipe/services/comfyui_templates/`. Do not create a new `backend/vidpipe/templates/` directory.
- **Forgetting regeneration paths:** CLAUDE.md CRITICAL: Both the main pipeline AND `_regenerate_clip`/`_regenerate_keyframe` in `routes.py` AND `asset_library.py` generation endpoints must check `COMFYUI_IMAGE_MODELS`. There are currently 6+ places that import and check this set.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Workflow JSON template | Hand-craft node graphs in Python code | Export from ComfyUI GUI, store as JSON | Templates are debuggable, version-controlled, can be tested in ComfyUI GUI directly |
| ComfyUI API client | Custom HTTP client for ComfyUI | Existing `ComfyUIClient` class | Already handles auth, upload, queue, poll, download with proper error handling |
| Reference image uploading | Custom upload logic | `comfy_client.upload_image()` | Handles multipart upload, returns server-side filename |
| Tag resolution | Custom binding lookup | `resolve_tags_with_assets()` from Phase 23 | Batch-loads all bindings in 3 queries, returns typed ResolvedAssetRef[] |
| Image output extraction | Custom history parsing | `find_comfyui_image_output()` | Already handles both direct and scanned SaveImage nodes |

**Key insight:** 90% of the infrastructure already exists. The new work is primarily: four JSON files, one builder function, set membership additions, and reference categorization logic in keyframes.py.

## Common Pitfalls

### Pitfall 1: Missing COMFYUI_IMAGE_MODELS Check in Regeneration Paths
**What goes wrong:** Flux model selected but regeneration routes to Vertex AI instead of ComfyUI.
**Why it happens:** `routes.py` has `_regenerate_keyframe()`, `asset_library.py` has generate-appearance endpoints -- all import and check `COMFYUI_IMAGE_MODELS`. New model IDs must be in the set or these paths break silently.
**How to avoid:** After adding Flux IDs to `COMFYUI_IMAGE_MODELS`, grep the entire codebase for `COMFYUI_IMAGE_MODELS` and verify every usage site handles the new model IDs.
**Warning signs:** Flux model selected in UI but keyframes look like Gemini output (different style, different resolution).

### Pitfall 2: Missing ALLOWED_IMAGE_MODELS Gatekeeper
**What goes wrong:** API rejects Flux model IDs with 422 error.
**Why it happens:** `ALLOWED_IMAGE_MODELS` in `routes.py` is the validation gatekeeper. If Flux IDs aren't added there, all API calls with Flux models fail validation.
**How to avoid:** Add all four Flux model IDs to `ALLOWED_IMAGE_MODELS` in `routes.py`.
**Warning signs:** 422 "Invalid image_model" errors when selecting Flux models.

### Pitfall 3: UNO Custom Node Not Available on ComfyUI Server
**What goes wrong:** Workflow submission fails with "node type not found" error.
**Why it happens:** The UNO reference injection templates use custom nodes (from ComfyUI-UNO-Flux extension) that must be installed on the ComfyUI server. This is an infrastructure dependency, not a code dependency.
**How to avoid:** The builder function should gracefully handle this: if UNO nodes are required but the server doesn't have them, fall back to the base template and log a warning. The `with_references` and `full` templates should only be used when the user explicitly selects a model ID that implies UNO support (flux-dev-redux, flux-dev-full).
**Warning signs:** "Error" status on ComfyUI job poll, error message mentioning unknown node type.

### Pitfall 4: Reference Image Download/Upload Latency
**What goes wrong:** Keyframe generation takes much longer with references because each reference image must be downloaded from storage and uploaded to ComfyUI before the workflow can be queued.
**Why it happens:** ResolvedAssetRef carries `reference_image_urls` (S3/local paths). These need to be read and then uploaded via `comfy_client.upload_image()`.
**How to avoid:** Upload reference images in parallel using `asyncio.gather()`. Limit to max 3 reference images per workflow (matching template capacity). Cache uploaded filenames within a session to avoid re-uploading the same image for multiple shots.
**Warning signs:** 10+ second delays before workflow even starts generating.

### Pitfall 5: Flux Resolution Mismatch
**What goes wrong:** Flux generates at 1024x1024 regardless of scene aspect ratio.
**Why it happens:** Unlike Gemini which accepts `aspect_ratio` string, Flux needs explicit pixel dimensions. The workflow template needs width/height injection.
**How to avoid:** Map aspect ratios to Flux-native resolutions: `16:9` -> 1024x576, `9:16` -> 576x1024, `1:1` -> 1024x1024. Use a resolution mapping dict similar to `_WAN_RESOLUTIONS`.
**Warning signs:** Square images when 16:9 was requested.

### Pitfall 6: LoRA File Not on ComfyUI Server
**What goes wrong:** Workflow fails because LoRA .safetensors file isn't in the ComfyUI server's loras directory.
**Why it happens:** Actor.lora_url will be an S3 path (Phase 25), but ComfyUI expects the file to already exist in its `models/loras/` directory. For cloud.comfy.org, the file may need to be uploaded or referenced differently.
**How to avoid:** For Phase 24 (without Phase 25 training), the LoRA filename is optional and typically None. When it IS present, the builder should assume the file exists on the server (Phase 25 handles the download/caching). Error handling should catch workflow failures related to missing LoRA and log clearly.
**Warning signs:** ComfyUI job fails with "LoRA not found" type error.

## Code Examples

### Flux.1 Dev Base Workflow Template (API Format)
```json
{
  "10": {
    "inputs": { "vae_name": "ae.safetensors" },
    "class_type": "VAELoader",
    "_meta": { "title": "Load VAE" }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp16.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux"
    },
    "class_type": "DualCLIPLoader",
    "_meta": { "title": "DualCLIPLoader" }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": { "title": "Load Diffusion Model" }
  },
  "6": {
    "inputs": {
      "text": "A cinematic scene...",
      "clip": ["11", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "CLIP Text Encode (Positive)" }
  },
  "7": {
    "inputs": {
      "text": "",
      "clip": ["11", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "CLIP Text Encode (Negative)" }
  },
  "5": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage",
    "_meta": { "title": "EmptySD3LatentImage" }
  },
  "13": {
    "inputs": { "shift": 1.15, "model": ["12", 0] },
    "class_type": "ModelSamplingFlux",
    "_meta": { "title": "ModelSamplingFlux" }
  },
  "3": {
    "inputs": {
      "seed": 0,
      "steps": 20,
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0,
      "model": ["13", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    },
    "class_type": "KSampler",
    "_meta": { "title": "KSampler" }
  },
  "8": {
    "inputs": { "samples": ["3", 0], "vae": ["10", 0] },
    "class_type": "VAEDecode",
    "_meta": { "title": "VAE Decode" }
  },
  "9": {
    "inputs": {
      "filename_prefix": "flux-output",
      "images": ["8", 0]
    },
    "class_type": "SaveImage",
    "_meta": { "title": "Save Image" }
  }
}
```

### Flux.1 Dev + LoRA Template (Additional Nodes)
```json
{
  "14": {
    "inputs": {
      "lora_name": "placeholder.safetensors",
      "strength_model": 0.8,
      "model": ["12", 0]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": { "title": "Load LoRA" }
  }
}
```
Note: In the LoRA template, node 13 (ModelSamplingFlux) connects to node 14's output instead of node 12 directly: `"model": ["14", 0]`.

### Flux Resolution Mapping
```python
_FLUX_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (1024, 1024),
}
```

### Frontend Constants Addition
```typescript
// In constants.ts IMAGE_MODELS array
{ id: "flux-dev", label: "Flux.1 Dev", costPerImage: 0.00 },
{ id: "flux-dev-lora", label: "Flux.1 Dev + LoRA", costPerImage: 0.00 },
{ id: "flux-dev-redux", label: "Flux.1 Dev + Refs", costPerImage: 0.00 },
{ id: "flux-dev-full", label: "Flux.1 Dev Full", costPerImage: 0.00 },
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Gemini native image gen only | ComfyUI for Qwen + Wan models | Pre-existing | ComfyUI is already a first-class generation backend |
| No reference image injection for ComfyUI | UNO/Redux for Flux | This phase | Enables asset-consistent generation via ComfyUI |
| Asset refs only as text descriptions | Typed ResolvedAssetRef with LoRA/ref URLs | Phase 23 | Foundation for LoRA + reference routing |

**Key infrastructure already in place:**
- ComfyUI client singleton with upload/queue/poll/download
- Three existing workflow templates (qwen-txt2img, qwen-image-edit, wan-i2v)
- `COMFYUI_IMAGE_MODELS` routing pattern in keyframes.py
- `ALLOWED_IMAGE_MODELS` validation in routes.py
- `ResolvedAssetRef` dataclass with `lora_url`, `reference_image_urls`, `asset_type`
- `resolve_tags_with_assets()` batch-loading function

## Open Questions

1. **Exact UNO node class_type names for cloud.comfy.org**
   - What we know: The ComfyUI-UNO-Flux extension provides `UNOModelLoader` and `UNOGenerate` nodes
   - What's unclear: Whether cloud.comfy.org supports this extension natively, or if a simpler LoadImage-based approach is needed
   - Recommendation: Build the `with_references` template using standard LoadImage nodes feeding into a conditioning path. If the ComfyUI server has UNO installed, the template works as-is. If not, the base and LoRA-only templates still work. The template can be swapped when we confirm server capabilities.

2. **LoRA file availability on ComfyUI server**
   - What we know: ComfyUI expects LoRA files in `models/loras/`. For cloud.comfy.org, files may need uploading.
   - What's unclear: Does cloud.comfy.org support dynamic LoRA file upload, or only pre-installed models?
   - Recommendation: For Phase 24, treat lora_filename as a server-side path that the user or Phase 25 ensures exists. Add error handling that degrades to base template if LoRA loading fails.

3. **Flux model file names on cloud.comfy.org**
   - What we know: Standard names are flux1-dev.safetensors, ae.safetensors, t5xxl_fp16.safetensors, clip_l.safetensors
   - What's unclear: cloud.comfy.org may use different filenames or have models pre-installed
   - Recommendation: Use standard names in templates. If cloud.comfy.org uses different names, the templates are easy to update (single JSON file edit).

## Touchpoint Inventory (All Files Requiring Changes)

This is critical for the planner -- every file that needs modification:

### Backend Changes
| File | Change | Reason |
|------|--------|--------|
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-base.json` | NEW | FLUX-01 |
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-with-lora.json` | NEW | FLUX-02 |
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-with-refs.json` | NEW | FLUX-03 |
| `backend/vidpipe/services/comfyui_templates/flux-txt2img-full.json` | NEW | FLUX-04 |
| `backend/vidpipe/services/comfyui_client.py` | ADD builder function + template loaders | FLUX-05 |
| `backend/vidpipe/pipeline/keyframes.py` | ADD Flux IDs to `COMFYUI_IMAGE_MODELS`, ADD binding-based categorization, ADD `_generate_image_comfyui_flux()` | FLUX-06, FLUX-07 |
| `backend/vidpipe/api/routes.py` | ADD Flux IDs to `ALLOWED_IMAGE_MODELS` | FLUX-06 |

### Frontend Changes
| File | Change | Reason |
|------|--------|--------|
| `frontend/src/lib/constants.ts` | ADD Flux entries to `IMAGE_MODELS` array | FLUX-08 |

### Verification Touchpoints (files that import COMFYUI_IMAGE_MODELS -- must work with new IDs)
| File | Import Site | Behavior |
|------|-------------|----------|
| `backend/vidpipe/pipeline/keyframes.py:514` | `is_comfyui = image_model in COMFYUI_IMAGE_MODELS` | Routes to ComfyUI -- needs Flux-specific handler |
| `backend/vidpipe/pipeline/video_gen.py:291` | `if image_model in COMFYUI_IMAGE_MODELS` | Regenerates safety escalation keyframes -- needs Flux handler |
| `backend/vidpipe/api/routes.py:4977-4979` | `_regenerate_keyframe` | Must route Flux to ComfyUI |
| `backend/vidpipe/api/asset_library.py:861,1426,1904` | Actor/Set/Prop generate-appearance | Must route Flux to ComfyUI |

## Sources

### Primary (HIGH confidence)
- Existing codebase: `comfyui_client.py`, `keyframes.py`, `video_gen.py`, `tag_resolver.py`, `routes.py`, `constants.ts` -- all read and analyzed in full
- `docs/assets_mapping.md` Section 5 (ComfyUI Workflow Design for Flux.1) -- PRD spec driving this phase

### Secondary (MEDIUM confidence)
- [ComfyUI Flux.1 Text-to-Image Tutorial](https://docs.comfy.org/tutorials/flux/flux-1-text-to-image) -- Official node types and configuration
- [Flux.1 Dev ComfyUI Guide](https://comfyui-wiki.com/en/tutorial/advanced/image/flux/flux-1-dev-t2i) -- DualCLIPLoader, UNETLoader, EmptySD3LatentImage nodes
- [ComfyUI LoRA Example](https://docs.comfy.org/tutorials/basic/lora) -- LoraLoaderModelOnly node inputs
- [UNOModelLoader Documentation](https://comfyai.run/documentation/UNOModelLoader) -- UNO node inputs and outputs
- [ComfyUI-UNO-Flux GitHub](https://github.com/alexgenovese/ComfyUI-UNO-Flux) -- UNO extension for Flux
- [Flux Resolution Comparisons](https://blog.segmind.com/image-resolutions-with-flux-1-dev-model-compared/) -- Resolution recommendations

### Tertiary (LOW confidence)
- [Flux UNO Multiple Image Reference Workflow](https://openart.ai/workflows/cat_untimely_42/flux-uno-multiple-image-reference/WfkqpbROylpsxuTedjAQ) -- Community workflow example
- Exact cloud.comfy.org model file names -- needs runtime verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, extending established patterns
- Architecture: HIGH -- follows exact codebase patterns (template + builder + routing set)
- Workflow JSON node structure: MEDIUM -- node class_types are well-documented but exact wiring needs validation against target ComfyUI server
- Pitfalls: HIGH -- identified from CLAUDE.md warnings and codebase analysis

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable domain, ComfyUI node types evolve slowly)
