# ADR: New ComfyUI Cloud models — multi-ref image generation and FLF2V video

**Date:** 2026-06-11
**Status:** Accepted

## Context

The ComfyUI integration previously offered txt2img-only Qwen, single-image
qwen-image-edit, style-only Flux Redux references, and start-frame-only WAN
2.2 video. Production-bible identity references (the Nano Banana machinery)
were gated to Gemini models, and the end keyframe was loaded but never used
on the ComfyUI video path.

This phase adds:

| Model ID | Capability |
|---|---|
| `qwen-image-edit-2509` | 1–3 identity reference images (Apache 2.0) |
| `flux-2-klein` | 0–4 reference images, plain txt2img at 0 (Apache 2.0, 4B distilled) |
| `wan-2.2-flf2v` | Start+end keyframe interpolation, same weights as wan-2.2-i2v |
| `ltx-2.3-flf2v` | Start+end keyframes @25fps, 4–10s, native audio (open weights) |
| `seedance-2.0-flf2v` | Start+(optional)end, 4–15s, audio — paid ByteDance partner node |

## Decisions

### 1. Committed API-format templates, no runtime conversion

Official Comfy-Org templates use the subgraph app format, which
`_convert_app_to_api` cannot handle (and whose `_WIDGET_NAMES` positional
mapping silently mis-maps unknown node classes). We hand-assembled minimal
flat API-format graphs (`backend/vidpipe/services/comfyui_templates/`) from
the official subgraph internals, with raw app-format copies archived under
`docs/comfyui-templates/`. Node input names were verified against ComfyUI
source (`comfy_extras/*`, `comfy_api_nodes/nodes_bytedance.py`).

### 2. Qwen 2509 refs-as-inputs mapping

Qwen-Image-Edit is an edit model: the input images ARE the references.
Mapping onto our two keyframe modes:

- **End-frame mode:** `image1` = start frame (drives output dimensions and
  visual conditioning via ReferenceLatent), `image2/3` = identity refs.
  This gives ComfyUI models true start-frame-conditioned end keyframes for
  the first time.
- **Start-frame mode:** the template KSampler runs at denoise 1.0, so the
  `latent_image` input only determines output dimensions. We patch in an
  `EmptySD3LatentImage` at the scene aspect ratio and use all 3 slots for
  identity refs. With zero refs the model cannot run — we fall back to
  qwen-fast txt2img (logged).
- **Ref-slot reservation:** identity-ref selection is capped at 2 for qwen
  (3rd slot reserved for the start frame in end-frame mode) and 3 for klein
  (1 of 4 reserved).

### 3. ComfyUI models now use the Nano Banana reference machinery

`COMFYUI_MULTIREF_IMAGE_MODELS` widens the gate at keyframes.py so
face/wardrobe candidate selection, identity-policy face cropping, emphasis
escalation, and post-generation face verification all apply to
qwen-image-edit-2509 and flux-2-klein. **Consequence:** identity
verification (and its retries) is newly active for ComfyUI image models on
production-bible scenes — intentional, watch generation latency.

### 4. Video adapter spec registry

`COMFY_VIDEO_SPECS` in `comfyui_adapter.py` holds per-model behavior: fps,
fixed-vs-requested duration, end-frame/audio/char-ref support, and the
framing-safety prompt prefix (moved out of video_gen.py). `submit()` gained
`duration_seconds`/`audio_enabled`; `download()` gained
`video_model`/`duration_seconds` with WAN-compatible defaults so in-flight
`comfyui:{prompt_id}` operations survive a deploy. The operation-ID format
is unchanged. `COMFYUI_VIDEO_MODELS` (video_gen.py) and `COMFY_VIDEO_SPECS`
must stay in sync (pinned by a unit test).

### 5. FLF2V soft degrade

A missing end keyframe on an FLF2V model logs a warning and degrades to
first-frame-only generation (wan-flf2v → i2v workflow; ltx → single guide;
seedance → omit last_frame) instead of failing the shot. Rationale:
uploaded/gap-filled shots can legitimately lack an end keyframe.

### 6. Seedance pricing and auth

Cost set to $0.22/s (silent and audio) from the partner node's own pricing
expression: 21,600 tokens/s @720p × ~$0.01001/1K tokens, audio included.
**Known caveat:** partner API nodes authenticate via a logged-in Comfy Org
account, not the plain `X-API-Key` our client sends — the live smoke test
failed with "Unauthorized: Please login first to use this node" after the
workflow itself validated. Account-level setup (Comfy login / credit
balance / API-key partner-node permission) is required before
seedance-2.0-flf2v works end-to-end. The error surfaces cleanly in
`clip.error_message`.

### 7. Frontend provider field

Model options now carry `provider: "vertex" | "comfyui"`; the Settings-page
groupings filter on it instead of `id.startsWith(...)` (which would have
orphaned `ltx-*`/`seedance-*` from every group).

## Verification

- 34 unit tests (builders, adapter dispatch, catalog consistency) green;
  full backend suite 84 passed with only two pre-existing failures.
- Live Comfy Cloud smoke tests (backend/tests/comfyui/test_new_models.py):
  - qwen-2509: SUCCESS — 1664×928 output at requested scene dims
  - flux2-klein: SUCCESS — single-ref generation in ~10s
  - ltx: SUCCESS — 1280×704 @25fps h264 with non-silent AAC audio
    (VAE rounds 720→704; consistent across clips)
  - seedance: workflow accepted, blocked on partner-node account auth (see §6)

## Consequences

- LTX outputs are 25fps/704p-class vs WAN's 16fps/480p-class — mixed-model
  scenes rely on the ffmpeg stitcher normalizing (same as Veo 24fps today).
- Seedance is the first ComfyUI model with a real per-second cost; UI cost
  estimates now reflect it.
- The five flux-dev legacy models remain unchanged for backward
  compatibility.
