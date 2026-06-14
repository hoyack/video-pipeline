# Character LoRA Plan — Identity Locking for Open Models

**Status:** proposal / not yet implemented
**Author:** generated 2026-06-13
**Goal:** Lock a character's *exact face* (e.g. `@BRANDON`) across generated keyframes by training a per-model identity **LoRA**, training it on the **local GPU**, and installing the resulting weights **manually into Comfy Cloud**.

---

## 1. Why this is needed

Reference images alone do not lock identity well enough on the open (ComfyUI) image models. Observed behavior:

- The real Brandon photos are **off-domain** (home-studio, casual) for cyberpunk-noir scenes — the model borrows wardrobe but drifts on the face.
- We added **on-domain synthetic cyberpunk refs** (base portrait scored 0.54 face-embedding similarity) and raised reference quality, which helps, but **flux-2-klein's reference adherence is weak**: in-scene faces still land at ~0.75 vision identity (embeddings 0.04–0.31 on stylized frames).
- The Gemini path (Nano Banana / `gemini-2.5-flash-image`) gives better identity (~0.85) but **cannot use a custom LoRA** — it is a closed API model. LoRAs only help the **open** models.

A character LoRA bakes the identity into the model weights → the strongest, most consistent face lock available for the open models.

> Identity LoRAs apply to the **image / keyframe** models. The video models (LTX, WAN, Seedance) are first-last-frame interpolators driven by the keyframes, so locking the keyframe identity is what matters; per-video-model character LoRAs are out of scope.

---

## 2. One LoRA per open model (architecture-specific)

LoRA weights are tied to a specific base-model architecture — a FLUX LoRA will not load on Qwen, and a FLUX.1 LoRA will not load on FLUX.2. So **each open image model we use needs its own LoRA per character.**

Open image models in use (`COMFYUI_IMAGE_MODELS` / `COMFYUI_MULTIREF_IMAGE_MODELS` in `backend/vidpipe/pipeline/keyframes.py`):

| Model id | Base architecture | LoRA needed? | Notes |
|---|---|---|---|
| `flux-dev-lora` | FLUX.1-dev | **Yes** (primary) | Already has a LoRA workflow template (`flux-txt2img-with-lora.json`). The natural first target. |
| `flux-2-klein` | FLUX.2 Klein | **Yes** | Confirm Klein LoRA training is supported by the training toolkit before committing. |
| `qwen-image-edit-2509` | Qwen-Image | **Yes** | Separate Qwen-Image LoRA; different trainer config. |

Closed models that **cannot** use a LoRA (skip): `gemini-2.5-flash-image` (Nano Banana), `gemini-3-pro-image-preview` (Nano Banana Pro), Veo.

**Matrix:** `LoRA = {character} × {open image model}`. For Brandon across all three open models = 3 LoRA files. Start with **FLUX.1-dev** (existing workflow support) and expand.

---

## 3. Training on the local GPU

**Hardware:** NVIDIA RTX 4070 Ti, **12 GB VRAM** (verified: WSL2 `/dev/dxg`, host `torch 2.10+cu128`, CUDA available, Docker `nvidia` runtime).

**Toolkit:** [ai-toolkit](https://github.com/ostris/ai-toolkit) (Ostris) — supports FLUX.1 and FLUX.2 LoRA training with low-VRAM modes; kohya `sd-scripts` is the alternative. Neither is installed yet.

**12 GB VRAM is the binding constraint.** FLUX LoRA training normally wants 16–24 GB. On 12 GB it is only feasible with aggressive settings:
- fp8 / quantized base weights
- 512 px training resolution
- batch size 1 + gradient accumulation
- gradient checkpointing, low-rank (rank 16–32)
- expect **multi-hour** runs and OOM risk — tune down if it OOMs.

**Dataset (per character):** the actor's reference images. Brandon currently has **9** (4 real + 5 on-domain synthetic cyberpunk; ≥ the 5-image minimum the app enforces). Prefer 15–30 varied, clear-face images for best results; the synthetic cyberpunk variants are valuable because they are on-domain. Auto-caption each image starting with a unique **trigger word** (the app already uses `ACTOR_{NAME}`, e.g. `ACTOR_BRANDON`) — see `lora_trainer.py::prepare_dataset`.

**Per-model configs:** maintain one training config per architecture (FLUX.1-dev, FLUX.2 Klein, Qwen-Image) — base checkpoint, rank, resolution, steps differ. Keep them under e.g. `scripts/lora/configs/`.

---

## 4. Manual install into Comfy Cloud

We use **Comfy Cloud** (`COMFY_UI_HOST=https://cloud.comfy.org`), not a self-hosted ComfyUI, so trained LoRAs must be placed on Comfy Cloud's model storage **manually** before a workflow can reference them:

1. Train locally → produce `{character}_{model}.safetensors` (e.g. `brandon_flux1dev.safetensors`).
2. Upload the `.safetensors` into the Comfy Cloud account's **LoRA model directory** (`models/loras/`) via the Comfy Cloud UI / model-upload mechanism. **Confirm Comfy Cloud allows custom LoRA upload** — this is the biggest open risk; if it does not, we must run **ComfyUI locally** on the 4070 Ti and point `COMFY_UI_HOST` at it instead (FLUX inference on 12 GB is tight but workable).
3. Note the **exact server filename** — the workflow references the LoRA by filename (`workflow["14"]["inputs"]["lora_name"]`), not by URL.
4. Record that filename in the actor/character record so the pipeline can inject it (see §5).

Because this step is manual, the pipeline should treat the LoRA filename as **configuration**, not something it uploads.

---

## 5. Pipeline integration

Today (`backend/vidpipe/`):
- `lora_trainer.py` — **Replicate-cloud only** (`ReplicateBackend`, requires `settings.replicate_api_token`); `download_and_store_weights()` stores a single `.safetensors` at `asset-library/actors/{id}/lora/model.safetensors` and sets the actor's `lora_url`.
- `keyframes.py:~2757` — for a CHARACTER reference with `aref.lora_url`, sets `flux_lora` → `build_flux_txt2img_workflow(lora_filename=...)` (template `flux-txt2img-with-lora.json`, node `14`, default `lora_strength=0.8`).

Changes needed:
1. **Per-model LoRA storage on the actor.** Replace the single `lora_url` with a map `lora_by_model: {model_id: {comfy_filename, strength, trigger_word, trained_at}}`. Migration: keep `lora_url` working as the FLUX.1-dev entry.
2. **Local-GPU training backend.** Add a `LocalLoRATrainingBackend` to `lora_trainer.py` (runs ai-toolkit on the host GPU) so training does not require Replicate. Selectable via settings (`lora_training_backend = local | replicate`).
3. **Manual-install registration endpoint.** Since Comfy Cloud install is manual, add an endpoint to register a trained LoRA against `(actor, model)` with its Comfy Cloud filename + trigger word, without re-uploading.
4. **Keyframe routing per model.** When generating with an open model that has a registered LoRA for the character, inject the right filename + prepend the trigger word to the prompt. Extend beyond `flux-dev-lora` to `flux-2-klein` and `qwen-image-edit-2509` (each needs a LoRA-enabled workflow template — currently only flux has one).

---

## 6. Verification

Use the existing QA harness (`vidpipe.qa`) to measure the win:
- Generate establishing keyframes with and without the LoRA for the same character/scene.
- Compare **vision identity_match** (the reliable gate; embeddings are unreliable on stylized frames) and eyeball.
- Acceptance: establishing keyframe `identity_match ≥ 0.85` and visually unmistakable, beating the refs-only baseline (~0.70–0.75 on flux).

---

## 7. Recommended sequence

1. **FLUX.1-dev first** (existing workflow support): install ai-toolkit → train `brandon_flux1dev.safetensors` from the 9 refs (low-VRAM config) → manually install into Comfy Cloud → register filename → switch a test scene to `flux-dev-lora` → QA-compare vs Nano Banana.
2. If Comfy Cloud rejects custom LoRA upload → stand up **local ComfyUI** on the 4070 Ti and repoint `COMFY_UI_HOST`.
3. Once FLUX.1-dev is proven, repeat for **FLUX.2 Klein** and **Qwen-Image** (confirm toolkit support per architecture).
4. Generalize storage/routing to the per-model `lora_by_model` map.

## 8. Open risks
- **Comfy Cloud custom-LoRA upload** may not be supported → fallback to local ComfyUI.
- **12 GB VRAM** for FLUX training is tight → may need fp8/512px or fall back to cloud training for FLUX, local for lighter architectures.
- **FLUX.2 Klein LoRA** training support in ai-toolkit/kohya — verify before committing.
- LoRA strength tuning (`lora_strength`) trades identity lock vs prompt flexibility — expose per-generation.
