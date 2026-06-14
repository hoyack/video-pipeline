# Face-Swap Plan — Model-Agnostic Exact-Face Locking

**Status:** POC validated 2026-06-13 — pipeline integration is the remaining work.
**Decided:** 2026-06-13 — use **face-swap** as the primary identity-locking method (chosen over LoRA for speed/flexibility; LoRA plan in `docs/lora-plan.md` remains the heavier alternative).

> **POC RESULT (validated):** `inswapper_128.onnx` + buffalo_l (CPU, onnxruntime 1.26 in the backend container) swapping Brandon's real primary ref onto an in-scene keyframe moved face-embedding similarity **0.17 → 0.87** and is visually the exact real Brandon. CPU is fast enough for a 128px swap. **Start this plan at §3 (integration); §6 step 1 is done.**

## 1. Goal & why this wins
Reference-based identity (synth refs, Nano Banana, wardrobe looks) tops out ~0.75–0.85 vision identity and never *replicates* a specific face — it approximates. A **post-generation face swap** pastes the character's **real** face onto each generated keyframe, giving near-exact identity.

Key consequence: **identity is decoupled from the image model.** Keyframes can be generated with any model (Qwen `qwen-image-edit-2509`, Flux `flux-2-klein`/`flux-dev`, etc.) chosen purely for composition/cost/quality — **Nano Banana is no longer required for face fidelity.** The face-swap step restores the exact face regardless of generator. Default keyframe model for testing → switch back to **Qwen / Flux** (cheaper, open).

Runs on the **local GPU** (RTX 4070 Ti, CUDA verified; InsightFace already used here for QA detection).

## 2. Approach
- **Library:** InsightFace `FaceAnalysis` (already installed, `buffalo_l`) + the **inswapper** model (`inswapper_128.onnx`) for the swap. onnxruntime-gpu on the 4070 Ti (CUDAExecutionProvider).
- **Source face:** the character's clearest real reference (e.g. Brandon's primary `2fa30ea8`); cache the source embedding/face per character.
- **Target:** each generated keyframe. Detect the largest face → swap source identity onto it → (optional) face-restore for quality.
- **Model acquisition:** `inswapper_128.onnx` is gated/licensed — source it deliberately (note license). Optional quality boost: GFPGAN/CodeFormer face-restoration after swap. Consider `facefusion` as a turnkey alternative if inswapper sourcing is a problem.

## 3. Pipeline integration
- New service `backend/vidpipe/services/face_swap_service.py`:
  - `swap_face(target_png: bytes, source_png: bytes) -> bytes | None` — returns swapped image, or `None` if no face detected in target (stylized renders sometimes have undetectable faces — then skip gracefully).
  - Lazy-load swapper + detector (singleton, CPU/GPU provider).
- Hook in `backend/vidpipe/pipeline/keyframes.py`: **after** a keyframe is generated/verified and **before** clip generation, if the shot has on-screen character(s) with a real reference and the feature flag is on, run the swap and persist the swapped keyframe. For multi-character shots, swap each detected face to its nearest character source (start single-character).
- Config: `settings.face_swap_enabled` (+ per-scene override), `face_swap_restore` (GFPGAN on/off), `face_swap_min_det_score`.
- Source-face selection: reuse the QA face-clarity ranking to pick the best real ref as the swap source.

## 4. Risks / mitigations
- **Stylized targets:** inswapper is trained on real faces; on heavily stylized cyberpunk renders the swap can look pasted or the detector can miss the face. Mitigate with face-restoration blending and a det-score threshold; **POC on one keyframe first** before integrating.
- **Pose/angle:** swaps best on near-frontal faces; extreme angles degrade — acceptable since establishing shots are usually frontal.
- **Temporal consistency:** we swap the *keyframes*, and the FLF2V video models interpolate from them, so the swapped identity carries into the clip. If the clip still drifts mid-shot, a later enhancement is per-frame swap on the rendered clip (heavier).
- **Licensing:** confirm inswapper usage terms for the project's context.

## 5. Verification
- QA harness (`vidpipe.qa`): `identity_match` should jump to ~0.9+ and **embeddings should rise materially** (the swapped face IS the real face — unlike stylized generations where embeddings stay low). This is the acceptance metric.

## 6. Sequence (next session)
1. ~~**POC:** acquire `inswapper_128.onnx`; swap Brandon's real face onto a keyframe; QA-score.~~ ✅ **DONE — 0.17→0.87, validated.** Model fetched from `huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx` (554MB) to `/root/.insightface/models/inswapper_128.onnx` in the backend container. **This is ephemeral (lost on rebuild) — first integration task: bake it into the backend image or mount a volume.**
2. Build `face_swap_service.py` + the keyframe post-hook + config flag.
3. Switch keyframe `image_model` to **Qwen or Flux** for a test production; enable face-swap; regenerate; QA-compare faces vs the Nano Banana baseline.
4. Tune (restoration, det threshold); roll into the standard pipeline.

## 7. Relationship to other plans
- Replaces the need for Nano Banana-for-faces and (for now) the LoRA (`docs/lora-plan.md`) — keep LoRA as a fallback for cases where face-swap underperforms (e.g. non-frontal heroes).
- Complements the wardrobe-look pipeline (`/actor-wardrobe-presets/{id}/generate-image`), which is built but was never exercised for Brandon (presets have 0 images, 0 looks bound) — wardrobe refs still help composition/wardrobe even though face is now handled by the swap.
