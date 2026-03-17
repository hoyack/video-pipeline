# Character Face Identity Preservation

How the pipeline ensures generated keyframes preserve character facial identity across scenes.

## Architecture Overview

```
ActorRef images ──► Prequalification ──► Qualified refs (with embeddings)
                        │                        │
Actor.base_appearance ──┤                        ▼
                        ▼               ┌────────────────────┐
               Identity Instruction     │  Image Generation   │
               (feature-anchored)       │  (Vertex or ComfyUI)│
                        │               └────────┬───────────┘
                        ▼                        │
                  Gemini/Flux                    ▼
                  generation  ◄──────── Reference images
                        │
                        ▼
               ┌────────────────────┐
               │  Face Verification  │
               │  (YOLO + ArcFace)   │
               └────────┬───────────┘
                        │
                   Pass? ──► Done
                   Fail? ──► Retry with escalation (up to Level 2)
                             Level 2+: feed best-so-far as conditioning
```

## Vertex AI Path (Nano Banana)

### Models

| Model ID | Alias | Cost/image | Best for |
|----------|-------|-----------|----------|
| `gemini-2.5-flash-image` | Nano Banana | $0.04 | Fast iteration, character consistency |
| `gemini-3-pro-image-preview` | Nano Banana Pro | $0.13 | Higher quality, better identity |

### How It Works

Gemini image generation accepts multimodal content: text + images mixed in a single request. We prepend reference photos with an identity instruction, then append the scene prompt.

**Contents sent to Gemini:**
```
[identity_instruction, ref_image_1, ref_image_2, ..., scene_prompt]
```

### Reference Image Prequalification

Not all reference images contain extractable faces. Sending bad refs causes "multi-reference confusion" where Gemini averages features across valid and invalid references.

**Service:** `vidpipe/services/ref_prequalification.py`

```python
qualified = await prequalify_refs(ref_urls, file_mgr, face_svc)
# Returns only refs where InsightFace detected a face
# Sorted by detection score (best face first)
```

Each `QualifiedRef` carries:
- `image_bytes` — for passing to Gemini
- `face_embedding` — 512-dim ArcFace vector for downstream verification
- `detection_score` — InsightFace confidence

**Impact:** In testing, prequalification improved match scores even when all 4 refs technically had faces — InsightFace detects faces that YOLO misses (upper-40% person bbox heuristic vs dedicated face analysis).

### Feature-Anchored Identity Prompts

Generic "match their face" instructions give weak identity retention. Injecting specific facial features from `Actor.base_appearance_prompt` improves retention ~41%.

**Function:** `_build_identity_instruction(text_description, emphasis_level)`

**Level 0 (standard):**
```
The following reference photo(s) show the EXACT person who must appear
in the generated image. This person's key features: {actor.base_appearance_prompt}
Match their face structure, skin tone, and distinguishing features precisely.
The generated character MUST be recognizable as the SAME PERSON.
```

**Level 1 (escalated):**
```
CRITICAL IDENTITY REQUIREMENT: The character's face must EXACTLY match
the reference photo(s). Key features: {actor.base_appearance_prompt}
Pay extreme attention to facial bone structure, eye shape and spacing,
nose bridge, jawline contour, and skin tone.
```

**Level 2 (maximum — sacrifice composition for identity):**
```
ABSOLUTE PRIORITY — FACE IDENTITY: Match the reference photos with
photographic accuracy. The person has: {actor.base_appearance_prompt}
The face is the most important element. Sacrifice background detail or
composition fidelity if needed to preserve exact face identity.
```

### Iterative Refinement Loop

The pipeline retries up to 4 times (Levels 0-3) with escalating emphasis:

| Level | Strategy | Identity instruction |
|-------|----------|---------------------|
| 0 | Standard text-to-image + refs | Feature-anchored (Level 0) |
| 1 | Escalated prompt prefix + refs | Feature-anchored (Level 1) |
| 2 | Image-to-image: best-so-far as conditioning + refs | Feature-anchored (Level 2) |
| 3 | Repeat Level 2 with different seed | Feature-anchored (Level 2) |

**Best-so-far tracking:** Each attempt is scored via ArcFace. The image with the highest similarity is kept regardless of pass/fail threshold.

**Level 2+ key insight:** At Level 2, the best attempt from Levels 0-1 is fed back to Gemini as a conditioning frame alongside the original references. This tells Gemini "keep this composition but improve the face match."

### Face Verification

**Function:** `_verify_keyframe_faces(keyframe_bytes, placed_char_assets, ref_embeddings)`

1. **YOLO detection:** `CVDetectionService.detect_faces_from_bytes()` — finds person bboxes, extracts upper-40% as face region
2. **Face crop:** 10% padding around detected face bbox
3. **ArcFace embedding:** `FaceMatchingService.generate_embedding_from_bytes()` — 512-dim normalized vector via InsightFace buffalo_l
4. **Cosine similarity:** Compare generated face embedding against each reference embedding
5. **Threshold:** `settings.cv_analysis.keyframe_face_match_threshold` (default 0.45)

**Embedding sources (dual path):**
- Legacy manifest: `Asset.face_embedding` bytes (precomputed at upload)
- CastBinding flow: `ref_embeddings` from prequalification service (computed on-the-fly, cached to `ActorRef.face_embedding`)

**Soft degradation:** Verification never blocks generation. Returns `(True, 0.0, reason)` when:
- No embeddings available
- No faces detected in generated image
- CV service errors

### ArcFace Similarity Score Reference

| Score | Interpretation |
|-------|---------------|
| > 0.60 | Definitely same person (production quality) |
| 0.45–0.60 | Probably same person (acceptable for keyframes) |
| 0.30–0.45 | Possibly same person (needs improvement) |
| 0.20–0.30 | Inconclusive |
| < 0.20 | Different person / noise floor |

**Our measured results:**

| Configuration | Best similarity |
|--------------|----------------|
| Baseline (generic prompt, all refs, no prequalification) | 0.3684 |
| With prequalification + feature-anchored prompt | **0.4744** |
| Target for production pipeline | 0.45–0.50 |

---

## ComfyUI Path (Flux Dev)

### Models

| Model ID | Identity mechanism | Notes |
|----------|-------------------|-------|
| `flux-dev` | unCLIPConditioning (CLIP Vision refs) | Reference-only, no training |
| `flux-dev-lora` | LoRA weight injection | Requires per-character training |
| `flux-dev-full` | LoRA + unCLIPConditioning | Highest fidelity (hybrid) |
| `qwen-fast` | Text only | No identity preservation |
| `qwen-image-edit` | Image conditioning | Edit-based, not identity-specific |

### Current Flux Reference Image Flow

```
ActorRef images ──► upload_image() ──► ComfyUI server
                                          │
Tag resolver ──► LoRA URL (.safetensors) ─┤
                                          ▼
                                 Template selection:
                                 ├─ LoRA + refs → flux-txt2img-full.json
                                 ├─ LoRA only  → flux-txt2img-with-lora.json
                                 ├─ refs only  → flux-txt2img-with-refs.json
                                 └─ neither    → flux-txt2img-base.json
```

**Ref image conditioning (in template):**
1. `LoadImage` nodes load uploaded refs (up to 3)
2. `ImageBatch` combines them
3. `CLIPVisionLoader` loads `clip_vision_g.safetensors`
4. `CLIPVisionEncode` produces vision embeddings from batch
5. `unCLIPConditioning` merges vision embeddings with text conditioning (strength 0.65)

**LoRA conditioning:**
1. `LoraLoaderModelOnly` loads actor-specific `.safetensors` weights
2. Merged into UNet at configurable strength (default 0.8)
3. LoRA trained via Replicate `ostris/flux-dev-lora-trainer` (25-30 images, ~30 min)

### DRY Status: What's Shared vs. Separate

| Component | Vertex AI | ComfyUI | Shared? |
|-----------|-----------|---------|---------|
| Ref prequalification | Used | **Not used** | Service is model-agnostic but only wired for Vertex |
| Identity instruction (`_build_identity_instruction`) | Used (3 levels) | **Not used** | ComfyUI prompt doesn't get feature anchoring |
| Face verification (`_verify_keyframe_faces`) | Called per attempt | **Not called** | Full DRY gap |
| Retry loop (`_max_identity_retries`) | 4 attempts with escalation | **Single attempt** | Loop exists but only wraps Vertex path |
| Best-so-far tracking | Tracks across attempts | **No tracking** | Vertex-only feature |

### Opportunities to Extend DRY to ComfyUI

1. **Wire face verification after ComfyUI generation** — `_verify_keyframe_faces()` is model-agnostic; it only needs image bytes + reference embeddings
2. **Wrap ComfyUI Flux generation in the same retry loop** — escalation could re-generate with increased `reference_strengths` (currently fixed at 0.65)
3. **Pass prequalified ref embeddings for verification** — the prequalification service already works for any image source
4. **Inject `_char_text_description` into ComfyUI prompts** — even without a formal identity instruction, appending facial feature text to the Flux prompt helps

---

## ComfyUI Identity Methods: Comparison for Future Implementation

### Available Approaches

| Method | Type | ArcFace Score | VRAM | Zero-shot? | Custom Nodes |
|--------|------|--------------|------|-----------|--------------|
| **PuLID Flux** | Embedding injection | 0.73–0.77 | High | Yes | `ComfyUI-PuLID-Flux` |
| **InstantID** | Adapter + ControlNet | 0.73–0.76 | High | Yes | `ComfyUI_InstantID` (SDXL only) |
| **IP-Adapter FaceID** | ArcFace embedding injection | 0.60–0.62 | Medium | Yes | `ComfyUI_IPAdapter_plus` |
| **Face-trained LoRA** | Weight fine-tuning | 0.85+ | Low | No (requires training) | None (built-in) |
| **ReActor** | Post-processing face swap | ~0.68 | Low | Yes | `ComfyUI-ReActor` |
| **Flux Redux** | CLIP Vision conditioning | Not face-specific | Medium | Yes | None (built-in) |
| **Nano Banana (current)** | Multimodal refs + text | 0.37–0.47 | N/A (API) | Yes | N/A |

### Recommended Integration Path

#### 1. PuLID Flux (Best zero-shot approach for Flux)

Highest identity scores among zero-shot methods. Natural fit for existing Flux pipeline.

**Integration:**
- New template: `flux-txt2img-with-pulid.json`
- New builder: `build_flux_pulid_workflow()` in `comfyui_client.py`
- Follows exact pattern of existing 4 Flux templates
- Nodes: `PulidFluxModelLoader` → `PulidFluxInsightFaceLoader` → `ApplyPulidFlux`
- Models: `pulid_flux_v0.9.1.safetensors` (~1.5 GB), EVA-CLIP (~400 MB), InsightFace antelopev2 (~300 MB)

**Template pattern:**
```json
{"class_type": "PulidFluxModelLoader", "inputs": {"pulid_file": "pulid_flux_v0.9.1.safetensors"}},
{"class_type": "PulidFluxInsightFaceLoader", "inputs": {"provider": "CPU"}},
{"class_type": "PulidFluxEvaClipLoader"},
{"class_type": "ApplyPulidFlux", "inputs": {
    "weight": 0.9,
    "start_at": 0.0,
    "end_at": 1.0,
    "model": ["flux_model", 0],
    "pulid_flux": ["pulid_loader", 0],
    "eva_clip": ["eva_clip_loader", 0],
    "face_analysis": ["insightface_loader", 0],
    "image": ["ref_image", 0]
}}
```

#### 2. Face-Trained LoRA (Already implemented — highest fidelity for known characters)

Already supported via `flux-dev-lora` and `flux-dev-full` model selections. Training happens via `lora_trainer.py` → Replicate. Best used for recurring Production Bible characters.

**Enhancement opportunity:** Wire face verification into the LoRA generation path so we can measure and compare LoRA identity scores against Vertex AI scores.

#### 3. ReActor Post-Processing (Quick win — works with any generation method)

Append ReActor face swap nodes to any existing workflow template. Low effort, moderate quality improvement.

**Integration:**
- Append 3-5 nodes after `VAEDecode` in any template
- Upload reference face via `comfy_client.upload_image()`
- Nodes: `LoadImage` (reference face) → `ReActorFaceSwap` → `SaveImage`
- Models: `inswapper_128.onnx` (~529 MB), optional face restoration

**Caution:** Post-processing face swap can look unnatural with large pose differences. Best used as a refinement pass, not a primary identity method.

### Combined Approach (Maximum Identity Preservation)

For critical shots (hero close-ups, emotional beats), combine methods:

```
1. Generate with PuLID Flux or LoRA (strong identity conditioning)
2. Verify with ArcFace (same _verify_keyframe_faces pipeline)
3. If score < threshold: ReActor face swap refinement pass
4. Re-verify → accept or retry
```

This reuses the existing verification infrastructure across all generation backends.

---

## Key Files

| File | Role |
|------|------|
| `vidpipe/services/ref_prequalification.py` | Screen refs for face detectability, cache embeddings |
| `vidpipe/pipeline/keyframes.py` | Identity instruction builder, retry loop, face verification |
| `vidpipe/services/face_matching.py` | ArcFace embedding generation + cosine similarity |
| `vidpipe/services/cv_detection.py` | YOLO face detection (upper-40% person bbox) |
| `vidpipe/services/comfyui_client.py` | Flux workflow builders, template selection |
| `vidpipe/services/comfyui_templates/` | JSON workflow templates for each Flux variant |
| `vidpipe/services/tag_resolver.py` | Resolves @tags to ActorRef images + LoRA URLs |
| `vidpipe/services/lora_trainer.py` | Per-character LoRA training via Replicate |
| `vidpipe/config.py` | `cv_analysis.keyframe_face_match_threshold` (0.45) |
| `vidpipe/db/models.py` | `ActorRef.face_embedding` (cached ArcFace vectors) |
