# Asset Library → Production Bible → Scene/Shot Integration Plan

**Author:** Manus AI
**Repository:** `hoyack/video-pipeline`
**Date:** March 14, 2026
**Status:** Proposed Architecture — Ready for Implementation

---

## 1. Executive Summary

This document defines the complete architectural plan for wiring the `video-pipeline` Asset Library into Production Bibles and Scenes, enabling tag-based asset referencing in shot descriptions and high-fidelity asset-consistent image generation via **Flux.1** and **ComfyUI**.

The system already has significant infrastructure in place from Phase 22 (Asset Library & Actor-Character Model). What remains is to close the loop between the **binding layer** and the **image generation layer**, specifically: ensuring that when a user writes a shot description like `@brandon steps out of his @2100_cybertruck in front of the @cyberpunk_hotel`, the generated keyframes faithfully reproduce Brandon's face, the Cybertruck's exact design, and the Cyberpunk Hotel's visual identity.

This plan is organized into six major areas: tag syntax standardization, binding-to-pipeline wiring, ComfyUI workflow design, LoRA training infrastructure, frontend enhancements, and a phased implementation roadmap.

---

## 2. Current State Assessment

Before defining what needs to be built, it is important to understand what already exists and what gaps remain.

### 2.1 What Is Already Built

The following table summarizes the current state of each system layer as of Phase 22.

| Layer | Component | Status |
|---|---|---|
| **Data Model** | `Actor`, `ActorRef`, `ActorVoiceProfile`, `ActorWardrobePreset` | Complete |
| **Data Model** | `LibrarySet`, `LibrarySetRef`, `LibrarySonicIdentity` | Complete |
| **Data Model** | `LibraryProp`, `LibraryPropRef` | Complete |
| **Data Model** | `SoundAsset` | Complete |
| **Data Model** | `CastBinding`, `SetBinding`, `PropBinding`, `SoundBinding` | Complete |
| **Backend API** | CRUD for all Asset Library entities | Complete |
| **Backend API** | Binding creation/update/delete endpoints | Complete |
| **Tag Resolver** | `[CHAR:TAG]`, `[SET:TAG]`, `[PROP:TAG]` resolution from bindings | Complete |
| **Storyboard Pipeline** | Tag resolution before LLM prompt injection | Complete |
| **Keyframe Pipeline** | Reference image loading for `CHARACTER` assets | Partial |
| **ComfyUI** | Wan 2.2 I2V video workflow | Complete |
| **ComfyUI** | Qwen image-edit workflow | Complete |
| **ComfyUI** | Flux.1 text-to-image workflow | Not started |
| **Frontend** | Asset Library UI (Actors, Sets, Props, Sound Assets) | Complete |
| **Frontend** | Production Bible binding sections (Casting, Art Dept, Sound) | Complete |
| **Frontend** | Tag autocomplete in scene editor | Not started |
| **LoRA Training** | Per-actor LoRA training pipeline | Not started |
| **Tag Syntax** | `[CHAR:TAG]` vs `@tag` unification | Needs decision |

### 2.2 The Core Gap

The existing `tag_resolver.py` resolves tags to **text descriptions** and collects **reference image URLs**. However, the keyframe pipeline currently only uses reference images for the existing `Asset` model (the old video-extraction-based system), not for the new `Actor`/`LibrarySet`/`LibraryProp` entities from the Asset Library. The bridge between the new binding system and the image generation adapter is the primary gap this plan addresses.

---

## 3. Tag Syntax Decision

The existing system uses `[CHAR:TAG]`, `[SET:TAG]`, and `[PROP:TAG]` syntax. The user's request uses `@tag` syntax. This plan recommends supporting **both** syntaxes, with `@tag` as the primary user-facing syntax and `[TYPE:TAG]` retained for backward compatibility and explicit type-scoped resolution.

### 3.1 Recommended Syntax

The `@tag` syntax is more natural for users writing shot descriptions and storyboards. The tag resolver should be extended to support it:

```
@brandon steps out of his @2100_cybertruck in front of the @cyberpunk_hotel
```

At resolution time, the system looks up `brandon`, `2100_cybertruck`, and `cyberpunk_hotel` across all binding types (Cast, Set, Prop) within the attached Production Bible. The binding's `tag` field (which is already stored as a plain string like `BRANDON` or `CYBERPUNK_HOTEL`) is matched case-insensitively.

The `[CHAR:TAG]` syntax remains supported for explicit type-scoped references, which is useful when the same tag name exists in multiple binding types. The updated tag resolver should handle both patterns:

```python
# Existing pattern (backward compatible)
TAG_PATTERN = re.compile(r"\[(CHAR|SET|PROP):([A-Z0-9_]+)\]")

# New @-mention pattern
AT_TAG_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")
```

When an `@tag` is encountered, the resolver performs a cross-type lookup in order: CastBinding → PropBinding → SetBinding. The first match wins. If no match is found, the tag is left as-is and logged as unresolved.

---

## 4. Binding-to-Pipeline Wiring

This section describes the changes needed to make the Asset Library bindings flow correctly through the storyboard and keyframe pipelines.

### 4.1 Extended Tag Resolution Result

The current `ResolvedPrompt` dataclass in `tag_resolver.py` returns text substitutions and collects `character_refs` (actor ref URLs) and `set_context` (lighting/style). This needs to be extended to carry structured asset metadata for the image generation adapter:

```python
@dataclass
class ResolvedAssetRef:
    """A resolved asset reference for image generation."""
    tag: str                      # e.g., "BRANDON"
    asset_type: str               # "CHARACTER", "PROP", "SET"
    display_name: str             # e.g., "Brandon"
    text_description: str         # base_appearance_prompt / appearance_prompt / reverse_prompt
    reference_image_urls: list[str]   # all uploaded reference images for this asset
    lora_url: str | None          # path to trained LoRA file (if available)
    wardrobe_override: dict | None    # for CastBindings with wardrobe overrides
    lighting_notes: str | None    # for SetBindings

@dataclass
class ResolvedPrompt:
    text: str
    asset_refs: list[ResolvedAssetRef]   # NEW: structured asset references
    character_refs: list = field(default_factory=list)   # backward compat
    set_context: list = field(default_factory=list)      # backward compat
    unresolved_tags: list = field(default_factory=list)
```

### 4.2 Keyframe Pipeline Updates

The `generate_keyframes()` function in `keyframes.py` needs to be updated to use the new `ResolvedAssetRef` list when building the image generation payload. The current code path that loads reference images from the old `Asset` model (via `manifest_service.load_manifest_assets`) should be supplemented with a parallel path that loads from the new binding system.

The decision tree for which path to use is:

1.  If the scene has a `production_bible_id` AND the scene prompt contains `@tags` or `[TYPE:TAG]` references, use the **binding-based path** (new Asset Library).
2.  If the scene has a `production_bible_id` and uses the old `ShotManifest` system with `CHAR_01` style tags, use the **manifest-based path** (existing system).
3.  Both paths can coexist within the same scene if needed.

The updated keyframe generation logic for the new path:

```python
# New path: resolve @tags and [TYPE:TAG] from bindings
if scene.production_bible_id and has_any_tags(shot.start_frame_prompt):
    from vidpipe.services.tag_resolver import resolve_tags_with_assets
    resolved = await resolve_tags_with_assets(
        shot.start_frame_prompt, scene.production_bible_id, session
    )
    asset_refs = resolved.asset_refs
    # Categorize by type for the image adapter
    char_refs = [r for r in asset_refs if r.asset_type == "CHARACTER"]
    prop_refs = [r for r in asset_refs if r.asset_type == "PROP"]
    set_refs  = [r for r in asset_refs if r.asset_type == "SET"]
```

### 4.3 Storyboard Pipeline Updates

The storyboard pipeline already calls `resolve_tags()` to enrich the LLM prompt. The update here is to also pass the `asset_refs` into the system prompt so the LLM understands exactly what assets are available and what their visual descriptions are. The existing `format_asset_registry()` function in `manifest_service.py` serves this purpose but currently only reads from the old `Asset` model. A new `format_binding_registry()` function should be added that reads from the binding tables:

```python
async def format_binding_registry(
    session: AsyncSession,
    bible_id: uuid.UUID,
) -> str:
    """Format all bound assets from a Production Bible for LLM context injection."""
    cast = await session.execute(
        select(CastBinding).where(CastBinding.production_bible_id == bible_id)
    )
    # ... load sets, props, format into structured text
```

This ensures the LLM generating shot descriptions knows to use `@brandon` and `@cyberpunk_hotel` as valid references.

---

## 5. ComfyUI Workflow Design for Flux.1

This is the most technically complex part of the plan. We need to design ComfyUI workflows that can accept multiple reference images (one per asset) and generate a coherent image that faithfully represents all of them simultaneously.

### 5.1 Technology Selection

Three approaches exist for multi-asset reference injection in Flux.1, each with different tradeoffs:

| Approach | Technology | Consistency | Training Required | Speed |
|---|---|---|---|---|
| **Reference Injection** | Flux.1 Redux + UNO | Good | No | Fast |
| **IP-Adapter** | InstantX FLUX.1-dev-IP-Adapter | Good (faces) | No | Fast |
| **Custom LoRA** | Kohya_ss / SimpleTuner | Excellent | Yes (1-3 hrs) | Fast (inference) |
| **Hybrid** | LoRA (character) + Redux (prop/set) | Excellent | Yes (character only) | Fast |

The **recommended approach** is a hybrid model:

- **Characters (Actors):** Custom LoRA training per actor, triggered manually by the user when they have sufficient reference images. The LoRA captures the actor's exact face and body proportions.
- **Props and Sets:** Flux.1 Redux or UNO reference injection at generation time. These assets tend to be more about visual style than exact identity, making reference injection sufficient.

### 5.2 New ComfyUI Workflow Templates

Three new workflow templates need to be created and stored in the `docs/` or `backend/vidpipe/templates/comfyui/` directory:

**Template 1: `flux_txt2img_base.json`**
A standard Flux.1 Dev text-to-image workflow with no reference images. Used as the fallback when no assets have reference images.

**Template 2: `flux_txt2img_with_lora.json`**
Flux.1 Dev with a dynamic LoRA loader node. The `comfyui_client.py` builder function will inject the LoRA filename and strength at runtime.

**Template 3: `flux_txt2img_with_references.json`**
Flux.1 Dev with UNO or Redux conditioning nodes accepting up to 3 reference images. The builder function will inject the uploaded reference image filenames at runtime.

**Template 4: `flux_txt2img_full.json`**
The full hybrid template: Flux.1 Dev + LoRA loader + UNO reference conditioning. This is the production-quality template used when a character has a trained LoRA and the scene also contains props/sets with reference images.

### 5.3 Workflow Builder Functions

New functions to be added to `comfyui_client.py`:

```python
def build_flux_txt2img_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    lora_filename: str | None = None,
    lora_strength: float = 0.8,
    reference_image_filenames: list[str] | None = None,
    reference_strengths: list[float] | None = None,
) -> dict:
    """Build a Flux.1 text-to-image workflow with optional LoRA and reference images.

    Dynamically selects the appropriate template based on what's provided:
    - No lora, no refs: flux_txt2img_base.json
    - lora only: flux_txt2img_with_lora.json
    - refs only: flux_txt2img_with_references.json
    - both: flux_txt2img_full.json
    """
```

### 5.4 Model Registry Update

The `COMFYUI_IMAGE_MODELS` set in `keyframes.py` needs to be updated to include Flux model identifiers:

```python
COMFYUI_IMAGE_MODELS = {
    "qwen-fast",
    "qwen-image-edit",
    "flux-dev",           # NEW: Flux.1 Dev base
    "flux-dev-lora",      # NEW: Flux.1 Dev with LoRA support
    "flux-dev-redux",     # NEW: Flux.1 Dev with Redux reference injection
    "flux-dev-full",      # NEW: Flux.1 Dev with LoRA + Redux
}
```

These model IDs can be selected in the Scene creation form, allowing users to choose their preferred generation approach per scene.

---

## 6. LoRA Training Infrastructure

This section describes the infrastructure needed to train per-actor LoRA models from the Actor's reference images.

### 6.1 Training Pipeline Overview

The LoRA training pipeline is an asynchronous background process triggered by the user from the Actor detail view. The high-level flow is:

```
User clicks "Train Identity Model"
    → Backend validates: actor has >= 10 reference images
    → Backend creates a TrainingJob record (status: QUEUED)
    → Backend dispatches job to GPU worker (via queue or direct API)
    → GPU worker: downloads images → captions → trains LoRA → uploads .safetensors
    → Worker updates TrainingJob status to COMPLETED
    → Backend updates Actor.lora_url with the S3 path
    → Frontend shows "Identity Model: Ready" on actor card
```

### 6.2 Data Model Additions

Two new fields are needed on the `Actor` model:

```python
class Actor(Base):
    # ... existing fields ...
    lora_url: Mapped[Optional[str]]      # S3 path to trained .safetensors file
    lora_trained_at: Mapped[Optional[datetime]]
    lora_training_status: Mapped[Optional[str]]  # QUEUED | TRAINING | COMPLETED | FAILED
```

Similar fields should be added to `LibraryProp` for props that warrant a dedicated LoRA (e.g., a hero vehicle that appears in many shots).

### 6.3 Training Job Service

A new `lora_trainer.py` service handles the training lifecycle:

```python
async def dispatch_actor_lora_training(
    session: AsyncSession,
    actor_id: uuid.UUID,
) -> TrainingJob:
    """Dispatch a LoRA training job for an actor.

    Downloads all ActorRef images, generates captions via VLM,
    packages the dataset, and submits to the training worker.
    """
```

### 6.4 Training Worker Options

The training worker can be implemented in one of three ways, depending on infrastructure:

| Option | Description | Pros | Cons |
|---|---|---|---|
| **Local GPU Worker** | A separate Python process running Kohya_ss on a local GPU machine | Full control, no per-job cost | Requires dedicated GPU hardware |
| **Replicate API** | Use `lucataco/simpletuner-flux` on Replicate | No infrastructure management | Per-job cost, external dependency |
| **RunPod / Lambda Labs** | Spin up a GPU pod on demand for each training job | Cost-effective for batch training | Latency to provision pod |
| **ComfyUI Flux Trainer** | Use the ComfyUI Flux Trainer custom node | Unified with existing ComfyUI setup | Requires ComfyUI to be running |

The recommended approach for initial implementation is the **Replicate API** path, as it requires no additional infrastructure. The `lora_trainer.py` service can be designed with a pluggable backend interface so that the training provider can be swapped later.

### 6.5 Dataset Preparation

Quality LoRA training requires properly captioned images. The dataset preparation step should:

1.  Download all `ActorRef` images for the actor.
2.  Resize images to 512×512 or 768×768 (maintaining aspect ratio with padding).
3.  Generate a caption for each image using a VLM (e.g., Qwen-VL or BLIP-2). The caption should describe the person's appearance without using their name, to avoid overfitting.
4.  Add a trigger word caption (e.g., `"a photo of ACTOR_BRANDON"`) to a subset of images.
5.  Package as a zip file and upload to S3 for the training worker.

Minimum recommended dataset sizes:

| Use Case | Min Images | Recommended Images | Training Steps |
|---|---|---|---|
| Face identity (close-ups) | 5 | 15–20 | 1000–1500 |
| Full body consistency | 10 | 20–30 | 1500–2000 |
| Prop/vehicle | 8 | 15–20 | 1000–1500 |
| Set/environment | 5 | 10–15 | 800–1200 |

---

## 7. Frontend Enhancements

### 7.1 Tag Autocomplete in Scene Editor

The scene prompt editor (currently a plain textarea or CodeMirror instance) needs tag autocomplete support. When the user types `@`, a dropdown should appear showing all assets bound to the attached Production Bible, filterable by name.

The implementation uses the existing `editorExtensions.ts` CodeMirror setup. A new extension should be added:

```typescript
// New file: frontend/src/components/codemirror/assetTagCompletion.ts
import { autocompletion, CompletionContext } from "@codemirror/autocomplete";

export function createAssetTagCompletion(
  bibleId: string,
  boundAssets: BoundAssetSummary[]
): Extension {
  return autocompletion({
    override: [
      async (context: CompletionContext) => {
        const before = context.matchBefore(/@\w*/);
        if (!before) return null;
        return {
          from: before.from,
          options: boundAssets.map((asset) => ({
            label: `@${asset.tag.toLowerCase()}`,
            detail: `${asset.type}: ${asset.name}`,
            info: asset.description,
          })),
        };
      },
    ],
  });
}
```

The `BoundAssetSummary` list is fetched from a new API endpoint: `GET /api/production-bibles/{id}/bound-assets/summary`, which returns all CastBindings, SetBindings, and PropBindings in a flat list with their tags, names, types, and primary reference image thumbnails.

### 7.2 Tag Preview Panel

In the Scene editor, a collapsible side panel should show a "Tag Preview" — when the user hovers over or clicks on an `@tag` in the editor, the panel shows the asset's primary reference image, name, and text description. This confirms to the user that the tag is correctly linked before generation.

### 7.3 Actor Detail View: Training Status

The Actor detail view should be updated to show:

- A "Train Identity Model" button (enabled when `refs.length >= 5`).
- A status badge: "No Model", "Training...", or "Model Ready" with the training date.
- A "Regenerate Model" button when the actor has been updated with new reference images since the last training.

### 7.4 Production Bible: Asset Tag Reference Sheet

The Production Bible detail view should include a "Tag Reference Sheet" tab that lists all bound assets with their `@tag` syntax, type, and thumbnail. This serves as a quick reference for writers composing shot descriptions.

---

## 8. End-to-End Example Walkthrough

This section traces the complete user journey for the example scenario: `@brandon steps out of his @2100_cybertruck in front of the @cyberpunk_hotel`.

### Step 1: Create Assets in the Asset Library

The user navigates to **Asset Library → Actors** and creates a new Actor named "Brandon":
- Uploads 15 reference photos (front, side, 3/4 views, various expressions).
- Sets `base_appearance_prompt`: `"tall athletic man in his 30s, short dark hair, strong jawline, brown eyes"`.
- Sets prompt tag: `BRANDON`.

The user then navigates to **Asset Library → Props** and creates "2100 Cybertruck":
- Uploads 8 reference images of the vehicle.
- Sets `appearance_prompt`: `"futuristic angular electric truck, matte black body, sharp geometric lines, glowing blue trim"`.
- Sets prompt tag: `2100_CYBERTRUCK`.

The user navigates to **Asset Library → Sets** and creates "Cyberpunk Hotel":
- Uploads 6 reference images of the hotel exterior.
- Sets `reverse_prompt`: `"neon-lit brutalist hotel facade, holographic signage, rain-slicked streets, cyberpunk aesthetic"`.
- Sets prompt tag: `CYBERPUNK_HOTEL`.

### Step 2: Create a Production Bible and Bind Assets

The user creates a new Production Bible named "Cyberpunk One":
- In the **Casting** tab, clicks "+ Add" → selects Brandon → fills in character name "Brandon Mercer", role "LEAD", tag `BRANDON`.
- In the **Art Department → Props** tab, clicks "+ Add" → selects 2100 Cybertruck → tag `2100_CYBERTRUCK`.
- In the **Art Department → Sets** tab, clicks "+ Add" → selects Cyberpunk Hotel → tag `CYBERPUNK_HOTEL`.

### Step 3: Create a Scene with the Production Bible

The user creates a new Scene, attaches the "Cyberpunk One" Production Bible, and writes the following prompt:

```
@brandon steps out of his @2100_cybertruck in front of the @cyberpunk_hotel.
He looks up at the neon signs, rain falling on his jacket.
```

The editor highlights `@brandon`, `@2100_cybertruck`, and `@cyberpunk_hotel` in blue, confirming they are resolved.

### Step 4: Storyboard Generation

The storyboard pipeline:
1.  Calls `resolve_tags_with_assets()` on the scene prompt.
2.  Resolves `@brandon` → `"Brandon Mercer (tall athletic man in his 30s, short dark hair, strong jawline, brown eyes)"` + ref images.
3.  Resolves `@2100_cybertruck` → `"2100 Cybertruck (futuristic angular electric truck, matte black body, sharp geometric lines, glowing blue trim)"` + ref images.
4.  Resolves `@cyberpunk_hotel` → `"Cyberpunk Hotel (neon-lit brutalist hotel facade, holographic signage, rain-slicked streets, cyberpunk aesthetic)"` + ref images.
5.  Injects the enriched prompt and asset registry into the LLM system prompt.
6.  The LLM generates shot descriptions that correctly reference `@brandon`, `@2100_cybertruck`, and `@cyberpunk_hotel` in each shot.

### Step 5: Keyframe Generation

For each shot, the keyframe pipeline:
1.  Checks if Brandon has a trained LoRA → yes, `brandon_v1.safetensors`.
2.  Downloads the Cybertruck reference image and the Hotel reference image.
3.  Builds a `flux_txt2img_full.json` workflow:
    - LoRA: `brandon_v1.safetensors` (strength: 0.85)
    - UNO Reference 1: `cybertruck_side.png` (strength: 0.65)
    - UNO Reference 2: `cyberpunk_hotel_exterior.png` (strength: 0.60)
    - Prompt: `"Brandon Mercer (tall athletic man, short dark hair) steps out of a futuristic angular electric truck in front of a neon-lit brutalist hotel facade, rain-slicked streets, cyberpunk aesthetic"`
4.  Queues the workflow to ComfyUI.
5.  The generated keyframe shows Brandon's exact face (via LoRA), the Cybertruck's angular design (via UNO), and the Cyberpunk Hotel's neon aesthetic (via UNO).

---

## 9. Implementation Roadmap

The work is organized into four implementation phases, each building on the previous.

### Phase A: Tag Syntax & Binding Registry (Estimated: 1–2 weeks)

This phase focuses on extending the tag resolver and pipeline wiring without touching the image generation layer.

| Task | File(s) | Description |
|---|---|---|
| A-01 | `tag_resolver.py` | Add `@tag` pattern support alongside `[TYPE:TAG]` |
| A-02 | `tag_resolver.py` | Extend `ResolvedPrompt` with `ResolvedAssetRef` list |
| A-03 | `tag_resolver.py` | Implement `resolve_tags_with_assets()` that loads from bindings |
| A-04 | `manifest_service.py` | Add `format_binding_registry()` for LLM context injection |
| A-05 | `storyboard.py` | Use `format_binding_registry()` when bible has bindings |
| A-06 | `api/routes.py` | Add `GET /api/production-bibles/{id}/bound-assets/summary` |
| A-07 | `frontend/src/api/client.ts` | Add `getBoundAssetsSummary()` API client function |
| A-08 | `frontend/src/api/types.ts` | Add `BoundAssetSummary` type |

### Phase B: ComfyUI Flux Workflows (Estimated: 2–3 weeks)

This phase introduces Flux.1 as an image generation backend in ComfyUI.

| Task | File(s) | Description |
|---|---|---|
| B-01 | `docs/comfyui/flux_txt2img_base.json` | Design and export Flux.1 Dev base workflow |
| B-02 | `docs/comfyui/flux_txt2img_with_lora.json` | Design Flux.1 Dev + LoRA loader workflow |
| B-03 | `docs/comfyui/flux_txt2img_with_references.json` | Design Flux.1 Dev + UNO/Redux reference workflow |
| B-04 | `docs/comfyui/flux_txt2img_full.json` | Design full hybrid workflow |
| B-05 | `comfyui_client.py` | Add `build_flux_txt2img_workflow()` builder function |
| B-06 | `keyframes.py` | Add Flux model IDs to `COMFYUI_IMAGE_MODELS` |
| B-07 | `keyframes.py` | Add binding-based reference resolution path |
| B-08 | `keyframes.py` | Route CHARACTER LoRA + PROP/SET refs to Flux workflow builder |
| B-09 | `frontend/src/lib/constants.ts` | Add Flux model options to `IMAGE_MODELS` catalog |

### Phase C: LoRA Training Infrastructure (Estimated: 3–4 weeks)

This phase adds the per-actor LoRA training capability.

| Task | File(s) | Description |
|---|---|---|
| C-01 | `db/models.py` | Add `lora_url`, `lora_trained_at`, `lora_training_status` to `Actor` |
| C-02 | `db/models.py` | Add `lora_url`, `lora_training_status` to `LibraryProp` |
| C-03 | `db/migrations/` | Alembic migration for new fields |
| C-04 | `services/lora_trainer.py` | New service: dataset prep, job dispatch, status polling |
| C-05 | `api/asset_library.py` | Add `POST /api/asset-library/actors/{id}/train-lora` endpoint |
| C-06 | `api/asset_library.py` | Add `GET /api/asset-library/actors/{id}/lora-status` endpoint |
| C-07 | `frontend/src/components/ActorLibraryDetail.tsx` | Add "Train Identity Model" button and status badge |
| C-08 | `frontend/src/api/client.ts` | Add `trainActorLora()` and `getActorLoraStatus()` functions |

### Phase D: Frontend Enhancements (Estimated: 1–2 weeks)

This phase delivers the user-facing improvements to the scene editor and Production Bible views.

| Task | File(s) | Description |
|---|---|---|
| D-01 | `codemirror/assetTagCompletion.ts` | New CodeMirror extension for `@tag` autocomplete |
| D-02 | `codemirror/editorExtensions.ts` | Wire autocomplete extension into editor setup |
| D-03 | `components/SceneEditor.tsx` | Pass `bibleId` and `boundAssets` to editor for autocomplete |
| D-04 | `components/TagPreviewPanel.tsx` | New component: hover/click preview of resolved tag |
| D-05 | `components/ProductionBibleDetail.tsx` | Add "Tag Reference Sheet" tab |
| D-06 | `components/ActorLibraryDetail.tsx` | Add LoRA training status UI |

---

## 10. Open Questions and Decisions Required

The following questions require a decision before or during implementation:

1.  **Tag Syntax:** Should `@tag` be the primary syntax going forward, with `[TYPE:TAG]` deprecated? Or should both be fully supported indefinitely? The `@tag` syntax is more natural for writers but loses the explicit type scoping.

2.  **LoRA Training Provider:** Which training backend should be used first — Replicate API (easy, per-job cost), local GPU worker (full control, hardware required), or ComfyUI Flux Trainer (unified stack)? This decision affects Phase C's implementation significantly.

3.  **Reference Image Strength Tuning:** The UNO/Redux reference image strengths (e.g., 0.65 for props, 0.60 for sets) need empirical tuning. Should these be hardcoded defaults, per-binding overrides, or scene-level settings?

4.  **Asset Versioning:** If an actor's reference images are updated after a LoRA has been trained, should the old LoRA be invalidated automatically? Should Production Bibles that reference this actor be notified?

5.  **Multiple Characters Per Shot:** When a shot contains two characters (e.g., `@brandon` and `@sarah`), the system can only load one LoRA at a time in standard ComfyUI. Options include: (a) merge LoRAs, (b) use one LoRA + one UNO reference, or (c) use UNO references for both. This needs a defined policy.

6.  **Minimum Reference Image Count:** Should the "Train Identity Model" button be disabled until a minimum number of reference images are uploaded? The recommended minimum is 10 images for good results, but 5 may be sufficient for face-only LoRAs.

---

## 11. Architecture Diagram

The following diagram illustrates the complete data flow from asset creation to keyframe generation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ASSET LIBRARY                               │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│  │    Actor    │   │ LibrarySet  │   │ LibraryProp │              │
│  │  @brandon   │   │@cyberpunk_  │   │@2100_cyber_ │              │
│  │             │   │   hotel     │   │   truck     │              │
│  │ ActorRefs[] │   │ SetRefs[]   │   │ PropRefs[]  │              │
│  │ lora_url    │   │             │   │             │              │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘              │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION BIBLE                               │
│                                                                     │
│  CastBinding(actor=@brandon, tag="BRANDON", char_name="Brandon")   │
│  SetBinding(set=@cyberpunk_hotel, tag="CYBERPUNK_HOTEL")           │
│  PropBinding(prop=@2100_cybertruck, tag="2100_CYBERTRUCK")         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ scene.production_bible_id
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           SCENE                                     │
│                                                                     │
│  prompt: "@brandon steps out of his @2100_cybertruck in front of   │
│           the @cyberpunk_hotel"                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  tag_resolver.py   │
                    │  resolve_tags_     │
                    │  with_assets()     │
                    └─────────┬──────────┘
                              │ ResolvedAssetRef[]
                    ┌─────────▼──────────┐
                    │  storyboard.py     │
                    │  LLM generates     │
                    │  shot descriptions │
                    └─────────┬──────────┘
                              │ Shot.start_frame_prompt
                    ┌─────────▼──────────┐
                    │  keyframes.py      │
                    │  build_flux_       │
                    │  workflow()        │
                    └─────────┬──────────┘
                              │ ComfyUI JSON workflow
                    ┌─────────▼──────────┐
                    │   ComfyUI          │
                    │   Flux.1 Dev       │
                    │   + brandon LoRA   │
                    │   + truck ref img  │
                    │   + hotel ref img  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Keyframe PNG     │
                    │   (brandon's face, │
                    │    cybertruck,     │
                    │    hotel visible)  │
                    └────────────────────┘
```

---

## References

[1] [How to use image prompts with Flux model (Redux) — Stable Diffusion Art](https://stable-diffusion-art.com/flux-redux/)
[2] [GitHub — bytedance/UNO: A Universal Customization Method for Both Single and Multi-Subject Conditioning](https://github.com/bytedance/UNO)
[3] [GitHub — bmaltais/kohya_ss: Kohya_ss GUI for LoRA training](https://github.com/bmaltais/kohya_ss)
[4] [GitHub — bghira/SimpleTuner: SimpleTuner for Flux LoRA training](https://github.com/bghira/SimpleTuner)
[5] [InstantX/FLUX.1-dev-IP-Adapter on Hugging Face](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter)
[6] [FLUX.1 Kontext: Character Consistency and In-Context Image Generation — Together AI](https://www.together.ai/blog/flux-1-kontext)
[7] [ComfyUI Flux.1 Text-to-Image Workflow Example — ComfyUI Docs](https://docs.comfy.org/tutorials/flux/flux-1-text-to-image)
[8] [FLUXGYM to ComfyUI: Building and Using a Custom LoRA — Interactive Immersive](https://interactiveimmersive.io/blog/artificial-intelligence/fluxgym-to-comfyui-building-and-using-custom-loras/)
