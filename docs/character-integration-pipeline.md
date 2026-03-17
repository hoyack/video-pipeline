# Character Integration Pipeline

Technical reference for how an Asset Library Actor flows through the system — from creation, to Production Bible binding, to scene generation with identity-conditioned keyframes.

## Pipeline Overview

```
Asset Library          Production Bible           Scene Editor              Pipeline
============          ================           ============              ========

1. Create Actor    2. Cast Actor as         3. Write prompt with     4. Storyboard LLM
   + upload refs      CastBinding              @TAG references          receives binding
                      (tag, role, name)                                  registry text

                                              5. User generates      6. Keyframe gen
                                                 scene                   resolves @tags
                                                                         to refs + LoRA

                                                                      7. Image adapter
                                                                         receives ref
                                                                         images as
                                                                         identity ground
```

---

## Stage 1: Actor Creation in Asset Library

**Models:** `Actor` + `ActorRef` (1:N)

An Actor is created via `POST /api/asset-library/actors` with:
- `name` (required)
- `base_appearance_prompt` (optional) — detailed visual description used for prompt injection
- `description` (optional) — general notes

Reference images are uploaded separately via `POST /api/asset-library/actors/{id}/refs`, creating `ActorRef` rows:
- `image_url` — S3 key or local path to the uploaded image
- `label` — optional tag like "front", "profile", "3-4"
- `is_primary` — determines thumbnail display

**Key files:**
- `backend/vidpipe/db/models.py:432-475` — Actor + ActorRef models
- `backend/vidpipe/api/asset_library.py:615-762` — CRUD + image upload endpoints

**Data at this stage:**
```
Actor(id=abc, name="Sarah", base_appearance_prompt="A woman in her 30s, auburn hair...")
  ActorRef(actor_id=abc, image_url="asset-library/actors/abc/refs/front.png", is_primary=True)
  ActorRef(actor_id=abc, image_url="asset-library/actors/abc/refs/profile.png")
```

---

## Stage 2: Casting — CastBinding in Production Bible

**Model:** `CastBinding`

A CastBinding connects an Actor to a Production Bible with production-specific metadata. Created via `POST /api/production-bibles/{bible_id}/cast`:

- `actor_id` — FK to Actor (validated exists)
- `tag` — unique within the bible, e.g. `SARAH_CONNOR` (used as `@SARAH_CONNOR` in prompts)
- `character_name` — display name for this production, e.g. "Sarah Connor"
- `role` — LEAD / SUPPORTING / EXTRA / NARRATOR
- `character_description`, `character_arc`, `behavioral_notes` — optional character context
- `wardrobe_override` — optional JSON for production-specific wardrobe

**CastBinding does NOT store reference images.** It always resolves through Actor -> ActorRef at generation time. This means Actor refs are shared across all productions the actor appears in.

**Unique constraints:** One actor per bible (no double-casting), one tag per bible (no conflicts).

**Key files:**
- `backend/vidpipe/db/models.py:635-670` — CastBinding model
- `backend/vidpipe/api/bindings.py:272-303` — create endpoint

---

## Stage 3: Production Bible Binding to Scene

The scene editor allows selecting a Production Bible via `ProductionBibleSelector`. When committed via `PATCH /api/scenes/{id}/edit`:

1. `scene.production_bible_id` is set
2. `scene.production_bible_version` records the bible version
3. `manifest_service.create_snapshot()` freezes the bible state at binding time
4. `manifest_service.increment_usage()` tracks usage count

The Production Bible ID on the scene is what triggers binding-aware mode in all downstream pipeline stages.

**Key files:**
- `backend/vidpipe/api/routes.py:793-819` — edit scene with bible binding
- `backend/vidpipe/services/manifest_service.py:682-758` — snapshot creation

---

## Stage 4: Storyboard Generation — Binding Registry Injection

When `scene.production_bible_id` is set, the storyboard stage injects a binding registry into the LLM system prompt.

### 4.1 Building the Registry Text

`format_binding_registry()` in `manifest_service.py:861-988`:

1. Queries all CastBindings, SetBindings, PropBindings for the bible
2. Batch-loads referenced Actor, LibrarySet, LibraryProp entities
3. Formats each binding as a text block:

```
AVAILABLE ASSETS FOR THIS PRODUCTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CHARACTER] @SARAH_CONNOR — "Sarah Connor"(LEAD)
  A woman in her 30s, auburn hair, athletic build, intense green eyes...

[SET] @BUNKER — "Underground Bunker"
  Concrete walls, dim fluorescent lighting, military equipment scattered...

[PROP] @PLASMA_RIFLE — "Plasma Rifle"
  Futuristic energy weapon with blue glow...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference assets by @TAG in your scene descriptions and shot prompts.
```

The `text_description` for characters comes from `Actor.base_appearance_prompt`.

### 4.2 Injection into Storyboard Prompt

In `storyboard.py:339-460`:

```python
binding_block = await format_binding_registry(session, scene.production_bible_id)
if binding_block:
    asset_registry_block = binding_block  # New path: @tag syntax
```

This block is injected into `ENHANCED_STORYBOARD_PROMPT` which instructs the LLM:
- Use registered `@TAG` references for all known characters/sets/props
- Do not invent new names for listed assets
- Generate `shot_manifest` entries referencing asset tags

### 4.3 Scene Prompt Tag Resolution

Before sending to the LLM, the user's scene prompt is resolved via `resolve_tags()`:
```
Input:  "Show @SARAH_CONNOR entering @BUNKER"
Output: "Show Sarah Connor (A woman in her 30s, auburn hair...) entering Underground Bunker (Concrete walls...)"
```

The LLM receives the expanded descriptions and composes detailed shot-level prompts informed by the character's appearance.

**Key files:**
- `backend/vidpipe/services/manifest_service.py:861-988` — format_binding_registry
- `backend/vidpipe/pipeline/storyboard.py:339-460` — injection logic
- `backend/vidpipe/services/tag_resolver.py:84-176` — resolve_tags (text substitution)

---

## Stage 5: Keyframe Generation — Tag Resolution to Reference Images

This is where text descriptions become visual identity. The keyframe generator resolves `@tags` to actual image bytes and passes them to the image adapter.

### 5.1 Two Resolution Paths

The keyframe stage has two paths depending on the image model:

**Path A: Vertex AI (Gemini/Imagen models)**
Uses the legacy manifest asset system. The prompt rewriter (`PromptRewriterService`) selects 3 reference tags, then `resolve_asset_image_bytes()` loads their images as raw bytes.

**Path B: ComfyUI Flux**
Uses `resolve_tags_with_assets()` which returns `ResolvedAssetRef` objects containing `reference_image_urls` and `lora_url`.

### 5.2 ResolvedAssetRef Structure

```python
@dataclass
class ResolvedAssetRef:
    tag: str                        # SARAH_CONNOR
    asset_type: str                 # CHARACTER
    display_name: str               # Sarah Connor
    text_description: str           # Actor.base_appearance_prompt
    reference_image_urls: list[str] # [ActorRef.image_url, ...]
    lora_url: str | None            # Trained identity model (Phase 25)
    wardrobe_override: dict | None  # CastBinding override
    lighting_notes: str | None      # SetBinding override
```

### 5.3 Tag Resolution Query Chain

```
resolve_tags_with_assets(prompt, production_bible_id, session)
  1. Batch load: select(CastBinding).where(bible_id=...)
  2. Batch load: select(Actor).where(id.in_(actor_ids))
  3. Batch load: select(ActorRef).where(actor_id.in_(actor_ids))
  4. For each @tag match in prompt:
     - Lookup CastBinding by tag (case-insensitive)
     - Get Actor via binding.actor_id
     - Get ActorRef[] via actor.id
     - Build ResolvedAssetRef with ref_urls = [r.image_url for r in refs]
```

### 5.4 Reference Image Delivery to Image Adapters

**Vertex AI path** (`_generate_image_from_text`, keyframes.py:89-144):
```python
contents = []
if reference_images:
    contents.append("The following reference photo(s) show the EXACT person(s)...")
    for ref_bytes in reference_images:
        contents.append(types.Part.from_bytes(data=ref_bytes, mime_type=...))
contents.append(prompt)
response = await client.aio.models.generate_content(model=image_model, contents=contents)
```
Reference images are prepended as multimodal parts before the text prompt. Gemini uses them for identity grounding.

**Vertex AI conditioned path** (`_generate_image_conditioned`, keyframes.py:153-210):
For end frames, the conditioning (previous) frame comes first (strongest weight), then asset references, then prompt.

**ComfyUI Flux path** (`_generate_image_comfyui_flux`, keyframes.py:444-522):
```python
for aref in resolved.asset_refs:
    if aref.asset_type == "CHARACTER" and aref.lora_url:
        flux_lora = aref.lora_url       # Use LoRA for trained identity
    elif aref.reference_image_urls:
        ref_url = aref.reference_image_urls[0]  # First ref only
        ref_data = await file_mgr.read_bytes(ref_url)
        uploaded_name = await comfy_client.upload_image(ref_data, f"flux_ref_{aref.tag}.png")
        flux_ref_filenames.append(uploaded_name)
        flux_ref_strengths.append(0.65)
```
Reference images are uploaded to ComfyUI cloud, then injected into the Flux workflow as `LoadImage` nodes with `unCLIPConditioning`.

### 5.5 Face Verification Loop

After keyframe generation, if placed CHARACTER assets exist, a face verification step runs:
1. YOLO face detection on the generated keyframe
2. ArcFace embedding comparison against Actor reference embeddings
3. If verification fails, retry with stronger identity emphasis prefix (up to 2 retries)
4. Soft degradation: if CV tools are unavailable or no faces detected, verification passes

**Key files:**
- `backend/vidpipe/pipeline/keyframes.py:525-1071` — main orchestrator
- `backend/vidpipe/services/tag_resolver.py:258-478` — resolve_tags_with_assets
- `backend/vidpipe/services/reference_selection.py:280-360` — resolve_asset_image_bytes
- `backend/vidpipe/services/prompt_rewriter.py:104-134` — rewrite_keyframe_prompt

---

## Image Adapter Reference Image Support

| Adapter | Refs Supported | How Refs Are Used | Max Refs |
|---------|---------------|-------------------|----------|
| Gemini (nano-banana, nano-banana-pro) | Yes | Multimodal parts prepended to prompt | Unlimited |
| ComfyUI Qwen (qwen-image-edit) | No | Text-only generation, no ref inputs | N/A |
| ComfyUI Flux (flux-dev, flux-dev-redux) | Yes | Uploaded to ComfyUI, LoadImage nodes + unCLIPConditioning | 3 |
| ComfyUI Flux + LoRA | Yes | LoRA weights loaded + ref images | 3 refs + 1 LoRA |

---

## Identified Gaps

### Gap 1: Actor Can Be Cast Without Reference Images

**Location:** `bindings.py:272-303`, `asset_library.py:615-627`

An Actor can be created with zero reference images, then cast in a Production Bible. The CastBinding creation endpoint only validates that the Actor exists, not that it has ActorRef rows. At generation time, the tag resolver returns an empty `reference_image_urls` list, and the image adapter generates without identity grounding.

**Impact:** The user believes they've configured a character, but keyframes won't visually match the intended actor. No warning in the UI.

### Gap 2: Flux Uses Only First Reference Image

**Location:** `keyframes.py:849`

When resolving Flux references, only `aref.reference_image_urls[0]` is used per asset. If an actor has 5 reference images (front, profile, 3/4, etc.), only the first is uploaded to ComfyUI. The Flux workflow supports up to 3 LoadImage nodes, but this limit applies across ALL assets in the shot, not per-asset.

**Impact:** Multi-angle reference images that would improve identity consistency are ignored for Flux.

### Gap 3: No Scene-Level Prompt @Tag Autocomplete

**Location:** `EditModeOverlay.tsx:1081-1086`

The scene-level prompt editor (where users write the initial scene concept) does NOT have @tag autocomplete extensions. Only shot-level editors (ShotEditorCard) have the CodeMirror `createAssetTagCompletion` extension. Users must type tags manually or reference the Tag Reference sheet.

**Impact:** Users may misspell tags or not know what tags are available when writing the scene prompt.

### Gap 4: No Validation That base_appearance_prompt Exists

**Location:** `tag_resolver.py:505-515`

The `text_description` field in `ResolvedAssetRef` comes from `Actor.base_appearance_prompt`, which is nullable. If empty, the tag resolution produces an empty description, and the storyboard LLM receives no visual guidance for that character beyond the character_name.

**Impact:** Characters without appearance prompts generate inconsistently — the LLM invents its own visual interpretation.

### Gap 5: Qwen Image Model Cannot Use References

**Location:** `keyframes.py:816-835`

When the image model is `qwen-image-edit` or `qwen-fast` (ComfyUI text-to-image), the keyframe generator calls `_generate_image_comfyui()` which only accepts a text prompt. No reference images are passed. This is the model currently in use for most scenes.

**Impact:** Even when an actor has reference images, they are completely unused if the image model is Qwen. The character's visual appearance is determined entirely by the text prompt.

### Gap 6: Storyboard Binding Registry Lacks Reference Image Metadata

**Location:** `manifest_service.py:940-951`

The `format_binding_registry()` function only includes the Actor's `base_appearance_prompt` text. It does not include information about whether reference images exist, how many there are, or their quality. The storyboard LLM has no way to know if visual identity grounding will be available downstream.

**Impact:** The LLM might generate shots that rely on specific visual details that can't be enforced because the image model doesn't support references.

### Gap 7: End Frame Reference Images Not Resolved for Flux

**Location:** `keyframes.py:947-1046`

The Flux reference resolution only runs for the start frame (shot 0). End frames use `_generate_image_conditioned()` for Vertex AI or basic `_generate_image_comfyui()` for ComfyUI. Flux end-frame generation doesn't re-resolve tags for reference conditioning.

**Impact:** End frames may drift from the established character identity, especially in multi-shot scenes.

---

## Plan to Close Gaps

### Phase A: Validation & Warnings (Low effort, high impact)

1. **Add ref count to CastBinding response** — Include `actor_ref_count` in the API response when listing cast bindings. The frontend can display a warning badge on actors with zero refs.

2. **Validate base_appearance_prompt on cast** — When creating a CastBinding, warn (not block) if the Actor has no `base_appearance_prompt`. Return a `warnings` array in the response.

3. **Add ref count to format_binding_registry** — Append `(N reference images)` or `(no reference images)` after each character entry so the storyboard LLM can factor this into its shot composition decisions.

### Phase B: Scene Editor UX (Medium effort)

4. **Add @tag autocomplete to scene prompt editor** — Pass the same `tagExtensions` (from `createAssetTagCompletion`) to the scene-level MarkdownEditorModal. The `boundAssets` data is already loaded via `getBoundAssetsSummary()`.

### Phase C: Multi-Ref Support for Flux (Medium effort)

5. **Upload multiple refs per actor to Flux** — Change the Flux resolution loop in `keyframes.py:849` to upload up to 3 ActorRef images per character (respecting the 3-node limit in the workflow). Prioritize `is_primary=True` ref first, then others.

6. **Resolve Flux refs for end frames** — Add Flux binding resolution to the end-frame generation path, mirroring the start-frame logic. Use the same `resolve_tags_with_assets()` call.

### Phase D: Adapter Abstraction (Higher effort)

7. **Create image adapter abstraction with ref support** — Define a common interface for image generation that declares reference image capability. Each adapter implements `supports_references() -> bool` and `generate(prompt, refs, ...) -> bytes`. This will make it straightforward to add ref support to new adapters (upcoming Imagen 4, DALL-E, etc.) without modifying keyframes.py.

8. **Qwen ref passthrough via IP-Adapter** — Investigate whether the Qwen ComfyUI workflow can accept reference images via IP-Adapter nodes. If feasible, wire up a `qwen-image-edit-with-refs` workflow template that includes LoadImage + IP-Adapter conditioning. This closes the biggest current gap since Qwen is the default image model.
