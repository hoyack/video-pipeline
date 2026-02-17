# VidPipe V2: Production-Grade AI Video Generation Pipeline

## Architecture Strategy Document

---

## 1. Executive Summary

VidPipe V2 evolves from a linear prompt→storyboard→keyframe→video pipeline into a **studio-grade production system** built around four core concepts:

1. **Manifesting** — The foundational Phase 0 that decomposes user uploads into a tagged asset pool. Users upload unlimited reference images; the system runs YOLO object/face detection, crops entities, reverse-engineers prompts via vision LLM, and registers everything in a manifest with unique tags. This is the project's DNA — everything downstream references it.
2. **Asset Registry** — A persistent, tagged pool of visual assets (characters, objects, environments, styles) seeded by Manifesting and enriched progressively as the pipeline generates content
3. **Scene Manifest** — A structured bill-of-materials for each scene, mapping tagged assets to spatial positions and roles
4. **Adaptive Prompt Rewriting** — Dynamic prompt enrichment that injects asset descriptions, reverse-engineered prompts, reference images, and manifest metadata into generation calls as scenes progress

The system draws from professional VFX pipeline practices — particularly asset management, previs/layout, and the iterative feedback loop between departments — but replaces human-in-the-loop handoffs with LLM-driven orchestration. The fork system fully inherits manifests and reference materials, allowing iterative refinement without re-uploading or re-tagging.

---

## 2. Veo 3.1 API Capabilities (Current as of Feb 2026)

### What We Can Actually Use

| Feature | API Support | Constraints |
|---------|------------|-------------|
| **Reference Images ("Ingredients")** | Up to **3 asset images** per generation | `referenceType: "asset"` — preserves subject appearance |
| **Style Images** | 1 style image | `referenceType: "style"` — Veo 2 only (`veo-2.0-generate-exp`), **not supported on Veo 3.1** |
| **First + Last Frame** | Start image + end image → transition video | Both images provided, Veo generates the motion between them with audio |
| **Scene Extension** | Extend existing clip by referencing its GCS URI | Continues from final 1 second / 24 frames of prior clip |
| **Native Audio** | Dialog, SFX, ambient — all prompt-driven | `generateAudio: true` — lip sync quality improving but imperfect |
| **Timestamp Prompting** | `[00:00-00:02] shot description` segments within single generation | Multi-shot within single 8s clip |
| **Seed** | `uint32` for deterministic regeneration | Same prompt + same references + same seed = same output |
| **Negative Prompt** | Describe what to exclude | `negativePrompt` parameter |
| **Resolution** | 720p or 1080p | 1080p on Veo 3.1, 4K upscale available in Flow |
| **Duration** | 4, 6, or 8 seconds | 8s mandatory when using reference images |
| **Aspect Ratio** | 16:9 or 9:16 | Native vertical support added Jan 2026 |
| **Sample Count** | 1-4 candidates per request | Generate multiple takes, pick best |
| **Add/Remove Object** | Insert or remove objects from generated video | **Veo 2 only**, no audio |

### Critical Constraints

- **3 reference images max** — This is the hard ceiling. Every generation call gets at most 3 asset images. This means we must be **strategic** about which 3 assets matter most for each scene.
- **8s clips with references** — When using asset reference images, clips are always 8 seconds.
- **No style images on Veo 3.1** — Style transfer requires falling back to `veo-2.0-generate-exp`.
- **GCS URIs preferred for production** — Avoids base64 transfer size limits and enables chaining/extension.
- **Billing: $0.40/s standard, $0.15/s fast** — Only charged on successful generation. Failed attempts may not be billed.

### The "Ingredients to Video" Workflow (Our Core Primitive)

Google's recommended pipeline, which maps directly to what we want to build:

1. **Generate reference images** with Gemini image models (character portraits, objects, environments)
2. **Pass best images as `referenceImages`** to Veo 3.1 with a descriptive prompt
3. **Reuse the same 3 images across shots** to preserve identity/continuity
4. **Keep reference images consistent**: same crop, same face angle, consistent clothing/background

This is essentially what the big studios do with "character sheets" and "environment plates" — we're just automating it.

---

## 3. The Asset Registry

### Concept

The Asset Registry is a project-scoped database of visual assets that grows throughout the pipeline. Assets enter the registry from two sources:

1. **User-uploaded reference images** (tagged at upload time)
2. **Pipeline-extracted assets** (detected and extracted from generated content)

Every asset gets a persistent ID, semantic tags, embedding vectors, and a canonical reference image suitable for feeding to Veo 3.1.

### Asset Schema

```
Asset {
  asset_id: UUID
  project_id: UUID
  
  # Classification
  asset_type: enum [CHARACTER, OBJECT, ENVIRONMENT, STYLE, PROP, VEHICLE, TEXT_ELEMENT]
  
  # Identity
  name: string                    # "Detective Marcus", "Red Sports Car", "Noir Office"
  manifest_tag: string            # "CHAR_01", "OBJ_REDCAR", "ENV_OFFICE" (auto-generated)
  user_tags: string[]             # User-provided tags from upload UI: ["protagonist", "detective"]
  
  # Source
  source: enum [USER_UPLOAD, USER_UPLOAD_EXTRACT, KEYFRAME_EXTRACT, CLIP_EXTRACT, GENERATED]
  source_upload_id: UUID | null   # Which uploaded image this was extracted from
  source_scene_index: int | null  # Which scene it was extracted from (pipeline extracts)
  source_timestamp: float | null  # Timestamp within clip
  
  # Visual Data
  reference_image_url: string     # GCS URI — the canonical "character sheet" image
  full_source_image_url: string   # GCS URI — the original uploaded image (before cropping)
  thumbnail_url: string
  crop_bbox: [x1, y1, x2, y2]    # Bounding box in source image (null if full-image asset)
  
  # Reverse-Engineered Prompt (LLM-generated from the image itself)
  reverse_prompt: string          # "A middle-aged man with salt-and-pepper close-cropped beard,
                                  #  wearing a rumpled brown trench coat and fedora hat. Tired 
                                  #  hazel eyes with crow's feet, strong jawline, 5 o'clock 
                                  #  shadow. Standing in a dimly lit doorway. Film noir aesthetic,
                                  #  high contrast, dramatic side lighting."
                                  #  
                                  # This is the KEY field — it's what gets injected into 
                                  # generation prompts. It describes the image as if you were 
                                  # writing a prompt TO RECREATE it. Different from a caption.
  
  # Semantic Description (LLM-generated, more narrative)
  visual_description: string      # "The main detective character. Appears weary and world-worn.
                                  #  His trench coat is his signature — always present. The 
                                  #  fedora is removable (scenes 1,3 = on, scenes 2,4 = off)."
  
  # Detection Metadata (from YOLO / CV analysis)
  detection_class: string | null  # YOLO class: "person", "car", "chair", etc.
  detection_confidence: float     # YOLO confidence score
  is_face_crop: bool              # True if this asset is specifically a face extraction
  
  # Embeddings (for similarity matching)
  face_embedding: float[512] | null     # ArcFace/FaceNet embedding for characters
  clip_embedding: float[768] | null     # CLIP embedding for general visual similarity
  
  # Pipeline Metadata
  appearances: [{scene_index, timestamp, bbox, confidence}]  # Where this asset appears
  created_at: datetime
  quality_score: float            # LLM-assessed quality of the reference image (1-10)
  
  # Inheritance (for fork system)
  inherited_from_project: UUID | null   # If forked, which project this asset came from
  is_inherited: bool                     # True = came from parent project, not re-generated
  
  # Cost tracking
  extraction_cost: float          # LLM + CV inference cost to create this asset
}
```

### Asset Lifecycle

```
PHASE 0: MANIFESTING (the project foundation)
  
  USER UPLOADS multiple reference images in GenerateForm
    → Each image gets user-provided metadata: name, type, optional tags
    → Example: user uploads 8 images — 3 character photos, 2 location shots,
      1 vehicle, 1 prop close-up, 1 mood/style reference
    → All stored in GCS immediately
  
  COLLAGE ASSEMBLY (Pillow)
    → All uploads assembled into a single numbered contact sheet
    → Grid layout: each image labeled with upload index + user-provided name
    → Purpose: gives the vision LLM spatial context of the full asset pool
    → Also useful for human review / project documentation
  
  YOLO OBJECT DETECTION SWEEP (local GPU, per-image)
    → Run YOLOv8/v12 on each uploaded image individually
    → Detect: persons, faces (YOLO-Face), vehicles, furniture, animals, etc.
    → For each detection above confidence threshold:
      - Crop the bounding box region with padding
      - Save crop as separate image → GCS
      - Record detection class, confidence, bbox coordinates
    → A single uploaded image may yield MULTIPLE assets:
      - Photo of two people at a table → CHAR_01, CHAR_02, OBJ_TABLE
      - Street scene → ENV_STREET, OBJ_CAR_BG, CHAR_PEDESTRIAN
  
  FACE EXTRACTION + EMBEDDING (local GPU)
    → For every detected person/face:
      - Extract face crop (aligned, normalized)
      - Generate ArcFace embedding (512-dim vector)
      - Generate CLIP embedding for the full person crop
    → Cross-match all face embeddings against each other:
      - Same person in multiple uploads? → Merge into single character asset
      - Multiple angles of same face = higher quality reference pool
  
  VISION LLM REVERSE-PROMPTING (Gemini, per-crop)
    → Each significant crop fed individually to Gemini 2.5 Flash/Pro
    → System prompt instructs the LLM to produce TWO outputs:
    
      1. REVERSE PROMPT: "Describe this image as if you were writing a 
         prompt to recreate it in an AI image generator. Include: physical 
         appearance, clothing, expression, pose, lighting, environment, 
         style, color palette, camera angle, and any distinguishing features.
         Be specific enough that the generated image would be recognizable
         as the same subject."
         
      2. VISUAL DESCRIPTION: "Provide a narrative description of this 
         subject for use in a production bible. What is distinctive about 
         them? What should remain consistent across scenes? What is 
         variable (removable hat, changeable expression)?"
    
    → The reverse_prompt is the critical output — it becomes the text 
      that gets injected into Veo/Imagen prompts to describe this asset
    → Cost: ~$0.01-0.03 per crop (Gemini 2.5 Flash with image input)
  
  MANIFEST TAG ASSIGNMENT (deterministic)
    → Auto-generate manifest tags based on type + sequence:
      - Characters: CHAR_01, CHAR_02, CHAR_03...
      - Objects: OBJ_01, OBJ_02... (or OBJ_REDCAR if user named it)
      - Environments: ENV_01, ENV_02...
      - Props: PROP_01, PROP_02...
      - Styles: STYLE_01...
    → User-provided names become the display name
    → Tags are the machine-readable identifiers used in manifests
  
  ASSET REGISTRY POPULATION
    → All assets registered with:
      - reference_image_url (the crop or full image)
      - reverse_prompt (LLM-generated recreation prompt)
      - visual_description (LLM-generated narrative description)
      - face_embedding / clip_embedding
      - detection metadata (class, confidence, bbox)
      - manifest_tag
      - user_tags
    → Registry is now the COMPLETE project foundation
    → Storyboarding phase receives this as structured context

PHASE 1: STORYBOARDING receives full Asset Registry
  → LLM creates scene manifests referencing manifest_tags
  → Can declare NEW assets not yet in registry (described but not visualized)
  → Manifest maps assets to scenes with spatial/role assignments

PHASE 2: KEYFRAME GENERATION
  → Generated keyframes analyzed by CV pipeline
  → New entities detected, reverse-prompted, and registered
  → Matched against existing assets via face/CLIP embedding similarity
  → Best extractions promoted to canonical reference images
  → Asset pool GROWS — later scenes benefit from earlier extractions

PHASE 3: VIDEO GENERATION
  → Up to 3 most relevant assets selected per scene (from enriched registry)
  → Reference images + enriched prompts (using reverse_prompts) sent to Veo 3.1
  → Generated clips analyzed post-generation
  → Asset appearances tracked, new entities registered

PROGRESSIVE ENRICHMENT (throughout Phases 2-3)
  → As pipeline progresses, asset pool grows richer
  → Reverse prompts get refined based on what models actually produce
  → Later scenes benefit from earlier scene's extractions
  → Prompts dynamically rewritten with accumulated knowledge

FORK INHERITANCE (when project is forked)
  → Entire Asset Registry is inherited by forked project
  → All manifests, reverse_prompts, embeddings carry forward
  → Forked project can add NEW reference uploads on top
  → Modified scenes re-run Manifesting only for new/changed uploads
  → Existing assets marked is_inherited=true, cost tracked as $0
```

---

## 4. Phase 0: Manifesting — The Project Foundation

### Why Manifesting Comes First

In professional VFX, nothing happens before the "asset bible" is built. The studio receives concept art, location photos, character references, and prop designs. These get cataloged, tagged, and distributed to every department before a single frame is rendered.

VidPipe V2's Manifesting phase is the automated equivalent. It happens **before storyboarding** because the storyboard LLM needs to know what assets exist and how to reference them. Without Manifesting, the storyboard generates generic descriptions. With it, the storyboard generates manifest-tagged, asset-aware scene breakdowns that directly map to the reference images the user uploaded.

### The Upload → Decompose → Reverse-Prompt → Tag Pipeline

```
USER UPLOADS (GenerateForm)
│
│  User uploads N images (no limit). For each image, user provides:
│    - Name (required): "Detective Marcus", "The Red Mustang"  
│    - Type (required): CHARACTER | OBJECT | ENVIRONMENT | STYLE | PROP
│    - Description (optional): "Main character, always wears trench coat"
│    - Tags (optional): ["protagonist", "noir", "recurring"]
│
│  Frontend sends all images + metadata in the generate request.
│  Images upload to GCS immediately, paths stored in reference_uploads table.
│
├─────────────────────────────────────────────────────────────────────┐
│                                                                     │
▼                                                                     │
STEP 1: CONTACT SHEET ASSEMBLY (Pillow)                               │
│                                                                     │
│  All uploaded images assembled into a numbered contact sheet:        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  PROJECT REFERENCE SHEET                                     │    │
│  │                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │ [1]      │  │ [2]      │  │ [3]      │  │ [4]      │   │    │
│  │  │ Det.     │  │ Femme    │  │ Red      │  │ Noir     │   │    │
│  │  │ Marcus   │  │ Fatale   │  │ Mustang  │  │ Office   │   │    │
│  │  │ CHAR     │  │ CHAR     │  │ VEHICLE  │  │ ENV      │   │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │
│  │                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │ [5]      │  │ [6]      │  │ [7]      │  │ [8]      │   │    │
│  │  │ Revolver │  │ Rain     │  │ Alley    │  │ Style    │   │    │
│  │  │          │  │ Scene    │  │ at Night │  │ Ref      │   │    │
│  │  │ PROP     │  │ ENV      │  │ ENV      │  │ STYLE    │   │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Contact sheet saved to GCS for:                                    │
│    - Project documentation / human review                           │
│    - Optional input to storyboard LLM (visual context of all refs) │
│                                                                     │
▼                                                                     │
STEP 2: YOLO DECOMPOSITION (local GPU, per-image)                     │
│                                                                     │
│  For each uploaded image individually:                               │
│                                                                     │
│  Input: uploaded_image_03.jpg ("Red Mustang")                       │
│  YOLO detects:                                                      │
│    ┌────────────────────────────────────────┐                       │
│    │  ┌─car (0.97)────────────────────┐    │                       │
│    │  │                                │    │                       │
│    │  │   ┌person (0.84)─┐            │    │                       │
│    │  │   │ (driver)     │            │    │                       │
│    │  │   └──────────────┘            │    │                       │
│    │  │                                │    │                       │
│    │  └────────────────────────────────┘    │                       │
│    │                      ┌tree (0.72)──┐  │                       │
│    │                      └─────────────┘  │                       │
│    └────────────────────────────────────────┘                       │
│                                                                     │
│  Output per image:                                                  │
│    - Primary subject crop (matches user's stated type)              │
│    - Additional detected entity crops (faces, objects, bg elements) │
│    - Detection metadata: class, confidence, bbox, area %            │
│                                                                     │
│  Crop strategy:                                                     │
│    - Faces: aligned crop with 30% padding for hair/shoulders        │
│    - Objects: tight bbox crop with 10% padding                      │
│    - Persons: full body crop when available                         │
│    - Environments: full image (no crop) — the whole image IS the    │
│      environment reference                                          │
│    - Small detections (<5% frame area): skip unless user-tagged     │
│                                                                     │
│  Face cross-matching (after all images processed):                  │
│    - Compute ArcFace embedding for every detected face              │
│    - Cosine similarity matrix across all faces                      │
│    - Similarity > 0.6 = same person → merge into single character   │
│    - Result: "Upload [1] face and Upload [5] background person      │
│      are the same person (similarity: 0.87) → merged as CHAR_01"   │
│                                                                     │
▼                                                                     │
STEP 3: VISION LLM REVERSE-PROMPTING (Gemini, per-crop)              │
│                                                                     │
│  Each significant crop is individually fed to Gemini 2.5 Flash.     │
│  NOT the collage — individual crops for maximum detail extraction.  │
│                                                                     │
│  System prompt (for CHARACTER type):                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ You are a visual prompt engineer for AI video generation.     │   │
│  │ Analyze this image and produce two outputs as JSON:           │   │
│  │                                                               │   │
│  │ 1. "reverse_prompt": Write a detailed prompt that would       │   │
│  │    recreate this subject in an AI image/video generator.      │   │
│  │    Include: physical build, skin tone, hair (color, style,    │   │
│  │    length), facial features (eye color/shape, nose, jaw,      │   │
│  │    facial hair), expression, clothing (every garment with     │   │
│  │    color and material), accessories, pose, lighting on the    │   │
│  │    subject, and camera angle. Be specific enough that the     │   │
│  │    generated result would be recognizable as this person.     │   │
│  │    Write in prompt style, not prose. ~100-150 words.          │   │
│  │                                                               │   │
│  │ 2. "visual_description": Narrative description for a          │   │
│  │    production bible. What is distinctive/signature about this │   │
│  │    subject? What must stay consistent across scenes? What is  │   │
│  │    variable (removable accessories, changeable expressions)?  │   │
│  │    ~50-80 words.                                              │   │
│  │                                                               │   │
│  │ 3. "suggested_name": If no name was provided, suggest one     │   │
│  │    based on appearance.                                       │   │
│  │                                                               │   │
│  │ 4. "quality_score": Rate 1-10 how suitable this image is as  │   │
│  │    a reference for AI generation (clear, well-lit, good       │   │
│  │    angle, unoccluded = higher score).                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Variant prompts for other types:                                   │
│    - OBJECT/PROP: focus on shape, material, color, condition, scale │
│    - ENVIRONMENT: focus on setting, architecture, lighting, mood,   │
│      time of day, weather, depth, key landmarks                     │
│    - STYLE: focus on color palette, contrast, grain, era,           │
│      cinematographic style, aspect ratio feel                       │
│    - VEHICLE: focus on make/model/era feel, color, condition,       │
│      angle, distinguishing features                                 │
│                                                                     │
│  Cost: ~$0.01-0.03 per crop × typical 10-20 crops = $0.10-0.60    │
│                                                                     │
▼                                                                     │
STEP 4: MANIFEST TAG ASSIGNMENT + REGISTRY POPULATION                 │
│                                                                     │
│  Auto-assign manifest tags:                                         │
│    Characters → CHAR_01, CHAR_02 (ordered by user upload order)     │
│    Objects → OBJ_01, OBJ_02 (or OBJ_MUSTANG if user named it)      │
│    Environments → ENV_01, ENV_02                                    │
│    Props → PROP_01, PROP_02                                         │
│    Vehicles → VEH_01, VEH_02                                       │
│    Styles → STYLE_01                                                │
│                                                                     │
│  Handle extracted sub-entities:                                     │
│    Upload [3] "Red Mustang" (VEHICLE) also yielded:                 │
│      → VEH_01 (the car itself — primary, inherits user's name)     │
│      → CHAR_03 (detected driver — secondary extraction)            │
│    User's name/type applies to primary subject.                     │
│    Extracted sub-entities get auto-names: "Person in Red Mustang"   │
│                                                                     │
│  Register ALL assets in Asset Registry with:                        │
│    - reference_image_url (crop or full image)                       │
│    - reverse_prompt (from Step 3)                                   │
│    - visual_description (from Step 3)                               │
│    - face_embedding / clip_embedding (from Step 2)                  │
│    - manifest_tag (from this step)                                  │
│    - detection metadata                                             │
│    - quality_score                                                  │
│                                                                     │
│  OUTPUT: Complete Asset Registry — the project's DNA                │
│  This is what the storyboard phase receives as context.             │
└─────────────────────────────────────────────────────────────────────┘
```

### Manifesting Output Example

For a user who uploads 6 images for a noir detective story:

```json
{
  "project_manifest": {
    "asset_count": 11,
    "from_uploads": 6,
    "from_extraction": 5,
    "face_merges": 1,
    
    "assets": [
      {
        "manifest_tag": "CHAR_01",
        "name": "Detective Marcus",
        "type": "CHARACTER",
        "source": "USER_UPLOAD (upload_01) + USER_UPLOAD (upload_04)",
        "note": "Same face detected in uploads 1 and 4 (similarity: 0.91), merged",
        "reverse_prompt": "A weathered middle-aged Caucasian man, mid-50s, salt-and-pepper close-cropped beard, deep-set hazel eyes with pronounced crow's feet, strong square jawline with 5 o'clock shadow. Wearing a rumpled dark brown wool trench coat over a loosened burgundy tie and white dress shirt with coffee stain on left cuff. Brown leather fedora hat. World-weary expression, slight downward turn of mouth. Shot from slightly below eye level, dramatic chiaroscuro side-lighting from the left, deep shadows on right side of face. Film noir aesthetic, high contrast, slight film grain.",
        "quality_score": 8.5,
        "reference_images": ["crop_upload01_face.jpg", "crop_upload04_full_body.jpg"]
      },
      {
        "manifest_tag": "CHAR_02", 
        "name": "Femme Fatale",
        "type": "CHARACTER",
        "source": "USER_UPLOAD (upload_02)",
        "reverse_prompt": "An elegant woman in her early 30s with jet-black hair styled in vintage finger waves, porcelain skin, deep red lipstick, high cheekbones, almond-shaped dark brown eyes with smoky eye makeup. Wearing a form-fitting emerald green satin evening dress with a plunging neckline and long black silk gloves. Pearl drop earrings. Confident half-smile, chin slightly raised. Three-quarter profile view, soft key light from above-right creating a glamorous catchlight in the eyes. 1940s Hollywood glamour lighting.",
        "quality_score": 9.0,
        "reference_images": ["crop_upload02_face.jpg", "crop_upload02_full.jpg"]
      },
      {
        "manifest_tag": "VEH_01",
        "name": "Red Mustang", 
        "type": "VEHICLE",
        "source": "USER_UPLOAD (upload_03)",
        "reverse_prompt": "A cherry red 1967 Ford Mustang Fastback, gleaming wet paint reflecting city lights, chrome bumpers and trim, round headlights, slightly lowered suspension. Parked at an angle on rain-slicked asphalt. Night scene, neon reflections on the wet surface, moody urban atmosphere.",
        "quality_score": 7.5
      },
      {
        "manifest_tag": "CHAR_03",
        "name": "Mustang Driver (extracted)",
        "type": "CHARACTER",
        "source": "USER_UPLOAD_EXTRACT (upload_03, bbox: [340,120,480,300])",
        "reverse_prompt": "Partial view of a young man visible through car windshield, dark hair, aviator sunglasses, leather jacket collar visible. Low quality reference — partially occluded by windshield glare.",
        "quality_score": 3.5,
        "note": "Low quality extraction. May need dedicated reference upload."
      }
      // ... ENV_01 (Noir Office), ENV_02 (Rain Scene), PROP_01 (Revolver), etc.
    ],
    
    "contact_sheet_url": "gs://project/manifesting/contact_sheet.jpg",
    "total_manifesting_cost": 0.42
  }
}
```

### What Gets Passed to Storyboarding

The storyboard LLM receives a structured context block:

```
AVAILABLE ASSETS FOR THIS PROJECT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CHAR_01] "Detective Marcus" (CHARACTER, quality: 8.5/10)
  Reverse prompt: "A weathered middle-aged Caucasian man, mid-50s, 
  salt-and-pepper close-cropped beard..."
  Production notes: Signature trench coat (always). Fedora is removable. 
  Two reference angles available.
  
[CHAR_02] "Femme Fatale" (CHARACTER, quality: 9.0/10)  
  Reverse prompt: "An elegant woman in her early 30s with jet-black hair 
  styled in vintage finger waves..."
  Production notes: Emerald dress is signature. Glamour lighting preferred.

[VEH_01] "Red Mustang" (VEHICLE, quality: 7.5/10)
  Reverse prompt: "A cherry red 1967 Ford Mustang Fastback..."

[CHAR_03] "Mustang Driver" (CHARACTER, quality: 3.5/10 — LOW)
  Note: Extracted from vehicle photo. Low quality reference.

[ENV_01] "Noir Office" (ENVIRONMENT, quality: 8.0/10)
  Reverse prompt: "A 1940s detective office with dark wood paneling..."

[PROP_01] "Revolver" (PROP, quality: 7.0/10)
  Reverse prompt: "A Smith & Wesson Model 10 revolver..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When creating scene manifests, reference assets by their [TAG].
You may declare NEW assets not yet in the registry — they will be
described textually and generated during keyframe phase.
```

---

## 5. The Scene Manifest

### Concept

The Scene Manifest is the bridge between Manifesting and generation. Built during storyboarding, it maps the tagged assets from Phase 0 into per-scene "bills of materials" — what assets appear, where they are spatially, what they're doing, and how they relate to each other. Think of it as a lightweight version of the "shot breakdown" in VFX, but now it has concrete references to actual images and reverse-prompts rather than generic descriptions.

### Manifest Schema

```
SceneManifest {
  scene_index: int
  
  # Spatial Layout
  composition: {
    shot_type: string           # "medium_shot", "wide_establishing", "close_up", "two_shot"
    camera_movement: string     # "slow_pan_left", "dolly_forward", "static", "crane_up"
    focal_point: string         # "CHAR_01 face", "OBJ_REDCAR", "center_frame"
  }
  
  # Asset Placements
  placements: [
    {
      asset_tag: "CHAR_01"        # References Asset Registry
      role: "subject"             # subject | background | prop | interaction_target
      position: "center_left"     # Spatial hint for prompt construction
      action: "sitting at desk, examining a letter"
      expression: "concerned, furrowed brow"
      wardrobe_note: "same trench coat as scene 1, hat removed"
    },
    {
      asset_tag: "ENV_OFFICE"
      role: "environment"
      lighting: "single desk lamp, warm pool of light, rest in shadow"
      time_of_day: "late night"
    },
    {
      asset_tag: "OBJ_LETTER"
      role: "prop"
      position: "on desk, under desk lamp"
      state: "slightly crumpled, handwritten text visible"
    }
  ]
  
  # Audio Direction
  audio: {
    dialogue: "Marcus mutters: 'This doesn't add up...'"
    sfx: ["paper rustling", "distant traffic through window"]
    ambient: "quiet office hum, ticking clock"
    music: "low, tense jazz piano"
  }
  
  # Continuity Notes
  continuity: {
    inherits_from: 2              # Continue from scene 2's end state
    lighting_match: "consistent with scene 2 — same office, same time"
    wardrobe_changes: []
    new_elements: ["OBJ_LETTER introduced"]
  }
  
  # Reference Image Selection (computed at generation time)
  selected_references: ["CHAR_01", "ENV_OFFICE", "OBJ_LETTER"]  # Max 3 for Veo
  reference_priority_reasoning: "Character face consistency is highest priority for 
    this medium shot. Office environment establishes setting. Letter is key plot prop."
}
```

### How the Manifest Gets Built

**Phase 1: LLM Manifest Generation (during storyboarding)**

The storyboard LLM call gets enhanced. Instead of just generating scene descriptions and prompts, it now also produces a structured manifest. The system prompt includes:

- The user's prompt
- All registered assets (user uploads + their descriptions)
- The asset tag namespace
- Instructions to assign manifest tags to recurring elements
- Instructions to declare new assets that will need to be created

**Phase 2: Manifest Enrichment (progressive)**

As scenes are generated and analyzed, the manifest is updated:
- Asset descriptions get refined based on what was actually generated
- Spatial relationships are validated against CV analysis
- Continuity notes are updated based on actual visual output

---

## 5. Computer Vision Analysis Pipeline

### Architecture Decision: Gemini Vision + YOLO Hybrid

Rather than choosing one approach, we use both for different strengths:

| Task | Tool | Reasoning |
|------|------|-----------|
| **Face detection + extraction** | YOLO-Face or RetinaFace | Fast, precise bounding boxes. Deterministic. No API cost per frame. |
| **Object detection** | YOLOv8/v12 (COCO classes) | 80+ object classes, real-time, runs locally on your RTX 4090 |
| **Semantic scene understanding** | Gemini 2.5 Flash | Rich natural language descriptions, spatial reasoning, character identification, context understanding |
| **Face identity matching** | ArcFace/FaceNet embeddings | Numeric similarity comparison — is this the same character across scenes? |
| **Asset description generation** | Gemini 2.5 Pro | Detailed, prompt-compatible descriptions of extracted crops |
| **Visual similarity** | CLIP embeddings | General-purpose visual matching for objects and environments |

### Analysis Pipeline (runs after each keyframe/clip generation)

```
GENERATED ASSET (keyframe image or video clip)
  │
  ├─► YOLO Object Detection (local GPU)
  │   ├─ Detect all objects with bounding boxes + confidence scores
  │   ├─ Detect all faces with bounding boxes
  │   └─ Output: detection_results[] with {class, bbox, confidence}
  │
  ├─► Face Processing (local GPU)
  │   ├─ Crop detected faces
  │   ├─ Generate ArcFace embedding per face
  │   ├─ Compare against Asset Registry face embeddings
  │   └─ Output: face_matches[] with {asset_id, similarity, is_new}
  │
  ├─► CLIP Embedding (local GPU)  
  │   ├─ Generate embedding for full frame
  │   ├─ Generate embedding for each significant crop
  │   └─ Output: frame_embedding, crop_embeddings[]
  │
  └─► Gemini Vision Analysis (API call)
      ├─ Input: full frame + detection overlay + existing manifest
      ├─ Prompt: "Analyze this generated scene. Identify all characters,
      │   objects, and environment details. For each detected entity,
      │   provide: (1) match to manifest tag if applicable, (2) detailed
      │   visual description suitable for re-prompting, (3) continuity
      │   assessment vs. intended manifest, (4) quality rating 1-10"
      └─ Output: structured JSON scene analysis

MERGE & RECONCILE
  ├─ Match YOLO detections with Gemini identifications
  ├─ Update Asset Registry with new appearances
  ├─ Register new assets not in registry
  ├─ Flag continuity issues (wrong wardrobe, missing prop, etc.)
  └─ Update cost tracker (YOLO: $0, Gemini: token cost)
```

### Video Clip Analysis (Frame Sampling Strategy)

For video clips (not just keyframes), we don't analyze every frame:

```
8-second clip at 24fps = 192 frames

Sampling strategy:
  - Frame 0 (first frame) — always analyze
  - Frame 48 (2s) — early action
  - Frame 96 (4s) — midpoint
  - Frame 144 (6s) — late action  
  - Frame 191 (last frame) — always analyze
  - Any frame with significant motion delta (computed via OpenCV frame diff)

Result: ~5-8 frames analyzed per clip instead of 192
  - YOLO runs on all sampled frames (fast, local)
  - Gemini Vision gets a composite: key frames + YOLO overlay + manifest context
```

### Cost Estimate for CV Pipeline

| Operation | Per Scene Cost | Notes |
|-----------|---------------|-------|
| YOLO detection (5-8 frames) | **$0.00** | Local GPU inference |
| ArcFace embedding (per face) | **$0.00** | Local GPU inference |
| CLIP embedding (per frame) | **$0.00** | Local GPU inference |
| Gemini 2.5 Flash analysis | **~$0.005-0.02** | ~2K input tokens (image + context), ~500 output |
| Gemini 2.5 Pro (detailed asset descriptions) | **~$0.01-0.05** | Only for new asset registration |

For a 10-scene project: approximately **$0.20-0.50** additional CV analysis cost. Negligible compared to the $32+ video generation cost.

---

## 6. Adaptive Prompt Rewriting

### The Core Innovation

Current VidPipe generates all scene prompts during storyboarding and never modifies them. VidPipe V2 **rewrites prompts dynamically** as the pipeline progresses, incorporating:

1. **Actual asset descriptions** (not just what was imagined, but what was generated)
2. **Reference image selections** (the 3 best-matching assets for this scene)
3. **Continuity corrections** (adjustments based on what previous scenes actually produced)
4. **Manifest-enriched detail** (spatial placement, wardrobe notes, prop states)

### Prompt Assembly Pipeline

```
ORIGINAL STORYBOARD PROMPT (from phase 1)
  "Detective Marcus sits at his desk, examining a mysterious letter"

MANIFEST ENRICHMENT (structural metadata)
  + shot_type: "medium shot"
  + camera_movement: "slow dolly forward"
  + lighting: "single desk lamp, warm pool, rest in shadow"
  + audio direction: dialog, sfx, ambient

ASSET INJECTION (from Asset Registry)
  + CHAR_01 description: "A middle-aged man with salt-and-pepper close-cropped 
    beard, wearing a rumpled brown trench coat. Tired hazel eyes with crow's 
    feet, strong jawline, 5 o'clock shadow. Fedora hat removed, placed on desk."
  + ENV_OFFICE description: "1940s noir detective office. Dark wood paneling, 
    frosted glass door reading 'Marcus Webb, Private Eye'. Venetian blinds 
    casting striped shadows. Cluttered desk with scattered papers and an empty 
    bourbon glass."
  + OBJ_LETTER: "A cream-colored envelope, slightly crumpled, with handwritten 
    address in blue ink. Wax seal broken."

CONTINUITY PATCH (from CV analysis of scene N-1)
  + "Marcus's trench coat should be unbuttoned (was unbuttoned in scene 2 
    end frame). The bourbon glass should be empty (was emptied in scene 2)."

REFERENCE IMAGE SELECTION
  + Select top 3 assets by relevance to this scene
  + Priority: CHAR_01 (face consistency), ENV_OFFICE (setting), OBJ_LETTER (plot)

═══════════════════════════════════════════════

FINAL ASSEMBLED PROMPT (sent to Veo 3.1):

  "Medium shot with slow dolly forward. A middle-aged man with a 
   salt-and-pepper beard in a rumpled, unbuttoned brown trench coat 
   sits behind a cluttered desk in a 1940s noir detective office. 
   Dark wood paneling, venetian blinds casting striped shadows. He 
   examines a cream-colored envelope with a broken wax seal under 
   the warm pool of a desk lamp, the rest of the office in shadow. 
   An empty bourbon glass sits nearby. He mutters: 'This doesn't 
   add up...' SFX: paper rustling, distant traffic through window. 
   Ambient: quiet office hum, ticking clock. Moody noir aesthetic, 
   cinematic, film grain."

  + referenceImages: [CHAR_01.png, ENV_OFFICE.png, OBJ_LETTER.png]
```

### The LLM Rewriter

A dedicated Gemini call assembles the final prompt. It receives:

```
System: You are a professional cinematographer and VFX director assembling 
a generation prompt for Veo 3.1. You will receive the original scene 
description, a scene manifest, asset descriptions, continuity notes, 
and reference image metadata. Your job is to compose a single, detailed, 
cinematic prompt that:

1. Follows the [Cinematography] + [Subject] + [Action] + [Context] + 
   [Style & Ambiance] formula
2. Incorporates specific visual details from asset descriptions 
   (clothing, features, distinguishing marks)
3. Enforces continuity with previous scenes
4. Includes audio direction (dialogue in quotes, SFX:, Ambient:)
5. Stays under 500 words (Veo prompt sweet spot)
6. Specifies which 3 reference images to attach and why

You are also selecting which 3 assets (from the full registry) should be 
attached as referenceImages. Choose based on:
- Subject face consistency (highest priority for character scenes)
- Environment matching (important for establishing shots)
- Key props (if they're plot-critical and close to camera)
```

### Cost for Prompt Rewriting

~$0.01-0.03 per scene (Gemini 2.5 Flash). For a 10-scene project: ~$0.10-0.30 additional.

---

## 7. Wireframe / Previs Composition (Advanced, Phase 2)

### Concept

Professional studios use "previs" (previsualization) — rough 3D mockups showing spatial layout before committing to expensive final renders. We can approximate this using **generated reference sheets and composition guides**.

### Approach: LLM-Directed Composition Sheet

Instead of actual 3D wireframes (which would require a completely different toolchain), we generate a **composition reference image** using Gemini's image generation:

```
Step 1: LLM generates a composition description from the manifest
  "Create a simple composition guide showing: [CHAR_01] at center-left 
   facing right, seated. [desk] at center. [OBJ_LETTER] on desk under 
   [lamp] at center-right. Camera at eye level, medium shot framing."

Step 2: Generate a rough composition image with Gemini Image
  - Low detail, focused on spatial relationships
  - Not photorealistic — more like a storyboard panel

Step 3: Use as FIRST FRAME for Veo generation
  - The composition image becomes the starting reference
  - Veo animates from this layout, preserving spatial arrangement
  
Step 4: Asset reference images guide identity/style
  - The 3 referenceImages ensure character/object consistency
  - The first frame ensures spatial arrangement
```

This effectively gives us:
- **Spatial control** via the first-frame image
- **Identity control** via the reference images
- **Motion control** via the prompt (camera movement, character actions)
- **Audio control** via prompt dialogue/SFX directions

### Reference Sheet Alternative

Another approach (simpler, fewer API calls): generate a single **reference sheet image** that composites all relevant assets onto one canvas:

```
┌────────────────────────────────────────────┐
│  SCENE 3 REFERENCE SHEET                   │
│                                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ CHAR_01 │  │ ENV_OFF │  │ OBJ_LTR │   │
│  │  (face) │  │ (wide)  │  │ (close) │   │
│  └─────────┘  └─────────┘  └─────────┘   │
│                                            │
│  Composition: Medium shot, CHAR_01 center- │
│  left, desk center, lamp right. Noir       │
│  lighting with single warm source.         │
└────────────────────────────────────────────┘
```

This could be assembled programmatically (PIL/Pillow) and fed as one of the 3 reference images, though this burns one of our precious reference image slots. Use only when spatial arrangement is critical.

---

## 8. Frontend Changes

### GenerateForm Enhancements

```
Current GenerateForm:
  [Prompt textarea]
  [Style picker]
  [Aspect ratio]
  [Duration controls]
  [Model selectors]
  [Generate button]

VidPipe V2 GenerateForm:
  [Prompt textarea]
  [Style picker]
  [Aspect ratio]
  [Duration controls]
  [Model selectors]
  
  ┌─ REFERENCE IMAGES (upload unlimited) ────────────────────────────┐
  │                                                                   │
  │  Drag & drop images here, or click to browse                     │
  │  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐          │
  │  │          📁 Drop reference images here              │          │
  │  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘          │
  │                                                                   │
  │  Uploaded (6 images):                                             │
  │  ┌──────────────────────────────────────────────────────────┐     │
  │  │ ┌──────┐  Name: [Detective Marcus___]                   │     │
  │  │ │ img1 │  Type: (•)CHAR ( )OBJ ( )ENV ( )PROP ( )VEH   │     │
  │  │ │      │  Desc: [Main character, always wears trench__] │     │
  │  │ └──────┘  Tags: [protagonist] [detective] [+]      [🗑] │     │
  │  ├──────────────────────────────────────────────────────────┤     │
  │  │ ┌──────┐  Name: [Femme Fatale________]                  │     │
  │  │ │ img2 │  Type: (•)CHAR ( )OBJ ( )ENV ( )PROP ( )VEH   │     │
  │  │ │      │  Desc: [Mysterious woman in green dress______] │     │
  │  │ └──────┘  Tags: [love-interest] [+]                [🗑] │     │
  │  ├──────────────────────────────────────────────────────────┤     │
  │  │ ┌──────┐  Name: [Red Mustang_________]                  │     │
  │  │ │ img3 │  Type: ( )CHAR ( )OBJ ( )ENV ( )PROP (•)VEH   │     │
  │  │ │      │  Desc: [1967 Mustang Fastback_______________]  │     │
  │  │ └──────┘  Tags: [chase-scene] [+]                  [🗑] │     │
  │  ├──────────────────────────────────────────────────────────┤     │
  │  │  ... 3 more images (scrollable) ...                     │     │
  │  └──────────────────────────────────────────────────────────┘     │
  │                                                                   │
  │  ℹ️ Images will be analyzed during Manifesting phase.             │
  │  Objects and faces will be auto-detected and tagged.              │
  │  The system will reverse-engineer prompts for each asset.         │
  └───────────────────────────────────────────────────────────────────┘
  
  [Generate Video]
  ↓
  First pipeline step: "Manifesting..." (progress shows contact sheet, 
  then YOLO detections, then reverse-prompts as they complete)
```

### New UI Components

1. **ReferenceImageUploader** — Multi-file drag-drop with per-image inline tagging form (name, type radio buttons, description textarea, tag chips). Scrollable list when >3 images. Delete button per image. No upload limit.

2. **ManifestingProgress** — New progress sub-view shown during Phase 0:
   - Shows contact sheet being assembled (live preview)
   - Shows YOLO detection boxes appearing on each image
   - Shows extracted crops appearing with auto-assigned tags
   - Shows reverse-prompts streaming in as Gemini completes each
   - Expandable per-image detail: "Upload 3 → 3 assets extracted (VEH_01, CHAR_03, ENV_BG)"

3. **AssetRegistryPanel** — Shows all project assets (uploaded + extracted + pipeline-generated), filterable by type and source. Each asset card shows: thumbnail, manifest_tag, name, reverse_prompt (collapsible), quality score badge, source indicator (📤 uploaded, 🔍 extracted, 🎬 pipeline). Appears on ProjectDetail and during Manifesting.

4. **ManifestViewer** — Read-only view of scene manifests during/after generation. Shows which assets map to which scenes, with visual lines connecting assets to scene cards. Highlights unresolved assets (declared but not yet generated).

5. **ContinuityTimeline** — Horizontal timeline showing where each asset appears across scenes. Color-coded bars per asset. Hover shows the reference image and its appearances.

6. **CVAnalysisOverlay** — Toggle on SceneCard to show YOLO detection boxes overlaid on keyframes, with manifest_tag labels.

### API Changes

```
POST /api/generate — enhanced request body
{
  prompt: string,
  style: string,
  // ... existing fields ...
  
  reference_uploads: [
    {
      image_data: base64,              // Raw image bytes
      name: "Detective Marcus",        // Required
      asset_type: "CHARACTER",         // Required: CHARACTER|OBJECT|ENVIRONMENT|PROP|VEHICLE|STYLE
      description: "Main character",   // Optional freeform
      tags: ["protagonist", "detective"]  // Optional user tags
    },
    {
      image_data: base64,
      name: "Red Mustang",
      asset_type: "VEHICLE",
      description: "1967 Mustang Fastback, cherry red",
      tags: ["chase-scene"]
    }
    // ... unlimited additional uploads
  ]
}

GET /api/projects/{id} — enhanced response
{
  // ... existing fields ...
  project_manifest: ProjectManifest,    // Phase 0 output summary
  assets: Asset[],                       // Full Asset Registry
  scene_manifests: SceneManifest[],      // Per-scene manifests
  contact_sheet_url: string,             // GCS URL for contact sheet
}

GET /api/projects/{id}/assets — list project assets
  ?type=CHARACTER                        // Filter by type
  ?source=USER_UPLOAD                    // Filter by source
  ?scene=3                               // Filter by scene appearance

GET /api/projects/{id}/assets/{asset_id} — asset detail
  Response includes: all appearances, reverse_prompt, visual_description,
  reference image URL, quality score, source tracking

POST /api/projects/{id}/assets — manually add asset mid-pipeline
  (same schema as reference_uploads entry, triggers incremental manifesting)

GET /api/projects/{id}/manifest — project manifest summary
GET /api/projects/{id}/scenes/{idx}/manifest — single scene manifest

POST /api/projects/{id}/fork — enhanced with asset_changes
  (see Fork System section for full schema)
```

---

## 9. Pipeline Flow (V2)

### State Machine: `pending → manifesting → storyboarding → keyframing → video_gen → stitching → complete`

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                    │
│  Prompt + Style + Models + Duration + Reference Images (N images,    │
│  each tagged with name, type, optional description + tags)           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 0: MANIFESTING (the project foundation)              NEW      │
│                                                                       │
│  This is the FIRST pipeline step. Nothing else runs until this       │
│  completes. It produces the Asset Registry that everything           │
│  downstream depends on.                                              │
│                                                                       │
│  Step 1: Upload all images to GCS                                    │
│  Step 2: Assemble contact sheet (Pillow grid with labels)            │
│  Step 3: YOLO detection sweep on each image (local GPU)              │
│    - Object detection (YOLOv8 COCO: 80 classes)                      │
│    - Face detection (YOLO-Face or RetinaFace)                        │
│    - Crop all significant detections with padding                    │
│    - Cross-match faces via ArcFace embeddings (merge same person)    │
│  Step 4: Vision LLM reverse-prompting (Gemini, per-crop)            │
│    - Generate reverse_prompt (recreation-style prompt text)          │
│    - Generate visual_description (production bible entry)            │
│    - Assess quality_score (1-10)                                     │
│  Step 5: Assign manifest tags (CHAR_01, OBJ_02, ENV_01, etc.)      │
│  Step 6: Populate Asset Registry (all fields)                        │
│                                                                       │
│  Commit: Asset Registry saved to DB. Contact sheet saved to GCS.    │
│  Cost: ~$0.10-0.60 depending on number of uploads/crops             │
│                                                                       │
│  OUTPUT: Complete Asset Registry with reverse_prompts, embeddings,   │
│  tags, and quality scores. This IS the project foundation.           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1: MANIFEST-AWARE STORYBOARDING                    ENHANCED   │
│                                                                       │
│  Enhanced Gemini call receives:                                       │
│    - User prompt                                                      │
│    - Full Asset Registry (names, reverse_prompts, tags, quality)     │
│    - Contact sheet image (optional, for visual context)              │
│    - System prompt requiring manifest-tagged scene breakdowns        │
│                                                                       │
│  Outputs (per scene):                                                │
│    - Scene description (existing)                                     │
│    - Scene manifest (asset placements referencing manifest_tags)     │
│    - Asset declarations (new assets described but not yet visualized)│
│    - Continuity notes (what carries from previous scene)             │
│    - Audio direction (dialog, SFX, ambient)                          │
│    - Reference selection rationale (which 3 of N assets matter most) │
│                                                                       │
│  The storyboard LLM uses reverse_prompts to write scene prompts     │
│  that are VISUALLY ALIGNED with the actual reference images.         │
│                                                                       │
│  Cost: ~$0.05-0.15 (larger prompt with asset context)               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 2: KEYFRAME GENERATION (enhanced)                             │
│                                                                       │
│  For each scene:                                                      │
│    1. PROMPT REWRITE: LLM assembles generation prompt from manifest   │
│       + asset reverse_prompts + continuity notes                     │
│    2. REFERENCE SELECTION: Pick 3 most relevant assets for this scene │
│    3. GENERATE: Imagen/Gemini Image with enriched prompt              │
│    4. CV ANALYSIS: Run detection pipeline on generated keyframe       │
│       - YOLO: object/face bounding boxes                              │
│       - Face embedding: match against Asset Registry                  │
│       - Gemini Vision: semantic analysis + continuity check           │
│    5. ASSET EXTRACTION: Crop new entities, reverse-prompt them,      │
│       register in Asset Registry (same flow as Manifesting Step 3-5) │
│    6. CONTINUITY CHECK: Flag issues for next scene's rewrite          │
│    7. Daisy-chain: end frame → next scene start (existing behavior)   │
│                                                                       │
│  Key insight: Scene 5's keyframe benefits from assets extracted       │
│  from scenes 1-4. The asset pool GROWS as the pipeline progresses.   │
│                                                                       │
│  Cost: existing keyframe cost + ~$0.03 per scene for CV + rewrite    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 3: VIDEO GENERATION (enhanced)                                │
│                                                                       │
│  For each scene:                                                      │
│    1. PROMPT REWRITE (again): Fresh rewrite with latest asset pool    │
│       - May differ from keyframe prompt (richer descriptions now)     │
│       - Includes audio direction from manifest                       │
│       - Injects reverse_prompts for the 3 selected reference assets  │
│    2. REFERENCE IMAGE ATTACHMENT:                                      │
│       - Up to 3 asset reference images from Registry                  │
│       - First frame from keyframe daisy-chain (separate `image` param)│
│       - Hybrid: first frame for spatial + 3 refs for identity         │
│    3. VEO GENERATION: Submit with enriched prompt + references        │
│    4. POST-GENERATION CV ANALYSIS:                                    │
│       - Sample 5-8 frames from clip                                   │
│       - YOLO detection sweep (local GPU, free)                        │
│       - Face matching against Registry                                │
│       - Gemini Vision: full scene analysis + quality assessment       │
│    5. ASSET EXTRACTION: New entities from video → Registry            │
│    6. CONTINUITY SCORING: Rate adherence to manifest                  │
│    7. Existing: safety remediation, retry logic                       │
│                                                                       │
│  Cost: existing video cost + ~$0.05 per scene for CV + rewrite       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 4: STITCHING (existing, unchanged)                            │
│                                                                       │
│  ffmpeg concat/xfade, audio preservation                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Cost Tracking Updates

### New Cost Categories

```python
COST_TABLE_V2 = {
    # Existing
    "text_generation": {...},
    "image_generation": {...},
    "video_generation": {...},
    
    # New — Phase 0: Manifesting
    "manifesting": {
        "gemini_vision_reverse_prompt": 0.0,  # Per-crop reverse-prompt call
        "gemini_vision_description": 0.0,     # Per-crop visual description
        "yolo_inference": 0.0,                # Local GPU — $0
        "face_embedding": 0.0,                # Local GPU — $0
        "clip_embedding": 0.0,                # Local GPU — $0
        "contact_sheet_assembly": 0.0,        # Pillow — $0
    },
    
    # New — Pipeline enhancements
    "asset_analysis": {
        "gemini_2.5_flash_vision": 0.0,     # Per-scene CV analysis (token-based)
        "gemini_2.5_pro_description": 0.0,   # Per-asset detailed description
        "yolo_inference": 0.0,                # Local GPU — $0
        "face_embedding": 0.0,                # Local GPU — $0
        "clip_embedding": 0.0,                # Local GPU — $0
    },
    "prompt_rewriting": {
        "keyframe_rewrite": 0.0,             # Per-scene LLM call
        "video_rewrite": 0.0,                # Per-scene LLM call
    },
    "manifest_generation": {
        "storyboard_manifest": 0.0,          # Enhanced storyboard call
    },
    "previs_composition": {
        "composition_image": 0.0,            # Optional composition guide generation
    },
}
```

### Estimated Total Cost Comparison

For a 10-scene, 80-second video:

| Category | V1 Cost | V2 Cost | Delta |
|----------|---------|---------|-------|
| **Manifesting (Phase 0)** | $0.00 | ~$0.40 | +$0.40 (YOLO free, ~15 Gemini crops) |
| Storyboard (Gemini text) | ~$0.05 | ~$0.15 | +$0.10 (manifest generation) |
| Keyframe generation | ~$1.00 | ~$1.00 | $0.00 (same) |
| Video generation (Veo 3.1) | ~$32.00 | ~$32.00 | $0.00 (same, but higher quality due to references) |
| CV Analysis (post-gen) | $0.00 | ~$0.50 | +$0.50 |
| Prompt rewriting | $0.00 | ~$0.30 | +$0.30 |
| **Total** | **~$33.05** | **~$34.35** | **+$1.30 (~4%)** |

**For ~4% additional cost, we get**: manifest-tagged asset pool with reverse-prompts, character consistency via reference images, continuity enforcement, enriched prompts, and a growing reference pool. The Manifesting phase alone ($0.40) delivers the entire project foundation that every subsequent phase builds on.

---

## 11. Technology Stack Additions

### Backend (Python)

```
New dependencies:
  - ultralytics          # YOLOv8/v12 inference
  - insightface          # ArcFace face embeddings (or deepface)
  - transformers         # CLIP embeddings (openai/clip-vit-large-patch14)
  - opencv-python        # Frame extraction, image processing
  - Pillow               # Image cropping, composition sheet assembly
  - numpy                # Embedding math, similarity computation
  - faiss-cpu            # Fast similarity search (optional, for large asset pools)
```

### GPU Requirements

Your RTX 4090 handles all local inference easily:
- YOLOv8 medium: ~5ms per frame
- ArcFace: ~2ms per face
- CLIP: ~10ms per image
- Total per scene (5-8 frames): <100ms

The dual 3090s could run these in parallel if the 4090 is busy with other workloads.

### Frontend

```
New dependencies:
  - react-dropzone       # Drag-drop file uploads
  - (or use native HTML5 drag-drop to avoid dependencies)
```

### Database Schema Additions

```sql
-- Asset Registry table (populated by Manifesting phase)
CREATE TABLE assets (
    asset_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id),
    asset_type TEXT NOT NULL,  -- CHARACTER, OBJECT, ENVIRONMENT, STYLE, PROP, VEHICLE
    name TEXT NOT NULL,
    manifest_tag TEXT NOT NULL, -- CHAR_01, OBJ_REDCAR, ENV_01, etc.
    user_tags TEXT[],           -- User-provided tags from upload UI
    
    -- Source tracking
    source TEXT NOT NULL,       -- USER_UPLOAD, USER_UPLOAD_EXTRACT, KEYFRAME_EXTRACT, etc.
    source_upload_id UUID REFERENCES reference_uploads(upload_id),
    source_scene_index INTEGER,
    
    -- Visual data
    reference_image_url TEXT NOT NULL,   -- GCS URI for canonical reference image (crop)
    full_source_image_url TEXT,          -- GCS URI for original full image
    thumbnail_url TEXT,
    crop_bbox REAL[],                    -- [x1, y1, x2, y2] in source image
    
    -- Reverse-engineered prompt (the KEY field for generation)
    reverse_prompt TEXT,                 -- Prompt-style description to recreate this asset
    visual_description TEXT,             -- Production bible narrative description
    
    -- Detection metadata
    detection_class TEXT,                -- YOLO class: "person", "car", etc.
    detection_confidence REAL,
    is_face_crop BOOLEAN DEFAULT FALSE,
    
    -- Embeddings (serialized numpy arrays)
    face_embedding BYTEA,               -- ArcFace 512-dim
    clip_embedding BYTEA,               -- CLIP 768-dim
    
    -- Quality & scoring
    quality_score REAL,                  -- LLM-assessed 1-10
    
    -- Fork inheritance
    inherited_from_project UUID,         -- Parent project if forked
    inherited_from_asset UUID,           -- Parent asset ID if forked
    is_inherited BOOLEAN DEFAULT FALSE,  -- True = inherited, not re-processed
    
    -- Cost & timestamps
    extraction_cost REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_assets_project ON assets(project_id);
CREATE INDEX idx_assets_manifest_tag ON assets(project_id, manifest_tag);

-- Asset appearances tracking (where each asset shows up across scenes)
CREATE TABLE asset_appearances (
    id SERIAL PRIMARY KEY,
    asset_id UUID REFERENCES assets(asset_id),
    scene_index INTEGER NOT NULL,
    timestamp_sec REAL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    confidence REAL,
    source TEXT  -- 'keyframe', 'clip', 'manifest_planned'
);

-- Scene manifests (produced by storyboarding, updated by pipeline)
CREATE TABLE scene_manifests (
    project_id UUID REFERENCES projects(project_id),
    scene_index INTEGER NOT NULL,
    manifest_json JSONB NOT NULL,          -- Full manifest (placements, roles, etc.)
    original_prompt TEXT,                   -- Raw storyboard prompt
    rewritten_keyframe_prompt TEXT,         -- After prompt assembly for keyframe gen
    rewritten_video_prompt TEXT,            -- After prompt assembly for video gen
    selected_reference_tags TEXT[],         -- Which 3 assets selected for this scene
    cv_analysis_json JSONB,                -- Post-generation analysis results
    continuity_score REAL,                 -- How well generation matched manifest
    PRIMARY KEY (project_id, scene_index)
);

-- User reference image uploads (raw uploads before decomposition)
CREATE TABLE reference_uploads (
    upload_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id),
    original_filename TEXT,
    gcs_url TEXT NOT NULL,
    
    -- User-provided metadata from upload UI
    user_name TEXT NOT NULL,               -- "Detective Marcus"
    user_type TEXT NOT NULL,               -- "CHARACTER", "OBJECT", etc.
    user_description TEXT,                 -- Optional freeform description
    user_tags TEXT[],                      -- Optional tags
    
    -- Processing results
    yolo_detections_json JSONB,            -- Raw YOLO results for this image
    assets_extracted INTEGER DEFAULT 0,    -- How many assets were extracted
    processing_cost REAL DEFAULT 0.0,
    
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Project manifest (the Phase 0 output — project-level summary)
CREATE TABLE project_manifests (
    project_id UUID PRIMARY KEY REFERENCES projects(project_id),
    
    -- Summary stats
    total_uploads INTEGER DEFAULT 0,
    total_assets INTEGER DEFAULT 0,
    total_from_extraction INTEGER DEFAULT 0,
    total_face_merges INTEGER DEFAULT 0,
    
    -- Contact sheet
    contact_sheet_url TEXT,                 -- GCS URL for the assembled contact sheet
    
    -- Cost
    manifesting_cost REAL DEFAULT 0.0,
    
    -- Fork tracking
    inherited_from_project UUID,           -- If forked, parent project's manifest
    inherited_asset_count INTEGER DEFAULT 0,
    new_upload_count INTEGER DEFAULT 0,    -- Uploads added in this fork
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 12. Implementation Phases

### Phase 1: Manifesting + Reference Upload (Foundation — build this first)
- Frontend: Multi-image upload in GenerateForm with per-image tagging (name, type, description, tags)
- Backend: `reference_uploads` table, GCS upload endpoint
- Pipeline: Manifesting phase (contact sheet, YOLO decomposition, Gemini reverse-prompting, tag assignment, Asset Registry population)
- Backend: `assets` table, `project_manifests` table, CRUD API for assets
- State machine: add `manifesting` state between `pending` and `storyboarding`
- Frontend: Asset Registry panel showing all manifested assets with reverse-prompts
- **This phase alone is valuable** — even without downstream pipeline changes, users can see their uploads decomposed, tagged, and described

### Phase 2: Manifest-Aware Storyboarding
- Enhanced storyboard system prompt receives full Asset Registry context
- Storyboard output includes per-scene manifests with manifest_tag references
- `scene_manifests` table for structured manifest storage
- Basic prompt rewriting (inject reverse_prompts into generation prompts)
- Frontend: ManifestViewer on ProjectDetail showing asset→scene mapping

### Phase 3: Reference Image Passthrough to Veo
- Wire the reference image selection logic (pick 3 per scene from manifest)
- Pass `referenceImages` with `referenceType: "asset"` to Veo 3.1 API
- Implement hybrid approach: first-frame daisy-chain + 3 reference images
- Update cost tracking for reference-aware generation
- Frontend: Show which 3 references were selected per scene in SceneCard

### Phase 4: CV Analysis Pipeline (Post-Generation)
- YOLO + face embedding + CLIP local inference setup
- Post-keyframe and post-clip analysis
- Asset extraction from generated content (same flow as Manifesting decomposition)
- Gemini Vision scene analysis integration
- Progressive asset enrichment (scenes 1-4 feed scene 5+)
- Frontend: CV analysis overlay on SceneCards, asset appearance timeline

### Phase 5: Adaptive Prompt Rewriting
- Full prompt assembly pipeline with dedicated LLM rewriter
- Continuity checking and correction between scenes
- Reference image selection optimization (quality + relevance scoring)
- Reverse-prompt refinement based on what models actually produce

### Phase 6: Previs / Composition Guides
- Composition image generation from manifests
- First-frame spatial control integration
- Reference sheet assembly (Pillow compositing for complex scenes)

### Phase 7: Fork System Integration (critical for production workflows)
- **See detailed fork design below**

---

## 13. Fork System: Manifest and Asset Inheritance

### The Problem

V1 forking already handles scene invalidation, keyframe/clip inheritance, and cost tracking. But V2 adds new artifacts that must also inherit correctly: the Asset Registry, scene manifests, reverse-prompts, reference uploads, and the contact sheet. Without proper inheritance, forking a project would lose the entire Manifesting foundation.

### Fork Inheritance Rules

```
WHEN USER FORKS A PROJECT:

1. ASSET REGISTRY — Full Copy
   ├─ ALL assets from parent project are copied to forked project
   ├─ Each copied asset: is_inherited=true, inherited_from_asset=parent_asset_id
   ├─ Cost for inherited assets: $0 (no re-processing)
   ├─ Inherited assets retain: reverse_prompt, visual_description, embeddings,
   │   reference_image_url (GCS URLs are shared, not duplicated)
   └─ Inherited assets are READ-ONLY unless user explicitly modifies them

2. PROJECT MANIFEST — Inherited + Extensible
   ├─ Parent's project_manifest copied with inherited_from_project set
   ├─ inherited_asset_count = parent's total_assets
   ├─ new_upload_count = 0 (increments as user adds new uploads in fork)
   └─ Contact sheet: inherited (shared GCS URL)

3. SCENE MANIFESTS — Selective Inheritance
   ├─ Unchanged scenes: manifest inherited as-is
   ├─ Modified scenes: manifest marked for regeneration
   ├─ New scenes: blank manifest, will be generated during storyboarding
   └─ Deleted scenes: manifest removed, but assets remain in Registry
       (they may be used by other scenes)

4. REFERENCE UPLOADS — Inherited + Addable
   ├─ Parent's reference_uploads are visible in forked project
   ├─ User CAN add new reference uploads in the fork
   ├─ New uploads trigger INCREMENTAL MANIFESTING:
   │   - Only the new uploads go through YOLO → reverse-prompt → tagging
   │   - Existing assets are NOT re-processed
   │   - New assets get tags that don't collide (CHAR_04, OBJ_05, etc.)
   │   - Face embeddings cross-matched against ALL assets (inherited + new)
   │   - Contact sheet regenerated to include new uploads
   └─ Cost: only the new uploads' processing cost

5. INVALIDATION INTERACTION
   ├─ Adding a new reference upload does NOT invalidate any scenes
   │   (it just adds to the asset pool)
   ├─ MODIFYING an existing asset's reverse_prompt or reference image:
   │   → Invalidates scenes where that asset appears (from scene manifest)
   │   → Uses existing _compute_invalidation logic
   ├─ DELETING an asset from the registry:
   │   → Scenes using that asset get flagged for manifest regeneration
   │   → Storyboard phase re-runs for affected scenes
   ├─ SWAPPING a reference image (replace CHAR_01's reference):
   │   → Re-runs reverse-prompting for just that asset
   │   → Invalidates scenes using CHAR_01 from the swap point forward
   └─ All invalidation logic extends existing fork system's
       _compute_invalidation method
```

### Fork UI Changes (EditForkPanel Additions)

```
EXISTING EditForkPanel:
  ┌─ Project-level edits (prompt, style, models)
  ├─ Per-scene edits (prompts, delete, keyframe clear)
  └─ Generate Fork button

V2 EditForkPanel ADDITIONS:
  ┌─ ASSET REGISTRY SECTION ─────────────────────────────────────┐
  │                                                               │
  │  Inherited Assets (from parent project):                      │
  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐          │
  │  │CHAR01│  │CHAR02│  │VEH01 │  │ENV01 │  │PROP01│          │
  │  │Marcus│  │Fatale│  │Mustng│  │Office│  │Revlvr│          │
  │  │ 🔒   │  │ 🔒   │  │ ✏️   │  │ 🔒   │  │ ❌   │          │
  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘          │
  │                                                               │
  │  🔒 = inherited unchanged                                    │
  │  ✏️ = modified (amber border, shows diff)                    │
  │  ❌ = marked for removal                                     │
  │                                                               │
  │  [+ Add New Reference Images]   ← triggers incremental       │
  │                                    manifesting on fork        │
  │                                                               │
  │  Per-asset actions:                                           │
  │    - Swap reference image (re-upload, re-reverse-prompt)      │
  │    - Edit reverse_prompt manually (override LLM output)       │
  │    - Edit name / tags                                         │
  │    - Remove from project (doesn't delete GCS files)           │
  │                                                               │
  │  ⚠️ Modified assets will invalidate scenes using them.       │
  └───────────────────────────────────────────────────────────────┘
```

### Fork Request Changes (buildForkRequest additions)

```javascript
// Existing fork request fields (unchanged)
{
  parent_project_id: "uuid",
  prompt_changes: {...},
  scene_changes: {...},
  
  // NEW: Asset changes
  asset_changes: {
    // Assets from parent that were modified
    modified_assets: [
      {
        asset_id: "uuid",
        changes: {
          reverse_prompt: "new prompt text...",  // Manual edit
          reference_image: base64 | null,        // Swapped image
          name: "New Name",                      // Renamed
        }
      }
    ],
    
    // Assets from parent that were removed
    removed_asset_ids: ["uuid1", "uuid2"],
    
    // Brand new reference uploads added in this fork
    new_uploads: [
      {
        image_data: base64,
        name: "New Character",
        type: "CHARACTER",
        description: "Introduced in act 2",
        tags: ["secondary", "villain"]
      }
    ]
  }
}
```

### Backend Fork Processing (enhanced _process_fork)

```python
async def _process_fork(self, fork_request):
    # 1. Existing: copy project, compute scene invalidation
    new_project = await self._copy_project(fork_request)
    
    # 2. NEW: Inherit Asset Registry
    parent_assets = await self._get_project_assets(fork_request.parent_id)
    for asset in parent_assets:
        if asset.asset_id in fork_request.removed_asset_ids:
            continue  # Skip removed assets
        
        inherited = asset.copy()
        inherited.project_id = new_project.id
        inherited.is_inherited = True
        inherited.inherited_from_asset = asset.asset_id
        inherited.inherited_from_project = fork_request.parent_id
        
        if asset.asset_id in fork_request.modified_assets:
            changes = fork_request.modified_assets[asset.asset_id]
            if changes.reverse_prompt:
                inherited.reverse_prompt = changes.reverse_prompt
            if changes.reference_image:
                # Upload new image, re-run reverse-prompting
                inherited.reference_image_url = await self._upload_to_gcs(changes.reference_image)
                inherited.reverse_prompt = await self._reverse_prompt(inherited.reference_image_url)
                inherited.is_inherited = False  # Modified = no longer pure inheritance
            inherited.extraction_cost = 0.0  # Track incremental cost
        
        await self._save_asset(inherited)
    
    # 3. NEW: Process new uploads (incremental manifesting)
    if fork_request.new_uploads:
        await self._run_incremental_manifesting(
            new_project.id, 
            fork_request.new_uploads,
            existing_assets=inherited_assets  # For face cross-matching
        )
    
    # 4. NEW: Inherit scene manifests (respecting invalidation)
    invalidation_point = self._compute_invalidation(fork_request)
    parent_manifests = await self._get_scene_manifests(fork_request.parent_id)
    for manifest in parent_manifests:
        if manifest.scene_index < invalidation_point:
            # Scene unchanged — inherit manifest as-is
            await self._copy_manifest(manifest, new_project.id)
        else:
            # Scene invalidated — manifest will be regenerated during storyboarding
            pass
    
    # 5. Existing: proceed with generation from invalidation point
    await self._run_pipeline(new_project, start_from=invalidation_point)
```

---

## 13. Key Design Decisions

### Decision 1: 3 Reference Image Strategy

Since Veo 3.1 only allows 3 reference images, every scene needs a **selection strategy**:

| Scene Type | Slot 1 | Slot 2 | Slot 3 |
|-----------|--------|--------|--------|
| Character close-up | Primary character face | Character full body | Environment/mood |
| Two-shot dialogue | Character A face | Character B face | Environment |
| Establishing shot | Environment wide | Key prop/vehicle | Style reference |
| Action scene | Primary character | Action context | Environment |

The LLM rewriter makes this decision per-scene based on the manifest.

### Decision 2: When to Use First+Last Frame vs. Reference Images

- **First+Last Frame**: Use for precise transitions between scenes (your existing daisy-chain approach). Gives spatial control but NO identity control from reference images.
- **Reference Images**: Use for identity/consistency. Gives character consistency but less spatial precision.
- **Hybrid**: Use first frame as `image` input + up to 3 `referenceImages`. The API supports this — the `image` parameter is separate from `referenceImages`.

**Recommendation**: Use hybrid approach. First frame from daisy-chain for spatial continuity + 3 reference images for identity. This gives us both.

### Decision 3: Gemini Vision vs. YOLO for Detection

Use both. YOLO is free (local GPU), fast, and gives precise bounding boxes. Gemini Vision is expensive per-call but gives semantic understanding ("this is the same detective from scene 1, his coat is now unbuttoned"). They complement each other.

### Decision 4: Where to Run CV Inference

Local GPU (your RTX 4090). YOLOv8, ArcFace, and CLIP are all small models that run in milliseconds. No reason to pay for cloud inference. The pipeline already needs local ffmpeg for stitching — adding Python CV inference to the same host is natural.

---

## 14. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Veo 3.1 reference images don't preserve identity well enough | Use seed parameter for determinism. Multiple candidates (sampleCount=4). Iterative refinement with quality scoring. |
| YOLO misdetects / false positives corrupt Asset Registry | Confidence thresholds. LLM validation pass. Manual curation option in UI. |
| Prompt rewriting makes prompts too long for Veo | LLM rewriter has 500-word target constraint. Truncation fallback. |
| CV analysis adds too much latency | All YOLO/embedding runs are <100ms. Gemini Vision is async. Can parallelize with next scene's generation. |
| Asset pool grows too large | Pruning by quality_score. Deduplication via embedding similarity. Per-project scoping. |
| 3-image limit is too constraining for complex scenes | Prioritization logic in LLM rewriter. Composition sheet approach to pack spatial info into 1 image. Prompt text carries remaining context. |
| Cost estimation becomes complex | Every new call type gets explicit cost tracking. Dashboard already shows cost breakdowns. |

---

## 15. What Makes This "Hollywood Ready"

Professional VFX pipelines are built around:

1. **Asset management systems** (ShotGrid/ftrack) — We have the Asset Registry
2. **Shot breakdowns** (what's in each shot) — We have the Scene Manifest
3. **Reference sheets** (character bibles, environment plates) — We have auto-generated asset descriptions + reference images
4. **Previs/layout** (spatial planning) — We have composition guides + first-frame control
5. **Continuity departments** (script supervisors tracking wardrobe/props) — We have CV analysis + continuity scoring
6. **Iterative refinement** (dailies → notes → rework) — We have the fork system + adaptive rewriting
7. **Cost tracking** (VFX budget management) — We have per-operation cost tracking

The innovation is that all of these traditionally-human roles are orchestrated by LLMs, with computer vision providing the "eyes" that keep the system grounded in what was actually generated rather than what was intended.

No other AI video pipeline I'm aware of combines manifest-driven generation, progressive asset extraction, adaptive prompt rewriting, and CV-validated continuity into a single automated workflow. This is genuinely novel.