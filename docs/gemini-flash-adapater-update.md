# VidPipe — Nano Banana Vertex Adapter Specification

**Document:** `ADAPTER-SPEC-001`  
**Version:** 1.0  
**Adapter ID:** `vertex_nano_banana`  
**Status:** Draft  

---

## 1. Overview

This document specifies the behavior of the **Vertex Nano Banana Adapter** — the image generation
adapter in VidPipe that calls the Google Vertex AI Gemini image generation models (Nano Banana
family). It receives a resolved reference payload from the screenwriter pipeline, constructs a
properly structured multimodal API call, and returns one or more generated keyframe images.

This specification defines the complete adapter contract: input schema, reference resolution,
prompt construction strategy, Vertex API call structure, output schema, error handling, and
token budget management.

Later adapters (ComfyUI/LoRA, fal.ai, etc.) implement the same input/output contract but
different internal execution logic. This document covers only the Vertex adapter.

---

## 2. Adapter Position in VidPipe Pipeline

```
Screenwriter
    │
    │  Writes scene with @tag references embedded in prose/directions
    ▼
Storyboard Builder
    │
    │  Resolves @tags → reference images from AssetRegistry
    │  Constructs KeyframeGenerationRequest
    ▼
Adapter Router
    │
    │  Reads adapter_id from Manifest scene config
    │  Routes to: vertex_nano_banana | comfyui_flux_lora | ...
    ▼
Vertex Nano Banana Adapter  ◄── THIS SPEC
    │
    │  Builds Vertex API payload
    │  Calls Gemini image model
    │  Returns KeyframeGenerationResponse
    ▼
Asset Registry
    │
    │  Stores generated keyframe(s) with provenance metadata
    ▼
Next pipeline stage (video gen, continuity check, etc.)
```

---

## 3. @Tag System Contract

### 3.1 Tag Syntax in Screenwriter Output

The screenwriter embeds asset references using the `@` prefix. Tags are resolved against the
**AssetRegistry** before reaching the adapter.

```
@char:alex          Character reference (one or more images)
@char:morgan        Character reference
@set:coffee_shop    Location/environment reference
@prop:red_cup       Prop reference
@logo:brandmark     Logo/graphic reference
```

Tags may appear inline in scene directions or in a dedicated references block:

```yaml
# Example scene manifest fragment
scene_id: "ep01_sc04"
directions: >
  @char:alex and @char:morgan are seated at a table in @set:coffee_shop.
  @char:alex holds @prop:red_cup bearing @logo:brandmark, drinking.
  
references:
  - tag: "@char:alex"
    asset_id: "char_001"
  - tag: "@char:morgan"
    asset_id: "char_002"
  - tag: "@set:coffee_shop"
    asset_id: "set_004"
  - tag: "@prop:red_cup"
    asset_id: "prop_012"
  - tag: "@logo:brandmark"
    asset_id: "logo_003"
```

### 3.2 Asset Registry Entry Schema

Each asset in the registry has a typed entry that determines how the adapter treats it:

```typescript
interface AssetEntry {
  asset_id: string;
  tag: string;
  asset_type: "character" | "set" | "prop" | "logo" | "style";
  display_name: string;
  
  // One or more reference images for this asset
  // Multiple images per asset allowed (e.g. different angles of a character)
  reference_images: ReferenceImage[];
  
  // Adapter-specific hints — optional
  adapter_hints?: {
    vertex_nano_banana?: VertexAssetHints;
  };
}

interface ReferenceImage {
  image_id: string;
  file_path: string;        // local path or GCS URI
  mime_type: "image/png" | "image/jpeg" | "image/webp" | "image/heic" | "image/heif";
  role_hint?: string;       // optional: "face", "full_body", "outfit_detail", "angle_left", etc.
  file_size_bytes: number;
}

interface VertexAssetHints {
  // How strongly to emphasize identity preservation for this asset
  identity_weight: "strict" | "moderate" | "loose";
  // What aspects must be preserved
  preserve: string[];       // e.g. ["facial_features", "hair", "outfit", "logo_colors"]
  // Optional: where this asset appears relative to others in the scene
  spatial_hint?: string;    // e.g. "left foreground", "right background", "center"
}
```

---

## 4. KeyframeGenerationRequest Schema

This is the normalized object the Adapter Router passes to every adapter. All adapters receive
the same schema — the adapter is responsible for interpreting it according to its own API.

```typescript
interface KeyframeGenerationRequest {
  // Routing
  adapter_id: "vertex_nano_banana";
  model_variant: "nano_banana" | "nano_banana_2" | "nano_banana_pro";
  // Default: nano_banana_pro for keyframe generation quality

  // Scene context
  scene_id: string;
  shot_id: string;
  
  // The resolved assets — already loaded from AssetRegistry
  assets: ResolvedAsset[];
  
  // The image generation prompt written by screenwriter/director agent
  // This describes the SCENE — not the assets (those are handled by reference structure)
  generation_prompt: GenerationPrompt;
  
  // Output configuration
  output_config: OutputConfig;
}

interface ResolvedAsset {
  tag: string;              // "@char:alex"
  asset_type: "character" | "set" | "prop" | "logo" | "style";
  display_name: string;     // "Alex"
  reference_images: LoadedImage[];   // pre-loaded binary content
  adapter_hints?: VertexAssetHints;
  
  // Relationships declared by screenwriter — parsed from directions
  // e.g. alex is "holding" red_cup, red_cup "bears" brandmark
  relationships: AssetRelationship[];
}

interface LoadedImage {
  image_id: string;
  data: Buffer;             // raw image bytes
  mime_type: string;
  role_hint?: string;
  file_size_bytes: number;
}

interface AssetRelationship {
  subject_tag: string;      // "@char:alex"
  relationship: "holding" | "wearing" | "in" | "near" | "on" | "interacting_with";
  object_tag: string;       // "@prop:red_cup"
  detail?: string;          // optional: "drinking from", "logo facing camera"
}

interface GenerationPrompt {
  // The core scene description — what's happening, mood, composition
  scene_description: string;
  
  // Cinematography direction
  camera: {
    shot_type: string;      // "medium shot", "wide establishing", "close-up", etc.
    angle?: string;         // "eye level", "low angle", "bird's eye"
    lens?: string;          // "50mm", "wide angle", "telephoto"
  };
  
  // Lighting direction
  lighting?: string;        // "warm afternoon light from left", "dramatic side lighting"
  
  // Style/mood
  style?: string;           // "photorealistic", "cinematic", "documentary"
  
  // Negative guidance — what NOT to generate
  negative_guidance?: string[];
}

interface OutputConfig {
  aspect_ratio: string;           // "16:9", "2:3", "1:1", etc.
  resolution: "0.5K" | "1K" | "2K" | "4K";
  num_images: number;             // default: 1
  output_format: "png" | "jpeg" | "webp";
  return_data_uri: boolean;       // default: false
}
```

---

## 5. Reference Image Limits and Token Budget

The adapter must enforce Nano Banana's hard limits and manage the token budget before
constructing the API call.

### 5.1 Hard Limits (Vertex AI / Google AI)

| Constraint | Limit |
|---|---|
| Maximum reference images | 14 per request |
| Input tokens (Nano Banana 2 / Vertex) | 131,072 (use 65,536 as safe ceiling) |
| Input tokens (Nano Banana Pro) | 65,536 |
| Output tokens | 32,768 |
| Max image file size | 7MB per image (direct upload) / 50MB (GCS) |
| Supported MIME types | image/png, image/jpeg, image/webp, image/heic, image/heif |

### 5.2 Token Budget Calculation

```
Tokens per image (HIGH resolution setting): 1,120
Tokens per image (MEDIUM): ~560
Tokens per image (LOW): ~258

Budget formula:
  safe_input_budget = 65,536
  text_prompt_tokens = estimate(prompt_text)  # ~500-1000 for a rich scene prompt
  available_for_images = safe_input_budget - text_prompt_tokens
  max_images_at_HIGH = floor(available_for_images / 1,120)
```

At HIGH resolution with a 1,000-token prompt: floor(64,536 / 1,120) = **57 images** — well above
the 14-image hard limit. Token budget is therefore NOT the binding constraint for normal scenes.
The 14-image hard limit is always the binding constraint.

### 5.3 Image Selection Strategy When Asset Count Exceeds 14

When the total number of reference images across all assets exceeds 14, the adapter applies
the following priority and reduction rules:

```
Priority order for image slot allocation:
  1. Characters (identity-critical) — max 2 images per character
  2. Props with logos/graphics applied — 1 image per prop
  3. Set/environment — 1-2 images
  4. Logos standalone — 1 image
  5. Additional character angles — use remaining slots

Reduction rules (applied in order until total ≤ 14):
  1. Drop role_hint="angle_*" images beyond the first per character
  2. Drop style reference images (least critical)
  3. Reduce character refs to 1 per character
  4. If still >14: log WARNING and prioritize by asset_type weight
```

The adapter MUST log a `REFERENCE_TRUNCATION` warning with the full list of dropped images
when truncation occurs, so the director agent or user can adjust.

---

## 6. Prompt Construction Strategy

This is the core logic of the adapter. The prompt passed to Vertex must perform two jobs:

1. **Role declarations** — tell the model what each image is for
2. **Scene direction** — describe what to generate

### 6.1 Prompt Assembly Order

```
[ROLE DECLARATIONS — one block per asset, images ordered first]
[RELATIONSHIP DECLARATIONS — nested references between assets]  
[SCENE DIRECTION — what to generate]
[IDENTITY PRESERVATION RULES — explicit constraints]
[CINEMATOGRAPHY — shot, lighting, style]
```

### 6.2 Image Ordering in the Parts Array

The Vertex API receives a flat array of `parts`. The adapter constructs this array in a
deliberate order that maps to the role declarations in the text prompt:

```
parts = [
  # Assets in priority order: characters → set → props → logos
  
  # Character 1 (all reference images grouped together)
  { inlineData: { mimeType: ..., data: char1_image1 } },
  { inlineData: { mimeType: ..., data: char1_image2 } },  # if multiple refs
  
  # Character 2
  { inlineData: { mimeType: ..., data: char2_image1 } },
  
  # Set/environment
  { inlineData: { mimeType: ..., data: set_image1 } },
  
  # Props
  { inlineData: { mimeType: ..., data: prop_image1 } },
  
  # Logos
  { inlineData: { mimeType: ..., data: logo_image1 } },
  
  # Text prompt (ALWAYS LAST)
  { text: <assembled_prompt_string> }
]
```

The text prompt explicitly numbers each image sequentially, matching the array order:

```
Image 1: Character reference (Alex) — face and outfit
Image 2: Character reference (Alex) — full body angle
Image 3: Character reference (Morgan)
Image 4: Environment reference (coffee shop)
Image 5: Prop reference (red cup)
Image 6: Logo reference (brandmark)
```

### 6.3 Prompt Template

The adapter uses the following template, populated programmatically:

```python
PROMPT_TEMPLATE = """
REFERENCE IMAGES — read all role assignments before generating:

{role_declarations}

{relationship_declarations}

SCENE DIRECTION:
{scene_description}

IDENTITY PRESERVATION RULES:
{preservation_rules}

CINEMATOGRAPHY:
Shot: {shot_type}{angle_clause}{lens_clause}
Lighting: {lighting}
Style: {style}
{negative_clause}
""".strip()
```

### 6.4 Role Declaration Generation

Each asset generates a role declaration block. The format varies by asset type:

```python
def build_role_declaration(
    asset: ResolvedAsset,
    image_indices: list[int],   # the 1-based indices of this asset's images in the parts array
) -> str:
    
    if asset.asset_type == "character":
        index_str = format_indices(image_indices)
        hints = asset.adapter_hints.get("vertex_nano_banana", {})
        preserve = hints.get("preserve", ["facial features", "hair", "outfit"])
        spatial = hints.get("spatial_hint", "")
        spatial_clause = f" They appear {spatial}." if spatial else ""
        
        return f"""
{index_str} — CHARACTER: {asset.display_name.upper()} (SUBJECT)
  This character must appear in the scene.{spatial_clause}
  Preserve exactly: {", ".join(preserve)}.
  Do not blend or merge any features with other characters.
""".strip()

    elif asset.asset_type == "set":
        index_str = format_indices(image_indices)
        return f"""
{index_str} — ENVIRONMENT REFERENCE: {asset.display_name.upper()}
  This is the location and setting for the scene.
  Match: architecture, ambient lighting direction, color temperature, depth of field.
  Do not introduce background elements not present in this reference.
""".strip()

    elif asset.asset_type == "prop":
        index_str = format_indices(image_indices)
        return f"""
{index_str} — PROP REFERENCE: {asset.display_name.upper()}
  This object must appear in the scene.
  Match: shape, material finish, color, and distinctive physical details exactly.
""".strip()

    elif asset.asset_type == "logo":
        index_str = format_indices(image_indices)
        return f"""
{index_str} — LOGO/GRAPHIC REFERENCE: {asset.display_name.upper()}
  This graphic must be applied to its target surface in the scene.
  Render it: sharp, correctly colored, legible, and faithful to this reference.
  Scale and orientation should match natural placement on the target object.
""".strip()


def format_indices(indices: list[int]) -> str:
    if len(indices) == 1:
        return f"Image {indices[0]}"
    elif len(indices) == 2:
        return f"Images {indices[0]} and {indices[1]}"
    else:
        return f"Images {', '.join(str(i) for i in indices[:-1])}, and {indices[-1]}"
```

### 6.5 Relationship Declaration Generation

Relationships are declared explicitly after all role blocks. This handles nested references
(e.g. logo on prop held by character):

```python
def build_relationship_declarations(assets: list[ResolvedAsset]) -> str:
    all_relationships = []
    for asset in assets:
        all_relationships.extend(asset.relationships)
    
    if not all_relationships:
        return ""
    
    lines = ["ASSET RELATIONSHIPS — spatial and physical connections:"]
    for rel in all_relationships:
        subject_name = resolve_display_name(rel.subject_tag, assets)
        object_name = resolve_display_name(rel.object_tag, assets)
        detail = f" ({rel.detail})" if rel.detail else ""
        lines.append(f"  - {subject_name} is {rel.relationship} {object_name}{detail}.")
    
    return "\n".join(lines)
```

For the scenario described (Alex holding red cup with brandmark logo), this produces:

```
ASSET RELATIONSHIPS — spatial and physical connections:
  - Alex is holding Red Cup (drinking from it).
  - Brandmark Logo is on Red Cup (printed on front face, logo facing camera).
```

### 6.6 Identity Preservation Rules Generation

```python
def build_preservation_rules(assets: list[ResolvedAsset]) -> str:
    rules = []
    
    characters = [a for a in assets if a.asset_type == "character"]
    props = [a for a in assets if a.asset_type == "prop"]
    logos = [a for a in assets if a.asset_type == "logo"]
    
    for char in characters:
        hints = char.adapter_hints.get("vertex_nano_banana", {}) if char.adapter_hints else {}
        weight = hints.get("identity_weight", "strict")
        preserve = hints.get("preserve", ["facial features", "hair color and style", "outfit"])
        
        if weight == "strict":
            rules.append(
                f"- {char.display_name}: face must be pixel-faithful to reference. "
                f"Preserve exactly: {', '.join(preserve)}. "
                f"No feature blending with other characters."
            )
        elif weight == "moderate":
            rules.append(
                f"- {char.display_name}: maintain recognizable likeness. "
                f"Preserve: {', '.join(preserve)}."
            )
    
    for prop in props:
        rules.append(
            f"- {prop.display_name}: shape, material, and finish must match reference exactly."
        )
    
    for logo in logos:
        rules.append(
            f"- {logo.display_name}: must be legible, correctly colored, "
            f"and faithfully match the reference graphic."
        )
    
    return "\n".join(rules) if rules else "Maintain visual consistency with all references."
```

---

## 7. Vertex API Call Construction

### 7.1 Model Variant Mapping

```python
MODEL_MAP = {
    "nano_banana":     "gemini-2.5-flash-image",
    "nano_banana_2":   "gemini-3.1-flash-image-preview",
    "nano_banana_pro": "gemini-3-pro-image-preview",
}
```

### 7.2 Full Adapter Implementation

```python
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from google.cloud import aiplatform

class VertexNanaBananaAdapter:
    
    ADAPTER_ID = "vertex_nano_banana"
    
    MODEL_MAP = {
        "nano_banana":     "gemini-2.5-flash-image",
        "nano_banana_2":   "gemini-3.1-flash-image-preview",
        "nano_banana_pro": "gemini-3-pro-image-preview",
    }
    
    MAX_REFERENCE_IMAGES = 14
    SAFE_INPUT_TOKEN_BUDGET = 65_536
    TOKENS_PER_IMAGE_HIGH = 1_120
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        vertexai.init(project=project_id, location=location)
    
    # ─────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────
    
    def generate(self, request: KeyframeGenerationRequest) -> KeyframeGenerationResponse:
        
        # 1. Validate and select images
        selected_assets = self._select_images(request.assets)
        
        # 2. Build image index map (asset_tag → list of 1-based part indices)
        image_index_map = self._build_image_index_map(selected_assets)
        
        # 3. Assemble prompt text
        prompt_text = self._build_prompt(
            assets=selected_assets,
            image_index_map=image_index_map,
            generation_prompt=request.generation_prompt
        )
        
        # 4. Build parts array
        parts = self._build_parts(selected_assets, prompt_text)
        
        # 5. Build generation config
        gen_config = self._build_generation_config(request.output_config)
        
        # 6. Call Vertex
        model_id = self.MODEL_MAP[request.model_variant]
        model = GenerativeModel(model_id)
        
        response = model.generate_content(
            contents=parts,
            generation_config=gen_config
        )
        
        # 7. Parse and return
        return self._parse_response(response, request)
    
    # ─────────────────────────────────────────────
    # Image selection
    # ─────────────────────────────────────────────
    
    def _select_images(self, assets: list[ResolvedAsset]) -> list[ResolvedAsset]:
        """
        Apply slot allocation and reduction rules.
        Returns a new list of assets with reference_images trimmed to fit
        within the 14-image hard limit.
        """
        # Count total images
        total = sum(len(a.reference_images) for a in assets)
        
        if total <= self.MAX_REFERENCE_IMAGES:
            return assets  # no reduction needed
        
        # Apply reduction rules
        working = [self._clone_asset(a) for a in assets]
        
        # Rule 1: Cap characters at 2 images each
        for asset in working:
            if asset.asset_type == "character" and len(asset.reference_images) > 2:
                dropped = asset.reference_images[2:]
                asset.reference_images = asset.reference_images[:2]
                self._log_truncation(asset.tag, dropped, "rule1_char_cap_2")
        
        if self._total_images(working) <= self.MAX_REFERENCE_IMAGES:
            return working
        
        # Rule 2: Drop style/angle supplementary images
        for asset in working:
            supplementary = [
                img for img in asset.reference_images
                if img.role_hint and img.role_hint.startswith("angle_")
            ]
            if supplementary:
                asset.reference_images = [
                    img for img in asset.reference_images
                    if not (img.role_hint and img.role_hint.startswith("angle_"))
                ]
                self._log_truncation(asset.tag, supplementary, "rule2_drop_angles")
        
        if self._total_images(working) <= self.MAX_REFERENCE_IMAGES:
            return working
        
        # Rule 3: Cap characters at 1 image each
        for asset in working:
            if asset.asset_type == "character" and len(asset.reference_images) > 1:
                dropped = asset.reference_images[1:]
                asset.reference_images = asset.reference_images[:1]
                self._log_truncation(asset.tag, dropped, "rule3_char_cap_1")
        
        if self._total_images(working) <= self.MAX_REFERENCE_IMAGES:
            return working
        
        # Rule 4: Priority culling — remove lowest priority assets entirely
        priority_order = ["logo", "style", "set", "prop", "character"]
        for asset_type in priority_order:
            if self._total_images(working) <= self.MAX_REFERENCE_IMAGES:
                break
            for asset in working:
                if asset.asset_type == asset_type and self._total_images(working) > self.MAX_REFERENCE_IMAGES:
                    if len(asset.reference_images) > 0:
                        dropped = [asset.reference_images.pop()]
                        self._log_truncation(asset.tag, dropped, "rule4_priority_cull")
        
        final_total = self._total_images(working)
        if final_total > self.MAX_REFERENCE_IMAGES:
            raise AdapterError(
                f"REFERENCE_LIMIT_EXCEEDED: Could not reduce to ≤14 images. "
                f"Final count: {final_total}. Review asset count for scene."
            )
        
        return working
    
    # ─────────────────────────────────────────────
    # Image index mapping
    # ─────────────────────────────────────────────
    
    def _build_image_index_map(
        self, assets: list[ResolvedAsset]
    ) -> dict[str, list[int]]:
        """
        Returns: { "@char:alex": [1, 2], "@char:morgan": [3], "@set:coffee_shop": [4], ... }
        Image indices are 1-based, matching the text prompt references.
        Images are ordered: characters → sets → props → logos.
        """
        ordered_assets = (
            [a for a in assets if a.asset_type == "character"] +
            [a for a in assets if a.asset_type == "set"] +
            [a for a in assets if a.asset_type == "prop"] +
            [a for a in assets if a.asset_type == "logo"] +
            [a for a in assets if a.asset_type == "style"]
        )
        
        index_map = {}
        counter = 1
        for asset in ordered_assets:
            indices = []
            for _ in asset.reference_images:
                indices.append(counter)
                counter += 1
            index_map[asset.tag] = indices
        
        return index_map
    
    # ─────────────────────────────────────────────
    # Prompt construction
    # ─────────────────────────────────────────────
    
    def _build_prompt(
        self,
        assets: list[ResolvedAsset],
        image_index_map: dict[str, list[int]],
        generation_prompt: GenerationPrompt
    ) -> str:
        
        sections = ["REFERENCE IMAGES — read all role assignments before generating:\n"]
        
        # Role declarations — one block per asset
        ordered_assets = (
            [a for a in assets if a.asset_type == "character"] +
            [a for a in assets if a.asset_type == "set"] +
            [a for a in assets if a.asset_type == "prop"] +
            [a for a in assets if a.asset_type == "logo"]
        )
        
        for asset in ordered_assets:
            indices = image_index_map[asset.tag]
            declaration = build_role_declaration(asset, indices)
            sections.append(declaration + "\n")
        
        # Relationship declarations
        rel_block = build_relationship_declarations(assets)
        if rel_block:
            sections.append(rel_block + "\n")
        
        # Scene direction
        sections.append("SCENE DIRECTION:")
        sections.append(generation_prompt.scene_description + "\n")
        
        # Identity rules
        rules = build_preservation_rules(assets)
        sections.append("IDENTITY PRESERVATION RULES:")
        sections.append(rules + "\n")
        
        # Cinematography
        camera = generation_prompt.camera
        angle_clause = f", {camera.angle}" if camera.angle else ""
        lens_clause = f", {camera.lens}" if camera.lens else ""
        sections.append("CINEMATOGRAPHY:")
        sections.append(f"Shot: {camera.shot_type}{angle_clause}{lens_clause}")
        
        if generation_prompt.lighting:
            sections.append(f"Lighting: {generation_prompt.lighting}")
        
        if generation_prompt.style:
            sections.append(f"Style: {generation_prompt.style}")
        
        if generation_prompt.negative_guidance:
            neg = ", ".join(generation_prompt.negative_guidance)
            sections.append(f"Avoid: {neg}")
        
        return "\n".join(sections)
    
    # ─────────────────────────────────────────────
    # Parts array construction
    # ─────────────────────────────────────────────
    
    def _build_parts(
        self, assets: list[ResolvedAsset], prompt_text: str
    ) -> list[Part]:
        
        ordered_assets = (
            [a for a in assets if a.asset_type == "character"] +
            [a for a in assets if a.asset_type == "set"] +
            [a for a in assets if a.asset_type == "prop"] +
            [a for a in assets if a.asset_type == "logo"]
        )
        
        parts = []
        
        # Image parts first, in declared order
        for asset in ordered_assets:
            for img in asset.reference_images:
                parts.append(
                    Part.from_data(
                        data=img.data,
                        mime_type=img.mime_type
                    )
                )
        
        # Text prompt always last
        parts.append(Part.from_text(prompt_text))
        
        return parts
    
    # ─────────────────────────────────────────────
    # Generation config
    # ─────────────────────────────────────────────
    
    def _build_generation_config(self, output_config: OutputConfig) -> GenerationConfig:
        return GenerationConfig(
            response_modalities=["IMAGE", "TEXT"],
            temperature=1.0,          # Google's recommended value for image generation
            # Aspect ratio and resolution are passed as generation params
            # when supported by the model variant — see Vertex API docs
        )
    
    # ─────────────────────────────────────────────
    # Response parsing
    # ─────────────────────────────────────────────
    
    def _parse_response(
        self, response, request: KeyframeGenerationRequest
    ) -> "KeyframeGenerationResponse":
        
        images = []
        model_commentary = []
        
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    images.append(GeneratedImage(
                        data=base64.b64decode(part.inline_data.data),
                        mime_type=part.inline_data.mime_type,
                    ))
                elif hasattr(part, "text") and part.text:
                    model_commentary.append(part.text)
        
        return KeyframeGenerationResponse(
            adapter_id=self.ADAPTER_ID,
            scene_id=request.scene_id,
            shot_id=request.shot_id,
            images=images,
            model_commentary=model_commentary,
            model_variant=request.model_variant,
            reference_image_count=sum(
                len(a.reference_images) for a in request.assets
            ),
        )
```

---

## 8. KeyframeGenerationResponse Schema

```typescript
interface KeyframeGenerationResponse {
  adapter_id: string;
  scene_id: string;
  shot_id: string;
  
  // Generated images
  images: GeneratedImage[];
  
  // Optional text the model returned (commentary, warnings)
  model_commentary: string[];
  
  // Provenance
  model_variant: string;
  reference_image_count: number;
  
  // Set by adapter if truncation occurred
  truncation_warnings?: TruncationWarning[];
  
  // Full prompt as sent (for debugging/logging)
  prompt_sent?: string;
}

interface GeneratedImage {
  data: Buffer;               // raw image bytes
  mime_type: string;
  width?: number;
  height?: number;
}

interface TruncationWarning {
  asset_tag: string;
  dropped_image_ids: string[];
  rule_applied: string;
}
```

---

## 9. End-to-End Example

Given the scenario: Alex and Morgan at a coffee shop, Alex drinking from a branded cup.

**Input assets (6 total images):**
```
@char:alex    → 2 reference images (face_close.png, full_body.png)
@char:morgan  → 1 reference image  (full_body.png)
@set:coffee   → 1 reference image  (interior_wide.jpg)
@prop:cup     → 1 reference image  (cup_side.png)
@logo:brand   → 1 reference image  (logo_flat.png)
```

**Constructed parts array (7 parts):**
```
Part 1: image/png  (alex face_close)
Part 2: image/png  (alex full_body)
Part 3: image/png  (morgan full_body)
Part 4: image/jpeg (coffee shop interior)
Part 5: image/png  (cup)
Part 6: image/png  (logo)
Part 7: text       (assembled prompt)
```

**Assembled prompt text:**
```
REFERENCE IMAGES — read all role assignments before generating:

Images 1 and 2 — CHARACTER: ALEX (SUBJECT)
  This character must appear in the scene. They appear left of center.
  Preserve exactly: facial features, hair color and style, outfit.
  Do not blend or merge any features with other characters.

Image 3 — CHARACTER: MORGAN (SUBJECT)
  This character must appear in the scene. They appear right of center.
  Preserve exactly: facial features, hair color and style, outfit.
  Do not blend or merge any features with other characters.

Image 4 — ENVIRONMENT REFERENCE: COFFEE SHOP
  This is the location and setting for the scene.
  Match: architecture, ambient lighting direction, color temperature, depth of field.
  Do not introduce background elements not present in this reference.

Image 5 — PROP REFERENCE: RED CUP
  This object must appear in the scene.
  Match: shape, material finish, color, and distinctive physical details exactly.

Image 6 — LOGO/GRAPHIC REFERENCE: BRANDMARK
  This graphic must be applied to its target surface in the scene.
  Render it: sharp, correctly colored, legible, and faithful to this reference.
  Scale and orientation should match natural placement on the target object.

ASSET RELATIONSHIPS — spatial and physical connections:
  - Alex is holding Red Cup (drinking from it).
  - Brandmark Logo is on Red Cup (printed on front face, logo facing camera).

SCENE DIRECTION:
  Alex and Morgan are seated at a small table in the coffee shop, engaged
  in relaxed conversation. Alex is mid-sip, cup raised. Morgan is leaning
  slightly forward, attentive. Morning atmosphere, relaxed and warm.

IDENTITY PRESERVATION RULES:
- Alex: face must be pixel-faithful to reference. Preserve exactly: facial
  features, hair color and style, outfit. No feature blending with other characters.
- Morgan: face must be pixel-faithful to reference. Preserve exactly: facial
  features, hair color and style, outfit. No feature blending with other characters.
- Red Cup: shape, material, and finish must match reference exactly.
- Brandmark: must be legible, correctly colored, and faithfully match the
  reference graphic.

CINEMATOGRAPHY:
Shot: medium shot, eye level
Lighting: warm morning light from window left, soft fill
Style: photorealistic, cinematic
Avoid: blurry faces, merged characters, missing logo, wrong outfits
```

---

## 10. Error Handling

| Error Code | Condition | Behavior |
|---|---|---|
| `REFERENCE_LIMIT_EXCEEDED` | Cannot reduce to ≤14 after all rules | Raise, do not call API |
| `MIME_TYPE_UNSUPPORTED` | Image file is not png/jpeg/webp/heic/heif | Raise with asset_id |
| `IMAGE_TOO_LARGE` | Image >7MB (direct) or >50MB (GCS) | Resize/compress before calling, log warning |
| `MODEL_REFUSAL` | Vertex returns safety block | Return response with `blocked: true`, log full prompt |
| `EMPTY_IMAGE_RESPONSE` | Response contains no image parts | Retry once, then raise |
| `TOKEN_BUDGET_EXCEEDED` | Prompt + images estimated >safe budget | Switch to MEDIUM resolution, retry |

---

## 11. Adapter Interface Contract

All VidPipe adapters implement this interface, enabling the Adapter Router to call any
adapter identically:

```python
class ImageGenAdapter(ABC):
    
    ADAPTER_ID: str  # class-level constant
    
    @abstractmethod
    def generate(
        self, request: KeyframeGenerationRequest
    ) -> KeyframeGenerationResponse:
        """Main generation entrypoint."""
        ...
    
    @abstractmethod
    def validate_request(
        self, request: KeyframeGenerationRequest
    ) -> list[str]:
        """
        Returns list of validation errors (empty = valid).
        Called by AdapterRouter before routing.
        """
        ...
    
    @abstractmethod
    def estimate_cost(
        self, request: KeyframeGenerationRequest
    ) -> CostEstimate:
        """
        Returns estimated cost/credits for this request.
        Used by pipeline budgeting layer.
        """
        ...
```

---

## 12. Configuration

```yaml
# vidpipe_config.yaml — adapter section

adapters:
  vertex_nano_banana:
    enabled: true
    project_id: "${GCP_PROJECT_ID}"
    location: "us-central1"
    
    defaults:
      model_variant: "nano_banana_pro"
      resolution: "1K"
      aspect_ratio: "16:9"
      output_format: "png"
      num_images: 1
      temperature: 1.0
    
    limits:
      max_reference_images: 14
      safe_token_budget: 65536
      tokens_per_image_high: 1120
      max_image_file_size_mb: 7
    
    image_reduction:
      strategy: "priority_cull"   # priority_cull | uniform_reduce | fail_fast
      char_max_refs: 2
      set_max_refs: 2
      prop_max_refs: 1
      logo_max_refs: 1
    
    logging:
      log_prompts: true           # log full assembled prompt for debugging
      log_truncations: true
      log_token_estimates: true
```

---

## 13. Future Adapter Compatibility Notes

When the ComfyUI/LoRA adapter is implemented, it will receive the same
`KeyframeGenerationRequest` but will interpret `assets` differently:

- `reference_images` → used as training data pointers or IP-Adapter references
- `adapter_hints.comfyui` → LoRA paths, trigger words, strength values
- The prompt construction will follow ComfyUI's token/trigger word pattern
  rather than the Nano Banana role-declaration pattern

The `KeyframeGenerationResponse` contract is identical across all adapters,
ensuring downstream pipeline stages (asset storage, continuity check, video gen)
are fully adapter-agnostic.