# Phase 8: Veo Reference Passthrough and Clean Sheets - Research

**Researched:** 2026-02-16
**Domain:** Veo 3.1 reference image API integration, reference selection logic, clean sheet preprocessing
**Confidence:** HIGH

## Summary

Phase 8 implements the critical bridge between the Asset Registry (Phase 5-7) and Veo 3.1 video generation by passing up to 3 reference images per scene to preserve character/object identity across clips. This phase addresses the "3-reference constraint" — Veo 3.1 accepts maximum 3 asset references with `referenceType: "asset"` — requiring strategic selection logic based on scene type and manifest placements.

The research confirms Veo 3.1's reference image capabilities are production-ready (as of early 2026) with documented API structure. The hybrid approach combining first-frame daisy-chaining (`image` parameter) with 3 reference images (`referenceImages` parameter) is officially supported. Clean sheet preprocessing is optional but valuable: background removal via rembg is free (local inference), while full Gemini Image clean sheet generation costs ~$0.02-0.05 per asset. Face similarity validation uses ArcFace embeddings with cosine distance (threshold ~0.6 for same identity).

Critical constraint discovered: When using reference images, Veo 3.1 **mandates 8-second clip duration** — no 4s or 6s options. This impacts pipeline configuration and must be enforced in Phase 8.

**Primary recommendation:** Implement reference selection logic first (uses existing Asset Registry data), add hybrid passthrough to Veo API, then optionally layer clean sheet preprocessing as quality enhancement. Store selected references in `scene_manifests.selected_reference_tags` for debugging and UI display.

## Standard Stack

### Core Dependencies (Already in Project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| google-genai | ^0.12.0+ | Veo 3.1 API client with reference image support | Official Google SDK, supports `referenceImages` parameter in `GenerateVideosConfig` |
| SQLAlchemy | 2.0 | Database ORM for reference tracking tables | Already used throughout project |
| Pillow | Latest | Image loading and manipulation for reference processing | Standard Python imaging library |

### New Optional Dependencies

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rembg | ^2.0.60 (Jan 2026) | Background removal for Tier 2 clean sheets | Optional quality enhancement, free local inference |
| insightface | 0.7+ (ArcFace) | Face similarity scoring for clean sheet validation | Required for Tier 3/4 clean sheets, validates generated sheets preserve identity |

### Veo 3.1 API Constraints (February 2026)

| Feature | Constraint | Impact |
|---------|-----------|--------|
| **Reference images** | Maximum 3 per generation | Must implement selection logic |
| **Reference type** | Only `"asset"` supported on Veo 3.1 | `"style"` reference only on Veo 2 (deprecated) |
| **Duration with references** | **Mandatory 8 seconds** | Cannot use 4s or 6s when references attached |
| **Format** | base64 or GCS URI | GCS URIs preferred for production (no size limits) |
| **Hybrid mode** | `image` + `referenceImages` both allowed | First frame for spatial + refs for identity |

**Source:** [Generate Veo videos from reference images - Vertex AI docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/use-reference-images-to-guide-video-generation)

## Architecture Patterns

### Reference Selection Strategy (3-Slot Allocation)

From v2-pipe-optimization.md Decision 1 (lines 1561-1572), the selection strategy adapts to scene type:

| Scene Type | Slot 1 | Slot 2 | Slot 3 |
|-----------|--------|--------|--------|
| **Character close-up** | Primary character face crop | Character full-body crop | Environment reference |
| **Two-shot dialogue** | Character A face | Character B face | Environment |
| **Establishing shot** | Environment wide | Key prop/vehicle | Style reference or secondary environment |
| **Action scene** | Primary character | Secondary character or vehicle | Environment context |
| **Object focus** | Primary object (multiple angles if available) | Environment | Character in context |

**Selection logic:**

```python
def select_references(
    scene_manifest: SceneManifestSchema,
    all_assets: list[Asset]
) -> list[Asset]:
    """Select up to 3 best references for this scene.

    Priority system:
    1. Subject role assets (main focus)
    2. Interaction_target assets (plot-critical props)
    3. Background role assets (environment, secondary characters)
    4. Quality score tiebreaker (higher = better)
    5. Face crop preference for characters (better identity preservation)
    """
    from vidpipe.schemas.storyboard_enhanced import AssetPlacement

    # Map placements to asset objects
    asset_map = {a.manifest_tag: a for a in all_assets}
    placements = scene_manifest.placements

    # Categorize by role
    subject_refs = [asset_map[p.asset_tag] for p in placements
                    if p.role == "subject" and p.asset_tag in asset_map]
    interaction_refs = [asset_map[p.asset_tag] for p in placements
                        if p.role == "interaction_target" and p.asset_tag in asset_map]
    background_refs = [asset_map[p.asset_tag] for p in placements
                       if p.role == "background" and p.asset_tag in asset_map]

    # Build selection pool
    pool = []
    pool.extend(subject_refs)      # Subjects first
    pool.extend(interaction_refs)  # Then plot-critical items
    pool.extend(background_refs)   # Then background elements

    # Deduplicate (same asset may appear multiple times)
    seen_tags = set()
    unique_pool = []
    for asset in pool:
        if asset.manifest_tag not in seen_tags:
            unique_pool.append(asset)
            seen_tags.add(asset.manifest_tag)

    # Sort by quality within role groups
    unique_pool.sort(
        key=lambda a: (a.quality_score or 0.0),
        reverse=True
    )

    # Take top 3
    selected = unique_pool[:3]

    return selected
```

**Special case: Character with multiple crops**

If CHAR_01 has both face crop and full-body crop (from multi-detection in Phase 5), prefer:
- Face crop for close-ups / medium shots (better identity detail)
- Full-body crop for wide shots (better composition context)

Query via `Asset.source_asset_id` to find child crops of a primary asset.

### Hybrid First-Frame + Reference Images Pattern

From v2-pipe-optimization.md Decision 2 (lines 1574-1580):

```python
# video_gen.py enhancement
video_config = types.GenerateVideosConfig(
    aspect_ratio=project.aspect_ratio,
    duration_seconds=8,  # MANDATORY when using reference images

    # First frame: spatial continuity from daisy-chain
    last_frame=types.Image(image_bytes=end_frame_bytes, mime_type="image/png"),

    # Reference images: identity preservation (max 3)
    reference_images=[
        types.ReferenceImage(
            reference_image=types.Image(
                image_bytes=ref_bytes,  # or gcs_uri=ref.reference_image_url
                mime_type="image/png"
            ),
            reference_type="asset"  # Only valid type for Veo 3.1
        )
        for ref in selected_references[:3]
    ],

    negative_prompt="...",
    generate_audio=bool(project.audio_enabled),
    seed=project.seed,
)

# Pass first frame separately (not in reference_images)
response = await client.aio.models.generate_videos(
    model=video_model,
    prompt=video_prompt,
    image=types.Image(image_bytes=start_frame_bytes, mime_type="image/png"),
    config=video_config,
)
```

**Critical:** `duration_seconds` MUST be 8 when `reference_images` is non-empty. From search results: "Base clips are 4, 6, or 8 seconds in the Gemini API (8 s with reference images)." This constraint must be enforced in pipeline configuration.

### Clean Sheet Generation Tiers

From v2-manifest.md lines 333-353, four quality tiers:

#### Tier 1: No Preprocessing (Default)
- **Cost:** $0
- **Method:** Use raw crop from Asset Registry as-is
- **When:** Asset quality_score ≥ 7.0, clean background already
- **Output:** Original `Asset.reference_image_url`

#### Tier 2: Background Removal (Fast, Local)
- **Cost:** $0 (local rembg inference)
- **Method:** rembg removes background, replace with neutral gray (#808080)
- **When:** Asset has busy/distracting background, quality_score 5.0-7.0
- **Output:** New image saved to `tmp/manifests/{manifest_id}/clean_sheets/tier2_{asset_id}.png`

```python
# Tier 2 implementation
from rembg import remove
from PIL import Image
import io

def generate_tier2_clean_sheet(asset: Asset) -> bytes:
    """Remove background, replace with neutral gray."""
    # Load original
    original_bytes = Path(asset.reference_image_url).read_bytes()

    # Remove background (rembg outputs PNG with alpha)
    no_bg_bytes = remove(original_bytes, force_return_bytes=True)

    # Convert to RGB with gray background
    img = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
    background = Image.new("RGB", img.size, (128, 128, 128))
    background.paste(img, mask=img.split()[3])  # Alpha channel as mask

    # Save to bytes
    output = io.BytesIO()
    background.save(output, format="PNG")
    return output.getvalue()
```

**Source:** [rembg GitHub](https://github.com/danielgatis/rembg) - Latest release Jan 2026, requires Python ≥3.11

#### Tier 3: Full Clean Sheet via Gemini Image (Best Quality)
- **Cost:** ~$0.02-0.05 per asset (Imagen 3 or Gemini 3 Pro Image)
- **Method:** Feed original crop + enhanced prompt to Gemini Image, generate idealized version
- **When:** Key characters (quality_score < 7.0), or user-requested upgrade
- **Validation:** Face similarity via ArcFace cosine distance ≥ 0.6
- **Output:** New image + similarity score stored in `asset_clean_references` table

```python
# Tier 3 implementation
async def generate_tier3_clean_sheet(
    asset: Asset,
    client,
    image_model: str
) -> tuple[bytes, float]:
    """Generate clean sheet via Gemini Image with face validation.

    Returns:
        (clean_image_bytes, face_similarity_score)
    """
    from google.genai import types

    # Build enhanced prompt
    conditioning_prompt = (
        f"{asset.reverse_prompt}\n\n"
        f"IMPORTANT: Generate this subject on a clean, neutral gray background "
        f"(#808080). Well-lit studio lighting with even, soft illumination. "
        f"Subject facing slightly toward camera, no occlusion. Full head and "
        f"shoulders visible. Preserve ALL distinguishing features exactly. "
        f"Photorealistic quality."
    )

    # Load reference image
    ref_bytes = Path(asset.reference_image_url).read_bytes()

    # Generate clean version (image-conditioned generation)
    response = await client.aio.models.generate_images(
        model=image_model,
        prompt=conditioning_prompt,
        reference_images=[
            types.Image(image_bytes=ref_bytes, mime_type="image/png")
        ],
        config=types.GenerateImagesConfig(
            number_of_images=1,
            include_rai_reason=True,
        )
    )

    clean_bytes = response.generated_images[0].image.image_bytes

    # Validate face similarity (if character)
    if asset.asset_type == "CHARACTER" and asset.face_embedding:
        similarity = compute_face_similarity(ref_bytes, clean_bytes, asset.face_embedding)
        if similarity < 0.6:
            logger.warning(f"Clean sheet similarity low ({similarity:.2f}) for {asset.manifest_tag}")
    else:
        similarity = None

    return clean_bytes, similarity
```

**Source:** [Gemini 3 Pro Image (Nano Banana Pro) docs](https://ai.google.dev/gemini-api/docs/image-generation)

#### Tier 4: Multi-Angle Sheet (Premium)
- **Cost:** ~$0.06-0.15 (3 generations)
- **Method:** Generate 3 angles: frontal, 3/4 left, 3/4 right
- **When:** Hero characters needing maximum flexibility
- **Output:** 3 clean references per asset, selection logic picks best per scene

**Not implemented in Phase 8 — deferred to future enhancement.**

### Face Similarity Validation with ArcFace

```python
def compute_face_similarity(
    ref_bytes: bytes,
    clean_bytes: bytes,
    stored_embedding: bytes
) -> float:
    """Compute cosine similarity between reference and clean sheet faces.

    Args:
        ref_bytes: Original reference image
        clean_bytes: Generated clean sheet
        stored_embedding: Pre-computed ArcFace embedding from Phase 5 (512-dim float32)

    Returns:
        Cosine similarity score (0.0 to 1.0)
        - ≥0.6: Same person (acceptable)
        - <0.6: Different identity (reject)
    """
    import numpy as np
    from insightface.app import FaceAnalysis

    # Lazy-load ArcFace model
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Extract embedding from clean sheet
    img = cv2.imdecode(np.frombuffer(clean_bytes, np.uint8), cv2.IMREAD_COLOR)
    faces = app.get(img)
    if not faces:
        return 0.0  # No face detected

    clean_embedding = faces[0].embedding  # 512-dim

    # Load stored embedding
    ref_embedding = np.frombuffer(stored_embedding, dtype=np.float32)

    # Cosine similarity
    similarity = np.dot(ref_embedding, clean_embedding) / (
        np.linalg.norm(ref_embedding) * np.linalg.norm(clean_embedding)
    )

    return float(similarity)
```

**Source:** [ArcFace face similarity with embeddings](https://medium.com/@ichigo.v.gen12/arcface-architecture-and-practical-example-how-to-calculate-the-face-similarity-between-images-183896a35957) - Cosine distance metric standard for face recognition

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Background removal | Custom segmentation models, OpenCV masking | rembg library | State-of-the-art U-Net/BiRefNet models, free local inference, supports GPU acceleration, 3-year production track record |
| Reference image format conversion | Manual base64 encoding, custom HTTP upload | google-genai SDK with GCS URIs | SDK handles upload, retry, format negotiation. GCS URIs avoid base64 size bloat (>30% overhead) |
| Face similarity scoring | Custom CNN training, feature extraction | ArcFace via insightface | Pre-trained on millions of faces, cosine distance threshold (0.6) is industry-standard for same-person matching |
| Clean sheet prompt engineering | Generic "remove background" prompts | Structured conditioning with reverse_prompt injection | Asset's reverse_prompt already optimized by Phase 5 LLM — reuse it with clean sheet directives |
| Reference selection logic | Random selection, first-3 assets | Role-based priority (subject > interaction_target > background) + quality scoring | Mirrors professional VFX shot breakdown practices, ensures plot-critical elements get identity preservation |

**Key insight:** Phase 8's complexity is in selection logic and validation, NOT in inference. Lean on existing models (rembg, ArcFace, Gemini Image) for heavy lifting. Focus custom code on business logic: which 3 references, when to upgrade quality, how to validate results.

## Common Pitfalls

### Pitfall 1: Forgetting 8-Second Duration Constraint

**What goes wrong:** Pipeline attempts to generate 4s or 6s clips with reference images, Veo API rejects with validation error.

**Why it happens:** Duration constraints are scene-specific in current pipeline. Adding reference images changes valid duration set from `[4, 6, 8]` to `[8]` only.

**How to avoid:**
- Check `len(selected_references) > 0` BEFORE setting `duration_seconds`
- Override `project.target_clip_duration` to 8 if references used
- Validate in submission code: `assert not (config.reference_images and config.duration_seconds != 8)`

**Warning signs:**
- Veo API errors mentioning "invalid duration"
- Inconsistent clip lengths in final stitched video
- Reference images ignored (fallback to no-reference mode)

**Source:** [Veo 3.1 duration constraints](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/use-reference-images-to-guide-video-generation)

### Pitfall 2: Base64 Size Limits with Multiple References

**What goes wrong:** 3 high-res reference images as base64 exceed Gemini API request size limits (~10MB), causing 413 errors.

**Why it happens:** Base64 encoding inflates file size by ~33%. 3 × 2MB PNGs = 6MB raw = 8MB base64 = near limit.

**How to avoid:**
- **Prefer GCS URIs for production:** `reference_image_url` already stored as GCS path
- Use `gcs_uri` parameter instead of `image_bytes` in ReferenceImage
- Only use base64 for transient/test scenarios or thumbnails
- Resize reference images to max 1024px longest edge before storage

**Code fix:**
```python
# BAD: Base64 for all 3 references
reference_images=[
    types.ReferenceImage(
        reference_image=types.Image(image_bytes=ref_bytes),  # Large!
        reference_type="asset"
    )
    for ref in selected_refs
]

# GOOD: GCS URIs (no size overhead)
reference_images=[
    types.ReferenceImage(
        reference_image=types.Image(gcs_uri=ref.reference_image_url),
        reference_type="asset"
    )
    for ref in selected_refs
]
```

### Pitfall 3: Clean Sheet Generation Loops Never Converge

**What goes wrong:** Tier 3 clean sheet fails validation (similarity < 0.6), retry with adjusted prompt, still fails, infinite loop.

**Why it happens:** Gemini Image may not preserve identity well for difficult cases (extreme angles, occlusion, low-res source).

**How to avoid:**
- **Max 3 attempts** per asset, then fall back to Tier 2 or Tier 1
- Log similarity scores for debugging: `logger.info(f"Clean sheet attempt {n}: similarity {score:.2f}")`
- Accept similarity ≥ 0.5 after 2 failures (loosen threshold)
- If all attempts fail, mark asset with `clean_sheet_status = "failed"`, use original

**Warning signs:**
- Clean sheet generation taking >60 seconds per asset
- Multiple Gemini Image calls for same asset in logs
- Similarity scores consistently 0.3-0.5 (borderline identity)

### Pitfall 4: Overwriting Original Assets with Clean Sheets

**What goes wrong:** Clean sheet saved to `Asset.reference_image_url`, original lost, can't revert if quality degrades.

**Why it happens:** Assuming clean sheet "replaces" the reference instead of being an alternative.

**How to avoid:**
- **Never modify `Asset.reference_image_url`** — that's the canonical source
- Store clean sheets in separate `asset_clean_references` table (schema below)
- Reference selection logic checks `asset_clean_references` first, falls back to original
- UI shows both original + clean versions, user can toggle

**Database schema (from v2-manifest.md lines 512-523):**
```sql
CREATE TABLE asset_clean_references (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES assets(id),
    tier VARCHAR(20) NOT NULL,  -- 'tier2_rembg', 'tier3_gemini', 'tier4_multi'
    clean_image_url VARCHAR(500) NOT NULL,
    generation_prompt TEXT,
    face_similarity_score FLOAT,
    quality_score FLOAT,
    is_primary BOOLEAN DEFAULT FALSE,  -- Which clean ref to use if multiple
    generation_cost FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Pitfall 5: Ignoring Scene Type in Reference Selection

**What goes wrong:** Establishing shot gets CHAR_01 face close-up as reference, wide environment shot gets character detail instead of spatial context.

**Why it happens:** Naive selection logic always picks highest quality_score, ignoring scene composition.

**How to avoid:**
- **Read `scene_manifest.composition.shot_type`** before selection
- Establishing shots: prioritize ENV and PROP over CHARACTER
- Close-ups: prioritize CHARACTER face crops over full-body
- Two-shots: ensure 2 different CHARACTER tags selected, not same character twice
- Use `composition.focal_point` to identify primary subject

**Example logic:**
```python
if scene_manifest.composition.shot_type == "establishing_shot":
    # Prioritize environments and props
    env_assets = [a for a in pool if a.asset_type == "ENVIRONMENT"]
    prop_assets = [a for a in pool if a.asset_type in ("PROP", "VEHICLE")]
    char_assets = [a for a in pool if a.asset_type == "CHARACTER"]
    selected = (env_assets[:2] + prop_assets[:1])[:3]
elif scene_manifest.composition.shot_type == "close_up":
    # Prioritize face crops
    face_crops = [a for a in pool if a.is_face_crop]
    selected = face_crops[:3]
```

## Code Examples

Verified patterns from design docs and existing codebase:

### Reference Selection and Passthrough

```python
# Source: video_gen.py enhancement for Phase 8
async def _generate_video_for_scene_with_references(
    session: AsyncSession,
    scene: Scene,
    project: Project,
    file_mgr: FileManager,
    client,
    video_model: str,
) -> None:
    """Enhanced video generation with reference image passthrough."""
    from vidpipe.services.manifest_service import load_manifest_assets
    from vidpipe.db.models import SceneManifest

    # Load scene manifest
    result = await session.execute(
        select(SceneManifest).where(
            SceneManifest.project_id == project.id,
            SceneManifest.scene_index == scene.scene_index
        )
    )
    scene_manifest = result.scalar_one_or_none()

    # Select references (if manifest exists and has placements)
    selected_refs = []
    if scene_manifest and project.manifest_id:
        all_assets = await load_manifest_assets(session, project.manifest_id)
        selected_refs = select_references_for_scene(
            scene_manifest.manifest_json,
            all_assets
        )

        # Store selected tags in manifest for debugging/UI
        scene_manifest.selected_reference_tags = [r.manifest_tag for r in selected_refs]
        await session.commit()

    # Load keyframes
    keyframes = await session.execute(
        select(Keyframe)
        .where(Keyframe.scene_id == scene.id)
        .order_by(Keyframe.position)
    )
    start_kf = next(k for k in keyframes if k.position == "start")
    end_kf = next(k for k in keyframes if k.position == "end")

    start_bytes = Path(start_kf.file_path).read_bytes()
    end_bytes = Path(end_kf.file_path).read_bytes()

    # Build Veo config with references
    video_config = types.GenerateVideosConfig(
        aspect_ratio=project.aspect_ratio,
        duration_seconds=8 if selected_refs else project.target_clip_duration,
        last_frame=types.Image(image_bytes=end_bytes, mime_type="image/png"),
        negative_prompt="photorealistic, photo, hyperrealistic, text, watermark",
    )

    # Add reference images if available
    if selected_refs:
        # Check for clean sheet overrides
        ref_images = []
        for asset in selected_refs:
            # Query asset_clean_references for primary clean sheet
            clean_ref = await get_primary_clean_reference(session, asset.id)
            if clean_ref:
                img_url = clean_ref.clean_image_url
            else:
                img_url = asset.reference_image_url

            ref_images.append(
                types.ReferenceImage(
                    reference_image=types.Image(gcs_uri=img_url),
                    reference_type="asset"
                )
            )

        video_config.reference_images = ref_images[:3]  # Max 3

    # Audio for Veo 3+
    if video_model != "veo-2.0-generate-001":
        video_config.generate_audio = bool(project.audio_enabled)

    # Seed for consistency
    if project.seed is not None:
        video_config.seed = project.seed

    # Submit job
    operation = await client.aio.models.generate_videos(
        model=video_model,
        prompt=scene.video_motion_prompt,
        image=types.Image(image_bytes=start_bytes, mime_type="image/png"),
        config=video_config,
    )

    # Existing polling logic continues...
```

### Clean Sheet Generation (Tier 2)

```python
# Source: Adapted from rembg docs + Phase 8 requirements
import asyncio
from pathlib import Path
from rembg import remove
from PIL import Image
import io

async def generate_tier2_clean_sheet(
    session: AsyncSession,
    asset: Asset,
    manifest_id: uuid.UUID,
) -> str:
    """Generate Tier 2 clean sheet: background removal via rembg.

    Returns:
        Path to saved clean sheet image
    """
    # Ensure directory exists
    clean_dir = Path("tmp/manifests") / str(manifest_id) / "clean_sheets"
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Load original
    original_bytes = Path(asset.reference_image_url).read_bytes()

    # Remove background (CPU-bound, run in thread pool)
    no_bg_bytes = await asyncio.to_thread(
        remove,
        original_bytes,
        force_return_bytes=True
    )

    # Composite onto gray background
    img = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
    background = Image.new("RGB", img.size, (128, 128, 128))
    background.paste(img, mask=img.split()[3])  # Alpha as mask

    # Save
    output_path = clean_dir / f"tier2_{asset.id}.png"
    background.save(output_path, format="PNG")

    # Create database record
    from vidpipe.db.models import AssetCleanReference

    clean_ref = AssetCleanReference(
        asset_id=asset.id,
        tier="tier2_rembg",
        clean_image_url=str(output_path),
        generation_prompt=None,  # No prompt for rembg
        face_similarity_score=None,  # No validation for Tier 2
        quality_score=None,
        is_primary=True,  # Mark as primary clean ref
        generation_cost=0.0,
    )
    session.add(clean_ref)
    await session.commit()

    return str(output_path)
```

### Clean Sheet Generation (Tier 3)

```python
# Source: Gemini Image conditioning pattern + face validation
async def generate_tier3_clean_sheet(
    session: AsyncSession,
    asset: Asset,
    manifest_id: uuid.UUID,
    image_model: str = "imagen-3.0-generate-001",
) -> Optional[str]:
    """Generate Tier 3 clean sheet: full Gemini Image generation with validation.

    Returns:
        Path to saved clean sheet, or None if validation failed
    """
    from vidpipe.services.vertex_client import get_vertex_client, location_for_model
    from google.genai import types
    import numpy as np

    client = get_vertex_client(location=location_for_model(image_model))

    # Build conditioning prompt
    conditioning_prompt = (
        f"{asset.reverse_prompt}\n\n"
        f"CRITICAL: Generate this subject on a clean, solid neutral gray background "
        f"(#808080 RGB). Studio lighting with even, soft illumination from multiple "
        f"angles. Subject centered, facing slightly toward camera at 15-degree angle. "
        f"No occlusion, shadows, or background elements. Full head and shoulders visible "
        f"for characters, or full object view for props/vehicles. Preserve ALL "
        f"distinguishing features, clothing details, and expressions EXACTLY as described. "
        f"Photorealistic quality, sharp focus."
    )

    # Load reference
    ref_bytes = Path(asset.reference_image_url).read_bytes()

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Generate clean version
            response = await client.aio.models.generate_images(
                model=image_model,
                prompt=conditioning_prompt,
                reference_images=[
                    types.Image(image_bytes=ref_bytes, mime_type="image/png")
                ],
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    include_rai_reason=True,
                    aspect_ratio="1:1",  # Square for character sheets
                )
            )

            clean_bytes = response.generated_images[0].image.image_bytes

            # Validate face similarity if character
            similarity = None
            if asset.asset_type == "CHARACTER" and asset.face_embedding:
                similarity = await asyncio.to_thread(
                    compute_face_similarity,
                    ref_bytes,
                    clean_bytes,
                    asset.face_embedding
                )

                threshold = 0.6 if attempt < 2 else 0.5  # Loosen on retry
                if similarity < threshold:
                    logger.warning(
                        f"Tier 3 clean sheet attempt {attempt+1}/{max_attempts} "
                        f"for {asset.manifest_tag}: similarity {similarity:.2f} "
                        f"< {threshold:.2f}, retrying..."
                    )
                    continue

            # Save clean sheet
            clean_dir = Path("tmp/manifests") / str(manifest_id) / "clean_sheets"
            clean_dir.mkdir(parents=True, exist_ok=True)
            output_path = clean_dir / f"tier3_{asset.id}.png"
            output_path.write_bytes(clean_bytes)

            # Create database record
            from vidpipe.db.models import AssetCleanReference

            clean_ref = AssetCleanReference(
                asset_id=asset.id,
                tier="tier3_gemini",
                clean_image_url=str(output_path),
                generation_prompt=conditioning_prompt,
                face_similarity_score=similarity,
                quality_score=None,  # Could add LLM quality assessment
                is_primary=True,
                generation_cost=0.03,  # Estimated Gemini Image cost
            )
            session.add(clean_ref)
            await session.commit()

            logger.info(
                f"Tier 3 clean sheet generated for {asset.manifest_tag} "
                f"(similarity: {similarity:.2f if similarity else 'N/A'})"
            )
            return str(output_path)

        except Exception as e:
            logger.error(f"Tier 3 attempt {attempt+1} failed: {e}")
            if attempt == max_attempts - 1:
                return None

    return None
```

### SceneCard Reference Display (Frontend)

```typescript
// Source: Phase 8 UI requirements
interface SceneCardProps {
  scene: Scene;
  sceneManifest?: SceneManifest;
  selectedReferences?: Asset[];
}

function SceneCard({ scene, sceneManifest, selectedReferences }: SceneCardProps) {
  return (
    <div className="scene-card">
      <h3>Scene {scene.scene_index}</h3>
      <p>{scene.scene_description}</p>

      {/* Existing: Keyframe thumbnails */}
      <div className="keyframes">
        {/* ... */}
      </div>

      {/* NEW: Reference images used for this scene */}
      {selectedReferences && selectedReferences.length > 0 && (
        <div className="reference-badges">
          <label>Identity References:</label>
          <div className="badge-row">
            {selectedReferences.map((ref) => (
              <div key={ref.id} className="ref-badge" title={ref.name}>
                <img
                  src={ref.thumbnail_url || ref.reference_image_url}
                  alt={ref.manifest_tag}
                  className="ref-thumb"
                />
                <span className="ref-tag">{ref.manifest_tag}</span>
                {ref.quality_score && (
                  <span className="quality-score">{ref.quality_score.toFixed(1)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Existing: Video clip player */}
      <div className="clip-player">
        {/* ... */}
      </div>
    </div>
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No reference images | Veo 3.1 asset reference support (max 3) | Veo 3.1 launch (Jan 2026) | Character identity preservation across scenes, eliminates "same character looks different" problem |
| Style images only | `referenceType: "asset"` for identity | Veo 2 → Veo 3.1 | Style transfer deprecated, replaced with identity-preserving references |
| First+last frame only | Hybrid: first frame + 3 references | Veo 3.1 API design | Spatial continuity (first frame) + identity consistency (references) in single generation |
| Manual background removal (Photoshop) | rembg automated removal | rembg 2.0 (2024-2026) | Free local inference, U-Net/BiRefNet models, production-ready |
| Fixed 4/6/8s duration | 8s mandatory with references | Veo 3.1 constraint | Pipeline must adapt duration based on reference usage |

**Deprecated/outdated:**
- **Style references on Veo 3.1:** `referenceType: "style"` only supported on `veo-2.0-generate-exp`, removed in Veo 3+
- **Base64-only uploads:** GCS URIs now preferred, avoid size limits
- **Manual clean sheet creation:** Gemini Image can generate clean backgrounds with conditioning

**Emerging patterns (AI video generation 2026):**
- **Reference budget optimization:** With 3-slot limit, strategic selection (subject > context > style) mirrors shot breakdown practices
- **Hybrid spatial+identity control:** Separate parameters for composition (first frame) vs. appearance (references)
- **Automated quality gates:** Face similarity validation (ArcFace cosine ≥0.6) prevents identity drift
- **Tiered preprocessing:** Free local (rembg) for bulk, paid API (Gemini) for quality-critical assets

## Open Questions

1. **GCS URI vs. base64 performance difference**
   - What we know: GCS URIs avoid 33% base64 overhead, no size limits
   - What's unclear: Latency difference in Veo API processing (GCS fetch vs. inline base64)
   - Recommendation: Default to GCS URIs for production. Use base64 only for testing with small images. Monitor Veo response times in both modes.

2. **Clean sheet quality vs. original for different asset types**
   - What we know: Tier 3 works well for characters (identity preservation verified via face similarity)
   - What's unclear: Quality impact for ENVIRONMENT, PROP, VEHICLE — no face validation metric
   - Recommendation: Phase 8 implements Tier 2/3 for CHARACTER only. Extend to other types in Phase 9 after observing Veo output quality with clean vs. original environment references.

3. **Reference image resolution requirements**
   - What we know: Veo accepts various resolutions, no official minimum documented
   - What's unclear: Optimal resolution for identity preservation (512px? 1024px? Original 2048px?)
   - Recommendation: Start with Asset Registry's native resolution (Phase 5 stores originals). Add optional downscaling to 1024px if GCS transfer times become issue. Monitor Veo quality degradation at different resolutions.

4. **Clean sheet validation for non-face assets**
   - What we know: ArcFace cosine distance works for CHARACTER faces (threshold 0.6)
   - What's unclear: How to validate PROP/VEHICLE clean sheets preserve visual identity without faces
   - Recommendation: Use CLIP embedding distance for non-character assets. Extract CLIP embedding from original + clean sheet, compute cosine similarity. Threshold TBD experimentally (likely 0.7-0.8 for general objects).

## Sources

### Primary (HIGH confidence)

**Veo 3.1 Reference Images:**
- [Generate Veo videos from reference images - Vertex AI docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/use-reference-images-to-guide-video-generation) - Official API documentation, `referenceImages` parameter structure
- [Veo on Vertex AI video generation API reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation) - Duration constraints, reference type enums
- [Generate videos with Veo 3.1 in Gemini API](https://ai.google.dev/gemini-api/docs/video) - 8-second mandatory duration with references confirmed

**Background Removal:**
- [rembg GitHub repository](https://github.com/danielgatis/rembg) - Latest release Jan 2026, Python ≥3.11, GPU support
- [rembg PyPI package](https://pypi.org/project/rembg/) - Installation, API usage, version history

**Gemini Image Generation:**
- [Gemini 3 Pro Image (Nano Banana Pro) overview](https://ai.google.dev/gemini-api/docs/image-generation) - Image conditioning, reference image support
- [Imagen 3 in Gemini API announcement](https://developers.googleblog.com/imagen-3-arrives-in-the-gemini-api/) - Capabilities, pricing, model ID

**Face Similarity:**
- [ArcFace face similarity with embeddings tutorial](https://medium.com/@ichigo.v.gen12/arcface-architecture-and-practical-example-how-to-calculate-the-face-similarity-between-images-183896a35957) - Cosine distance computation, 512-dim vectors
- [Face Recognition with ArcFace - LearnOpenCV](https://learnopencv.com/face-recognition-with-arcface/) - Threshold guidelines, embedding extraction

**V2 Architecture (Internal):**
- `/home/ubuntu/work/video-pipeline/docs/v2-manifest.md` - Clean sheet tiers (lines 297-356), database schema (lines 512-523)
- `/home/ubuntu/work/video-pipeline/docs/v2-pipe-optimization.md` - Reference selection strategy (lines 1561-1572), hybrid approach (lines 1574-1580)

### Secondary (MEDIUM confidence)

**Veo 3.1 Capabilities:**
- [Google Veo 3.1 Explained: Last-Frame Support & Reference Images](https://getimg.ai/blog/google-veo-3-1-review) - Community analysis of reference image quality
- [Ultimate prompting guide for Veo 3.1](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1) - Best practices for reference image prompting

**Python SDK Issues:**
- [GitHub Issue #1988 - Veo 3.1 Reference Images support in python-genai](https://github.com/googleapis/python-genai/issues/1988) - Known SDK limitations (resolved as of v0.12.0+)

### Tertiary (LOW confidence)

- Community blog posts on Veo reference image workflows - Anecdotal quality reports, not official documentation

## Metadata

**Confidence breakdown:**
- Veo 3.1 reference API: HIGH - Official Google Cloud documentation, multiple sources confirm
- 8-second duration constraint: HIGH - Explicitly stated in Vertex AI docs
- rembg background removal: HIGH - Mature library, latest release Jan 2026, extensive documentation
- Clean sheet generation: MEDIUM - Gemini Image supports conditioning but quality varies by asset type
- Face similarity thresholds: MEDIUM - ArcFace standard is 0.6, but may need tuning per use case
- GCS URI preference: MEDIUM - Implied by documentation, not explicitly benchmarked

**Research date:** 2026-02-16
**Valid until:** 30 days (Veo 3.1 API stable, rembg mature, but Google may release Veo 4.0 updates)

**Key dependencies validated:**
- Veo 3.1 `referenceImages` parameter: ✅ Confirmed in official API docs
- 8-second mandatory duration: ✅ Confirmed in Vertex AI video generation guide
- rembg Python ≥3.11 support: ✅ Verified in PyPI package metadata (Jan 2026 release)
- ArcFace cosine similarity: ✅ Standard face recognition metric, well-documented
- GCS URI support in google-genai SDK: ✅ Confirmed in SDK source and examples
- SceneManifest table schema: ✅ Exists in models.py (lines 153-167)
- Asset.face_embedding field: ✅ Exists in models.py (line 77), stores bytes

**Critical constraint flagged:**
- **8-second duration mandatory with references** — Must be enforced in video_gen.py config validation
