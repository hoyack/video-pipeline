"""Clean sheet generation service for reference image preprocessing.

Tier 2: local background removal via rembg.
Tier 3: full clean sheet via Gemini Image with face similarity validation.

Spec reference: Phase 8 - Clean Sheets
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, AssetCleanReference
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

logger = logging.getLogger(__name__)

# Lazy-loaded globals for optional dependencies
_rembg_remove = None
_face_app = None


def _load_rembg():
    """Lazy-load rembg module for background removal."""
    global _rembg_remove
    if _rembg_remove is None:
        try:
            from rembg import remove
            _rembg_remove = remove
            logger.info("Loaded rembg for Tier 2 clean sheet generation")
        except ImportError:
            logger.warning(
                "rembg not installed - Tier 2 clean sheets unavailable. "
                "Install with: pip install rembg"
            )
            _rembg_remove = False
    return _rembg_remove if _rembg_remove is not False else None


def _load_face_analyzer():
    """Lazy-load insightface FaceAnalysis for face similarity validation."""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Loaded insightface for face similarity validation")
        except ImportError:
            logger.warning(
                "insightface not installed - face validation unavailable. "
                "Install with: pip install insightface"
            )
            _face_app = False
    return _face_app if _face_app is not False else None


async def generate_tier2_clean_sheet(
    session: AsyncSession,
    asset: Asset,
    manifest_id: uuid.UUID,
) -> Optional[str]:
    """Generate Tier 2 clean sheet using rembg background removal.

    Process:
    - Load original image from asset.reference_image_url
    - Run rembg.remove() in thread pool (CPU-bound)
    - Convert RGBA to RGB with #808080 gray background
    - Save to tmp/manifests/{manifest_id}/clean_sheets/tier2_{asset.id}.png
    - Create AssetCleanReference record

    Args:
        session: SQLAlchemy async session
        asset: Asset to generate clean sheet for
        manifest_id: Manifest UUID for directory structure

    Returns:
        Path string to clean sheet, or None if rembg unavailable
    """
    rembg_remove = _load_rembg()
    if rembg_remove is None:
        return None

    if not asset.reference_image_url:
        logger.warning(f"Asset {asset.id} has no reference_image_url")
        return None

    # Load original image
    try:
        from PIL import Image
        import io

        original_bytes = Path(asset.reference_image_url).read_bytes()

        # Run rembg in thread pool (CPU-bound operation)
        removed_bytes = await asyncio.to_thread(rembg_remove, original_bytes)

        # Convert RGBA to RGB with gray background
        rgba_img = Image.open(io.BytesIO(removed_bytes))
        rgb_img = Image.new("RGB", rgba_img.size, (128, 128, 128))  # #808080
        rgb_img.paste(rgba_img, mask=rgba_img.split()[3])  # alpha channel as mask

        # Save to tmp/manifests/{manifest_id}/clean_sheets/tier2_{asset_id}.png
        output_dir = Path(f"tmp/manifests/{manifest_id}/clean_sheets")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tier2_{asset.id}.png"

        rgb_img.save(output_path, "PNG")
        logger.info(f"Generated Tier 2 clean sheet for asset {asset.id}: {output_path}")

        # Create AssetCleanReference record
        clean_ref = AssetCleanReference(
            asset_id=asset.id,
            tier="tier2_rembg",
            clean_image_url=str(output_path),
            generation_prompt=None,
            face_similarity_score=None,
            quality_score=None,
            is_primary=True,
            generation_cost=0.0,
        )
        session.add(clean_ref)
        await session.commit()

        return str(output_path)

    except Exception as e:
        logger.error(f"Tier 2 clean sheet generation failed for asset {asset.id}: {e}")
        return None


def compute_face_similarity(
    original_bytes: bytes,
    clean_bytes: bytes,
    stored_embedding: bytes,
) -> float:
    """Compute face similarity between clean sheet and stored embedding.

    Synchronous helper called via asyncio.to_thread.

    Args:
        original_bytes: Original image bytes (unused, for future reference)
        clean_bytes: Clean sheet image bytes
        stored_embedding: Stored face embedding from Asset.face_embedding

    Returns:
        Float 0.0-1.0 similarity score
    """
    face_app = _load_face_analyzer()
    if face_app is None:
        logger.warning("insightface unavailable - skipping validation")
        return 1.0  # Skip validation

    try:
        from PIL import Image
        import io
        import numpy as np

        # Extract face from clean sheet
        clean_img = Image.open(io.BytesIO(clean_bytes))
        clean_arr = np.array(clean_img)

        faces = face_app.get(clean_arr)
        if not faces:
            logger.warning("No face detected in clean sheet")
            return 0.0

        # Get embedding from detected face
        clean_embedding = faces[0].embedding

        # Load stored embedding
        stored_arr = np.frombuffer(stored_embedding, dtype=np.float32)

        # Compute cosine similarity
        cosine_sim = np.dot(clean_embedding, stored_arr) / (
            np.linalg.norm(clean_embedding) * np.linalg.norm(stored_arr)
        )

        return float(cosine_sim)

    except Exception as e:
        logger.error(f"Face similarity computation failed: {e}")
        return 0.0


async def generate_tier3_clean_sheet(
    session: AsyncSession,
    asset: Asset,
    manifest_id: uuid.UUID,
    image_model: str = "imagen-3.0-generate-002",
) -> Optional[str]:
    """Generate Tier 3 clean sheet using Gemini Image with face similarity validation.

    Process:
    - Build conditioning prompt from asset.reverse_prompt with clean sheet directives
    - Call Gemini Image with reference image
    - Validate face similarity for CHARACTER assets (3 attempts with threshold loosening)
    - Save to tmp/manifests/{manifest_id}/clean_sheets/tier3_{asset.id}.png
    - Create AssetCleanReference record

    Args:
        session: SQLAlchemy async session
        asset: Asset to generate clean sheet for
        manifest_id: Manifest UUID for directory structure
        image_model: Gemini Image model ID

    Returns:
        Path string to clean sheet, or None on failure
    """
    if not asset.reference_image_url:
        logger.warning(f"Asset {asset.id} has no reference_image_url")
        return None

    # Load reference image
    reference_bytes = Path(asset.reference_image_url).read_bytes()

    # Build conditioning prompt
    conditioning_prompt = (
        f"Generate a clean, idealized reference image of: {asset.reverse_prompt or asset.name}.\n\n"
        f"CRITICAL REQUIREMENTS:\n"
        f"- Neutral gray background (#808080)\n"
        f"- Studio lighting, no harsh shadows\n"
        f"- No occlusion or obstructions\n"
        f"- Preserve all distinguishing features (face, clothing, markings)\n"
        f"- Clear, unobstructed view\n"
        f"- Professional reference quality\n\n"
        f"Visual description: {asset.visual_description or 'N/A'}"
    )

    # Get Vertex AI client
    client = get_vertex_client(location=location_for_model(image_model))

    # For CHARACTER assets with face_embedding, validate face similarity
    max_attempts = 3
    threshold = 0.6
    best_image = None
    best_score = 0.0

    for attempt in range(max_attempts):
        try:
            # Generate image via Gemini Image
            from google.genai import types

            response = await client.aio.models.generate_images(
                model=image_model,
                prompt=conditioning_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="block_low_and_above",
                    person_generation="allow_adult",
                ),
                reference_images=[
                    types.RawReferenceImage(
                        reference_image=types.Image(
                            image_bytes=reference_bytes,
                            mime_type="image/png"
                        ),
                        reference_id=1,
                    )
                ],
            )

            if not response.generated_images:
                logger.warning(f"Tier 3 attempt {attempt + 1}: No images generated")
                continue

            gen_img = response.generated_images[0]
            clean_bytes = gen_img.image.image_bytes

            # Validate face similarity for CHARACTER assets
            if asset.asset_type == "CHARACTER" and asset.face_embedding:
                # Compute similarity in thread pool (CPU-bound)
                score = await asyncio.to_thread(
                    compute_face_similarity,
                    reference_bytes,
                    clean_bytes,
                    asset.face_embedding,
                )

                logger.info(
                    f"Tier 3 attempt {attempt + 1} for asset {asset.id}: "
                    f"face similarity = {score:.3f} (threshold = {threshold:.2f})"
                )

                # Track best result
                if score > best_score:
                    best_score = score
                    best_image = clean_bytes

                # Loosen threshold on final attempt
                if attempt == 2:
                    threshold = 0.5

                # Check if passed threshold
                if score >= threshold:
                    break
            else:
                # No validation needed for non-CHARACTER or no embedding
                best_image = clean_bytes
                best_score = 1.0
                break

        except Exception as e:
            logger.warning(f"Tier 3 attempt {attempt + 1} failed: {e}")
            continue

    # Check if validation succeeded
    if best_image is None:
        logger.error(f"Tier 3 clean sheet generation failed for asset {asset.id} after {max_attempts} attempts")
        return None

    if asset.asset_type == "CHARACTER" and asset.face_embedding and best_score < threshold:
        logger.warning(
            f"Tier 3 clean sheet for asset {asset.id} failed validation: "
            f"best score {best_score:.3f} < threshold {threshold:.2f}"
        )
        return None

    # Save clean sheet
    try:
        output_dir = Path(f"tmp/manifests/{manifest_id}/clean_sheets")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tier3_{asset.id}.png"

        output_path.write_bytes(best_image)
        logger.info(
            f"Generated Tier 3 clean sheet for asset {asset.id}: {output_path} "
            f"(similarity: {best_score:.3f})"
        )

        # Create AssetCleanReference record
        clean_ref = AssetCleanReference(
            asset_id=asset.id,
            tier="tier3_gemini",
            clean_image_url=str(output_path),
            generation_prompt=conditioning_prompt,
            face_similarity_score=best_score if asset.face_embedding else None,
            quality_score=None,
            is_primary=True,
            generation_cost=0.03,  # Estimated cost for Imagen generation
        )
        session.add(clean_ref)
        await session.commit()

        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to save Tier 3 clean sheet for asset {asset.id}: {e}")
        return None


async def generate_clean_sheet(
    session: AsyncSession,
    asset: Asset,
    manifest_id: uuid.UUID,
    tier: str = "tier2",
) -> Optional[str]:
    """Convenience dispatcher for clean sheet generation.

    Args:
        session: SQLAlchemy async session
        asset: Asset to generate clean sheet for
        manifest_id: Manifest UUID
        tier: "tier2" or "tier3"

    Returns:
        Path string to clean sheet, or None on failure
    """
    if tier == "tier2":
        return await generate_tier2_clean_sheet(session, asset, manifest_id)
    elif tier == "tier3":
        return await generate_tier3_clean_sheet(session, asset, manifest_id)
    else:
        logger.error(f"Unknown tier: {tier}")
        return None
