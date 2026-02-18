"""Reference selection service for Veo 3-reference passthrough.

Implements scene-type-aware selection logic that picks up to 3 optimal reference
images per scene based on asset roles and scene composition type.

Spec reference: Phase 8 - Veo Reference Passthrough and Clean Sheets
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, AssetCleanReference
from vidpipe.schemas.storyboard_enhanced import SceneManifestSchema

logger = logging.getLogger(__name__)


def select_references_for_scene(
    scene_manifest_json: dict,
    all_assets: list[Asset],
) -> list[Asset]:
    """Select up to 3 optimal reference images for a scene.

    Implements scene-type-aware selection:
    - close_up: Prioritize face crops of subject role
    - two_shot: Up to 2 unique characters, fill with environment
    - establishing: Prioritize environments, then props, characters last
    - default (medium_shot, wide_shot): Subject > interaction_target > background

    Args:
        scene_manifest_json: Scene manifest JSON (from SceneManifest.manifest_json)
        all_assets: All manifest assets available for this project

    Returns:
        List of up to 3 Asset objects selected for Veo reference passthrough
    """
    try:
        # Parse scene manifest into pydantic model
        manifest = SceneManifestSchema(**scene_manifest_json)
    except Exception as e:
        logger.warning(f"Failed to parse scene manifest: {e}")
        return []

    if not manifest.placements:
        return []

    # Build asset map for quick lookup
    asset_map = {asset.manifest_tag: asset for asset in all_assets}

    # Map placements to asset objects (skip new_asset_declarations not in registry)
    placed_assets = []
    for placement in manifest.placements:
        asset = asset_map.get(placement.asset_tag)
        if asset:
            placed_assets.append((asset, placement))

    if not placed_assets:
        return []

    # Extract shot type for scene-aware selection
    shot_type = manifest.composition.shot_type.lower()

    # Scene-type-aware selection logic
    if shot_type == "close_up":
        # Prioritize face crops of subject role
        return _select_close_up_references(placed_assets)
    elif shot_type == "two_shot":
        # Get up to 2 unique CHARACTER assets, fill with environment
        return _select_two_shot_references(placed_assets)
    elif shot_type == "establishing":
        # Prioritize environments, then props, then characters
        return _select_establishing_references(placed_assets)
    else:
        # Default: medium_shot, wide_shot, and any other type
        return _select_default_references(placed_assets)


def _select_close_up_references(placed_assets: list[tuple[Asset, object]]) -> list[Asset]:
    """Select references for close_up shots.

    Priority:
    1. Face crops of subject role (is_face_crop=True)
    2. Full-body subject assets
    3. Environment assets
    """
    selected = []

    # Group by priority
    face_crops_subject = []
    full_body_subject = []
    environments = []

    for asset, placement in placed_assets:
        if placement.role == "subject":
            if asset.is_face_crop:
                face_crops_subject.append(asset)
            else:
                full_body_subject.append(asset)
        elif asset.asset_type == "ENVIRONMENT":
            environments.append(asset)

    # Sort each group by quality_score descending
    face_crops_subject.sort(key=lambda a: a.quality_score or 0, reverse=True)
    full_body_subject.sort(key=lambda a: a.quality_score or 0, reverse=True)
    environments.sort(key=lambda a: a.quality_score or 0, reverse=True)

    # Build selection (up to 3 total)
    selected.extend(face_crops_subject[:3])
    if len(selected) < 3:
        selected.extend(full_body_subject[: 3 - len(selected)])
    if len(selected) < 3:
        selected.extend(environments[: 3 - len(selected)])

    return _deduplicate_by_tag(selected)[:3]


def _select_two_shot_references(placed_assets: list[tuple[Asset, object]]) -> list[Asset]:
    """Select references for two_shot scenes.

    Priority:
    1. Up to 2 unique CHARACTER assets from subject/interaction_target roles
    2. Fill remaining slot with environment asset
    """
    selected = []

    # Group by type
    characters = []
    environments = []

    for asset, placement in placed_assets:
        if asset.asset_type == "CHARACTER":
            if placement.role in ("subject", "interaction_target"):
                characters.append(asset)
        elif asset.asset_type == "ENVIRONMENT":
            environments.append(asset)

    # Sort by quality_score
    characters.sort(key=lambda a: a.quality_score or 0, reverse=True)
    environments.sort(key=lambda a: a.quality_score or 0, reverse=True)

    # Deduplicate characters (prefer face crops for two_shot)
    unique_characters = _deduplicate_by_tag(characters)

    # Take up to 2 unique characters
    selected.extend(unique_characters[:2])

    # Fill remaining slot with environment
    if len(selected) < 3:
        selected.extend(environments[: 3 - len(selected)])

    return selected[:3]


def _select_establishing_references(placed_assets: list[tuple[Asset, object]]) -> list[Asset]:
    """Select references for establishing shots.

    Priority:
    1. ENVIRONMENT type assets
    2. PROP/VEHICLE type assets
    3. Characters last (prefer full-body over face crops)
    """
    selected = []

    # Group by priority
    environments = []
    props = []
    characters = []

    for asset, placement in placed_assets:
        if asset.asset_type == "ENVIRONMENT":
            environments.append(asset)
        elif asset.asset_type in ("PROP", "VEHICLE"):
            props.append(asset)
        elif asset.asset_type == "CHARACTER":
            # For establishing shots, prefer full-body over face crops
            characters.append(asset)

    # Sort each group by quality_score descending
    environments.sort(key=lambda a: a.quality_score or 0, reverse=True)
    props.sort(key=lambda a: a.quality_score or 0, reverse=True)
    characters.sort(key=lambda a: a.quality_score or 0, reverse=True)

    # Build selection (up to 3 total)
    selected.extend(environments[:3])
    if len(selected) < 3:
        selected.extend(props[: 3 - len(selected)])
    if len(selected) < 3:
        # For establishing, prefer full-body characters (filter out face crops)
        full_body_chars = [c for c in characters if not c.is_face_crop]
        selected.extend(full_body_chars[: 3 - len(selected)])

    return _deduplicate_by_tag(selected)[:3]


def _select_default_references(placed_assets: list[tuple[Asset, object]]) -> list[Asset]:
    """Select references for default scene types (medium_shot, wide_shot, etc.).

    Priority:
    1. Subject role first
    2. Interaction_target role
    3. Background role
    Sort by quality_score descending within each priority group.
    """
    selected = []

    # Group by role priority
    subject = []
    interaction_target = []
    background = []

    for asset, placement in placed_assets:
        if placement.role == "subject":
            subject.append(asset)
        elif placement.role == "interaction_target":
            interaction_target.append(asset)
        elif placement.role in ("background", "environment"):
            background.append(asset)

    # Sort each group by quality_score descending
    subject.sort(key=lambda a: a.quality_score or 0, reverse=True)
    interaction_target.sort(key=lambda a: a.quality_score or 0, reverse=True)
    background.sort(key=lambda a: a.quality_score or 0, reverse=True)

    # Build selection (up to 3 total)
    selected.extend(subject[:3])
    if len(selected) < 3:
        selected.extend(interaction_target[: 3 - len(selected)])
    if len(selected) < 3:
        selected.extend(background[: 3 - len(selected)])

    return _deduplicate_by_tag(selected)[:3]


def _deduplicate_by_tag(assets: list[Asset]) -> list[Asset]:
    """Deduplicate assets by manifest_tag, keeping first occurrence.

    For characters with both face crop and full-body, this keeps whichever
    was prioritized first in the selection logic.
    """
    seen_tags = set()
    unique = []
    for asset in assets:
        if asset.manifest_tag not in seen_tags:
            seen_tags.add(asset.manifest_tag)
            unique.append(asset)
    return unique


async def get_primary_clean_reference(
    session: AsyncSession,
    asset_id: uuid.UUID,
) -> Optional[AssetCleanReference]:
    """Get primary clean reference for an asset.

    Queries asset_clean_references table for the given asset_id where is_primary=True.
    Used by video_gen to check if a clean sheet override exists.

    Args:
        session: SQLAlchemy async session
        asset_id: Asset UUID to query

    Returns:
        AssetCleanReference if found, None otherwise
    """
    result = await session.execute(
        select(AssetCleanReference).where(
            AssetCleanReference.asset_id == asset_id,
            AssetCleanReference.is_primary == True,
        )
    )
    return result.scalar_one_or_none()


async def resolve_asset_image_bytes(
    session: AsyncSession,
    asset: Asset,
) -> Optional[bytes]:
    """Resolve an asset's reference image to raw bytes.

    Checks for a clean sheet override first, then falls back to
    asset.reference_image_url.  Resolves ``/api/assets/...`` URLs to
    local files under ``tmp/manifests/{manifest_id}/{uploads|crops}/``.

    Returns None on any failure (soft failure — caller should skip).
    """
    try:
        # Prefer clean sheet override
        clean_ref = await get_primary_clean_reference(session, asset.id)
        img_url = clean_ref.clean_image_url if clean_ref else asset.reference_image_url

        if not img_url:
            return None

        # Resolve /api/assets/... URLs to local file paths
        if img_url.startswith("/api/assets/"):
            # Extract the primary asset UUID from the URL pattern:
            # /api/assets/{uuid}/image  (or /api/assets/{uuid}/clean-image)
            # This is needed because duplicate asset rows share the same
            # reference_image_url pointing to the primary asset's file,
            # but asset.id may differ from the primary.
            url_match = re.search(
                r"/api/assets/([0-9a-f-]{36})/", img_url
            )
            file_asset_id = url_match.group(1) if url_match else str(asset.id)

            manifest_dir = Path("tmp/manifests") / str(asset.manifest_id)
            resolved = None
            for subdir in ("uploads", "crops"):
                d = manifest_dir / subdir
                if d.exists():
                    matches = list(d.glob(f"{file_asset_id}_*"))
                    if matches:
                        resolved = matches[0]
                        break
            if not resolved:
                logger.warning(
                    f"Reference image not found on disk for asset {asset.id} "
                    f"(file_id={file_asset_id})"
                )
                return None
            return resolved.read_bytes()
        else:
            p = Path(img_url)
            if p.exists():
                return p.read_bytes()
            logger.warning(f"Reference image path does not exist: {img_url}")
            return None

    except Exception as e:
        logger.warning(f"Failed to resolve reference image for asset {asset.id}: {e}")
        return None
