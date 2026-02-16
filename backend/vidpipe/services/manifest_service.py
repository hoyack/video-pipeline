"""
Manifest service layer for business logic and CRUD operations.

Handles manifest and asset lifecycle management including creation, updates,
deletion, duplication, and asset tagging. All functions accept an AsyncSession
parameter for transaction management by the caller.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, Manifest, ManifestSnapshot, Project

# Valid enum constants
VALID_CATEGORIES = {"CHARACTERS", "ENVIRONMENT", "FULL_PRODUCTION", "STYLE_KIT", "BRAND_KIT", "CUSTOM"}
VALID_ASSET_TYPES = {"CHARACTER", "OBJECT", "ENVIRONMENT", "PROP", "STYLE", "VEHICLE", "OTHER"}
TAG_PREFIX_MAP = {
    "CHARACTER": "CHAR",
    "OBJECT": "OBJ",
    "ENVIRONMENT": "ENV",
    "PROP": "PROP",
    "STYLE": "STYLE",
    "VEHICLE": "VEH",
    "OTHER": "OTHER",
}


async def create_manifest(
    session: AsyncSession,
    name: str,
    description: Optional[str] = None,
    category: str = "CUSTOM",
    tags: Optional[list] = None,
) -> Manifest:
    """Create a new manifest in DRAFT status.

    Args:
        session: Active database session
        name: Manifest name
        description: Optional description
        category: One of VALID_CATEGORIES
        tags: Optional list of tag strings

    Returns:
        Created Manifest instance

    Raises:
        ValueError: If category is invalid
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of {VALID_CATEGORIES}")

    manifest = Manifest(
        name=name,
        description=description,
        category=category,
        tags=tags,
        status="DRAFT",
    )
    session.add(manifest)
    await session.flush()
    return manifest


async def list_manifests(
    session: AsyncSession,
    category: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "updated_at",
    sort_order: str = "desc",
) -> list[Manifest]:
    """List non-deleted manifests with optional filters and sorting.

    Args:
        session: Active database session
        category: Filter by category
        status: Filter by status
        sort_by: Column to sort by (updated_at, created_at, name, times_used, asset_count)
        sort_order: Sort direction (asc or desc)

    Returns:
        List of Manifest instances
    """
    query = select(Manifest).where(Manifest.deleted_at.is_(None))

    if category:
        query = query.where(Manifest.category == category)
    if status:
        query = query.where(Manifest.status == status)

    # Apply sorting
    sort_col = getattr(Manifest, sort_by, Manifest.updated_at)
    if sort_order == "asc":
        query = query.order_by(sort_col.asc())
    else:
        query = query.order_by(sort_col.desc())

    result = await session.execute(query)
    return list(result.scalars().all())


async def get_manifest(
    session: AsyncSession,
    manifest_id: uuid.UUID,
) -> Optional[Manifest]:
    """Get single manifest by ID (only if not deleted).

    Args:
        session: Active database session
        manifest_id: Manifest UUID

    Returns:
        Manifest instance or None if not found or deleted
    """
    result = await session.execute(
        select(Manifest).where(
            Manifest.id == manifest_id,
            Manifest.deleted_at.is_(None)
        )
    )
    return result.scalar_one_or_none()


async def update_manifest(
    session: AsyncSession,
    manifest_id: uuid.UUID,
    **kwargs,
) -> Manifest:
    """Update manifest fields.

    Allowed fields: name, description, category, tags

    Args:
        session: Active database session
        manifest_id: Manifest UUID
        **kwargs: Fields to update

    Returns:
        Updated Manifest instance

    Raises:
        ValueError: If manifest not found or invalid category
    """
    manifest = await get_manifest(session, manifest_id)
    if not manifest:
        raise ValueError(f"Manifest {manifest_id} not found")

    # Validate category if provided
    if "category" in kwargs and kwargs["category"] not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{kwargs['category']}'. Must be one of {VALID_CATEGORIES}")

    # Only allow updating specific fields
    allowed_fields = {"name", "description", "category", "tags"}
    for field in allowed_fields:
        if field in kwargs:
            setattr(manifest, field, kwargs[field])

    await session.flush()
    return manifest


async def delete_manifest(
    session: AsyncSession,
    manifest_id: uuid.UUID,
) -> None:
    """Soft delete manifest by setting deleted_at timestamp.

    Raises:
        ValueError: If manifest not found or if referenced by active projects
    """
    manifest = await get_manifest(session, manifest_id)
    if not manifest:
        raise ValueError(f"Manifest {manifest_id} not found")

    # Check if any projects reference this manifest
    result = await session.execute(
        select(func.count(Project.id)).where(Project.manifest_id == manifest_id)
    )
    project_count = result.scalar()

    if project_count > 0:
        raise ValueError(f"Cannot delete manifest: referenced by {project_count} project(s)")

    manifest.deleted_at = func.now()
    await session.flush()


async def duplicate_manifest(
    session: AsyncSession,
    manifest_id: uuid.UUID,
    new_name: Optional[str] = None,
) -> Manifest:
    """Create a copy of a manifest with all its assets.

    Args:
        session: Active database session
        manifest_id: Source manifest UUID
        new_name: Optional name for the copy (defaults to "{original_name} (Copy)")

    Returns:
        New Manifest instance with copied assets

    Raises:
        ValueError: If source manifest not found
    """
    source = await get_manifest(session, manifest_id)
    if not source:
        raise ValueError(f"Source manifest {manifest_id} not found")

    # Create new manifest
    copy_name = new_name or f"{source.name} (Copy)"
    new_manifest = Manifest(
        name=copy_name,
        description=source.description,
        category=source.category,
        tags=source.tags,
        status="DRAFT",
        version=1,
        parent_manifest_id=source.id,
    )
    session.add(new_manifest)
    await session.flush()

    # Copy all assets
    assets_result = await session.execute(
        select(Asset).where(Asset.manifest_id == manifest_id)
    )
    assets = assets_result.scalars().all()

    for asset in assets:
        new_asset = Asset(
            manifest_id=new_manifest.id,
            asset_type=asset.asset_type,
            name=asset.name,
            manifest_tag=asset.manifest_tag,
            user_tags=asset.user_tags,
            reference_image_url=asset.reference_image_url,
            thumbnail_url=asset.thumbnail_url,
            description=asset.description,
            source=asset.source,
            sort_order=asset.sort_order,
        )
        session.add(new_asset)

    new_manifest.asset_count = len(assets)
    await session.flush()
    return new_manifest


async def create_asset(
    session: AsyncSession,
    manifest_id: uuid.UUID,
    name: str,
    asset_type: str,
    description: Optional[str] = None,
    user_tags: Optional[list] = None,
) -> Asset:
    """Create an asset within a manifest with auto-generated manifest_tag.

    Args:
        session: Active database session
        manifest_id: Parent manifest UUID
        name: Asset name
        asset_type: One of VALID_ASSET_TYPES
        description: Optional description
        user_tags: Optional list of user-defined tags

    Returns:
        Created Asset instance

    Raises:
        ValueError: If asset_type invalid or manifest not found
    """
    if asset_type not in VALID_ASSET_TYPES:
        raise ValueError(f"Invalid asset_type '{asset_type}'. Must be one of {VALID_ASSET_TYPES}")

    manifest = await get_manifest(session, manifest_id)
    if not manifest:
        raise ValueError(f"Manifest {manifest_id} not found")

    # Auto-generate manifest_tag by counting existing assets of same type
    result = await session.execute(
        select(func.count(Asset.id)).where(
            Asset.manifest_id == manifest_id,
            Asset.asset_type == asset_type
        )
    )
    count = result.scalar()
    prefix = TAG_PREFIX_MAP[asset_type]
    manifest_tag = f"{prefix}_{count + 1:02d}"

    asset = Asset(
        manifest_id=manifest_id,
        asset_type=asset_type,
        name=name,
        manifest_tag=manifest_tag,
        description=description,
        user_tags=user_tags,
    )
    session.add(asset)

    # Update manifest asset count
    manifest.asset_count += 1
    await session.flush()
    return asset


async def list_assets(
    session: AsyncSession,
    manifest_id: uuid.UUID,
) -> list[Asset]:
    """List all assets for a manifest, ordered by sort_order then created_at.

    Args:
        session: Active database session
        manifest_id: Manifest UUID

    Returns:
        List of Asset instances
    """
    result = await session.execute(
        select(Asset)
        .where(Asset.manifest_id == manifest_id)
        .order_by(Asset.sort_order, Asset.created_at)
    )
    return list(result.scalars().all())


async def get_asset(
    session: AsyncSession,
    asset_id: uuid.UUID,
) -> Optional[Asset]:
    """Get single asset by ID.

    Args:
        session: Active database session
        asset_id: Asset UUID

    Returns:
        Asset instance or None if not found
    """
    result = await session.execute(
        select(Asset).where(Asset.id == asset_id)
    )
    return result.scalar_one_or_none()


async def update_asset(
    session: AsyncSession,
    asset_id: uuid.UUID,
    **kwargs,
) -> Asset:
    """Update asset fields.

    Allowed fields: name, description, asset_type, user_tags, sort_order
    If asset_type changes, manifest_tag is regenerated.

    Args:
        session: Active database session
        asset_id: Asset UUID
        **kwargs: Fields to update

    Returns:
        Updated Asset instance

    Raises:
        ValueError: If asset not found or invalid asset_type
    """
    asset = await get_asset(session, asset_id)
    if not asset:
        raise ValueError(f"Asset {asset_id} not found")

    # Validate asset_type if provided
    if "asset_type" in kwargs:
        new_type = kwargs["asset_type"]
        if new_type not in VALID_ASSET_TYPES:
            raise ValueError(f"Invalid asset_type '{new_type}'. Must be one of {VALID_ASSET_TYPES}")

        # Regenerate manifest_tag if type changed
        if new_type != asset.asset_type:
            result = await session.execute(
                select(func.count(Asset.id)).where(
                    Asset.manifest_id == asset.manifest_id,
                    Asset.asset_type == new_type
                )
            )
            count = result.scalar()
            prefix = TAG_PREFIX_MAP[new_type]
            asset.manifest_tag = f"{prefix}_{count + 1:02d}"
            asset.asset_type = new_type

    # Update allowed fields
    allowed_fields = {"name", "description", "user_tags", "sort_order", "reverse_prompt", "visual_description"}
    for field in allowed_fields:
        if field in kwargs:
            setattr(asset, field, kwargs[field])

    await session.flush()
    return asset


async def delete_asset(
    session: AsyncSession,
    asset_id: uuid.UUID,
) -> None:
    """Hard delete an asset and update parent manifest asset_count.

    Args:
        session: Active database session
        asset_id: Asset UUID

    Raises:
        ValueError: If asset not found
    """
    asset = await get_asset(session, asset_id)
    if not asset:
        raise ValueError(f"Asset {asset_id} not found")

    manifest_id = asset.manifest_id

    # Delete asset
    await session.delete(asset)

    # Update manifest asset count
    manifest = await get_manifest(session, manifest_id)
    if manifest:
        manifest.asset_count = max(0, manifest.asset_count - 1)

    await session.flush()


def save_asset_image(
    manifest_id: uuid.UUID,
    asset_id: uuid.UUID,
    file_content: bytes,
    filename: str,
) -> str:
    """Save uploaded image to disk.

    NOT async - pure filesystem I/O. Caller should wrap in asyncio.to_thread().

    Args:
        manifest_id: Parent manifest UUID
        asset_id: Asset UUID
        file_content: Image file bytes
        filename: Original filename

    Returns:
        Path string to saved file
    """
    # Create directory structure: tmp/manifests/{manifest_id}/uploads/
    base_dir = Path("tmp/manifests") / str(manifest_id) / "uploads"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save with asset_id prefix to ensure uniqueness
    filepath = base_dir / f"{asset_id}_{filename}"
    filepath.write_bytes(file_content)

    return str(filepath)


async def create_snapshot(
    session: AsyncSession,
    manifest_id: uuid.UUID,
    project_id: uuid.UUID,
) -> ManifestSnapshot:
    """Create a snapshot of manifest state at generation time.

    Args:
        session: Active database session
        manifest_id: Manifest UUID to snapshot
        project_id: Project UUID this snapshot belongs to

    Returns:
        Created ManifestSnapshot instance

    Raises:
        ValueError: If manifest not found or is deleted
    """
    # Query manifest
    manifest = await get_manifest(session, manifest_id)
    if not manifest:
        raise ValueError(f"Manifest {manifest_id} not found")

    # Query all assets for this manifest
    assets = await list_assets(session, manifest_id)

    # Serialize manifest fields into snapshot_data
    snapshot_data = {
        "manifest": {
            "id": str(manifest.id),
            "name": manifest.name,
            "description": manifest.description,
            "category": manifest.category,
            "tags": manifest.tags,
            "contact_sheet_url": manifest.contact_sheet_url,
            "version": manifest.version,
            "status": manifest.status,
            "asset_count": manifest.asset_count,
            "total_processing_cost": manifest.total_processing_cost,
        },
        "assets": [],
    }

    # Serialize each asset
    for asset in assets:
        asset_data = {
            "id": str(asset.id),
            "asset_type": asset.asset_type,
            "name": asset.name,
            "manifest_tag": asset.manifest_tag,
            "user_tags": asset.user_tags,
            "reference_image_url": asset.reference_image_url,
            "thumbnail_url": asset.thumbnail_url,
            "description": asset.description,
            "source": asset.source,
            "sort_order": asset.sort_order,
            "reverse_prompt": asset.reverse_prompt,
            "visual_description": asset.visual_description,
            "detection_class": asset.detection_class,
            "detection_confidence": asset.detection_confidence,
            "is_face_crop": asset.is_face_crop,
            "crop_bbox": asset.crop_bbox,
            "quality_score": asset.quality_score,
        }
        snapshot_data["assets"].append(asset_data)

    # Create snapshot
    snapshot = ManifestSnapshot(
        manifest_id=manifest_id,
        project_id=project_id,
        version_at_snapshot=manifest.version,
        snapshot_data=snapshot_data,
    )
    session.add(snapshot)
    await session.flush()

    return snapshot


async def increment_usage(
    session: AsyncSession,
    manifest_id: uuid.UUID,
) -> None:
    """Increment manifest usage tracking.

    Args:
        session: Active database session
        manifest_id: Manifest UUID to update

    Raises:
        ValueError: If manifest not found
    """
    manifest = await get_manifest(session, manifest_id)
    if not manifest:
        raise ValueError(f"Manifest {manifest_id} not found")

    manifest.times_used += 1
    manifest.last_used_at = datetime.now(timezone.utc)
    await session.flush()
