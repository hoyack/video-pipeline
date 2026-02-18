"""
Manifest service layer for business logic and CRUD operations.

Handles manifest and asset lifecycle management including creation, updates,
deletion, duplication, and asset tagging. All functions accept an AsyncSession
parameter for transaction management by the caller.
"""
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, Keyframe, Manifest, ManifestSnapshot, Project, Scene

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


async def create_manifest_from_project(
    session: AsyncSession,
    project_id: uuid.UUID,
    name: Optional[str] = None,
) -> tuple[Manifest, list[Asset]]:
    """Create a manifest pre-populated from a project's storyboard data.

    Extracts characters, scene environments, and style guide from
    storyboard_raw and creates corresponding assets.

    Args:
        session: Active database session
        project_id: Source project UUID
        name: Optional manifest name (defaults to truncated project prompt)

    Returns:
        Tuple of (created Manifest, list of created Assets)

    Raises:
        ValueError: If project not found or has no storyboard data
    """
    result = await session.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    if not project.storyboard_raw:
        raise ValueError(f"Project {project_id} has no storyboard data")

    storyboard = project.storyboard_raw

    # Derive manifest name from project prompt if not provided
    if not name:
        prompt_text = project.prompt or "Untitled"
        name = prompt_text[:80] + ("..." if len(prompt_text) > 80 else "")

    manifest = await create_manifest(
        session,
        name=name,
        description=f"Auto-imported from project {project_id}",
        category="FULL_PRODUCTION",
    )

    assets_list: list[Asset] = []

    # --- Characters ---
    characters = storyboard.get("characters", [])
    for char in characters:
        char_name = char.get("name", "Unknown Character")
        phys = char.get("physical_description", "")
        cloth = char.get("clothing_description", "")
        reverse_prompt = ". ".join(filter(None, [phys, cloth]))

        asset = await create_asset(
            session,
            manifest_id=manifest.id,
            name=char_name,
            asset_type="CHARACTER",
            description=f"Character from project import: {char_name}",
        )
        asset.source = "project_import"
        if reverse_prompt:
            asset.reverse_prompt = reverse_prompt
        assets_list.append(asset)

    # --- Environments (one per scene, using start keyframe) ---
    scenes_data = storyboard.get("scenes", [])

    # Query actual scenes + keyframes from the database for file paths
    scene_result = await session.execute(
        select(Scene)
        .where(Scene.project_id == project_id)
        .order_by(Scene.scene_index)
    )
    db_scenes = list(scene_result.scalars().all())

    # Build map of scene_index -> start keyframe file_path
    keyframe_map: dict[int, str] = {}
    if db_scenes:
        scene_ids = [s.id for s in db_scenes]
        kf_result = await session.execute(
            select(Keyframe).where(
                Keyframe.scene_id.in_(scene_ids),
                Keyframe.position == "start",
            )
        )
        keyframes = kf_result.scalars().all()
        scene_id_to_index = {s.id: s.scene_index for s in db_scenes}
        for kf in keyframes:
            idx = scene_id_to_index.get(kf.scene_id)
            if idx is not None:
                keyframe_map[idx] = kf.file_path

    for i, scene_data in enumerate(scenes_data):
        scene_desc = scene_data.get("scene_description", "")
        start_prompt = scene_data.get("start_frame_prompt", "")

        asset = await create_asset(
            session,
            manifest_id=manifest.id,
            name=f"Scene {i + 1} Environment",
            asset_type="ENVIRONMENT",
            description=scene_desc or None,
        )
        asset.source = "project_import"
        if start_prompt:
            asset.reverse_prompt = start_prompt

        # Copy keyframe image if available
        src_path = keyframe_map.get(i)
        if src_path and Path(src_path).exists():
            dest_dir = Path("tmp/manifests") / str(manifest.id) / "uploads"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{asset.id}_{Path(src_path).name}"
            shutil.copy2(src_path, dest_path)
            asset.reference_image_url = f"/api/assets/{asset.id}/image"

        assets_list.append(asset)

    # --- Style guide ---
    style_guide = storyboard.get("style_guide", {})
    if style_guide:
        parts = filter(None, [
            style_guide.get("visual_style"),
            style_guide.get("color_palette"),
            style_guide.get("camera_style"),
        ])
        style_reverse_prompt = ". ".join(parts)

        asset = await create_asset(
            session,
            manifest_id=manifest.id,
            name="Visual Style",
            asset_type="STYLE",
            description="Style guide from project import",
        )
        asset.source = "project_import"
        if style_reverse_prompt:
            asset.reverse_prompt = style_reverse_prompt
        assets_list.append(asset)

    await session.flush()
    return manifest, assets_list


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

    # Delete child assets first (extracted crops referencing this asset)
    # Must flush children before deleting parent to satisfy FK constraints
    children = await session.execute(
        select(Asset).where(Asset.source_asset_id == asset_id)
    )
    child_list = list(children.scalars().all())
    if child_list:
        for child in child_list:
            await session.delete(child)
        await session.flush()

    # Delete parent asset
    await session.delete(asset)
    await session.flush()

    # Update manifest asset count (parent + children)
    deleted_count = 1 + len(child_list)
    manifest = await get_manifest(session, manifest_id)
    if manifest:
        manifest.asset_count = max(0, manifest.asset_count - deleted_count)
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


async def load_manifest_assets(
    session: AsyncSession,
    manifest_id: uuid.UUID,
) -> list[Asset]:
    """Load canonical (non-inherited) assets for a manifest, ordered by quality score descending.

    Used for LLM context injection where highest-quality assets should
    appear first in the system prompt for better attention distribution.

    Filters out inherited copies (created during fork) to avoid duplicate
    tags in the asset registry. Inherited copies share the same manifest_id
    but are marked with is_inherited=True.

    Args:
        session: Active database session
        manifest_id: Manifest UUID

    Returns:
        List of Asset instances ordered by quality_score desc (nulls last)
    """
    result = await session.execute(
        select(Asset)
        .where(Asset.manifest_id == manifest_id, Asset.is_inherited == False)
        .order_by(Asset.quality_score.desc().nullslast())
    )
    return list(result.scalars().all())


def format_asset_registry(assets: list[Asset]) -> str:
    """Format asset list as structured text block for LLM system prompt injection.

    For each asset, includes:
    - Header: [TAG] "Name" (type, quality: X/10)
    - Reverse prompt: Truncated to 200 chars
    - Production notes (visual_description): Only for quality >= 7.0, truncated to 150 chars

    Args:
        assets: List of Asset instances to format

    Returns:
        Formatted text block for LLM context injection
    """
    if not assets:
        return "No assets registered. Describe all visual elements in scenes."

    lines = ["AVAILABLE ASSETS FOR THIS PROJECT:", "━" * 40]

    for asset in assets:
        # Header line with quality score
        quality_str = f"{asset.quality_score:.1f}/10" if asset.quality_score is not None else "N/A"
        lines.append(f"[{asset.manifest_tag}] \"{asset.name}\" ({asset.asset_type}, quality: {quality_str})")

        # Reverse prompt (truncated to 200 chars)
        if asset.reverse_prompt:
            reverse_prompt = asset.reverse_prompt
            if len(reverse_prompt) > 200:
                reverse_prompt = reverse_prompt[:200] + "..."
            lines.append(f"  {reverse_prompt}")

        # Production notes only for high-quality assets (>= 7.0)
        if asset.visual_description and asset.quality_score is not None and asset.quality_score >= 7.0:
            visual_desc = asset.visual_description
            if len(visual_desc) > 150:
                visual_desc = visual_desc[:150] + "..."
            lines.append(f"  Production notes: {visual_desc}")

        lines.append("")  # Blank line between assets

    lines.append("━" * 40)
    lines.append("Reference assets by [TAG]. You may declare NEW assets not in the registry.")

    return "\n".join(lines)
