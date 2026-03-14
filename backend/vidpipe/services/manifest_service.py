"""
Manifest service layer for business logic and CRUD operations.

Handles manifest and asset lifecycle management including creation, updates,
deletion, duplication, and asset tagging. All functions accept an AsyncSession
parameter for transaction management by the caller.
"""
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import (
    Actor,
    Asset,
    CastBinding,
    Keyframe,
    LibraryProp,
    LibrarySet,
    ManifestSnapshot,
    ProductionBible,
    PropBinding,
    Scene,
    SetBinding,
    Shot,
)

# Backwards-compat alias used internally in this module
Manifest = ProductionBible

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


async def create_manifest_from_scene(
    session: AsyncSession,
    scene_id: uuid.UUID,
    name: Optional[str] = None,
) -> tuple[Manifest, list[Asset]]:
    """Create a manifest pre-populated from a scene's storyboard data.

    Extracts characters, shot environments, and style guide from
    storyboard_raw and creates corresponding assets.

    Args:
        session: Active database session
        scene_id: Source scene UUID
        name: Optional manifest name (defaults to truncated scene prompt)

    Returns:
        Tuple of (created Manifest, list of created Assets)

    Raises:
        ValueError: If scene not found or has no storyboard data
    """
    result = await session.execute(
        select(Scene).where(Scene.id == scene_id)
    )
    scene = result.scalar_one_or_none()
    if not scene:
        raise ValueError(f"Scene {scene_id} not found")

    if not scene.storyboard_raw:
        raise ValueError(f"Scene {scene_id} has no storyboard data")

    storyboard = scene.storyboard_raw

    # Derive manifest name from scene prompt if not provided
    if not name:
        prompt_text = scene.prompt or "Untitled"
        name = prompt_text[:80] + ("..." if len(prompt_text) > 80 else "")

    manifest = await create_manifest(
        session,
        name=name,
        description=f"Auto-imported from scene {scene_id}",
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
            description=f"Character from scene import: {char_name}",
        )
        asset.source = "scene_import"
        if reverse_prompt:
            asset.reverse_prompt = reverse_prompt
        assets_list.append(asset)

    # --- Environments (one per shot, using start keyframe) ---
    shots_data = storyboard.get("shots", [])

    # Query actual shots + keyframes from the database for file paths
    shot_result = await session.execute(
        select(Shot)
        .where(Shot.scene_id == scene_id)
        .order_by(Shot.shot_index)
    )
    db_shots = list(shot_result.scalars().all())

    # Build map of shot_index -> start keyframe file_path
    keyframe_map: dict[int, str] = {}
    if db_shots:
        shot_ids = [s.id for s in db_shots]
        kf_result = await session.execute(
            select(Keyframe).where(
                Keyframe.shot_id.in_(shot_ids),
                Keyframe.position == "start",
            )
        )
        keyframes = kf_result.scalars().all()
        shot_id_to_index = {s.id: s.shot_index for s in db_shots}
        for kf in keyframes:
            idx = shot_id_to_index.get(kf.shot_id)
            if idx is not None:
                keyframe_map[idx] = kf.file_path

    for i, shot_data in enumerate(shots_data):
        shot_desc = shot_data.get("shot_description", "")
        start_prompt = shot_data.get("start_frame_prompt", "")

        asset = await create_asset(
            session,
            manifest_id=manifest.id,
            name=f"Shot {i + 1} Environment",
            asset_type="ENVIRONMENT",
            description=shot_desc or None,
        )
        asset.source = "scene_import"
        if start_prompt:
            asset.reverse_prompt = start_prompt

        # Copy keyframe image if available
        src_path = keyframe_map.get(i)
        if src_path:
            from vidpipe.services.storage_backend import get_storage_backend, LocalStorageBackend
            from vidpipe.services.file_manager import FileManager
            storage = get_storage_backend()
            file_mgr = FileManager()
            try:
                src_data = await file_mgr.read_bytes(src_path)
                if isinstance(storage, LocalStorageBackend):
                    dest_dir = Path("tmp/manifests") / str(manifest.id) / "uploads"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / f"{asset.id}_{Path(src_path).name}"
                    dest_path.write_bytes(src_data)
                    asset.reference_image_url = f"/api/assets/{asset.id}/image"
                else:
                    key = f"manifests/{manifest.id}/uploads/{asset.id}_{Path(src_path).name}"
                    await storage.put(key, src_data, "image/png")
                    asset.reference_image_url = key
            except FileNotFoundError:
                pass  # Source keyframe not available

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
            description="Style guide from scene import",
        )
        asset.source = "scene_import"
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
        ValueError: If manifest not found or if referenced by active scenes
    """
    manifest = await get_manifest(session, manifest_id)
    if not manifest:
        raise ValueError(f"Manifest {manifest_id} not found")

    # Check if any scenes reference this production bible
    result = await session.execute(
        select(func.count(Scene.id)).where(Scene.production_bible_id == manifest_id)
    )
    scene_count = result.scalar()

    if scene_count > 0:
        raise ValueError(f"Cannot delete manifest: referenced by {scene_count} scene(s)")

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
    new_manifest = ProductionBible(
        name=copy_name,
        description=source.description,
        category=source.category,
        tags=source.tags,
        status="DRAFT",
        version=1,
        parent_production_bible_id=source.id,
    )
    session.add(new_manifest)
    await session.flush()

    # Copy all assets
    assets_result = await session.execute(
        select(Asset).where(Asset.production_bible_id == manifest_id)
    )
    assets = assets_result.scalars().all()

    for asset in assets:
        new_asset = Asset(
            production_bible_id=new_manifest.id,
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
            Asset.production_bible_id == manifest_id,
            Asset.asset_type == asset_type
        )
    )
    count = result.scalar()
    prefix = TAG_PREFIX_MAP[asset_type]
    manifest_tag = f"{prefix}_{count + 1:02d}"

    asset = Asset(
        production_bible_id=manifest_id,
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
        .where(Asset.production_bible_id == manifest_id)
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
                    Asset.production_bible_id == asset.production_bible_id,
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

    Also removes associated files from S3 storage if applicable.

    Args:
        session: Active database session
        asset_id: Asset UUID

    Raises:
        ValueError: If asset not found
    """
    from vidpipe.services.storage_backend import get_storage_backend

    asset = await get_asset(session, asset_id)
    if not asset:
        raise ValueError(f"Asset {asset_id} not found")

    manifest_id = asset.production_bible_id

    # Collect S3 keys to delete (parent + children)
    s3_keys_to_delete: list[str] = []
    storage = get_storage_backend()

    def _collect_s3_key(url: str | None) -> None:
        """Collect an S3 key if it's not a local API path."""
        if url and not url.startswith("/api/") and not storage.is_local():
            s3_keys_to_delete.append(url)

    _collect_s3_key(asset.reference_image_url)

    # Delete child assets first (extracted crops referencing this asset)
    # Must flush children before deleting parent to satisfy FK constraints
    children = await session.execute(
        select(Asset).where(Asset.source_asset_id == asset_id)
    )
    child_list = list(children.scalars().all())
    if child_list:
        for child in child_list:
            _collect_s3_key(child.reference_image_url)
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

    # Clean up S3 files after DB operations
    for key in s3_keys_to_delete:
        try:
            await storage.delete(key)
        except Exception:
            logger.warning(f"Failed to delete S3 asset file {key}")


def save_asset_image(
    manifest_id: uuid.UUID,
    asset_id: uuid.UUID,
    file_content: bytes,
    filename: str,
) -> str:
    """Save uploaded image to disk (local backend only).

    NOT async - pure filesystem I/O. Caller should wrap in asyncio.to_thread().
    For S3 backend, caller should use storage.put() directly instead.

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
    scene_id: uuid.UUID,
) -> ManifestSnapshot:
    """Create a snapshot of manifest state at generation time.

    Args:
        session: Active database session
        manifest_id: Manifest UUID to snapshot
        scene_id: Scene UUID this snapshot belongs to

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
        production_bible_id=manifest_id,
        scene_id=scene_id,
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
        .where(Asset.production_bible_id == manifest_id, Asset.is_inherited == False)
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
        return "No assets registered. Describe all visual elements in shots."

    lines = ["AVAILABLE ASSETS FOR THIS SCENE:", "━" * 40]

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
    lines.append(
        "Reference assets by [TAG]. You MUST use existing CHARACTER tags — do NOT "
        "create new CHARACTER tags for people already represented in the registry. "
        "You may declare new ENVIRONMENT or PROP assets not in the registry."
    )

    return "\n".join(lines)


async def format_binding_registry(
    session: AsyncSession,
    production_bible_id: uuid.UUID,
) -> str | None:
    """Format all bound assets for a production bible as structured text for LLM context injection.

    Uses the @tag syntax instead of [TAG] manifest tags. Queries CastBindings,
    SetBindings, and PropBindings, batch-loads referenced entities, and formats
    a text block suitable for system prompt injection.

    Returns None if no bindings exist (signals caller to fall back to legacy
    asset registry path).

    Args:
        session: Active database session
        production_bible_id: Production bible UUID

    Returns:
        Formatted text block with @TAG references, or None if no bindings
    """
    # Query all binding types
    cast_result = await session.execute(
        select(CastBinding)
        .where(CastBinding.production_bible_id == production_bible_id)
        .order_by(CastBinding.created_at)
    )
    cast_bindings = list(cast_result.scalars().all())

    set_result = await session.execute(
        select(SetBinding)
        .where(SetBinding.production_bible_id == production_bible_id)
        .order_by(SetBinding.created_at)
    )
    set_bindings = list(set_result.scalars().all())

    prop_result = await session.execute(
        select(PropBinding)
        .where(PropBinding.production_bible_id == production_bible_id)
        .order_by(PropBinding.created_at)
    )
    prop_bindings = list(prop_result.scalars().all())

    # If no bindings at all, return None to signal fallback
    if not cast_bindings and not set_bindings and not prop_bindings:
        return None

    # Batch-load referenced entities
    actors_by_id: dict[uuid.UUID, Actor] = {}
    if cast_bindings:
        actor_ids = [b.actor_id for b in cast_bindings]
        actor_result = await session.execute(
            select(Actor).where(Actor.id.in_(actor_ids))
        )
        actors_by_id = {a.id: a for a in actor_result.scalars().all()}

    lib_sets_by_id: dict[uuid.UUID, LibrarySet] = {}
    if set_bindings:
        set_ids = [b.library_set_id for b in set_bindings]
        lib_set_result = await session.execute(
            select(LibrarySet).where(LibrarySet.id.in_(set_ids))
        )
        lib_sets_by_id = {s.id: s for s in lib_set_result.scalars().all()}

    lib_props_by_id: dict[uuid.UUID, LibraryProp] = {}
    if prop_bindings:
        prop_ids = [b.library_prop_id for b in prop_bindings]
        lib_prop_result = await session.execute(
            select(LibraryProp).where(LibraryProp.id.in_(prop_ids))
        )
        lib_props_by_id = {p.id: p for p in lib_prop_result.scalars().all()}

    # Format output
    lines = ["AVAILABLE ASSETS FOR THIS PRODUCTION:", "━" * 40]

    def _truncate(text: str | None, max_len: int = 200) -> str:
        if not text:
            return ""
        return text[:max_len] + "..." if len(text) > max_len else text

    # CHARACTER bindings
    for cb in cast_bindings:
        actor = actors_by_id.get(cb.actor_id)
        role_str = f" ({cb.role})" if cb.role else ""
        if actor:
            description = _truncate(actor.base_appearance_prompt)
            lines.append(f'[CHARACTER] @{cb.tag} — "{cb.character_name}"{role_str}')
            if description:
                lines.append(f"  {description}")
        else:
            lines.append(f'[CHARACTER] @{cb.tag} — "{cb.character_name}"{role_str} (asset deleted)')
        lines.append("")

    # SET bindings
    for sb in set_bindings:
        lib_set = lib_sets_by_id.get(sb.library_set_id)
        display_name = sb.production_name or (lib_set.name if lib_set else "Unknown Set")
        if lib_set:
            description = _truncate(lib_set.reverse_prompt)
            lines.append(f'[SET] @{sb.tag} — "{display_name}"')
            if description:
                lines.append(f"  {description}")
            if lib_set.lighting_notes:
                lines.append(f"  Lighting: {_truncate(lib_set.lighting_notes, 150)}")
        else:
            lines.append(f'[SET] @{sb.tag} — "{display_name}" (asset deleted)')
        lines.append("")

    # PROP bindings
    for pb in prop_bindings:
        lib_prop = lib_props_by_id.get(pb.library_prop_id)
        display_name = pb.production_name or (lib_prop.name if lib_prop else "Unknown Prop")
        if lib_prop:
            description = _truncate(lib_prop.appearance_prompt)
            lines.append(f'[PROP] @{pb.tag} — "{display_name}"')
            if description:
                lines.append(f"  {description}")
        else:
            lines.append(f'[PROP] @{pb.tag} — "{display_name}" (asset deleted)')
        lines.append("")

    lines.append("━" * 40)
    lines.append(
        "Reference assets by @TAG in your scene descriptions and shot prompts.\n"
        "You MUST use existing @TAG references for characters, sets, and props\n"
        "already listed above. Do NOT invent new names for listed assets."
    )

    return "\n".join(lines)
