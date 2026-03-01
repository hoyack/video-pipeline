"""Asset-to-entity migration service for Production Bible Entity Expansion.

Converts existing CHARACTER and ENVIRONMENT assets into structured Character
and Set entities. Migration is idempotent -- calling twice does not create
duplicates (checks by name within the same production bible).

Spec reference: Phase 17 - PBEX-06, PBEX-12
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, Character, Set

logger = logging.getLogger(__name__)


async def migrate_character_assets(session: AsyncSession, bible_id: uuid.UUID) -> int:
    """Migrate CHARACTER assets to Character entities.

    For each CHARACTER asset in the given production bible, create a Character
    entity if one with the same name does not already exist.

    Args:
        session: Active async session (caller controls commit).
        bible_id: Production bible UUID.

    Returns:
        Count of Character entities created.
    """
    # Fetch all CHARACTER assets for this bible
    result = await session.execute(
        select(Asset).where(
            Asset.production_bible_id == bible_id,
            Asset.asset_type == "CHARACTER",
        )
    )
    character_assets = result.scalars().all()

    if not character_assets:
        return 0

    # Fetch existing character names for idempotency check
    existing_result = await session.execute(
        select(Character.name).where(Character.production_bible_id == bible_id)
    )
    existing_names = {row[0] for row in existing_result.all()}

    created = 0
    for asset in character_assets:
        name = asset.name or asset.manifest_tag
        if not name:
            logger.warning("Skipping CHARACTER asset %s with no name or manifest_tag", asset.id)
            continue

        if name in existing_names:
            logger.debug("Character '%s' already exists for bible %s, skipping", name, bible_id)
            continue

        character = Character(
            production_bible_id=bible_id,
            name=name,
            role="SUPPORTING",  # safe default
            description=asset.visual_description,
            base_appearance=asset.reverse_prompt,
            prompt_tags=[asset.manifest_tag] if asset.manifest_tag else None,
            actor_refs=[asset.reference_image_url] if asset.reference_image_url else None,
        )
        session.add(character)
        existing_names.add(name)  # prevent duplicates within same batch
        created += 1

    if created:
        await session.flush()
        logger.info("Created %d Character entities from CHARACTER assets for bible %s", created, bible_id)

    return created


async def migrate_environment_assets(session: AsyncSession, bible_id: uuid.UUID) -> int:
    """Migrate ENVIRONMENT assets to Set entities.

    For each ENVIRONMENT asset in the given production bible, create a Set
    entity if one with the same name does not already exist.

    Args:
        session: Active async session (caller controls commit).
        bible_id: Production bible UUID.

    Returns:
        Count of Set entities created.
    """
    # Fetch all ENVIRONMENT assets for this bible
    result = await session.execute(
        select(Asset).where(
            Asset.production_bible_id == bible_id,
            Asset.asset_type == "ENVIRONMENT",
        )
    )
    environment_assets = result.scalars().all()

    if not environment_assets:
        return 0

    # Fetch existing set names for idempotency check
    existing_result = await session.execute(
        select(Set.name).where(Set.production_bible_id == bible_id)
    )
    existing_names = {row[0] for row in existing_result.all()}

    created = 0
    for asset in environment_assets:
        name = asset.name or asset.manifest_tag
        if not name:
            logger.warning("Skipping ENVIRONMENT asset %s with no name or manifest_tag", asset.id)
            continue

        if name in existing_names:
            logger.debug("Set '%s' already exists for bible %s, skipping", name, bible_id)
            continue

        set_entity = Set(
            production_bible_id=bible_id,
            name=name,
            reference_image=asset.reference_image_url,
            reverse_prompt=asset.reverse_prompt,
            style_tags=[asset.manifest_tag] if asset.manifest_tag else None,
            prompt_tags=[asset.manifest_tag] if asset.manifest_tag else None,
        )
        session.add(set_entity)
        existing_names.add(name)  # prevent duplicates within same batch
        created += 1

    if created:
        await session.flush()
        logger.info("Created %d Set entities from ENVIRONMENT assets for bible %s", created, bible_id)

    return created


async def migrate_all_assets(session: AsyncSession, bible_id: uuid.UUID) -> dict:
    """Migrate all CHARACTER and ENVIRONMENT assets to structured entities.

    Calls migrate_character_assets and migrate_environment_assets. Caller
    controls the transaction (commit/rollback).

    Args:
        session: Active async session.
        bible_id: Production bible UUID.

    Returns:
        Dict with keys 'characters_created' and 'sets_created'.
    """
    characters_created = await migrate_character_assets(session, bible_id)
    sets_created = await migrate_environment_assets(session, bible_id)

    return {
        "characters_created": characters_created,
        "sets_created": sets_created,
    }
