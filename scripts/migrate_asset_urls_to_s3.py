#!/usr/bin/env python3
"""Migrate legacy /api/assets/... URLs in the database to S3 storage keys.

For each Asset with a reference_image_url like ``/api/assets/{id}/image``,
searches S3 for the matching file under ``manifests/{pb_id}/{uploads,crops}/``
and updates the column to the S3 key.

Also migrates:
- ProductionBible.contact_sheet_url (``/api/production-bibles/{id}/contact-sheet`` -> S3 key)
- AssetCleanReference.clean_image_url (legacy paths -> S3 keys)

Usage:
    python scripts/migrate_asset_urls_to_s3.py [--dry-run]

Requires VIDPIPE_STORAGE__STORAGE_BACKEND=s3 and S3 env vars.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(dry_run: bool = False):
    from vidpipe.db import async_session
    from vidpipe.db.models import Asset, AssetCleanReference, ProductionBible
    from vidpipe.services.storage_backend import get_storage_backend
    from sqlalchemy import select

    storage = get_storage_backend()
    if storage.is_local():
        logger.error("Storage backend is 'local'. Set VIDPIPE_STORAGE__STORAGE_BACKEND=s3")
        sys.exit(1)

    logger.info(f"{'DRY RUN — ' if dry_run else ''}Migrating asset URLs to S3 keys...")

    async with async_session() as session:
        # --- 1. Migrate Asset.reference_image_url ---
        result = await session.execute(
            select(Asset).where(
                Asset.reference_image_url.like("/api/assets/%")
            )
        )
        assets = result.scalars().all()
        logger.info(f"Found {len(assets)} assets with legacy /api/ URLs")

        updated_assets = 0
        not_found_assets = 0
        for asset in assets:
            s3_key = None
            for subdir in ("uploads", "crops"):
                prefix = f"manifests/{asset.production_bible_id}/{subdir}/{asset.id}_"
                keys = await storage.list_prefix(prefix, limit=1)
                if keys:
                    s3_key = keys[0]
                    break

            if s3_key:
                if dry_run:
                    logger.info(f"  [dry-run] Asset {asset.id}: {asset.reference_image_url} -> {s3_key}")
                else:
                    asset.reference_image_url = s3_key
                updated_assets += 1
            else:
                logger.warning(f"  Asset {asset.id}: no S3 file found (pb={asset.production_bible_id})")
                not_found_assets += 1

        logger.info(f"Assets: {updated_assets} updated, {not_found_assets} not found in S3")

        # --- 2. Migrate ProductionBible.contact_sheet_url ---
        result = await session.execute(
            select(ProductionBible).where(
                ProductionBible.contact_sheet_url.like("/api/%")
            )
        )
        pbs = result.scalars().all()
        logger.info(f"Found {len(pbs)} production bibles with legacy contact sheet URLs")

        updated_pbs = 0
        for pb in pbs:
            s3_key = f"manifests/{pb.id}/contact_sheet.jpg"
            exists = await storage.exists(s3_key)
            if exists:
                if dry_run:
                    logger.info(f"  [dry-run] PB {pb.id}: {pb.contact_sheet_url} -> {s3_key}")
                else:
                    pb.contact_sheet_url = s3_key
                updated_pbs += 1
            else:
                logger.warning(f"  PB {pb.id}: contact sheet not found in S3")

        logger.info(f"Production Bibles: {updated_pbs} contact sheets updated")

        # --- 3. Migrate AssetCleanReference.clean_image_url ---
        result = await session.execute(
            select(AssetCleanReference).where(
                AssetCleanReference.clean_image_url.like("/api/%")
                | AssetCleanReference.clean_image_url.like("/tmp/%")
                | AssetCleanReference.clean_image_url.like("tmp/%")
            )
        )
        clean_refs = result.scalars().all()
        logger.info(f"Found {len(clean_refs)} clean references with legacy URLs")

        updated_clean = 0
        for cr in clean_refs:
            # Clean refs are stored at manifests/{pb_id}/clean_sheets/{asset_id}_{tier}.png
            # Get the asset to find the pb_id
            asset = await session.get(Asset, cr.asset_id)
            if not asset:
                logger.warning(f"  CleanRef {cr.id}: asset {cr.asset_id} not found")
                continue

            prefix = f"manifests/{asset.production_bible_id}/clean_sheets/{cr.asset_id}_"
            keys = await storage.list_prefix(prefix, limit=1)
            if keys:
                if dry_run:
                    logger.info(f"  [dry-run] CleanRef {cr.id}: {cr.clean_image_url} -> {keys[0]}")
                else:
                    cr.clean_image_url = keys[0]
                updated_clean += 1
            else:
                logger.warning(f"  CleanRef {cr.id}: not found in S3 (asset={cr.asset_id})")

        logger.info(f"Clean References: {updated_clean} updated")

        if not dry_run:
            await session.commit()
            logger.info("All changes committed.")
        else:
            logger.info("Dry run complete — no changes made.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
