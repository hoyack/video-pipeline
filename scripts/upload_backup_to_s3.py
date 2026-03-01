#!/usr/bin/env python3
"""Upload local backup manifest files to S3 and update asset DB URLs.

Walks .temp/tmp/manifests/ (or a given directory), uploads each file to S3,
then updates Asset.reference_image_url from legacy /api/ paths to S3 keys.

Usage:
    python scripts/upload_backup_to_s3.py [--dry-run] [--source-dir DIR]
"""

import argparse
import asyncio
import logging
import mimetypes
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(source_dir: str, dry_run: bool = False):
    from vidpipe.db import async_session
    from vidpipe.db.models import Asset, ProductionBible
    from vidpipe.services.storage_backend import get_storage_backend
    from sqlalchemy import select

    storage = get_storage_backend()
    if storage.is_local():
        logger.error("Storage backend is 'local'. Set VIDPIPE_STORAGE__STORAGE_BACKEND=s3")
        sys.exit(1)

    src = Path(source_dir).resolve()
    if not src.exists():
        logger.error(f"Source directory does not exist: {src}")
        sys.exit(1)

    logger.info(f"{'DRY RUN — ' if dry_run else ''}Uploading from {src} to S3...")

    # Phase 1: Upload all files to S3
    uploaded = 0
    skipped = 0
    asset_key_map: dict[str, dict[str, str]] = {}  # pb_id -> {asset_id -> s3_key}

    for pb_dir in sorted(src.iterdir()):
        if not pb_dir.is_dir():
            continue
        pb_id = pb_dir.name

        for root, _dirs, files in os.walk(pb_dir):
            for fname in files:
                local_path = Path(root) / fname
                rel = local_path.relative_to(src)
                key = f"manifests/{rel}"
                content_type = mimetypes.guess_type(fname)[0] or "application/octet-stream"

                # Check if already in S3
                exists = await storage.exists(key)
                if exists:
                    skipped += 1
                    # Still record for DB update
                    match = re.match(r"^([0-9a-f-]{36})_", fname)
                    if match:
                        asset_key_map.setdefault(pb_id, {})[match.group(1)] = key
                    continue

                if dry_run:
                    logger.info(f"  [dry-run] Would upload: {key} ({local_path.stat().st_size} bytes)")
                else:
                    data = local_path.read_bytes()
                    await storage.put(key, data, content_type)
                uploaded += 1

                # Track asset file mappings
                match = re.match(r"^([0-9a-f-]{36})_", fname)
                if match:
                    asset_key_map.setdefault(pb_id, {})[match.group(1)] = key

    logger.info(f"Upload: {uploaded} new, {skipped} already existed")

    # Phase 2: Update Asset.reference_image_url in DB
    async with async_session() as session:
        updated_assets = 0
        for pb_id, mappings in asset_key_map.items():
            import uuid as _uuid
            try:
                pb_uuid = _uuid.UUID(pb_id)
            except ValueError:
                continue

            result = await session.execute(
                select(Asset).where(
                    Asset.production_bible_id == pb_uuid,
                    Asset.reference_image_url.isnot(None),
                )
            )
            assets = result.scalars().all()

            for asset in assets:
                aid = str(asset.id)
                if aid in mappings:
                    s3_key = mappings[aid]
                    if asset.reference_image_url != s3_key:
                        if dry_run:
                            logger.info(f"  [dry-run] Asset {aid}: {asset.reference_image_url} -> {s3_key}")
                        else:
                            asset.reference_image_url = s3_key
                        updated_assets += 1
                else:
                    # Check if this asset references another asset's image
                    ref = asset.reference_image_url
                    if ref and ref.startswith("/api/assets/"):
                        ref_match = re.search(r"/api/assets/([0-9a-f-]{36})/", ref)
                        if ref_match:
                            ref_aid = ref_match.group(1)
                            if ref_aid in mappings:
                                s3_key = mappings[ref_aid]
                                if dry_run:
                                    logger.info(
                                        f"  [dry-run] Asset {aid} (refs {ref_aid}): {ref} -> {s3_key}"
                                    )
                                else:
                                    asset.reference_image_url = s3_key
                                updated_assets += 1

        # Phase 3: Update contact sheet URLs
        updated_contact = 0
        for pb_id in asset_key_map:
            import uuid as _uuid
            try:
                pb_uuid = _uuid.UUID(pb_id)
            except ValueError:
                continue

            pb = await session.get(ProductionBible, pb_uuid)
            if not pb:
                continue

            cs_key = f"manifests/{pb_id}/contact_sheet.jpg"
            cs_exists = await storage.exists(cs_key)
            if cs_exists and pb.contact_sheet_url != cs_key:
                if dry_run:
                    logger.info(f"  [dry-run] PB {pb_id} contact_sheet: {pb.contact_sheet_url} -> {cs_key}")
                else:
                    pb.contact_sheet_url = cs_key
                updated_contact += 1

        if not dry_run:
            await session.commit()

        logger.info(f"DB updates: {updated_assets} asset URLs, {updated_contact} contact sheets")
        if dry_run:
            logger.info("Dry run complete — no changes written.")
        else:
            logger.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-dir", default=".temp/tmp/manifests",
                        help="Local directory with manifest backups")
    args = parser.parse_args()
    asyncio.run(main(source_dir=args.source_dir, dry_run=args.dry_run))
