#!/usr/bin/env python3
"""Migrate local filesystem assets to Supabase Storage (S3).

Walks the local tmp/ directory, uploads each file to the configured S3 bucket,
then updates database path columns to relative keys.

Usage:
    python scripts/migrate_assets_to_s3.py [--dry-run] [--database-url URL] [--tmp-dir DIR]

Requires VIDPIPE_STORAGE__S3_* env vars to be set (or .env file).
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(dry_run: bool = False):
    from vidpipe.config import settings
    from vidpipe.services.storage_backend import S3StorageBackend

    if settings.storage.storage_backend != "s3":
        logger.error("Storage backend is not 's3'. Set VIDPIPE_STORAGE__STORAGE_BACKEND=s3")
        sys.exit(1)

    tmp_dir = settings.storage.tmp_dir.resolve()
    if not tmp_dir.exists():
        logger.error(f"tmp_dir does not exist: {tmp_dir}")
        sys.exit(1)

    storage = S3StorageBackend(
        endpoint=settings.storage.s3_endpoint,
        bucket=settings.storage.s3_bucket,
        service_key=settings.storage.s3_service_key,
    )

    # Collect all files to upload
    files_to_upload: list[tuple[Path, str, str]] = []  # (local_path, key, content_type)

    for root, _dirs, filenames in os.walk(tmp_dir):
        for fname in filenames:
            local_path = Path(root) / fname
            # Derive key by stripping tmp_dir prefix
            key = str(local_path.relative_to(tmp_dir))

            # Determine content type
            suffix = local_path.suffix.lower()
            content_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".txt": "text/plain",
            }
            content_type = content_types.get(suffix, "application/octet-stream")
            files_to_upload.append((local_path, key, content_type))

    logger.info(f"Found {len(files_to_upload)} files to upload from {tmp_dir}")

    if dry_run:
        for local_path, key, ct in files_to_upload[:20]:
            logger.info(f"  [DRY RUN] {key} ({ct}, {local_path.stat().st_size} bytes)")
        if len(files_to_upload) > 20:
            logger.info(f"  ... and {len(files_to_upload) - 20} more")
        return

    # Upload files
    uploaded = 0
    failed = 0
    for i, (local_path, key, content_type) in enumerate(files_to_upload):
        try:
            data = local_path.read_bytes()
            await storage.put(key, data, content_type)
            uploaded += 1
            if (i + 1) % 50 == 0:
                logger.info(f"  Uploaded {i + 1}/{len(files_to_upload)}...")
        except Exception as e:
            logger.error(f"  Failed to upload {key}: {e}")
            failed += 1

    logger.info(f"Upload complete: {uploaded} uploaded, {failed} failed")

    # Update database paths
    logger.info("Updating database paths...")
    from vidpipe.db import async_session
    from vidpipe.db.models import Keyframe, VideoClip, Scene

    tmp_prefix = str(tmp_dir) + "/"

    async with async_session() as session:
        from sqlalchemy import select, update

        # Update Keyframe.file_path
        kf_result = await session.execute(select(Keyframe).where(Keyframe.file_path.like(f"{tmp_prefix}%")))
        kf_count = 0
        for kf in kf_result.scalars().all():
            kf.file_path = kf.file_path[len(tmp_prefix):]
            kf_count += 1
        logger.info(f"  Updated {kf_count} keyframe paths")

        # Update VideoClip.local_path
        clip_result = await session.execute(select(VideoClip).where(VideoClip.local_path.like(f"{tmp_prefix}%")))
        clip_count = 0
        for clip in clip_result.scalars().all():
            clip.local_path = clip.local_path[len(tmp_prefix):]
            clip_count += 1
        logger.info(f"  Updated {clip_count} clip paths")

        # Update Scene.output_path
        scene_result = await session.execute(select(Scene).where(Scene.output_path.like(f"{tmp_prefix}%")))
        scene_count = 0
        for scene in scene_result.scalars().all():
            scene.output_path = scene.output_path[len(tmp_prefix):]
            scene_count += 1
        logger.info(f"  Updated {scene_count} scene output paths")

        await session.commit()

    logger.info("Migration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate assets to S3")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
