"""Background processing tasks for manifest manifesting pipeline.

This module provides background task execution for the manifesting engine
with in-memory progress tracking.
"""

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

from vidpipe.db import async_session
from vidpipe.db.models import Asset, Manifest
from vidpipe.services.manifesting_engine import ManifestingEngine

logger = logging.getLogger(__name__)

# Module-level dict for in-memory progress tracking
TASK_STATUS: dict[str, dict] = {}


async def process_manifest_task(manifest_id: str) -> None:
    """Run manifesting pipeline for a manifest in background.

    Args:
        manifest_id: Manifest UUID as string
    """
    task_id = f"manifest_{manifest_id}"
    TASK_STATUS[task_id] = {
        "status": "processing",
        "current_step": "initializing",
        "progress": {},
    }

    try:
        async with async_session() as session:
            engine = ManifestingEngine(session)
            # Share progress reference so engine updates TASK_STATUS directly
            TASK_STATUS[task_id] = engine.progress
            await engine.process_manifest(uuid.UUID(manifest_id))
            TASK_STATUS[task_id]["status"] = "complete"
    except Exception as e:
        logger.error(f"Manifesting failed for {manifest_id}: {e}", exc_info=True)
        current_step = TASK_STATUS.get(task_id, {}).get("current_step", "unknown")
        TASK_STATUS[task_id] = {
            "status": "error",
            "error": str(e),
            "current_step": current_step,
        }

        # Mark manifest as ERROR in database
        try:
            async with async_session() as err_session:
                manifest = await err_session.get(Manifest, uuid.UUID(manifest_id))
                if manifest:
                    manifest.status = "ERROR"
                    await err_session.commit()
        except Exception as db_err:
            logger.error(f"Failed to mark manifest as ERROR: {db_err}")


async def extract_video_frames_task(manifest_id: str, video_path: str) -> None:
    """Extract unique frames from uploaded video and create assets.

    Args:
        manifest_id: Manifest UUID as string
        video_path: Path to uploaded video file
    """
    from vidpipe.services.video_frame_extractor import VideoFrameExtractor
    from vidpipe.config import settings

    task_id = f"extract_{manifest_id}"
    TASK_STATUS[task_id] = {
        "status": "extracting",
        "current_step": "initializing",
        "progress": {
            "candidate_frames": 0,
            "unique_frames": 0,
        },
    }

    try:
        threshold = settings.cv_analysis.video_frame_dedup_threshold
        extractor = VideoFrameExtractor(dedup_threshold=threshold)
        # Share progress reference so polling reads live updates
        TASK_STATUS[task_id] = extractor.progress

        output_dir = str(Path("tmp/manifests") / manifest_id)

        # Run extraction in a thread (cv2 + CLIP are CPU-bound)
        unique_paths, video_info = await asyncio.to_thread(
            extractor.extract_unique_frames, video_path, output_dir
        )

        # Create asset records for each unique frame
        extractor.progress["current_step"] = "saving"
        uploads_dir = Path("tmp/manifests") / manifest_id / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        async with async_session() as session:
            manifest = await session.get(Manifest, uuid.UUID(manifest_id))
            if not manifest:
                raise ValueError(f"Manifest {manifest_id} not found")

            manifest.source_video_duration = video_info["duration"]

            from sqlalchemy import func as sa_func, select

            for i, frame_path in enumerate(unique_paths):
                # Auto-generate manifest_tag
                result = await session.execute(
                    select(sa_func.count(Asset.id)).where(
                        Asset.manifest_id == uuid.UUID(manifest_id),
                        Asset.asset_type == "CHARACTER",
                    )
                )
                count = result.scalar()
                manifest_tag = f"CHAR_{count + 1:02d}"

                asset = Asset(
                    manifest_id=uuid.UUID(manifest_id),
                    asset_type="CHARACTER",
                    name=f"Frame {i + 1}",
                    manifest_tag=manifest_tag,
                    source="video_frame",
                    sort_order=i,
                )
                session.add(asset)
                await session.flush()

                # Copy frame to uploads directory
                dest = uploads_dir / f"{asset.id}_frame_{i + 1:03d}.jpg"
                shutil.copy2(frame_path, str(dest))

                asset.reference_image_url = f"/api/assets/{asset.id}/image"
                manifest.asset_count += 1

            manifest.status = "DRAFT"
            await session.commit()

        extractor.progress["status"] = "complete"
        extractor.progress["current_step"] = "complete"
        logger.info(
            f"Video extraction complete for {manifest_id}: "
            f"{len(unique_paths)} assets created"
        )

    except Exception as e:
        logger.error(
            f"Video extraction failed for {manifest_id}: {e}", exc_info=True
        )
        current_step = TASK_STATUS.get(task_id, {}).get("current_step", "unknown")
        TASK_STATUS[task_id] = {
            "status": "error",
            "error": str(e),
            "current_step": current_step,
            "progress": TASK_STATUS.get(task_id, {}).get("progress", {}),
        }

        # Mark manifest as ERROR
        try:
            async with async_session() as err_session:
                manifest = await err_session.get(Manifest, uuid.UUID(manifest_id))
                if manifest:
                    manifest.status = "ERROR"
                    await err_session.commit()
        except Exception as db_err:
            logger.error(f"Failed to mark manifest as ERROR: {db_err}")
