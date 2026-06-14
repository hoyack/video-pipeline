"""Startup recovery for scenes orphaned by a server restart.

The pipeline runs as in-process background tasks. When the server restarts
mid-run (e.g. a container rebuild), scenes are left stranded in an in-flight
status with no task driving them — the UI shows them generating forever and
the production never finishes. Every stage checkpoints to the database, so
these scenes are safely resumable: ComfyUI jobs are re-polled via the
persisted ``comfyui:`` operation id, completed stages are skipped.
"""

import logging
import uuid

from sqlalchemy import select

from vidpipe.db import async_session
from vidpipe.db.models import Scene
from vidpipe.orchestrator.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Statuses that imply a pipeline task should be driving the scene
IN_FLIGHT_STATUSES = (
    "pending",
    "storyboarding",
    "keyframing",
    "generating_video",
    "video_gen",
    "stitching",
)


async def find_orphaned_scene_ids() -> list[uuid.UUID]:
    """Return ids of scenes stranded in in-flight statuses."""
    async with async_session() as session:
        result = await session.execute(
            select(Scene.id)
            .where(Scene.status.in_(IN_FLIGHT_STATUSES))
            .where(Scene.deleted_at.is_(None))
            .order_by(Scene.created_at)
        )
        return [row[0] for row in result]


async def recover_orphaned_scenes() -> int:
    """Resume every orphaned scene, sequentially.

    Runs as a background task after startup. Failures are persisted per-scene
    by the orchestrator (scene.status="failed" + error_message) and do not
    stop recovery of the remaining scenes.

    Returns:
        Number of scenes for which recovery was attempted.
    """
    scene_ids = await find_orphaned_scene_ids()
    if not scene_ids:
        logger.info("Startup recovery: no orphaned in-flight scenes found")
        return 0

    logger.warning(
        "Startup recovery: resuming %d scene(s) orphaned by a restart: %s",
        len(scene_ids), [str(s) for s in scene_ids],
    )
    for scene_id in scene_ids:
        async with async_session() as session:
            try:
                await run_pipeline(session, scene_id)
                logger.info("Startup recovery: scene %s resumed to completion", scene_id)
            except Exception as e:
                # Error state already persisted by the orchestrator
                logger.error(
                    "Startup recovery: scene %s failed: %s: %s",
                    scene_id, type(e).__name__, e,
                )
    return len(scene_ids)
