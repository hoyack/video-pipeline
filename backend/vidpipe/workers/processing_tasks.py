"""Background processing tasks for manifest manifesting pipeline.

This module provides background task execution for the manifesting engine
with in-memory progress tracking.
"""

import logging
import uuid

from vidpipe.db import async_session
from vidpipe.db.models import Manifest
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
