"""
Database module for vidpipe.

Provides async SQLAlchemy engine with SQLite WAL mode,
session management, and schema initialization.
"""
import logging

from sqlalchemy import text

from vidpipe.db.engine import async_session, engine, get_session, shutdown
from vidpipe.db.models import Base, SceneManifest, SceneAudioManifest, AssetCleanReference

logger = logging.getLogger(__name__)


async def _run_migrations(conn) -> None:
    """Run safe ALTER TABLE migrations for new columns (idempotent)."""
    migrations = [
        "ALTER TABLE projects ADD COLUMN forked_from_id TEXT REFERENCES projects(id)",
        "ALTER TABLE video_clips ADD COLUMN source VARCHAR(20) DEFAULT 'generated'",
        "ALTER TABLE video_clips ADD COLUMN veo_submission_count INTEGER DEFAULT 0",
        "ALTER TABLE video_clips ADD COLUMN safety_regen_count INTEGER DEFAULT 0",
        "ALTER TABLE projects ADD COLUMN manifest_id TEXT REFERENCES manifests(id)",
        "ALTER TABLE projects ADD COLUMN manifest_version INTEGER",
        # Phase 5: Manifesting Engine fields
        "ALTER TABLE assets ADD COLUMN reverse_prompt TEXT",
        "ALTER TABLE assets ADD COLUMN visual_description TEXT",
        "ALTER TABLE assets ADD COLUMN detection_class VARCHAR(50)",
        "ALTER TABLE assets ADD COLUMN detection_confidence REAL",
        "ALTER TABLE assets ADD COLUMN is_face_crop INTEGER DEFAULT 0",
        "ALTER TABLE assets ADD COLUMN crop_bbox TEXT",  # JSON stored as TEXT in SQLite
        "ALTER TABLE assets ADD COLUMN face_embedding BLOB",
        "ALTER TABLE assets ADD COLUMN quality_score REAL",
        "ALTER TABLE assets ADD COLUMN source_asset_id TEXT REFERENCES assets(id)",
    ]
    for sql in migrations:
        try:
            await conn.execute(text(sql))
        except Exception:
            # Column already exists — safe to ignore
            pass


async def init_database():
    """Initialize database schema on first run."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_migrations(conn)


__all__ = [
    "Base",
    "engine",
    "async_session",
    "get_session",
    "shutdown",
    "init_database",
]
