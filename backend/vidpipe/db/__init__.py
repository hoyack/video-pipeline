"""
Database module for vidpipe.

Provides async SQLAlchemy engine with SQLite WAL mode,
session management, and schema initialization.
"""
import logging

from sqlalchemy import text

from vidpipe.db.engine import async_session, engine, get_session, shutdown
from vidpipe.db.models import Base

logger = logging.getLogger(__name__)


async def _run_migrations(conn) -> None:
    """Run safe ALTER TABLE migrations for new columns (idempotent)."""
    migrations = [
        "ALTER TABLE projects ADD COLUMN forked_from_id TEXT REFERENCES projects(id)",
        "ALTER TABLE video_clips ADD COLUMN source VARCHAR(20) DEFAULT 'generated'",
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
