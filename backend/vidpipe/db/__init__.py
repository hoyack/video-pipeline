"""
Database module for vidpipe.

Provides async SQLAlchemy engine with SQLite WAL mode,
session management, and schema initialization.
"""
from vidpipe.db.engine import async_session, engine, get_session, shutdown
from vidpipe.db.models import Base


async def init_database():
    """Initialize database schema on first run."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


__all__ = [
    "Base",
    "engine",
    "async_session",
    "get_session",
    "shutdown",
    "init_database",
]
