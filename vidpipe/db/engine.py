"""
Database engine configuration for vidpipe.

Provides async SQLAlchemy engine with SQLite WAL mode,
crash-safe PRAGMA configuration, and session management.
"""
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vidpipe.config import settings


def configure_sqlite_pragmas(dbapi_conn, connection_record):
    """
    Configure SQLite PRAGMA settings for crash safety and performance.

    - WAL mode: Write-Ahead Logging for better concurrency
    - FULL synchronous: Maximum crash safety (FOUND-04 requirement)
    - Foreign keys: Enable referential integrity (Pitfall 2)
    - Busy timeout: Wait up to 5s for locks
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


# Create async engine
engine = create_async_engine(
    settings.storage.database_url,
    echo=False,
)

# Register PRAGMA configuration on connect
# CRITICAL: Use engine.sync_engine for aiosqlite compatibility
event.listens_for(engine.sync_engine, "connect")(configure_sqlite_pragmas)

# Create session factory
# CRITICAL: expire_on_commit=False prevents greenlet errors (Pitfall 1)
async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session():
    """
    Dependency injection function for async sessions.

    Yields an async session and ensures proper cleanup.
    """
    async with async_session() as session:
        yield session


async def shutdown():
    """Dispose of engine and close all connections."""
    await engine.dispose()
