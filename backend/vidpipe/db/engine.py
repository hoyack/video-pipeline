"""
Database engine configuration for vidpipe.

Supports both SQLite (aiosqlite) and PostgreSQL (asyncpg) backends.
Driver is auto-detected from the configured database URL.
"""
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vidpipe.config import settings


def _is_sqlite() -> bool:
    """Check if the configured database URL is SQLite."""
    return settings.storage.database_url.startswith("sqlite")


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


# Build engine kwargs based on driver
_engine_kwargs: dict = {"echo": False}

if not _is_sqlite():
    # PostgreSQL connection pool tuning
    _engine_kwargs.update(
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        # Disable asyncpg prepared statement cache — required for
        # connection poolers (Supavisor, PgBouncer) in transaction mode
        connect_args={"statement_cache_size": 0},
    )

# Create async engine
engine = create_async_engine(settings.storage.database_url, **_engine_kwargs)

# Register PRAGMA configuration on connect — SQLite only
if _is_sqlite():
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
