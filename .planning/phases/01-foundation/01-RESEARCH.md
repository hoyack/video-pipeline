# Phase 1: Foundation - Research

**Researched:** 2026-02-14
**Domain:** Python async data persistence, configuration management, and filesystem artifact handling
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundational data persistence, configuration, and filesystem layers for a Python async video generation pipeline. The core stack consists of **SQLAlchemy 2.0+** with async support via **aiosqlite** for SQLite database operations, **Pydantic 2.0+** with **pydantic-settings** for type-safe configuration loading from YAML and environment variables, and **pathlib** for structured filesystem artifact management.

SQLite with WAL (Write-Ahead Logging) mode provides crash-safe local storage suitable for single-user CLI tools, eliminating the need for cloud databases or multi-user auth. The async-first architecture using Python 3.11+ asyncio patterns ensures the foundation is ready for later phases involving long-running API calls to Gemini and Veo services.

**Primary recommendation:** Use SQLAlchemy 2.0 declarative models with async sessions (`create_async_engine`, `async_sessionmaker`), enable SQLite WAL mode with `synchronous=FULL` for crash safety, configure `expire_on_commit=False` to prevent implicit lazy loading in async contexts, and use pydantic-settings with `YamlConfigSettingsSource` for configuration loading.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.0+ | Async ORM and database toolkit | Industry standard Python ORM; 2.0 has native async support, type-safe Mapped annotations, and mature ecosystem |
| aiosqlite | 0.22.1+ | Async SQLite driver | Required for async SQLite; 0.22.1+ has fixes for SQLAlchemy 2.0 worker thread management |
| Pydantic | 2.0+ | Data validation and settings | Type-safe models with runtime validation; v2 has major performance improvements and cleaner API |
| pydantic-settings | 2.12.0+ | Configuration management | Official Pydantic extension for loading config from env vars, YAML, JSON, TOML |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Alembic | 1.18.0+ | Database migrations | Schema evolution after initial release; has async template for SQLAlchemy 2.0 |
| python-dotenv | 1.0+ | .env file loading | Development environment config; complements pydantic-settings |
| edwh-uuid7 | latest | Time-ordered UUIDs | Optional: if you want time-sortable UUIDs instead of uuid4; compatible with PostgreSQL uuid7 |
| pytest-asyncio | latest | Async testing support | Testing async database code with pytest |
| pytest-async-sqlalchemy | latest | SQLAlchemy async fixtures | Provides test database fixtures with transaction rollback |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SQLAlchemy | SQLModel | SQLModel is simpler but less mature; better for FastAPI-only projects |
| aiosqlite | asyncpg + PostgreSQL | PostgreSQL offers more features but adds deployment complexity for local-first tool |
| uuid4 | uuid7 / edwh-uuid7 | uuid7 provides time-ordering which aids debugging but requires extra dependency until Python 3.14 |
| Alembic | Manual schema.sql | Migrations provide version control for schema changes but add complexity for v1 |

**Installation:**
```bash
# Core dependencies
pip install sqlalchemy[asyncio]>=2.0 aiosqlite>=0.22.1 pydantic>=2.0 pydantic-settings>=2.12.0

# Optional: migrations and testing
pip install alembic>=1.18.0 pytest-asyncio pytest-async-sqlalchemy

# Optional: time-ordered UUIDs
pip install edwh-uuid7
```

## Architecture Patterns

### Recommended Project Structure
```
vidpipe/
├── db/
│   ├── __init__.py
│   ├── engine.py              # Engine creation, session factory
│   ├── models.py              # ORM models (Project, Scene, Keyframe, VideoClip, PipelineRun)
│   └── migrations/            # Alembic migrations (optional for v1)
│       └── env.py             # Async migration environment
├── config.py                  # Pydantic settings model
├── services/
│   └── file_manager.py        # Filesystem artifact helpers
└── config.yaml                # Default configuration
```

### Pattern 1: Async Engine and Session Factory

**What:** Create a single async engine at application startup, use `async_sessionmaker` for session creation per operation.

**When to use:** All database operations in async code.

**Example:**
```python
# db/engine.py
# Source: https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

engine = create_async_engine(
    "sqlite+aiosqlite:///vidpipe.db",
    echo=True,  # Set False in production
)

async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,  # CRITICAL for async to prevent lazy loads
    class_=AsyncSession,
)

async def get_session() -> AsyncSession:
    """Dependency injection for async sessions."""
    async with async_session() as session:
        yield session

async def shutdown():
    """Clean up engine on application shutdown."""
    await engine.dispose()
```

**Critical:** Always set `expire_on_commit=False` in async contexts to prevent implicit lazy loading after commits.

### Pattern 2: Declarative Models with Mapped Annotations

**What:** Use SQLAlchemy 2.0 style declarative models with `Mapped[Type]` annotations.

**When to use:** All ORM model definitions.

**Example:**
```python
# db/models.py
# Source: https://docs.sqlalchemy.org/en/21/orm/quickstart.html
from sqlalchemy import String, Text, JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    prompt: Mapped[str] = mapped_column(Text)
    style: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50))
    style_guide: Mapped[dict] = mapped_column(JSON, nullable=True)
    storyboard_raw: Mapped[dict] = mapped_column(JSON, nullable=True)
    output_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
```

**Note:** Use `Mapped[str | None]` for nullable columns, `Mapped[str]` for NOT NULL.

### Pattern 3: SQLite Configuration with WAL and Crash Safety

**What:** Configure SQLite connection with WAL mode, synchronous=FULL, and foreign keys enabled.

**When to use:** Engine creation for production use.

**Example:**
```python
# db/engine.py
# Source: https://sqlite.org/wal.html and https://docs.sqlalchemy.org/en/21/dialects/sqlite.html
from sqlalchemy import event, create_engine
from sqlalchemy.ext.asyncio import create_async_engine

def configure_sqlite_pragmas(dbapi_conn, connection_record):
    """Set SQLite PRAGMAs for crash safety and performance."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")        # Enable WAL mode
    cursor.execute("PRAGMA synchronous=FULL")        # Maximum crash safety
    cursor.execute("PRAGMA foreign_keys=ON")         # Enable FK constraints
    cursor.execute("PRAGMA busy_timeout=5000")       # Wait 5s on locks
    cursor.close()

# For async engine
engine = create_async_engine("sqlite+aiosqlite:///vidpipe.db")

# Register event listener
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    configure_sqlite_pragmas(dbapi_conn, connection_record)
```

**Why:** WAL mode enables concurrent reads during writes. `synchronous=FULL` ensures commits survive crashes/power loss. `foreign_keys=ON` must be set per connection (default is OFF).

### Pattern 4: Pydantic Settings with YAML and Environment Variables

**What:** Use `BaseSettings` with `YamlConfigSettingsSource` to load config from YAML file with env var overrides.

**When to use:** Application configuration loading.

**Example:**
```python
# config.py
# Source: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource, PydanticBaseSettingsSource
from typing import Tuple

class DatabaseConfig(BaseSettings):
    url: str = "sqlite+aiosqlite:///vidpipe.db"
    echo: bool = False

class PipelineConfig(BaseSettings):
    max_scenes: int = 15
    retry_max_attempts: int = 5
    retry_base_delay: int = 2

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",  # Allow DATABASE__URL env var
        env_prefix="VIDPIPE_",      # Prefix for env vars
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Priority: env vars > YAML file > defaults
        return (
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            init_settings,
        )

# Usage
settings = Settings()
```

**Priority order:** Environment variables override YAML, YAML overrides defaults.

### Pattern 5: Structured Filesystem Artifacts with Pathlib

**What:** Use `pathlib.Path` for all filesystem operations, create structured directories per project.

**When to use:** Saving binary artifacts (keyframes, clips, output videos).

**Example:**
```python
# services/file_manager.py
# Source: https://docs.python.org/3/library/pathlib.html
from pathlib import Path
import uuid

class FileManager:
    def __init__(self, base_dir: str | Path = "./tmp"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_project_dir(self, project_id: uuid.UUID) -> Path:
        """Get structured directory for project artifacts."""
        project_dir = self.base_dir / str(project_id)
        project_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (project_dir / "keyframes").mkdir(exist_ok=True)
        (project_dir / "clips").mkdir(exist_ok=True)
        (project_dir / "output").mkdir(exist_ok=True)

        return project_dir

    def save_keyframe(self, project_id: uuid.UUID, scene_idx: int, position: str, data: bytes) -> Path:
        """Save keyframe image and return path."""
        project_dir = self.get_project_dir(project_id)
        filename = f"scene_{scene_idx}_{position}.png"
        filepath = project_dir / "keyframes" / filename

        filepath.write_bytes(data)  # Atomic write
        return filepath
```

**Why:** `pathlib.Path` provides cross-platform path handling, atomic file operations, and cleaner syntax than `os.path`.

### Pattern 6: Schema Initialization

**What:** Create database schema at first run using `metadata.create_all()` with async engine.

**When to use:** Application startup before first database operation.

**Example:**
```python
# db/engine.py
# Source: https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html
from sqlalchemy.ext.asyncio import create_async_engine
from db.models import Base

async def init_database():
    """Initialize database schema on first run."""
    engine = create_async_engine("sqlite+aiosqlite:///vidpipe.db")

    async with engine.begin() as conn:
        # Run sync operation in async context
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
```

**Alternative:** Use Alembic migrations for schema evolution after v1.

### Anti-Patterns to Avoid

- **Lazy loading in async code:** Never rely on lazy relationship loading. Use `selectinload()` or `joinedload()` to eagerly load relationships.
- **Forgetting `expire_on_commit=False`:** Default is True, which causes lazy loads after commit in async contexts (will fail).
- **Not enabling SQLite PRAGMAs:** Foreign keys are OFF by default; must enable on every connection.
- **Using `os.path` instead of `pathlib`:** Old-style path handling is error-prone and platform-specific.
- **Hard-coded `./tmp` paths:** Use configurable base directory from settings.
- **Not cleaning up async engine:** Always call `await engine.dispose()` on shutdown to prevent event loop warnings.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Database migrations | Custom schema versioning | Alembic | Handles version tracking, rollbacks, auto-generation; well-tested |
| Configuration loading | Custom YAML parser + env var logic | pydantic-settings | Type validation, nested models, source priority, dotenv support |
| Async session lifecycle | Manual session creation/cleanup | `async_sessionmaker` + context managers | Prevents leaks, handles errors, standard pattern |
| Path manipulation | String concatenation | pathlib.Path | Cross-platform, prevents path traversal, cleaner API |
| Connection pooling | Custom connection manager | SQLAlchemy engine pools | Handles concurrency, timeouts, recycling automatically |
| Retry logic with backoff | Custom retry loops | tenacity library | Configurable strategies, async support, well-tested |

**Key insight:** SQLAlchemy's async support and Pydantic's validation eliminate most custom boilerplate. Don't reinvent these wheels.

## Common Pitfalls

### Pitfall 1: Lazy Loading After Commit in Async Sessions

**What goes wrong:** After calling `await session.commit()`, accessing a relationship attribute triggers a lazy load, which fails in async contexts with "greenlet_spawn has not been called" error.

**Why it happens:** SQLAlchemy's default `expire_on_commit=True` marks all objects as expired after commit, causing attribute access to trigger new queries. In async, you can't issue synchronous queries.

**How to avoid:** Always configure `async_sessionmaker` with `expire_on_commit=False`:
```python
async_session = async_sessionmaker(engine, expire_on_commit=False)
```

**Warning signs:** Errors like `sqlalchemy.exc.MissingGreenlet` or `greenlet_spawn has not been called` when accessing model attributes after commit.

**Source:** [SQLAlchemy async documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)

### Pitfall 2: SQLite Foreign Keys Not Enabled

**What goes wrong:** Foreign key constraints are silently ignored; orphaned records or invalid references persist in database.

**Why it happens:** SQLite's `PRAGMA foreign_keys` defaults to OFF for backward compatibility.

**How to avoid:** Use event listener to enable on every connection:
```python
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

**Warning signs:** Able to delete parent records without cascade; referential integrity violations not caught.

**Source:** [SQLite documentation](https://www.sqlite.org/pragma.html)

### Pitfall 3: Concurrent Access Without WAL Mode

**What goes wrong:** "Database is locked" errors under concurrent reads/writes.

**Why it happens:** SQLite's default DELETE journal mode locks the entire database for writes.

**How to avoid:** Enable WAL mode which allows concurrent reads during writes:
```python
cursor.execute("PRAGMA journal_mode=WAL")
```

**Warning signs:** `sqlite3.OperationalError: database is locked` during concurrent operations.

**Source:** [SQLite WAL documentation](https://sqlite.org/wal.html)

### Pitfall 4: Not Disposing Async Engine

**What goes wrong:** Application hangs on exit or emits warnings like `RuntimeError: Event loop is closed`.

**Why it happens:** aiosqlite uses background threads; engine holds open connections that must be cleaned up.

**How to avoid:** Call `await engine.dispose()` on shutdown:
```python
async def shutdown():
    await engine.dispose()
```

**Warning signs:** Application doesn't exit cleanly; pytest warnings about unclosed connections.

**Source:** [SQLAlchemy async I/O documentation](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html)

### Pitfall 5: Path Traversal in Artifact Storage

**What goes wrong:** User-controlled project IDs could escape `tmp/` directory and write to arbitrary filesystem locations.

**Why it happens:** Direct string concatenation without validation.

**How to avoid:** Use `pathlib.Path.resolve()` and validate paths stay within base directory:
```python
project_dir = (self.base_dir / str(project_id)).resolve()
if not project_dir.is_relative_to(self.base_dir):
    raise ValueError("Invalid project path")
```

**Warning signs:** Project IDs with `..` or absolute paths work without errors.

### Pitfall 6: Wrong Synchronous Mode for Use Case

**What goes wrong:** Either poor performance (synchronous=FULL) or data loss after crashes (synchronous=NORMAL/OFF).

**Why it happens:** Misunderstanding WAL + synchronous mode interactions.

**How to avoid:** For crash safety with acceptable performance, use WAL + synchronous=FULL. For maximum performance with some durability risk, use WAL + synchronous=NORMAL.

**Spec requirement:** Use `synchronous=FULL` per FOUND-04 requirement for crash-safe operations.

**Warning signs:** Database corruption after crashes/power loss, or unexpectedly slow writes.

**Source:** [SQLite synchronous pragma](https://sqlite.org/pragma.html#pragma_synchronous)

### Pitfall 7: Nested Pydantic Models Without BaseModel

**What goes wrong:** Nested configuration sections don't load properly from YAML; values are missing or incorrect.

**Why it happens:** pydantic-settings requires nested models to inherit from `pydantic.BaseModel`, not `BaseSettings`.

**How to avoid:** Use `BaseModel` for nested config classes:
```python
from pydantic import BaseModel

class DatabaseConfig(BaseModel):  # Not BaseSettings
    url: str
    echo: bool

class Settings(BaseSettings):
    database: DatabaseConfig
```

**Warning signs:** Nested config values not loading from YAML; unexpected initialization behavior.

**Source:** [pydantic-settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

## Code Examples

Verified patterns from official sources:

### Complete Engine Setup with Event Listeners
```python
# db/engine.py
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

def configure_sqlite(dbapi_conn, connection_record):
    """Configure SQLite for crash safety and performance."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

engine = create_async_engine(
    "sqlite+aiosqlite:///vidpipe.db",
    echo=False,
)

# Register PRAGMA setter for sync engine (aiosqlite wraps it)
event.listens_for(engine.sync_engine, "connect")(configure_sqlite)

async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)
```
**Source:** [SQLAlchemy async extensions](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html), [SQLite WAL](https://sqlite.org/wal.html)

### Async CRUD Operations
```python
# Example CRUD operations
from sqlalchemy import select
from db.models import Project
from db.engine import async_session

async def create_project(prompt: str, style: str) -> Project:
    """Create new project with async session."""
    async with async_session() as session:
        project = Project(prompt=prompt, style=style, status="pending")
        session.add(project)
        await session.commit()
        # Safe to access attributes because expire_on_commit=False
        return project

async def get_project(project_id: uuid.UUID) -> Project | None:
    """Retrieve project by ID with eager loading."""
    async with async_session() as session:
        stmt = select(Project).where(Project.id == project_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
```
**Source:** [SQLAlchemy async documentation](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html)

### Configuration Loading with YAML
```python
# config.py
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",
    )

    database_url: str = "sqlite+aiosqlite:///vidpipe.db"
    tmp_dir: Path = Path("./tmp")

    @field_validator("tmp_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (env_settings, YamlConfigSettingsSource(settings_cls), init_settings)
```
**Source:** [Pydantic settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SQLAlchemy 1.4 with `declarative_base()` | SQLAlchemy 2.0 with `DeclarativeBase` and `Mapped[]` | Released Jan 2023 | Type safety, better IDE support, cleaner syntax |
| `Query` API | `select()` statement API | SQLAlchemy 2.0 | Unified query construction for Core and ORM |
| Pydantic 1.x with custom config loaders | Pydantic 2.0 with pydantic-settings | Pydantic 2.0 release Jun 2023 | 5-50x validation performance, native Settings support |
| Manual async session management | `async_sessionmaker` factory | SQLAlchemy 1.4+ | Consistent session lifecycle, easier testing |
| UUID v4 random | UUID v7 time-ordered | RFC 9562 (May 2024), Python 3.14 | Better database indexing, sortable by creation time |

**Deprecated/outdated:**
- `declarative_base()`: Use `DeclarativeBase` class for SQLAlchemy 2.0
- `Query` API (e.g., `session.query(Model)`): Use `select(Model)` statement pattern
- `expire_on_commit=True` (default) in async: Must be False for async sessions
- Pydantic 1.x `BaseSettings`: Moved to separate `pydantic-settings` package in v2
- `os.path` module: Use `pathlib.Path` for modern Python code

## Open Questions

1. **Should we use Alembic migrations for v1 or defer to v2?**
   - What we know: Alembic adds complexity but provides schema versioning
   - What's unclear: Whether schema will change during v1 development
   - Recommendation: Start with `metadata.create_all()` for simplicity. Add Alembic if schema changes become frequent. Migration to Alembic is straightforward later.

2. **UUID v4 vs UUID v7 for primary keys?**
   - What we know: UUID v7 provides time-ordering (better for indexes and debugging); Python 3.14 adds native support
   - What's unclear: Whether time-ordering is worth the extra dependency (edwh-uuid7)
   - Recommendation: Use `uuid.uuid4()` for v1 (no dependencies). Consider uuid7 if database performance or debugging benefits are needed.

3. **In-memory SQLite for tests vs file-based?**
   - What we know: In-memory is faster but can't test WAL mode or file-specific behaviors
   - What's unclear: Whether we need to test WAL-specific features
   - Recommendation: Use in-memory for unit tests, file-based for integration tests that verify crash safety.

## Sources

### Primary (HIGH confidence)
- [SQLAlchemy 2.1 Async I/O Documentation](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html) - Async patterns, session management, engine configuration
- [SQLite WAL Mode Documentation](https://sqlite.org/wal.html) - Write-ahead logging, checkpoint behavior, crash safety
- [SQLite PRAGMA Documentation](https://www.sqlite.org/pragma.html) - journal_mode, synchronous, foreign_keys settings
- [Pydantic Settings Management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - BaseSettings, YAML loading, environment variables
- [Python pathlib Documentation](https://docs.python.org/3/library/pathlib.html) - Path operations, cross-platform compatibility

### Secondary (MEDIUM confidence)
- [SQLAlchemy Complete Guide - DevToolbox](https://devtoolbox.dedyn.io/blog/sqlalchemy-complete-guide) - ORM patterns, best practices
- [Python asyncio Complete Guide - DevToolbox](https://devtoolbox.dedyn.io/blog/python-asyncio-complete-guide) - Python 3.11+ async patterns, TaskGroup
- [Pydantic Complete Guide - DevToolbox](https://devtoolbox.dedyn.io/blog/pydantic-complete-guide) - Field validators, nested models
- [Python pathlib Complete Guide - DevToolbox](https://devtoolbox.dedyn.io/blog/python-pathlib-complete-guide) - Best practices, security considerations
- [Building High-Performance Async APIs - Leapcell](https://leapcell.io/blog/building-high-performance-async-apis-with-fastapi-sqlalchemy-2-0-and-asyncpg) - FastAPI + SQLAlchemy 2.0 patterns
- [pytest-async-sqlalchemy GitHub](https://github.com/igortg/pytest-async-sqlalchemy) - Testing patterns for async SQLAlchemy

### Tertiary (LOW confidence)
- [Medium: Mastering SQLAlchemy](https://medium.com/@ramanbazhanau/mastering-sqlalchemy-a-comprehensive-guide-for-python-developers-ddb3d9f2e829) - General patterns
- [Medium: How to Load Configuration in Pydantic](https://medium.com/@wihlarkop/how-to-load-configuration-in-pydantic-3693d0ee81a3) - YAML loading examples
- [aiosqlite GitHub Issues](https://github.com/openai/codex/issues/9906) - Known issues with aiosqlite + SQLAlchemy

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are official, well-documented, and actively maintained as of Feb 2026
- Architecture: HIGH - Patterns verified against official SQLAlchemy 2.1, Pydantic 2.x, and SQLite documentation
- Pitfalls: HIGH - Common issues documented in official docs, GitHub discussions, and verified with official sources

**Research date:** 2026-02-14
**Valid until:** ~2026-03-14 (30 days; stable ecosystem with infrequent breaking changes)
