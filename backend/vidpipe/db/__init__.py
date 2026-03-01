"""
Database module for vidpipe.

Provides async SQLAlchemy engine with driver-aware configuration
(SQLite WAL mode or PostgreSQL connection pooling),
session management, and schema initialization.
"""
import logging

from sqlalchemy import text

from vidpipe.db.engine import async_session, engine, get_session, shutdown, _is_sqlite
from vidpipe.db.models import Base, ShotManifest, ShotAudioManifest, AssetCleanReference, AssetAppearance, SceneCheckpoint, Production, ProductionBible, Sequence, DEFAULT_USER_ID

logger = logging.getLogger(__name__)


async def _seed_default_user(conn) -> None:
    """Idempotent: ensure default user + settings rows exist.

    SQLite stores UUIDs as 32-char hex (no dashes) via SQLAlchemy's Uuid type.
    PostgreSQL uses native UUID type (dashed format).
    """
    import uuid as _uuid

    if _is_sqlite():
        uid = DEFAULT_USER_ID.hex  # 32-char hex, no dashes — matches SQLAlchemy Uuid type
        uid_dashed = str(DEFAULT_USER_ID)  # clean up any rows from old dashed format

        # Remove stale rows inserted with dashed UUID format
        await conn.execute(text("DELETE FROM user_settings WHERE user_id = :uid"), {"uid": uid_dashed})
        await conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": uid_dashed})
    else:
        # PostgreSQL: native UUID type uses dashed format
        uid = str(DEFAULT_USER_ID)

    row = await conn.execute(text("SELECT id FROM users WHERE id = :uid"), {"uid": uid})
    if row.fetchone() is None:
        await conn.execute(
            text("INSERT INTO users (id, name) VALUES (:uid, 'default')"),
            {"uid": uid},
        )
        sid = _uuid.uuid4().hex if _is_sqlite() else str(_uuid.uuid4())
        await conn.execute(
            text(
                "INSERT INTO user_settings (id, user_id) VALUES (:sid, :uid)"
            ),
            {"sid": sid, "uid": uid},
        )
        logger.info("Seeded default user %s", uid)


async def _run_rename_migrations(conn) -> None:
    """Run table and column RENAME migrations (idempotent).

    MUST run BEFORE create_all() so SQLAlchemy finds tables under their new names.

    Phase 16: Renames 'manifests' table to 'production_bibles' and renames all
    FK columns that referenced manifests to use production_bible_id naming.

    SQLite 3.25+ supports RENAME COLUMN. Older versions require table recreation
    but we use try/except for idempotency — if the column already has the new name,
    the rename fails silently.
    """
    # -------------------------------------------------------------------------
    # Phase 16: manifests → production_bibles table rename
    # Must run before create_all() so SA finds the table under the new name.
    # -------------------------------------------------------------------------
    try:
        await conn.execute(text("ALTER TABLE manifests RENAME TO production_bibles"))
        logger.info("Renamed table: manifests -> production_bibles")
    except Exception:
        pass  # Already renamed or doesn't exist yet (fresh DB)

    # -------------------------------------------------------------------------
    # Phase 16: FK column renames (idempotent via try/except)
    # SQLite 3.25+ and PostgreSQL both support RENAME COLUMN.
    # -------------------------------------------------------------------------
    rename_columns = [
        # production_bibles self-referential FK
        ("production_bibles", "parent_manifest_id", "parent_production_bible_id"),
        # assets.manifest_id → production_bible_id
        ("assets", "manifest_id", "production_bible_id"),
        # manifest_snapshots.manifest_id → production_bible_id
        ("manifest_snapshots", "manifest_id", "production_bible_id"),
        # scenes.manifest_id → production_bible_id
        ("scenes", "manifest_id", "production_bible_id"),
        # scenes.manifest_version → production_bible_version
        ("scenes", "manifest_version", "production_bible_version"),
    ]
    for table, old_col, new_col in rename_columns:
        try:
            await conn.execute(
                text(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}")
            )
            logger.info("Renamed column: %s.%s -> %s", table, old_col, new_col)
        except Exception:
            pass  # Already renamed, column doesn't exist, or table doesn't exist yet


async def _run_migrations(conn) -> None:
    """Run safe ALTER TABLE migrations for new columns (idempotent).

    Runs on both SQLite and PostgreSQL. PostgreSQL uses ADD COLUMN IF NOT EXISTS
    for idempotency; SQLite doesn't support IF NOT EXISTS so we catch exceptions.

    Note: create_all() only creates NEW tables — it won't add columns to
    existing tables, so these migrations are needed on both drivers.
    """
    is_sqlite = _is_sqlite()

    # Migrations use SQLite types (TEXT, INTEGER, REAL, BLOB).
    # PostgreSQL also accepts these via implicit type aliases.
    migrations = [
        "ALTER TABLE scenes ADD COLUMN forked_from_id TEXT REFERENCES scenes(id)",
        "ALTER TABLE video_clips ADD COLUMN source VARCHAR(20) DEFAULT 'generated'",
        "ALTER TABLE video_clips ADD COLUMN veo_submission_count INTEGER DEFAULT 0",
        "ALTER TABLE video_clips ADD COLUMN safety_regen_count INTEGER DEFAULT 0",
        # Phase 6: manifest → production bible association on scenes
        # Note: column may be named manifest_id (pre-16) or production_bible_id (post-16).
        # If rename migration above ran, this ADD will fail silently (already exists).
        # If starting fresh (no old DB), use the new name directly.
        "ALTER TABLE scenes ADD COLUMN production_bible_id TEXT REFERENCES production_bibles(id)",
        "ALTER TABLE scenes ADD COLUMN production_bible_version INTEGER",
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
        # Phase 9: CV Analysis Pipeline fields
        "ALTER TABLE assets ADD COLUMN clip_embedding BLOB",
        "ALTER TABLE shot_manifests ADD COLUMN cv_analysis_json TEXT",
        "ALTER TABLE shot_manifests ADD COLUMN continuity_score REAL",
        # Phase 10: Adaptive Prompt Rewriting
        "ALTER TABLE shot_manifests ADD COLUMN rewritten_keyframe_prompt TEXT",
        "ALTER TABLE shot_manifests ADD COLUMN rewritten_video_prompt TEXT",
        # Phase 11: Multi-Candidate Quality Mode
        "ALTER TABLE scenes ADD COLUMN quality_mode INTEGER DEFAULT 0",
        "ALTER TABLE scenes ADD COLUMN candidate_count INTEGER DEFAULT 1",
        # ComfyUI integration
        "ALTER TABLE user_settings ADD COLUMN comfyui_host VARCHAR(500)",
        "ALTER TABLE user_settings ADD COLUMN comfyui_api_key TEXT",
        "ALTER TABLE user_settings ADD COLUMN comfyui_cost_per_second REAL",
        # Phase 13: LLM Provider Abstraction & Ollama Integration
        "ALTER TABLE user_settings ADD COLUMN ollama_use_cloud INTEGER DEFAULT 0",
        "ALTER TABLE user_settings ADD COLUMN ollama_api_key TEXT",
        "ALTER TABLE user_settings ADD COLUMN ollama_endpoint VARCHAR(500)",
        "ALTER TABLE user_settings ADD COLUMN ollama_models TEXT",  # JSON stored as TEXT in SQLite
        "ALTER TABLE scenes ADD COLUMN vision_model VARCHAR(100)",
        "ALTER TABLE scenes ADD COLUMN deleted_at TIMESTAMP",
        # Selective stage execution
        "ALTER TABLE scenes ADD COLUMN run_through VARCHAR(20)",
        # Scene title
        "ALTER TABLE scenes ADD COLUMN title VARCHAR(200)",
        # PipeSVN: version control
        "ALTER TABLE scenes ADD COLUMN head_sha VARCHAR(40)",
        "ALTER TABLE video_clips ADD COLUMN prompt_used TEXT",
        "ALTER TABLE shots ADD COLUMN generation_status VARCHAR(32)",
        # Display preferences
        "ALTER TABLE user_settings ADD COLUMN show_cost INTEGER DEFAULT 1",
        # Productions
        "ALTER TABLE scenes ADD COLUMN production_id TEXT REFERENCES productions(id)",
        # ElevenLabs configuration
        "ALTER TABLE user_settings ADD COLUMN elevenlabs_api_key TEXT",
        # Phase 16: Sequence grouping
        "ALTER TABLE scenes ADD COLUMN sequence_id TEXT REFERENCES sequences(id)",
        "ALTER TABLE scenes ADD COLUMN scene_order INTEGER DEFAULT 0",
    ]
    for sql in migrations:
        try:
            if not is_sqlite:
                # PostgreSQL supports IF NOT EXISTS for idempotent column adds
                sql = sql.replace("ADD COLUMN ", "ADD COLUMN IF NOT EXISTS ")
            await conn.execute(text(sql))
        except Exception:
            # Column already exists (SQLite) or other safe-to-ignore error
            pass


async def init_database():
    """Initialize database schema on first run."""
    async with engine.begin() as conn:
        # Phase 16: Table/column renames MUST run before create_all() so SQLAlchemy
        # finds tables under their new names (production_bibles, production_bible_id).
        await _run_rename_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)
        await _run_migrations(conn)
        await _seed_default_user(conn)


__all__ = [
    "Base",
    "engine",
    "async_session",
    "get_session",
    "shutdown",
    "init_database",
    "AssetAppearance",
]
