"""
Database module for vidpipe.

Provides async SQLAlchemy engine with SQLite WAL mode,
session management, and schema initialization.
"""
import logging

from sqlalchemy import text

from vidpipe.db.engine import async_session, engine, get_session, shutdown
from vidpipe.db.models import Base, SceneManifest, SceneAudioManifest, AssetCleanReference, AssetAppearance, DEFAULT_USER_ID

logger = logging.getLogger(__name__)


async def _seed_default_user(conn) -> None:
    """Idempotent: ensure default user + settings rows exist.

    Uses .hex (no dashes) to match SQLAlchemy's Uuid storage format in SQLite.
    """
    import uuid as _uuid
    uid = DEFAULT_USER_ID.hex  # 32-char hex, no dashes — matches SQLAlchemy Uuid type
    uid_dashed = str(DEFAULT_USER_ID)  # clean up any rows from old dashed format

    # Remove stale rows inserted with dashed UUID format
    await conn.execute(text("DELETE FROM user_settings WHERE user_id = :uid"), {"uid": uid_dashed})
    await conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": uid_dashed})

    row = await conn.execute(text("SELECT id FROM users WHERE id = :uid"), {"uid": uid})
    if row.fetchone() is None:
        await conn.execute(
            text("INSERT INTO users (id, name) VALUES (:uid, 'default')"),
            {"uid": uid},
        )
        await conn.execute(
            text(
                "INSERT INTO user_settings (id, user_id) VALUES (:sid, :uid)"
            ),
            {"sid": _uuid.uuid4().hex, "uid": uid},
        )
        logger.info("Seeded default user %s", uid)


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
        # Phase 9: CV Analysis Pipeline fields
        "ALTER TABLE assets ADD COLUMN clip_embedding BLOB",
        "ALTER TABLE scene_manifests ADD COLUMN cv_analysis_json TEXT",
        "ALTER TABLE scene_manifests ADD COLUMN continuity_score REAL",
        # Phase 10: Adaptive Prompt Rewriting
        "ALTER TABLE scene_manifests ADD COLUMN rewritten_keyframe_prompt TEXT",
        "ALTER TABLE scene_manifests ADD COLUMN rewritten_video_prompt TEXT",
        # Phase 11: Multi-Candidate Quality Mode
        "ALTER TABLE projects ADD COLUMN quality_mode INTEGER DEFAULT 0",
        "ALTER TABLE projects ADD COLUMN candidate_count INTEGER DEFAULT 1",
        # ComfyUI integration
        "ALTER TABLE user_settings ADD COLUMN comfyui_host VARCHAR(500)",
        "ALTER TABLE user_settings ADD COLUMN comfyui_api_key TEXT",
        "ALTER TABLE user_settings ADD COLUMN comfyui_cost_per_second REAL",
        # Phase 13: LLM Provider Abstraction & Ollama Integration
        "ALTER TABLE user_settings ADD COLUMN ollama_use_cloud INTEGER DEFAULT 0",
        "ALTER TABLE user_settings ADD COLUMN ollama_api_key TEXT",
        "ALTER TABLE user_settings ADD COLUMN ollama_endpoint VARCHAR(500)",
        "ALTER TABLE user_settings ADD COLUMN ollama_models TEXT",  # JSON stored as TEXT in SQLite
        "ALTER TABLE projects ADD COLUMN vision_model VARCHAR(100)",
        "ALTER TABLE projects ADD COLUMN deleted_at TIMESTAMP",
        # Selective stage execution
        "ALTER TABLE projects ADD COLUMN run_through VARCHAR(20)",
        # Project title
        "ALTER TABLE projects ADD COLUMN title VARCHAR(200)",
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
