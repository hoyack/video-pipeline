"""
Database module for vidpipe.

Provides async SQLAlchemy engine with driver-aware configuration
(SQLite WAL mode or PostgreSQL connection pooling),
session management, and schema initialization.
"""
import logging

from sqlalchemy import text

from vidpipe.db.engine import async_session, engine, get_session, shutdown, _is_sqlite
from vidpipe.db.models import (
    Base, ShotManifest, ShotAudioManifest, AssetCleanReference, AssetAppearance,
    SceneCheckpoint, Production, ProductionBible, Sequence, Screenplay, DEFAULT_USER_ID,
    Character, Wardrobe, VoiceProfile, Set, SonicIdentity, Prop, ScoreTheme, SFXItem,
    Actor, ActorRef, ActorVoiceProfile, ActorWardrobePreset,
    LibrarySet, LibrarySetRef, LibrarySonicIdentity,
    LibraryProp, LibraryPropRef, SoundAsset,
    CastBinding, SetBinding, PropBinding, SoundBinding,
)

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
    # Uses SAVEPOINT on PostgreSQL so a failed ALTER doesn't abort the transaction.
    # -------------------------------------------------------------------------
    is_pg = not _is_sqlite()
    if is_pg:
        await conn.execute(text("SAVEPOINT rename_table_sp"))
    try:
        await conn.execute(text("ALTER TABLE manifests RENAME TO production_bibles"))
        logger.info("Renamed table: manifests -> production_bibles")
        if is_pg:
            await conn.execute(text("RELEASE SAVEPOINT rename_table_sp"))
    except Exception:
        if is_pg:
            await conn.execute(text("ROLLBACK TO SAVEPOINT rename_table_sp"))
        # Already renamed or doesn't exist yet (fresh DB)

    # -------------------------------------------------------------------------
    # Phase 16: FK column renames (idempotent via SAVEPOINT + try/except)
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
        sp_name = f"rename_{table}_{old_col}_sp"
        if is_pg:
            await conn.execute(text(f"SAVEPOINT {sp_name}"))
        try:
            await conn.execute(
                text(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}")
            )
            logger.info("Renamed column: %s.%s -> %s", table, old_col, new_col)
            if is_pg:
                await conn.execute(text(f"RELEASE SAVEPOINT {sp_name}"))
        except Exception:
            if is_pg:
                await conn.execute(text(f"ROLLBACK TO SAVEPOINT {sp_name}"))
            # Already renamed, column doesn't exist, or table doesn't exist yet


async def _run_migrations(conn) -> None:
    """Run safe ALTER TABLE migrations for new columns (idempotent).

    Runs on both SQLite and PostgreSQL. PostgreSQL uses ADD COLUMN IF NOT EXISTS
    for idempotency; SQLite doesn't support IF NOT EXISTS so we catch exceptions.

    Note: create_all() only creates NEW tables — it won't add columns to
    existing tables, so these migrations are needed on both drivers.
    """
    is_sqlite = _is_sqlite()
    uuid_type = "TEXT" if is_sqlite else "UUID"

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
        # GCP service account JSON (full credential file for Vertex AI)
        "ALTER TABLE user_settings ADD COLUMN gcp_service_account_json TEXT",
        "ALTER TABLE user_settings ADD COLUMN default_voice_id VARCHAR(200)",
        "ALTER TABLE user_settings ADD COLUMN default_vision_model VARCHAR(100)",
        # Phase 16: Sequence grouping
        # Note: SQLite uses TEXT for UUIDs, PostgreSQL uses UUID type.
        # The correct type is set below based on driver detection.
        "ALTER TABLE scenes ADD COLUMN sequence_id {uuid_type} REFERENCES sequences(id)",
        "ALTER TABLE scenes ADD COLUMN scene_order INTEGER DEFAULT 0",
        # Phase 17: score_theme_id FK on scenes for Director agent compatibility
        "ALTER TABLE scenes ADD COLUMN score_theme_id {uuid_type} REFERENCES score_themes(id)",
        # Phase 18: Screenplay linkage columns on scenes
        "ALTER TABLE scenes ADD COLUMN screenplay_breakdown_index INTEGER",
        "ALTER TABLE scenes ADD COLUMN screenplay_context TEXT",
        # Phase 18: Unique index on screenplays.production_id for existing DBs
        # (create_all() handles this for new DBs via unique=True on the column)
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_screenplays_production_id ON screenplays(production_id)",
        # Phase 19: Direct Production → Production Bible FK
        "ALTER TABLE productions ADD COLUMN production_bible_id {uuid_type} REFERENCES production_bibles(id)",
        # Phase 22: Promotion tracking columns on existing Phase 17 tables
        "ALTER TABLE characters ADD COLUMN promoted_to_actor_id {uuid_type} REFERENCES actors(id)",
        "ALTER TABLE sets ADD COLUMN promoted_to_library_set_id {uuid_type} REFERENCES library_sets(id)",
        "ALTER TABLE props ADD COLUMN promoted_to_library_prop_id {uuid_type} REFERENCES library_props(id)",
        "ALTER TABLE score_themes ADD COLUMN promoted_to_sound_asset_id {uuid_type} REFERENCES sound_assets(id)",
        "ALTER TABLE sfx_items ADD COLUMN promoted_to_sound_asset_id {uuid_type} REFERENCES sound_assets(id)",
        # Phase 25: LoRA training infrastructure
        "ALTER TABLE actors ADD COLUMN lora_url TEXT",
        "ALTER TABLE actors ADD COLUMN lora_trained_at TIMESTAMP",
        "ALTER TABLE actors ADD COLUMN lora_training_status VARCHAR(20)",
        "ALTER TABLE actors ADD COLUMN lora_training_job_id VARCHAR(200)",
        "ALTER TABLE user_settings ADD COLUMN replicate_api_token TEXT",
        "ALTER TABLE user_settings ADD COLUMN replicate_username VARCHAR(200)",
        # Phase 27: Cache ArcFace embeddings on ActorRef for identity verification
        # BLOB for SQLite, BYTEA for PostgreSQL (BLOB is not a valid PG type)
        "ALTER TABLE actor_refs ADD COLUMN face_embedding {blob_type}",
        # Screenwriter Agent: script analysis and shot-level character assignment
        "ALTER TABLE scenes ADD COLUMN script_analysis TEXT",
        "ALTER TABLE shots ADD COLUMN characters_present TEXT",
        "ALTER TABLE shots ADD COLUMN beat_index INTEGER",
        "ALTER TABLE shots ADD COLUMN narrative_intent TEXT",
        "ALTER TABLE shots ADD COLUMN emotional_weight REAL",
        # Dynamic shot count: let screenwriter decide shot count vs user-fixed
        "ALTER TABLE scenes ADD COLUMN dynamic_shot_count BOOLEAN DEFAULT false",
        # Cast binding identity handling for character grounding policy
        "ALTER TABLE cast_bindings ADD COLUMN identity_type VARCHAR(20) DEFAULT 'HUMAN'",
        # Keyframe verification metadata
        "ALTER TABLE keyframes ADD COLUMN verification_status VARCHAR(64)",
        "ALTER TABLE keyframes ADD COLUMN verification_attempts INTEGER DEFAULT 0",
        "ALTER TABLE keyframes ADD COLUMN verification_summary TEXT",
    ]
    blob_type = "BLOB" if is_sqlite else "BYTEA"

    for i, raw_sql in enumerate(migrations):
        if "{" in raw_sql:
            sql = raw_sql.format(uuid_type=uuid_type, blob_type=blob_type)
        else:
            sql = raw_sql
        sp = f"mig_sp_{i}"
        try:
            if not is_sqlite:
                pg_sql = sql.replace("ADD COLUMN ", "ADD COLUMN IF NOT EXISTS ")
                await conn.execute(text(f"SAVEPOINT {sp}"))
                await conn.execute(text(pg_sql))
                await conn.execute(text(f"RELEASE SAVEPOINT {sp}"))
            else:
                await conn.execute(text(sql))
        except Exception as e:
            if not is_sqlite:
                logger.warning("Migration rollback: %s — %s", sql[:60], e)
                await conn.execute(text(f"ROLLBACK TO SAVEPOINT {sp}"))
            # Column already exists (SQLite) or other safe-to-ignore error

    if not is_sqlite:
        await _promote_json_text_columns(conn)
    await _backfill_cast_binding_identity_types(conn)
    await _widen_keyframe_verification_status_column(conn)


async def _backfill_cast_binding_identity_types(conn) -> None:
    """Ensure legacy cast bindings have an explicit identity_type."""
    savepoint = "cast_binding_identity_type_backfill"
    try:
        if not _is_sqlite():
            await conn.execute(text(f"SAVEPOINT {savepoint}"))
        await conn.execute(
            text(
                """
                UPDATE cast_bindings
                SET identity_type = 'HUMAN'
                WHERE identity_type IS NULL
                   OR TRIM(identity_type) = ''
                """
            )
        )
        if not _is_sqlite():
            await conn.execute(text(f"RELEASE SAVEPOINT {savepoint}"))
    except Exception as e:
        logger.warning("Cast binding identity_type backfill rollback — %s", e)
        if not _is_sqlite():
            await conn.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint}"))


async def _widen_keyframe_verification_status_column(conn) -> None:
    """Allow additive verification states longer than the legacy 20-char limit."""
    if _is_sqlite():
        return

    savepoint = "keyframe_verification_status_widen"
    try:
        await conn.execute(text(f"SAVEPOINT {savepoint}"))
        await conn.execute(
            text(
                """
                ALTER TABLE keyframes
                ALTER COLUMN verification_status TYPE VARCHAR(64)
                """
            )
        )
        await conn.execute(text(f"RELEASE SAVEPOINT {savepoint}"))
    except Exception as e:
        logger.warning("Keyframe verification_status widen rollback — %s", e)
        await conn.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint}"))


async def _promote_json_text_columns(conn) -> None:
    """Convert legacy TEXT JSON columns to JSONB on PostgreSQL."""
    conversions = [
        ("scenes", "script_analysis"),
        ("shots", "characters_present"),
    ]

    for table_name, column_name in conversions:
        result = await conn.execute(
            text(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = :table_name
                  AND column_name = :column_name
                """
            ),
            {"table_name": table_name, "column_name": column_name},
        )
        data_type = result.scalar_one_or_none()
        if data_type != "text":
            continue

        savepoint = f"json_fix_{table_name}_{column_name}"
        try:
            await conn.execute(text(f"SAVEPOINT {savepoint}"))
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {table_name}
                    ALTER COLUMN {column_name}
                    TYPE JSONB
                    USING CASE
                        WHEN {column_name} IS NULL OR btrim({column_name}) = '' THEN NULL
                        ELSE {column_name}::jsonb
                    END
                    """
                )
            )
            await conn.execute(text(f"RELEASE SAVEPOINT {savepoint}"))
            logger.info(
                "Migrated %s.%s from TEXT to JSONB",
                table_name,
                column_name,
            )
        except Exception as e:
            logger.warning(
                "JSON column promotion rollback: %s.%s — %s",
                table_name,
                column_name,
                e,
            )
            await conn.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint}"))


async def _enable_rls(conn) -> None:
    """Enable Row Level Security on tables containing secrets (PostgreSQL only).

    RLS with no policies blocks all access via the Supabase anon/PostgREST key.
    The backend connects via the postgres service role which bypasses RLS.
    Idempotent — ENABLE ROW LEVEL SECURITY is a no-op if already enabled.
    """
    if _is_sqlite():
        return

    tables = ["user_settings"]
    for table in tables:
        sp = f"rls_{table}_sp"
        try:
            await conn.execute(text(f"SAVEPOINT {sp}"))
            await conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
            await conn.execute(text(f"RELEASE SAVEPOINT {sp}"))
            logger.info("Enabled RLS on %s", table)
        except Exception as e:
            await conn.execute(text(f"ROLLBACK TO SAVEPOINT {sp}"))
            logger.debug("RLS on %s: %s", table, e)


async def init_database():
    """Initialize database schema on first run."""
    async with engine.begin() as conn:
        # Phase 16: Table/column renames MUST run before create_all() so SQLAlchemy
        # finds tables under their new names (production_bibles, production_bible_id).
        await _run_rename_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)
        await _run_migrations(conn)
        await _seed_default_user(conn)
        await _enable_rls(conn)


__all__ = [
    "Base",
    "engine",
    "async_session",
    "get_session",
    "shutdown",
    "init_database",
    "AssetAppearance",
]
