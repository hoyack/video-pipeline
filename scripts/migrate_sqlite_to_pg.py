#!/usr/bin/env python3
"""
Migrate vidpipe data from SQLite to PostgreSQL.

Reads all tables from a SQLite database, converts UUID columns from
32-char hex (SQLAlchemy SQLite format) to dashed format (PostgreSQL native UUID),
and bulk-inserts data into PostgreSQL in FK dependency order.

Usage:
    python scripts/migrate_sqlite_to_pg.py \
        --sqlite "sqlite+aiosqlite:///vidpipe.db" \
        --pg "postgresql+asyncpg://postgres:pw@localhost:5432/vidpipe_master"

Handles:
    - UUID hex → dashed conversion
    - bytes (BLOB → BYTEA automatic via asyncpg)
    - JSON (transparent)
    - Booleans (0/1 → true/false)
"""
import argparse
import asyncio
import json
import re
import sys
import uuid

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

# Import vidpipe models so create_all() uses the full ORM schema
sys.path.insert(0, "backend")
from vidpipe.db.models import Base  # noqa: E402


# Tables in FK dependency order (parents before children)
TABLE_ORDER = [
    "productions",
    "users",
    "user_settings",
    "manifests",
    "scenes",
    "manifest_snapshots",
    "shots",
    "assets",
    "asset_clean_references",
    "asset_appearances",
    "keyframes",
    "video_clips",
    "generation_candidates",
    "pipeline_runs",
    "shot_manifests",
    "shot_audio_manifests",
    "scene_checkpoints",
]

# Columns known to contain UUID values (32-char hex in SQLite)
UUID_COLUMNS = {
    "id",
    "scene_id",
    "shot_id",
    "manifest_id",
    "asset_id",
    "user_id",
    "parent_manifest_id",
    "forked_from_id",
    "source_asset_id",
    "production_id",
    "inherited_from_asset",
    "inherited_from_scene",
}

# 32-char hex pattern for UUID detection
HEX_UUID_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def hex_to_dashed_uuid(value: str) -> str:
    """Convert 32-char hex UUID to dashed format."""
    if value and isinstance(value, str) and HEX_UUID_RE.match(value):
        return str(uuid.UUID(value))
    return value


def convert_row(row: dict, column_types: dict, json_text_columns: set) -> dict:
    """Convert a row's values for PostgreSQL compatibility."""
    converted = {}
    for col, val in row.items():
        if val is None:
            converted[col] = val
        elif col in UUID_COLUMNS and isinstance(val, str):
            converted[col] = hex_to_dashed_uuid(val)
        elif isinstance(val, int) and column_types.get(col) == "BOOLEAN":
            converted[col] = bool(val)
        elif col in json_text_columns and isinstance(val, str):
            # JSON stored as TEXT in SQLite — parse to native Python for PG JSON column
            try:
                converted[col] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                converted[col] = val
        else:
            converted[col] = val
    return converted


async def migrate(sqlite_url: str, pg_url: str, dry_run: bool = False):
    """Run the migration from SQLite to PostgreSQL."""

    # --- Read from SQLite (sync engine for simplicity) ---
    sync_sqlite_url = sqlite_url.replace("sqlite+aiosqlite:", "sqlite:")
    sqlite_engine = create_engine(sync_sqlite_url)
    sqlite_meta = MetaData()
    sqlite_meta.reflect(bind=sqlite_engine)

    sqlite_tables = set(sqlite_meta.tables.keys())
    print(f"SQLite tables found: {sorted(sqlite_tables)}")

    # Build column type map from SQLite inspector
    sqlite_inspector = inspect(sqlite_engine)

    # --- Create schema on PostgreSQL ---
    # Disable prepared statement cache for pooler compatibility (Supavisor/PgBouncer)
    pg_engine = create_async_engine(
        pg_url, connect_args={"statement_cache_size": 0}
    )

    print("\nCreating PostgreSQL schema from ORM models...")
    async with pg_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Schema created.")

    # Build ORM table reference map for typed inserts
    orm_tables = {t.name: t for t in Base.metadata.sorted_tables}

    # --- Migrate data table by table ---
    total_rows = 0
    for table_name in TABLE_ORDER:
        if table_name not in sqlite_tables:
            print(f"\n  {table_name}: not in SQLite (skipping)")
            continue

        # Get column types for boolean conversion
        columns_info = sqlite_inspector.get_columns(table_name)
        column_types = {}
        for col_info in columns_info:
            col_type = str(col_info["type"]).upper()
            if "BOOL" in col_type or col_info.get("name") in (
                "is_face_crop", "is_inherited", "quality_mode", "audio_enabled",
                "has_dialogue", "has_music", "is_primary", "is_selected",
                "show_cost", "ollama_use_cloud",
            ):
                column_types[col_info["name"]] = "BOOLEAN"

        # Detect JSON columns stored as TEXT in SQLite but JSON in ORM
        json_text_columns = set()
        if table_name in orm_tables:
            orm_table = orm_tables[table_name]
            for col in orm_table.columns:
                if str(col.type) == "JSON":
                    # Check if SQLite stores this as TEXT
                    sqlite_col_info = {c["name"]: c for c in columns_info}
                    if col.name in sqlite_col_info:
                        sqlite_type = str(sqlite_col_info[col.name]["type"]).upper()
                        if "TEXT" in sqlite_type or "VARCHAR" in sqlite_type:
                            json_text_columns.add(col.name)

        # Read all rows
        table = sqlite_meta.tables[table_name]
        with sqlite_engine.connect() as sqlite_conn:
            result = sqlite_conn.execute(table.select())
            rows = [dict(row._mapping) for row in result]

        if not rows:
            print(f"\n  {table_name}: 0 rows (empty)")
            continue

        # Convert rows
        converted_rows = [convert_row(row, column_types, json_text_columns) for row in rows]

        print(f"\n  {table_name}: {len(rows)} rows", end="")

        if dry_run:
            print(" (dry run — skipped)")
            continue

        # Use ORM table for typed insert (handles JSON, UUID, etc.)
        if table_name in orm_tables:
            orm_table = orm_tables[table_name]
            # Filter columns to only those that exist in both SQLite and ORM
            orm_col_names = {c.name for c in orm_table.columns}
            filtered_rows = [
                {k: v for k, v in row.items() if k in orm_col_names}
                for row in converted_rows
            ]
            insert_stmt = orm_table.insert().prefix_with("OR IGNORE") if False else orm_table.insert()
            async with pg_engine.begin() as conn:
                # Use executemany-style via insert().values() for proper type handling
                for row in filtered_rows:
                    try:
                        await conn.execute(insert_stmt.values(**row))
                    except Exception as e:
                        if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                            continue
                        raise
        else:
            # Fallback: raw text SQL for tables not in ORM
            columns = list(converted_rows[0].keys())
            placeholders = ", ".join(f":{col}" for col in columns)
            col_list = ", ".join(f'"{col}"' for col in columns)
            insert_sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'
            async with pg_engine.begin() as conn:
                await conn.execute(text(insert_sql), converted_rows)

        print(f" -> inserted")
        total_rows += len(rows)

    # --- Verify row counts ---
    if not dry_run:
        print("\n--- Verification ---")
        mismatches = 0
        async with pg_engine.connect() as pg_conn:
            for table_name in TABLE_ORDER:
                if table_name not in sqlite_tables:
                    continue

                # SQLite count
                with sqlite_engine.connect() as sqlite_conn:
                    sqlite_count = sqlite_conn.execute(
                        text(f'SELECT COUNT(*) FROM "{table_name}"')
                    ).scalar()

                # PostgreSQL count
                pg_count = (
                    await pg_conn.execute(
                        text(f'SELECT COUNT(*) FROM "{table_name}"')
                    )
                ).scalar()

                status = "OK" if sqlite_count == pg_count else "MISMATCH"
                if status == "MISMATCH":
                    mismatches += 1
                print(f"  {table_name}: SQLite={sqlite_count} PG={pg_count} [{status}]")

        if mismatches:
            print(f"\nWARNING: {mismatches} table(s) have row count mismatches!")
        else:
            print(f"\nAll tables match. {total_rows} total rows migrated.")

    sqlite_engine.dispose()
    await pg_engine.dispose()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Migrate vidpipe from SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite",
        required=True,
        help="SQLite connection URL (e.g. sqlite+aiosqlite:///vidpipe.db)",
    )
    parser.add_argument(
        "--pg",
        required=True,
        help="PostgreSQL connection URL (e.g. postgresql+asyncpg://user:pw@host:5432/db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and convert data but don't write to PostgreSQL",
    )
    args = parser.parse_args()
    asyncio.run(migrate(args.sqlite, args.pg, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
