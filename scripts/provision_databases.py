#!/usr/bin/env python3
"""
Provision vidpipe databases on a PostgreSQL instance (e.g. Supabase).

Creates: vidpipe_master, vidpipe_worker1, vidpipe_worker2, vidpipe_worker3

Usage:
    python scripts/provision_databases.py

Environment variables:
    SUPABASE_HOST     (default: localhost)
    SUPABASE_PORT     (default: 5432)
    SUPABASE_USER     (default: postgres)
    SUPABASE_PASSWORD (required)
    SUPABASE_TENANT   (optional, for Supavisor pooler URLs)
"""
import asyncio
import os
import sys

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg is required. Install with: pip install asyncpg>=0.30.0")
    sys.exit(1)


DATABASES = [
    "vidpipe_master",
    "vidpipe_worker1",
    "vidpipe_worker2",
    "vidpipe_worker3",
]


async def main():
    host = os.environ.get("SUPABASE_HOST", "localhost")
    port = int(os.environ.get("SUPABASE_PORT", "5432"))
    user = os.environ.get("SUPABASE_USER", "postgres")
    password = os.environ.get("SUPABASE_PASSWORD")
    tenant = os.environ.get("SUPABASE_TENANT", "")

    if not password:
        print("ERROR: SUPABASE_PASSWORD environment variable is required.")
        sys.exit(1)

    print(f"Connecting to PostgreSQL at {host}:{port} as {user}...")

    # Connect to the default 'postgres' database to create new databases
    conn = await asyncpg.connect(
        host=host, port=port, user=user, password=password, database="postgres"
    )

    try:
        for db_name in DATABASES:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            if exists:
                print(f"  {db_name}: already exists (skipping)")
            else:
                # CREATE DATABASE cannot run inside a transaction
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                print(f"  {db_name}: created")
    finally:
        await conn.close()

    # Verify connectivity to each database
    print("\nVerifying connectivity...")
    for db_name in DATABASES:
        try:
            test_conn = await asyncpg.connect(
                host=host, port=port, user=user, password=password, database=db_name
            )
            version = await test_conn.fetchval("SELECT version()")
            await test_conn.close()
            print(f"  {db_name}: OK ({version.split(',')[0]})")
        except Exception as e:
            print(f"  {db_name}: FAILED ({e})")

    # Print connection URLs for .env files
    print("\n--- Connection URLs for .env files ---")
    print("# Direct connection (port 5432):")
    for db_name in DATABASES:
        print(
            f"# VIDPIPE_STORAGE__DATABASE_URL=postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        )

    if tenant:
        pooler_port = 6543
        print(f"\n# Supavisor pooler connection (port {pooler_port}):")
        for db_name in DATABASES:
            print(
                f"# VIDPIPE_STORAGE__DATABASE_URL=postgresql+asyncpg://{user}.{tenant}:{password}@{host}:{pooler_port}/{db_name}"
            )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
