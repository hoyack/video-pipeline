"""Tests for cast binding identity_type migration/backfill behavior."""

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.db import _backfill_cast_binding_identity_types
from vidpipe.db.models import Actor, Base, CastBinding, ProductionBible


@pytest_asyncio.fixture
async def engine_and_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield engine, factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_identity_type_backfill_sets_empty_values_to_human(engine_and_factory):
    engine, factory = engine_and_factory

    async with factory() as session:
        bible = ProductionBible(name="Identity Bible")
        actor = Actor(name="Brandon Cross")
        session.add_all([bible, actor])
        await session.flush()

        binding = CastBinding(
            production_bible_id=bible.id,
            actor_id=actor.id,
            tag="BRANDON_CROSS",
            character_name="Brandon Cross",
            role="LEAD",
        )
        session.add(binding)
        await session.commit()

        await session.execute(
            text("UPDATE cast_bindings SET identity_type = '' WHERE id = :id"),
            {"id": str(binding.id)},
        )
        await session.commit()

    async with engine.begin() as conn:
        await _backfill_cast_binding_identity_types(conn)

    async with factory() as session:
        refreshed = await session.get(CastBinding, binding.id)
        assert refreshed.identity_type == "HUMAN"
