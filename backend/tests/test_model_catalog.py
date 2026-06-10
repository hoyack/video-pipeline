"""Tests for model catalog compatibility and UI-visible smoke scenes."""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.api import routes
from vidpipe.db import _canonicalize_legacy_model_ids
from vidpipe.db.models import Base, DEFAULT_USER_ID, Scene, Shot, User, UserSettings
from vidpipe.services.model_catalog import canonical_model_id
from vidpipe.services.vertex_client import location_for_model


@pytest_asyncio.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield engine, factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(test_db):
    _, factory = test_db
    yield factory


def test_legacy_google_models_are_mapped_to_supported_ids():
    assert canonical_model_id("gemini-3-pro-preview") == "gemini-3.1-pro-preview"
    assert canonical_model_id("veo-3.1-generate-preview") == "veo-3.1-generate-001"
    assert canonical_model_id("veo-3.1-fast-generate-preview") == "veo-3.1-fast-generate-001"
    assert canonical_model_id("wan-2.2-ref-i2v") == "wan-2.2-i2v"


def test_current_global_model_routing():
    assert location_for_model("gemini-3.1-pro-preview") == "global"
    assert location_for_model("gemini-3.5-flash") == "global"
    assert location_for_model("gemini-3-pro-preview") == "global"
    assert location_for_model("veo-3.1-fast-generate-001") == "us-central1"


def test_catalog_excludes_known_bad_preview_ids_from_new_validation():
    assert "gemini-3-pro-preview" not in routes.ALLOWED_TEXT_MODELS
    assert "gemini-3.1-pro-preview" in routes.ALLOWED_TEXT_MODELS
    assert "gemini-3.5-flash" in routes.ALLOWED_TEXT_MODELS
    assert "veo-3.1-generate-preview" not in routes.ALLOWED_VIDEO_MODELS
    assert "veo-3.1-fast-generate-preview" not in routes.ALLOWED_VIDEO_MODELS
    assert "veo-3.1-generate-001" in routes.ALLOWED_VIDEO_MODELS
    assert "veo-3.1-fast-generate-001" in routes.ALLOWED_VIDEO_MODELS


@pytest.mark.asyncio
async def test_create_draft_scene_with_legacy_models_lands_in_scene_list(
    session_factory,
    monkeypatch,
):
    monkeypatch.setattr(routes, "async_session", session_factory)

    response = await routes.create_draft_scene(
        routes.CreateSceneRequest(
            title="UI Smoke Test Scene",
            prompt="A deterministic draft scene that should appear in the dashboard.",
            style="cinematic",
            aspect_ratio="16:9",
            clip_duration=4,
            shot_count=2,
            text_model="gemini-3-pro-preview",
            image_model="gemini-2.5-flash-image",
            video_model="veo-3.1-fast-generate-preview",
            vision_model="gemini-3-pro-preview",
        )
    )

    assert response.status == "draft"
    assert response.shot_count == 2

    async with session_factory() as session:
        scene = await session.get(Scene, uuid.UUID(response.scene_id))
        assert scene is not None
        assert scene.text_model == "gemini-3.1-pro-preview"
        assert scene.video_model == "veo-3.1-fast-generate-001"
        assert scene.vision_model == "gemini-3.1-pro-preview"

    listed = await routes.list_scenes(page=1, per_page=12, view="cards")
    assert listed.total == 1
    assert listed.items[0].scene_id == response.scene_id
    assert listed.items[0].title == "UI Smoke Test Scene"
    assert listed.items[0].status == "draft"
    assert listed.items[0].text_model == "gemini-3.1-pro-preview"
    assert listed.items[0].video_model == "veo-3.1-fast-generate-001"

    async with session_factory() as session:
        shots = [
            shot
            for shot in (
                await session.execute(
                    select(Shot).where(Shot.scene_id == uuid.UUID(response.scene_id))
                )
            ).scalars()
        ]
    assert len(shots) == 2


@pytest.mark.asyncio
async def test_legacy_model_migration_updates_persisted_scene_and_settings(test_db):
    engine, session_factory = test_db

    async with session_factory() as session:
        session.add(User(id=DEFAULT_USER_ID, name="default"))
        session.add(
            UserSettings(
                user_id=DEFAULT_USER_ID,
                enabled_text_models=["gemini-3-pro-preview", "gemini-3.1-pro-preview"],
                enabled_image_models=["gemini-2.5-flash-image"],
                enabled_video_models=[
                    "veo-3.1-fast-generate-preview",
                    "veo-3.1-fast-generate-001",
                    "wan-2.2-ref-i2v",
                ],
                default_text_model="gemini-3-pro-preview",
                default_video_model="veo-3.1-generate-preview",
                default_vision_model="gemini-3-pro-preview",
            )
        )
        scene = Scene(
            title="Legacy Persisted Scene",
            prompt="Legacy row",
            style="cinematic",
            aspect_ratio="16:9",
            target_clip_duration=6,
            target_shot_count=1,
            total_duration=6,
            text_model="gemini-3-pro-preview",
            image_model="gemini-2.5-flash-image",
            video_model="veo-3.1-fast-generate-preview",
            vision_model="gemini-3-pro-preview",
            status="draft",
        )
        session.add(scene)
        await session.commit()

    async with engine.begin() as conn:
        await _canonicalize_legacy_model_ids(conn)

    async with session_factory() as session:
        scene = (await session.execute(select(Scene))).scalar_one()
        assert scene.text_model == "gemini-3.1-pro-preview"
        assert scene.video_model == "veo-3.1-fast-generate-001"
        assert scene.vision_model == "gemini-3.1-pro-preview"

        settings = (await session.execute(select(UserSettings))).scalar_one()
        assert settings.enabled_text_models == ["gemini-3.1-pro-preview"]
        assert settings.enabled_video_models == [
            "veo-3.1-fast-generate-001",
            "wan-2.2-i2v",
        ]
        assert settings.default_text_model == "gemini-3.1-pro-preview"
        assert settings.default_video_model == "veo-3.1-generate-001"
        assert settings.default_vision_model == "gemini-3.1-pro-preview"
