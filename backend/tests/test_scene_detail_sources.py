"""Tests for scene detail keyframe source reporting."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.api import routes
from vidpipe.db.models import Base, Keyframe, Scene, Shot


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_scene_detail_reports_keyframe_sources(session_factory, monkeypatch):
    async with session_factory() as session:
        scene = Scene(
            title="Continuity Scene",
            prompt="A two-shot continuity test.",
            style="cinematic",
            aspect_ratio="16:9",
            target_clip_duration=5,
            target_shot_count=2,
            seed=42,
            status="draft",
        )
        session.add(scene)
        await session.flush()

        shot_0 = Shot(
            scene_id=scene.id,
            shot_index=0,
            shot_description="Opening shot.",
            start_frame_prompt="start 0",
            end_frame_prompt="end 0",
            video_motion_prompt="move 0",
            status="keyframes_done",
        )
        shot_1 = Shot(
            scene_id=scene.id,
            shot_index=1,
            shot_description="Follow-up shot.",
            start_frame_prompt="start 1",
            end_frame_prompt="end 1",
            video_motion_prompt="move 1",
            status="keyframes_done",
        )
        session.add_all([shot_0, shot_1])
        await session.flush()

        session.add_all([
            Keyframe(
                shot_id=shot_0.id,
                position="start",
                prompt_used="start 0",
                file_path="/tmp/shot0-start.png",
                mime_type="image/png",
                source="generated",
                verification_status="passed",
                verification_attempts=2,
                verification_summary="start ok",
            ),
            Keyframe(
                shot_id=shot_0.id,
                position="end",
                prompt_used="end 0",
                file_path="/tmp/shot0-end.png",
                mime_type="image/png",
                source="generated",
                verification_status="passed",
                verification_attempts=1,
                verification_summary="end ok",
            ),
            Keyframe(
                shot_id=shot_1.id,
                position="start",
                prompt_used="start 1",
                file_path="/tmp/shot1-start.png",
                mime_type="image/png",
                source="inherited",
                verification_status="inherited",
                verification_attempts=0,
                verification_summary="Inherited from previous shot end keyframe.",
            ),
            Keyframe(
                shot_id=shot_1.id,
                position="end",
                prompt_used="end 1",
                file_path="/tmp/shot1-end.png",
                mime_type="image/png",
                source="generated",
                verification_status="accepted_with_warnings",
                verification_attempts=3,
                verification_summary="best_effort_fallback accepted attempt 2/3",
            ),
        ])
        await session.commit()

        monkeypatch.setattr(routes, "async_session", session_factory)
        detail = await routes.get_scene_detail(scene.id)

        assert detail.shots[0].start_keyframe_source == "generated"
        assert detail.shots[0].end_keyframe_source == "generated"
        assert detail.shots[0].start_verification_status == "passed"
        assert detail.shots[0].start_verification_attempts == 2
        assert detail.shots[0].end_verification_summary == "end ok"
        assert detail.shots[1].start_keyframe_source == "inherited"
        assert detail.shots[1].end_keyframe_source == "generated"
        assert detail.shots[1].start_verification_status == "inherited"
        assert detail.shots[1].end_verification_status == "accepted_with_warnings"
        assert detail.shots[1].end_verification_attempts == 3


@pytest.mark.asyncio
async def test_all_phases_failure_marks_scene_failed(session_factory, monkeypatch):
    async with session_factory() as session:
        scene = Scene(
            title="Failure Scene",
            prompt="A scene that fails during keyframes.",
            style="cinematic",
            aspect_ratio="16:9",
            target_clip_duration=5,
            target_shot_count=1,
            seed=42,
            status="draft",
        )
        session.add(scene)
        await session.commit()

    async def fake_storyboard(scene_id, text_model_override=None, *, _emit_complete=True):
        del scene_id, text_model_override, _emit_complete
        return None

    async def fake_keyframes(
        scene_id,
        image_model_override=None,
        text_model_override=None,
        *,
        _emit_complete=True,
    ):
        del scene_id, image_model_override, text_model_override, _emit_complete
        raise RuntimeError("verification boom")

    monkeypatch.setattr(routes, "async_session", session_factory)
    monkeypatch.setattr(routes, "_run_storyboard_regeneration", fake_storyboard)
    monkeypatch.setattr(routes, "_run_keyframes_regeneration", fake_keyframes)

    await routes._run_all_phases_regeneration(scene.id, run_through="keyframes")

    async with session_factory() as verify_session:
        loaded = await verify_session.get(Scene, scene.id)
        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.error_message == "verification boom"
