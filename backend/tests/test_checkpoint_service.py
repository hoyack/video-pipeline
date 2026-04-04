"""Tests for checkpoint snapshots preserving structured storyboard metadata."""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vidpipe.db.models import Base, Scene, Shot
from vidpipe.services.checkpoint_service import build_snapshot, restore_from_snapshot


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
async def test_snapshot_and_restore_preserve_structured_storyboard_fields(session_factory):
    async with session_factory() as session:
        scene = Scene(
            title="Checkpoint Scene",
            prompt="A scene with structured metadata.",
            style="cinematic",
            aspect_ratio="16:9",
            target_clip_duration=6,
            target_shot_count=2,
            dynamic_shot_count=True,
            total_duration=12,
            text_model="gemini-3-flash-preview",
            image_model="gemini-2.5-flash-image",
            video_model="wan-2.2-i2v",
            audio_enabled=False,
            seed=1234,
            script_analysis={"story_beats": [{"index": 0}, {"index": 1}]},
            screenplay_context={"slugline": "INT. PENTHOUSE - NIGHT"},
            style_guide={"visual_style": "cinematic"},
            storyboard_raw={"shots": [{"shot_index": 0}]},
            status="draft",
        )
        session.add(scene)
        await session.flush()

        shot = Shot(
            id=uuid.uuid4(),
            scene_id=scene.id,
            shot_index=0,
            shot_description="Opening shot",
            start_frame_prompt="start",
            end_frame_prompt="end",
            video_motion_prompt="move",
            transition_notes="cut",
            status="pending",
            generation_status="generating_text",
            characters_present=["@BRANDON_CROSS", "@DRACULA_NORMAL"],
            beat_index=1,
            narrative_intent="Set the confrontation.",
            emotional_weight=7,
        )
        session.add(shot)
        await session.commit()

        snapshot = await build_snapshot(session, scene)
        assert snapshot["scene"]["dynamic_shot_count"] is True
        assert snapshot["scene"]["script_analysis"] == {
            "story_beats": [{"index": 0}, {"index": 1}]
        }
        assert snapshot["scene"]["style_guide"] == {"visual_style": "cinematic"}
        assert snapshot["shots"][0]["characters_present"] == [
            "@BRANDON_CROSS",
            "@DRACULA_NORMAL",
        ]
        assert snapshot["shots"][0]["generation_status"] == "generating_text"

        scene.dynamic_shot_count = False
        scene.script_analysis = None
        scene.style_guide = None
        scene.storyboard_raw = None
        shot.characters_present = None
        shot.beat_index = None
        shot.narrative_intent = None
        shot.emotional_weight = None
        shot.generation_status = None
        await session.commit()

        await restore_from_snapshot(session, scene, snapshot)
        await session.commit()

        refreshed_scene = await session.get(Scene, scene.id)
        assert refreshed_scene.dynamic_shot_count is True
        assert refreshed_scene.script_analysis == {
            "story_beats": [{"index": 0}, {"index": 1}]
        }
        assert refreshed_scene.style_guide == {"visual_style": "cinematic"}
        assert refreshed_scene.storyboard_raw == {"shots": [{"shot_index": 0}]}

        restored_shots = (
            await session.execute(
                select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
            )
        ).scalars().all()
        assert len(restored_shots) == 1
        assert restored_shots[0].characters_present == [
            "@BRANDON_CROSS",
            "@DRACULA_NORMAL",
        ]
        assert restored_shots[0].beat_index == 1
        assert restored_shots[0].narrative_intent == "Set the confrontation."
        assert restored_shots[0].emotional_weight == 7
        assert restored_shots[0].generation_status == "generating_text"
