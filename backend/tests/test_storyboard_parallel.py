"""Tests for the parallel manifest-aware storyboard pipeline."""

import asyncio
import re
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vidpipe.db.models import Asset, Base, ProductionBible, Scene, Shot, ShotAudioManifest, ShotManifest
from vidpipe.pipeline.storyboard import _character_bible_block, generate_storyboard
from vidpipe.schemas.screenwriter_agent import ScriptAnalysis, ShotBreakdown
from vidpipe.schemas.storyboard_enhanced import ShotManifestPackageOutput, ShotPromptPackageOutput
from vidpipe.services.event_bus import event_bus
from vidpipe.services.llm.base import LLMAdapter


def _extract_shot_index(prompt: str) -> int:
    match = re.search(r'"shot_index"\s*:\s*(\d+)', prompt)
    if not match:
        raise AssertionError(f"shot_index missing from prompt: {prompt[:200]}")
    return int(match.group(1))


class FakeStoryboardAdapter(LLMAdapter):
    """Deterministic adapter for storyboard pipeline tests."""

    def __init__(
        self,
        *,
        fail_prompt_shots: set[int] | None = None,
        prompt_delays: dict[int, float] | None = None,
    ) -> None:
        self.fail_prompt_shots = fail_prompt_shots or set()
        self.prompt_delays = prompt_delays or {}
        self.manifest_calls: list[int] = []
        self.prompt_calls: list[int] = []

    async def generate_text(
        self,
        prompt: str,
        schema,
        *,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        max_retries: int = 3,
    ):
        if schema is ScriptAnalysis:
            return ScriptAnalysis.model_validate({
                "narrative_summary": "A lone character moves through a city at dawn.",
                "tone": "cinematic",
                "genre": "drama",
                "pacing": "steady",
                "characters": [
                    {
                        "tag": "CHAR_01",
                        "role": "protagonist",
                        "screen_time_hint": "heavy",
                        "first_appearance_beat": 0,
                    }
                ],
                "settings": ["City street"],
                "story_beats": [
                    {
                        "index": 0,
                        "description": "The character appears at dawn.",
                        "characters_involved": ["CHAR_01"],
                        "emotional_tone": "awe",
                        "is_climax": False,
                    },
                    {
                        "index": 1,
                        "description": "The city wakes up around them.",
                        "characters_involved": [],
                        "emotional_tone": "neutral",
                        "is_climax": False,
                    },
                ],
                "emotional_arc": "quiet rise",
            })

        if schema is ShotBreakdown:
            return ShotBreakdown.model_validate({
                "shots": [
                    {
                        "shot_index": 0,
                        "beat_index": 0,
                        "narrative_intent": "Introduce the character at dawn.",
                        "characters_present": ["CHAR_01"],
                        "setting": "Rooftop",
                        "time_of_day": "dawn",
                        "emotional_weight": 6,
                        "duration_hint": 4,
                        "transition_from_previous": None,
                    },
                    {
                        "shot_index": 1,
                        "beat_index": 1,
                        "narrative_intent": "Show the city without characters.",
                        "characters_present": [],
                        "setting": "Street",
                        "time_of_day": "morning",
                        "emotional_weight": 3,
                        "duration_hint": 4,
                        "transition_from_previous": "Cut from the rooftop to the street below.",
                    },
                ],
                "arc_coverage": "Both beats are covered.",
                "uncovered_beats": [],
            })

        shot_index = _extract_shot_index(prompt)

        if schema is ShotManifestPackageOutput:
            self.manifest_calls.append(shot_index)
            if shot_index == 0:
                return ShotManifestPackageOutput.model_validate({
                    "shot_index": 0,
                    "shot_description": "A determined figure stands on a rooftop at dawn.",
                    "key_details": ["dawn", "rooftop", "city skyline"],
                    "shot_manifest": {
                        "shot_index": 0,
                        "composition": {
                            "shot_type": "wide_shot",
                            "camera_movement": "slow_pan_left",
                            "focal_point": "CHAR_99",
                        },
                        "placements": [
                            {
                                "asset_tag": "CHAR_99",
                                "role": "subject",
                                "position": "center",
                                "action": "looking over the city",
                                "expression": "determined",
                                "wardrobe_note": "blue coat",
                            }
                        ],
                        "continuity_notes": "Establish the hero before moving into the city.",
                        "new_asset_declarations": [
                            {
                                "tag": "CHAR_99",
                                "type": "CHARACTER",
                                "description": "Duplicate invented character",
                            }
                        ],
                    },
                    "audio_manifest": {
                        "shot_index": 0,
                        "dialogue_lines": [
                            {
                                "speaker_tag": "CHAR_99",
                                "speaker_name": "Hero",
                                "line": "The city is waking up.",
                                "delivery": "softly",
                                "timing": "mid-shot",
                                "emphasis": None,
                            }
                        ],
                        "sfx": [],
                        "ambient": {
                            "base_layer": "early morning wind",
                            "environmental": "distant traffic",
                            "weather": None,
                            "time_cues": "dawn birds",
                        },
                        "music": None,
                        "audio_continuity": None,
                    },
                })

            return ShotManifestPackageOutput.model_validate({
                "shot_index": 1,
                "shot_description": "A quiet street begins to fill with movement.",
                "key_details": ["street", "dawn light", "waking city"],
                "shot_manifest": {
                    "shot_index": 1,
                    "composition": {
                        "shot_type": "establishing",
                        "camera_movement": "static",
                        "focal_point": "ENV_01",
                    },
                    "placements": [
                        {
                            "asset_tag": "ENV_01",
                            "role": "environment",
                            "position": "background",
                            "action": "sunlight spilling between buildings",
                            "expression": None,
                            "wardrobe_note": None,
                        }
                    ],
                    "continuity_notes": "Carry the rooftop dawn light into the city below.",
                    "new_asset_declarations": [],
                },
                "audio_manifest": {
                    "shot_index": 1,
                    "dialogue_lines": [],
                    "sfx": [],
                    "ambient": {
                        "base_layer": "city ambience",
                        "environmental": "subway rumble",
                        "weather": None,
                        "time_cues": None,
                    },
                    "music": None,
                    "audio_continuity": None,
                },
            })

        if schema is ShotPromptPackageOutput:
            self.prompt_calls.append(shot_index)
            await asyncio.sleep(self.prompt_delays.get(shot_index, 0))
            if shot_index in self.fail_prompt_shots:
                raise RuntimeError(f"prompt generation failed for shot {shot_index}")
            return ShotPromptPackageOutput.model_validate({
                "shot_index": shot_index,
                "start_frame_prompt": f"A comic book rendering of shot {shot_index} at dawn.",
                "end_frame_prompt": f"A comic book rendering of shot {shot_index} as the moment progresses.",
                "video_motion_prompt": f"Slow camera movement for shot {shot_index}.",
                "transition_notes": f"Transition out of shot {shot_index}.",
            })

        raise AssertionError(f"Unexpected schema {schema}")

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        schema,
        *,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_retries: int = 3,
    ):
        raise NotImplementedError


class DoubleAtPromptAdapter(FakeStoryboardAdapter):
    """Adapter variant that returns duplicated @tags in prompt fields."""

    async def generate_text(
        self,
        prompt: str,
        schema,
        *,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        max_retries: int = 3,
    ):
        if schema is ShotPromptPackageOutput:
            shot_index = _extract_shot_index(prompt)
            return ShotPromptPackageOutput.model_validate({
                "shot_index": shot_index,
                "start_frame_prompt": (
                    f"A comic book rendering of @@BRANDON_CROSS_BUSINESS_CASUAL in shot {shot_index}."
                ),
                "end_frame_prompt": (
                    f"A comic book rendering of @@BRANDON_CROSS_BUSINESS_CASUAL at the end of shot {shot_index}."
                ),
                "video_motion_prompt": (
                    f"Camera pushes toward @@BRANDON_CROSS_BUSINESS_CASUAL in shot {shot_index}."
                ),
                "transition_notes": "Cut away from @@BRANDON_CROSS_BUSINESS_CASUAL.",
            })
        return await super().generate_text(
            prompt,
            schema,
            temperature=temperature,
            system_prompt=system_prompt,
            max_retries=max_retries,
        )


@pytest_asyncio.fixture
async def session(tmp_path: Path):
    db_path = tmp_path / "storyboard_parallel.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as db_session:
        yield db_session

    await engine.dispose()


async def _seed_scene(
    session: AsyncSession,
    *,
    with_existing_shots: bool = False,
) -> Scene:
    bible = ProductionBible(name="Test Bible", status="DRAFT")
    session.add(bible)
    await session.flush()

    session.add_all([
        Asset(
            production_bible_id=bible.id,
            asset_type="CHARACTER",
            name="Hero",
            manifest_tag="CHAR_01",
            reverse_prompt="athletic adult with short dark hair wearing a blue coat",
            description="Hero asset",
            source="uploaded",
        ),
        Asset(
            production_bible_id=bible.id,
            asset_type="ENVIRONMENT",
            name="City Street",
            manifest_tag="ENV_01",
            reverse_prompt="quiet city street at dawn with long shadows",
            description="Environment asset",
            source="uploaded",
        ),
    ])

    scene = Scene(
        title="Parallel Storyboard",
        prompt="At dawn, the hero surveys the city before the streets wake up.",
        style="comic_book",
        aspect_ratio="16:9",
        target_clip_duration=4,
        target_shot_count=2,
        production_bible_id=bible.id,
        status="storyboarding",
    )
    session.add(scene)
    await session.flush()

    if with_existing_shots:
        session.add_all([
            Shot(
                scene_id=scene.id,
                shot_index=0,
                shot_description="User-provided rooftop opening.",
                start_frame_prompt="User start prompt.",
                end_frame_prompt="User end prompt.",
                video_motion_prompt="User motion prompt.",
                transition_notes="User transition.",
                status="pending",
            ),
            Shot(
                scene_id=scene.id,
                shot_index=1,
                shot_description="",
                start_frame_prompt="",
                end_frame_prompt="",
                video_motion_prompt="",
                transition_notes="",
                status="pending",
            ),
        ])
        scene.storyboard_raw = {
            "style_guide": {
                "visual_style": "comic book",
                "color_palette": "warm dawn tones",
                "camera_style": "cinematic",
            },
            "characters": [
                {
                    "name": "Hero",
                    "physical_description": "athletic adult with short dark hair",
                    "clothing_description": "blue coat",
                }
            ],
            "shots": [
                {
                    "shot_index": 0,
                    "shot_description": "User-provided rooftop opening.",
                    "key_details": ["rooftop"],
                    "start_frame_prompt": "User start prompt.",
                    "end_frame_prompt": "User end prompt.",
                    "video_motion_prompt": "User motion prompt.",
                    "transition_notes": "User transition.",
                }
            ],
        }

    await session.commit()
    return scene


def test_character_bible_block_does_not_double_prefix_existing_at_tags():
    block = _character_bible_block(
        [
            {
                "tag": "@BRANDON_CROSS_BUSINESS_CASUAL",
                "name": "Brandon Cross",
                "physical_description": "sharp features and close-cropped dark hair",
                "clothing_description": "a charcoal gray suit",
            }
        ],
        ["@BRANDON_CROSS_BUSINESS_CASUAL"],
    )

    assert "@@BRANDON_CROSS_BUSINESS_CASUAL" not in block
    assert "- @BRANDON_CROSS_BUSINESS_CASUAL:" in block


@pytest.mark.asyncio
async def test_manifest_parallel_storyboard_persists_and_emits_progress(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = await _seed_scene(session)
    adapter = FakeStoryboardAdapter()
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        event_bus,
        "emit",
        lambda scene_id, event_type, **payload: events.append((event_type, payload)),
    )

    await generate_storyboard(session, scene, text_adapter=adapter)
    await session.refresh(scene)

    shots = list((await session.execute(
        select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
    )).scalars())
    manifests = list((await session.execute(
        select(ShotManifest).where(ShotManifest.scene_id == scene.id).order_by(ShotManifest.shot_index)
    )).scalars())
    audio_manifests = list((await session.execute(
        select(ShotAudioManifest).where(ShotAudioManifest.scene_id == scene.id).order_by(ShotAudioManifest.shot_index)
    )).scalars())

    assert scene.status == "keyframing"
    assert len(shots) == 2
    assert len(manifests) == 2
    assert len(audio_manifests) == 2
    assert manifests[0].asset_tags == ["CHAR_01"]
    assert audio_manifests[0].speaker_tags == ["CHAR_01"]
    assert scene.storyboard_raw is not None
    assert len(scene.storyboard_raw["shots"]) == 2
    assert scene.storyboard_raw["characters"][0]["name"] == "Hero"
    assert adapter.manifest_calls == [0, 1]

    event_types = [event_type for event_type, _ in events]
    assert "phase_progress" in event_types
    assert event_types.count("shot_text_ready") == 2
    assert event_types[-2:] == ["phase_completed", "refresh"]
    assert events[0] == (
        "phase_started",
        {
            "phase": "storyboard",
            "total_shots": 2,
            "message": "Starting storyboard generation for 2 shot(s)",
        },
    )
    assert any(
        event_type == "shot_status"
        and payload["message"] == "Shot 1: generating manifest and audio plan"
        for event_type, payload in events
    )
    assert any(
        event_type == "shot_status"
        and payload["message"] == "Shot 2: writing storyboard prompts"
        for event_type, payload in events
    )
    assert any(
        event_type == "task_log"
        and payload["source"] == "screenwriter.analysis.prompt"
        and payload["kind"] == "prompt"
        and "Analyze the script and identify" in payload["detail"]
        for event_type, payload in events
    )
    assert any(
        event_type == "task_log"
        and payload["source"] == "storyboard.manifest.prompt"
        and payload["shot_index"] == 0
        and payload["kind"] == "prompt"
        and "Return ONLY this shot's package." in payload["detail"]
        for event_type, payload in events
    )
    assert any(
        event_type == "task_log"
        and payload["source"] == "storyboard.prompts.response"
        and payload["shot_index"] == 1
        and payload["kind"] == "response"
        and "\"transition_notes\": \"Transition out of shot 1.\"" in payload["detail"]
        for event_type, payload in events
    )
    assert any(
        event_type == "shot_text_ready"
        and payload["message"] == "Shot 1 storyboard text ready"
        for event_type, payload in events
    )
    assert events[-2] == (
        "phase_completed",
        {
            "phase": "storyboard",
            "message": "Storyboard generation complete",
        },
    )


@pytest.mark.asyncio
async def test_manifest_parallel_storyboard_preserves_completed_shots_on_failure(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = await _seed_scene(session)
    adapter = FakeStoryboardAdapter(
        fail_prompt_shots={1},
        prompt_delays={0: 0.0, 1: 0.05},
    )
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        event_bus,
        "emit",
        lambda scene_id, event_type, **payload: events.append((event_type, payload)),
    )

    with pytest.raises(RuntimeError, match="Storyboard generation failed for shot\\(s\\): 2"):
        await generate_storyboard(session, scene, text_adapter=adapter)

    await session.refresh(scene)
    shots = list((await session.execute(
        select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
    )).scalars())

    assert scene.status == "failed"
    assert scene.error_message == "Storyboard generation failed for shot(s): 2"
    assert len(shots) == 1
    assert shots[0].shot_index == 0
    assert shots[0].start_frame_prompt == "A comic book rendering of shot 0 at dawn."
    assert any(event_type == "shot_text_ready" and payload["shot_index"] == 0 for event_type, payload in events)


@pytest.mark.asyncio
async def test_manifest_parallel_storyboard_gap_fill_only_generates_empty_shots(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = await _seed_scene(session, with_existing_shots=True)
    adapter = FakeStoryboardAdapter()
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        event_bus,
        "emit",
        lambda scene_id, event_type, **payload: events.append((event_type, payload)),
    )

    await generate_storyboard(session, scene, text_adapter=adapter)
    await session.refresh(scene)

    shots = list((await session.execute(
        select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
    )).scalars())

    assert adapter.manifest_calls == [1]
    assert shots[0].shot_description == "User-provided rooftop opening."
    assert shots[0].start_frame_prompt == "User start prompt."
    assert shots[1].shot_description == "A quiet street begins to fill with movement."
    assert shots[1].start_frame_prompt == "A comic book rendering of shot 1 at dawn."
    assert len(scene.storyboard_raw["shots"]) == 2
    assert any(event_type == "shot_text_ready" and payload["shot_index"] == 1 for event_type, payload in events)


@pytest.mark.asyncio
async def test_manifest_parallel_storyboard_sanitizes_duplicate_at_tags_in_prompt_fields(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = await _seed_scene(session)
    adapter = DoubleAtPromptAdapter()

    monkeypatch.setattr(event_bus, "emit", lambda *_args, **_kwargs: None)

    await generate_storyboard(session, scene, text_adapter=adapter)
    await session.refresh(scene)

    shots = list((await session.execute(
        select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
    )).scalars())

    for shot in shots:
        assert "@@BRANDON_CROSS_BUSINESS_CASUAL" not in shot.start_frame_prompt
        assert "@@BRANDON_CROSS_BUSINESS_CASUAL" not in shot.end_frame_prompt
        assert "@@BRANDON_CROSS_BUSINESS_CASUAL" not in shot.video_motion_prompt
        assert "@@BRANDON_CROSS_BUSINESS_CASUAL" not in shot.transition_notes
        assert "@BRANDON_CROSS_BUSINESS_CASUAL" in shot.start_frame_prompt

    storyboard_shots = scene.storyboard_raw["shots"]
    assert all("@@BRANDON_CROSS_BUSINESS_CASUAL" not in shot["start_frame_prompt"] for shot in storyboard_shots)
    assert all("@@BRANDON_CROSS_BUSINESS_CASUAL" not in shot["end_frame_prompt"] for shot in storyboard_shots)
