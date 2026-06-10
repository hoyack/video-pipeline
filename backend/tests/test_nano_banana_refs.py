"""Tests for Nano Banana multi-character reference assembly."""

import json
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vidpipe.db.models import (
    Actor,
    ActorRef,
    ActorWardrobePreset,
    Base,
    CastBinding,
    CastLook,
    LibrarySet,
    LibrarySetRef,
    ProductionBible,
    SetBinding,
    Shot,
)
from vidpipe.pipeline.keyframes import (
    _GeneratedKeyframeAttempt,
    _CharacterCropPlan,
    _CharacterCropSelection,
    _CharacterVisionVerificationResult,
    _HumanFaceVerificationResult,
    _KeyframeVerificationReport,
    _NANO_BANANA_MAX_REFERENCE_IMAGES,
    _ReferenceCandidate,
    _VisionVerificationReport,
    _apply_identity_policy_to_reference_candidates,
    _assemble_nano_banana_reference_context,
    _build_best_effort_fallback_result,
    _build_best_effort_detail,
    _passes_partial_visibility_human_check,
    _build_retry_reference_candidates,
    _select_best_effort_attempt,
    _verify_generated_keyframe,
)
from vidpipe.services.file_manager import FileManager
from vidpipe.services.ref_prequalification import QualifiedRef
from vidpipe.services.tag_resolver import resolve_tags_with_assets


def _write_ref(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode("utf-8"))
    return str(path)


async def _create_character_binding(
    session: AsyncSession,
    *,
    bible_id,
    actor_name: str,
    binding_tag: str,
    ref_urls: list[str],
    base_ref_urls: list[str] | None = None,
    look_tag: str | None = None,
    look_is_default: bool = False,
    identity_type: str = "HUMAN",
) -> None:
    actor = Actor(
        name=actor_name,
        base_appearance_prompt=f"{actor_name} signature face and wardrobe details.",
    )
    session.add(actor)
    await session.flush()

    binding = CastBinding(
        production_bible_id=bible_id,
        actor_id=actor.id,
        tag=binding_tag,
        character_name=actor_name,
        role="LEAD",
        identity_type=identity_type,
    )
    session.add(binding)
    await session.flush()

    for index, ref_url in enumerate(base_ref_urls or []):
        session.add(ActorRef(
            actor_id=actor.id,
            image_url=ref_url,
            is_primary=index == 0,
            label=f"base-angle-{index}",
        ))

    if look_tag:
        preset = ActorWardrobePreset(
            actor_id=actor.id,
            label=f"{actor_name} look",
            description=f"{actor_name} wardrobe look",
            reference_images=ref_urls,
        )
        session.add(preset)
        await session.flush()
        session.add(CastLook(
            cast_binding_id=binding.id,
            wardrobe_preset_id=preset.id,
            tag=look_tag,
            is_default=look_is_default,
        ))
    else:
        for index, ref_url in enumerate(ref_urls):
            session.add(ActorRef(
                actor_id=actor.id,
                image_url=ref_url,
                is_primary=index == 0,
                label=f"angle-{index}",
            ))

    await session.flush()


async def _create_set_binding(
    session: AsyncSession,
    *,
    bible_id,
    tag: str,
    ref_url: str,
) -> None:
    lib_set = LibrarySet(
        name="Lavish Gothic Penthouse",
        reverse_prompt="lavish, decaying Gothic penthouse atop a cyberpunk tower",
    )
    session.add(lib_set)
    await session.flush()
    session.add(LibrarySetRef(
        library_set_id=lib_set.id,
        image_url=ref_url,
        is_primary=True,
        label="hero",
    ))
    session.add(SetBinding(
        production_bible_id=bible_id,
        library_set_id=lib_set.id,
        tag=tag,
        production_name="Gothic Penthouse",
    ))
    await session.flush()


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
async def test_multi_character_refs_include_all_visible_tags(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Test Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Brandon Cross",
            binding_tag="BRANDON_CROSS",
            look_tag="BRANDON_CROSS_CYBERPUNK",
            ref_urls=[
                _write_ref(tmp_path, "brandon-look-1.png"),
                _write_ref(tmp_path, "brandon-look-2.png"),
            ],
        )
        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Dracula",
            binding_tag="DRACULA_NORMAL",
            ref_urls=[
                _write_ref(tmp_path, "dracula-1.png"),
                _write_ref(tmp_path, "dracula-2.png"),
            ],
        )
        await session.commit()

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="Two characters face off.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=["@BRANDON_CROSS_CYBERPUNK", "DRACULA_NORMAL"],
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt=(
                "A cinematic commercial featuring @BRANDON_CROSS_CYBERPUNK and "
                "@DRACULA_NORMAL standing opposite each other."
            ),
            shot=shot,
            shot_manifest_json={
                "placements": [
                    {"asset_tag": "BRANDON_CROSS_CYBERPUNK"},
                    {"asset_tag": "DRACULA_NORMAL"},
                ]
            },
            selected_reference_tags=[],
            all_assets=[],
            file_mgr=FileManager(),
        )

        assert context.mandatory_character_tags == [
            "BRANDON_CROSS_CYBERPUNK",
            "DRACULA_NORMAL",
        ]
        assert len(context.ref_image_bytes_list) == 4
        assert context.final_reference_tags == [
            "BRANDON_CROSS_CYBERPUNK",
            "DRACULA_NORMAL",
            "BRANDON_CROSS_CYBERPUNK",
            "DRACULA_NORMAL",
        ]


@pytest.mark.asyncio
async def test_binding_tag_with_default_look_keeps_requested_character_ref(
    session_factory,
    tmp_path: Path,
):
    async with session_factory() as session:
        bible = ProductionBible(name="Default Look Alias Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Brandon Cross",
            binding_tag="BRANDON_CROSS",
            look_tag="BRANDON_CROSS_CYBERPUNK",
            look_is_default=True,
            ref_urls=[_write_ref(tmp_path, "brandon-look.png")],
            base_ref_urls=[_write_ref(tmp_path, "brandon-face.png")],
        )
        await session.commit()

        resolved = await resolve_tags_with_assets(
            "@BRANDON_CROSS enters frame.",
            bible.id,
            session,
        )
        char_ref = next(ref for ref in resolved.asset_refs if ref.asset_type == "CHARACTER")
        assert char_ref.tag == "BRANDON_CROSS_CYBERPUNK"

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="The recurring hero enters.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=[],
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt=None,
            shot=shot,
            shot_manifest_json={
                "placements": [
                    {"asset_tag": "BRANDON_CROSS"},
                ]
            },
            selected_reference_tags=["BRANDON_CROSS"],
            all_assets=[],
            file_mgr=FileManager(),
        )

        assert context.mandatory_character_tags == ["BRANDON_CROSS"]
        assert context.final_reference_tags == ["BRANDON_CROSS", "BRANDON_CROSS"]
        assert context.identity_types_by_tag["BRANDON_CROSS"] == "HUMAN"
        assert context.ref_image_bytes_list == [
            Path(tmp_path / "brandon-face.png").read_bytes(),
            Path(tmp_path / "brandon-look.png").read_bytes(),
        ]


@pytest.mark.asyncio
async def test_character_tag_typo_is_canonicalized_to_bound_look(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Typo Recovery Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Brandon Cross",
            binding_tag="BRANDON_CROSS",
            look_tag="BRANDON_CROSS_CYERPUNK",
            ref_urls=[_write_ref(tmp_path, "brandon-cyerpunk.png")],
        )
        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Dracula",
            binding_tag="DRACULA_NORMAL",
            ref_urls=[_write_ref(tmp_path, "dracula-normal.png")],
        )
        await session.commit()

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="Brandon and Dracula face off.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=["@BRANDON_CROSS_CYBERPUNK", "@DRACULA_NORMAL"],
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt=(
                "A cinematic commercial featuring @BRANDON_CROSS_CYBERPUNK and "
                "@DRACULA_NORMAL standing opposite each other."
            ),
            shot=shot,
            shot_manifest_json={
                "placements": [
                    {"asset_tag": "BRANDON_CROSS_CYBERPUNK"},
                    {"asset_tag": "DRACULA_NORMAL"},
                ]
            },
            selected_reference_tags=["BRANDON_CROSS_CYBERPUNK", "DRACULA_NORMAL"],
            all_assets=[],
            file_mgr=FileManager(),
        )

        assert context.canonical_tag_remaps == {
            "BRANDON_CROSS_CYBERPUNK": "BRANDON_CROSS_CYERPUNK",
        }
        assert context.mandatory_character_tags == [
            "BRANDON_CROSS_CYERPUNK",
            "DRACULA_NORMAL",
        ]
        assert context.final_reference_tags == [
            "BRANDON_CROSS_CYERPUNK",
            "DRACULA_NORMAL",
        ]


@pytest.mark.asyncio
async def test_stringified_characters_present_is_normalized(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Legacy Stringified Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Brandon Cross",
            binding_tag="BRANDON_CROSS",
            look_tag="BRANDON_CROSS_CYBERPUNK",
            ref_urls=[_write_ref(tmp_path, "brandon-legacy.png")],
        )
        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Dracula",
            binding_tag="DRACULA_NORMAL",
            ref_urls=[_write_ref(tmp_path, "dracula-legacy.png")],
        )
        await session.commit()

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="Legacy row with stringified characters_present.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=json.dumps([
                "@BRANDON_CROSS_CYBERPUNK",
                "@DRACULA_NORMAL",
            ]),
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt=None,
            shot=shot,
            shot_manifest_json=None,
            selected_reference_tags=[],
            all_assets=[],
            file_mgr=FileManager(),
        )

        assert context.mandatory_character_tags == [
            "BRANDON_CROSS_CYBERPUNK",
            "DRACULA_NORMAL",
        ]


@pytest.mark.asyncio
async def test_multi_character_refs_trim_round_robin_over_budget(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Overflow Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Brandon Cross",
            binding_tag="BRANDON_CROSS",
            look_tag="BRANDON_CROSS_CYBERPUNK",
            ref_urls=[_write_ref(tmp_path, f"brandon-{i}.png") for i in range(8)],
        )
        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Dracula",
            binding_tag="DRACULA_NORMAL",
            ref_urls=[_write_ref(tmp_path, f"dracula-{i}.png") for i in range(8)],
        )
        await session.commit()

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="Two characters face off.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=["BRANDON_CROSS_CYBERPUNK", "DRACULA_NORMAL"],
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt=None,
            shot=shot,
            shot_manifest_json={"placements": []},
            selected_reference_tags=[],
            all_assets=[],
            file_mgr=FileManager(),
            max_reference_images=_NANO_BANANA_MAX_REFERENCE_IMAGES,
        )

        counts = Counter(context.final_reference_tags)
        assert len(context.ref_image_bytes_list) == _NANO_BANANA_MAX_REFERENCE_IMAGES
        assert counts["BRANDON_CROSS_CYBERPUNK"] == 7
        assert counts["DRACULA_NORMAL"] == 6
        assert context.trimmed_reference_counts == {
            "BRANDON_CROSS_CYBERPUNK": 1,
            "DRACULA_NORMAL": 2,
        }


@pytest.mark.asyncio
async def test_optional_non_character_ref_appends_after_characters(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Mixed Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Dracula",
            binding_tag="DRACULA_NORMAL",
            ref_urls=[
                _write_ref(tmp_path, "dracula-a.png"),
                _write_ref(tmp_path, "dracula-b.png"),
            ],
        )
        await _create_set_binding(
            session,
            bible_id=bible.id,
            tag="GOTHIC_PENTHOUSE",
            ref_url=_write_ref(tmp_path, "penthouse.png"),
        )
        await session.commit()

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="Dracula stands in the penthouse.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=["DRACULA_NORMAL"],
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt=None,
            shot=shot,
            shot_manifest_json={
                "placements": [
                    {"asset_tag": "DRACULA_NORMAL"},
                    {"asset_tag": "GOTHIC_PENTHOUSE"},
                ]
            },
            selected_reference_tags=["GOTHIC_PENTHOUSE"],
            all_assets=[],
            file_mgr=FileManager(),
            max_reference_images=3,
        )

        assert context.final_reference_tags == [
            "DRACULA_NORMAL",
            "DRACULA_NORMAL",
            "GOTHIC_PENTHOUSE",
        ]


@pytest.mark.asyncio
async def test_scene_prompt_fallback_resolves_character_refs(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Fallback Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Dracula",
            binding_tag="DRACULA_NORMAL",
            ref_urls=[
                _write_ref(tmp_path, "fallback-dracula-a.png"),
                _write_ref(tmp_path, "fallback-dracula-b.png"),
            ],
        )
        await session.commit()

        shot = Shot(
            scene_id=uuid.uuid4(),
            shot_index=0,
            shot_description="Fallback shot.",
            start_frame_prompt="",
            end_frame_prompt="",
            video_motion_prompt="",
            status="draft",
            characters_present=[],
        )

        context = await _assemble_nano_banana_reference_context(
            session,
            production_bible_id=bible.id,
            scene_prompt="A cinematic portrait of @DRACULA_NORMAL in neon rain.",
            shot=shot,
            shot_manifest_json=None,
            selected_reference_tags=[],
            all_assets=[],
            file_mgr=FileManager(),
        )

        assert context.mandatory_character_tags == ["DRACULA_NORMAL"]
        assert len(context.ref_image_bytes_list) == 2


@pytest.mark.asyncio
async def test_human_look_refs_merge_wardrobe_and_base_actor_refs(session_factory, tmp_path: Path):
    async with session_factory() as session:
        bible = ProductionBible(name="Merged Human Look Bible")
        session.add(bible)
        await session.flush()

        await _create_character_binding(
            session,
            bible_id=bible.id,
            actor_name="Brandon Cross",
            binding_tag="BRANDON_CROSS",
            look_tag="BRANDON_CROSS_CYBERPUNK",
            ref_urls=[
                _write_ref(tmp_path, "look-1.png"),
                _write_ref(tmp_path, "look-2.png"),
            ],
            base_ref_urls=[
                _write_ref(tmp_path, "base-1.png"),
                _write_ref(tmp_path, "base-2.png"),
            ],
        )
        await session.commit()

        resolved = await resolve_tags_with_assets(
            "@BRANDON_CROSS_CYBERPUNK enters frame.",
            bible.id,
            session,
        )

        char_ref = next(ref for ref in resolved.asset_refs if ref.asset_type == "CHARACTER")
        assert char_ref.identity_type == "HUMAN"
        assert char_ref.reference_image_urls == [
            str(tmp_path / "look-1.png"),
            str(tmp_path / "base-1.png"),
            str(tmp_path / "look-2.png"),
            str(tmp_path / "base-2.png"),
        ]
        assert char_ref.face_reference_image_urls == [
            str(tmp_path / "base-1.png"),
            str(tmp_path / "base-2.png"),
        ]
        assert char_ref.wardrobe_reference_image_urls == [
            str(tmp_path / "look-1.png"),
            str(tmp_path / "look-2.png"),
        ]


@pytest.mark.asyncio
async def test_identity_policy_preserves_human_wardrobe_and_non_human_refs(
    monkeypatch,
    tmp_path: Path,
):
    human_url = _write_ref(tmp_path, "human-ref.png")
    wardrobe_url = _write_ref(tmp_path, "human-wardrobe.png")
    cat_url = _write_ref(tmp_path, "cat-ref.png")
    seen_urls: list[str] = []

    async def fake_prequalify(ref_urls, _file_mgr, face_svc=None):
        del face_svc
        seen_urls.extend(ref_urls)
        return [
            QualifiedRef(
                image_url=human_url,
                image_bytes=Path(human_url).read_bytes(),
                face_crop_bytes=b"face-crop",
                face_embedding=np.ones(4, dtype=np.float32),
                detection_score=0.99,
            )
        ]

    monkeypatch.setattr(
        "vidpipe.services.ref_prequalification.prequalify_refs",
        fake_prequalify,
    )

    filtered_candidates, prequalified_embeddings, policy_report = (
        await _apply_identity_policy_to_reference_candidates(
            selected_candidates=[
                _ReferenceCandidate(
                    tag="BRANDON_CROSS",
                    image_url=human_url,
                    image_bytes=Path(human_url).read_bytes(),
                    asset_type="CHARACTER",
                    source="binding",
                    identity_type="HUMAN",
                    reference_kind="face",
                ),
                _ReferenceCandidate(
                    tag="BRANDON_CROSS",
                    image_url=wardrobe_url,
                    image_bytes=Path(wardrobe_url).read_bytes(),
                    asset_type="CHARACTER",
                    source="binding",
                    identity_type="HUMAN",
                    reference_kind="wardrobe",
                ),
                _ReferenceCandidate(
                    tag="DRACULA_NORMAL",
                    image_url=cat_url,
                    image_bytes=Path(cat_url).read_bytes(),
                    asset_type="CHARACTER",
                    source="binding",
                    identity_type="ANIMAL",
                    reference_kind="wardrobe",
                ),
            ],
            file_mgr=FileManager(),
        )
    )

    assert seen_urls == [human_url]
    assert [candidate.tag for candidate in filtered_candidates] == [
        "BRANDON_CROSS",
        "BRANDON_CROSS",
        "DRACULA_NORMAL",
    ]
    assert filtered_candidates[0].image_bytes == b"face-crop"
    assert filtered_candidates[1].image_bytes == Path(wardrobe_url).read_bytes()
    assert len(prequalified_embeddings["BRANDON_CROSS"]) == 1
    assert policy_report["human_face_tags"] == ["BRANDON_CROSS"]
    assert policy_report["human_wardrobe_tags"] == ["BRANDON_CROSS"]
    assert policy_report["passthrough_tags"] == [
        "BRANDON_CROSS",
        "DRACULA_NORMAL",
    ]


def test_retry_reference_packing_prefers_face_and_wardrobe():
    candidates = [
        _ReferenceCandidate(
            tag="BRANDON_CROSS",
            image_url="face-1",
            image_bytes=b"face-1",
            asset_type="CHARACTER",
            source="binding",
            identity_type="HUMAN",
            reference_kind="face",
        ),
        _ReferenceCandidate(
            tag="BRANDON_CROSS",
            image_url="wardrobe-1",
            image_bytes=b"wardrobe-1",
            asset_type="CHARACTER",
            source="binding",
            identity_type="HUMAN",
            reference_kind="wardrobe",
        ),
        _ReferenceCandidate(
            tag="BRANDON_CROSS",
            image_url="set-1",
            image_bytes=b"set-1",
            asset_type="SET",
            source="binding",
            reference_kind="supplemental",
        ),
    ]

    packed = _build_retry_reference_candidates(
        candidates,
        mandatory_tags=["BRANDON_CROSS"],
        retry_level=2,
    )

    assert [candidate.reference_kind for candidate in packed] == ["face", "wardrobe"]


@pytest.mark.asyncio
async def test_crowded_human_verification_uses_vision_as_authority(monkeypatch):
    async def fake_vision_report(**kwargs):
        del kwargs
        return _VisionVerificationReport(
            passed=True,
            detail="vision ok",
            results=[
                _CharacterVisionVerificationResult(
                    tag="BRANDON_CROSS",
                    passed=True,
                    character_visible=True,
                    identity_match=True,
                    wardrobe_match=True,
                    identity_score=9.0,
                    wardrobe_score=9.0,
                    issues=[],
                    used_full_frame=False,
                )
            ],
        )

    def fake_crop_plan(_keyframe_bytes, _targets):
        return _CharacterCropPlan(
            selections={
                "BRANDON_CROSS": _CharacterCropSelection(
                    tag="BRANDON_CROSS",
                    image_bytes=b"crop",
                    bbox=[0, 0, 10, 10],
                    used_full_frame=False,
                )
            },
            detected_face_count=5,
            detected_person_count=5,
            detected_object_count=5,
        )

    async def fake_verify_target_face(_crop_bytes, _ref_embeddings, threshold=None):
        del threshold
        return False, 0.21, "best_sim=0.210 threshold=0.450 refs_checked=1"

    monkeypatch.setattr(
        "vidpipe.pipeline.keyframes._verify_keyframe_characters_with_vision",
        fake_vision_report,
    )
    monkeypatch.setattr(
        "vidpipe.pipeline.keyframes._select_character_candidate_boxes",
        fake_crop_plan,
    )
    monkeypatch.setattr(
        "vidpipe.pipeline.keyframes._verify_target_face",
        fake_verify_target_face,
    )

    report = await _verify_generated_keyframe(
        scene=SimpleNamespace(vision_model="gemini", text_model="gemini"),
        shot=SimpleNamespace(shot_index=1),
        position="end",
        keyframe_bytes=b"frame",
        shot_manifest_json={"placements": [{"asset_tag": "BRANDON_CROSS", "position": "left"}]},
        selected_candidates=[
            _ReferenceCandidate(
                tag="BRANDON_CROSS",
                image_url="face",
                image_bytes=b"face",
                asset_type="CHARACTER",
                source="binding",
                identity_type="HUMAN",
                reference_kind="face",
            )
        ],
        identity_types_by_tag={"BRANDON_CROSS": "HUMAN"},
        placed_char_assets=[],
        prequalified_ref_embeddings_by_tag={"BRANDON_CROSS": [np.ones(4, dtype=np.float32)]},
    )

    assert report.passed is True
    assert report.verification_mode == "vision_primary_face_advisory"
    assert report.face_results[0].advisory is True
    assert report.face_results[0].passed is False


@pytest.mark.asyncio
async def test_single_human_verification_stays_strict(monkeypatch):
    async def fake_vision_report(**kwargs):
        del kwargs
        return _VisionVerificationReport(
            passed=True,
            detail="vision ok",
            results=[
                _CharacterVisionVerificationResult(
                    tag="BRANDON_CROSS",
                    passed=True,
                    character_visible=True,
                    identity_match=True,
                    wardrobe_match=True,
                    identity_score=9.0,
                    wardrobe_score=9.0,
                    issues=[],
                    used_full_frame=False,
                )
            ],
        )

    def fake_crop_plan(_keyframe_bytes, _targets):
        return _CharacterCropPlan(
            selections={
                "BRANDON_CROSS": _CharacterCropSelection(
                    tag="BRANDON_CROSS",
                    image_bytes=b"crop",
                    bbox=[0, 0, 10, 10],
                    used_full_frame=False,
                )
            },
            detected_face_count=1,
            detected_person_count=1,
            detected_object_count=1,
        )

    async def fake_verify_target_face(_crop_bytes, _ref_embeddings, threshold=None):
        del threshold
        return False, 0.21, "best_sim=0.210 threshold=0.450 refs_checked=1"

    monkeypatch.setattr(
        "vidpipe.pipeline.keyframes._verify_keyframe_characters_with_vision",
        fake_vision_report,
    )
    monkeypatch.setattr(
        "vidpipe.pipeline.keyframes._select_character_candidate_boxes",
        fake_crop_plan,
    )
    monkeypatch.setattr(
        "vidpipe.pipeline.keyframes._verify_target_face",
        fake_verify_target_face,
    )

    report = await _verify_generated_keyframe(
        scene=SimpleNamespace(vision_model="gemini", text_model="gemini"),
        shot=SimpleNamespace(shot_index=1),
        position="end",
        keyframe_bytes=b"frame",
        shot_manifest_json={"placements": [{"asset_tag": "BRANDON_CROSS", "position": "left"}]},
        selected_candidates=[
            _ReferenceCandidate(
                tag="BRANDON_CROSS",
                image_url="face",
                image_bytes=b"face",
                asset_type="CHARACTER",
                source="binding",
                identity_type="HUMAN",
                reference_kind="face",
            )
        ],
        identity_types_by_tag={"BRANDON_CROSS": "HUMAN"},
        placed_char_assets=[],
        prequalified_ref_embeddings_by_tag={"BRANDON_CROSS": [np.ones(4, dtype=np.float32)]},
    )

    assert report.passed is False
    assert report.verification_mode == "strict_face_and_vision"
    assert report.face_results[0].advisory is False


def test_best_effort_selection_prefers_visible_identity_match():
    weak_visible_attempt = _GeneratedKeyframeAttempt(
        attempt_number=1,
        retry_level=0,
        keyframe_bytes=b"attempt-1",
        verification_report=_KeyframeVerificationReport(
            passed=False,
            detail="cat visible but eyes wrong",
            verification_mode="vision_only",
            face_results=[],
            vision_report=_VisionVerificationReport(
                passed=False,
                detail="vision fail",
                results=[
                    _CharacterVisionVerificationResult(
                        tag="DRACULA_NORMAL",
                        passed=False,
                        character_visible=True,
                        identity_match=False,
                        wardrobe_match=False,
                        identity_score=6.0,
                        wardrobe_score=6.0,
                        issues=["eye color mismatch"],
                        used_full_frame=False,
                    )
                ],
            ),
        ),
    )
    invisible_attempt = _GeneratedKeyframeAttempt(
        attempt_number=2,
        retry_level=1,
        keyframe_bytes=b"attempt-2",
        verification_report=_KeyframeVerificationReport(
            passed=False,
            detail="cat missing",
            verification_mode="vision_only",
            face_results=[],
            vision_report=_VisionVerificationReport(
                passed=False,
                detail="vision fail",
                results=[
                    _CharacterVisionVerificationResult(
                        tag="DRACULA_NORMAL",
                        passed=False,
                        character_visible=False,
                        identity_match=False,
                        wardrobe_match=False,
                        identity_score=0.0,
                        wardrobe_score=0.0,
                        issues=["character is not visible"],
                        used_full_frame=False,
                    )
                ],
            ),
        ),
    )

    selected = _select_best_effort_attempt([invisible_attempt, weak_visible_attempt])

    assert selected is weak_visible_attempt


def test_best_effort_detail_includes_selected_attempt_and_scores():
    attempt = _GeneratedKeyframeAttempt(
        attempt_number=2,
        retry_level=1,
        keyframe_bytes=b"attempt-2",
        verification_report=_KeyframeVerificationReport(
            passed=False,
            detail="vision_check=False @BRANDON: passed=False visible=True",
            verification_mode="strict_face_and_vision",
            face_results=[
                _HumanFaceVerificationResult(
                    tag="BRANDON",
                    passed=False,
                    advisory=False,
                    similarity=0.41,
                    detail="best_sim=0.410 threshold=0.450 refs_checked=3",
                )
            ],
            vision_report=_VisionVerificationReport(
                passed=False,
                detail="vision fail",
                results=[
                    _CharacterVisionVerificationResult(
                        tag="BRANDON",
                        passed=False,
                        character_visible=True,
                        identity_match=True,
                        wardrobe_match=False,
                        identity_score=9.0,
                        wardrobe_score=5.0,
                        issues=["wardrobe drift"],
                        used_full_frame=False,
                    )
                ],
            ),
        ),
    )

    detail = _build_best_effort_detail(
        position="end",
        attempts=[attempt, attempt, attempt],
        selected_attempt=attempt,
    )

    assert "best_effort_fallback accepted after full verification miss" in detail
    assert "selected_attempt=2/3" in detail
    assert "position=end" in detail
    assert "wardrobe_total=5.0" in detail


def test_best_effort_fallback_result_can_capture_transport_exhaustion():
    attempt = _GeneratedKeyframeAttempt(
        attempt_number=1,
        retry_level=0,
        keyframe_bytes=b"attempt-1",
        verification_report=_KeyframeVerificationReport(
            passed=False,
            detail="vision fail",
            verification_mode="vision_primary_face_advisory",
            face_results=[],
            vision_report=_VisionVerificationReport(
                passed=False,
                detail="vision fail",
                results=[
                    _CharacterVisionVerificationResult(
                        tag="BRANDON",
                        passed=False,
                        character_visible=True,
                        identity_match=False,
                        wardrobe_match=True,
                        identity_score=5.0,
                        wardrobe_score=8.0,
                        issues=["Character is seen from behind"],
                        used_full_frame=True,
                    )
                ],
            ),
        ),
    )

    fallback = _build_best_effort_fallback_result(
        position="end",
        attempts=[attempt],
        transport_detail="ClientError(code=429): Resource exhausted",
    )

    assert fallback is not None
    selected_attempt, verification = fallback
    assert selected_attempt is attempt
    assert verification.status == "accepted_with_warnings"
    assert "transport_exhausted=ClientError(code=429): Resource exhausted" in verification.detail


def test_partial_visibility_human_check_allows_back_view_when_wardrobe_is_strong():
    target = SimpleNamespace(identity_type="HUMAN")
    crop = _CharacterCropSelection(
        tag="BRANDON",
        image_bytes=b"crop",
        bbox=None,
        used_full_frame=True,
    )
    result = SimpleNamespace(
        character_visible=True,
        wardrobe_score=7.0,
        issues=["Character is seen from behind, face and head are not visible for identity verification."],
    )

    assert _passes_partial_visibility_human_check(
        target=target,
        crop=crop,
        result=result,
    ) is True
