"""Tests for dynamic screenwriter shot breakdown behavior."""

import pytest

from vidpipe.schemas.screenwriter_agent import ScriptAnalysis, ShotBreakdown
from vidpipe.services.llm.base import LLMAdapter
from vidpipe.services.screenwriter_agent import ScreenwriterAgentService, validate_screenplay


def _analysis_with_beats(count: int) -> ScriptAnalysis:
    return ScriptAnalysis.model_validate({
        "narrative_summary": "Two rivals square off across an ornate room.",
        "tone": "cinematic",
        "genre": "drama",
        "pacing": "steady",
        "characters": [
            {
                "tag": "BRANDON_CROSS",
                "role": "protagonist",
                "screen_time_hint": "heavy",
                "first_appearance_beat": 0,
            },
            {
                "tag": "DRACULA_NORMAL",
                "role": "antagonist",
                "screen_time_hint": "heavy",
                "first_appearance_beat": 0,
            },
        ],
        "settings": ["Gothic penthouse"],
        "story_beats": [
            {
                "index": idx,
                "description": f"Beat {idx}",
                "characters_involved": ["BRANDON_CROSS", "DRACULA_NORMAL"],
                "emotional_tone": "tense",
                "is_climax": idx == count - 1,
            }
            for idx in range(count)
        ],
        "emotional_arc": "rising confrontation",
    })


class FakeScreenwriterAdapter(LLMAdapter):
    """Adapter that returns a compressed breakdown first, then a corrected retry."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls = 0

    async def generate_text(
        self,
        prompt: str,
        schema,
        *,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        max_retries: int = 3,
    ):
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls == 1:
            return schema.model_validate({
                "shots": [
                    {
                        "shot_index": 0,
                        "beat_index": 0,
                        "narrative_intent": "Play the whole scene as one master shot.",
                        "characters_present": ["BRANDON_CROSS", "DRACULA_NORMAL"],
                        "setting": "Gothic penthouse",
                        "time_of_day": "night",
                        "emotional_weight": 9,
                        "duration_hint": 12,
                        "transition_from_previous": None,
                    }
                ],
                "arc_coverage": "Compressed into one take.",
                "uncovered_beats": [],
            })

        return schema.model_validate({
            "shots": [
                {
                    "shot_index": 0,
                    "beat_index": 0,
                    "narrative_intent": "Introduce the faceoff.",
                    "characters_present": ["BRANDON_CROSS", "DRACULA_NORMAL"],
                    "setting": "Gothic penthouse",
                    "time_of_day": "night",
                    "emotional_weight": 5,
                    "duration_hint": 4,
                    "transition_from_previous": None,
                },
                {
                    "shot_index": 1,
                    "beat_index": 1,
                    "narrative_intent": "Cover the first dialogue turn.",
                    "characters_present": ["DRACULA_NORMAL"],
                    "setting": "Gothic penthouse",
                    "time_of_day": "night",
                    "emotional_weight": 7,
                    "duration_hint": 4,
                    "transition_from_previous": "Cut to Dracula.",
                },
                {
                    "shot_index": 2,
                    "beat_index": 2,
                    "narrative_intent": "Escalate as Brandon takes the product.",
                    "characters_present": ["BRANDON_CROSS"],
                    "setting": "Gothic penthouse",
                    "time_of_day": "night",
                    "emotional_weight": 8,
                    "duration_hint": 4,
                    "transition_from_previous": "Cut back to Brandon.",
                },
            ],
            "arc_coverage": "Three shots cover the confrontation cleanly.",
            "uncovered_beats": [],
        })

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


@pytest.mark.asyncio
async def test_dynamic_breakdown_retries_when_multi_beat_scene_is_overcompressed():
    adapter = FakeScreenwriterAdapter()
    service = ScreenwriterAgentService(adapter, model_label="test-model")
    analysis = _analysis_with_beats(4)

    breakdown = await service.break_into_shots(
        scene_prompt="A long confrontation between two rivals.",
        analysis=analysis,
        target_shot_count=1,
        dynamic_shot_count=True,
    )

    assert adapter.calls == 2
    assert len(breakdown.shots) == 3
    assert "Use between 2 and" in adapter.prompts[0]
    assert "previous breakdown was too compressed" in adapter.prompts[1]


def test_validate_screenplay_uses_expected_story_beats():
    warnings = validate_screenplay(
        "A confrontation unfolds across several beats.",
        ShotBreakdown.model_validate({
            "shots": [
                {
                    "shot_index": 0,
                    "beat_index": 0,
                    "narrative_intent": "Cover only the opening beat.",
                    "characters_present": ["BRANDON_CROSS"],
                    "setting": "Penthouse",
                    "time_of_day": "night",
                    "emotional_weight": 5,
                    "duration_hint": 4,
                    "transition_from_previous": None,
                }
            ],
            "arc_coverage": "Only the opening beat is covered.",
            "uncovered_beats": [],
        }),
        {"BRANDON_CROSS", "DRACULA_NORMAL"},
        expected_story_beats=4,
    )

    assert any("Story beats not covered by any shot" in warning for warning in warnings)


def test_validate_screenplay_canonicalizes_binding_registry_tags():
    warnings = validate_screenplay(
        "Brandon and Nexus negotiate in a club.",
        ShotBreakdown.model_validate({
            "shots": [
                {
                    "shot_index": 0,
                    "beat_index": 0,
                    "narrative_intent": "Brandon sits at the booth.",
                    "characters_present": ["BRANDON_CROSS_CYERPUNK"],
                    "setting": "Club",
                    "time_of_day": "night",
                    "emotional_weight": 5,
                    "duration_hint": 4,
                    "transition_from_previous": None,
                }
            ],
            "arc_coverage": "Only Brandon appears.",
            "uncovered_beats": [],
        }),
        {"@BRANDON_CROSS_CYBERPUNK", "Nexus"},
        expected_story_beats=1,
        binding_registry_tags={"BRANDON_CROSS_CYERPUNK"},
    )

    assert not any("BRANDON_CROSS" in warning for warning in warnings)
    assert any("NEXUS" in warning for warning in warnings)
