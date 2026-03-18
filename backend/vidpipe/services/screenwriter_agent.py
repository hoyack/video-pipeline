"""Screenwriter Agent Service — 2-step script analysis and shot breakdown.

Transforms scene prompts into shot-level screenplays with explicit character
assignment (characters_present[]) before the main storyboard LLM call.

Step 1: analyze_script() — narrative structure, characters, beats, tone
Step 2: break_into_shots() — per-shot character/narrative assignment

Spec reference: Screenwriter Agent Plan (Phases A-E)
"""

import json
import logging
import uuid

from vidpipe.schemas.screenwriter_agent import (
    ScriptAnalysis,
    ShotBreakdown,
)
from vidpipe.services.event_bus import emit_task_log
from vidpipe.services.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


ANALYZE_SCRIPT_PROMPT = """\
You are a screenwriter analyzing a scene script for shot-level video generation.

{binding_registry_block}

Analyze the script and identify:
- Narrative summary (1-2 sentences)
- Tone, genre, and pacing
- All characters referenced (from @tags AND implied by text)
- Settings/locations mentioned or implied
- Story beats (key narrative moments, in order)
- Emotional arc across the scene

For characters, use EXACT @tag names from the Available Assets list when they match.
Characters without an @tag should use a descriptive name (e.g. "narrator", "crowd").

Script:
{scene_prompt}"""


BREAK_INTO_SHOTS_PROMPT = """\
You are a director breaking a scene into exactly {target_shot_count} visual shots.

SCRIPT ANALYSIS:
{analysis_json}

{binding_registry_block}

{existing_shot_context}

For each shot, assign:
- shot_index: 0-based index
- beat_index: which story beat this shot serves
- narrative_intent: what the shot communicates (1 sentence)
- characters_present: EXACT @tags from Available Assets of characters VISIBLE in frame.
  CRITICAL: This list MUST be empty [] for shots where NO characters appear
  (establishing shots, scenery, object close-ups, building exteriors).
  Only list characters who are physically visible in the frame.
- setting, time_of_day, emotional_weight (0-10), duration_hint (seconds)
- transition_from_previous (optional)

Ensure every story beat is covered by at least one shot.
Distribute emotional weight to create a compelling arc.

Script:
{scene_prompt}"""


class ScreenwriterAgentService:
    """Transforms scene prompts into shot-level screenplays with explicit character assignment."""

    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        scene_id: uuid.UUID | str | None = None,
        model_label: str | None = None,
    ):
        self._adapter = adapter
        self._scene_id = scene_id
        self._model_label = model_label

    def _emit_llm_log(
        self,
        *,
        source: str,
        summary: str,
        detail: str,
        kind: str,
        level: str = "info",
    ) -> None:
        if not self._scene_id:
            return
        emit_task_log(
            self._scene_id,
            phase="storyboard",
            level=level,
            kind=kind,
            source=source,
            summary=summary,
            detail=detail,
        )

    async def analyze_script(
        self,
        scene_prompt: str,
        binding_registry_block: str = "",
    ) -> ScriptAnalysis:
        """Step 1: Analyze scene prompt for narrative structure, characters, beats."""
        prompt = ANALYZE_SCRIPT_PROMPT.format(
            binding_registry_block=binding_registry_block,
            scene_prompt=scene_prompt,
        )
        self._emit_llm_log(
            source="screenwriter.analysis.prompt",
            summary="Screenwriter analysis prompt sent",
            kind="prompt",
            detail=(
                f"MODEL: {self._model_label or 'unknown'}\n"
                f"SCHEMA: {ScriptAnalysis.__name__}\n\n"
                f"{prompt}"
            ),
        )

        analysis = await self._adapter.generate_text(
            prompt=prompt,
            schema=ScriptAnalysis,
            temperature=0.5,
            max_retries=1,
        )
        self._emit_llm_log(
            source="screenwriter.analysis.response",
            summary="Screenwriter analysis response received",
            kind="response",
            detail=(
                f"MODEL: {self._model_label or 'unknown'}\n"
                f"SCHEMA: {ScriptAnalysis.__name__}\n\n"
                f"{json.dumps(analysis.model_dump(), indent=2)}"
            ),
            level="success",
        )
        logger.info(
            "Script analysis: %d characters, %d beats, tone=%s",
            len(analysis.characters),
            len(analysis.story_beats),
            analysis.tone,
        )
        return analysis

    async def break_into_shots(
        self,
        scene_prompt: str,
        analysis: ScriptAnalysis,
        target_shot_count: int,
        binding_registry_block: str = "",
        existing_shot_context: str = "",
    ) -> ShotBreakdown:
        """Step 2: Break analysis into shot assignments with explicit characters_present."""
        analysis_json = json.dumps(analysis.model_dump(), indent=2)

        prompt = BREAK_INTO_SHOTS_PROMPT.format(
            target_shot_count=target_shot_count,
            analysis_json=analysis_json,
            binding_registry_block=binding_registry_block,
            existing_shot_context=existing_shot_context,
            scene_prompt=scene_prompt,
        )
        self._emit_llm_log(
            source="screenwriter.breakdown.prompt",
            summary="Screenwriter shot-breakdown prompt sent",
            kind="prompt",
            detail=(
                f"MODEL: {self._model_label or 'unknown'}\n"
                f"SCHEMA: {ShotBreakdown.__name__}\n\n"
                f"{prompt}"
            ),
        )

        breakdown = await self._adapter.generate_text(
            prompt=prompt,
            schema=ShotBreakdown,
            temperature=0.5,
            max_retries=1,
        )
        self._emit_llm_log(
            source="screenwriter.breakdown.response",
            summary="Screenwriter shot-breakdown response received",
            kind="response",
            detail=(
                f"MODEL: {self._model_label or 'unknown'}\n"
                f"SCHEMA: {ShotBreakdown.__name__}\n\n"
                f"{json.dumps(breakdown.model_dump(), indent=2)}"
            ),
            level="success",
        )
        logger.info(
            "Shot breakdown: %d shots, arc_coverage=%s, uncovered_beats=%s",
            len(breakdown.shots),
            breakdown.arc_coverage,
            breakdown.uncovered_beats,
        )

        # Run deterministic validation
        available_tags = set()
        for char in analysis.characters:
            available_tags.add(char.tag)
        warnings = validate_screenplay(scene_prompt, breakdown, available_tags)
        for w in warnings:
            logger.warning("Screenplay validation: %s", w)

        return breakdown


def validate_screenplay(
    scene_prompt: str,
    breakdown: ShotBreakdown,
    available_tags: set[str],
) -> list[str]:
    """Deterministic post-generation checks. Returns list of warnings (non-blocking)."""
    warnings: list[str] = []

    # 1. Every character tag should appear in at least one shot
    tags_in_shots: set[str] = set()
    for sa in breakdown.shots:
        for tag in (sa.characters_present or []):
            tags_in_shots.add(tag)

    unused_tags = available_tags - tags_in_shots
    if unused_tags:
        warnings.append(f"Characters never appear in any shot: {unused_tags}")

    # 2. Every beat_index should be referenced by at least one shot
    beat_indices = {sa.beat_index for sa in breakdown.shots if sa.beat_index is not None}
    all_beat_indices = set(range(max(beat_indices) + 1)) if beat_indices else set()
    uncovered = all_beat_indices - beat_indices
    if uncovered:
        warnings.append(f"Story beats not covered by any shot: {uncovered}")

    # 3. Check for declared uncovered beats
    if breakdown.uncovered_beats:
        warnings.append(
            f"Agent self-reported uncovered beats: {breakdown.uncovered_beats}"
        )

    # 4. Shot indices should be sequential starting from 0
    shot_indices = sorted(sa.shot_index for sa in breakdown.shots)
    expected = list(range(len(breakdown.shots)))
    if shot_indices != expected:
        warnings.append(
            f"Non-sequential shot indices: got {shot_indices}, expected {expected}"
        )

    return warnings
