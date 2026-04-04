"""Screenwriter Agent Service — 2-step script analysis and shot breakdown.

Transforms scene prompts into shot-level screenplays with explicit character
assignment (characters_present[]) before the main storyboard LLM call.

Step 1: analyze_script() — narrative structure, characters, beats, tone
Step 2: break_into_shots() — per-shot character/narrative assignment

Spec reference: Screenwriter Agent Plan (Phases A-E)
"""

import json
import logging
import re
import uuid

from vidpipe.schemas.screenwriter_agent import (
    ScriptAnalysis,
    ShotBreakdown,
)
from vidpipe.services.event_bus import emit_task_log
from vidpipe.services.llm.base import LLMAdapter

logger = logging.getLogger(__name__)
_BINDING_TAG_PATTERN = re.compile(r"@([A-Z0-9_]+)")


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
You are a director breaking a scene into visual shots.

{shot_count_instruction}

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


def _dynamic_shot_budget(
    analysis: ScriptAnalysis,
    target_shot_count: int,
) -> tuple[int, int]:
    """Estimate a healthy dynamic shot range from beat count."""
    beat_count = max(1, len(analysis.story_beats))
    min_shots = 1 if beat_count <= 1 else 2
    recommended = min(8, max(min_shots, beat_count))
    if target_shot_count > 1:
        recommended = max(recommended, min(target_shot_count, 12))
    max_shots = min(12, max(recommended, beat_count + 2))
    return recommended, max_shots


def _shot_count_instruction(
    analysis: ScriptAnalysis,
    target_shot_count: int,
    *,
    dynamic_shot_count: bool,
    retrying: bool = False,
) -> str:
    """Build the shot-count instruction for fixed or dynamic breakdowns."""
    if not dynamic_shot_count:
        return f"Break the scene into exactly {target_shot_count} visual shots."

    beat_count = max(1, len(analysis.story_beats))
    min_shots = 1 if beat_count <= 1 else 2
    recommended, max_shots = _dynamic_shot_budget(analysis, target_shot_count)

    instruction = (
        "Choose the optimal number of visual shots needed to cover the full scene clearly. "
        f"Use between {min_shots} and {max_shots} shots, aiming for about {recommended} shots "
        f"based on the script's {beat_count} story beats. "
        "Do not collapse a multi-beat scene into a single continuous master shot unless the analysis "
        "truly describes one uninterrupted beat with no meaningful visual, blocking, or emotional shift. "
        "Dialogue turns, major actions, emotional reversals, or changes in camera emphasis usually deserve new shots."
    )

    if retrying:
        instruction += (
            " The previous breakdown was too compressed. Re-break the scene with clearer visual progression, "
            "and ensure the shot list actually covers the narrative arc beat by beat."
        )

    return instruction


def _needs_dynamic_retry(
    analysis: ScriptAnalysis,
    breakdown: ShotBreakdown,
) -> bool:
    """Detect obviously over-compressed dynamic breakdowns."""
    beat_count = len(analysis.story_beats)
    covered_beats = {
        shot.beat_index
        for shot in breakdown.shots
        if shot.beat_index is not None
    }

    if beat_count >= 3 and len(breakdown.shots) <= 1:
        return True
    if beat_count >= 3 and len(covered_beats) < min(beat_count, 2):
        return True
    if breakdown.uncovered_beats:
        return True
    return False


def _canonicalize_validation_tag(
    raw_tag: str,
    *,
    binding_registry_tags: set[str],
) -> str:
    normalized = raw_tag.strip().lstrip("@").upper()
    if not normalized:
        return normalized
    if normalized in binding_registry_tags:
        return normalized
    if not binding_registry_tags:
        return normalized

    def _shared_prefix_tokens(candidate: str) -> int:
        left = normalized.split("_")
        right = candidate.split("_")
        shared = 0
        for ltok, rtok in zip(left, right):
            if ltok != rtok:
                break
            shared += 1
        return shared

    ranked = sorted(
        (
            (_shared_prefix_tokens(candidate), candidate)
            for candidate in binding_registry_tags
        ),
        reverse=True,
    )
    if not ranked:
        return normalized
    best_prefix, best_candidate = ranked[0]
    if best_prefix >= 2:
        return best_candidate
    return normalized


def _extract_binding_registry_tags(binding_registry_block: str) -> set[str]:
    return {match.upper() for match in _BINDING_TAG_PATTERN.findall(binding_registry_block or "")}


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
        dynamic_shot_count: bool = False,
        binding_registry_block: str = "",
        existing_shot_context: str = "",
    ) -> ShotBreakdown:
        """Step 2: Break analysis into shot assignments with explicit characters_present."""
        analysis_json = json.dumps(analysis.model_dump(), indent=2)

        def _build_prompt(*, retrying: bool = False) -> str:
            return BREAK_INTO_SHOTS_PROMPT.format(
                shot_count_instruction=_shot_count_instruction(
                    analysis,
                    target_shot_count,
                    dynamic_shot_count=dynamic_shot_count,
                    retrying=retrying,
                ),
                analysis_json=analysis_json,
                binding_registry_block=binding_registry_block,
                existing_shot_context=existing_shot_context,
                scene_prompt=scene_prompt,
            )

        prompt = _build_prompt()
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
        if dynamic_shot_count and _needs_dynamic_retry(analysis, breakdown):
            logger.warning(
                "Screenwriter dynamic breakdown too compressed (%d shots for %d beats); retrying once",
                len(breakdown.shots),
                len(analysis.story_beats),
            )
            retry_prompt = _build_prompt(retrying=True)
            self._emit_llm_log(
                source="screenwriter.breakdown.retry_prompt",
                summary="Screenwriter shot-breakdown retry prompt sent",
                kind="prompt",
                detail=(
                    f"MODEL: {self._model_label or 'unknown'}\n"
                    f"SCHEMA: {ShotBreakdown.__name__}\n\n"
                    f"{retry_prompt}"
                ),
                level="warning",
            )
            breakdown = await self._adapter.generate_text(
                prompt=retry_prompt,
                schema=ShotBreakdown,
                temperature=0.4,
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
        warnings = validate_screenplay(
            scene_prompt,
            breakdown,
            available_tags,
            expected_story_beats=len(analysis.story_beats),
            binding_registry_tags=_extract_binding_registry_tags(binding_registry_block),
        )
        for w in warnings:
            logger.warning("Screenplay validation: %s", w)

        return breakdown


def validate_screenplay(
    scene_prompt: str,
    breakdown: ShotBreakdown,
    available_tags: set[str],
    expected_story_beats: int | None = None,
    binding_registry_tags: set[str] | None = None,
) -> list[str]:
    """Deterministic post-generation checks. Returns list of warnings (non-blocking)."""
    warnings: list[str] = []
    registry_tags = {tag.upper() for tag in (binding_registry_tags or set()) if tag}
    normalized_available_tags = {
        _canonicalize_validation_tag(tag, binding_registry_tags=registry_tags)
        for tag in available_tags
        if tag and tag.strip()
    }

    # 1. Every character tag should appear in at least one shot
    tags_in_shots: set[str] = set()
    for sa in breakdown.shots:
        for tag in (sa.characters_present or []):
            if not tag or not tag.strip():
                continue
            tags_in_shots.add(
                _canonicalize_validation_tag(tag, binding_registry_tags=registry_tags)
            )

    unused_tags = normalized_available_tags - tags_in_shots
    if unused_tags:
        warnings.append(f"Characters never appear in any shot: {unused_tags}")

    # 2. Every beat_index should be referenced by at least one shot
    beat_indices = {sa.beat_index for sa in breakdown.shots if sa.beat_index is not None}
    if expected_story_beats is not None:
        all_beat_indices = set(range(expected_story_beats))
    else:
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
