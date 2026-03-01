"""Screenwriter service for sequential LLM-based screenplay generation.

Generates screenplay components incrementally: logline, treatment, character
breakdowns, scene breakdown, shot list, and script. Each step commits
independently so users see progress immediately via polling.

Follows the CVAnalysisService pattern: class-based with adapter injection.

Spec reference: Phase 18 Screenplay System (SCRN-07, SCRN-08, SCRN-09, SCRN-10)
"""

import logging
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import (
    Character,
    Prop,
    Screenplay,
    Set,
)
from vidpipe.schemas.screenplay import (
    CharacterBreakdownsOutput,
    LoglineOutput,
    SceneBreakdownOutput,
    ScriptOutput,
    ShotListOutput,
    TreatmentOutput,
)
from vidpipe.services.llm.base import LLMAdapter
from vidpipe.services.manifest_service import format_asset_registry, load_manifest_assets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------


def _build_logline_prompt(screenplay: Screenplay, bible_context: str) -> str:
    """Build prompt for logline generation."""
    parts = [
        "You are a professional screenwriter. Create a compelling one-sentence logline "
        "and identify the primary genre for the following production.",
    ]
    if screenplay.title:
        parts.append(f"\nProduction title: {screenplay.title}")
    if bible_context:
        parts.append(f"\nProduction Bible context:\n{bible_context}")
    parts.append(
        "\nWrite a logline that captures the central conflict, protagonist, and stakes. "
        "The genre should be one of: Action, Drama, Comedy, Sci-Fi, Thriller, Horror, "
        "Romance, Documentary, Fantasy, Mystery, Animation."
    )
    return "\n".join(parts)


def _build_treatment_prompt(screenplay: Screenplay, bible_context: str) -> str:
    """Build prompt for treatment generation. Requires logline."""
    parts = [
        "You are a professional screenwriter. Expand the following logline into a "
        "narrative treatment of 2-5 paragraphs.",
        f"\nLogline: {screenplay.logline}",
        f"Genre: {screenplay.genre}",
    ]
    if screenplay.title:
        parts.append(f"Production title: {screenplay.title}")
    if bible_context:
        parts.append(f"\nProduction Bible context:\n{bible_context}")
    parts.append(
        "\nThe treatment should outline the story arc, key plot points, and emotional "
        "journey. Write in present tense, third person."
    )
    return "\n".join(parts)


def _build_character_breakdowns_prompt(screenplay: Screenplay, bible_context: str) -> str:
    """Build prompt for character breakdowns. Requires logline + treatment."""
    parts = [
        "You are a professional screenwriter. Create detailed character breakdowns "
        "for all characters in the following production.",
        f"\nLogline: {screenplay.logline}",
        f"Genre: {screenplay.genre}",
        f"\nTreatment:\n{screenplay.treatment}",
    ]
    if bible_context:
        parts.append(
            f"\nProduction Bible context (use manifest_tags for characters that "
            f"already exist):\n{bible_context}"
        )
    parts.append(
        "\nFor each character, provide: name, role (PROTAGONIST/ANTAGONIST/SUPPORTING/EXTRA), "
        "arc (brief character journey), description (physical and personality), and "
        "manifest_tag (if the character exists in the Production Bible, use their tag like CHAR_01; "
        "otherwise leave manifest_tag null)."
    )
    return "\n".join(parts)


def _build_scene_breakdown_prompt(screenplay: Screenplay, bible_context: str) -> str:
    """Build prompt for scene breakdown. Requires logline + treatment + character_breakdowns."""
    # Summarize characters for context
    char_summary = ""
    if screenplay.character_breakdowns:
        chars = screenplay.character_breakdowns
        if isinstance(chars, dict) and "characters" in chars:
            chars = chars["characters"]
        if isinstance(chars, list):
            char_lines = []
            for c in chars:
                tag = c.get("manifest_tag", "")
                tag_str = f" [{tag}]" if tag else ""
                char_lines.append(f"- {c.get('name', 'Unknown')}{tag_str} ({c.get('role', 'SUPPORTING')})")
            char_summary = "\n".join(char_lines)

    parts = [
        "You are a professional screenwriter. Break the following production into "
        "individual scenes with detailed metadata for each.",
        f"\nLogline: {screenplay.logline}",
        f"Genre: {screenplay.genre}",
        f"\nTreatment:\n{screenplay.treatment}",
    ]
    if char_summary:
        parts.append(f"\nCharacters:\n{char_summary}")
    if bible_context:
        parts.append(
            f"\nProduction Bible context (use manifest_tags for characters, sets, "
            f"and props):\n{bible_context}"
        )
    parts.append(
        "\nFor each scene provide: scene_number (starting at 1), slugline (e.g. 'INT. OFFICE - DAY'), "
        "intent (narrative purpose — this becomes the scene prompt for video generation), "
        "emotional_beat, story_state_in, story_state_out, characters_present (list of manifest_tags "
        "or character names), set_ref (set manifest_tag or location name), and props_required "
        "(list of prop tags or names). Also provide the total scene_count."
    )
    return "\n".join(parts)


def _build_shot_list_prompt(screenplay: Screenplay, bible_context: str) -> str:
    """Build prompt for shot list generation. Requires scene_breakdown."""
    # Format scene breakdown entries for context
    scene_lines = []
    if screenplay.scene_breakdown:
        for entry in screenplay.scene_breakdown:
            scene_lines.append(
                f"Scene {entry.get('scene_number', '?')}: {entry.get('slugline', '')} — "
                f"{entry.get('intent', '')}"
            )

    parts = [
        "You are a professional director planning the shot list for a production.",
        f"\nLogline: {screenplay.logline}",
        f"Genre: {screenplay.genre}",
    ]
    if scene_lines:
        parts.append(f"\nScene breakdown:\n" + "\n".join(scene_lines))
    if bible_context:
        parts.append(f"\nProduction Bible context:\n{bible_context}")
    parts.append(
        "\nFor each scene, plan the individual shots. For each shot provide: "
        "scene_number, shot_number (sequential within the scene), shot_type "
        "(WIDE, CLOSE-UP, MEDIUM, OTS, POV, INSERT, AERIAL), camera_movement "
        "(STATIC, PAN LEFT, PAN RIGHT, DOLLY IN, DOLLY OUT, TRACKING, CRANE UP, "
        "CRANE DOWN, HANDHELD), action_notes (what happens in the shot), and "
        "duration_hint (e.g. '3-5s', '2s'). Plan 2-4 shots per scene."
    )
    return "\n".join(parts)


def _build_script_prompt(screenplay: Screenplay, bible_context: str) -> str:
    """Build prompt for script generation. Requires all prior steps."""
    # Summarize characters (not full treatment to save context window)
    char_summary = ""
    if screenplay.character_breakdowns:
        chars = screenplay.character_breakdowns
        if isinstance(chars, dict) and "characters" in chars:
            chars = chars["characters"]
        if isinstance(chars, list):
            char_lines = [f"- {c.get('name', 'Unknown')} ({c.get('role', '')}): {c.get('arc', '')}"
                          for c in chars[:10]]  # Limit to prevent overflow
            char_summary = "\n".join(char_lines)

    # Summarize scene breakdown entries
    scene_summary = ""
    if screenplay.scene_breakdown:
        scene_lines = []
        for entry in screenplay.scene_breakdown[:20]:  # Limit for context window
            scene_lines.append(
                f"Scene {entry.get('scene_number', '?')}: {entry.get('slugline', '')} — "
                f"{entry.get('intent', '')}"
            )
        scene_summary = "\n".join(scene_lines)

    parts = [
        "You are a professional screenwriter. Write the full screenplay script for "
        "the following production in standard screenplay format.",
        f"\nLogline: {screenplay.logline}",
        f"Genre: {screenplay.genre}",
    ]
    if char_summary:
        parts.append(f"\nCharacters:\n{char_summary}")
    if scene_summary:
        parts.append(f"\nScene breakdown:\n{scene_summary}")
    if bible_context:
        parts.append(f"\nProduction Bible context:\n{bible_context}")
    parts.append(
        "\nWrite the complete script including scene headings, action lines, and "
        "dialogue. Use standard screenplay format with character names in CAPS "
        "before their dialogue."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Bible context loading helper
# ---------------------------------------------------------------------------


async def load_bible_context(
    session: AsyncSession,
    production_id: uuid.UUID,
) -> str:
    """Load Production Bible context for a production.

    Finds scenes under the production that have a production_bible_id set,
    loads the bible's assets, and returns a formatted asset registry string.

    Args:
        session: Active database session.
        production_id: Production UUID.

    Returns:
        Formatted asset registry string, or empty string if no bible found.
    """
    from vidpipe.db.models import Scene

    # Find the first scene under this production with a production_bible_id
    result = await session.execute(
        select(Scene.production_bible_id)
        .where(
            Scene.production_id == production_id,
            Scene.production_bible_id.isnot(None),
        )
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return ""

    try:
        assets = await load_manifest_assets(session, row)
        return format_asset_registry(assets)
    except Exception:
        logger.warning("Failed to load bible context for production %s", production_id)
        return ""


# ---------------------------------------------------------------------------
# Entity validation helpers (SCRN-06)
# ---------------------------------------------------------------------------


async def _validate_breakdown_entities(
    session: AsyncSession,
    breakdown_entries: list[dict],
    production_bible_id: uuid.UUID,
) -> list[dict]:
    """Validate entity tags in scene breakdown against Production Bible entities.

    For each SceneBreakdownEntry, checks characters_present, set_ref, and
    props_required tags against actual Character, Set, and Prop prompt_tags.
    Logs warnings for unresolved tags but never crashes — the LLM may invent
    entities not yet in the bible.

    Returns the entries with _resolved_characters, _resolved_set, and
    _resolved_props boolean flags added.
    """
    # Load all entity prompt_tags from the production bible
    char_result = await session.execute(
        select(Character.prompt_tags).where(
            Character.production_bible_id == production_bible_id
        )
    )
    char_tags: set[str] = set()
    for (tags,) in char_result.all():
        if tags:
            char_tags.update(tags)

    set_result = await session.execute(
        select(Set.prompt_tags).where(
            Set.production_bible_id == production_bible_id
        )
    )
    set_tags: set[str] = set()
    for (tags,) in set_result.all():
        if tags:
            set_tags.update(tags)

    prop_result = await session.execute(
        select(Prop.prompt_tags).where(
            Prop.production_bible_id == production_bible_id
        )
    )
    prop_tags: set[str] = set()
    for (tags,) in prop_result.all():
        if tags:
            prop_tags.update(tags)

    # Validate each entry
    for entry in breakdown_entries:
        # Characters
        chars_present = entry.get("characters_present", [])
        all_chars_resolved = True
        for tag in chars_present:
            if tag not in char_tags:
                logger.warning(
                    "Unresolved tag '%s' in scene %s — no matching Character entity in Production Bible",
                    tag,
                    entry.get("scene_number", "?"),
                )
                all_chars_resolved = False
        entry["_resolved_characters"] = all_chars_resolved

        # Set
        set_ref = entry.get("set_ref")
        if set_ref:
            if set_ref not in set_tags:
                logger.warning(
                    "Unresolved tag '%s' in scene %s — no matching Set entity in Production Bible",
                    set_ref,
                    entry.get("scene_number", "?"),
                )
                entry["_resolved_set"] = False
            else:
                entry["_resolved_set"] = True
        else:
            entry["_resolved_set"] = True  # No set_ref to validate

        # Props
        props_req = entry.get("props_required", [])
        all_props_resolved = True
        for tag in props_req:
            if tag not in prop_tags:
                logger.warning(
                    "Unresolved tag '%s' in scene %s — no matching Prop entity in Production Bible",
                    tag,
                    entry.get("scene_number", "?"),
                )
                all_props_resolved = False
        entry["_resolved_props"] = all_props_resolved

    return breakdown_entries


# ---------------------------------------------------------------------------
# ScreenwriterService
# ---------------------------------------------------------------------------


class ScreenwriterService:
    """Generates screenplay components via sequential LLM chain.

    Each method:
    1. Checks screenplay.status != "LOCKED"
    2. Builds a prompt with Production Bible context
    3. Sets screenplay.generating_step for progress polling
    4. Calls adapter.generate_text() with the appropriate schema
    5. Updates the relevant Screenplay field
    6. Clears generating_step
    7. Commits independently

    Constructor accepts an LLMAdapter instance, constructed by the caller
    via get_adapter(screenplay.text_model or settings.models.storyboard_llm, user_settings).
    """

    def __init__(self, adapter: LLMAdapter):
        self._adapter = adapter

    def _check_locked(self, screenplay: Screenplay) -> None:
        """Raise ValueError if screenplay is LOCKED."""
        if screenplay.status == "LOCKED":
            raise ValueError("Screenplay is LOCKED — regeneration blocked")

    async def generate_logline(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
    ) -> None:
        """Generate logline and genre from production context.

        Temperature 0.8 for creative generation.
        """
        self._check_locked(screenplay)

        screenplay.generating_step = "logline"
        await session.commit()

        prompt = _build_logline_prompt(screenplay, bible_context)
        result: LoglineOutput = await self._adapter.generate_text(
            prompt=prompt,
            schema=LoglineOutput,
            temperature=0.8,
            max_retries=3,
        )

        screenplay.logline = result.logline
        screenplay.genre = result.genre
        screenplay.generating_step = None
        await session.commit()

        logger.info("Generated logline for screenplay %s", screenplay.id)

    async def generate_treatment(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
    ) -> None:
        """Generate treatment from logline. Requires logline to exist.

        Temperature 0.7 for structured creative output.
        """
        self._check_locked(screenplay)
        if not screenplay.logline:
            raise ValueError("Cannot generate treatment: logline is required")

        screenplay.generating_step = "treatment"
        await session.commit()

        prompt = _build_treatment_prompt(screenplay, bible_context)
        result: TreatmentOutput = await self._adapter.generate_text(
            prompt=prompt,
            schema=TreatmentOutput,
            temperature=0.7,
            max_retries=3,
        )

        screenplay.treatment = result.treatment
        screenplay.generating_step = None
        await session.commit()

        logger.info("Generated treatment for screenplay %s", screenplay.id)

    async def generate_character_breakdowns(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
    ) -> None:
        """Generate character breakdowns. Requires logline + treatment.

        Stores result as list of dicts in screenplay.character_breakdowns.
        Temperature 0.7.
        """
        self._check_locked(screenplay)
        if not screenplay.logline or not screenplay.treatment:
            raise ValueError("Cannot generate character breakdowns: logline and treatment are required")

        screenplay.generating_step = "character_breakdowns"
        await session.commit()

        prompt = _build_character_breakdowns_prompt(screenplay, bible_context)
        result: CharacterBreakdownsOutput = await self._adapter.generate_text(
            prompt=prompt,
            schema=CharacterBreakdownsOutput,
            temperature=0.7,
            max_retries=3,
        )

        screenplay.character_breakdowns = [c.model_dump() for c in result.characters]
        screenplay.generating_step = None
        await session.commit()

        logger.info(
            "Generated %d character breakdowns for screenplay %s",
            len(result.characters),
            screenplay.id,
        )

    async def generate_scene_breakdown(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
        production_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Generate scene breakdown. Requires logline + treatment + character_breakdowns.

        If production_id is provided, validates entity tags against Production Bible
        entities (SCRN-06) and annotates entries with resolution flags.

        Stores breakdown entries as list of dicts in screenplay.scene_breakdown.
        Temperature 0.7.
        """
        self._check_locked(screenplay)
        if not screenplay.logline or not screenplay.treatment or not screenplay.character_breakdowns:
            raise ValueError(
                "Cannot generate scene breakdown: logline, treatment, and "
                "character breakdowns are required"
            )

        screenplay.generating_step = "scene_breakdown"
        await session.commit()

        prompt = _build_scene_breakdown_prompt(screenplay, bible_context)
        result: SceneBreakdownOutput = await self._adapter.generate_text(
            prompt=prompt,
            schema=SceneBreakdownOutput,
            temperature=0.7,
            max_retries=3,
        )

        entries = [s.model_dump() for s in result.scenes]

        # Post-LLM entity validation (SCRN-06)
        if production_id is not None:
            # Find the production bible for this production
            from vidpipe.db.models import Scene as SceneModel

            pb_result = await session.execute(
                select(SceneModel.production_bible_id)
                .where(
                    SceneModel.production_id == production_id,
                    SceneModel.production_bible_id.isnot(None),
                )
                .limit(1)
            )
            pb_id = pb_result.scalar_one_or_none()
            if pb_id:
                entries = await _validate_breakdown_entities(session, entries, pb_id)

        screenplay.scene_breakdown = entries
        screenplay.generating_step = None
        await session.commit()

        logger.info(
            "Generated %d scene breakdowns for screenplay %s",
            len(entries),
            screenplay.id,
        )

    async def generate_shot_list(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
    ) -> None:
        """Generate shot list. Requires scene_breakdown to exist.

        Independently regeneratable per SCRN-09.
        Stores result as list of dicts in screenplay.shot_list.
        Temperature 0.7.
        """
        self._check_locked(screenplay)
        if not screenplay.scene_breakdown:
            raise ValueError("Cannot generate shot list: scene breakdown is required")

        screenplay.generating_step = "shot_list"
        await session.commit()

        prompt = _build_shot_list_prompt(screenplay, bible_context)
        result: ShotListOutput = await self._adapter.generate_text(
            prompt=prompt,
            schema=ShotListOutput,
            temperature=0.7,
            max_retries=3,
        )

        screenplay.shot_list = [s.model_dump() for s in result.shots]
        screenplay.generating_step = None
        await session.commit()

        logger.info(
            "Generated %d shots in shot list for screenplay %s",
            len(result.shots),
            screenplay.id,
        )

    async def generate_script(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
    ) -> None:
        """Generate full script. Requires all prior steps.

        For long productions, passes a summarized form of prior steps to avoid
        context window overflow.
        Temperature 0.7.
        """
        self._check_locked(screenplay)
        if not screenplay.logline or not screenplay.treatment or not screenplay.scene_breakdown:
            raise ValueError(
                "Cannot generate script: logline, treatment, and scene breakdown are required"
            )

        screenplay.generating_step = "script"
        await session.commit()

        prompt = _build_script_prompt(screenplay, bible_context)
        result: ScriptOutput = await self._adapter.generate_text(
            prompt=prompt,
            schema=ScriptOutput,
            temperature=0.7,
            max_retries=3,
        )

        screenplay.script = result.script
        screenplay.generating_step = None
        await session.commit()

        logger.info("Generated script for screenplay %s", screenplay.id)

    async def generate_full(
        self,
        session: AsyncSession,
        screenplay: Screenplay,
        bible_context: str = "",
        production_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Run all 6 generation steps in sequence.

        logline -> treatment -> character_breakdowns -> scene_breakdown ->
        shot_list -> script

        Each step commits independently so users see progress immediately.
        Passes production_id to generate_scene_breakdown for entity validation.
        """
        await self.generate_logline(session, screenplay, bible_context)
        await self.generate_treatment(session, screenplay, bible_context)
        await self.generate_character_breakdowns(session, screenplay, bible_context)
        await self.generate_scene_breakdown(session, screenplay, bible_context, production_id)
        await self.generate_shot_list(session, screenplay, bible_context)
        await self.generate_script(session, screenplay, bible_context)
