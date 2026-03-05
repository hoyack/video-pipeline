"""Tag resolver service for scene prompt enrichment.

Resolves [CHAR:TAG], [SET:TAG], [PROP:TAG] placeholders in prompts
to bound asset data from the Asset Library via CastBinding, SetBinding,
and PropBinding lookup.

Spec reference: Phase 22 - ALIB-07
"""

import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import (
    Actor,
    ActorRef,
    CastBinding,
    LibraryProp,
    LibrarySet,
    PropBinding,
    SetBinding,
)

logger = logging.getLogger(__name__)

TAG_PATTERN = re.compile(r"\[(CHAR|SET|PROP):([A-Z0-9_]+)\]")


def has_tags(text: str) -> bool:
    """Return True if text contains any [CHAR:TAG], [SET:TAG], or [PROP:TAG] patterns."""
    if not text:
        return False
    return bool(TAG_PATTERN.search(text))


@dataclass
class ResolvedPrompt:
    """Result of tag resolution on a prompt string."""

    text: str
    character_refs: list = field(default_factory=list)  # [{actor_id, ref_urls}]
    set_context: list = field(default_factory=list)  # [{set_id, lighting, style}]
    unresolved_tags: list = field(default_factory=list)  # tags that didn't match


async def resolve_tags(
    prompt: str,
    production_bible_id: uuid.UUID,
    session: AsyncSession,
) -> ResolvedPrompt:
    """Resolve [CHAR:TAG], [SET:TAG], [PROP:TAG] to bound asset data.

    For each tag found:
    - CHAR: substitutes with "character_name (base_appearance_prompt)", collects actor ref URLs
    - SET: substitutes with "production_name/set.name (reverse_prompt)", collects lighting context
    - PROP: substitutes with "production_name/prop.name (appearance_prompt)"

    Unmatched tags are left in the unresolved list but removed from the text
    to avoid polluting generation prompts.

    Args:
        prompt: The raw prompt text containing [TYPE:TAG] placeholders
        production_bible_id: UUID of the production bible to resolve bindings from
        session: Async SQLAlchemy session

    Returns:
        ResolvedPrompt with substituted text and collected references
    """
    matches = TAG_PATTERN.findall(prompt)

    if not matches:
        return ResolvedPrompt(text=prompt)

    character_refs: list[dict] = []
    set_context: list[dict] = []
    unresolved_tags: list[str] = []
    resolved_text = prompt

    for tag_type, tag_name in matches:
        full_tag = f"[{tag_type}:{tag_name}]"

        if tag_type == "CHAR":
            replacement = await _resolve_char_tag(
                tag_name, production_bible_id, session, character_refs
            )
        elif tag_type == "SET":
            replacement = await _resolve_set_tag(
                tag_name, production_bible_id, session, set_context
            )
        elif tag_type == "PROP":
            replacement = await _resolve_prop_tag(
                tag_name, production_bible_id, session
            )
        else:
            replacement = None

        if replacement is not None:
            resolved_text = resolved_text.replace(full_tag, replacement, 1)
        else:
            unresolved_tags.append(full_tag)
            # Remove unresolved tag from text to avoid polluting prompts
            resolved_text = resolved_text.replace(full_tag, tag_name, 1)

    if unresolved_tags:
        logger.warning("Unresolved tags in prompt: %s", unresolved_tags)

    return ResolvedPrompt(
        text=resolved_text,
        character_refs=character_refs,
        set_context=set_context,
        unresolved_tags=unresolved_tags,
    )


async def _resolve_char_tag(
    tag_name: str,
    bible_id: uuid.UUID,
    session: AsyncSession,
    character_refs: list[dict],
) -> str | None:
    """Resolve a CHAR tag to character appearance text and collect actor refs."""
    result = await session.execute(
        select(CastBinding).where(
            CastBinding.production_bible_id == bible_id,
            CastBinding.tag == tag_name,
        )
    )
    binding = result.scalars().first()
    if binding is None:
        return None

    # Load the Actor for base_appearance_prompt
    actor = await session.get(Actor, binding.actor_id)
    if actor is None:
        return binding.character_name

    # Collect actor ref URLs
    ref_result = await session.execute(
        select(ActorRef).where(ActorRef.actor_id == actor.id)
    )
    ref_urls = [r.image_url for r in ref_result.scalars().all()]

    if ref_urls:
        character_refs.append(
            {"actor_id": str(actor.id), "ref_urls": ref_urls}
        )

    # Build substitution text
    appearance = actor.base_appearance_prompt or ""
    if appearance:
        return f"{binding.character_name} ({appearance})"
    return binding.character_name


async def _resolve_set_tag(
    tag_name: str,
    bible_id: uuid.UUID,
    session: AsyncSession,
    set_context: list[dict],
) -> str | None:
    """Resolve a SET tag to set description and collect lighting context."""
    result = await session.execute(
        select(SetBinding).where(
            SetBinding.production_bible_id == bible_id,
            SetBinding.tag == tag_name,
        )
    )
    binding = result.scalars().first()
    if binding is None:
        return None

    # Load the LibrarySet for reverse_prompt
    lib_set = await session.get(LibrarySet, binding.library_set_id)
    if lib_set is None:
        return binding.production_name or tag_name

    # Collect set context
    context_entry: dict = {"set_id": str(lib_set.id)}
    if binding.lighting_override:
        context_entry["lighting"] = binding.lighting_override
    elif lib_set.lighting_notes:
        context_entry["lighting"] = lib_set.lighting_notes
    if lib_set.style_tags:
        context_entry["style"] = lib_set.style_tags
    set_context.append(context_entry)

    # Build substitution text
    display_name = binding.production_name or lib_set.name
    reverse = lib_set.reverse_prompt or ""
    if reverse:
        return f"{display_name} ({reverse})"
    return display_name


async def _resolve_prop_tag(
    tag_name: str,
    bible_id: uuid.UUID,
    session: AsyncSession,
) -> str | None:
    """Resolve a PROP tag to prop appearance text."""
    result = await session.execute(
        select(PropBinding).where(
            PropBinding.production_bible_id == bible_id,
            PropBinding.tag == tag_name,
        )
    )
    binding = result.scalars().first()
    if binding is None:
        return None

    # Load the LibraryProp for appearance_prompt
    lib_prop = await session.get(LibraryProp, binding.library_prop_id)
    if lib_prop is None:
        return binding.production_name or tag_name

    # Build substitution text
    display_name = binding.production_name or lib_prop.name
    appearance = lib_prop.appearance_prompt or ""
    if appearance:
        return f"{display_name} ({appearance})"
    return display_name


async def resolve_scene_tags(
    scene_description: str | None,
    start_frame_prompt: str | None,
    end_frame_prompt: str | None,
    video_motion_prompt: str | None,
    production_bible_id: uuid.UUID,
    session: AsyncSession,
) -> dict[str, str]:
    """Resolve tags in all prompt fields for a scene/shot at once.

    Returns a dict with resolved versions of each field.
    Fields without tags or None fields are returned unchanged.
    This is a convenience wrapper to avoid calling resolve_tags 4 times.

    Args:
        scene_description: Scene description (may contain tags)
        start_frame_prompt: Start keyframe prompt
        end_frame_prompt: End keyframe prompt
        video_motion_prompt: Video motion prompt
        production_bible_id: UUID of the production bible
        session: Async SQLAlchemy session

    Returns:
        Dict with keys: scene_description, start_frame_prompt, end_frame_prompt,
        video_motion_prompt — each resolved. Also 'character_refs' and 'set_context'
        aggregated from all resolutions, and 'unresolved_tags'.
    """
    fields = {
        "scene_description": scene_description,
        "start_frame_prompt": start_frame_prompt,
        "end_frame_prompt": end_frame_prompt,
        "video_motion_prompt": video_motion_prompt,
    }

    result: dict = {}
    all_character_refs: list[dict] = []
    all_set_context: list[dict] = []
    all_unresolved: list[str] = []

    for key, text in fields.items():
        if not text or not has_tags(text):
            result[key] = text or ""
            continue

        resolved = await resolve_tags(text, production_bible_id, session)
        result[key] = resolved.text
        all_character_refs.extend(resolved.character_refs)
        all_set_context.extend(resolved.set_context)
        all_unresolved.extend(resolved.unresolved_tags)

    result["character_refs"] = all_character_refs
    result["set_context"] = all_set_context
    result["unresolved_tags"] = all_unresolved

    if all_unresolved:
        logger.warning(
            "Unresolved tags during scene tag resolution: %s", all_unresolved
        )

    return result
