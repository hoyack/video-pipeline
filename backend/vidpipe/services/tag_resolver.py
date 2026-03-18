"""Tag resolver service for scene prompt enrichment.

Resolves [CHAR:TAG], [SET:TAG], [PROP:TAG] and @tag placeholders in prompts
to bound asset data from the Asset Library via CastBinding, SetBinding,
and PropBinding lookup.

@tag syntax performs cross-type lookup: CastBinding -> PropBinding -> SetBinding.
[TYPE:TAG] syntax uses explicit type-scoped resolution.

Spec reference: Phase 22 - ALIB-07, Phase 23 - ATAG-01/02/03
"""

import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import (
    Actor,
    ActorRef,
    ActorWardrobePreset,
    CastBinding,
    CastLook,
    LibraryProp,
    LibraryPropRef,
    LibrarySet,
    LibrarySetRef,
    PropBinding,
    SetBinding,
)

logger = logging.getLogger(__name__)

TAG_PATTERN = re.compile(r"\[(CHAR|SET|PROP):([A-Z0-9_]+)\]")
AT_TAG_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")


@dataclass
class ResolvedAssetRef:
    """Structured asset metadata for a resolved tag.

    Carries all information downstream image generation needs about a
    bound asset: text description for prompt injection, reference images
    for visual conditioning, and optional overrides.
    """

    tag: str  # Uppercase-normalized tag name
    asset_type: str  # CHARACTER, PROP, SET
    display_name: str
    text_description: str  # Actor.base_appearance_prompt / LibraryProp.appearance_prompt / LibrarySet.reverse_prompt
    reference_image_urls: list[str] = field(default_factory=list)  # From ActorRef / LibraryPropRef / LibrarySetRef
    lora_url: str | None = None  # Phase 25: S3 path to trained .safetensors
    wardrobe_override: dict | None = None  # From CastBinding.wardrobe_override
    lighting_notes: str | None = None  # From SetBinding.lighting_override or LibrarySet.lighting_notes


def has_tags(text: str) -> bool:
    """Return True if text contains any [CHAR:TAG], [SET:TAG], or [PROP:TAG] patterns."""
    if not text:
        return False
    return bool(TAG_PATTERN.search(text))


def has_any_tags(text: str) -> bool:
    """Return True if text contains @tag or [TYPE:TAG] patterns."""
    if not text:
        return False
    return bool(TAG_PATTERN.search(text) or AT_TAG_PATTERN.search(text))


@dataclass
class ResolvedPrompt:
    """Result of tag resolution on a prompt string."""

    text: str
    asset_refs: list[ResolvedAssetRef] = field(default_factory=list)
    character_refs: list = field(default_factory=list)  # [{actor_id, ref_urls}]
    set_context: list = field(default_factory=list)  # [{set_id, lighting, style}]
    unresolved_tags: list = field(default_factory=list)  # tags that didn't match


async def resolve_tags(
    prompt: str,
    production_bible_id: uuid.UUID,
    session: AsyncSession,
) -> ResolvedPrompt:
    """Resolve [CHAR:TAG], [SET:TAG], [PROP:TAG] and @tag to bound asset data.

    Processes [TYPE:TAG] matches first (explicit type), then @tag matches
    (cross-type lookup: CastBinding -> PropBinding -> SetBinding).
    Deduplicates: if both [CHAR:BRANDON] and @brandon appear, the @tag
    is skipped since the typed tag already resolved it.

    Unmatched tags are left in the unresolved list but removed from the text
    to avoid polluting generation prompts.

    Args:
        prompt: The raw prompt text containing tag placeholders
        production_bible_id: UUID of the production bible to resolve bindings from
        session: Async SQLAlchemy session

    Returns:
        ResolvedPrompt with substituted text and collected references
    """
    typed_matches = TAG_PATTERN.findall(prompt)
    at_matches = AT_TAG_PATTERN.findall(prompt)

    if not typed_matches and not at_matches:
        return ResolvedPrompt(text=prompt)

    character_refs: list[dict] = []
    set_context: list[dict] = []
    unresolved_tags: list[str] = []
    resolved_text = prompt
    resolved_tag_names: set[str] = set()  # Track resolved tags (uppercase) for dedup

    # Phase 1: Process [TYPE:TAG] matches first (more specific, explicit type)
    for tag_type, tag_name in typed_matches:
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
            resolved_tag_names.add(tag_name.upper())
        else:
            unresolved_tags.append(full_tag)
            # Remove unresolved tag from text to avoid polluting prompts
            resolved_text = resolved_text.replace(full_tag, tag_name, 1)

    # Phase 2: Process @tag matches (cross-type lookup)
    for tag_name in at_matches:
        upper_tag = tag_name.upper()

        # Skip if already resolved by [TYPE:TAG]
        if upper_tag in resolved_tag_names:
            # Remove the @tag from text since it was already resolved
            resolved_text = resolved_text.replace(f"@{tag_name}", "", 1)
            continue

        replacement, resolved_type = await _resolve_at_tag_cross_type(
            tag_name, production_bible_id, session, character_refs, set_context
        )

        if replacement is not None:
            resolved_text = resolved_text.replace(f"@{tag_name}", replacement, 1)
            resolved_tag_names.add(upper_tag)
        else:
            unresolved_tags.append(f"@{tag_name}")
            # Leave @tag as-is in text for unresolved tags
            logger.warning("Unresolved @tag '%s' in prompt", tag_name)

    if unresolved_tags:
        logger.warning("Unresolved tags in prompt: %s", unresolved_tags)

    return ResolvedPrompt(
        text=resolved_text,
        character_refs=character_refs,
        set_context=set_context,
        unresolved_tags=unresolved_tags,
    )


async def _resolve_at_tag_cross_type(
    tag_name: str,
    bible_id: uuid.UUID,
    session: AsyncSession,
    character_refs: list[dict],
    set_context: list[dict],
) -> tuple[str | None, str | None]:
    """Cross-type lookup for @tag: CastLook -> CastBinding -> PropBinding -> SetBinding.

    Returns (replacement_text, asset_type) or (None, None).
    Case-insensitive matching via func.upper() on the tag field.
    """
    upper_tag = tag_name.upper()

    # 0. CastLook first (wardrobe-specific character look)
    look_result = await session.execute(
        select(CastLook).join(CastBinding).where(
            CastBinding.production_bible_id == bible_id,
            func.upper(CastLook.tag) == upper_tag,
        )
    )
    cast_look = look_result.scalars().first()
    if cast_look:
        # Resolve via parent CastBinding's character identity + wardrobe description
        binding = await session.get(CastBinding, cast_look.cast_binding_id)
        if binding:
            actor = await session.get(Actor, binding.actor_id)
            if actor:
                appearance = actor.base_appearance_prompt or ""
                wp = await session.get(ActorWardrobePreset, cast_look.wardrobe_preset_id)
                wardrobe_desc = (wp.description or wp.label) if wp else ""
                text = f"{appearance}; wearing: {wardrobe_desc}" if appearance else f"wearing: {wardrobe_desc}"
                replacement = f"{binding.character_name} ({text})"
                return replacement, "CHARACTER"

    match_count = 0

    # 1. CastBinding (character identity is highest priority)
    result = await session.execute(
        select(CastBinding).where(
            CastBinding.production_bible_id == bible_id,
            func.upper(CastBinding.tag) == upper_tag,
        )
    )
    cast_binding = result.scalars().first()
    if cast_binding:
        match_count += 1

    # 2. PropBinding second
    result = await session.execute(
        select(PropBinding).where(
            PropBinding.production_bible_id == bible_id,
            func.upper(PropBinding.tag) == upper_tag,
        )
    )
    prop_binding = result.scalars().first()
    if prop_binding:
        match_count += 1

    # 3. SetBinding last
    result = await session.execute(
        select(SetBinding).where(
            SetBinding.production_bible_id == bible_id,
            func.upper(SetBinding.tag) == upper_tag,
        )
    )
    set_binding = result.scalars().first()
    if set_binding:
        match_count += 1

    # Log warning if @tag matches multiple binding types
    if match_count > 1:
        logger.warning(
            "@tag '%s' matches %d binding types (cast=%s, prop=%s, set=%s). "
            "Using first match per priority order.",
            tag_name, match_count,
            cast_binding is not None, prop_binding is not None, set_binding is not None,
        )

    # Resolve using priority: CastBinding -> PropBinding -> SetBinding
    if cast_binding:
        replacement = await _resolve_char_tag(
            cast_binding.tag, bible_id, session, character_refs
        )
        return replacement, "CHARACTER"

    if prop_binding:
        replacement = await _resolve_prop_tag(
            prop_binding.tag, bible_id, session
        )
        return replacement, "PROP"

    if set_binding:
        replacement = await _resolve_set_tag(
            set_binding.tag, bible_id, session, set_context
        )
        return replacement, "SET"

    return None, None


async def resolve_tags_with_assets(
    prompt: str,
    production_bible_id: uuid.UUID,
    session: AsyncSession,
) -> ResolvedPrompt:
    """Resolve tags and populate ResolvedAssetRef objects with structured metadata.

    This is a superset of resolve_tags() -- it resolves tag text AND builds
    ResolvedAssetRef objects with full asset metadata from the library entities.

    Uses batch pre-loading: 3 queries for bindings + bulk entity/ref queries
    to avoid N+1 query problems.

    Args:
        prompt: The raw prompt text containing tag placeholders
        production_bible_id: UUID of the production bible to resolve bindings from
        session: Async SQLAlchemy session

    Returns:
        ResolvedPrompt with substituted text, asset_refs, character_refs,
        set_context, and unresolved_tags all populated.
    """
    typed_matches = TAG_PATTERN.findall(prompt)
    at_matches = AT_TAG_PATTERN.findall(prompt)

    if not typed_matches and not at_matches:
        return ResolvedPrompt(text=prompt)

    # --- Batch pre-load all bindings for this bible (3 queries) ---
    cast_result = await session.execute(
        select(CastBinding).where(CastBinding.production_bible_id == production_bible_id)
    )
    cast_bindings = cast_result.scalars().all()
    cast_by_tag: dict[str, CastBinding] = {b.tag.upper(): b for b in cast_bindings}

    prop_result = await session.execute(
        select(PropBinding).where(PropBinding.production_bible_id == production_bible_id)
    )
    prop_bindings = prop_result.scalars().all()
    prop_by_tag: dict[str, PropBinding] = {b.tag.upper(): b for b in prop_bindings}

    set_result = await session.execute(
        select(SetBinding).where(SetBinding.production_bible_id == production_bible_id)
    )
    set_bindings = set_result.scalars().all()
    set_by_tag: dict[str, SetBinding] = {b.tag.upper(): b for b in set_bindings}

    # --- Load CastLooks for wardrobe-specific tags ---
    looks_by_tag: dict[str, CastLook] = {}
    default_look_by_binding: dict[uuid.UUID, CastLook] = {}
    wardrobe_presets_by_id: dict[uuid.UUID, ActorWardrobePreset] = {}
    if cast_bindings:
        binding_ids = [b.id for b in cast_bindings]
        looks_result = await session.execute(
            select(CastLook).where(CastLook.cast_binding_id.in_(binding_ids))
        )
        all_looks = looks_result.scalars().all()
        wp_ids_needed: set[uuid.UUID] = set()
        for lk in all_looks:
            looks_by_tag[lk.tag.upper()] = lk
            if lk.is_default:
                default_look_by_binding[lk.cast_binding_id] = lk
            wp_ids_needed.add(lk.wardrobe_preset_id)
        if wp_ids_needed:
            wp_result = await session.execute(
                select(ActorWardrobePreset).where(ActorWardrobePreset.id.in_(list(wp_ids_needed)))
            )
            wardrobe_presets_by_id = {wp.id: wp for wp in wp_result.scalars().all()}

    # --- Batch-load all referenced entities and their refs ---
    # Actors
    actor_ids = [b.actor_id for b in cast_bindings]
    actors_by_id: dict[uuid.UUID, Actor] = {}
    actor_refs_by_actor_id: dict[uuid.UUID, list[ActorRef]] = defaultdict(list)
    if actor_ids:
        actor_result = await session.execute(
            select(Actor).where(Actor.id.in_(actor_ids))
        )
        actors_by_id = {a.id: a for a in actor_result.scalars().all()}
        ref_result = await session.execute(
            select(ActorRef).where(ActorRef.actor_id.in_(actor_ids))
        )
        for r in ref_result.scalars().all():
            actor_refs_by_actor_id[r.actor_id].append(r)

    # LibraryProps
    prop_ids = [b.library_prop_id for b in prop_bindings]
    props_by_id: dict[uuid.UUID, LibraryProp] = {}
    prop_refs_by_prop_id: dict[uuid.UUID, list[LibraryPropRef]] = defaultdict(list)
    if prop_ids:
        prop_entity_result = await session.execute(
            select(LibraryProp).where(LibraryProp.id.in_(prop_ids))
        )
        props_by_id = {p.id: p for p in prop_entity_result.scalars().all()}
        prop_ref_result = await session.execute(
            select(LibraryPropRef).where(LibraryPropRef.library_prop_id.in_(prop_ids))
        )
        for r in prop_ref_result.scalars().all():
            prop_refs_by_prop_id[r.library_prop_id].append(r)

    # LibrarySets
    set_ids = [b.library_set_id for b in set_bindings]
    sets_by_id: dict[uuid.UUID, LibrarySet] = {}
    set_refs_by_set_id: dict[uuid.UUID, list[LibrarySetRef]] = defaultdict(list)
    if set_ids:
        set_entity_result = await session.execute(
            select(LibrarySet).where(LibrarySet.id.in_(set_ids))
        )
        sets_by_id = {s.id: s for s in set_entity_result.scalars().all()}
        set_ref_result = await session.execute(
            select(LibrarySetRef).where(LibrarySetRef.library_set_id.in_(set_ids))
        )
        for r in set_ref_result.scalars().all():
            set_refs_by_set_id[r.library_set_id].append(r)

    # --- Resolve tags ---
    character_refs: list[dict] = []
    set_context: list[dict] = []
    asset_refs: list[ResolvedAssetRef] = []
    unresolved_tags: list[str] = []
    resolved_text = prompt
    resolved_tag_names: set[str] = set()

    # Phase 1: Process [TYPE:TAG] matches first
    for tag_type, tag_name in typed_matches:
        full_tag = f"[{tag_type}:{tag_name}]"
        upper_tag = tag_name.upper()

        if tag_type == "CHAR" and upper_tag in cast_by_tag:
            binding = cast_by_tag[upper_tag]
            replacement, asset_ref = _build_char_resolution(
                binding, actors_by_id, actor_refs_by_actor_id, character_refs
            )
            if replacement is not None:
                resolved_text = resolved_text.replace(full_tag, replacement, 1)
                resolved_tag_names.add(upper_tag)
                if asset_ref:
                    asset_refs.append(asset_ref)
                continue

        elif tag_type == "SET" and upper_tag in set_by_tag:
            binding = set_by_tag[upper_tag]
            replacement, asset_ref = _build_set_resolution(
                binding, sets_by_id, set_refs_by_set_id, set_context
            )
            if replacement is not None:
                resolved_text = resolved_text.replace(full_tag, replacement, 1)
                resolved_tag_names.add(upper_tag)
                if asset_ref:
                    asset_refs.append(asset_ref)
                continue

        elif tag_type == "PROP" and upper_tag in prop_by_tag:
            binding = prop_by_tag[upper_tag]
            replacement, asset_ref = _build_prop_resolution(
                binding, props_by_id, prop_refs_by_prop_id
            )
            if replacement is not None:
                resolved_text = resolved_text.replace(full_tag, replacement, 1)
                resolved_tag_names.add(upper_tag)
                if asset_ref:
                    asset_refs.append(asset_ref)
                continue

        # Unresolved typed tag
        unresolved_tags.append(full_tag)
        resolved_text = resolved_text.replace(full_tag, tag_name, 1)

    # Phase 2: Process @tag matches (cross-type lookup from pre-loaded dicts)
    # Priority: CastLook -> CastBinding -> PropBinding -> SetBinding
    for tag_name in at_matches:
        upper_tag = tag_name.upper()

        if upper_tag in resolved_tag_names:
            resolved_text = resolved_text.replace(f"@{tag_name}", "", 1)
            continue

        resolved = False

        # Check CastLook tags first (wardrobe-specific looks)
        if upper_tag in looks_by_tag:
            look = looks_by_tag[upper_tag]
            # Find the parent CastBinding
            parent_binding = None
            for cb in cast_bindings:
                if cb.id == look.cast_binding_id:
                    parent_binding = cb
                    break
            if parent_binding:
                replacement, asset_ref = _build_look_resolution(
                    look, parent_binding, actors_by_id, wardrobe_presets_by_id, character_refs
                )
                if replacement is not None:
                    resolved_text = resolved_text.replace(f"@{tag_name}", replacement, 1)
                    resolved_tag_names.add(upper_tag)
                    if asset_ref:
                        asset_refs.append(asset_ref)
                    resolved = True

        # Cross-type lookup: CastBinding -> PropBinding -> SetBinding
        if not resolved:
            match_types = []
            if upper_tag in cast_by_tag:
                match_types.append("CHAR")
            if upper_tag in prop_by_tag:
                match_types.append("PROP")
            if upper_tag in set_by_tag:
                match_types.append("SET")

            if len(match_types) > 1:
                logger.warning(
                    "@tag '%s' matches %d binding types: %s. Using first match per priority.",
                    tag_name, len(match_types), match_types,
                )

        if not resolved and upper_tag in cast_by_tag:
            binding = cast_by_tag[upper_tag]
            # Check if this binding has a default look override
            default_lk = default_look_by_binding.get(binding.id)
            if default_lk:
                replacement, asset_ref = _build_look_resolution(
                    default_lk, binding, actors_by_id, wardrobe_presets_by_id, character_refs
                )
            else:
                replacement, asset_ref = _build_char_resolution(
                    binding, actors_by_id, actor_refs_by_actor_id, character_refs
                )
            if replacement is not None:
                resolved_text = resolved_text.replace(f"@{tag_name}", replacement, 1)
                resolved_tag_names.add(upper_tag)
                if asset_ref:
                    asset_refs.append(asset_ref)
                resolved = True

        if not resolved and upper_tag in prop_by_tag:
            binding = prop_by_tag[upper_tag]
            replacement, asset_ref = _build_prop_resolution(
                binding, props_by_id, prop_refs_by_prop_id
            )
            if replacement is not None:
                resolved_text = resolved_text.replace(f"@{tag_name}", replacement, 1)
                resolved_tag_names.add(upper_tag)
                if asset_ref:
                    asset_refs.append(asset_ref)
                resolved = True

        if not resolved and upper_tag in set_by_tag:
            binding = set_by_tag[upper_tag]
            replacement, asset_ref = _build_set_resolution(
                binding, sets_by_id, set_refs_by_set_id, set_context
            )
            if replacement is not None:
                resolved_text = resolved_text.replace(f"@{tag_name}", replacement, 1)
                resolved_tag_names.add(upper_tag)
                if asset_ref:
                    asset_refs.append(asset_ref)
                resolved = True

        if not resolved:
            unresolved_tags.append(f"@{tag_name}")
            logger.warning("Unresolved @tag '%s' in prompt", tag_name)

    if unresolved_tags:
        logger.warning("Unresolved tags in prompt: %s", unresolved_tags)

    return ResolvedPrompt(
        text=resolved_text,
        asset_refs=asset_refs,
        character_refs=character_refs,
        set_context=set_context,
        unresolved_tags=unresolved_tags,
    )


# ---------------------------------------------------------------------------
# Build helpers for resolve_tags_with_assets (batch-loaded, no DB queries)
# ---------------------------------------------------------------------------

def _build_char_resolution(
    binding: CastBinding,
    actors_by_id: dict[uuid.UUID, Actor],
    actor_refs_by_actor_id: dict[uuid.UUID, list],
    character_refs: list[dict],
) -> tuple[str | None, ResolvedAssetRef | None]:
    """Build replacement text and ResolvedAssetRef for a CHARACTER binding."""
    actor = actors_by_id.get(binding.actor_id)
    if actor is None:
        # Deleted entity -- treat as partially resolved (name only)
        logger.warning("Actor %s referenced by CastBinding %s not found", binding.actor_id, binding.id)
        return binding.character_name, None

    ref_urls = [r.image_url for r in actor_refs_by_actor_id.get(binding.actor_id, [])]
    if ref_urls:
        character_refs.append({"actor_id": str(actor.id), "ref_urls": ref_urls})

    appearance = actor.base_appearance_prompt or ""
    replacement = f"{binding.character_name} ({appearance})" if appearance else binding.character_name

    asset_ref = ResolvedAssetRef(
        tag=binding.tag.upper(),
        asset_type="CHARACTER",
        display_name=binding.character_name,
        text_description=actor.base_appearance_prompt or "",
        reference_image_urls=ref_urls,
        lora_url=getattr(actor, 'lora_url', None),
        wardrobe_override=binding.wardrobe_override,
        lighting_notes=None,
    )
    return replacement, asset_ref


def _build_look_resolution(
    look: CastLook,
    binding: CastBinding,
    actors_by_id: dict[uuid.UUID, Actor],
    wardrobe_presets_by_id: dict[uuid.UUID, ActorWardrobePreset],
    character_refs: list[dict],
) -> tuple[str | None, ResolvedAssetRef | None]:
    """Build replacement text and ResolvedAssetRef for a CastLook (wardrobe-specific tag)."""
    actor = actors_by_id.get(binding.actor_id)
    if actor is None:
        logger.warning("Actor %s referenced by CastBinding %s not found", binding.actor_id, binding.id)
        return binding.character_name, None

    preset = wardrobe_presets_by_id.get(look.wardrobe_preset_id)
    if preset is None:
        logger.warning("WardrobePreset %s referenced by CastLook %s not found", look.wardrobe_preset_id, look.id)
        # Fall back to base actor resolution
        return binding.character_name, None

    # Use wardrobe preset's reference_images for visual conditioning
    ref_urls = list(preset.reference_images or [])
    if ref_urls:
        character_refs.append({"actor_id": str(actor.id), "ref_urls": ref_urls})

    # Build combined text description: base appearance + wardrobe
    appearance = actor.base_appearance_prompt or ""
    wardrobe_desc = preset.description or preset.label
    text_description = f"{appearance}; wearing: {wardrobe_desc}" if appearance else f"wearing: {wardrobe_desc}"
    replacement = f"{binding.character_name} ({text_description})"

    asset_ref = ResolvedAssetRef(
        tag=look.tag.upper(),
        asset_type="CHARACTER",
        display_name=binding.character_name,
        text_description=text_description,
        reference_image_urls=ref_urls,
        lora_url=getattr(actor, 'lora_url', None),
        wardrobe_override={"wardrobe_description": wardrobe_desc},
        lighting_notes=None,
    )
    return replacement, asset_ref


def _build_set_resolution(
    binding: SetBinding,
    sets_by_id: dict[uuid.UUID, LibrarySet],
    set_refs_by_set_id: dict[uuid.UUID, list],
    set_context: list[dict],
) -> tuple[str | None, ResolvedAssetRef | None]:
    """Build replacement text and ResolvedAssetRef for a SET binding."""
    lib_set = sets_by_id.get(binding.library_set_id)
    if lib_set is None:
        logger.warning("LibrarySet %s referenced by SetBinding %s not found", binding.library_set_id, binding.id)
        display_name = binding.production_name or binding.tag
        return display_name, None

    # Collect set context
    context_entry: dict = {"set_id": str(lib_set.id)}
    if binding.lighting_override:
        context_entry["lighting"] = binding.lighting_override
    elif lib_set.lighting_notes:
        context_entry["lighting"] = lib_set.lighting_notes
    if lib_set.style_tags:
        context_entry["style"] = lib_set.style_tags
    set_context.append(context_entry)

    display_name = binding.production_name or lib_set.name
    reverse = lib_set.reverse_prompt or ""
    replacement = f"{display_name} ({reverse})" if reverse else display_name

    ref_urls = [r.image_url for r in set_refs_by_set_id.get(binding.library_set_id, [])]

    asset_ref = ResolvedAssetRef(
        tag=binding.tag.upper(),
        asset_type="SET",
        display_name=display_name,
        text_description=lib_set.reverse_prompt or "",
        reference_image_urls=ref_urls,
        lora_url=None,
        wardrobe_override=None,
        lighting_notes=binding.lighting_override or lib_set.lighting_notes,
    )
    return replacement, asset_ref


def _build_prop_resolution(
    binding: PropBinding,
    props_by_id: dict[uuid.UUID, LibraryProp],
    prop_refs_by_prop_id: dict[uuid.UUID, list],
) -> tuple[str | None, ResolvedAssetRef | None]:
    """Build replacement text and ResolvedAssetRef for a PROP binding."""
    lib_prop = props_by_id.get(binding.library_prop_id)
    if lib_prop is None:
        logger.warning("LibraryProp %s referenced by PropBinding %s not found", binding.library_prop_id, binding.id)
        display_name = binding.production_name or binding.tag
        return display_name, None

    display_name = binding.production_name or lib_prop.name
    appearance = lib_prop.appearance_prompt or ""
    replacement = f"{display_name} ({appearance})" if appearance else display_name

    ref_urls = [r.image_url for r in prop_refs_by_prop_id.get(binding.library_prop_id, [])]

    asset_ref = ResolvedAssetRef(
        tag=binding.tag.upper(),
        asset_type="PROP",
        display_name=display_name,
        text_description=lib_prop.appearance_prompt or "",
        reference_image_urls=ref_urls,
        lora_url=None,
        wardrobe_override=None,
        lighting_notes=None,
    )
    return replacement, asset_ref


# ---------------------------------------------------------------------------
# Per-tag resolution helpers (used by resolve_tags for non-batch path)
# ---------------------------------------------------------------------------

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

    Uses has_any_tags() to detect both @tag and [TYPE:TAG] patterns.

    Args:
        scene_description: Scene description (may contain tags)
        start_frame_prompt: Start keyframe prompt
        end_frame_prompt: End keyframe prompt
        video_motion_prompt: Video motion prompt
        production_bible_id: UUID of the production bible
        session: Async SQLAlchemy session

    Returns:
        Dict with keys: scene_description, start_frame_prompt, end_frame_prompt,
        video_motion_prompt -- each resolved. Also 'character_refs' and 'set_context'
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
        if not text or not has_any_tags(text):
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
