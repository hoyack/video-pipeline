"""Binding CRUD API routes for Asset Library.

Bindings connect standalone library entities (Actor, LibrarySet, LibraryProp,
SoundAsset) to Production Bibles. Each binding type carries a tag for prompt
injection and optional overrides.

Spec reference: Phase 22 - ALIB-02, ALIB-05, ALIB-07
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from vidpipe.db import async_session
from vidpipe.db.models import (
    Actor,
    ActorRef,
    ActorVoiceProfile,
    ActorWardrobePreset,
    CastBinding,
    LibraryProp,
    LibraryPropRef,
    LibrarySet,
    LibrarySetRef,
    ProductionBible,
    PropBinding,
    SetBinding,
    SoundAsset,
    SoundBinding,
)

logger = logging.getLogger(__name__)

bindings_router = APIRouter(prefix="/api", tags=["bindings"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CastBindingCreate(BaseModel):
    actor_id: str
    tag: str
    character_name: str
    character_description: Optional[str] = None
    character_arc: Optional[str] = None
    role: str
    wardrobe_override: Optional[dict] = None
    voice_profile_id: Optional[str] = None
    behavioral_notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class CastBindingUpdate(BaseModel):
    tag: Optional[str] = None
    character_name: Optional[str] = None
    character_description: Optional[str] = None
    character_arc: Optional[str] = None
    role: Optional[str] = None
    wardrobe_override: Optional[dict] = None
    voice_profile_id: Optional[str] = None
    behavioral_notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class SetBindingCreate(BaseModel):
    library_set_id: str
    tag: str
    production_name: Optional[str] = None
    lighting_override: Optional[str] = None
    sonic_override: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class SetBindingUpdate(BaseModel):
    tag: Optional[str] = None
    production_name: Optional[str] = None
    lighting_override: Optional[str] = None
    sonic_override: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class PropBindingCreate(BaseModel):
    library_prop_id: str
    tag: str
    production_name: Optional[str] = None
    notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class PropBindingUpdate(BaseModel):
    tag: Optional[str] = None
    production_name: Optional[str] = None
    notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class SoundBindingCreate(BaseModel):
    sound_asset_id: str
    tag: Optional[str] = None
    usage_notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class SoundBindingUpdate(BaseModel):
    tag: Optional[str] = None
    usage_notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cast_binding_to_dict(
    binding: CastBinding,
    actor_name: Optional[str] = None,
    primary_ref_url: Optional[str] = None,
) -> dict:
    return {
        "id": str(binding.id),
        "production_bible_id": str(binding.production_bible_id),
        "actor_id": str(binding.actor_id),
        "tag": binding.tag,
        "character_name": binding.character_name,
        "character_description": binding.character_description,
        "character_arc": binding.character_arc,
        "role": binding.role,
        "wardrobe_override": binding.wardrobe_override,
        "voice_profile_id": str(binding.voice_profile_id) if binding.voice_profile_id else None,
        "behavioral_notes": binding.behavioral_notes,
        "prompt_tags": binding.prompt_tags,
        "actor_name": actor_name,
        "primary_ref_url": primary_ref_url,
        "created_at": binding.created_at.isoformat(),
        "updated_at": binding.updated_at.isoformat(),
    }


def _set_binding_to_dict(
    binding: SetBinding,
    set_name: Optional[str] = None,
    primary_ref_url: Optional[str] = None,
) -> dict:
    return {
        "id": str(binding.id),
        "production_bible_id": str(binding.production_bible_id),
        "library_set_id": str(binding.library_set_id),
        "tag": binding.tag,
        "production_name": binding.production_name,
        "lighting_override": binding.lighting_override,
        "sonic_override": binding.sonic_override,
        "prompt_tags": binding.prompt_tags,
        "set_name": set_name,
        "primary_ref_url": primary_ref_url,
        "created_at": binding.created_at.isoformat(),
        "updated_at": binding.updated_at.isoformat(),
    }


def _prop_binding_to_dict(
    binding: PropBinding,
    prop_name: Optional[str] = None,
    primary_ref_url: Optional[str] = None,
) -> dict:
    return {
        "id": str(binding.id),
        "production_bible_id": str(binding.production_bible_id),
        "library_prop_id": str(binding.library_prop_id),
        "tag": binding.tag,
        "production_name": binding.production_name,
        "notes": binding.notes,
        "prompt_tags": binding.prompt_tags,
        "prop_name": prop_name,
        "primary_ref_url": primary_ref_url,
        "created_at": binding.created_at.isoformat(),
        "updated_at": binding.updated_at.isoformat(),
    }


def _sound_binding_to_dict(
    binding: SoundBinding,
    sound_name: Optional[str] = None,
    sound_category: Optional[str] = None,
) -> dict:
    return {
        "id": str(binding.id),
        "production_bible_id": str(binding.production_bible_id),
        "sound_asset_id": str(binding.sound_asset_id),
        "tag": binding.tag,
        "usage_notes": binding.usage_notes,
        "prompt_tags": binding.prompt_tags,
        "sound_name": sound_name,
        "sound_category": sound_category,
        "created_at": binding.created_at.isoformat(),
        "updated_at": binding.updated_at.isoformat(),
    }


async def _validate_bible(session, bible_id: str) -> uuid.UUID:
    """Validate production bible exists and return its UUID."""
    uid = uuid.UUID(bible_id)
    bible = await session.get(ProductionBible, uid)
    if bible is None:
        raise HTTPException(status_code=404, detail="Production Bible not found")
    return uid


async def _check_tag_unique(session, model, bible_id: uuid.UUID, tag: str, exclude_id: uuid.UUID | None = None):
    """Check that tag is unique within a bible for a given binding model."""
    stmt = select(model).where(model.production_bible_id == bible_id, model.tag == tag)
    if exclude_id is not None:
        stmt = stmt.where(model.id != exclude_id)
    result = await session.execute(stmt)
    existing = result.scalars().first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Tag '{tag}' already exists in this production bible")


# ---------------------------------------------------------------------------
# CastBinding endpoints
# ---------------------------------------------------------------------------


@bindings_router.get("/production-bibles/{bible_id}/cast")
async def list_cast_bindings(bible_id: str):
    """List all cast bindings for a production bible with actor name and primary ref."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        result = await session.execute(
            select(CastBinding)
            .where(CastBinding.production_bible_id == uid)
            .order_by(CastBinding.created_at)
        )
        bindings = result.scalars().all()

        if not bindings:
            return []

        # Bulk fetch actor names and primary refs
        actor_ids = [b.actor_id for b in bindings]
        actor_result = await session.execute(
            select(Actor).where(Actor.id.in_(actor_ids))
        )
        actors_by_id = {a.id: a for a in actor_result.scalars().all()}

        ref_result = await session.execute(
            select(ActorRef).where(
                ActorRef.actor_id.in_(actor_ids),
                ActorRef.is_primary == True,  # noqa: E712
            )
        )
        primary_refs = {r.actor_id: r.image_url for r in ref_result.scalars().all()}

        return [
            _cast_binding_to_dict(
                b,
                actor_name=actors_by_id.get(b.actor_id, Actor(name="")).name if b.actor_id in actors_by_id else None,
                primary_ref_url=primary_refs.get(b.actor_id),
            )
            for b in bindings
        ]


@bindings_router.post("/production-bibles/{bible_id}/cast", status_code=201)
async def create_cast_binding(bible_id: str, body: CastBindingCreate):
    """Create a cast binding. Validates actor exists and tag uniqueness."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        # Validate actor exists
        actor = await session.get(Actor, uuid.UUID(body.actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        # Validate tag uniqueness
        await _check_tag_unique(session, CastBinding, uid, body.tag)

        binding = CastBinding(
            production_bible_id=uid,
            actor_id=uuid.UUID(body.actor_id),
            tag=body.tag,
            character_name=body.character_name,
            character_description=body.character_description,
            character_arc=body.character_arc,
            role=body.role,
            wardrobe_override=body.wardrobe_override,
            voice_profile_id=uuid.UUID(body.voice_profile_id) if body.voice_profile_id else None,
            behavioral_notes=body.behavioral_notes,
            prompt_tags=body.prompt_tags,
        )
        session.add(binding)
        await session.commit()
        await session.refresh(binding)

        return _cast_binding_to_dict(binding, actor_name=actor.name)


@bindings_router.get("/cast-bindings/{binding_id}")
async def get_cast_binding(binding_id: str):
    """Get a single cast binding with full actor detail."""
    async with async_session() as session:
        binding = await session.get(CastBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Cast binding not found")

        actor = await session.get(Actor, binding.actor_id)

        # Fetch refs, voice profiles, wardrobe presets
        ref_result = await session.execute(
            select(ActorRef).where(ActorRef.actor_id == binding.actor_id)
        )
        refs = [
            {"id": str(r.id), "image_url": r.image_url, "label": r.label, "is_primary": r.is_primary}
            for r in ref_result.scalars().all()
        ]

        vp_result = await session.execute(
            select(ActorVoiceProfile).where(ActorVoiceProfile.actor_id == binding.actor_id)
        )
        voice_profiles = [
            {"id": str(vp.id), "voice_id": vp.voice_id, "adapter_type": vp.adapter_type, "style_notes": vp.style_notes}
            for vp in vp_result.scalars().all()
        ]

        wp_result = await session.execute(
            select(ActorWardrobePreset).where(ActorWardrobePreset.actor_id == binding.actor_id)
        )
        wardrobe_presets = [
            {"id": str(wp.id), "label": wp.label, "description": wp.description}
            for wp in wp_result.scalars().all()
        ]

        result = _cast_binding_to_dict(
            binding,
            actor_name=actor.name if actor else None,
        )
        result["actor_detail"] = {
            "refs": refs,
            "voice_profiles": voice_profiles,
            "wardrobe_presets": wardrobe_presets,
            "description": actor.description if actor else None,
            "base_appearance_prompt": actor.base_appearance_prompt if actor else None,
        }
        return result


@bindings_router.put("/cast-bindings/{binding_id}")
async def update_cast_binding(binding_id: str, body: CastBindingUpdate):
    """Update cast binding fields."""
    async with async_session() as session:
        binding = await session.get(CastBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Cast binding not found")

        # If tag changed, validate uniqueness
        if body.tag is not None and body.tag != binding.tag:
            await _check_tag_unique(session, CastBinding, binding.production_bible_id, body.tag, exclude_id=binding.id)
            binding.tag = body.tag

        if body.character_name is not None:
            binding.character_name = body.character_name
        if "character_description" in body.model_fields_set:
            binding.character_description = body.character_description
        if "character_arc" in body.model_fields_set:
            binding.character_arc = body.character_arc
        if body.role is not None:
            binding.role = body.role
        if "wardrobe_override" in body.model_fields_set:
            binding.wardrobe_override = body.wardrobe_override
        if "voice_profile_id" in body.model_fields_set:
            binding.voice_profile_id = uuid.UUID(body.voice_profile_id) if body.voice_profile_id else None
        if "behavioral_notes" in body.model_fields_set:
            binding.behavioral_notes = body.behavioral_notes
        if "prompt_tags" in body.model_fields_set:
            binding.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(binding)

        return _cast_binding_to_dict(binding)


@bindings_router.delete("/cast-bindings/{binding_id}", status_code=204)
async def delete_cast_binding(binding_id: str):
    """Delete a cast binding."""
    async with async_session() as session:
        binding = await session.get(CastBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Cast binding not found")

        await session.delete(binding)
        await session.commit()
        return None


# ---------------------------------------------------------------------------
# SetBinding endpoints
# ---------------------------------------------------------------------------


@bindings_router.get("/production-bibles/{bible_id}/set-bindings")
async def list_set_bindings(bible_id: str):
    """List set bindings with library set name and primary ref."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        result = await session.execute(
            select(SetBinding)
            .where(SetBinding.production_bible_id == uid)
            .order_by(SetBinding.created_at)
        )
        bindings = result.scalars().all()

        if not bindings:
            return []

        set_ids = [b.library_set_id for b in bindings]
        set_result = await session.execute(
            select(LibrarySet).where(LibrarySet.id.in_(set_ids))
        )
        sets_by_id = {s.id: s for s in set_result.scalars().all()}

        ref_result = await session.execute(
            select(LibrarySetRef).where(
                LibrarySetRef.library_set_id.in_(set_ids),
                LibrarySetRef.is_primary == True,  # noqa: E712
            )
        )
        primary_refs = {r.library_set_id: r.image_url for r in ref_result.scalars().all()}

        return [
            _set_binding_to_dict(
                b,
                set_name=sets_by_id[b.library_set_id].name if b.library_set_id in sets_by_id else None,
                primary_ref_url=primary_refs.get(b.library_set_id),
            )
            for b in bindings
        ]


@bindings_router.post("/production-bibles/{bible_id}/set-bindings", status_code=201)
async def create_set_binding(bible_id: str, body: SetBindingCreate):
    """Create a set binding."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        lib_set = await session.get(LibrarySet, uuid.UUID(body.library_set_id))
        if lib_set is None:
            raise HTTPException(status_code=404, detail="Library set not found")

        await _check_tag_unique(session, SetBinding, uid, body.tag)

        binding = SetBinding(
            production_bible_id=uid,
            library_set_id=uuid.UUID(body.library_set_id),
            tag=body.tag,
            production_name=body.production_name,
            lighting_override=body.lighting_override,
            sonic_override=body.sonic_override,
            prompt_tags=body.prompt_tags,
        )
        session.add(binding)
        await session.commit()
        await session.refresh(binding)

        return _set_binding_to_dict(binding, set_name=lib_set.name)


@bindings_router.put("/set-bindings/{binding_id}")
async def update_set_binding(binding_id: str, body: SetBindingUpdate):
    """Update set binding fields."""
    async with async_session() as session:
        binding = await session.get(SetBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Set binding not found")

        if body.tag is not None and body.tag != binding.tag:
            await _check_tag_unique(session, SetBinding, binding.production_bible_id, body.tag, exclude_id=binding.id)
            binding.tag = body.tag

        if "production_name" in body.model_fields_set:
            binding.production_name = body.production_name
        if "lighting_override" in body.model_fields_set:
            binding.lighting_override = body.lighting_override
        if "sonic_override" in body.model_fields_set:
            binding.sonic_override = body.sonic_override
        if "prompt_tags" in body.model_fields_set:
            binding.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(binding)

        return _set_binding_to_dict(binding)


@bindings_router.delete("/set-bindings/{binding_id}", status_code=204)
async def delete_set_binding(binding_id: str):
    """Delete a set binding."""
    async with async_session() as session:
        binding = await session.get(SetBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Set binding not found")

        await session.delete(binding)
        await session.commit()
        return None


# ---------------------------------------------------------------------------
# PropBinding endpoints
# ---------------------------------------------------------------------------


@bindings_router.get("/production-bibles/{bible_id}/prop-bindings")
async def list_prop_bindings(bible_id: str):
    """List prop bindings with library prop name and primary ref."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        result = await session.execute(
            select(PropBinding)
            .where(PropBinding.production_bible_id == uid)
            .order_by(PropBinding.created_at)
        )
        bindings = result.scalars().all()

        if not bindings:
            return []

        prop_ids = [b.library_prop_id for b in bindings]
        prop_result = await session.execute(
            select(LibraryProp).where(LibraryProp.id.in_(prop_ids))
        )
        props_by_id = {p.id: p for p in prop_result.scalars().all()}

        ref_result = await session.execute(
            select(LibraryPropRef).where(
                LibraryPropRef.library_prop_id.in_(prop_ids),
                LibraryPropRef.is_primary == True,  # noqa: E712
            )
        )
        primary_refs = {r.library_prop_id: r.image_url for r in ref_result.scalars().all()}

        return [
            _prop_binding_to_dict(
                b,
                prop_name=props_by_id[b.library_prop_id].name if b.library_prop_id in props_by_id else None,
                primary_ref_url=primary_refs.get(b.library_prop_id),
            )
            for b in bindings
        ]


@bindings_router.post("/production-bibles/{bible_id}/prop-bindings", status_code=201)
async def create_prop_binding(bible_id: str, body: PropBindingCreate):
    """Create a prop binding."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        lib_prop = await session.get(LibraryProp, uuid.UUID(body.library_prop_id))
        if lib_prop is None:
            raise HTTPException(status_code=404, detail="Library prop not found")

        await _check_tag_unique(session, PropBinding, uid, body.tag)

        binding = PropBinding(
            production_bible_id=uid,
            library_prop_id=uuid.UUID(body.library_prop_id),
            tag=body.tag,
            production_name=body.production_name,
            notes=body.notes,
            prompt_tags=body.prompt_tags,
        )
        session.add(binding)
        await session.commit()
        await session.refresh(binding)

        return _prop_binding_to_dict(binding, prop_name=lib_prop.name)


@bindings_router.put("/prop-bindings/{binding_id}")
async def update_prop_binding(binding_id: str, body: PropBindingUpdate):
    """Update prop binding fields."""
    async with async_session() as session:
        binding = await session.get(PropBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Prop binding not found")

        if body.tag is not None and body.tag != binding.tag:
            await _check_tag_unique(session, PropBinding, binding.production_bible_id, body.tag, exclude_id=binding.id)
            binding.tag = body.tag

        if "production_name" in body.model_fields_set:
            binding.production_name = body.production_name
        if "notes" in body.model_fields_set:
            binding.notes = body.notes
        if "prompt_tags" in body.model_fields_set:
            binding.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(binding)

        return _prop_binding_to_dict(binding)


@bindings_router.delete("/prop-bindings/{binding_id}", status_code=204)
async def delete_prop_binding(binding_id: str):
    """Delete a prop binding."""
    async with async_session() as session:
        binding = await session.get(PropBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Prop binding not found")

        await session.delete(binding)
        await session.commit()
        return None


# ---------------------------------------------------------------------------
# Bound Assets Summary (Phase 23)
# ---------------------------------------------------------------------------


@bindings_router.get("/production-bibles/{bible_id}/bound-assets/summary")
async def get_bound_assets_summary(bible_id: str):
    """Return flat list of all bindings with tags, names, types, and primary thumbnails.

    Combines CastBindings, SetBindings, and PropBindings into a single list
    for frontend consumption (autocomplete, tag reference sheets).
    No pagination -- typical bibles have 5-20 bindings.
    """
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        # Query all binding types
        cast_result = await session.execute(
            select(CastBinding)
            .where(CastBinding.production_bible_id == uid)
            .order_by(CastBinding.tag)
        )
        cast_bindings = list(cast_result.scalars().all())

        set_result = await session.execute(
            select(SetBinding)
            .where(SetBinding.production_bible_id == uid)
            .order_by(SetBinding.tag)
        )
        set_bindings = list(set_result.scalars().all())

        prop_result = await session.execute(
            select(PropBinding)
            .where(PropBinding.production_bible_id == uid)
            .order_by(PropBinding.tag)
        )
        prop_bindings = list(prop_result.scalars().all())

        # Batch-load referenced entities for names and descriptions
        actors_by_id: dict = {}
        actor_primary_refs: dict = {}
        if cast_bindings:
            actor_ids = [b.actor_id for b in cast_bindings]
            actor_result = await session.execute(
                select(Actor).where(Actor.id.in_(actor_ids))
            )
            actors_by_id = {a.id: a for a in actor_result.scalars().all()}

            ref_result = await session.execute(
                select(ActorRef).where(
                    ActorRef.actor_id.in_(actor_ids),
                    ActorRef.is_primary == True,  # noqa: E712
                )
            )
            actor_primary_refs = {r.actor_id: r.image_url for r in ref_result.scalars().all()}

        sets_by_id: dict = {}
        set_primary_refs: dict = {}
        if set_bindings:
            set_ids = [b.library_set_id for b in set_bindings]
            lib_set_result = await session.execute(
                select(LibrarySet).where(LibrarySet.id.in_(set_ids))
            )
            sets_by_id = {s.id: s for s in lib_set_result.scalars().all()}

            ref_result = await session.execute(
                select(LibrarySetRef).where(
                    LibrarySetRef.library_set_id.in_(set_ids),
                    LibrarySetRef.is_primary == True,  # noqa: E712
                )
            )
            set_primary_refs = {r.library_set_id: r.image_url for r in ref_result.scalars().all()}

        props_by_id: dict = {}
        prop_primary_refs: dict = {}
        if prop_bindings:
            prop_ids = [b.library_prop_id for b in prop_bindings]
            lib_prop_result = await session.execute(
                select(LibraryProp).where(LibraryProp.id.in_(prop_ids))
            )
            props_by_id = {p.id: p for p in lib_prop_result.scalars().all()}

            ref_result = await session.execute(
                select(LibraryPropRef).where(
                    LibraryPropRef.library_prop_id.in_(prop_ids),
                    LibraryPropRef.is_primary == True,  # noqa: E712
                )
            )
            prop_primary_refs = {r.library_prop_id: r.image_url for r in ref_result.scalars().all()}

        def _truncate(text: Optional[str], max_len: int = 200) -> Optional[str]:
            if not text:
                return None
            return text[:max_len] + "..." if len(text) > max_len else text

        # Build flat result list: CHARACTER first, then SET, then PROP
        items: list[dict] = []

        for cb in cast_bindings:
            actor = actors_by_id.get(cb.actor_id)
            items.append({
                "tag": cb.tag,
                "name": cb.character_name,
                "type": "CHARACTER",
                "primary_thumbnail_url": actor_primary_refs.get(cb.actor_id),
                "description": _truncate(actor.base_appearance_prompt) if actor else None,
            })

        for sb in set_bindings:
            lib_set = sets_by_id.get(sb.library_set_id)
            items.append({
                "tag": sb.tag,
                "name": sb.production_name or (lib_set.name if lib_set else "Unknown"),
                "type": "SET",
                "primary_thumbnail_url": set_primary_refs.get(sb.library_set_id),
                "description": _truncate(lib_set.reverse_prompt) if lib_set else None,
            })

        for pb in prop_bindings:
            lib_prop = props_by_id.get(pb.library_prop_id)
            items.append({
                "tag": pb.tag,
                "name": pb.production_name or (lib_prop.name if lib_prop else "Unknown"),
                "type": "PROP",
                "primary_thumbnail_url": prop_primary_refs.get(pb.library_prop_id),
                "description": _truncate(lib_prop.appearance_prompt) if lib_prop else None,
            })

        return items


# ---------------------------------------------------------------------------
# SoundBinding endpoints
# ---------------------------------------------------------------------------


@bindings_router.get("/production-bibles/{bible_id}/sound-bindings")
async def list_sound_bindings(bible_id: str):
    """List sound bindings with sound asset name and category."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        result = await session.execute(
            select(SoundBinding)
            .where(SoundBinding.production_bible_id == uid)
            .order_by(SoundBinding.created_at)
        )
        bindings = result.scalars().all()

        if not bindings:
            return []

        sound_ids = [b.sound_asset_id for b in bindings]
        sound_result = await session.execute(
            select(SoundAsset).where(SoundAsset.id.in_(sound_ids))
        )
        sounds_by_id = {s.id: s for s in sound_result.scalars().all()}

        return [
            _sound_binding_to_dict(
                b,
                sound_name=sounds_by_id[b.sound_asset_id].name if b.sound_asset_id in sounds_by_id else None,
                sound_category=sounds_by_id[b.sound_asset_id].category if b.sound_asset_id in sounds_by_id else None,
            )
            for b in bindings
        ]


@bindings_router.post("/production-bibles/{bible_id}/sound-bindings", status_code=201)
async def create_sound_binding(bible_id: str, body: SoundBindingCreate):
    """Create a sound binding."""
    async with async_session() as session:
        uid = await _validate_bible(session, bible_id)

        sound = await session.get(SoundAsset, uuid.UUID(body.sound_asset_id))
        if sound is None:
            raise HTTPException(status_code=404, detail="Sound asset not found")

        # SoundBinding tag is optional; only check uniqueness if provided
        if body.tag:
            await _check_tag_unique(session, SoundBinding, uid, body.tag)

        binding = SoundBinding(
            production_bible_id=uid,
            sound_asset_id=uuid.UUID(body.sound_asset_id),
            tag=body.tag,
            usage_notes=body.usage_notes,
            prompt_tags=body.prompt_tags,
        )
        session.add(binding)
        await session.commit()
        await session.refresh(binding)

        return _sound_binding_to_dict(binding, sound_name=sound.name, sound_category=sound.category)


@bindings_router.put("/sound-bindings/{binding_id}")
async def update_sound_binding(binding_id: str, body: SoundBindingUpdate):
    """Update sound binding fields."""
    async with async_session() as session:
        binding = await session.get(SoundBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Sound binding not found")

        if "tag" in body.model_fields_set:
            if body.tag and body.tag != binding.tag:
                await _check_tag_unique(session, SoundBinding, binding.production_bible_id, body.tag, exclude_id=binding.id)
            binding.tag = body.tag

        if "usage_notes" in body.model_fields_set:
            binding.usage_notes = body.usage_notes
        if "prompt_tags" in body.model_fields_set:
            binding.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(binding)

        return _sound_binding_to_dict(binding)


@bindings_router.delete("/sound-bindings/{binding_id}", status_code=204)
async def delete_sound_binding(binding_id: str):
    """Delete a sound binding."""
    async with async_session() as session:
        binding = await session.get(SoundBinding, uuid.UUID(binding_id))
        if binding is None:
            raise HTTPException(status_code=404, detail="Sound binding not found")

        await session.delete(binding)
        await session.commit()
        return None
