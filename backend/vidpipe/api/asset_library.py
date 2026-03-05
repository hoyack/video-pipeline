"""Asset Library CRUD API routes.

Manages standalone library entities: Actors (with refs, voice profiles, wardrobe
presets), LibrarySets (with refs, sonic identity), LibraryProps (with refs), and
SoundAssets. These are global entities not scoped to any production bible; they
get connected via binding tables (Plan 03).

Follows the same CRUD pattern as characters.py:
  - Pydantic Create/Update request models
  - _entity_to_dict() helpers for serialization
  - List endpoints with search param and bulk sub-entity fetch
  - Delete endpoints check for bindings (409 if bound)

Spec reference: Phase 22 - ALIB-01, ALIB-03, ALIB-04
"""

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, func, select

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
    LibrarySonicIdentity,
    PropBinding,
    SetBinding,
    SoundAsset,
    SoundBinding,
)
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend

logger = logging.getLogger(__name__)

asset_library_router = APIRouter(prefix="/api/asset-library", tags=["asset-library"])


# ---------------------------------------------------------------------------
# Valid sound categories
# ---------------------------------------------------------------------------
VALID_SOUND_CATEGORIES = {"SCORE_THEME", "SFX", "AMBIENCE", "FOLEY", "UI"}


# ---------------------------------------------------------------------------
# Request schemas — Actors
# ---------------------------------------------------------------------------


class ActorCreate(BaseModel):
    name: str
    description: Optional[str] = None
    base_appearance_prompt: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class ActorUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    base_appearance_prompt: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class VoiceProfileCreate(BaseModel):
    voice_id: Optional[str] = None
    adapter_type: Optional[str] = None
    style_notes: Optional[str] = None
    sample_url: Optional[str] = None


class VoiceProfileUpdate(BaseModel):
    voice_id: Optional[str] = None
    adapter_type: Optional[str] = None
    style_notes: Optional[str] = None
    sample_url: Optional[str] = None


class WardrobePresetCreate(BaseModel):
    label: str
    description: Optional[str] = None
    reference_images: Optional[list[str]] = None


class WardrobePresetUpdate(BaseModel):
    label: Optional[str] = None
    description: Optional[str] = None
    reference_images: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Request schemas — LibrarySets
# ---------------------------------------------------------------------------


class LibrarySetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    reverse_prompt: Optional[str] = None
    style_tags: Optional[list[str]] = None
    prompt_tags: Optional[list[str]] = None
    lighting_notes: Optional[str] = None


class LibrarySetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    reverse_prompt: Optional[str] = None
    style_tags: Optional[list[str]] = None
    prompt_tags: Optional[list[str]] = None
    lighting_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Request schemas — LibraryProps
# ---------------------------------------------------------------------------


class LibraryPropCreate(BaseModel):
    name: str
    description: Optional[str] = None
    appearance_prompt: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class LibraryPropUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    appearance_prompt: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Request schemas — SoundAssets
# ---------------------------------------------------------------------------


class SoundAssetCreate(BaseModel):
    name: str
    category: str
    subcategory: Optional[str] = None
    description: Optional[str] = None
    audio_url: Optional[str] = None
    generation_prompt: Optional[str] = None
    tags: Optional[list[str]] = None


class SoundAssetUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    description: Optional[str] = None
    audio_url: Optional[str] = None
    generation_prompt: Optional[str] = None
    tags: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Helpers — file upload
# ---------------------------------------------------------------------------


async def _save_upload(
    content: bytes,
    content_type: str,
    filename: str,
    storage_prefix: str,
) -> str:
    """Save uploaded file via dual-backend storage pattern. Returns stored path/key."""
    storage = get_storage_backend()

    if isinstance(storage, LocalStorageBackend):
        from vidpipe.config import settings as _settings

        local_dir = _settings.storage.tmp_dir / "asset-library" / storage_prefix
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename
        await asyncio.to_thread(local_path.write_bytes, content)
        return str(local_path)
    else:
        key = f"asset-library/{storage_prefix}/{filename}"
        await storage.put(key, content, content_type)
        # Write local copy for serving
        from vidpipe.config import settings as _settings

        local_path = _settings.storage.tmp_dir / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(local_path.write_bytes, content)
        return key


def _validate_image_upload(file: UploadFile, content: bytes) -> None:
    """Validate image file type and size (10MB limit)."""
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be image/png, image/jpeg, or image/webp",
        )
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 10MB")


def _validate_audio_upload(file: UploadFile, content: bytes) -> None:
    """Validate audio file type and size (20MB limit)."""
    allowed = ("audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/webm", "audio/flac")
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be one of {', '.join(allowed)}",
        )
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 20MB")


# ---------------------------------------------------------------------------
# Helpers — serialization
# ---------------------------------------------------------------------------


def _actor_ref_to_dict(r: ActorRef) -> dict:
    return {
        "id": str(r.id),
        "actor_id": str(r.actor_id),
        "image_url": r.image_url,
        "label": r.label,
        "is_primary": r.is_primary,
        "created_at": r.created_at.isoformat(),
    }


def _voice_profile_to_dict(vp: ActorVoiceProfile) -> dict:
    return {
        "id": str(vp.id),
        "actor_id": str(vp.actor_id),
        "voice_id": vp.voice_id,
        "adapter_type": vp.adapter_type,
        "style_notes": vp.style_notes,
        "sample_url": vp.sample_url,
        "created_at": vp.created_at.isoformat(),
    }


def _wardrobe_preset_to_dict(wp: ActorWardrobePreset) -> dict:
    return {
        "id": str(wp.id),
        "actor_id": str(wp.actor_id),
        "label": wp.label,
        "description": wp.description,
        "reference_images": wp.reference_images,
        "created_at": wp.created_at.isoformat(),
    }


def _actor_detail_to_dict(
    actor: Actor,
    refs: list[ActorRef],
    voice_profiles: list[ActorVoiceProfile],
    wardrobe_presets: list[ActorWardrobePreset],
    binding_count: int = 0,
) -> dict:
    return {
        "id": str(actor.id),
        "name": actor.name,
        "description": actor.description,
        "base_appearance_prompt": actor.base_appearance_prompt,
        "prompt_tags": actor.prompt_tags,
        "refs": [_actor_ref_to_dict(r) for r in refs],
        "voice_profiles": [_voice_profile_to_dict(vp) for vp in voice_profiles],
        "wardrobe_presets": [_wardrobe_preset_to_dict(wp) for wp in wardrobe_presets],
        "binding_count": binding_count,
        "created_at": actor.created_at.isoformat(),
        "updated_at": actor.updated_at.isoformat(),
    }


def _library_set_ref_to_dict(r: LibrarySetRef) -> dict:
    return {
        "id": str(r.id),
        "library_set_id": str(r.library_set_id),
        "image_url": r.image_url,
        "label": r.label,
        "is_primary": r.is_primary,
        "created_at": r.created_at.isoformat(),
    }


def _sonic_identity_to_dict(si: LibrarySonicIdentity) -> dict:
    return {
        "id": str(si.id),
        "library_set_id": str(si.library_set_id),
        "ambience_description": si.ambience_description,
        "reference_audio_url": si.reference_audio_url,
        "generation_prompt": si.generation_prompt,
        "created_at": si.created_at.isoformat(),
    }


def _library_set_detail_to_dict(
    ls: LibrarySet,
    refs: list[LibrarySetRef],
    sonic_identity: Optional[LibrarySonicIdentity],
    binding_count: int = 0,
) -> dict:
    return {
        "id": str(ls.id),
        "name": ls.name,
        "description": ls.description,
        "reverse_prompt": ls.reverse_prompt,
        "style_tags": ls.style_tags,
        "prompt_tags": ls.prompt_tags,
        "lighting_notes": ls.lighting_notes,
        "refs": [_library_set_ref_to_dict(r) for r in refs],
        "sonic_identity": _sonic_identity_to_dict(sonic_identity) if sonic_identity else None,
        "binding_count": binding_count,
        "created_at": ls.created_at.isoformat(),
        "updated_at": ls.updated_at.isoformat(),
    }


def _library_prop_ref_to_dict(r: LibraryPropRef) -> dict:
    return {
        "id": str(r.id),
        "library_prop_id": str(r.library_prop_id),
        "image_url": r.image_url,
        "label": r.label,
        "is_primary": r.is_primary,
        "created_at": r.created_at.isoformat(),
    }


def _library_prop_detail_to_dict(
    lp: LibraryProp,
    refs: list[LibraryPropRef],
    binding_count: int = 0,
) -> dict:
    return {
        "id": str(lp.id),
        "name": lp.name,
        "description": lp.description,
        "appearance_prompt": lp.appearance_prompt,
        "prompt_tags": lp.prompt_tags,
        "refs": [_library_prop_ref_to_dict(r) for r in refs],
        "binding_count": binding_count,
        "created_at": lp.created_at.isoformat(),
        "updated_at": lp.updated_at.isoformat(),
    }


def _sound_asset_to_dict(sa: SoundAsset, binding_count: int = 0) -> dict:
    return {
        "id": str(sa.id),
        "name": sa.name,
        "category": sa.category,
        "subcategory": sa.subcategory,
        "description": sa.description,
        "audio_url": sa.audio_url,
        "generation_prompt": sa.generation_prompt,
        "tags": sa.tags,
        "binding_count": binding_count,
        "created_at": sa.created_at.isoformat(),
        "updated_at": sa.updated_at.isoformat(),
    }


# ===========================================================================
# Actor endpoints
# ===========================================================================


@asset_library_router.get("/actors")
async def list_actors(search: Optional[str] = None):
    """List all actors with ref counts and binding counts."""
    async with async_session() as session:
        query = select(Actor).order_by(Actor.created_at.desc())
        if search:
            query = query.where(Actor.name.ilike(f"%{search}%"))
        result = await session.execute(query)
        actors = result.scalars().all()

        if not actors:
            return []

        actor_ids = [a.id for a in actors]

        # Bulk fetch ref counts
        ref_counts_q = (
            select(ActorRef.actor_id, func.count(ActorRef.id))
            .where(ActorRef.actor_id.in_(actor_ids))
            .group_by(ActorRef.actor_id)
        )
        ref_counts_result = await session.execute(ref_counts_q)
        ref_counts = dict(ref_counts_result.all())

        # Bulk fetch binding counts
        bind_counts_q = (
            select(CastBinding.actor_id, func.count(CastBinding.id))
            .where(CastBinding.actor_id.in_(actor_ids))
            .group_by(CastBinding.actor_id)
        )
        bind_counts_result = await session.execute(bind_counts_q)
        bind_counts = dict(bind_counts_result.all())

        # Fetch primary refs for thumbnails
        primary_refs_q = (
            select(ActorRef)
            .where(ActorRef.actor_id.in_(actor_ids), ActorRef.is_primary == True)  # noqa: E712
        )
        primary_refs_result = await session.execute(primary_refs_q)
        primary_refs = {r.actor_id: r for r in primary_refs_result.scalars().all()}

        return [
            {
                "id": str(a.id),
                "name": a.name,
                "description": a.description,
                "prompt_tags": a.prompt_tags,
                "ref_count": ref_counts.get(a.id, 0),
                "binding_count": bind_counts.get(a.id, 0),
                "primary_ref_url": primary_refs[a.id].image_url if a.id in primary_refs else None,
                "created_at": a.created_at.isoformat(),
            }
            for a in actors
        ]


@asset_library_router.post("/actors", status_code=201)
async def create_actor(body: ActorCreate):
    """Create a new actor."""
    async with async_session() as session:
        actor = Actor(
            name=body.name,
            description=body.description,
            base_appearance_prompt=body.base_appearance_prompt,
            prompt_tags=body.prompt_tags,
        )
        session.add(actor)
        await session.commit()
        await session.refresh(actor)
        return _actor_detail_to_dict(actor, [], [], [])


@asset_library_router.get("/actors/{actor_id}")
async def get_actor(actor_id: str):
    """Get actor with all sub-entities and binding count."""
    async with async_session() as session:
        actor = await session.get(Actor, uuid.UUID(actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        refs_result = await session.execute(
            select(ActorRef).where(ActorRef.actor_id == actor.id).order_by(ActorRef.created_at)
        )
        refs = list(refs_result.scalars().all())

        vp_result = await session.execute(
            select(ActorVoiceProfile).where(ActorVoiceProfile.actor_id == actor.id)
        )
        voice_profiles = list(vp_result.scalars().all())

        wp_result = await session.execute(
            select(ActorWardrobePreset).where(ActorWardrobePreset.actor_id == actor.id)
        )
        wardrobe_presets = list(wp_result.scalars().all())

        bind_result = await session.execute(
            select(func.count(CastBinding.id)).where(CastBinding.actor_id == actor.id)
        )
        binding_count = bind_result.scalar() or 0

        return _actor_detail_to_dict(actor, refs, voice_profiles, wardrobe_presets, binding_count)


@asset_library_router.put("/actors/{actor_id}")
async def update_actor(actor_id: str, body: ActorUpdate):
    """Update actor top-level fields."""
    async with async_session() as session:
        actor = await session.get(Actor, uuid.UUID(actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        if body.name is not None:
            actor.name = body.name
        if "description" in body.model_fields_set:
            actor.description = body.description
        if "base_appearance_prompt" in body.model_fields_set:
            actor.base_appearance_prompt = body.base_appearance_prompt
        if "prompt_tags" in body.model_fields_set:
            actor.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(actor)

        # Fetch sub-entities for response
        refs_result = await session.execute(
            select(ActorRef).where(ActorRef.actor_id == actor.id)
        )
        vp_result = await session.execute(
            select(ActorVoiceProfile).where(ActorVoiceProfile.actor_id == actor.id)
        )
        wp_result = await session.execute(
            select(ActorWardrobePreset).where(ActorWardrobePreset.actor_id == actor.id)
        )
        bind_result = await session.execute(
            select(func.count(CastBinding.id)).where(CastBinding.actor_id == actor.id)
        )

        return _actor_detail_to_dict(
            actor,
            list(refs_result.scalars().all()),
            list(vp_result.scalars().all()),
            list(wp_result.scalars().all()),
            bind_result.scalar() or 0,
        )


@asset_library_router.delete("/actors/{actor_id}", status_code=204)
async def delete_actor(actor_id: str):
    """Delete actor. Returns 409 if cast bindings exist."""
    async with async_session() as session:
        actor = await session.get(Actor, uuid.UUID(actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        bind_result = await session.execute(
            select(func.count(CastBinding.id)).where(CastBinding.actor_id == actor.id)
        )
        binding_count = bind_result.scalar() or 0
        if binding_count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete actor with {binding_count} cast binding(s). Remove bindings first.",
            )

        # Cascade delete sub-entities
        await session.execute(delete(ActorRef).where(ActorRef.actor_id == actor.id))
        await session.execute(delete(ActorVoiceProfile).where(ActorVoiceProfile.actor_id == actor.id))
        await session.execute(delete(ActorWardrobePreset).where(ActorWardrobePreset.actor_id == actor.id))
        await session.delete(actor)
        await session.commit()
        return None


# --- Actor Refs ---


@asset_library_router.post("/actors/{actor_id}/refs", status_code=201)
async def upload_actor_ref(actor_id: str, file: UploadFile = File(...)):
    """Upload reference image for an actor. Creates ActorRef row."""
    content = await file.read()
    _validate_image_upload(file, content)

    async with async_session() as session:
        actor = await session.get(Actor, uuid.UUID(actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        filename = file.filename or "ref.png"
        stored_path = await _save_upload(
            content,
            file.content_type or "image/png",
            filename,
            f"actors/{actor.id}/refs",
        )

        ref = ActorRef(
            actor_id=actor.id,
            image_url=stored_path,
            label=None,
            is_primary=False,
        )
        session.add(ref)
        await session.commit()
        await session.refresh(ref)
        return _actor_ref_to_dict(ref)


@asset_library_router.delete("/actor-refs/{ref_id}", status_code=204)
async def delete_actor_ref(ref_id: str):
    """Delete an actor reference image."""
    async with async_session() as session:
        ref = await session.get(ActorRef, uuid.UUID(ref_id))
        if ref is None:
            raise HTTPException(status_code=404, detail="Actor ref not found")
        await session.delete(ref)
        await session.commit()
        return None


# --- Actor Voice Profiles ---


@asset_library_router.post("/actors/{actor_id}/voice-profiles", status_code=201)
async def create_voice_profile(actor_id: str, body: VoiceProfileCreate):
    """Create a voice profile for an actor."""
    async with async_session() as session:
        actor = await session.get(Actor, uuid.UUID(actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        vp = ActorVoiceProfile(
            actor_id=actor.id,
            voice_id=body.voice_id,
            adapter_type=body.adapter_type,
            style_notes=body.style_notes,
            sample_url=body.sample_url,
        )
        session.add(vp)
        await session.commit()
        await session.refresh(vp)
        return _voice_profile_to_dict(vp)


@asset_library_router.put("/actor-voice-profiles/{profile_id}")
async def update_voice_profile(profile_id: str, body: VoiceProfileUpdate):
    """Update an actor voice profile."""
    async with async_session() as session:
        vp = await session.get(ActorVoiceProfile, uuid.UUID(profile_id))
        if vp is None:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        if "voice_id" in body.model_fields_set:
            vp.voice_id = body.voice_id
        if "adapter_type" in body.model_fields_set:
            vp.adapter_type = body.adapter_type
        if "style_notes" in body.model_fields_set:
            vp.style_notes = body.style_notes
        if "sample_url" in body.model_fields_set:
            vp.sample_url = body.sample_url

        await session.commit()
        await session.refresh(vp)
        return _voice_profile_to_dict(vp)


@asset_library_router.delete("/actor-voice-profiles/{profile_id}", status_code=204)
async def delete_voice_profile(profile_id: str):
    """Delete an actor voice profile."""
    async with async_session() as session:
        vp = await session.get(ActorVoiceProfile, uuid.UUID(profile_id))
        if vp is None:
            raise HTTPException(status_code=404, detail="Voice profile not found")
        await session.delete(vp)
        await session.commit()
        return None


# --- Actor Wardrobe Presets ---


@asset_library_router.post("/actors/{actor_id}/wardrobe-presets", status_code=201)
async def create_wardrobe_preset(actor_id: str, body: WardrobePresetCreate):
    """Create a wardrobe preset for an actor."""
    async with async_session() as session:
        actor = await session.get(Actor, uuid.UUID(actor_id))
        if actor is None:
            raise HTTPException(status_code=404, detail="Actor not found")

        wp = ActorWardrobePreset(
            actor_id=actor.id,
            label=body.label,
            description=body.description,
            reference_images=body.reference_images,
        )
        session.add(wp)
        await session.commit()
        await session.refresh(wp)
        return _wardrobe_preset_to_dict(wp)


@asset_library_router.put("/actor-wardrobe-presets/{preset_id}")
async def update_wardrobe_preset(preset_id: str, body: WardrobePresetUpdate):
    """Update an actor wardrobe preset."""
    async with async_session() as session:
        wp = await session.get(ActorWardrobePreset, uuid.UUID(preset_id))
        if wp is None:
            raise HTTPException(status_code=404, detail="Wardrobe preset not found")

        if body.label is not None:
            wp.label = body.label
        if "description" in body.model_fields_set:
            wp.description = body.description
        if "reference_images" in body.model_fields_set:
            wp.reference_images = body.reference_images

        await session.commit()
        await session.refresh(wp)
        return _wardrobe_preset_to_dict(wp)


@asset_library_router.delete("/actor-wardrobe-presets/{preset_id}", status_code=204)
async def delete_wardrobe_preset(preset_id: str):
    """Delete an actor wardrobe preset."""
    async with async_session() as session:
        wp = await session.get(ActorWardrobePreset, uuid.UUID(preset_id))
        if wp is None:
            raise HTTPException(status_code=404, detail="Wardrobe preset not found")
        await session.delete(wp)
        await session.commit()
        return None


# ===========================================================================
# LibrarySet endpoints
# ===========================================================================


@asset_library_router.get("/sets")
async def list_sets(search: Optional[str] = None):
    """List all library sets with ref counts and binding counts."""
    async with async_session() as session:
        query = select(LibrarySet).order_by(LibrarySet.created_at.desc())
        if search:
            query = query.where(LibrarySet.name.ilike(f"%{search}%"))
        result = await session.execute(query)
        sets = result.scalars().all()

        if not sets:
            return []

        set_ids = [s.id for s in sets]

        ref_counts_q = (
            select(LibrarySetRef.library_set_id, func.count(LibrarySetRef.id))
            .where(LibrarySetRef.library_set_id.in_(set_ids))
            .group_by(LibrarySetRef.library_set_id)
        )
        ref_counts = dict((await session.execute(ref_counts_q)).all())

        bind_counts_q = (
            select(SetBinding.library_set_id, func.count(SetBinding.id))
            .where(SetBinding.library_set_id.in_(set_ids))
            .group_by(SetBinding.library_set_id)
        )
        bind_counts = dict((await session.execute(bind_counts_q)).all())

        primary_refs_q = (
            select(LibrarySetRef)
            .where(LibrarySetRef.library_set_id.in_(set_ids), LibrarySetRef.is_primary == True)  # noqa: E712
        )
        primary_refs = {
            r.library_set_id: r
            for r in (await session.execute(primary_refs_q)).scalars().all()
        }

        return [
            {
                "id": str(s.id),
                "name": s.name,
                "description": s.description,
                "ref_count": ref_counts.get(s.id, 0),
                "binding_count": bind_counts.get(s.id, 0),
                "primary_ref_url": primary_refs[s.id].image_url if s.id in primary_refs else None,
                "created_at": s.created_at.isoformat(),
            }
            for s in sets
        ]


@asset_library_router.post("/sets", status_code=201)
async def create_set(body: LibrarySetCreate):
    """Create a new library set."""
    async with async_session() as session:
        ls = LibrarySet(
            name=body.name,
            description=body.description,
            reverse_prompt=body.reverse_prompt,
            style_tags=body.style_tags,
            prompt_tags=body.prompt_tags,
            lighting_notes=body.lighting_notes,
        )
        session.add(ls)
        await session.commit()
        await session.refresh(ls)
        return _library_set_detail_to_dict(ls, [], None)


@asset_library_router.get("/sets/{set_id}")
async def get_set(set_id: str):
    """Get library set with refs, sonic identity, and binding count."""
    async with async_session() as session:
        ls = await session.get(LibrarySet, uuid.UUID(set_id))
        if ls is None:
            raise HTTPException(status_code=404, detail="Library set not found")

        refs_result = await session.execute(
            select(LibrarySetRef).where(LibrarySetRef.library_set_id == ls.id).order_by(LibrarySetRef.created_at)
        )
        refs = list(refs_result.scalars().all())

        si_result = await session.execute(
            select(LibrarySonicIdentity).where(LibrarySonicIdentity.library_set_id == ls.id)
        )
        sonic_identity = si_result.scalars().first()

        bind_result = await session.execute(
            select(func.count(SetBinding.id)).where(SetBinding.library_set_id == ls.id)
        )
        binding_count = bind_result.scalar() or 0

        return _library_set_detail_to_dict(ls, refs, sonic_identity, binding_count)


@asset_library_router.put("/sets/{set_id}")
async def update_set(set_id: str, body: LibrarySetUpdate):
    """Update library set fields."""
    async with async_session() as session:
        ls = await session.get(LibrarySet, uuid.UUID(set_id))
        if ls is None:
            raise HTTPException(status_code=404, detail="Library set not found")

        if body.name is not None:
            ls.name = body.name
        if "description" in body.model_fields_set:
            ls.description = body.description
        if "reverse_prompt" in body.model_fields_set:
            ls.reverse_prompt = body.reverse_prompt
        if "style_tags" in body.model_fields_set:
            ls.style_tags = body.style_tags
        if "prompt_tags" in body.model_fields_set:
            ls.prompt_tags = body.prompt_tags
        if "lighting_notes" in body.model_fields_set:
            ls.lighting_notes = body.lighting_notes

        await session.commit()
        await session.refresh(ls)

        refs_result = await session.execute(
            select(LibrarySetRef).where(LibrarySetRef.library_set_id == ls.id)
        )
        si_result = await session.execute(
            select(LibrarySonicIdentity).where(LibrarySonicIdentity.library_set_id == ls.id)
        )
        bind_result = await session.execute(
            select(func.count(SetBinding.id)).where(SetBinding.library_set_id == ls.id)
        )

        return _library_set_detail_to_dict(
            ls,
            list(refs_result.scalars().all()),
            si_result.scalars().first(),
            bind_result.scalar() or 0,
        )


@asset_library_router.delete("/sets/{set_id}", status_code=204)
async def delete_set(set_id: str):
    """Delete library set. Returns 409 if set bindings exist."""
    async with async_session() as session:
        ls = await session.get(LibrarySet, uuid.UUID(set_id))
        if ls is None:
            raise HTTPException(status_code=404, detail="Library set not found")

        bind_result = await session.execute(
            select(func.count(SetBinding.id)).where(SetBinding.library_set_id == ls.id)
        )
        binding_count = bind_result.scalar() or 0
        if binding_count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete set with {binding_count} binding(s). Remove bindings first.",
            )

        await session.execute(delete(LibrarySetRef).where(LibrarySetRef.library_set_id == ls.id))
        await session.execute(delete(LibrarySonicIdentity).where(LibrarySonicIdentity.library_set_id == ls.id))
        await session.delete(ls)
        await session.commit()
        return None


# --- LibrarySet Refs ---


@asset_library_router.post("/sets/{set_id}/refs", status_code=201)
async def upload_set_ref(set_id: str, file: UploadFile = File(...)):
    """Upload reference image for a library set."""
    content = await file.read()
    _validate_image_upload(file, content)

    async with async_session() as session:
        ls = await session.get(LibrarySet, uuid.UUID(set_id))
        if ls is None:
            raise HTTPException(status_code=404, detail="Library set not found")

        filename = file.filename or "ref.png"
        stored_path = await _save_upload(
            content,
            file.content_type or "image/png",
            filename,
            f"sets/{ls.id}/refs",
        )

        ref = LibrarySetRef(
            library_set_id=ls.id,
            image_url=stored_path,
            label=None,
            is_primary=False,
        )
        session.add(ref)
        await session.commit()
        await session.refresh(ref)
        return _library_set_ref_to_dict(ref)


@asset_library_router.delete("/set-refs/{ref_id}", status_code=204)
async def delete_set_ref(ref_id: str):
    """Delete a library set reference image."""
    async with async_session() as session:
        ref = await session.get(LibrarySetRef, uuid.UUID(ref_id))
        if ref is None:
            raise HTTPException(status_code=404, detail="Set ref not found")
        await session.delete(ref)
        await session.commit()
        return None


# ===========================================================================
# LibraryProp endpoints
# ===========================================================================


@asset_library_router.get("/props")
async def list_props(search: Optional[str] = None):
    """List all library props with ref counts and binding counts."""
    async with async_session() as session:
        query = select(LibraryProp).order_by(LibraryProp.created_at.desc())
        if search:
            query = query.where(LibraryProp.name.ilike(f"%{search}%"))
        result = await session.execute(query)
        props = result.scalars().all()

        if not props:
            return []

        prop_ids = [p.id for p in props]

        ref_counts_q = (
            select(LibraryPropRef.library_prop_id, func.count(LibraryPropRef.id))
            .where(LibraryPropRef.library_prop_id.in_(prop_ids))
            .group_by(LibraryPropRef.library_prop_id)
        )
        ref_counts = dict((await session.execute(ref_counts_q)).all())

        bind_counts_q = (
            select(PropBinding.library_prop_id, func.count(PropBinding.id))
            .where(PropBinding.library_prop_id.in_(prop_ids))
            .group_by(PropBinding.library_prop_id)
        )
        bind_counts = dict((await session.execute(bind_counts_q)).all())

        primary_refs_q = (
            select(LibraryPropRef)
            .where(LibraryPropRef.library_prop_id.in_(prop_ids), LibraryPropRef.is_primary == True)  # noqa: E712
        )
        primary_refs = {
            r.library_prop_id: r
            for r in (await session.execute(primary_refs_q)).scalars().all()
        }

        return [
            {
                "id": str(p.id),
                "name": p.name,
                "description": p.description,
                "ref_count": ref_counts.get(p.id, 0),
                "binding_count": bind_counts.get(p.id, 0),
                "primary_ref_url": primary_refs[p.id].image_url if p.id in primary_refs else None,
                "created_at": p.created_at.isoformat(),
            }
            for p in props
        ]


@asset_library_router.post("/props", status_code=201)
async def create_prop(body: LibraryPropCreate):
    """Create a new library prop."""
    async with async_session() as session:
        lp = LibraryProp(
            name=body.name,
            description=body.description,
            appearance_prompt=body.appearance_prompt,
            prompt_tags=body.prompt_tags,
        )
        session.add(lp)
        await session.commit()
        await session.refresh(lp)
        return _library_prop_detail_to_dict(lp, [])


@asset_library_router.get("/props/{prop_id}")
async def get_prop(prop_id: str):
    """Get library prop with refs and binding count."""
    async with async_session() as session:
        lp = await session.get(LibraryProp, uuid.UUID(prop_id))
        if lp is None:
            raise HTTPException(status_code=404, detail="Library prop not found")

        refs_result = await session.execute(
            select(LibraryPropRef).where(LibraryPropRef.library_prop_id == lp.id).order_by(LibraryPropRef.created_at)
        )
        refs = list(refs_result.scalars().all())

        bind_result = await session.execute(
            select(func.count(PropBinding.id)).where(PropBinding.library_prop_id == lp.id)
        )
        binding_count = bind_result.scalar() or 0

        return _library_prop_detail_to_dict(lp, refs, binding_count)


@asset_library_router.put("/props/{prop_id}")
async def update_prop(prop_id: str, body: LibraryPropUpdate):
    """Update library prop fields."""
    async with async_session() as session:
        lp = await session.get(LibraryProp, uuid.UUID(prop_id))
        if lp is None:
            raise HTTPException(status_code=404, detail="Library prop not found")

        if body.name is not None:
            lp.name = body.name
        if "description" in body.model_fields_set:
            lp.description = body.description
        if "appearance_prompt" in body.model_fields_set:
            lp.appearance_prompt = body.appearance_prompt
        if "prompt_tags" in body.model_fields_set:
            lp.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(lp)

        refs_result = await session.execute(
            select(LibraryPropRef).where(LibraryPropRef.library_prop_id == lp.id)
        )
        bind_result = await session.execute(
            select(func.count(PropBinding.id)).where(PropBinding.library_prop_id == lp.id)
        )

        return _library_prop_detail_to_dict(
            lp, list(refs_result.scalars().all()), bind_result.scalar() or 0
        )


@asset_library_router.delete("/props/{prop_id}", status_code=204)
async def delete_prop(prop_id: str):
    """Delete library prop. Returns 409 if prop bindings exist."""
    async with async_session() as session:
        lp = await session.get(LibraryProp, uuid.UUID(prop_id))
        if lp is None:
            raise HTTPException(status_code=404, detail="Library prop not found")

        bind_result = await session.execute(
            select(func.count(PropBinding.id)).where(PropBinding.library_prop_id == lp.id)
        )
        binding_count = bind_result.scalar() or 0
        if binding_count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete prop with {binding_count} binding(s). Remove bindings first.",
            )

        await session.execute(delete(LibraryPropRef).where(LibraryPropRef.library_prop_id == lp.id))
        await session.delete(lp)
        await session.commit()
        return None


# --- LibraryProp Refs ---


@asset_library_router.post("/props/{prop_id}/refs", status_code=201)
async def upload_prop_ref(prop_id: str, file: UploadFile = File(...)):
    """Upload reference image for a library prop."""
    content = await file.read()
    _validate_image_upload(file, content)

    async with async_session() as session:
        lp = await session.get(LibraryProp, uuid.UUID(prop_id))
        if lp is None:
            raise HTTPException(status_code=404, detail="Library prop not found")

        filename = file.filename or "ref.png"
        stored_path = await _save_upload(
            content,
            file.content_type or "image/png",
            filename,
            f"props/{lp.id}/refs",
        )

        ref = LibraryPropRef(
            library_prop_id=lp.id,
            image_url=stored_path,
            label=None,
            is_primary=False,
        )
        session.add(ref)
        await session.commit()
        await session.refresh(ref)
        return _library_prop_ref_to_dict(ref)


@asset_library_router.delete("/prop-refs/{ref_id}", status_code=204)
async def delete_prop_ref(ref_id: str):
    """Delete a library prop reference image."""
    async with async_session() as session:
        ref = await session.get(LibraryPropRef, uuid.UUID(ref_id))
        if ref is None:
            raise HTTPException(status_code=404, detail="Prop ref not found")
        await session.delete(ref)
        await session.commit()
        return None


# ===========================================================================
# SoundAsset endpoints
# ===========================================================================


@asset_library_router.get("/sounds")
async def list_sounds(search: Optional[str] = None, category: Optional[str] = None):
    """List all sound assets with binding counts."""
    async with async_session() as session:
        query = select(SoundAsset).order_by(SoundAsset.created_at.desc())
        if search:
            query = query.where(SoundAsset.name.ilike(f"%{search}%"))
        if category:
            query = query.where(SoundAsset.category == category)
        result = await session.execute(query)
        sounds = result.scalars().all()

        if not sounds:
            return []

        sound_ids = [s.id for s in sounds]

        bind_counts_q = (
            select(SoundBinding.sound_asset_id, func.count(SoundBinding.id))
            .where(SoundBinding.sound_asset_id.in_(sound_ids))
            .group_by(SoundBinding.sound_asset_id)
        )
        bind_counts = dict((await session.execute(bind_counts_q)).all())

        return [
            {
                "id": str(s.id),
                "name": s.name,
                "category": s.category,
                "subcategory": s.subcategory,
                "description": s.description,
                "has_audio": s.audio_url is not None,
                "binding_count": bind_counts.get(s.id, 0),
                "created_at": s.created_at.isoformat(),
            }
            for s in sounds
        ]


@asset_library_router.post("/sounds", status_code=201)
async def create_sound(body: SoundAssetCreate):
    """Create a new sound asset."""
    if body.category not in VALID_SOUND_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{body.category}'. Must be one of {sorted(VALID_SOUND_CATEGORIES)}.",
        )

    async with async_session() as session:
        sa = SoundAsset(
            name=body.name,
            category=body.category,
            subcategory=body.subcategory,
            description=body.description,
            audio_url=body.audio_url,
            generation_prompt=body.generation_prompt,
            tags=body.tags,
        )
        session.add(sa)
        await session.commit()
        await session.refresh(sa)
        return _sound_asset_to_dict(sa)


@asset_library_router.get("/sounds/{sound_id}")
async def get_sound(sound_id: str):
    """Get sound asset with binding count."""
    async with async_session() as session:
        sa = await session.get(SoundAsset, uuid.UUID(sound_id))
        if sa is None:
            raise HTTPException(status_code=404, detail="Sound asset not found")

        bind_result = await session.execute(
            select(func.count(SoundBinding.id)).where(SoundBinding.sound_asset_id == sa.id)
        )
        binding_count = bind_result.scalar() or 0

        return _sound_asset_to_dict(sa, binding_count)


@asset_library_router.put("/sounds/{sound_id}")
async def update_sound(sound_id: str, body: SoundAssetUpdate):
    """Update sound asset fields."""
    if body.category is not None and body.category not in VALID_SOUND_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{body.category}'. Must be one of {sorted(VALID_SOUND_CATEGORIES)}.",
        )

    async with async_session() as session:
        sa = await session.get(SoundAsset, uuid.UUID(sound_id))
        if sa is None:
            raise HTTPException(status_code=404, detail="Sound asset not found")

        if body.name is not None:
            sa.name = body.name
        if body.category is not None:
            sa.category = body.category
        if "subcategory" in body.model_fields_set:
            sa.subcategory = body.subcategory
        if "description" in body.model_fields_set:
            sa.description = body.description
        if "audio_url" in body.model_fields_set:
            sa.audio_url = body.audio_url
        if "generation_prompt" in body.model_fields_set:
            sa.generation_prompt = body.generation_prompt
        if "tags" in body.model_fields_set:
            sa.tags = body.tags

        await session.commit()
        await session.refresh(sa)

        bind_result = await session.execute(
            select(func.count(SoundBinding.id)).where(SoundBinding.sound_asset_id == sa.id)
        )

        return _sound_asset_to_dict(sa, bind_result.scalar() or 0)


@asset_library_router.delete("/sounds/{sound_id}", status_code=204)
async def delete_sound(sound_id: str):
    """Delete sound asset. Returns 409 if sound bindings exist."""
    async with async_session() as session:
        sa = await session.get(SoundAsset, uuid.UUID(sound_id))
        if sa is None:
            raise HTTPException(status_code=404, detail="Sound asset not found")

        bind_result = await session.execute(
            select(func.count(SoundBinding.id)).where(SoundBinding.sound_asset_id == sa.id)
        )
        binding_count = bind_result.scalar() or 0
        if binding_count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete sound with {binding_count} binding(s). Remove bindings first.",
            )

        await session.delete(sa)
        await session.commit()
        return None


@asset_library_router.post("/sounds/{sound_id}/audio", status_code=200)
async def upload_sound_audio(sound_id: str, file: UploadFile = File(...)):
    """Upload audio file for a sound asset (20MB limit). Updates audio_url."""
    content = await file.read()
    _validate_audio_upload(file, content)

    async with async_session() as session:
        sa = await session.get(SoundAsset, uuid.UUID(sound_id))
        if sa is None:
            raise HTTPException(status_code=404, detail="Sound asset not found")

        filename = file.filename or "audio.mp3"
        stored_path = await _save_upload(
            content,
            file.content_type or "audio/mpeg",
            filename,
            f"sounds/{sa.id}",
        )

        sa.audio_url = stored_path
        await session.commit()
        await session.refresh(sa)

        bind_result = await session.execute(
            select(func.count(SoundBinding.id)).where(SoundBinding.sound_asset_id == sa.id)
        )

        return _sound_asset_to_dict(sa, bind_result.scalar() or 0)
