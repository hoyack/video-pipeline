"""Character, Wardrobe, and VoiceProfile CRUD API routes.

Characters are cast members within a Production Bible with role, appearance,
wardrobe items, and optional voice profile. Sub-entities:
  - Wardrobe (1:N) — costume/outfit entries per character
  - VoiceProfile (1:1) — TTS adapter config for dialogue generation

Spec reference: Phase 17 - PBEX-01, PBEX-02, PBEX-03, PBEX-04
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select

from vidpipe.db import async_session
from vidpipe.db.models import Character, ProductionBible, VoiceProfile, Wardrobe

logger = logging.getLogger(__name__)

character_router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Valid character roles
# ---------------------------------------------------------------------------
VALID_ROLES = {"PROTAGONIST", "ANTAGONIST", "SUPPORTING", "EXTRA"}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CharacterCreate(BaseModel):
    name: str
    role: str
    description: Optional[str] = None
    arc: Optional[str] = None
    base_appearance: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    arc: Optional[str] = None
    base_appearance: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class WardrobeCreate(BaseModel):
    label: str
    scene_context: Optional[str] = None
    prompt_descriptor: Optional[str] = None
    is_default: bool = False


class WardrobeUpdate(BaseModel):
    label: Optional[str] = None
    scene_context: Optional[str] = None
    prompt_descriptor: Optional[str] = None
    is_default: Optional[bool] = None


class VoiceProfileUpsert(BaseModel):
    voice_id: Optional[str] = None
    adapter_type: str = "ELEVENLABS"
    style_notes: Optional[str] = None
    sample_audio: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _character_to_dict(
    char: Character,
    wardrobes: list[Wardrobe],
    voice_profile: Optional[VoiceProfile],
) -> dict:
    """Build response dict for a character with nested sub-entities."""
    wardrobe_list = [
        {
            "wardrobe_id": str(w.id),
            "character_id": str(w.character_id),
            "label": w.label,
            "reference_images": w.reference_images,
            "scene_context": w.scene_context,
            "prompt_descriptor": w.prompt_descriptor,
            "is_default": w.is_default,
            "created_at": w.created_at.isoformat(),
        }
        for w in wardrobes
    ]

    vp_dict = None
    if voice_profile is not None:
        vp_dict = {
            "voice_profile_id": str(voice_profile.id),
            "character_id": str(voice_profile.character_id),
            "voice_id": voice_profile.voice_id,
            "adapter_type": voice_profile.adapter_type,
            "style_notes": voice_profile.style_notes,
            "sample_audio": voice_profile.sample_audio,
            "created_at": voice_profile.created_at.isoformat(),
        }

    return {
        "character_id": str(char.id),
        "production_bible_id": str(char.production_bible_id),
        "name": char.name,
        "role": char.role,
        "description": char.description,
        "arc": char.arc,
        "actor_refs": char.actor_refs,
        "base_appearance": char.base_appearance,
        "prompt_tags": char.prompt_tags,
        "wardrobe": wardrobe_list,
        "voice_profile": vp_dict,
        "created_at": char.created_at.isoformat(),
        "updated_at": char.updated_at.isoformat(),
    }


def _wardrobe_to_dict(w: Wardrobe) -> dict:
    return {
        "wardrobe_id": str(w.id),
        "character_id": str(w.character_id),
        "label": w.label,
        "reference_images": w.reference_images,
        "scene_context": w.scene_context,
        "prompt_descriptor": w.prompt_descriptor,
        "is_default": w.is_default,
        "created_at": w.created_at.isoformat(),
    }


def _voice_profile_to_dict(vp: VoiceProfile) -> dict:
    return {
        "voice_profile_id": str(vp.id),
        "character_id": str(vp.character_id),
        "voice_id": vp.voice_id,
        "adapter_type": vp.adapter_type,
        "style_notes": vp.style_notes,
        "sample_audio": vp.sample_audio,
        "created_at": vp.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Character endpoints
# ---------------------------------------------------------------------------


@character_router.get("/production-bibles/{production_bible_id}/characters")
async def list_characters(production_bible_id: str):
    """List all characters for a production bible with nested wardrobes and voice profiles."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        # Fetch characters
        result = await session.execute(
            select(Character)
            .where(Character.production_bible_id == uuid.UUID(production_bible_id))
            .order_by(Character.created_at)
        )
        characters = result.scalars().all()

        # Fetch all wardrobes and voice profiles for these characters in bulk
        char_ids = [c.id for c in characters]
        if char_ids:
            ward_result = await session.execute(
                select(Wardrobe).where(Wardrobe.character_id.in_(char_ids))
            )
            all_wardrobes = ward_result.scalars().all()

            vp_result = await session.execute(
                select(VoiceProfile).where(VoiceProfile.character_id.in_(char_ids))
            )
            all_voice_profiles = vp_result.scalars().all()
        else:
            all_wardrobes = []
            all_voice_profiles = []

        # Group by character_id
        wardrobes_by_char: dict[uuid.UUID, list[Wardrobe]] = {}
        for w in all_wardrobes:
            wardrobes_by_char.setdefault(w.character_id, []).append(w)

        vp_by_char: dict[uuid.UUID, VoiceProfile] = {}
        for vp in all_voice_profiles:
            vp_by_char[vp.character_id] = vp

        return [
            _character_to_dict(
                c,
                wardrobes_by_char.get(c.id, []),
                vp_by_char.get(c.id),
            )
            for c in characters
        ]


@character_router.post(
    "/production-bibles/{production_bible_id}/characters", status_code=201
)
async def create_character(production_bible_id: str, body: CharacterCreate):
    """Create a new character within a production bible."""
    if body.role not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid role '{body.role}'. Must be one of {sorted(VALID_ROLES)}.",
        )

    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        char = Character(
            production_bible_id=uuid.UUID(production_bible_id),
            name=body.name,
            role=body.role,
            description=body.description,
            arc=body.arc,
            base_appearance=body.base_appearance,
            prompt_tags=body.prompt_tags,
        )
        session.add(char)
        await session.commit()
        await session.refresh(char)

        return _character_to_dict(char, [], None)


@character_router.get("/characters/{character_id}")
async def get_character(character_id: str):
    """Get a single character with wardrobes and voice profile."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        ward_result = await session.execute(
            select(Wardrobe).where(Wardrobe.character_id == char.id)
        )
        wardrobes = ward_result.scalars().all()

        vp_result = await session.execute(
            select(VoiceProfile).where(VoiceProfile.character_id == char.id)
        )
        voice_profile = vp_result.scalars().first()

        return _character_to_dict(char, list(wardrobes), voice_profile)


@character_router.put("/characters/{character_id}")
async def update_character(character_id: str, body: CharacterUpdate):
    """Update character fields. Use model_fields_set for optional clearing."""
    if body.role is not None and body.role not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid role '{body.role}'. Must be one of {sorted(VALID_ROLES)}.",
        )

    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        if body.name is not None:
            char.name = body.name
        if body.role is not None:
            char.role = body.role
        if "description" in body.model_fields_set:
            char.description = body.description
        if "arc" in body.model_fields_set:
            char.arc = body.arc
        if "base_appearance" in body.model_fields_set:
            char.base_appearance = body.base_appearance
        if "prompt_tags" in body.model_fields_set:
            char.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(char)

        # Fetch sub-entities for response
        ward_result = await session.execute(
            select(Wardrobe).where(Wardrobe.character_id == char.id)
        )
        wardrobes = ward_result.scalars().all()

        vp_result = await session.execute(
            select(VoiceProfile).where(VoiceProfile.character_id == char.id)
        )
        voice_profile = vp_result.scalars().first()

        return _character_to_dict(char, list(wardrobes), voice_profile)


@character_router.delete("/characters/{character_id}", status_code=204)
async def delete_character(character_id: str):
    """Delete character and cascade-delete wardrobes and voice profile."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        # Delete sub-entities first (no relationship() cascade configured)
        await session.execute(
            delete(Wardrobe).where(Wardrobe.character_id == char.id)
        )
        await session.execute(
            delete(VoiceProfile).where(VoiceProfile.character_id == char.id)
        )

        await session.delete(char)
        await session.commit()

        return None


@character_router.get("/characters/{character_id}/prompt-context")
async def character_prompt_context(character_id: str):
    """Compute and return injection string for a character on demand."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        # Find default wardrobe
        ward_result = await session.execute(
            select(Wardrobe).where(
                Wardrobe.character_id == char.id,
                Wardrobe.is_default == True,  # noqa: E712
            )
        )
        default_wardrobe = ward_result.scalars().first()

        parts = [f"CHARACTER [{char.name}]: {char.role}."]
        if char.description:
            parts.append(char.description + ".")
        if char.base_appearance:
            parts.append(f"Base appearance: {char.base_appearance}.")
        if default_wardrobe:
            wd_text = f"Wardrobe (default): '{default_wardrobe.label}'"
            if default_wardrobe.prompt_descriptor:
                wd_text += f" - {default_wardrobe.prompt_descriptor}"
            parts.append(wd_text + ".")
        if char.prompt_tags:
            parts.append(f"Prompt tags: {', '.join(char.prompt_tags)}.")

        injection_string = " ".join(parts)

        return {
            "entity_id": str(char.id),
            "injection_string": injection_string,
        }


# ---------------------------------------------------------------------------
# Wardrobe endpoints
# ---------------------------------------------------------------------------


@character_router.get("/characters/{character_id}/wardrobes")
async def list_wardrobes(character_id: str):
    """List all wardrobe items for a character."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        result = await session.execute(
            select(Wardrobe)
            .where(Wardrobe.character_id == char.id)
            .order_by(Wardrobe.created_at)
        )
        wardrobes = result.scalars().all()

        return [_wardrobe_to_dict(w) for w in wardrobes]


@character_router.post("/characters/{character_id}/wardrobes", status_code=201)
async def create_wardrobe(character_id: str, body: WardrobeCreate):
    """Create a new wardrobe item for a character."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        w = Wardrobe(
            character_id=uuid.UUID(character_id),
            label=body.label,
            scene_context=body.scene_context,
            prompt_descriptor=body.prompt_descriptor,
            is_default=body.is_default,
        )
        session.add(w)
        await session.commit()
        await session.refresh(w)

        return _wardrobe_to_dict(w)


@character_router.put("/wardrobes/{wardrobe_id}")
async def update_wardrobe(wardrobe_id: str, body: WardrobeUpdate):
    """Update a wardrobe item."""
    async with async_session() as session:
        w = await session.get(Wardrobe, uuid.UUID(wardrobe_id))
        if w is None:
            raise HTTPException(status_code=404, detail="Wardrobe not found")

        if body.label is not None:
            w.label = body.label
        if "scene_context" in body.model_fields_set:
            w.scene_context = body.scene_context
        if "prompt_descriptor" in body.model_fields_set:
            w.prompt_descriptor = body.prompt_descriptor
        if body.is_default is not None:
            w.is_default = body.is_default

        await session.commit()
        await session.refresh(w)

        return _wardrobe_to_dict(w)


@character_router.delete("/wardrobes/{wardrobe_id}", status_code=204)
async def delete_wardrobe(wardrobe_id: str):
    """Delete a wardrobe item."""
    async with async_session() as session:
        w = await session.get(Wardrobe, uuid.UUID(wardrobe_id))
        if w is None:
            raise HTTPException(status_code=404, detail="Wardrobe not found")

        await session.delete(w)
        await session.commit()

        return None


# ---------------------------------------------------------------------------
# VoiceProfile endpoints
# ---------------------------------------------------------------------------


@character_router.get("/characters/{character_id}/voice-profile")
async def get_voice_profile(character_id: str):
    """Get the voice profile for a character."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        result = await session.execute(
            select(VoiceProfile).where(
                VoiceProfile.character_id == uuid.UUID(character_id)
            )
        )
        vp = result.scalars().first()
        if vp is None:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        return _voice_profile_to_dict(vp)


@character_router.put("/characters/{character_id}/voice-profile")
async def upsert_voice_profile(character_id: str, body: VoiceProfileUpsert):
    """Upsert voice profile for a character. Creates if not exists, updates if exists."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        result = await session.execute(
            select(VoiceProfile).where(
                VoiceProfile.character_id == uuid.UUID(character_id)
            )
        )
        vp = result.scalars().first()

        if vp is None:
            # Create new
            vp = VoiceProfile(
                character_id=uuid.UUID(character_id),
                voice_id=body.voice_id,
                adapter_type=body.adapter_type,
                style_notes=body.style_notes,
                sample_audio=body.sample_audio,
            )
            session.add(vp)
        else:
            # Update existing
            if "voice_id" in body.model_fields_set:
                vp.voice_id = body.voice_id
            if "adapter_type" in body.model_fields_set:
                vp.adapter_type = body.adapter_type
            if "style_notes" in body.model_fields_set:
                vp.style_notes = body.style_notes
            if "sample_audio" in body.model_fields_set:
                vp.sample_audio = body.sample_audio

        await session.commit()
        await session.refresh(vp)

        return _voice_profile_to_dict(vp)


@character_router.delete("/characters/{character_id}/voice-profile", status_code=204)
async def delete_voice_profile(character_id: str):
    """Delete the voice profile for a character."""
    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        result = await session.execute(
            select(VoiceProfile).where(
                VoiceProfile.character_id == uuid.UUID(character_id)
            )
        )
        vp = result.scalars().first()
        if vp is None:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        await session.delete(vp)
        await session.commit()

        return None
