"""Set, SonicIdentity, and Prop CRUD API routes.

Sets are location/environment entities within a Production Bible with visual
references, reverse prompts, and sonic identity sub-entities.
Props are physical objects used in production.

Sub-entities:
  - SonicIdentity (1:1 per Set) — ambient audio characteristics
  - Reference image upload with LLM Vision reverse-prompting for Sets

Spec reference: Phase 17 - PBEX-07, PBEX-08, PBEX-09, PBEX-13, PBEX-14
"""

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, select

from vidpipe.db import async_session
from vidpipe.db.models import ProductionBible, Prop, Set, SonicIdentity
from vidpipe.services.storage_backend import get_storage_backend, LocalStorageBackend

logger = logging.getLogger(__name__)

sets_props_router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SetCreate(BaseModel):
    name: str
    style_tags: Optional[list[str]] = None
    lighting_notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class SetUpdate(BaseModel):
    name: Optional[str] = None
    reverse_prompt: Optional[str] = None
    style_tags: Optional[list[str]] = None
    lighting_notes: Optional[str] = None
    prompt_tags: Optional[list[str]] = None


class SonicIdentityUpsert(BaseModel):
    ambience_description: Optional[str] = None
    reference_audio: Optional[str] = None
    generation_prompt: Optional[str] = None


class PropCreate(BaseModel):
    name: str
    description: Optional[str] = None
    associated_characters: Optional[list[str]] = None
    prompt_tags: Optional[list[str]] = None


class PropUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    associated_characters: Optional[list[str]] = None
    prompt_tags: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_to_dict(s: Set, sonic_identity: Optional[SonicIdentity] = None) -> dict:
    """Build response dict for a set with optional nested sonic identity."""
    si_dict = None
    if sonic_identity is not None:
        si_dict = {
            "sonic_identity_id": str(sonic_identity.id),
            "set_id": str(sonic_identity.set_id),
            "ambience_description": sonic_identity.ambience_description,
            "reference_audio": sonic_identity.reference_audio,
            "generation_prompt": sonic_identity.generation_prompt,
            "created_at": sonic_identity.created_at.isoformat(),
        }

    return {
        "set_id": str(s.id),
        "production_bible_id": str(s.production_bible_id),
        "name": s.name,
        "reference_image": s.reference_image,
        "reverse_prompt": s.reverse_prompt,
        "style_tags": s.style_tags,
        "lighting_notes": s.lighting_notes,
        "prompt_tags": s.prompt_tags,
        "sonic_identity": si_dict,
        "created_at": s.created_at.isoformat(),
        "updated_at": s.updated_at.isoformat(),
    }


def _sonic_identity_to_dict(si: SonicIdentity) -> dict:
    return {
        "sonic_identity_id": str(si.id),
        "set_id": str(si.set_id),
        "ambience_description": si.ambience_description,
        "reference_audio": si.reference_audio,
        "generation_prompt": si.generation_prompt,
        "created_at": si.created_at.isoformat(),
    }


def _prop_to_dict(p: Prop) -> dict:
    return {
        "prop_id": str(p.id),
        "production_bible_id": str(p.production_bible_id),
        "name": p.name,
        "reference_image": p.reference_image,
        "description": p.description,
        "associated_characters": p.associated_characters,
        "prompt_tags": p.prompt_tags,
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Set endpoints
# ---------------------------------------------------------------------------


@sets_props_router.get("/production-bibles/{production_bible_id}/sets")
async def list_sets(production_bible_id: str):
    """List all sets for a production bible with nested sonic identities."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        result = await session.execute(
            select(Set)
            .where(Set.production_bible_id == uuid.UUID(production_bible_id))
            .order_by(Set.created_at)
        )
        sets = result.scalars().all()

        # Bulk fetch sonic identities
        set_ids = [s.id for s in sets]
        if set_ids:
            si_result = await session.execute(
                select(SonicIdentity).where(SonicIdentity.set_id.in_(set_ids))
            )
            all_si = si_result.scalars().all()
        else:
            all_si = []

        si_by_set: dict[uuid.UUID, SonicIdentity] = {}
        for si in all_si:
            si_by_set[si.set_id] = si

        return [_set_to_dict(s, si_by_set.get(s.id)) for s in sets]


@sets_props_router.post(
    "/production-bibles/{production_bible_id}/sets", status_code=201
)
async def create_set(production_bible_id: str, body: SetCreate):
    """Create a new set within a production bible."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        s = Set(
            production_bible_id=uuid.UUID(production_bible_id),
            name=body.name,
            style_tags=body.style_tags,
            lighting_notes=body.lighting_notes,
            prompt_tags=body.prompt_tags,
        )
        session.add(s)
        await session.commit()
        await session.refresh(s)

        return _set_to_dict(s, None)


@sets_props_router.get("/sets/{set_id}")
async def get_set(set_id: str):
    """Get a single set with sonic identity."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        si_result = await session.execute(
            select(SonicIdentity).where(SonicIdentity.set_id == s.id)
        )
        sonic_identity = si_result.scalars().first()

        return _set_to_dict(s, sonic_identity)


@sets_props_router.put("/sets/{set_id}")
async def update_set(set_id: str, body: SetUpdate):
    """Update set fields. Use model_fields_set for optional clearing."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        if body.name is not None:
            s.name = body.name
        if "reverse_prompt" in body.model_fields_set:
            s.reverse_prompt = body.reverse_prompt
        if "style_tags" in body.model_fields_set:
            s.style_tags = body.style_tags
        if "lighting_notes" in body.model_fields_set:
            s.lighting_notes = body.lighting_notes
        if "prompt_tags" in body.model_fields_set:
            s.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(s)

        si_result = await session.execute(
            select(SonicIdentity).where(SonicIdentity.set_id == s.id)
        )
        sonic_identity = si_result.scalars().first()

        return _set_to_dict(s, sonic_identity)


@sets_props_router.delete("/sets/{set_id}", status_code=204)
async def delete_set(set_id: str):
    """Delete set and its sonic identity."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        # Delete sonic identity first (no relationship() cascade)
        await session.execute(
            delete(SonicIdentity).where(SonicIdentity.set_id == s.id)
        )

        await session.delete(s)
        await session.commit()

        return None


@sets_props_router.post("/sets/{set_id}/upload-reference")
async def upload_set_reference(set_id: str, file: UploadFile = File(...)):
    """Upload reference image for a set, then run inline LLM Vision reverse-prompting."""
    # Validate content type
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be image/png, image/jpeg, or image/webp",
        )

    content = await file.read()

    # Validate file size (max 10MB)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 10MB")

    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        storage = get_storage_backend()
        filename = file.filename or "upload.png"

        if isinstance(storage, LocalStorageBackend):
            # Local: write to disk
            from vidpipe.config import settings as _settings
            local_dir = _settings.storage.tmp_dir / "manifests" / str(s.production_bible_id) / "sets" / str(s.id)
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            await asyncio.to_thread(local_path.write_bytes, content)
            s.reference_image = str(local_path)
            reverse_prompt_path = str(local_path)
        else:
            # S3: upload to storage + keep local copy for pipeline reads
            key = f"manifests/{s.production_bible_id}/sets/{s.id}/{filename}"
            await storage.put(key, content, file.content_type or "image/png")
            # Keep local copy for LLM Vision / pipeline reads
            from vidpipe.config import settings as _settings
            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            s.reference_image = key
            reverse_prompt_path = str(local_path)

        # Inline LLM Vision reverse-prompt (graceful degradation)
        try:
            from vidpipe.services.reverse_prompt_service import ReversePromptService
            svc = ReversePromptService()
            result = await svc.reverse_prompt_asset(
                reverse_prompt_path, "ENVIRONMENT", user_name=s.name
            )
            s.reverse_prompt = result["reverse_prompt"]
        except Exception as e:
            logger.warning("Set reverse-prompt failed for %s: %s", set_id, e)
            # Continue -- reverse_prompt stays None (graceful degradation)

        await session.commit()
        await session.refresh(s)

        si_result = await session.execute(
            select(SonicIdentity).where(SonicIdentity.set_id == s.id)
        )
        sonic_identity = si_result.scalars().first()

        return _set_to_dict(s, sonic_identity)


@sets_props_router.get("/sets/{set_id}/prompt-context")
async def set_prompt_context(set_id: str):
    """Compute and return injection string for a set on demand."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        parts = [f"SET [{s.name}]:"]
        if s.reverse_prompt:
            parts.append(s.reverse_prompt + ".")
        if s.lighting_notes:
            parts.append(f"Lighting: {s.lighting_notes}.")
        if s.style_tags:
            parts.append(f"Style: {', '.join(s.style_tags)}.")
        if s.prompt_tags:
            parts.append(f"Prompt tags: {', '.join(s.prompt_tags)}.")

        injection_string = " ".join(parts)

        return {
            "entity_id": str(s.id),
            "injection_string": injection_string,
        }


# ---------------------------------------------------------------------------
# SonicIdentity endpoints
# ---------------------------------------------------------------------------


@sets_props_router.get("/sets/{set_id}/sonic-identity")
async def get_sonic_identity(set_id: str):
    """Get the sonic identity for a set."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        result = await session.execute(
            select(SonicIdentity).where(SonicIdentity.set_id == uuid.UUID(set_id))
        )
        si = result.scalars().first()
        if si is None:
            raise HTTPException(status_code=404, detail="Sonic identity not found")

        return _sonic_identity_to_dict(si)


@sets_props_router.put("/sets/{set_id}/sonic-identity")
async def upsert_sonic_identity(set_id: str, body: SonicIdentityUpsert):
    """Upsert sonic identity for a set. Creates if not exists, updates if exists."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        result = await session.execute(
            select(SonicIdentity).where(SonicIdentity.set_id == uuid.UUID(set_id))
        )
        si = result.scalars().first()

        if si is None:
            si = SonicIdentity(
                set_id=uuid.UUID(set_id),
                ambience_description=body.ambience_description,
                reference_audio=body.reference_audio,
                generation_prompt=body.generation_prompt,
            )
            session.add(si)
        else:
            if "ambience_description" in body.model_fields_set:
                si.ambience_description = body.ambience_description
            if "reference_audio" in body.model_fields_set:
                si.reference_audio = body.reference_audio
            if "generation_prompt" in body.model_fields_set:
                si.generation_prompt = body.generation_prompt

        await session.commit()
        await session.refresh(si)

        return _sonic_identity_to_dict(si)


@sets_props_router.delete("/sets/{set_id}/sonic-identity", status_code=204)
async def delete_sonic_identity(set_id: str):
    """Delete the sonic identity for a set."""
    async with async_session() as session:
        s = await session.get(Set, uuid.UUID(set_id))
        if s is None:
            raise HTTPException(status_code=404, detail="Set not found")

        result = await session.execute(
            select(SonicIdentity).where(SonicIdentity.set_id == uuid.UUID(set_id))
        )
        si = result.scalars().first()
        if si is None:
            raise HTTPException(status_code=404, detail="Sonic identity not found")

        await session.delete(si)
        await session.commit()

        return None


# ---------------------------------------------------------------------------
# Prop endpoints
# ---------------------------------------------------------------------------


@sets_props_router.get("/production-bibles/{production_bible_id}/props")
async def list_props(production_bible_id: str):
    """List all props for a production bible."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        result = await session.execute(
            select(Prop)
            .where(Prop.production_bible_id == uuid.UUID(production_bible_id))
            .order_by(Prop.created_at)
        )
        props = result.scalars().all()

        return [_prop_to_dict(p) for p in props]


@sets_props_router.post(
    "/production-bibles/{production_bible_id}/props", status_code=201
)
async def create_prop(production_bible_id: str, body: PropCreate):
    """Create a new prop within a production bible."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        p = Prop(
            production_bible_id=uuid.UUID(production_bible_id),
            name=body.name,
            description=body.description,
            associated_characters=body.associated_characters,
            prompt_tags=body.prompt_tags,
        )
        session.add(p)
        await session.commit()
        await session.refresh(p)

        return _prop_to_dict(p)


@sets_props_router.get("/props/{prop_id}")
async def get_prop(prop_id: str):
    """Get a single prop."""
    async with async_session() as session:
        p = await session.get(Prop, uuid.UUID(prop_id))
        if p is None:
            raise HTTPException(status_code=404, detail="Prop not found")

        return _prop_to_dict(p)


@sets_props_router.put("/props/{prop_id}")
async def update_prop(prop_id: str, body: PropUpdate):
    """Update prop fields."""
    async with async_session() as session:
        p = await session.get(Prop, uuid.UUID(prop_id))
        if p is None:
            raise HTTPException(status_code=404, detail="Prop not found")

        if body.name is not None:
            p.name = body.name
        if "description" in body.model_fields_set:
            p.description = body.description
        if "associated_characters" in body.model_fields_set:
            p.associated_characters = body.associated_characters
        if "prompt_tags" in body.model_fields_set:
            p.prompt_tags = body.prompt_tags

        await session.commit()
        await session.refresh(p)

        return _prop_to_dict(p)


@sets_props_router.delete("/props/{prop_id}", status_code=204)
async def delete_prop(prop_id: str):
    """Delete a prop."""
    async with async_session() as session:
        p = await session.get(Prop, uuid.UUID(prop_id))
        if p is None:
            raise HTTPException(status_code=404, detail="Prop not found")

        await session.delete(p)
        await session.commit()

        return None


@sets_props_router.post("/generate-reverse-prompt")
async def generate_reverse_prompt(file: UploadFile = File(...)):
    """Standalone reverse-prompt endpoint. Accepts an image and returns reverse_prompt text."""
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be image/png, image/jpeg, or image/webp",
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 10MB")

    # Save to a temp location for LLM Vision read
    from vidpipe.config import settings as _settings

    temp_dir = _settings.storage.tmp_dir / "reverse_prompt_uploads" / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "upload.png"
    temp_path = temp_dir / filename
    await asyncio.to_thread(temp_path.write_bytes, content)

    try:
        from vidpipe.services.reverse_prompt_service import ReversePromptService

        svc = ReversePromptService()
        result = await svc.reverse_prompt_asset(str(temp_path), "OBJECT")
        return {
            "reverse_prompt": result["reverse_prompt"],
            "visual_description": result.get("visual_description", ""),
        }
    except Exception as e:
        logger.error("Generate reverse-prompt failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate reverse prompt: {e}",
        )


@sets_props_router.post("/props/{prop_id}/upload-reference")
async def upload_prop_reference(prop_id: str, file: UploadFile = File(...)):
    """Upload reference image for a prop. No LLM Vision call (props lack reverse_prompt)."""
    # Validate content type
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be image/png, image/jpeg, or image/webp",
        )

    content = await file.read()

    # Validate file size (max 10MB)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 10MB")

    async with async_session() as session:
        p = await session.get(Prop, uuid.UUID(prop_id))
        if p is None:
            raise HTTPException(status_code=404, detail="Prop not found")

        storage = get_storage_backend()
        filename = file.filename or "upload.png"

        if isinstance(storage, LocalStorageBackend):
            from vidpipe.config import settings as _settings
            local_dir = _settings.storage.tmp_dir / "manifests" / str(p.production_bible_id) / "props" / str(p.id)
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            await asyncio.to_thread(local_path.write_bytes, content)
            p.reference_image = str(local_path)
        else:
            key = f"manifests/{p.production_bible_id}/props/{p.id}/{filename}"
            await storage.put(key, content, file.content_type or "image/png")
            from vidpipe.config import settings as _settings
            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            p.reference_image = key

        await session.commit()
        await session.refresh(p)

        return _prop_to_dict(p)
