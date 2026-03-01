"""Sound Department CRUD API routes.

ScoreTheme and SFXItem entities within a Production Bible.
ScoreThemes represent musical identities for scenes; SFXItems are categorised
sound-effect entries with optional source audio and generation prompts.

Spec reference: Phase 17 - PBEX-16, PBEX-17
"""

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select

from vidpipe.db import async_session
from vidpipe.db.models import ProductionBible, ScoreTheme, Set, SFXItem, SonicIdentity
from vidpipe.services.storage_backend import get_storage_backend, LocalStorageBackend

logger = logging.getLogger(__name__)

sound_router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Valid SFX categories
# ---------------------------------------------------------------------------
VALID_SFX_CATEGORIES = {"IMPACT", "MECHANICAL", "NATURAL", "UI", "FOLEY", "AMBIENCE"}

ALLOWED_AUDIO_TYPES = ("audio/mpeg", "audio/wav", "audio/ogg", "audio/webm", "audio/mp4", "audio/x-m4a")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ScoreThemeCreate(BaseModel):
    name: str
    mood_descriptors: Optional[list[str]] = None
    tempo_notes: Optional[str] = None
    usage_notes: Optional[str] = None
    generation_prompt: Optional[str] = None
    adapter_type: str = "MUSIC_GEN"


class ScoreThemeUpdate(BaseModel):
    name: Optional[str] = None
    mood_descriptors: Optional[list[str]] = None
    tempo_notes: Optional[str] = None
    usage_notes: Optional[str] = None
    reference_audio: Optional[str] = None
    generation_prompt: Optional[str] = None
    adapter_type: Optional[str] = None


class SFXItemCreate(BaseModel):
    name: str
    category: str  # validated against VALID_SFX_CATEGORIES
    generation_prompt: Optional[str] = None
    tags: Optional[list[str]] = None


class SFXItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None  # validated if provided
    source_audio: Optional[str] = None
    generation_prompt: Optional[str] = None
    tags: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# ScoreTheme endpoints
# ---------------------------------------------------------------------------


@sound_router.get("/production-bibles/{production_bible_id}/score-themes")
async def list_score_themes(production_bible_id: str):
    """List all score themes for a production bible."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        result = await session.execute(
            select(ScoreTheme)
            .where(ScoreTheme.production_bible_id == uuid.UUID(production_bible_id))
            .order_by(ScoreTheme.created_at)
        )
        themes = result.scalars().all()

        return [
            {
                "score_theme_id": str(t.id),
                "production_bible_id": str(t.production_bible_id),
                "name": t.name,
                "mood_descriptors": t.mood_descriptors,
                "tempo_notes": t.tempo_notes,
                "usage_notes": t.usage_notes,
                "reference_audio": t.reference_audio,
                "generation_prompt": t.generation_prompt,
                "adapter_type": t.adapter_type,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in themes
        ]


@sound_router.post("/production-bibles/{production_bible_id}/score-themes", status_code=201)
async def create_score_theme(production_bible_id: str, body: ScoreThemeCreate):
    """Create a new score theme for a production bible."""
    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        theme = ScoreTheme(
            production_bible_id=uuid.UUID(production_bible_id),
            name=body.name,
            mood_descriptors=body.mood_descriptors,
            tempo_notes=body.tempo_notes,
            usage_notes=body.usage_notes,
            generation_prompt=body.generation_prompt,
            adapter_type=body.adapter_type,
        )
        session.add(theme)
        await session.commit()
        await session.refresh(theme)

        return {
            "score_theme_id": str(theme.id),
            "production_bible_id": str(theme.production_bible_id),
            "name": theme.name,
            "mood_descriptors": theme.mood_descriptors,
            "tempo_notes": theme.tempo_notes,
            "usage_notes": theme.usage_notes,
            "reference_audio": theme.reference_audio,
            "generation_prompt": theme.generation_prompt,
            "adapter_type": theme.adapter_type,
            "created_at": theme.created_at.isoformat(),
            "updated_at": theme.updated_at.isoformat(),
        }


@sound_router.get("/score-themes/{score_theme_id}")
async def get_score_theme(score_theme_id: str):
    """Get a single score theme by ID."""
    async with async_session() as session:
        theme = await session.get(ScoreTheme, uuid.UUID(score_theme_id))
        if theme is None:
            raise HTTPException(status_code=404, detail="Score theme not found")

        return {
            "score_theme_id": str(theme.id),
            "production_bible_id": str(theme.production_bible_id),
            "name": theme.name,
            "mood_descriptors": theme.mood_descriptors,
            "tempo_notes": theme.tempo_notes,
            "usage_notes": theme.usage_notes,
            "reference_audio": theme.reference_audio,
            "generation_prompt": theme.generation_prompt,
            "adapter_type": theme.adapter_type,
            "created_at": theme.created_at.isoformat(),
            "updated_at": theme.updated_at.isoformat(),
        }


@sound_router.put("/score-themes/{score_theme_id}")
async def update_score_theme(score_theme_id: str, body: ScoreThemeUpdate):
    """Update a score theme. Uses model_fields_set for optional clearing."""
    async with async_session() as session:
        theme = await session.get(ScoreTheme, uuid.UUID(score_theme_id))
        if theme is None:
            raise HTTPException(status_code=404, detail="Score theme not found")

        if body.name is not None:
            theme.name = body.name
        if "mood_descriptors" in body.model_fields_set:
            theme.mood_descriptors = body.mood_descriptors
        if "tempo_notes" in body.model_fields_set:
            theme.tempo_notes = body.tempo_notes
        if "usage_notes" in body.model_fields_set:
            theme.usage_notes = body.usage_notes
        if "reference_audio" in body.model_fields_set:
            theme.reference_audio = body.reference_audio
        if "generation_prompt" in body.model_fields_set:
            theme.generation_prompt = body.generation_prompt
        if body.adapter_type is not None:
            theme.adapter_type = body.adapter_type

        await session.commit()
        await session.refresh(theme)

        return {
            "score_theme_id": str(theme.id),
            "production_bible_id": str(theme.production_bible_id),
            "name": theme.name,
            "mood_descriptors": theme.mood_descriptors,
            "tempo_notes": theme.tempo_notes,
            "usage_notes": theme.usage_notes,
            "reference_audio": theme.reference_audio,
            "generation_prompt": theme.generation_prompt,
            "adapter_type": theme.adapter_type,
            "created_at": theme.created_at.isoformat(),
            "updated_at": theme.updated_at.isoformat(),
        }


@sound_router.delete("/score-themes/{score_theme_id}", status_code=204)
async def delete_score_theme(score_theme_id: str):
    """Delete a score theme."""
    async with async_session() as session:
        theme = await session.get(ScoreTheme, uuid.UUID(score_theme_id))
        if theme is None:
            raise HTTPException(status_code=404, detail="Score theme not found")

        await session.delete(theme)
        await session.commit()


# ---------------------------------------------------------------------------
# SFXItem endpoints
# ---------------------------------------------------------------------------


@sound_router.get("/production-bibles/{production_bible_id}/sfx")
async def list_sfx(production_bible_id: str, category: Optional[str] = None):
    """List SFX items for a production bible. Optional category filter."""
    if category is not None and category not in VALID_SFX_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{category}'. Must be one of {sorted(VALID_SFX_CATEGORIES)}.",
        )

    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        query = select(SFXItem).where(
            SFXItem.production_bible_id == uuid.UUID(production_bible_id)
        )
        if category is not None:
            query = query.where(SFXItem.category == category)
        query = query.order_by(SFXItem.created_at)

        result = await session.execute(query)
        items = result.scalars().all()

        return [
            {
                "sfx_item_id": str(i.id),
                "production_bible_id": str(i.production_bible_id),
                "name": i.name,
                "category": i.category,
                "source_audio": i.source_audio,
                "generation_prompt": i.generation_prompt,
                "tags": i.tags,
                "created_at": i.created_at.isoformat(),
                "updated_at": i.updated_at.isoformat(),
            }
            for i in items
        ]


@sound_router.post("/production-bibles/{production_bible_id}/sfx", status_code=201)
async def create_sfx_item(production_bible_id: str, body: SFXItemCreate):
    """Create a new SFX item for a production bible."""
    if body.category not in VALID_SFX_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{body.category}'. Must be one of {sorted(VALID_SFX_CATEGORIES)}.",
        )

    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if bible is None:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        item = SFXItem(
            production_bible_id=uuid.UUID(production_bible_id),
            name=body.name,
            category=body.category,
            generation_prompt=body.generation_prompt,
            tags=body.tags,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)

        return {
            "sfx_item_id": str(item.id),
            "production_bible_id": str(item.production_bible_id),
            "name": item.name,
            "category": item.category,
            "source_audio": item.source_audio,
            "generation_prompt": item.generation_prompt,
            "tags": item.tags,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }


@sound_router.get("/sfx/{sfx_item_id}")
async def get_sfx_item(sfx_item_id: str):
    """Get a single SFX item by ID."""
    async with async_session() as session:
        item = await session.get(SFXItem, uuid.UUID(sfx_item_id))
        if item is None:
            raise HTTPException(status_code=404, detail="SFX item not found")

        return {
            "sfx_item_id": str(item.id),
            "production_bible_id": str(item.production_bible_id),
            "name": item.name,
            "category": item.category,
            "source_audio": item.source_audio,
            "generation_prompt": item.generation_prompt,
            "tags": item.tags,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }


@sound_router.put("/sfx/{sfx_item_id}")
async def update_sfx_item(sfx_item_id: str, body: SFXItemUpdate):
    """Update an SFX item."""
    if body.category is not None and body.category not in VALID_SFX_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category '{body.category}'. Must be one of {sorted(VALID_SFX_CATEGORIES)}.",
        )

    async with async_session() as session:
        item = await session.get(SFXItem, uuid.UUID(sfx_item_id))
        if item is None:
            raise HTTPException(status_code=404, detail="SFX item not found")

        if body.name is not None:
            item.name = body.name
        if body.category is not None:
            item.category = body.category
        if "source_audio" in body.model_fields_set:
            item.source_audio = body.source_audio
        if "generation_prompt" in body.model_fields_set:
            item.generation_prompt = body.generation_prompt
        if "tags" in body.model_fields_set:
            item.tags = body.tags

        await session.commit()
        await session.refresh(item)

        return {
            "sfx_item_id": str(item.id),
            "production_bible_id": str(item.production_bible_id),
            "name": item.name,
            "category": item.category,
            "source_audio": item.source_audio,
            "generation_prompt": item.generation_prompt,
            "tags": item.tags,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }


@sound_router.delete("/sfx/{sfx_item_id}", status_code=204)
async def delete_sfx_item(sfx_item_id: str):
    """Delete an SFX item."""
    async with async_session() as session:
        item = await session.get(SFXItem, uuid.UUID(sfx_item_id))
        if item is None:
            raise HTTPException(status_code=404, detail="SFX item not found")

        await session.delete(item)
        await session.commit()


# ---------------------------------------------------------------------------
# Asset-to-entity migration endpoint
# ---------------------------------------------------------------------------


@sound_router.post("/production-bibles/{production_bible_id}/migrate-entities", status_code=200)
async def migrate_entities(production_bible_id: str):
    """Migrate CHARACTER and ENVIRONMENT assets to Character and Set entities.

    Idempotent: calling twice does not create duplicates.
    """
    from vidpipe.services.production_bible_entity_service import migrate_all_assets

    async with async_session() as session:
        bible = await session.get(ProductionBible, uuid.UUID(production_bible_id))
        if not bible:
            raise HTTPException(status_code=404, detail="Production Bible not found")

        result = await migrate_all_assets(session, bible.id)
        await session.commit()
        return result


# ---------------------------------------------------------------------------
# Audio upload endpoints
# ---------------------------------------------------------------------------


def _score_theme_to_dict(t: ScoreTheme) -> dict:
    return {
        "score_theme_id": str(t.id),
        "production_bible_id": str(t.production_bible_id),
        "name": t.name,
        "mood_descriptors": t.mood_descriptors,
        "tempo_notes": t.tempo_notes,
        "usage_notes": t.usage_notes,
        "reference_audio": t.reference_audio,
        "generation_prompt": t.generation_prompt,
        "adapter_type": t.adapter_type,
        "created_at": t.created_at.isoformat(),
        "updated_at": t.updated_at.isoformat(),
    }


def _sfx_item_to_dict(item: SFXItem) -> dict:
    return {
        "sfx_item_id": str(item.id),
        "production_bible_id": str(item.production_bible_id),
        "name": item.name,
        "category": item.category,
        "source_audio": item.source_audio,
        "generation_prompt": item.generation_prompt,
        "tags": item.tags,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
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


@sound_router.post("/score-themes/{score_theme_id}/upload-audio")
async def upload_score_theme_audio(
    score_theme_id: str, file: UploadFile = File(...)
):
    """Upload reference audio for a score theme."""
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be one of {ALLOWED_AUDIO_TYPES}",
        )

    content = await file.read()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=422, detail="File too large. Maximum size is 20MB"
        )

    async with async_session() as session:
        theme = await session.get(ScoreTheme, uuid.UUID(score_theme_id))
        if theme is None:
            raise HTTPException(status_code=404, detail="Score theme not found")

        storage = get_storage_backend()
        filename = file.filename or "reference_audio.mp3"

        if isinstance(storage, LocalStorageBackend):
            from vidpipe.config import settings as _settings

            local_dir = (
                _settings.storage.tmp_dir
                / "manifests"
                / str(theme.production_bible_id)
                / "score_themes"
                / str(theme.id)
            )
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            await asyncio.to_thread(local_path.write_bytes, content)
            theme.reference_audio = str(local_path)
        else:
            key = f"manifests/{theme.production_bible_id}/score_themes/{theme.id}/{filename}"
            await storage.put(key, content, file.content_type or "audio/mpeg")
            from vidpipe.config import settings as _settings

            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(local_path.write_bytes, content)
            theme.reference_audio = key

        await session.commit()
        await session.refresh(theme)

        return _score_theme_to_dict(theme)


@sound_router.post("/sfx/{sfx_item_id}/upload-audio")
async def upload_sfx_audio(sfx_item_id: str, file: UploadFile = File(...)):
    """Upload source audio for an SFX item."""
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be one of {ALLOWED_AUDIO_TYPES}",
        )

    content = await file.read()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=422, detail="File too large. Maximum size is 20MB"
        )

    async with async_session() as session:
        item = await session.get(SFXItem, uuid.UUID(sfx_item_id))
        if item is None:
            raise HTTPException(status_code=404, detail="SFX item not found")

        storage = get_storage_backend()
        filename = file.filename or "source_audio.mp3"

        if isinstance(storage, LocalStorageBackend):
            from vidpipe.config import settings as _settings

            local_dir = (
                _settings.storage.tmp_dir
                / "manifests"
                / str(item.production_bible_id)
                / "sfx"
                / str(item.id)
            )
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            await asyncio.to_thread(local_path.write_bytes, content)
            item.source_audio = str(local_path)
        else:
            key = f"manifests/{item.production_bible_id}/sfx/{item.id}/{filename}"
            await storage.put(key, content, file.content_type or "audio/mpeg")
            from vidpipe.config import settings as _settings

            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(local_path.write_bytes, content)
            item.source_audio = key

        await session.commit()
        await session.refresh(item)

        return _sfx_item_to_dict(item)


@sound_router.post("/sonic-identities/{sonic_identity_id}/upload-audio")
async def upload_sonic_identity_audio(
    sonic_identity_id: str, file: UploadFile = File(...)
):
    """Upload reference audio for a sonic identity."""
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content type {file.content_type}. Must be one of {ALLOWED_AUDIO_TYPES}",
        )

    content = await file.read()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=422, detail="File too large. Maximum size is 20MB"
        )

    async with async_session() as session:
        si = await session.get(SonicIdentity, uuid.UUID(sonic_identity_id))
        if si is None:
            raise HTTPException(status_code=404, detail="Sonic identity not found")

        # Look up parent Set to get production_bible_id for storage path
        set_obj = await session.get(Set, si.set_id)

        storage = get_storage_backend()
        filename = file.filename or "reference_audio.mp3"

        if isinstance(storage, LocalStorageBackend):
            from vidpipe.config import settings as _settings

            local_dir = (
                _settings.storage.tmp_dir
                / "manifests"
                / str(set_obj.production_bible_id)
                / "sets"
                / str(set_obj.id)
                / "sonic_identity"
            )
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            await asyncio.to_thread(local_path.write_bytes, content)
            si.reference_audio = str(local_path)
        else:
            key = f"manifests/{set_obj.production_bible_id}/sets/{set_obj.id}/sonic_identity/{filename}"
            await storage.put(key, content, file.content_type or "audio/mpeg")
            from vidpipe.config import settings as _settings

            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(local_path.write_bytes, content)
            si.reference_audio = key

        await session.commit()
        await session.refresh(si)

        return _sonic_identity_to_dict(si)
