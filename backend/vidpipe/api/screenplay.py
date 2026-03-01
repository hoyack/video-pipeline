"""Screenplay CRUD, generation, and scene creation API routes.

Screenplays are 1:1 with Productions and provide structured narrative content
(logline, treatment, character breakdowns, scene breakdown, shot list, script)
generated incrementally by the ScreenwriterService.

Spec reference: Phase 18 Screenplay System (SCRN-03, SCRN-06, SCRN-09, SCRN-12, SCRN-13, SCRN-14)
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from vidpipe.config import settings
from vidpipe.db import async_session
from vidpipe.db.models import Production, Scene, Screenplay, UserSettings, DEFAULT_USER_ID
from vidpipe.services.screenwriter import ScreenwriterService, load_bible_context
from vidpipe.services.llm import get_adapter

logger = logging.getLogger(__name__)

screenplay_router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ScreenplayResponse(BaseModel):
    id: str
    production_id: str
    title: Optional[str] = None
    genre: Optional[str] = None
    status: str
    logline: Optional[str] = None
    treatment: Optional[str] = None
    character_breakdowns: Optional[list] = None
    scene_breakdown: Optional[list] = None
    script: Optional[str] = None
    shot_list: Optional[list] = None
    text_model: Optional[str] = None
    generating_step: Optional[str] = None
    created_at: str
    updated_at: str


class ScreenplayUpdate(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None
    logline: Optional[str] = None
    treatment: Optional[str] = None
    character_breakdowns: Optional[list] = None
    scene_breakdown: Optional[list] = None
    script: Optional[str] = None
    shot_list: Optional[list] = None
    text_model: Optional[str] = None


class StatusUpdate(BaseModel):
    status: str  # DRAFT, IN_REVIEW, LOCKED


class GenerateRequest(BaseModel):
    text_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: Screenplay ORM -> response dict
# ---------------------------------------------------------------------------


def _screenplay_to_response(sp: Screenplay) -> dict:
    return {
        "id": str(sp.id),
        "production_id": str(sp.production_id),
        "title": sp.title,
        "genre": sp.genre,
        "status": sp.status,
        "logline": sp.logline,
        "treatment": sp.treatment,
        "character_breakdowns": sp.character_breakdowns,
        "scene_breakdown": sp.scene_breakdown,
        "script": sp.script,
        "shot_list": sp.shot_list,
        "text_model": sp.text_model,
        "generating_step": sp.generating_step,
        "created_at": sp.created_at.isoformat(),
        "updated_at": sp.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# CRUD Endpoints
# ---------------------------------------------------------------------------


@screenplay_router.get(
    "/productions/{production_id}/screenplay",
    response_model=ScreenplayResponse,
)
async def get_or_create_screenplay(production_id: str):
    """Get existing screenplay or create a DRAFT one (upsert pattern)."""
    async with async_session() as session:
        prod = await session.get(Production, uuid.UUID(production_id))
        if prod is None:
            raise HTTPException(status_code=404, detail="Production not found")

        result = await session.execute(
            select(Screenplay).where(
                Screenplay.production_id == uuid.UUID(production_id)
            )
        )
        sp = result.scalar_one_or_none()

        if sp is None:
            sp = Screenplay(
                production_id=uuid.UUID(production_id),
                title=prod.name,
                status="DRAFT",
            )
            session.add(sp)
            await session.commit()
            await session.refresh(sp)

        return _screenplay_to_response(sp)


@screenplay_router.put(
    "/productions/{production_id}/screenplay",
    response_model=ScreenplayResponse,
)
async def update_screenplay(production_id: str, body: ScreenplayUpdate):
    """Update screenplay fields. Returns 409 when LOCKED."""
    async with async_session() as session:
        result = await session.execute(
            select(Screenplay).where(
                Screenplay.production_id == uuid.UUID(production_id)
            )
        )
        sp = result.scalar_one_or_none()
        if sp is None:
            raise HTTPException(status_code=404, detail="Screenplay not found")

        if sp.status == "LOCKED":
            raise HTTPException(
                status_code=409,
                detail="Screenplay is locked -- unlock before editing",
            )

        # Apply non-None fields from request body
        for field_name in body.model_fields_set:
            setattr(sp, field_name, getattr(body, field_name))

        await session.commit()
        await session.refresh(sp)

        return _screenplay_to_response(sp)


@screenplay_router.patch(
    "/productions/{production_id}/screenplay/status",
    response_model=ScreenplayResponse,
)
async def update_screenplay_status(production_id: str, body: StatusUpdate):
    """Transition screenplay status (DRAFT/IN_REVIEW/LOCKED)."""
    valid_statuses = {"DRAFT", "IN_REVIEW", "LOCKED"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{body.status}'. Must be one of {sorted(valid_statuses)}",
        )

    async with async_session() as session:
        result = await session.execute(
            select(Screenplay).where(
                Screenplay.production_id == uuid.UUID(production_id)
            )
        )
        sp = result.scalar_one_or_none()
        if sp is None:
            raise HTTPException(status_code=404, detail="Screenplay not found")

        sp.status = body.status
        await session.commit()
        await session.refresh(sp)

        return _screenplay_to_response(sp)


# ---------------------------------------------------------------------------
# Scene Creation from Screenplay
# ---------------------------------------------------------------------------


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-scenes",
)
async def generate_scenes_from_screenplay(
    production_id: str,
    force: bool = Query(False, description="Delete existing screenplay-linked scenes first"),
):
    """Create Scenes from a LOCKED screenplay's scene breakdown."""
    async with async_session() as session:
        result = await session.execute(
            select(Screenplay).where(
                Screenplay.production_id == uuid.UUID(production_id)
            )
        )
        sp = result.scalar_one_or_none()
        if sp is None:
            raise HTTPException(status_code=404, detail="Screenplay not found")

        if sp.status != "LOCKED":
            raise HTTPException(
                status_code=409,
                detail="Screenplay must be LOCKED before generating scenes",
            )

        if not sp.scene_breakdown:
            raise HTTPException(
                status_code=400,
                detail="Screenplay has no scene breakdown to generate scenes from",
            )

        # Check for existing screenplay-linked scenes
        existing_result = await session.execute(
            select(Scene).where(
                Scene.production_id == uuid.UUID(production_id),
                Scene.screenplay_breakdown_index.isnot(None),
            )
        )
        existing_scenes = existing_result.scalars().all()

        if existing_scenes and not force:
            raise HTTPException(
                status_code=409,
                detail="Scenes already generated from this screenplay. Delete existing scenes first or use force=true query param.",
            )

        if existing_scenes and force:
            for s in existing_scenes:
                await session.delete(s)
            await session.flush()

        # Fetch the production's bible_id for propagation to new scenes
        prod_obj = await session.get(Production, uuid.UUID(production_id))
        prod_bible_id = prod_obj.production_bible_id if prod_obj else None

        # Create one Scene per breakdown entry
        created = []
        for idx, entry in enumerate(sp.scene_breakdown):
            scene = Scene(
                production_id=uuid.UUID(production_id),
                prompt=entry.get("intent", ""),
                title=entry.get("slugline", f"Scene {idx + 1}"),
                style="cinematic",
                aspect_ratio="16:9",
                target_clip_duration=8,
                status="draft",
                screenplay_breakdown_index=entry.get("scene_number", idx + 1),
                screenplay_context=entry,
                scene_order=idx,
                production_bible_id=prod_bible_id,
            )
            session.add(scene)
            created.append({"entry": entry, "scene": scene})

        await session.commit()

        # Build response after commit (IDs assigned)
        scene_summaries = []
        for item in created:
            await session.refresh(item["scene"])
            scene_summaries.append({
                "scene_id": str(item["scene"].id),
                "title": item["scene"].title,
                "scene_number": item["entry"].get("scene_number", 0),
            })

        return scene_summaries


# ---------------------------------------------------------------------------
# Generation Endpoints
# ---------------------------------------------------------------------------


async def _get_user_settings(session):
    """Load singleton UserSettings row."""
    us_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
    )
    return us_result.scalar_one_or_none()


async def _get_or_create_screenplay(session, production_id: str) -> Screenplay:
    """Get screenplay or create a DRAFT one. Returns ORM instance."""
    result = await session.execute(
        select(Screenplay).where(
            Screenplay.production_id == uuid.UUID(production_id)
        )
    )
    sp = result.scalar_one_or_none()
    if sp is None:
        prod = await session.get(Production, uuid.UUID(production_id))
        if prod is None:
            raise HTTPException(status_code=404, detail="Production not found")
        sp = Screenplay(
            production_id=uuid.UUID(production_id),
            title=prod.name,
            status="DRAFT",
        )
        session.add(sp)
        await session.commit()
        await session.refresh(sp)
    return sp


async def _run_background_generate(
    production_id_str: str,
    screenplay_id_str: str,
    text_model: Optional[str],
    bible_context: str,
    user_settings_dict: Optional[dict],
):
    """Background task for full screenplay generation.

    Accepts only plain values -- no ORM objects or sessions from the request.
    Opens its own async_session for the full chain.
    """
    async with async_session() as session:
        sp = await session.get(Screenplay, uuid.UUID(screenplay_id_str))
        if sp is None:
            logger.error("Background generate: screenplay %s not found", screenplay_id_str)
            return

        model_id = text_model or sp.text_model or settings.models.storyboard_llm
        # Reconstruct user_settings for adapter if needed
        us = None
        if user_settings_dict is not None:
            us_result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
            )
            us = us_result.scalar_one_or_none()

        adapter = get_adapter(model_id, user_settings=us)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_full(
                session,
                sp,
                bible_context,
                production_id=uuid.UUID(production_id_str),
            )
            logger.info(
                "Background generate complete for screenplay %s",
                screenplay_id_str,
            )
        except Exception:
            logger.exception(
                "Background generate failed for screenplay %s",
                screenplay_id_str,
            )
            # Clear generating_step so UI doesn't get stuck
            sp.generating_step = None
            await session.commit()


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate",
    status_code=202,
)
async def generate_full_screenplay(
    production_id: str,
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """Run full ScreenwriterService chain as BackgroundTask (202 response)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        # Extract plain values before launching background task
        screenplay_id_str = str(sp.id)
        text_model = request.text_model
        us_dict = {"id": str(user_settings.id)} if user_settings else None

    background_tasks.add_task(
        _run_background_generate,
        production_id,
        screenplay_id_str,
        text_model,
        bible_context,
        us_dict,
    )

    return {"status": "generating", "screenplay_id": screenplay_id_str}


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-logline",
    response_model=ScreenplayResponse,
)
async def generate_logline(production_id: str, request: GenerateRequest):
    """Generate logline and genre (inline, ~5-10 seconds)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        model_id = request.text_model or sp.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_logline(session, sp, bible_context)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return _screenplay_to_response(sp)


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-treatment",
    response_model=ScreenplayResponse,
)
async def generate_treatment(production_id: str, request: GenerateRequest):
    """Generate treatment from logline (inline)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        model_id = request.text_model or sp.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_treatment(session, sp, bible_context)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return _screenplay_to_response(sp)


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-character-breakdowns",
    response_model=ScreenplayResponse,
)
async def generate_character_breakdowns(production_id: str, request: GenerateRequest):
    """Generate character breakdowns (inline)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        model_id = request.text_model or sp.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_character_breakdowns(session, sp, bible_context)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return _screenplay_to_response(sp)


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-scene-breakdown",
    response_model=ScreenplayResponse,
)
async def generate_scene_breakdown(production_id: str, request: GenerateRequest):
    """Generate scene breakdown with entity validation (inline)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        model_id = request.text_model or sp.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_scene_breakdown(
                session, sp, bible_context, production_id=uuid.UUID(production_id)
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return _screenplay_to_response(sp)


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-shot-list",
    response_model=ScreenplayResponse,
)
async def generate_shot_list(production_id: str, request: GenerateRequest):
    """Generate shot list from scene breakdown (inline, independently regeneratable per SCRN-09)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        model_id = request.text_model or sp.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_shot_list(session, sp, bible_context)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return _screenplay_to_response(sp)


@screenplay_router.post(
    "/productions/{production_id}/screenplay/generate-script",
    response_model=ScreenplayResponse,
)
async def generate_script(production_id: str, request: GenerateRequest):
    """Generate full script (inline)."""
    async with async_session() as session:
        sp = await _get_or_create_screenplay(session, production_id)
        bible_context = await load_bible_context(session, uuid.UUID(production_id))
        user_settings = await _get_user_settings(session)

        model_id = request.text_model or sp.text_model or settings.models.storyboard_llm
        adapter = get_adapter(model_id, user_settings=user_settings)
        screenwriter = ScreenwriterService(adapter)

        try:
            await screenwriter.generate_script(session, sp, bible_context)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        return _screenplay_to_response(sp)
