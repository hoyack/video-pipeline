"""Production master rendering API routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from vidpipe.db import async_session
from vidpipe.schemas.production_master import ProductionMasterResponse
from vidpipe.services.production_master import ProductionMasterError, ProductionMasterService
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend

production_master_router = APIRouter(prefix="/api")
production_master_service = ProductionMasterService()


@production_master_router.post(
    "/productions/{production_id}/render-master",
    response_model=ProductionMasterResponse,
)
async def render_production_master(production_id: str) -> ProductionMasterResponse:
    """Render a final MP4 from completed production scenes and voice stems."""
    async with async_session() as session:
        try:
            result = await production_master_service.render_master(
                session,
                uuid.UUID(production_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ProductionMasterError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ProductionMasterResponse(
        production_id=str(result.production_id),
        video_path=result.video_path,
        video_url=result.video_url,
        scene_count=result.scene_count,
        duration_seconds=result.duration_seconds,
        audio_stem_count=result.audio_stem_count,
    )


@production_master_router.get(
    "/productions/{production_id}/master",
    response_model=ProductionMasterResponse,
)
async def get_production_master(production_id: str) -> ProductionMasterResponse:
    """Return metadata for an already-rendered production master video."""
    async with async_session() as session:
        try:
            result = await production_master_service.get_existing_master(
                session,
                uuid.UUID(production_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ProductionMasterResponse(
        production_id=str(result.production_id),
        video_path=result.video_path,
        video_url=result.video_url,
        scene_count=result.scene_count,
        duration_seconds=result.duration_seconds,
        audio_stem_count=result.audio_stem_count,
    )


@production_master_router.get("/productions/{production_id}/master-video")
async def get_production_master_video(production_id: str):
    """Serve a rendered production master video."""
    storage_key = production_master_service.master_storage_key(uuid.UUID(production_id))
    storage = get_storage_backend()
    if isinstance(storage, LocalStorageBackend):
        path = storage.resolve_local_path(storage_key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Production master not found")
        return FileResponse(Path(path), media_type="video/mp4")

    try:
        return Response(await storage.get(storage_key), media_type="video/mp4")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Production master not found") from exc
