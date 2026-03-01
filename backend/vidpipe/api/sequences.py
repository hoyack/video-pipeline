"""Sequence CRUD API routes.

Sequences are optional containers above Scenes within a Production,
enabling narrative chapter/act organization. Scenes with sequence_id=null
remain in a flat (unsequenced) list.

Spec reference: Issue #24 - Sequence Grouping Layer
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, update

from vidpipe.db import async_session
from vidpipe.db.models import Production, Scene, Sequence

logger = logging.getLogger(__name__)

sequence_router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Valid act values
# ---------------------------------------------------------------------------
VALID_ACTS = {"ACT_1", "ACT_2", "ACT_3"}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class SequenceCreate(BaseModel):
    title: str
    description: Optional[str] = None
    act: Optional[str] = None  # ACT_1, ACT_2, ACT_3
    color: Optional[str] = None  # hex color e.g. "#3B82F6"


class SequenceUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    act: Optional[str] = None
    color: Optional[str] = None


class SequenceResponse(BaseModel):
    id: str
    production_id: str
    title: str
    description: Optional[str]
    order: int
    act: Optional[str]
    color: Optional[str]
    scene_count: int
    total_duration: Optional[int]  # sum of scene durations in seconds
    created_at: str
    updated_at: str


class SequenceWithScenes(SequenceResponse):
    scenes: list[dict]  # lightweight scene objects


class ReorderRequest(BaseModel):
    sequence_ids: list[str]  # ordered list of sequence UUIDs


class AssignSequenceRequest(BaseModel):
    sequence_id: Optional[str] = None  # null = unsequence
    scene_order: Optional[int] = None


# ---------------------------------------------------------------------------
# Helper: build SequenceResponse dict from a Sequence ORM row
# ---------------------------------------------------------------------------


def _seq_to_dict(seq: Sequence, scene_count: int, total_duration: Optional[int]) -> dict:
    return {
        "id": str(seq.id),
        "production_id": str(seq.production_id),
        "title": seq.title,
        "description": seq.description,
        "order": seq.order,
        "act": seq.act,
        "color": seq.color,
        "scene_count": scene_count,
        "total_duration": total_duration,
        "created_at": seq.created_at.isoformat(),
        "updated_at": seq.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /api/productions/{production_id}/sequences
# ---------------------------------------------------------------------------


@sequence_router.get("/productions/{production_id}/sequences", response_model=list[SequenceResponse])
async def list_sequences(production_id: str):
    """List all sequences for a production ordered by `order` field."""
    async with async_session() as session:
        # Verify production exists
        prod = await session.get(Production, uuid.UUID(production_id))
        if prod is None:
            raise HTTPException(status_code=404, detail="Production not found")

        # Fetch sequences with scene counts and total durations
        result = await session.execute(
            select(
                Sequence,
                func.count(Scene.id).label("scene_count"),
                func.sum(Scene.total_duration).label("total_duration"),
            )
            .outerjoin(Scene, Scene.sequence_id == Sequence.id)
            .where(Sequence.production_id == uuid.UUID(production_id))
            .group_by(Sequence.id)
            .order_by(Sequence.order)
        )
        rows = result.all()

        return [
            _seq_to_dict(row.Sequence, row.scene_count or 0, row.total_duration)
            for row in rows
        ]


# ---------------------------------------------------------------------------
# POST /api/productions/{production_id}/sequences
# ---------------------------------------------------------------------------


@sequence_router.post("/productions/{production_id}/sequences", response_model=SequenceResponse, status_code=201)
async def create_sequence(production_id: str, body: SequenceCreate):
    """Create a new sequence. Auto-assigns order to max(existing orders) + 1."""
    if body.act is not None and body.act not in VALID_ACTS:
        raise HTTPException(status_code=422, detail=f"Invalid act value '{body.act}'. Must be one of {sorted(VALID_ACTS)} or null.")

    async with async_session() as session:
        prod = await session.get(Production, uuid.UUID(production_id))
        if prod is None:
            raise HTTPException(status_code=404, detail="Production not found")

        # Determine next order value
        result = await session.execute(
            select(func.max(Sequence.order)).where(Sequence.production_id == uuid.UUID(production_id))
        )
        max_order = result.scalar() or -1
        next_order = max_order + 1

        seq = Sequence(
            production_id=uuid.UUID(production_id),
            title=body.title,
            description=body.description,
            order=next_order,
            act=body.act,
            color=body.color,
        )
        session.add(seq)
        await session.commit()
        await session.refresh(seq)

        return _seq_to_dict(seq, 0, None)


# ---------------------------------------------------------------------------
# GET /api/sequences/{sequence_id}
# ---------------------------------------------------------------------------


@sequence_router.get("/sequences/{sequence_id}", response_model=SequenceWithScenes)
async def get_sequence(sequence_id: str):
    """Get a single sequence with its scenes ordered by scene_order."""
    async with async_session() as session:
        seq = await session.get(Sequence, uuid.UUID(sequence_id))
        if seq is None:
            raise HTTPException(status_code=404, detail="Sequence not found")

        # Fetch scenes belonging to this sequence
        result = await session.execute(
            select(Scene)
            .where(Scene.sequence_id == uuid.UUID(sequence_id))
            .order_by(Scene.scene_order)
        )
        scenes = result.scalars().all()

        scene_list = [
            {
                "id": str(s.id),
                "title": s.title,
                "prompt": s.prompt,
                "status": s.status,
                "scene_order": s.scene_order,
                "total_duration": s.total_duration,
                "created_at": s.created_at.isoformat(),
            }
            for s in scenes
        ]

        total_duration = sum(s.total_duration or 0 for s in scenes) or None

        data = _seq_to_dict(seq, len(scenes), total_duration)
        data["scenes"] = scene_list
        return data


# ---------------------------------------------------------------------------
# PUT /api/sequences/{sequence_id}
# ---------------------------------------------------------------------------


@sequence_router.put("/sequences/{sequence_id}", response_model=SequenceResponse)
async def update_sequence(sequence_id: str, body: SequenceUpdate):
    """Update sequence fields."""
    if body.act is not None and body.act not in VALID_ACTS:
        raise HTTPException(status_code=422, detail=f"Invalid act value '{body.act}'. Must be one of {sorted(VALID_ACTS)} or null.")

    async with async_session() as session:
        seq = await session.get(Sequence, uuid.UUID(sequence_id))
        if seq is None:
            raise HTTPException(status_code=404, detail="Sequence not found")

        if body.title is not None:
            seq.title = body.title
        if body.description is not None:
            seq.description = body.description
        # Allow explicitly setting act/color to None to clear them
        if "act" in body.model_fields_set:
            seq.act = body.act
        if "color" in body.model_fields_set:
            seq.color = body.color

        await session.commit()
        await session.refresh(seq)

        # Fetch updated scene_count and total_duration
        result = await session.execute(
            select(
                func.count(Scene.id).label("scene_count"),
                func.sum(Scene.total_duration).label("total_duration"),
            ).where(Scene.sequence_id == uuid.UUID(sequence_id))
        )
        row = result.one()
        return _seq_to_dict(seq, row.scene_count or 0, row.total_duration)


# ---------------------------------------------------------------------------
# DELETE /api/sequences/{sequence_id}
# ---------------------------------------------------------------------------


@sequence_router.delete("/sequences/{sequence_id}")
async def delete_sequence(sequence_id: str):
    """Delete sequence and unsequence its child scenes (does not delete scenes)."""
    async with async_session() as session:
        seq = await session.get(Sequence, uuid.UUID(sequence_id))
        if seq is None:
            raise HTTPException(status_code=404, detail="Sequence not found")

        # Count scenes that will be unsequenced
        result = await session.execute(
            select(func.count(Scene.id)).where(Scene.sequence_id == uuid.UUID(sequence_id))
        )
        unsequenced_count = result.scalar() or 0

        # Unsequence scenes before deleting the sequence
        await session.execute(
            update(Scene)
            .where(Scene.sequence_id == uuid.UUID(sequence_id))
            .values(sequence_id=None, scene_order=0)
        )

        await session.delete(seq)
        await session.commit()

        return {"status": "deleted", "unsequenced_scenes": unsequenced_count}


# ---------------------------------------------------------------------------
# PUT /api/productions/{production_id}/sequences/reorder
# ---------------------------------------------------------------------------


@sequence_router.put("/productions/{production_id}/sequences/reorder")
async def reorder_sequences(production_id: str, body: ReorderRequest):
    """Bulk reorder sequences by updating their order fields."""
    async with async_session() as session:
        prod = await session.get(Production, uuid.UUID(production_id))
        if prod is None:
            raise HTTPException(status_code=404, detail="Production not found")

        # Validate all IDs belong to this production
        result = await session.execute(
            select(Sequence.id).where(Sequence.production_id == uuid.UUID(production_id))
        )
        existing_ids = {str(row[0]) for row in result.all()}

        for sid in body.sequence_ids:
            if sid not in existing_ids:
                raise HTTPException(
                    status_code=422,
                    detail=f"Sequence {sid} does not belong to production {production_id}",
                )

        # Update order fields to match the provided list's index
        for idx, sid in enumerate(body.sequence_ids):
            await session.execute(
                update(Sequence)
                .where(Sequence.id == uuid.UUID(sid))
                .values(order=idx)
            )

        await session.commit()
        return {"status": "reordered"}


# ---------------------------------------------------------------------------
# PUT /api/scenes/{scene_id}/sequence
# ---------------------------------------------------------------------------


@sequence_router.put("/scenes/{scene_id}/sequence")
async def assign_scene_to_sequence(scene_id: str, body: AssignSequenceRequest):
    """Assign or unassign a scene to/from a sequence."""
    async with async_session() as session:
        scene = await session.get(Scene, uuid.UUID(scene_id))
        if scene is None:
            raise HTTPException(status_code=404, detail="Scene not found")

        if body.sequence_id is None:
            # Unassign: clear sequence_id and reset scene_order
            scene.sequence_id = None
            scene.scene_order = 0
            await session.commit()
            return {"status": "unassigned", "sequence_id": None, "scene_order": 0}

        # Assign: validate sequence exists and belongs to the same production
        seq = await session.get(Sequence, uuid.UUID(body.sequence_id))
        if seq is None:
            raise HTTPException(status_code=404, detail="Sequence not found")

        if scene.production_id is None or seq.production_id != scene.production_id:
            raise HTTPException(
                status_code=422,
                detail="Sequence does not belong to the same production as the scene",
            )

        # Determine scene_order: use provided value or auto-increment
        if body.scene_order is not None:
            scene_order = body.scene_order
        else:
            result = await session.execute(
                select(func.max(Scene.scene_order)).where(
                    Scene.sequence_id == uuid.UUID(body.sequence_id)
                )
            )
            max_order = result.scalar()
            scene_order = (max_order + 1) if max_order is not None else 0

        scene.sequence_id = uuid.UUID(body.sequence_id)
        scene.scene_order = scene_order
        await session.commit()

        return {
            "status": "assigned",
            "sequence_id": body.sequence_id,
            "scene_order": scene_order,
        }
