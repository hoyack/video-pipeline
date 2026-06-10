"""Schemas for production master rendering."""

from pydantic import BaseModel, Field


class ProductionMasterResponse(BaseModel):
    """Response returned after rendering a production master video."""

    production_id: str
    video_path: str
    video_url: str
    scene_count: int = Field(ge=1)
    duration_seconds: float | None = None
    audio_stem_count: int = Field(ge=0)
