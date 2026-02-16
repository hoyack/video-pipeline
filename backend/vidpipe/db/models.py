"""SQLAlchemy 2.0 ORM models for Video Pipeline."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, JSON, Integer, Float, Boolean, ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Project(Base):
    """Project model representing a video generation project.

    Spec reference: Section 4.1
    """
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    prompt: Mapped[str] = mapped_column(Text)
    style: Mapped[str] = mapped_column(String(50))
    aspect_ratio: Mapped[str] = mapped_column(String(10))
    target_clip_duration: Mapped[int] = mapped_column(Integer)
    target_scene_count: Mapped[int] = mapped_column(Integer, default=3)
    total_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    image_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    video_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    audio_enabled: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    forked_from_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("projects.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(50))
    style_guide: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    storyboard_raw: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class Scene(Base):
    """Scene model representing a single scene within a project.

    Spec reference: Section 4.2
    """
    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    scene_index: Mapped[int] = mapped_column(Integer)
    scene_description: Mapped[str] = mapped_column(Text)
    start_frame_prompt: Mapped[str] = mapped_column(Text)
    end_frame_prompt: Mapped[str] = mapped_column(Text)
    video_motion_prompt: Mapped[str] = mapped_column(Text)
    transition_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50))


class Keyframe(Base):
    """Keyframe model representing start/end frame images for scenes.

    Spec reference: Section 4.3
    """
    __tablename__ = "keyframes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    position: Mapped[str] = mapped_column(String(10))  # 'start' or 'end'
    prompt_used: Mapped[str] = mapped_column(Text)
    file_path: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(20))  # 'generated' or 'inherited'
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class VideoClip(Base):
    """VideoClip model representing generated video clips for scenes.

    Spec reference: Section 4.4
    """
    __tablename__ = "video_clips"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    operation_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="generated")
    status: Mapped[str] = mapped_column(String(50))
    gcs_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    local_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    poll_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    veo_submission_count: Mapped[int] = mapped_column(Integer, default=0)
    safety_regen_count: Mapped[int] = mapped_column(Integer, default=0)


class PipelineRun(Base):
    """PipelineRun model tracking execution metrics for a project.

    Spec reference: Section 4.5
    """
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    total_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_api_cost_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
