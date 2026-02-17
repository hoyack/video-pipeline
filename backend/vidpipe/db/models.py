"""SQLAlchemy 2.0 ORM models for Video Pipeline."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, JSON, Integer, Float, Boolean, ForeignKey, Index, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Manifest(Base):
    """Manifest model representing a standalone, reusable asset collection.

    Spec reference: V2 Manifest System
    """
    __tablename__ = "manifests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    category: Mapped[str] = mapped_column(String(50), default="CUSTOM")
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="DRAFT")
    processing_progress: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    contact_sheet_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    asset_count: Mapped[int] = mapped_column(Integer, default=0)
    total_processing_cost: Mapped[float] = mapped_column(Float, default=0.0)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_manifest_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("manifests.id"), nullable=True, index=True
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class Asset(Base):
    """Asset model representing tagged visual elements within a manifest.

    Spec reference: V2 Manifest System
    """
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    manifest_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("manifests.id"), index=True
    )
    asset_type: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(Text)
    manifest_tag: Mapped[str] = mapped_column(String(50))
    user_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    reference_image_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="uploaded")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Phase 5: Manifesting Engine fields
    reverse_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    visual_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detection_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    detection_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_face_crop: Mapped[bool] = mapped_column(Boolean, default=False)
    crop_bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [x1, y1, x2, y2]
    face_embedding: Mapped[Optional[bytes]] = mapped_column(nullable=True)  # numpy.tobytes() 512-dim float32
    clip_embedding: Mapped[Optional[bytes]] = mapped_column(nullable=True)  # numpy.tobytes() 512-dim float32
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("assets.id"), nullable=True)  # parent asset if extracted crop


class AssetCleanReference(Base):
    """Clean reference image generated for an asset at a specific quality tier.

    Stores preprocessed reference images separately from originals.
    Never overwrites Asset.reference_image_url.

    Spec reference: Phase 8 - Clean Sheets
    """
    __tablename__ = "asset_clean_references"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), index=True)
    tier: Mapped[str] = mapped_column(String(20))  # 'tier2_rembg', 'tier3_gemini'
    clean_image_url: Mapped[str] = mapped_column(String(500))
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    face_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    generation_cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AssetAppearance(Base):
    """Track where each asset appears across scenes in generated content.

    Enables:
    - UI timeline view (show which assets appear in which scenes)
    - Debugging queries (find all scenes containing CHAR_01)
    - Continuity validation (did expected asset appear?)

    Spec reference: Phase 9 - CV Analysis Pipeline
    """
    __tablename__ = "asset_appearances"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    scene_index: Mapped[int] = mapped_column(Integer)
    frame_index: Mapped[int] = mapped_column(Integer)  # Which sampled frame (0-7)
    timestamp_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [x1, y1, x2, y2]
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20))  # "yolo", "face_match", "clip_match"
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ManifestSnapshot(Base):
    """ManifestSnapshot model capturing immutable state of a manifest at generation time.

    Spec reference: Phase 6 - GenerateForm Integration
    """
    __tablename__ = "manifest_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    manifest_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("manifests.id"), index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    version_at_snapshot: Mapped[int] = mapped_column(Integer)
    snapshot_data: Mapped[dict] = mapped_column(JSON)  # Full manifest + assets serialized
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


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
    manifest_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("manifests.id"), nullable=True, index=True
    )
    manifest_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    candidate_count: Mapped[int] = mapped_column(Integer, default=1)

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


class SceneManifest(Base):
    """Per-scene asset placement manifest with composition metadata.

    Spec reference: Phase 7
    """
    __tablename__ = "scene_manifests"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    scene_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    manifest_json: Mapped[dict] = mapped_column(JSON)
    composition_shot_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    composition_camera_movement: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    asset_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    new_asset_count: Mapped[int] = mapped_column(Integer, default=0)
    selected_reference_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    cv_analysis_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    continuity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Phase 10: Adaptive Prompt Rewriting
    rewritten_keyframe_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rewritten_video_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class SceneAudioManifest(Base):
    """Per-scene audio direction manifest with dialogue, SFX, ambient, and music.

    Spec reference: Phase 7
    """
    __tablename__ = "scene_audio_manifests"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    scene_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    dialogue_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    sfx_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    ambient_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    music_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    audio_continuity_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    speaker_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    has_dialogue: Mapped[bool] = mapped_column(Boolean, default=False)
    has_music: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


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


class GenerationCandidate(Base):
    """Stores per-candidate video clips with individual and composite quality scores.

    Spec reference: Phase 11 - Multi-Candidate Quality Mode
    """
    __tablename__ = "generation_candidates"
    __table_args__ = (
        Index("idx_candidates_project_scene", "project_id", "scene_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    scene_index: Mapped[int] = mapped_column(Integer)
    candidate_number: Mapped[int] = mapped_column(Integer)  # 0-based index within batch
    local_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # First frame JPEG

    # Individual dimension scores (0-10)
    manifest_adherence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visual_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    continuity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prompt_adherence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Scoring details JSON blob for debugging and UI display
    scoring_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Selection state
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    selected_by: Mapped[str] = mapped_column(String(20), default="auto")  # 'auto' or 'user'

    # Cost tracking
    generation_cost: Mapped[float] = mapped_column(Float, default=0.0)
    scoring_cost: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


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
