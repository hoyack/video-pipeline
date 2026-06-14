"""SQLAlchemy 2.0 ORM models for Video Pipeline."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, JSON, Integer, BigInteger, Float, Boolean, ForeignKey, Index, UniqueConstraint, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Production(Base):
    """Production model representing a named collection of scenes.

    Domain hierarchy: Production → Scene → Shot
    """
    __tablename__ = "productions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    production_bible_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("production_bibles.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class ProductionBible(Base):
    """ProductionBible model representing a standalone, reusable asset collection.

    Formerly called Manifest. Renamed in Phase 16 to align with production
    terminology — a Production Bible is the authoritative reference document
    for characters, environments, props, and style in a video production.

    Spec reference: V2 Manifest System / Phase 16 Production Bible Foundation
    """
    __tablename__ = "production_bibles"

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
    parent_production_bible_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("production_bibles.id"), nullable=True, index=True
    )
    source_video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_video_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


# Backwards-compatibility alias — remove after frontend migration in Plan 03
Manifest = ProductionBible


class Asset(Base):
    """Asset model representing tagged visual elements within a manifest.

    Spec reference: V2 Manifest System
    """
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
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

    # Phase 12: Fork inheritance tracking
    is_inherited: Mapped[bool] = mapped_column(Boolean, default=False)
    inherited_from_asset: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("assets.id"), nullable=True)
    inherited_from_scene: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("scenes.id"), nullable=True)


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
    """Track where each asset appears across shots in generated content.

    Enables:
    - UI timeline view (show which assets appear in which shots)
    - Debugging queries (find all shots containing CHAR_01)
    - Continuity validation (did expected asset appear?)

    Spec reference: Phase 9 - CV Analysis Pipeline
    """
    __tablename__ = "asset_appearances"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), index=True)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    shot_index: Mapped[int] = mapped_column(Integer)
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
    production_bible_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("production_bibles.id"), index=True)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    version_at_snapshot: Mapped[int] = mapped_column(Integer)
    snapshot_data: Mapped[dict] = mapped_column(JSON)  # Full production bible + assets serialized
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Sequence(Base):
    """Sequence model representing an optional grouping layer above Scenes.

    Sequences belong to a Production and contain an ordered set of Scenes.
    Scenes without a sequence_id remain in a flat list (unsequenced).

    Spec reference: Issue #24 - Sequence Grouping Layer
    """
    __tablename__ = "sequences"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("productions.id"), index=True
    )
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    order: Mapped[int] = mapped_column(Integer, default=0)
    act: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # ACT_1, ACT_2, ACT_3, or null
    color: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # hex color e.g. "#3B82F6"
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


# ---------------------------------------------------------------------------
# Phase 17: Production Bible Entity Expansion
# Characters, Sets, Props, ScoreThemes, SFXItems and their sub-entities.
# These entities provide structured production data for LLM context injection.
# ---------------------------------------------------------------------------


class Character(Base):
    """Character entity within a Production Bible.

    Represents a named cast member with role, appearance, wardrobe, and voice.
    Sub-entities: Wardrobe (1:N), VoiceProfile (1:1).

    Spec reference: Phase 17 - PBEX-01
    """
    __tablename__ = "characters"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(30))  # PROTAGONIST, ANTAGONIST, SUPPORTING, EXTRA
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    arc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor_refs: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of image URLs/keys
    base_appearance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of strings
    # Phase 22: Promotion tracking — links to library Actor
    promoted_to_actor_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("actors.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class Wardrobe(Base):
    """Wardrobe item belonging to a Character (1:N relationship).

    Each character can have multiple wardrobe items; one is flagged is_default.

    Spec reference: Phase 17 - PBEX-02
    """
    __tablename__ = "wardrobes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    character_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("characters.id"), index=True
    )
    label: Mapped[str] = mapped_column(Text)
    reference_images: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of image keys
    scene_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_descriptor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class VoiceProfile(Base):
    """Voice profile for a Character — 1:1 enforced via unique constraint.

    Stores TTS adapter config and style notes for dialogue generation.

    Spec reference: Phase 17 - PBEX-03
    """
    __tablename__ = "voice_profiles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    character_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("characters.id"), unique=True, index=True  # 1:1 enforced
    )
    voice_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(50), default="ELEVENLABS")
    style_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Set(Base):
    """Set (location/environment) entity within a Production Bible.

    Stores visual reference, reverse prompt, and sonic identity sub-entity.
    Sub-entity: SonicIdentity (1:1).

    Spec reference: Phase 17 - PBEX-07
    """
    __tablename__ = "sets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    reference_image: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    reverse_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    style_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    lighting_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Phase 22: Promotion tracking — links to library LibrarySet
    promoted_to_library_set_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("library_sets.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class SonicIdentity(Base):
    """Sonic identity for a Set — 1:1 enforced via unique constraint.

    Captures ambient audio characteristics for a location.

    Spec reference: Phase 17 - PBEX-08
    """
    __tablename__ = "sonic_identities"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    set_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sets.id"), unique=True, index=True  # 1:1 enforced
    )
    ambience_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Prop(Base):
    """Prop entity within a Production Bible.

    Represents a physical object used in production with optional character associations.

    Spec reference: Phase 17 - PBEX-13
    """
    __tablename__ = "props"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    reference_image: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    associated_characters: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of character UUID strings
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Phase 22: Promotion tracking — links to library LibraryProp
    promoted_to_library_prop_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("library_props.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class ScoreTheme(Base):
    """Score theme for a Production Bible — musical identity for scenes.

    Stores mood, tempo, and generation hints for music generation adapters.

    Spec reference: Phase 17 - PBEX-16
    """
    __tablename__ = "score_themes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    mood_descriptors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of strings
    tempo_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    usage_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(50), default="MUSIC_GEN")
    # Phase 22: Promotion tracking — links to library SoundAsset
    promoted_to_sound_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("sound_assets.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class SFXItem(Base):
    """Sound effects item within a Production Bible.

    Categorised SFX entry with optional source audio and generation prompt.

    Spec reference: Phase 17 - PBEX-17
    """
    __tablename__ = "sfx_items"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))  # IMPACT, MECHANICAL, NATURAL, UI, FOLEY, AMBIENCE
    source_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of strings
    # Phase 22: Promotion tracking — links to library SoundAsset
    promoted_to_sound_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("sound_assets.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


# ---------------------------------------------------------------------------
# Phase 22: Asset Library & Actor-Character Model
# Standalone library entities (Actor, LibrarySet, LibraryProp, SoundAsset)
# with sub-entity tables and binding tables to Production Bibles.
# ---------------------------------------------------------------------------


class Actor(Base):
    """Standalone Actor entity in the Asset Library.

    Actors exist independently of any Production Bible and can be bound
    to multiple bibles via CastBinding. Sub-entities: ActorRef (1:N),
    ActorVoiceProfile (1:N), ActorWardrobePreset (1:N).

    Spec reference: Phase 22 - ALIB-01
    """
    __tablename__ = "actors"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    base_appearance_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )

    # Phase 25: LoRA training state
    lora_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lora_trained_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    lora_training_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # QUEUED/TRAINING/COMPLETED/FAILED
    lora_training_job_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)


class ActorRef(Base):
    """Reference image for an Actor (1:N sub-entity).

    Stores front/profile/3-4 angle reference images for visual consistency.

    Spec reference: Phase 22 - ALIB-01
    """
    __tablename__ = "actor_refs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("actors.id"), index=True)
    image_url: Mapped[str] = mapped_column(String(500))
    label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # front/profile/3-4
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    face_embedding: Mapped[Optional[bytes]] = mapped_column(nullable=True)  # numpy.tobytes() 512-dim float32
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ActorVoiceProfile(Base):
    """Voice profile for an Actor (1:N sub-entity).

    Separate from existing VoiceProfile (which has character_id FK) to avoid
    coupling library entities to bible-scoped entities.

    Spec reference: Phase 22 - ALIB-01
    """
    __tablename__ = "actor_voice_profiles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("actors.id"), index=True)
    voice_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    adapter_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)  # ELEVENLABS/BARK/XTTS/CUSTOM
    style_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ActorWardrobePreset(Base):
    """Wardrobe preset for an Actor (1:N sub-entity).

    Separate from existing Wardrobe (which has character_id FK) to avoid
    coupling library entities to bible-scoped entities.

    Spec reference: Phase 22 - ALIB-01
    """
    __tablename__ = "actor_wardrobe_presets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("actors.id"), index=True)
    label: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_images: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of image URLs
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LibrarySet(Base):
    """Standalone Set entity in the Asset Library.

    Named 'library_sets' to avoid collision with existing 'sets' table
    (bible-scoped). Can be bound to bibles via SetBinding.

    Spec reference: Phase 22 - ALIB-03
    """
    __tablename__ = "library_sets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reverse_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    style_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    lighting_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class LibrarySetRef(Base):
    """Reference image for a LibrarySet (1:N sub-entity).

    Spec reference: Phase 22 - ALIB-03
    """
    __tablename__ = "library_set_refs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    library_set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library_sets.id"), index=True)
    image_url: Mapped[str] = mapped_column(String(500))
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LibrarySonicIdentity(Base):
    """Sonic identity for a LibrarySet (1:1 sub-entity).

    Separate from existing SonicIdentity (which has set_id FK).

    Spec reference: Phase 22 - ALIB-03
    """
    __tablename__ = "library_sonic_identities"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    library_set_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("library_sets.id"), unique=True, index=True  # 1:1 enforced
    )
    ambience_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_audio_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LibraryProp(Base):
    """Standalone Prop entity in the Asset Library.

    Named 'library_props' to avoid collision with existing 'props' table.
    Can be bound to bibles via PropBinding.

    Spec reference: Phase 22 - ALIB-05
    """
    __tablename__ = "library_props"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    appearance_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class LibraryPropRef(Base):
    """Reference image for a LibraryProp (1:N sub-entity).

    Spec reference: Phase 22 - ALIB-05
    """
    __tablename__ = "library_prop_refs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    library_prop_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library_props.id"), index=True)
    image_url: Mapped[str] = mapped_column(String(500))
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class SoundAsset(Base):
    """Standalone Sound entity in the Asset Library (unified).

    Combines score themes, SFX, ambience, foley, and UI sounds into a
    single entity with category discrimination.

    Spec reference: Phase 22 - ALIB-09
    """
    __tablename__ = "sound_assets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))  # SCORE_THEME/SFX/AMBIENCE/FOLEY/UI
    subcategory: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audio_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class CastBinding(Base):
    """Binds an Actor to a Production Bible with role and overrides.

    The tag column enables tag resolution in prompt injection (e.g. KING_ALDRIC).
    UniqueConstraints prevent duplicate actor or tag within the same bible.

    Spec reference: Phase 22 - ALIB-01
    """
    __tablename__ = "cast_bindings"
    __table_args__ = (
        UniqueConstraint("production_bible_id", "actor_id", name="uq_cast_bible_actor"),
        UniqueConstraint("production_bible_id", "tag", name="uq_cast_bible_tag"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("actors.id"), index=True)
    tag: Mapped[str] = mapped_column(String(100))  # e.g. KING_ALDRIC
    character_name: Mapped[str] = mapped_column(Text)
    character_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    character_arc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String(30))  # LEAD/SUPPORTING/EXTRA/NARRATOR
    identity_type: Mapped[str] = mapped_column(
        String(20),
        default="HUMAN",
        server_default=text("'HUMAN'"),
    )  # HUMAN/ANIMAL/CREATURE/OBJECT
    wardrobe_override: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    voice_profile_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("actor_voice_profiles.id"), nullable=True
    )
    behavioral_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class CastLook(Base):
    """A wardrobe-specific look for a CastBinding.

    Each CastLook ties a tag (e.g. BRANDON_PIRATE) to a wardrobe preset,
    enabling @BRANDON_PIRATE in screenplays to resolve to wardrobe-specific
    reference images and descriptions instead of the base actor refs.

    If is_default=True, the parent CastBinding's own tag resolves to this
    wardrobe preset instead of base ActorRefs.
    """
    __tablename__ = "cast_looks"
    __table_args__ = (
        UniqueConstraint("cast_binding_id", "tag", name="uq_cast_look_binding_tag"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    cast_binding_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cast_bindings.id"), index=True)
    wardrobe_preset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("actor_wardrobe_presets.id"), index=True)
    tag: Mapped[str] = mapped_column(String(100))  # e.g. BRANDON_PIRATE
    is_default: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class SetBinding(Base):
    """Binds a LibrarySet to a Production Bible with overrides.

    Spec reference: Phase 22 - ALIB-03
    """
    __tablename__ = "set_bindings"
    __table_args__ = (
        UniqueConstraint("production_bible_id", "library_set_id", name="uq_set_bible_library_set"),
        UniqueConstraint("production_bible_id", "tag", name="uq_set_bible_tag"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    library_set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library_sets.id"), index=True)
    tag: Mapped[str] = mapped_column(String(100))
    production_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lighting_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sonic_override: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class PropBinding(Base):
    """Binds a LibraryProp to a Production Bible with overrides.

    Spec reference: Phase 22 - ALIB-05
    """
    __tablename__ = "prop_bindings"
    __table_args__ = (
        UniqueConstraint("production_bible_id", "library_prop_id", name="uq_prop_bible_library_prop"),
        UniqueConstraint("production_bible_id", "tag", name="uq_prop_bible_tag"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    library_prop_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library_props.id"), index=True)
    tag: Mapped[str] = mapped_column(String(100))
    production_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class SoundBinding(Base):
    """Binds a SoundAsset to a Production Bible with usage notes.

    Spec reference: Phase 22 - ALIB-09
    """
    __tablename__ = "sound_bindings"
    __table_args__ = (
        UniqueConstraint("production_bible_id", "sound_asset_id", name="uq_sound_bible_sound_asset"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    sound_asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sound_assets.id"), index=True)
    tag: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    usage_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class Screenplay(Base):
    """Screenplay entity attached 1:1 to a Production.

    Stores structured narrative content: logline, treatment, character breakdowns,
    scene breakdown, script, and shot list. Generated incrementally by ScreenwriterService.

    Spec reference: Phase 18 Screenplay System (SCRN-01, SCRN-02, SCRN-05)
    """
    __tablename__ = "screenplays"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("productions.id"), unique=True, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    genre: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="DRAFT")  # DRAFT/IN_REVIEW/LOCKED
    logline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    treatment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    character_breakdowns: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    scene_breakdown: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    script: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    shot_list: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    text_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    generating_step: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Current step being generated (for progress)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class VoiceScript(Base):
    """Generated voice script attached 1:1 to a screenplay.

    Stores narration and dialogue lines separately from the screenplay text so
    they can be edited, bound to cast voices, rendered, mixed, and lip-synced.
    """
    __tablename__ = "voice_scripts"
    __table_args__ = (
        UniqueConstraint("screenplay_id", name="uq_voice_scripts_screenplay_id"),
        Index("idx_voice_scripts_production", "production_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    screenplay_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("screenplays.id"), index=True)
    production_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("productions.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="DRAFT")
    version: Mapped[int] = mapped_column(Integer, default=1)
    script_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_screenplay_updated_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    voice_generation_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    mix_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    lip_sync_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class VoiceLine(Base):
    """A narration or dialogue line that can be rendered by a TTS provider."""
    __tablename__ = "voice_lines"
    __table_args__ = (
        Index("idx_voice_lines_script_order", "voice_script_id", "line_index"),
        Index("idx_voice_lines_scene_order", "scene_id", "line_index"),
        Index("idx_voice_lines_shot_order", "shot_id", "line_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    voice_script_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("voice_scripts.id"), index=True)
    production_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("productions.id"), index=True)
    scene_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    scene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("scenes.id"), nullable=True, index=True)
    shot_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shot_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("shots.id"), nullable=True, index=True)
    line_index: Mapped[int] = mapped_column(Integer)
    line_type: Mapped[str] = mapped_column(String(20), default="DIALOGUE")
    speaker_tag: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    speaker_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cast_binding_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("cast_bindings.id"), nullable=True, index=True
    )
    actor_voice_profile_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("actor_voice_profiles.id"), nullable=True, index=True
    )
    character_voice_profile_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("voice_profiles.id"), nullable=True, index=True
    )
    voice_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(50), default="ELEVENLABS")
    text: Mapped[str] = mapped_column(Text)
    delivery_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timing_hint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    audio_mime_type: Mapped[str] = mapped_column(String(50), default="audio/mpeg")
    generation_status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    lip_sync_mode: Mapped[str] = mapped_column(String(20), default="AUTO")
    lip_sync_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    provider_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class VoiceMixArtifact(Base):
    """Rendered voice stem or mixed audio artifact for a voice script."""
    __tablename__ = "voice_mix_artifacts"
    __table_args__ = (
        Index("idx_voice_mix_script", "voice_script_id"),
        Index("idx_voice_mix_scene", "scene_id"),
        Index("idx_voice_mix_shot", "shot_id"),
        Index("idx_voice_mix_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    voice_script_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("voice_scripts.id"), index=True)
    scene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("scenes.id"), nullable=True, index=True)
    shot_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("shots.id"), nullable=True, index=True)
    artifact_type: Mapped[str] = mapped_column(String(40), default="SCENE_VOICE_STEM")
    audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="READY")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class SoundEffectCue(Base):
    """Editable production sound-deck cue for SFX, ambience, foley, or UI audio."""
    __tablename__ = "sound_effect_cues"
    __table_args__ = (
        Index("idx_sound_cue_production", "production_id"),
        Index("idx_sound_cue_scene", "scene_id"),
        Index("idx_sound_cue_shot", "shot_id"),
        Index("idx_sound_cue_status", "generation_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("productions.id"), index=True)
    scene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("scenes.id"), nullable=True, index=True)
    shot_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("shots.id"), nullable=True, index=True)
    scene_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shot_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cue_index: Mapped[int] = mapped_column(Integer, default=0)
    cue_type: Mapped[str] = mapped_column(String(30), default="SFX")
    name: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    timing_hint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_db: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sound_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("sound_assets.id"), nullable=True, index=True
    )
    sfx_item_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("sfx_items.id"), nullable=True, index=True
    )
    audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    audio_mime_type: Mapped[str] = mapped_column(String(50), default="audio/mpeg")
    generation_status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    provider_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class SoundMixArtifact(Base):
    """Rendered sound-deck stem for a production or scene."""
    __tablename__ = "sound_mix_artifacts"
    __table_args__ = (
        Index("idx_sound_mix_production", "production_id"),
        Index("idx_sound_mix_scene", "scene_id"),
        Index("idx_sound_mix_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("productions.id"), index=True)
    scene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("scenes.id"), nullable=True, index=True)
    artifact_type: Mapped[str] = mapped_column(String(40), default="SCENE_SFX_STEM")
    audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="READY")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LipSyncJob(Base):
    """Tracks a post-video lip-sync operation for one voice line and shot."""
    __tablename__ = "lip_sync_jobs"
    __table_args__ = (
        Index("idx_lip_sync_line", "voice_line_id"),
        Index("idx_lip_sync_shot", "shot_id"),
        Index("idx_lip_sync_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    voice_line_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("voice_lines.id"), nullable=True, index=True
    )
    shot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("shots.id"), index=True)
    input_clip_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("video_clips.id"), nullable=True, index=True
    )
    input_audio_path: Mapped[str] = mapped_column(String(500))
    output_clip_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(50), default="FAKE")
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    eligibility_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class Scene(Base):
    """Scene model representing a video generation scene.

    Spec reference: Section 4.1
    """
    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    prompt: Mapped[str] = mapped_column(Text)
    style: Mapped[str] = mapped_column(String(50))
    aspect_ratio: Mapped[str] = mapped_column(String(10))
    target_clip_duration: Mapped[int] = mapped_column(Integer)
    target_shot_count: Mapped[int] = mapped_column(Integer, default=3)
    dynamic_shot_count: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    total_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    image_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    video_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    audio_enabled: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    forked_from_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("scenes.id"), nullable=True
    )
    production_bible_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("production_bibles.id"), nullable=True, index=True
    )
    production_bible_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Phase 11: Multi-Candidate Quality Mode
    quality_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    candidate_count: Mapped[int] = mapped_column(Integer, default=1)

    # Phase 13: LLM Provider Abstraction
    vision_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Selective stage execution ("Generate Through")
    run_through: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    deleted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # PipeSVN: version control
    head_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    # Production association
    production_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("productions.id"), nullable=True, index=True
    )

    # Phase 16: Sequence grouping (optional)
    sequence_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("sequences.id"), nullable=True, index=True
    )
    scene_order: Mapped[int] = mapped_column(Integer, default=0)

    # Phase 17: Score theme association for Director agent
    score_theme_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("score_themes.id"), nullable=True, index=True
    )

    # Phase 18: Screenplay linkage
    screenplay_breakdown_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    screenplay_context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Screenwriter Agent: script analysis output
    script_analysis: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

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


class Shot(Base):
    """Shot model representing a single shot within a scene.

    Spec reference: Section 4.2
    """
    __tablename__ = "shots"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    shot_index: Mapped[int] = mapped_column(Integer)
    shot_description: Mapped[str] = mapped_column(Text)
    start_frame_prompt: Mapped[str] = mapped_column(Text)
    end_frame_prompt: Mapped[str] = mapped_column(Text)
    video_motion_prompt: Mapped[str] = mapped_column(Text)
    transition_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50))
    generation_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default=None
    )

    # Screenwriter Agent: per-shot character assignment and narrative metadata
    characters_present: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    beat_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    narrative_intent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emotional_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ShotManifest(Base):
    """Per-shot asset placement manifest with composition metadata.

    Spec reference: Phase 7
    """
    __tablename__ = "shot_manifests"

    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), primary_key=True)
    shot_index: Mapped[int] = mapped_column(Integer, primary_key=True)
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


class ShotAudioManifest(Base):
    """Per-shot audio direction manifest with dialogue, SFX, ambient, and music.

    Spec reference: Phase 7
    """
    __tablename__ = "shot_audio_manifests"

    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), primary_key=True)
    shot_index: Mapped[int] = mapped_column(Integer, primary_key=True)
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
    """Keyframe model representing start/end frame images for shots.

    Spec reference: Section 4.3
    """
    __tablename__ = "keyframes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    shot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("shots.id"), index=True)
    position: Mapped[str] = mapped_column(String(10))  # 'start' or 'end'
    prompt_used: Mapped[str] = mapped_column(Text)
    file_path: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(20))  # 'generated' or 'inherited'
    verification_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    verification_attempts: Mapped[int] = mapped_column(Integer, default=0)
    verification_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Post-generation face-swap pass marker (idempotent resume): True once the
    # face-swap pass has finalized this keyframe (swapped or deliberately skipped).
    face_swapped: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class VideoClip(Base):
    """VideoClip model representing generated video clips for shots.

    Spec reference: Section 4.4
    """
    __tablename__ = "video_clips"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    shot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("shots.id"), index=True)
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
    prompt_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    veo_submission_count: Mapped[int] = mapped_column(Integer, default=0)
    safety_regen_count: Mapped[int] = mapped_column(Integer, default=0)


class GenerationCandidate(Base):
    """Stores per-candidate video clips with individual and composite quality scores.

    Spec reference: Phase 11 - Multi-Candidate Quality Mode
    """
    __tablename__ = "generation_candidates"
    __table_args__ = (
        Index("idx_candidates_scene_shot", "scene_id", "shot_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    shot_index: Mapped[int] = mapped_column(Integer)
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
    """PipelineRun model tracking execution metrics for a scene.

    Spec reference: Section 4.5
    """
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    total_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_api_cost_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# ---------------------------------------------------------------------------
# User Settings (single-user, no auth)
# ---------------------------------------------------------------------------

DEFAULT_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class User(Base):
    """Single default user for settings storage."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), default="default")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class UserSettings(Base):
    """Per-user settings with model enable/disable and GCP credentials."""
    __tablename__ = "user_settings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"), unique=True, index=True
    )

    # Enabled models per category — null = all enabled (zero-config default)
    enabled_text_models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    enabled_image_models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    enabled_video_models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Default model selections
    default_text_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    default_image_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    default_video_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    default_vision_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # GCP configuration
    gcp_project_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    gcp_location: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    vertex_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ComfyUI configuration
    comfyui_host: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    comfyui_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    comfyui_cost_per_second: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Display preferences
    show_cost: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))

    # Phase 13: Ollama configuration
    ollama_use_cloud: Mapped[bool] = mapped_column(Boolean, default=False)
    ollama_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ollama_endpoint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ollama_models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # ^ list of {"id": "ollama/llama3.1", "label": "Llama 3.1", "enabled": true, "vision": false}

    # GCP service account JSON (full credential file contents)
    gcp_service_account_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ElevenLabs configuration
    elevenlabs_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    default_voice_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Phase 25: Replicate configuration
    replicate_api_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    replicate_username: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class SceneCheckpoint(Base):
    """Immutable snapshot of scene state for version control.

    Spec reference: PipeSVN
    """
    __tablename__ = "scene_checkpoints"
    __table_args__ = (
        UniqueConstraint("scene_id", "sha", name="uq_checkpoint_scene_sha"),
        Index("idx_checkpoint_scene_created", "scene_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)
    sha: Mapped[str] = mapped_column(String(40))
    parent_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    snapshot_data: Mapped[dict] = mapped_column(JSON)
    message: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
