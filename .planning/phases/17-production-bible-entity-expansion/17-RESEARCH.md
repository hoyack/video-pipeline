# Phase 17: Production Bible Entity Expansion - Research

**Researched:** 2026-02-28
**Domain:** SQLAlchemy ORM entity expansion, FastAPI CRUD API design, React tabbed UI with sub-entities, LLM Vision auto-prompting
**Confidence:** HIGH

## Summary

Phase 17 adds structured first-class entities to the Production Bible: Character (with Wardrobe and VoiceProfile sub-entities), Set (with SonicIdentity sub-entity), Prop, ScoreTheme, and SFXItem. It also adds a `score_theme_id` nullable FK on `Scene` for forward compatibility with a future Director agent. These entities replace the existing flat `Asset` system for structured production data while the `Asset` model continues to serve the raw upload/embedding workflow.

The phase divides cleanly into four layers: (1) ORM models + DB migrations, (2) dedicated API route files per domain, (3) a service layer for business logic, and (4) React UI split across Casting, Art Department, and Sound tabs inside `ProductionBibleCreator`. The existing `sequences.py` file is the best architectural template for new route files — it is self-contained, imports only what it needs, and uses `async_session()` directly.

LLM Vision reverse-prompt generation for Set entities on image upload reuses `ReversePromptService` with a Set-specific system prompt, exactly matching how asset reprocessing works today. Audio generation buttons for VoiceProfile and ScoreTheme must be disabled with a clear "ElevenLabs adapter not yet available" tooltip until Issue #12 ships.

**Primary recommendation:** Implement in four plans: (1) ORM + migrations, (2) API route files, (3) service + migration logic for existing asset migration, (4) frontend tabs.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PBEX-01 | Character entity: name, role (PROTAGONIST/ANTAGONIST/SUPPORTING/EXTRA), description, arc, actor_refs images, base_appearance, wardrobe items, voice_profile, prompt_tags | ORM Mapped[] columns + JSON arrays; role is a String enum |
| PBEX-02 | Wardrobe sub-entity per character: label, reference_images, scene_context, prompt_descriptor, is_default toggle | Separate ORM table with character_id FK; JSON for reference_images list |
| PBEX-03 | VoiceProfile sub-entity per character: voice_id, adapter_type (ELEVENLABS), style_notes, sample_audio | Separate ORM table with character_id FK; 1:1 with Character |
| PBEX-04 | Character CRUD API under `/api/production-bibles/:id/characters` + `/api/characters/:id` with prompt-context endpoint | New `characters.py` route file per CLAUDE.md domain separation; inject router in app.py |
| PBEX-05 | Character detail UI: Overview, Actor References, Wardrobe, Voice Profile tabs in Casting tab | React tabbed panel; reuse AssetUploader component for image uploads |
| PBEX-06 | Existing manifest CHARACTER assets migrated to Character entities on first load | Migration service function: query Asset WHERE asset_type='CHARACTER', create Character row, mark migrated |
| PBEX-07 | Set entity: name, reference_image, reverse_prompt, style_tags, lighting_notes, prompt_tags, sonic_identity | ORM model; reverse_prompt auto-generated via LLM Vision on upload |
| PBEX-08 | SonicIdentity sub-entity per set: ambience_description, reference_audio, generation_prompt | Separate ORM table with set_id FK; 1:1 with Set |
| PBEX-09 | LLM Vision reverse-prompt auto-generation for Sets on reference image upload | Reuse ReversePromptService with ENVIRONMENT system prompt; trigger async after upload commit |
| PBEX-10 | Set CRUD API under `/api/production-bibles/:id/sets` + `/api/sets/:id` with prompt-context endpoint | New `sets_props.py` route file |
| PBEX-11 | Set detail UI: Visual and Sonic Identity tabs in Art Department tab | React two-tab panel inside Art Department |
| PBEX-12 | Existing background/scene/ENVIRONMENT assets migrated to Set entities | Same migration pattern as PBEX-06; query Asset WHERE asset_type='ENVIRONMENT' |
| PBEX-13 | Prop entity: name, reference_image, description, associated_characters, prompt_tags | ORM model; associated_characters is JSON list of character IDs |
| PBEX-14 | Prop CRUD API under `/api/production-bibles/:id/props` + `/api/props/:id` | Included in `sets_props.py` route file |
| PBEX-15 | Prop list/detail UI in Art Department tab with thumbnail grid | Reuse thumbnail grid pattern from existing AssetEditor |
| PBEX-16 | ScoreTheme entity: name, mood_descriptors, tempo_notes, usage_notes, reference_audio, generation_prompt, adapter_type | ORM model; JSON for mood_descriptors; adapter_type is String (MUSIC_GEN etc.) |
| PBEX-17 | SFXItem entity: name, category (IMPACT/MECHANICAL/NATURAL/UI/FOLEY/AMBIENCE), source_audio, generation_prompt, tags | ORM model; category is String enum; tags is JSON |
| PBEX-18 | ScoreTheme and SFXItem CRUD API under `/api/production-bibles/:id/score-themes` and `/api/production-bibles/:id/sfx` | New `sound.py` route file |
| PBEX-19 | Sound Department tab UI: Score Themes and SFX Library sections with category filters | New Sound tab in ProductionBibleCreator; category filter as pills (existing UI pattern) |
| PBEX-20 | Scene.score_theme_id nullable FK for forward compatibility with Director agent | ALTER TABLE migration: `ALTER TABLE scenes ADD COLUMN score_theme_id TEXT REFERENCES score_themes(id)` |
</phase_requirements>

---

## Standard Stack

### Core (already in use — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy 2.0 | >=2.0 | ORM with `Mapped[Type]` annotations | Existing pattern in `models.py` |
| FastAPI | >=0.115 | Route handlers, Pydantic request/response schemas | Existing app framework |
| Pydantic v2 | >=2.0 | Request/response models with validation | Existing pattern |
| aiosqlite + asyncpg | existing | Dual-driver async DB | Existing infrastructure |
| React 19 + TypeScript | existing | Frontend tabs and entity detail views | Existing frontend stack |
| Tailwind CSS 4 | existing | Styling | Existing convention |

### No New Dependencies Required

All capabilities needed for Phase 17 exist in the current stack:
- JSON columns for list fields (already used for `tags`, `user_tags`, `prompt_tags`)
- LLM Vision via `ReversePromptService` (already exists for set reverse-prompting)
- File uploads via `UploadFile` (already used for `upload_asset_image`)
- Tabbed UI via existing component patterns in `ProductionBibleCreator.tsx`

---

## Architecture Patterns

### Recommended Project Structure Changes

```
backend/vidpipe/api/
├── app.py                    # Add: include_router for new route files
├── routes.py                 # UNCHANGED (existing production bible CRUD stays here)
├── sequences.py              # Existing template
├── characters.py             # NEW: Character + Wardrobe + VoiceProfile CRUD
├── sets_props.py             # NEW: Set + SonicIdentity + Prop CRUD
└── sound.py                  # NEW: ScoreTheme + SFXItem CRUD

backend/vidpipe/db/
├── models.py                 # ADD: Character, Wardrobe, VoiceProfile, Set, SonicIdentity, Prop, ScoreTheme, SFXItem
└── __init__.py               # ADD: ALTER TABLE migrations for new tables + score_theme_id on scenes

backend/vidpipe/services/
└── production_bible_entity_service.py   # NEW: business logic for entity migration + prompt-context generation

frontend/src/
├── api/
│   ├── types.ts              # ADD: Character, Set, Prop, ScoreTheme, SFXItem types
│   └── client.ts             # ADD: API functions for new entities
└── components/
    ├── ProductionBibleCreator.tsx   # MODIFY: wire up Casting, Art Dept, Sound tabs
    ├── CharacterDetail.tsx          # NEW: four-tab character editor
    ├── SetDetail.tsx                # NEW: two-tab set editor
    └── SoundDepartment.tsx          # NEW: ScoreTheme + SFX sections
```

### Pattern 1: ORM Models with Sub-Entities (HIGH confidence)

Follow the exact `Mapped[Type]` pattern from `models.py`. All new entities live in `production_bibles` via FK, not in `scenes` or `assets`. Sub-entities use `character_id`, `set_id` FKs.

```python
# Source: backend/vidpipe/db/models.py (existing pattern)
class Character(Base):
    __tablename__ = "characters"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(30))  # PROTAGONIST/ANTAGONIST/SUPPORTING/EXTRA
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    arc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor_refs: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)   # list of image URLs/keys
    base_appearance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Wardrobe(Base):
    __tablename__ = "wardrobes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    character_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("characters.id"), index=True)
    label: Mapped[str] = mapped_column(Text)
    reference_images: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    scene_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_descriptor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    character_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("characters.id"), unique=True, index=True  # 1:1
    )
    voice_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(50), default="ELEVENLABS")
    style_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Set(Base):
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
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class SonicIdentity(Base):
    __tablename__ = "sonic_identities"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    set_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sets.id"), unique=True, index=True  # 1:1
    )
    ambience_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Prop(Base):
    __tablename__ = "props"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    reference_image: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    associated_characters: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list of character UUIDs as strings
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class ScoreTheme(Base):
    __tablename__ = "score_themes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    mood_descriptors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tempo_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    usage_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(50), default="MUSIC_GEN")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class SFXItem(Base):
    __tablename__ = "sfx_items"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))  # IMPACT/MECHANICAL/NATURAL/UI/FOLEY/AMBIENCE
    source_audio: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
```

### Pattern 2: DB Migrations (HIGH confidence)

New tables are created by `create_all()` on fresh DBs. For `score_theme_id` on the existing `scenes` table, add an ALTER TABLE entry to `_run_migrations()` in `db/__init__.py`. This is the established pattern used for all previous column additions.

```python
# Source: backend/vidpipe/db/__init__.py (existing _run_migrations pattern)
# Add inside migrations list:
"ALTER TABLE scenes ADD COLUMN score_theme_id {uuid_type} REFERENCES score_themes(id)",
```

Note: The `{uuid_type}` placeholder is already handled by the migration loop (`sql.format(uuid_type=uuid_type)`), so use TEXT for SQLite and UUID for PostgreSQL automatically.

New tables (`characters`, `wardrobes`, `voice_profiles`, `sets`, `sonic_identities`, `props`, `score_themes`, `sfx_items`) are automatically created by `Base.metadata.create_all()` once ORM models are added to `models.py`. No explicit migration needed for new tables.

### Pattern 3: API Route Files (HIGH confidence)

Follow `sequences.py` exactly:
- `APIRouter(prefix="/api")` — same prefix as main router
- Pydantic `BaseModel` for request/response shapes defined in the same file
- `async with async_session() as session:` for DB access
- Register in `app.py` via `app.include_router(character_router)`

```python
# Source: backend/vidpipe/api/sequences.py (existing pattern)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from vidpipe.db import async_session
from vidpipe.db.models import Character, Wardrobe, VoiceProfile, ProductionBible

character_router = APIRouter(prefix="/api")

@character_router.get("/production-bibles/{production_bible_id}/characters")
async def list_characters(production_bible_id: str): ...

@character_router.post("/production-bibles/{production_bible_id}/characters", status_code=201)
async def create_character(production_bible_id: str, body: CharacterCreate): ...

@character_router.get("/characters/{character_id}")
async def get_character(character_id: str): ...

@character_router.put("/characters/{character_id}")
async def update_character(character_id: str, body: CharacterUpdate): ...

@character_router.delete("/characters/{character_id}")
async def delete_character(character_id: str): ...

@character_router.get("/characters/{character_id}/prompt-context")
async def get_character_prompt_context(character_id: str): ...
```

### Pattern 4: Prompt-Context Endpoints (MEDIUM confidence — designed from requirements)

The prompt-context endpoint returns a structured injection string that downstream pipeline stages (storyboarding, prompt rewriting) can include in their system prompts. Based on requirements PBEX-04 and PBEX-10, these endpoints return a formatted string describing the entity for use in LLM prompts.

```python
# Recommended response shape for /api/characters/{id}/prompt-context
{
  "character_id": "...",
  "injection_string": "CHARACTER [Alice]: PROTAGONIST. Base appearance: tall, red hair, Victorian dress. Wardrobe (default): 'Field Outfit' - worn linen blouse, mud-stained boots. Prompt tags: [victorian, female, protagonist]"
}

# Recommended response shape for /api/sets/{id}/prompt-context
{
  "set_id": "...",
  "injection_string": "SET [Abandoned Warehouse]: Industrial decay, broken windows, shafts of dusty light. Lighting: chiaroscuro, single overhead source. Reverse prompt: <generated_text>. Prompt tags: [industrial, dark, interior]"
}
```

### Pattern 5: LLM Vision Reverse-Prompting for Sets (HIGH confidence)

Reuse `ReversePromptService` with `asset_type="ENVIRONMENT"` for Sets. This triggers on reference image upload, runs after the file is saved, and updates `set.reverse_prompt` before committing. The service returns `{"reverse_prompt": str, "visual_description": str, "quality_score": float, "suggested_name": str}`.

```python
# Source: backend/vidpipe/services/reverse_prompt_service.py (existing)
from vidpipe.services.reverse_prompt_service import ReversePromptService

async def auto_reverse_prompt_set(set_id: uuid.UUID, image_path: str, session: AsyncSession):
    svc = ReversePromptService()  # uses default gemini-2.5-flash
    result = await svc.reverse_prompt_asset(image_path, "ENVIRONMENT", user_name=set.name)
    set_row.reverse_prompt = result["reverse_prompt"]
    await session.commit()
```

This call should be made **after** the file is saved and committed, **within the same upload request handler** (not as a background task), so the client receives the populated reverse_prompt in the response. If the LLM call fails, log and continue — the field stays null (graceful degradation per project conventions).

### Pattern 6: Asset Migration to Character/Set Entities (MEDIUM confidence)

On first load of a ProductionBible that has `assets` of type `CHARACTER` or `ENVIRONMENT`, check if corresponding Character/Set rows exist. If not, create them from the asset data.

```python
# In production_bible_entity_service.py
async def migrate_character_assets(session: AsyncSession, bible_id: uuid.UUID):
    """Migrate CHARACTER-type assets to Character entities. Idempotent."""
    existing_chars = await session.execute(
        select(Character.name).where(Character.production_bible_id == bible_id)
    )
    existing_names = {r[0] for r in existing_chars.all()}

    assets = await session.execute(
        select(Asset).where(
            Asset.production_bible_id == bible_id,
            Asset.asset_type == "CHARACTER",
        )
    )
    for asset in assets.scalars().all():
        if asset.name not in existing_names:
            char = Character(
                production_bible_id=bible_id,
                name=asset.name,
                role="SUPPORTING",  # default
                description=asset.visual_description,
                base_appearance=asset.reverse_prompt,
            )
            session.add(char)
    await session.flush()
```

Call this from the GET `/api/production-bibles/{id}` handler when returning the bible detail, OR from a dedicated POST `/api/production-bibles/{id}/migrate-entities` endpoint. The GET approach is simpler but adds latency; a dedicated endpoint is safer. **Recommendation: dedicated migration endpoint called once on Casting/Art tabs first open.**

### Pattern 7: Frontend Tab Structure (HIGH confidence)

`ProductionBibleCreator.tsx` already has `DEPARTMENT_TABS` array config with `casting`, `art`, and `sound` tabs. The Casting tab currently renders CHARACTER assets only. Phase 17 replaces this with Character entity CRUD. Art Department tab replaces ENVIRONMENT/PROP/OBJECT assets with Set + Prop entity CRUD. Sound tab (currently placeholder) gets ScoreTheme + SFX sections.

The `AssetUploader` and `AssetEditor` components remain for the raw asset processing workflow. New entity-specific components (`CharacterDetail.tsx`, `SetDetail.tsx`, `SoundDepartment.tsx`) are purpose-built for structured entity editing.

### Anti-Patterns to Avoid

- **Avoid reusing Asset rows for Character/Set/Prop** — the Asset model is for the processing pipeline (face detection, CLIP embeddings, reverse prompting). The new entities are structured production data consumed by LLM context injection.
- **Avoid running LLM Vision for Set reverse-prompting in background task** — the response should include the populated field; run inline with a try/except fallback.
- **Avoid putting all new routes in routes.py** — CLAUDE.md requires domain-separated route files.
- **Avoid cascade-delete of assets when migrating** — migration creates Character/Set entities from assets, does NOT delete assets. The pipeline still uses assets for reference image selection.
- **Avoid adding `score_theme_id` FK before `score_themes` table exists** — in `_run_migrations()`, migrations are ordered; the score_themes table is created by `create_all()` which runs before `_run_migrations()`, so this is safe.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Set reverse-prompt generation | Custom Gemini call | `ReversePromptService` | Already handles MIME detection, retry, schema validation |
| File upload for actor_refs/reference_image | Custom multipart | `UploadFile` + existing `save_asset_image()` pattern | Existing infra handles local vs S3 routing |
| UUID generation | `str(uuid.uuid4())` inline | `default=uuid.uuid4` on `mapped_column` | ORM-level default, consistent with all other models |
| JSON column for lists | Separate join table for tags | `Mapped[Optional[list]] = mapped_column(JSON)` | Established pattern for `tags`, `prompt_tags`, `actor_refs` |
| Enum validation for role/category | Python Enum class | String column + validation in route handler | Consistent with `Sequence.act` pattern |

**Key insight:** The project deliberately avoids Python Enum types in ORM models — validation is done at the API layer (route handler raises HTTPException 422), not at the DB layer. Follow this for `Character.role`, `SFXItem.category`, and `ScoreTheme.adapter_type`.

---

## Common Pitfalls

### Pitfall 1: score_theme_id FK References Table Created by create_all()

**What goes wrong:** `_run_migrations()` runs AFTER `create_all()`. The `score_themes` table is created by `create_all()`. The ALTER TABLE that adds `score_theme_id` to `scenes` references `score_themes(id)`. This ordering is correct — but only if the SQLite FK pragma is not in strict mode. SQLite does not enforce FK constraints by default anyway. PostgreSQL will enforce it, but since `score_themes` is created before the ALTER runs, this is safe.

**Why it happens:** Confusion about whether `create_all()` or migrations run first.

**How to avoid:** The ordering in `init_database()` is: rename migrations → `create_all()` → column migrations. New tables appear via `create_all()`. New FK columns on existing tables appear via `_run_migrations()`. This ordering is correct and must be preserved.

### Pitfall 2: 1:1 Sub-Entity UniqueConstraint

**What goes wrong:** `VoiceProfile` and `SonicIdentity` are 1:1 with their parent. Without `unique=True` on the FK, multiple rows could be inserted.

**How to avoid:** Use `unique=True` on the `character_id` / `set_id` FK column in the ORM model. This creates a UNIQUE constraint at the DB level. The route handler should check for existence before creating: if a VoiceProfile already exists for the character, return it (upsert semantics) or 409 Conflict.

```python
# Correct: 1:1 enforced at DB level
class VoiceProfile(Base):
    character_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("characters.id"), unique=True, index=True
    )
```

### Pitfall 3: SQLite TEXT vs PostgreSQL UUID for New FK Columns

**What goes wrong:** New FK columns added via `_run_migrations()` use the `{uuid_type}` placeholder (TEXT for SQLite, UUID for PostgreSQL). But if a developer hard-codes `TEXT` for a new FK referencing `characters(id)`, it breaks on PostgreSQL.

**How to avoid:** Add all new UUID FK columns in `_run_migrations()` using the `{uuid_type}` placeholder:
```python
"ALTER TABLE scenes ADD COLUMN score_theme_id {uuid_type} REFERENCES score_themes(id)",
```
The loop already handles format substitution.

### Pitfall 4: LLM Vision Call Blocking Upload Response

**What goes wrong:** Reverse-prompting for Set reference images can take 2-5 seconds. If done synchronously in the upload handler without a timeout, slow Gemini responses block the client.

**How to avoid:** Wrap the `ReversePromptService.reverse_prompt_asset()` call in a `try/except` with a reasonable timeout (tenacity already handles retries). If it fails or times out, the `reverse_prompt` field stays `null` and the client gets a successful upload response with the image URL. The user can trigger reprocessing manually.

### Pitfall 5: app.py Missing Router Registration

**What goes wrong:** New route files (`characters.py`, `sets_props.py`, `sound.py`) are created but not registered in `app.py` — routes return 404 silently.

**How to avoid:** Every new route file must be imported and registered in `app.py`:
```python
from vidpipe.api.characters import character_router
from vidpipe.api.sets_props import sets_props_router
from vidpipe.api.sound import sound_router

app.include_router(character_router)
app.include_router(sets_props_router)
app.include_router(sound_router)
```

### Pitfall 6: Disabled Audio Generation UI State

**What goes wrong:** VoiceProfile and ScoreTheme have audio generation buttons that require ElevenLabs (#12) and Music (#22) adapters. If these render as active buttons, users click them and get cryptic errors.

**How to avoid:** Render audio generation buttons as `disabled` with `title="ElevenLabs adapter coming soon"` tooltip. Use a feature flag constant `AUDIO_ADAPTERS_ENABLED = false` in the component. This is explicitly stated in the issue constraints.

---

## Code Examples

Verified patterns from existing codebase:

### DB Migration for New FK Column on Existing Table

```python
# Source: backend/vidpipe/db/__init__.py _run_migrations()
# Pattern: add to migrations list, use {uuid_type} for UUID FKs
migrations = [
    # ... existing migrations ...
    # Phase 17: score_theme_id on scenes for Director agent compatibility
    "ALTER TABLE scenes ADD COLUMN score_theme_id {uuid_type} REFERENCES score_themes(id)",
]
```

### Route File Registration in app.py

```python
# Source: backend/vidpipe/api/app.py (existing pattern for sequences.py)
from vidpipe.api.characters import character_router
from vidpipe.api.sets_props import sets_props_router
from vidpipe.api.sound import sound_router

app.include_router(character_router)
app.include_router(sets_props_router)
app.include_router(sound_router)
```

### Inline LLM Vision on Upload (Set reverse-prompt)

```python
# Source: backend/vidpipe/api/routes.py reprocess_asset pattern
# Pattern: run LLM Vision inline, catch all errors gracefully
from vidpipe.services.reverse_prompt_service import ReversePromptService

@sets_props_router.post("/sets/{set_id}/upload-reference")
async def upload_set_reference(set_id: str, file: UploadFile = File(...)):
    # ... save file ...
    async with async_session() as session:
        set_row = await session.get(Set, uuid.UUID(set_id))
        set_row.reference_image = image_key_or_path
        await session.flush()

        # Auto-generate reverse_prompt from uploaded image
        try:
            svc = ReversePromptService()
            result = await svc.reverse_prompt_asset(
                local_image_path, "ENVIRONMENT", user_name=set_row.name
            )
            set_row.reverse_prompt = result["reverse_prompt"]
        except Exception as e:
            logger.warning("Set reverse-prompt failed: %s", e)
            # Continue — reverse_prompt stays None

        await session.commit()
        await session.refresh(set_row)
        return _set_to_response(set_row)
```

### TypeScript Types for New Entities

```typescript
// Source: frontend/src/api/types.ts (existing AssetResponse pattern)

export interface CharacterResponse {
  character_id: string;
  production_bible_id: string;
  name: string;
  role: "PROTAGONIST" | "ANTAGONIST" | "SUPPORTING" | "EXTRA";
  description: string | null;
  arc: string | null;
  actor_refs: string[] | null;
  base_appearance: string | null;
  prompt_tags: string[] | null;
  wardrobe: WardrobeResponse[];
  voice_profile: VoiceProfileResponse | null;
  created_at: string;
  updated_at: string;
}

export interface WardrobeResponse {
  wardrobe_id: string;
  character_id: string;
  label: string;
  reference_images: string[] | null;
  scene_context: string | null;
  prompt_descriptor: string | null;
  is_default: boolean;
  created_at: string;
}

export interface VoiceProfileResponse {
  voice_profile_id: string;
  character_id: string;
  voice_id: string | null;
  adapter_type: string;
  style_notes: string | null;
  sample_audio: string | null;
  created_at: string;
}

export interface SetResponse {
  set_id: string;
  production_bible_id: string;
  name: string;
  reference_image: string | null;
  reverse_prompt: string | null;
  style_tags: string[] | null;
  lighting_notes: string | null;
  prompt_tags: string[] | null;
  sonic_identity: SonicIdentityResponse | null;
  created_at: string;
  updated_at: string;
}

export interface SonicIdentityResponse {
  sonic_identity_id: string;
  set_id: string;
  ambience_description: string | null;
  reference_audio: string | null;
  generation_prompt: string | null;
  created_at: string;
}

export interface PropResponse {
  prop_id: string;
  production_bible_id: string;
  name: string;
  reference_image: string | null;
  description: string | null;
  associated_characters: string[] | null;
  prompt_tags: string[] | null;
  created_at: string;
  updated_at: string;
}

export interface ScoreThemeResponse {
  score_theme_id: string;
  production_bible_id: string;
  name: string;
  mood_descriptors: string[] | null;
  tempo_notes: string | null;
  usage_notes: string | null;
  reference_audio: string | null;
  generation_prompt: string | null;
  adapter_type: string;
  created_at: string;
  updated_at: string;
}

export interface SFXItemResponse {
  sfx_item_id: string;
  production_bible_id: string;
  name: string;
  category: "IMPACT" | "MECHANICAL" | "NATURAL" | "UI" | "FOLEY" | "AMBIENCE";
  source_audio: string | null;
  generation_prompt: string | null;
  tags: string[] | null;
  created_at: string;
  updated_at: string;
}

export interface PromptContextResponse {
  entity_id: string;
  injection_string: string;
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flat Asset rows for all entity types | Structured Character/Set/Prop entities | Phase 17 | Characters have structured tabs, sub-entities; pipeline can inject richer context |
| Asset.asset_type = "CHARACTER" | `Character` ORM table with sub-entities | Phase 17 | Backward compat via migration; assets remain for pipeline use |
| Sound = placeholder tab | ScoreTheme + SFXItem entities | Phase 17 | Structured sound library usable by Director agent |

**Deprecated/outdated:**
- `DEPARTMENT_TABS[2].assetTypes = []` (empty sound tab): Will be replaced by ScoreTheme/SFX entity sections.

---

## Open Questions

1. **How should the GET /api/production-bibles/{id} response change?**
   - What we know: It currently returns `assets: AssetResponse[]`. Phase 17 adds Character, Set, Prop, ScoreTheme, SFXItem entities.
   - What's unclear: Should the bible detail response nest all entity lists inline, or should the frontend make separate calls per entity type?
   - Recommendation: Separate calls per tab (lazy load) is cleaner and avoids large payloads. The Casting tab calls `/characters`, Art Dept calls `/sets` + `/props`, Sound calls `/score-themes` + `/sfx`. The main bible detail endpoint stays unchanged.

2. **Should existing Asset rows for CHARACTER/ENVIRONMENT be deleted after migration?**
   - What we know: The pipeline uses Asset rows for reference image selection (`COMFYUI_VIDEO_MODELS` check, Veo reference_images construction). Deleting assets would break the pipeline.
   - What's unclear: Whether the requirements intend co-existence (assets for pipeline, entities for UI) or full replacement.
   - Recommendation: Co-existence. Migration creates Character/Set entities from Asset data but leaves Asset rows intact. The "migration" is additive, not destructive. Mark migrated assets (e.g., add `is_migrated_to_entity: bool` column, or simply check if a Character with the same name exists).

3. **How should image uploads for actor_refs (multi-image) be handled?**
   - What we know: `actor_refs` on Character is `JSON` (list of image keys/URLs). The existing upload pattern handles single-image per asset.
   - What's unclear: Should actor_refs use the same `tmp/manifests/{bible_id}/uploads/` directory, or a new `tmp/bibles/{bible_id}/characters/{character_id}/actor_refs/` structure?
   - Recommendation: Use `tmp/manifests/{bible_id}/characters/{character_id}/actor_refs/` (extend existing pattern, keep `manifests/` prefix for storage key compatibility with existing S3 bucket layout). Each upload appends a key to the `actor_refs` JSON list.

4. **prompt-context endpoint: eager vs on-demand generation?**
   - What we know: The endpoint is for pipeline injection; it could be computed dynamically from entity fields or stored as a pre-computed column.
   - Recommendation: Compute on-demand in the route handler (no stored column). The injection string is a simple string concatenation of entity fields — no LLM call needed. This keeps the data model simple.

---

## Validation Architecture

> Nyquist validation is NOT enabled in `.planning/config.json` (no `nyquist_validation` key, `workflow` only has `research`, `plan_check`, `verifier`). Skipping this section.

---

## Sources

### Primary (HIGH confidence)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/models.py` — ORM model patterns, Mapped[] annotations, JSON column usage
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/__init__.py` — migration pattern, `_run_migrations()`, `{uuid_type}` placeholder
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/sequences.py` — route file template (APIRouter, Pydantic schemas, async_session)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/app.py` — router registration pattern
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/services/reverse_prompt_service.py` — LLM Vision pattern for Set auto-prompting
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/routes.py` — asset upload, production bible CRUD patterns
- `/home/ubuntu/work/video-pipeline/frontend/src/components/ProductionBibleCreator.tsx` — DEPARTMENT_TABS config, existing tab structure
- `/home/ubuntu/work/video-pipeline/frontend/src/api/types.ts` — TypeScript type patterns
- `/home/ubuntu/work/video-pipeline/.planning/REQUIREMENTS.md` — PBEX-01 through PBEX-20 definitions
- `/home/ubuntu/work/video-pipeline/.planning/STATE.md` — accumulated architectural decisions

### Secondary (MEDIUM confidence)
- Phase 16 STATE.md decisions: `DEPARTMENT_TABS array config in ProductionBibleCreator: Casting=CHARACTER, Art Dept=ENV/PROP/OBJECT/VEHICLE/STYLE, Sound=placeholder`
- Issue constraint from task description: blocked-by graph, sub-entity ownership, audio generation buttons disabled, `Scene.score_theme_id` nullable FK

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all patterns directly verified in codebase
- Architecture: HIGH — ORM, migration, and route file patterns verified from working code
- LLM Vision for sets: HIGH — `ReversePromptService` exists and is tested in production
- Migration of existing assets: MEDIUM — the approach is logical but the exact trigger point (GET vs dedicated endpoint) needs a planning decision
- Frontend component structure: HIGH — `DEPARTMENT_TABS` structure and component import patterns verified in `ProductionBibleCreator.tsx`
- Pitfalls: HIGH — all pitfalls derived from documented codebase decisions in STATE.md

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (stable stack, low churn)
