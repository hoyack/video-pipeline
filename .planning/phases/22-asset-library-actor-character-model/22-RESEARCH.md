# Phase 22: Asset Library & Actor-Character Model - Research

**Researched:** 2026-03-05
**Domain:** Database schema refactoring, REST API design, React UI for asset management
**Confidence:** HIGH

## Summary

This phase transforms the current production-bible-scoped entity model (Character, Set, Prop, ScoreTheme, SFXItem from Phase 17) into a two-layer architecture: standalone library entities (Actor, Set, Prop, SoundAsset) that exist independently, plus binding tables (CastBinding, SetBinding, PropBinding, SoundBinding) that connect library assets to Production Bibles with production-specific overrides.

The existing Phase 17 models are tightly coupled to Production Bibles via `production_bible_id` FK on every entity. The new architecture introduces global entities without this FK, plus binding tables that reference both the library entity and the Production Bible. The existing Character model maps closely to the new Actor model but lacks the "casting" indirection layer. Scene tag resolution (`[CHAR:TAG]`, `[SET:TAG]`, `[PROP:TAG]`) is a new feature that resolves tags to bound assets at generation time.

**Primary recommendation:** Create new standalone Actor/Set/Prop/SoundAsset tables alongside existing Phase 17 tables. Add binding tables. Implement "Promote to Library" migration that converts existing bible-scoped entities into library entities + bindings. Do NOT rename or alter existing tables -- they continue to work for backward compatibility until promotion.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Actor is a standalone entity with: id, name, description, base_appearance_prompt, prompt_tags[], appearance_refs (ActorRef[]), voice_profiles (VoiceProfile[]), wardrobe_presets (WardrobeItem[])
- ActorRef stores image_url, label (front/profile/3-4), is_primary flag
- VoiceProfile stores voice_id, adapter_type (ELEVENLABS|BARK|XTTS|CUSTOM), style_notes, sample_url
- WardrobeItem stores label, description, reference_images[]
- Set: name, description, reverse_prompt, style_tags[], prompt_tags[], reference_images (SetRef[]), lighting_notes, sonic_identity
- Prop: name, description, appearance_prompt, prompt_tags[], reference_images (PropRef[])
- Sound Asset: name, category (SCORE_THEME|SFX|AMBIENCE|FOLEY|UI), subcategory, description, audio_url, generation_prompt, tags[]
- CastBinding: bible_id, actor_id, character_name, character_description, character_arc, role (LEAD|SUPPORTING|EXTRA|NARRATOR), wardrobe_override[], voice_profile_id, behavioral_notes, prompt_tags[]
- SetBinding: bible_id, set_id, production_name, lighting_override, sonic_override, prompt_tags[]
- PropBinding: bible_id, prop_id, production_name, notes, prompt_tags[]
- SoundBinding: bible_id, sound_asset_id, usage_notes, prompt_tags[]
- ProductionBible gains: cast (CastBinding[]), sets (SetBinding[]), props (PropBinding[]), sound (SoundBinding[])
- Tag syntax: [CHAR:TAG], [SET:TAG], [PROP:TAG] in scene prompts
- Tag resolution at generation time
- New top-level Asset Library navigation with sub-sections: Actors, Sets, Props, Sound Assets
- Production Bible view adds Casting, Art Department, Sound sections with +Add picker
- "Promote to Actor Library" action on existing characters/sets/props
- Existing scenes without tags continue to work (additive)

### Claude's Discretion
- Database migration strategy (ALTER TABLE vs new tables with migration service)
- Whether to reuse existing Character/Set/Prop ORM models from Phase 17 or create new Actor-level models alongside them
- API route organization for Asset Library endpoints
- Frontend routing structure for Asset Library views
- How to handle the relationship between Phase 17's Character model and the new Actor + CastBinding model
- Tag autocomplete implementation details (debounce, matching strategy)
- Asset Library search/filter implementation
- Inline asset creation UX in Production Bible picker

### Deferred Ideas (OUT OF SCOPE)
- Audio pipeline integration (dialogue generation, SFX placement, score/ambience, audio mix)
- Asset versioning (whether bound Production Bibles see changes or pin to a version)
- Tag syntax finalization ([CHAR:TAG] vs @TAG vs {{TAG}})
- Multiple Production Bibles per Production
- Prop extraction accuracy (YOLO pipeline vs manual-only)
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy 2.0 | 2.0+ | ORM with Mapped[] annotations | Already used project-wide |
| FastAPI | 0.100+ | REST API framework | Already used project-wide |
| Pydantic | 2.0+ | Request/response validation | Already used project-wide |
| React 19 | 19 | Frontend UI | Already used project-wide |
| wouter | latest | Frontend routing | Already used project-wide |
| Tailwind CSS 4 | 4 | Styling | Already used project-wide |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiosqlite/asyncpg | existing | Async DB drivers | Already configured in engine.py |

No new dependencies needed. This phase is purely domain modeling and CRUD.

## Architecture Patterns

### Recommended Project Structure

#### Backend

```
backend/vidpipe/
├── db/
│   └── models.py           # Add: Actor, ActorRef, LibrarySet, LibraryProp, SoundAsset,
│                            #       CastBinding, SetBinding, PropBinding, SoundBinding
├── api/
│   ├── asset_library.py     # NEW: Actor/Set/Prop/SoundAsset CRUD (library-level)
│   ├── bindings.py          # NEW: CastBinding/SetBinding/PropBinding/SoundBinding CRUD
│   ├── characters.py        # KEEP: existing bible-scoped character routes (backward compat)
│   ├── sets_props.py        # KEEP: existing bible-scoped routes
│   └── sound.py             # KEEP: existing bible-scoped routes
└── services/
    └── tag_resolver.py      # NEW: [CHAR:TAG] → Actor appearance resolution
```

#### Frontend

```
frontend/src/
├── components/
│   ├── AssetLibrary.tsx           # NEW: top-level library view with tabs
│   ├── ActorList.tsx              # NEW: actor list/grid with search
│   ├── ActorDetail.tsx            # NEW: actor detail with tabs
│   ├── LibrarySetList.tsx         # NEW: library set list
│   ├── LibraryPropList.tsx        # NEW: library prop list
│   ├── SoundAssetList.tsx         # NEW: sound asset list
│   ├── AssetPicker.tsx            # NEW: modal picker for binding assets to bibles
│   ├── ProductionBibleCreator.tsx # MODIFY: add binding sections
│   └── CharacterDetail.tsx        # KEEP: for existing bible-scoped editing
└── api/
    ├── client.ts                  # ADD: library entity + binding CRUD functions
    └── types.ts                   # ADD: Actor, LibrarySet, LibraryProp, SoundAsset,
                                   #       CastBinding, SetBinding, PropBinding, SoundBinding types
```

### Pattern 1: New Tables Alongside Existing (Migration Strategy)

**What:** Create new standalone entity tables (`actors`, `library_sets`, `library_props`, `sound_assets`) and binding tables (`cast_bindings`, `set_bindings`, `prop_bindings`, `sound_bindings`). Existing Phase 17 tables (`characters`, `sets`, `props`, `score_themes`, `sfx_items`) remain unchanged.

**Why this approach:**
1. **No ALTER TABLE on existing tables** -- avoids migration complexity on both SQLite and PostgreSQL
2. **Existing Production Bible views continue to work** -- Phase 17 entities are still valid
3. **"Promote to Library" is a data copy** -- copies entity data from bible-scoped table to library table, creates a binding back to the bible
4. **Gradual migration** -- no big-bang cutover required

**Key naming decisions:**
- `actors` table (not `library_actors`) since Actor is inherently standalone
- `library_sets` table (not `sets`) to avoid collision with existing `sets` table
- `library_props` table (not `props`) to avoid collision
- `sound_assets` table (not `sound_items`) to match PRD terminology
- Binding tables: `cast_bindings`, `set_bindings`, `prop_bindings`, `sound_bindings`

**Example ORM model:**
```python
class Actor(Base):
    """Standalone actor entity in the global Asset Library.

    Actors are persistent identities that can be "cast" as Characters
    in multiple Production Bibles via CastBinding.
    """
    __tablename__ = "actors"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    base_appearance_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class ActorRef(Base):
    """Reference image for an Actor (1:N)."""
    __tablename__ = "actor_refs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("actors.id"), index=True
    )
    image_url: Mapped[str] = mapped_column(String(500))
    label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # front, profile, 3-4
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CastBinding(Base):
    """Binds an Actor to a Production Bible as a named Character."""
    __tablename__ = "cast_bindings"
    __table_args__ = (
        UniqueConstraint("production_bible_id", "actor_id", name="uq_cast_bible_actor"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_bible_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_bibles.id"), index=True
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("actors.id"), index=True
    )
    character_name: Mapped[str] = mapped_column(Text)
    character_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    character_arc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String(30))  # LEAD, SUPPORTING, EXTRA, NARRATOR
    wardrobe_override: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    voice_profile_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("voice_profiles.id"), nullable=True
    )
    behavioral_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )
```

### Pattern 2: API Route Organization

**What:** Separate route files for library-level CRUD vs binding CRUD.

```python
# asset_library.py - Global library endpoints
asset_library_router = APIRouter(prefix="/api/asset-library")

# Actors
GET    /api/asset-library/actors              # list with search/filter
POST   /api/asset-library/actors              # create
GET    /api/asset-library/actors/{id}         # get with refs, voice profiles, wardrobe
PUT    /api/asset-library/actors/{id}         # update
DELETE /api/asset-library/actors/{id}         # delete (check bindings first)
POST   /api/asset-library/actors/{id}/refs    # upload reference image

# Sets
GET    /api/asset-library/sets               # list
POST   /api/asset-library/sets               # create
GET    /api/asset-library/sets/{id}          # get
PUT    /api/asset-library/sets/{id}          # update
DELETE /api/asset-library/sets/{id}          # delete

# Props
GET    /api/asset-library/props              # list
POST   /api/asset-library/props              # create
GET    /api/asset-library/props/{id}         # get
PUT    /api/asset-library/props/{id}         # update
DELETE /api/asset-library/props/{id}         # delete

# Sound Assets
GET    /api/asset-library/sounds             # list with category filter
POST   /api/asset-library/sounds             # create
GET    /api/asset-library/sounds/{id}        # get
PUT    /api/asset-library/sounds/{id}        # update
DELETE /api/asset-library/sounds/{id}        # delete

# bindings.py - Production Bible binding endpoints
bindings_router = APIRouter(prefix="/api")

# Cast bindings
GET    /api/production-bibles/{id}/cast      # list cast bindings
POST   /api/production-bibles/{id}/cast      # bind actor as character
PUT    /api/cast-bindings/{id}               # update binding overrides
DELETE /api/cast-bindings/{id}               # unbind

# Set/Prop/Sound bindings follow same pattern
GET    /api/production-bibles/{id}/set-bindings
POST   /api/production-bibles/{id}/set-bindings
# etc.

# Promotion endpoint
POST   /api/characters/{id}/promote-to-library  # promote bible-scoped → library
POST   /api/sets/{id}/promote-to-library
POST   /api/props/{id}/promote-to-library
```

### Pattern 3: Frontend Routing

**What:** Add `/asset-library` top-level route with sub-routes.

```
/asset-library                    → AssetLibrary (tabs: Actors, Sets, Props, Sounds)
/asset-library/actors/:id         → ActorDetail (tabs: Overview, Refs, Voice, Wardrobe, Usage)
/asset-library/sets/:id           → LibrarySetDetail
/asset-library/props/:id          → LibraryPropDetail
/asset-library/sounds/:id         → SoundAssetDetail
```

Add navigation item in Layout.tsx after "Production Bibles" link.

### Pattern 4: Tag Resolution Service

**What:** Service that resolves `[CHAR:TAG]`, `[SET:TAG]`, `[PROP:TAG]` in scene prompts to actual asset data.

```python
# tag_resolver.py
import re
from typing import Optional

TAG_PATTERN = re.compile(r'\[(CHAR|SET|PROP):([A-Z0-9_]+)\]')

async def resolve_tags(prompt: str, production_bible_id: uuid.UUID, session) -> dict:
    """Resolve all tags in a prompt to their bound asset data.

    Returns dict with:
      - resolved_prompt: prompt with tags expanded to appearance text
      - char_refs: list of actor reference images for visual injection
      - set_context: lighting/style notes from bound sets
    """
    matches = TAG_PATTERN.findall(prompt)
    # Look up CastBindings, SetBindings, PropBindings for this bible
    # Substitute or annotate the prompt with resolved data
```

### Anti-Patterns to Avoid

- **Modifying existing Phase 17 tables:** Adding/removing columns on `characters`, `sets`, `props` tables risks breaking existing data and API surface. Create new tables instead.
- **Shared ORM model for both bible-scoped and library entities:** The data models have different semantics (owned by bible vs standalone). Separate ORM classes are clearer.
- **Eager-loading all bindings on Production Bible list:** Only load binding counts for list view; full binding data on detail view.
- **Cascade-deleting library entities when bible is deleted:** Bindings should be deleted, but the library entity persists.
- **Hard-wiring voice_profile_id FK from CastBinding to existing VoiceProfile table:** The existing VoiceProfile has `character_id` FK. The new binding needs its own voice profile reference -- store `voice_profile_id` on CastBinding pointing to the Actor's VoiceProfile, or use a new `actor_voice_profiles` table.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Search/filter | Custom full-text search | SQL LIKE with SQLAlchemy `icontains` | Good enough for asset counts < 10K; no need for Elasticsearch |
| Tag parsing | Manual string splitting | `re.compile` regex pattern | Handles edge cases, already standard in Python |
| Debounced autocomplete | Custom debounce hook | `setTimeout`/`clearTimeout` in useEffect | Standard React pattern, no library needed |

## Common Pitfalls

### Pitfall 1: VoiceProfile FK Collision
**What goes wrong:** The existing `VoiceProfile` table has `character_id` FK pointing to `characters` table (Phase 17 bible-scoped). New Actor-level voice profiles need a different FK.
**Why it happens:** Reusing the same VoiceProfile table for both contexts creates FK confusion.
**How to avoid:** Create `actor_voice_profiles` table with `actor_id` FK, separate from the existing `voice_profiles` table. When promoting a Character to an Actor, copy the VoiceProfile data to the new table.

### Pitfall 2: Table Name Collision (Set, Prop)
**What goes wrong:** Existing `sets` and `props` tables already exist from Phase 17 with `production_bible_id` FK.
**Why it happens:** The PRD uses the same entity names for both bible-scoped and library-scoped entities.
**How to avoid:** Use `library_sets` and `library_props` table names for standalone entities. Or prefix with `asset_` (e.g., `asset_sets`, `asset_props`). The `library_` prefix is clearer.

### Pitfall 3: Wardrobe Override Complexity
**What goes wrong:** CastBinding has `wardrobe_override[]` which is a JSON list of wardrobe overrides. If the override references WardrobeItems from the Actor, changes to the Actor's wardrobe may break overrides.
**Why it happens:** Overrides reference base items by label or ID, but those items can change.
**How to avoid:** Store wardrobe overrides as self-contained descriptions in the JSON (not references to wardrobe IDs). Each override should contain all the data needed without looking up the Actor's wardrobe.

### Pitfall 4: Migration Data Integrity
**What goes wrong:** "Promote to Library" creates a library entity + binding, but the original bible-scoped entity still exists. Users may edit both, causing data drift.
**Why it happens:** No link between the promoted library entity and the original bible-scoped entity.
**How to avoid:** Add `promoted_to_actor_id` (nullable FK) on the `characters` table so the UI can show "Promoted -- edit in Library" instead of inline editing. Same pattern for sets/props.

### Pitfall 5: Binding Deletion Cascade
**What goes wrong:** Deleting a library entity (Actor) while bindings exist in Production Bibles causes FK violations or orphaned bindings.
**Why it happens:** No referential integrity check before deletion.
**How to avoid:** Check for active bindings before deletion. Return 409 Conflict if bindings exist (same pattern as manifest deletion with project references). Provide "Force delete" option that also removes bindings.

### Pitfall 6: Dual-Driver Schema
**What goes wrong:** New tables must work on both SQLite and PostgreSQL.
**Why it happens:** Project uses dual-driver architecture.
**How to avoid:** Follow existing patterns: use `Mapped[uuid.UUID]` for UUIDs (SQLAlchemy handles TEXT vs native UUID), `server_default=text("false")` for booleans, `JSON` for list/dict fields. Add migrations to `_run_migrations()` in `db/__init__.py` with the `{uuid_type}` template pattern for FK columns.

## Code Examples

### Existing Pattern: Entity CRUD Route (from characters.py)

The existing character routes follow a consistent pattern:
1. Pydantic `Create` and `Update` request models
2. Helper `_entity_to_dict()` function for serialization
3. List endpoint with bulk sub-entity fetching (avoids N+1)
4. Create/Update/Delete with explicit sub-entity cleanup
5. Routes organized as `GET /production-bibles/{id}/characters` (scoped list) + `GET /characters/{id}` (direct access)

New library routes should follow the same pattern but with `/api/asset-library/actors` prefix instead of production bible scoping.

### Existing Pattern: Storage Upload (from characters.py)

```python
# Dual-backend storage pattern used in all upload endpoints
storage = get_storage_backend()
if isinstance(storage, LocalStorageBackend):
    local_dir = _settings.storage.tmp_dir / "asset-library" / "actors" / str(actor.id) / "refs"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename
    await asyncio.to_thread(local_path.write_bytes, content)
    stored_path = str(local_path)
else:
    key = f"asset-library/actors/{actor.id}/refs/{filename}"
    await storage.put(key, content, file.content_type or "image/png")
    # Also write local copy for pipeline access
    local_path = _settings.storage.tmp_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(local_path.write_bytes, content)
    stored_path = key
```

### Existing Pattern: Migration in db/__init__.py

```python
# New table creation happens automatically via create_all().
# Only ALTER TABLE migrations needed for columns on EXISTING tables.
# New tables (actors, actor_refs, etc.) are handled by create_all().
# Only need migration for the promoted_to_actor_id column on characters table:
migrations = [
    "ALTER TABLE characters ADD COLUMN promoted_to_actor_id {uuid_type} REFERENCES actors(id)",
    "ALTER TABLE sets ADD COLUMN promoted_to_library_set_id {uuid_type} REFERENCES library_sets(id)",
    "ALTER TABLE props ADD COLUMN promoted_to_library_prop_id {uuid_type} REFERENCES library_props(id)",
]
```

### Tag Resolution Example

```python
async def resolve_tags(prompt: str, bible_id: uuid.UUID, session) -> ResolvedPrompt:
    matches = TAG_PATTERN.findall(prompt)
    resolved = prompt
    char_refs = []

    for tag_type, tag_name in matches:
        if tag_type == "CHAR":
            # Find CastBinding with matching prompt_tag
            binding = await session.execute(
                select(CastBinding).where(
                    CastBinding.production_bible_id == bible_id,
                    CastBinding.prompt_tags.contains([tag_name])  # JSON array contains
                )
            )
            cast = binding.scalars().first()
            if cast:
                actor = await session.get(Actor, cast.actor_id)
                appearance = actor.base_appearance_prompt or ""
                resolved = resolved.replace(
                    f"[CHAR:{tag_name}]",
                    f"{cast.character_name} ({appearance})"
                )
                # Collect actor refs for image generation
                refs = await session.execute(
                    select(ActorRef).where(ActorRef.actor_id == actor.id)
                )
                char_refs.extend(refs.scalars().all())

    return ResolvedPrompt(text=resolved, character_refs=char_refs)
```

## State of the Art

| Old Approach (Phase 17) | New Approach (Phase 22) | Impact |
|--------------------------|-------------------------|--------|
| Character owned by ProductionBible | Actor standalone + CastBinding | Same actor reusable across productions |
| Set owned by ProductionBible | LibrarySet standalone + SetBinding | Same set reusable |
| Direct entity editing in bible | Library editing + production-specific overrides in bindings | Changes propagate unless overridden |
| No asset browsing across bibles | Global Asset Library with search/filter | Faster production setup |
| No tag system | [CHAR:TAG] resolution in prompts | Structured prompt composition |

## Open Questions

1. **VoiceProfile table strategy**
   - What we know: Existing VoiceProfile has `character_id` FK. Actors need their own voice profiles.
   - What's unclear: Should Actor voice profiles share the same table with a nullable `actor_id` column, or use a separate `actor_voice_profiles` table?
   - Recommendation: Separate table (`actor_voice_profiles`) is cleaner. The existing VoiceProfile table stays untouched. When promoting, copy the data.

2. **Wardrobe sub-entity for Actors vs Characters**
   - What we know: Existing Wardrobe has `character_id` FK. Actors also need wardrobe presets.
   - What's unclear: Same table with nullable `actor_id`, or separate `actor_wardrobe_presets` table?
   - Recommendation: Separate table. Keeps Phase 17 untouched. When promoting, copy data.

3. **SoundAsset unification**
   - What we know: PRD defines SoundAsset with category (SCORE_THEME|SFX|AMBIENCE|FOLEY|UI). Phase 17 has separate ScoreTheme and SFXItem tables.
   - What's unclear: Should the library have a single `sound_assets` table or keep separate types?
   - Recommendation: Single `sound_assets` table with `category` enum. Simpler to browse and bind. Promotion from ScoreTheme/SFXItem copies data into sound_assets.

4. **Tag matching strategy**
   - What we know: Tags like `[CHAR:KING_ALDRIC]` need to resolve to a CastBinding.
   - What's unclear: Should tag matching use `prompt_tags` JSON array on bindings, or a dedicated `tag` column?
   - Recommendation: Add a dedicated `tag` column (String, unique per bible+type) on each binding table. This is simpler and faster than JSON array searching. The `prompt_tags` list remains for additional tag injection into prompts.

## Sources

### Primary (HIGH confidence)
- `backend/vidpipe/db/models.py` -- Current ORM models, Phase 17 entity schema
- `backend/vidpipe/api/characters.py` -- Current character CRUD pattern (657 lines)
- `backend/vidpipe/api/sets_props.py` -- Current set/prop CRUD pattern (608 lines)
- `backend/vidpipe/api/sound.py` -- Current sound CRUD pattern (618 lines)
- `backend/vidpipe/db/__init__.py` -- Migration pattern with `_run_migrations()`
- `frontend/src/components/CharacterDetail.tsx` -- Current entity detail UI pattern (1058 lines)
- `frontend/src/api/types.ts` -- Current TypeScript types for entities
- `frontend/src/components/ProductionBibleCreator.tsx` -- Current department tab layout

### Secondary (MEDIUM confidence)
- `.planning/phases/22-asset-library-actor-character-model/22-CONTEXT.md` -- PRD decisions and data model spec

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, uses existing project patterns
- Architecture: HIGH -- clear separation pattern with new tables alongside existing
- Pitfalls: HIGH -- derived from direct codebase analysis of existing FK constraints and dual-driver patterns
- Tag resolution: MEDIUM -- design is sound but implementation details need validation during development

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain, no external dependencies changing)
