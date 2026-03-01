# Phase 19: Bible Context Fix + Code Cleanup - Research

**Researched:** 2026-03-01
**Domain:** Backend schema migration, service layer fix, frontend dead code removal
**Confidence:** HIGH

## Summary

Phase 19 addresses one integration gap (`SCRN-10-bible-context`) and one flow gap (`bible-before-scenes`) identified in the v1.0 milestone audit, plus cleanup of stale code from the Phase 16 manifest-to-Production-Bible rename.

The core bug is in `load_bible_context()` in `backend/vidpipe/services/screenwriter.py`. It looks up a Production Bible by querying `Scene.production_bible_id` for scenes belonging to the production. When a production has no scenes yet (the "Bible -> Screenplay -> Generate Full" flow), the query returns nothing and bible context silently becomes an empty string. The fix per the roadmap is to add a `Production.production_bible_id` FK column, then rewrite `load_bible_context()` to query `Production.production_bible_id` directly. A secondary indirect lookup in `generate_scene_breakdown()` for entity validation also needs the same fix.

The cleanup items are straightforward deletions and string replacements: four orphan Manifest*.tsx files (1,885 lines total, no external imports), user-facing "manifest" strings in ShotCard.tsx and EditForkPanel.tsx, and a dead `sound_router` try/except guard in app.py.

**Primary recommendation:** Add `production_bible_id` nullable FK to the `Production` model, add an ALTER TABLE migration, rewrite `load_bible_context()` and the `generate_scene_breakdown()` entity-validation lookup, then delete dead files and fix strings.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCRN-10 | Production Bible Characters and Sets injected as context into Screenwriter generation prompts | Fix `load_bible_context()` to use `Production.production_bible_id` directly; fix entity validation lookup in `generate_scene_breakdown()`. Both currently use indirect Scene FK lookup that fails when no scenes exist. |
</phase_requirements>

## Standard Stack

No new libraries needed. This phase modifies existing code in the project's established stack.

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy 2.0 | existing | ORM model + migration | Already used throughout |
| FastAPI | existing | API endpoints | Already used throughout |
| React 19 + TypeScript | existing | Frontend components | Already used throughout |

## Architecture Patterns

### Pattern 1: Nullable FK Column on Production Model
**What:** Add `production_bible_id: Mapped[Optional[uuid.UUID]]` to the `Production` model with `ForeignKey("production_bibles.id")`, nullable, indexed.
**When to use:** When the Production needs a direct link to its Production Bible.
**Why:** The `Production` model at line 16 of `models.py` currently has no `production_bible_id`. The only linkage from Production to ProductionBible is through `Scene.production_bible_id`, which requires scenes to exist first.

```python
# backend/vidpipe/db/models.py — Production class (line 16)
class Production(Base):
    __tablename__ = "productions"
    # ... existing columns ...
    production_bible_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("production_bibles.id"), nullable=True, index=True
    )
```

### Pattern 2: Idempotent ALTER TABLE Migration
**What:** Add migration SQL to `_run_migrations()` in `backend/vidpipe/db/__init__.py`.
**When to use:** Any new column on an existing table.
**How:** Follow the existing pattern — SQLite uses raw SQL; PostgreSQL uses `ADD COLUMN IF NOT EXISTS`.

```python
# In _run_migrations(), add to the migrations list:
"ALTER TABLE productions ADD COLUMN production_bible_id {uuid_type} REFERENCES production_bibles(id)",
```

### Pattern 3: Direct FK Lookup in load_bible_context
**What:** Query `Production.production_bible_id` directly instead of going through Scene.
**When to use:** Replace the current indirect Scene-based lookup.

```python
async def load_bible_context(
    session: AsyncSession,
    production_id: uuid.UUID,
) -> str:
    from vidpipe.db.models import Production
    result = await session.execute(
        select(Production.production_bible_id)
        .where(Production.id == production_id)
    )
    bible_id = result.scalar_one_or_none()
    if not bible_id:
        return ""
    try:
        assets = await load_manifest_assets(session, bible_id)
        return format_asset_registry(assets)
    except Exception:
        logger.warning("Failed to load bible context for production %s", production_id)
        return ""
```

### Pattern 4: Entity Validation Bible Lookup
**What:** The `generate_scene_breakdown()` method at line 522-536 of `screenwriter.py` also does the same indirect lookup via `Scene.production_bible_id`. It needs the same fix.

```python
# Replace Scene-based lookup with Production-based lookup:
from vidpipe.db.models import Production as ProductionModel
pb_result = await session.execute(
    select(ProductionModel.production_bible_id)
    .where(ProductionModel.id == production_id)
)
pb_id = pb_result.scalar_one_or_none()
```

### Anti-Patterns to Avoid
- **Removing backward-compat aliases from types.ts prematurely:** The `ManifestListItem = ProductionBibleListItem` type aliases in `types.ts` are still referenced by the orphan files. After deleting the orphan files, check if any aliases are still imported elsewhere. Only remove aliases that become fully unused.
- **Renaming the `manifest_adherence_score` DB column:** The user-facing "Manifest" label in `ShotCard.tsx` at line 373 refers to `manifest_adherence_score`. The label should change to "Bible" or "Adherence" but the DB column/API field name should NOT be renamed in this phase (it would require a DB migration across all existing data).
- **Deleting `fetchManifestAssets` alias from client.ts:** `EditForkPanel.tsx` uses this import. Change the import to `fetchProductionBibleAssets` first, then consider removing the alias if no other files use it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema migration | Manual SQL outside `_run_migrations()` | Add to existing `migrations` list in `_run_migrations()` | The existing migration framework handles both SQLite and PostgreSQL idempotently |
| Bible-to-Production linking | New join table or service | Single nullable FK on Production | Simplest correct solution; matches existing pattern (Scene has same FK) |

## Common Pitfalls

### Pitfall 1: Migration Must Use {uuid_type} Placeholder
**What goes wrong:** Hard-coding `TEXT` in the migration SQL breaks PostgreSQL (needs `UUID` type).
**Why it happens:** SQLite stores UUIDs as TEXT, PostgreSQL uses native UUID.
**How to avoid:** Use `{uuid_type}` placeholder in migration string, like line 193 and 196 in `db/__init__.py`.
**Warning signs:** Tests pass on SQLite but fail on PostgreSQL.

### Pitfall 2: ProductionResponse Schema Needs production_bible_id
**What goes wrong:** Frontend can't display or use the new FK if the API doesn't return it.
**Why it happens:** `ProductionResponse` at line 6029 of `routes.py` doesn't include `production_bible_id`.
**How to avoid:** Add `production_bible_id: Optional[str] = None` to `ProductionResponse` and include it in all `ProductionResponse(...)` constructors (lines 6046, 6068, 6091, 6121).
**Warning signs:** Frontend has no way to know which bible is linked to a production.

### Pitfall 3: Setting production_bible_id on Production When Scenes Are Created
**What goes wrong:** The new `Production.production_bible_id` is never populated for existing workflows.
**Why it happens:** Scenes already have `production_bible_id` set via the generate/draft endpoints, but the Production level never gets updated.
**How to avoid:** When a scene's `production_bible_id` is set (in `POST /api/generate` and `POST /api/projects`), also set the parent production's `production_bible_id` if it's currently null. This ensures backward compatibility.
**Warning signs:** New flow works, but existing productions with scenes still show no bible context.

### Pitfall 4: Dead Import Check After Orphan Deletion
**What goes wrong:** Deleting `ManifestLibrary.tsx` etc. may leave dangling imports.
**Why it happens:** These files are self-contained but you must verify no route or component references them.
**How to avoid:** Search for all imports of these four component names across the codebase. Current state: only `ManifestLibrary.tsx` imports `ManifestCard.tsx`; no external file imports any of the four.
**Warning signs:** Build errors after deletion.

### Pitfall 5: sound_router Import Actually Works
**What goes wrong:** Removing the try/except guard assumes the import always succeeds.
**Why it happens:** The `sound.py` file now exists and the import succeeds. The guard was for Phase 17-03 forward-compatibility during development.
**How to avoid:** Verify `backend/vidpipe/api/sound.py` exists (it does, 15,241 bytes), then change from try/except to direct import. The guard was explicitly noted as dead code in the v1.0 audit.

## Code Examples

### Current load_bible_context (BROKEN for no-scenes case)
```python
# backend/vidpipe/services/screenwriter.py lines 224-260
async def load_bible_context(session, production_id):
    from vidpipe.db.models import Scene
    result = await session.execute(
        select(Scene.production_bible_id)
        .where(Scene.production_id == production_id,
               Scene.production_bible_id.isnot(None))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return ""  # <-- BUG: returns empty when no scenes exist
    assets = await load_manifest_assets(session, row)
    return format_asset_registry(assets)
```

### Files to Delete (orphan Manifest components)
```
frontend/src/components/ManifestLibrary.tsx   (233 lines)
frontend/src/components/ManifestCreator.tsx   (1294 lines)
frontend/src/components/ManifestCard.tsx      (150 lines)
frontend/src/components/ManifestSelector.tsx  (208 lines)
```
Verification: `grep -r "ManifestLibrary\|ManifestCreator\|ManifestCard\|ManifestSelector" frontend/src/` shows only self-references within these four files.

### User-Facing "manifest" Strings to Fix
1. **ShotCard.tsx line 261:** `title={canNavigate ? \`${ref.name} — Click to view manifest\` : ref.name}` -- Change "manifest" to "Production Bible"
2. **ShotCard.tsx line 373:** `<span>Manifest</span>` (label for manifest_adherence_score) -- Change to "Bible" or "Adherence"
3. **EditForkPanel.tsx line 599:** `"No assets in manifest"` -- Change to "No assets in Production Bible"
4. **EditForkPanel.tsx line 3:** `import { forkScene, fetchManifestAssets, ... }` -- Change to `fetchProductionBibleAssets`

### Dead sound_router Guard to Remove
```python
# backend/vidpipe/api/app.py lines 75-80
# Phase 17-03: Sound router (may not exist yet)
try:
    from vidpipe.api.sound import sound_router
    app.include_router(sound_router)
except ImportError:
    pass  # Plan 17-03 not yet executed; sound_router registered on next startup
```
Replace with direct import (no try/except):
```python
from vidpipe.api.sound import sound_router
app.include_router(sound_router)
```

## Affected Files Summary

### Backend Modifications
| File | Change |
|------|--------|
| `backend/vidpipe/db/models.py` | Add `production_bible_id` FK to `Production` class |
| `backend/vidpipe/db/__init__.py` | Add ALTER TABLE migration for `productions.production_bible_id` |
| `backend/vidpipe/services/screenwriter.py` | Rewrite `load_bible_context()` and entity validation lookup in `generate_scene_breakdown()` |
| `backend/vidpipe/api/routes.py` | Add `production_bible_id` to `ProductionResponse`, propagate in scene creation |
| `backend/vidpipe/api/app.py` | Remove `sound_router` try/except guard, use direct import |

### Frontend Modifications
| File | Change |
|------|--------|
| `frontend/src/components/ShotCard.tsx` | Replace "manifest" user-facing strings |
| `frontend/src/components/EditForkPanel.tsx` | Replace "manifest" string, update import alias |

### Frontend Deletions
| File | Reason |
|------|--------|
| `frontend/src/components/ManifestLibrary.tsx` | Orphan from Phase 16 rename |
| `frontend/src/components/ManifestCreator.tsx` | Orphan from Phase 16 rename |
| `frontend/src/components/ManifestCard.tsx` | Orphan from Phase 16 rename |
| `frontend/src/components/ManifestSelector.tsx` | Orphan from Phase 16 rename |

## Open Questions

1. **Should `production_bible_id` propagate from Production to newly created Scenes?**
   - What we know: Currently scenes get `production_bible_id` individually via the generate endpoint's `production_bible_id` field. If the Production has a `production_bible_id`, new scenes created from a screenplay (via `generate_scenes_from_screenplay`) could inherit it automatically.
   - What's unclear: Whether the phase scope includes auto-propagation or just the lookup fix.
   - Recommendation: Auto-propagate `production_bible_id` from Production to new scenes created via `generate_scenes_from_screenplay` endpoint. This makes the "Bible -> Screenplay -> Generate Scenes" flow fully connected. The `generate_scenes_from_screenplay` endpoint at line 254 of `screenplay.py` doesn't currently set `production_bible_id` on new scenes.

2. **How do users link a Production Bible to a Production?**
   - What we know: Currently the link only happens when generating a scene with a `production_bible_id` in the generate request. There's no UI for "assign bible to production" at the Production level.
   - What's unclear: Whether Phase 19 should add an API endpoint to set Production.production_bible_id.
   - Recommendation: Add `production_bible_id` to `ProductionUpdate` request schema and the PUT endpoint. This is minimal and enables the frontend to set it later. Phase 19 doesn't need to add the frontend UI for this (the existing flow through scene creation still works).

## Sources

### Primary (HIGH confidence)
- `backend/vidpipe/services/screenwriter.py` -- contains `load_bible_context()` (lines 224-260) and entity validation lookup (lines 522-536)
- `backend/vidpipe/db/models.py` -- Production model (lines 16-31, no `production_bible_id`), Scene model (lines 433-505, has `production_bible_id`)
- `backend/vidpipe/db/__init__.py` -- migration patterns (lines 120-220)
- `backend/vidpipe/api/screenplay.py` -- all 8 generation endpoints call `load_bible_context()`
- `backend/vidpipe/api/app.py` -- sound_router guard (lines 75-80)
- `.planning/v1.0-MILESTONE-AUDIT.md` -- gap definitions

### Verified Findings
- `ManifestLibrary.tsx`, `ManifestCreator.tsx`, `ManifestCard.tsx`, `ManifestSelector.tsx` have zero external imports (verified via grep)
- `sound.py` exists at 15,241 bytes (verified via ls)
- `ProductionResponse` schema lacks `production_bible_id` (verified at line 6029 of routes.py)
- ShotCard.tsx has 4 manifest references: 2 user-facing strings ("Click to view manifest", "Manifest" score label), 2 data field references (`manifest_tag`, `manifest_adherence_score`)
- EditForkPanel.tsx has 2 manifest references: 1 user-facing string ("No assets in manifest"), 1 import alias (`fetchManifestAssets`)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies, all changes within existing patterns
- Architecture: HIGH - Directly follows existing FK and migration patterns in the codebase
- Pitfalls: HIGH - All findings verified by reading actual source code

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (stable codebase, no external dependency changes)
