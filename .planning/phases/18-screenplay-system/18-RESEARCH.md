# Phase 18: Screenplay System - Research

**Researched:** 2026-02-28
**Domain:** Screenplay data model, LLM generation chain, Scene creation pipeline wiring
**Confidence:** HIGH

## Summary

Phase 18 introduces a Screenplay entity as a structured narrative document attached 1:1 to a Production. A Screenwriter service generates screenplay components incrementally using the existing `LLMAdapter` abstraction (not CrewAI — that decision is locked). The Scene Breakdown within the screenplay then drives automated Scene creation under the Production, enriching Shot generation with Character, Set, and Prop references from the Production Bible.

The codebase is well-prepared: the `Production` model (`productions` table) exists, `Scene` already has `production_id` FK, the `LLMAdapter` ABC with `VertexAIAdapter`/`OllamaAdapter` is fully implemented and used for storyboarding, and the `sequences.py` route file demonstrates the domain-split route pattern required by CLAUDE.md. The primary work is: (1) new `Screenplay` ORM model + migration, (2) `ScreenwriterService` class that calls the LLM sequentially, (3) REST API in a new `screenplay.py` route file, (4) "Generate Scenes from Screenplay" endpoint that creates `Scene` rows, (5) storyboard enrichment when a Scene has a screenplay breakdown reference, and (6) React UI with tabbed editor.

**Primary recommendation:** Follow the CVAnalysisService/sequences.py patterns exactly — a class-based service with sequential async LLM calls, a dedicated route file registered in `app.py`, incremental DB commits after each LLM step for live progress visibility, and Pydantic schemas for every LLM-structured output shape.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCRN-01 | Screenplay entity attached 1:1 to Production with title, genre, status (DRAFT/IN_REVIEW/LOCKED), logline, treatment, character_breakdowns, scene_breakdown, script, shot_list | ORM model pattern from models.py; store large text fields as `Text`, structured sub-data as `JSON` columns |
| SCRN-02 | Scene Breakdown sub-structure per scene: scene_number, slugline, intent, emotional_beat, story_state_in, story_state_out, characters_present (Character refs), set_ref (Set ref), props_required (Prop refs) | JSON column on Screenplay model; Pydantic schema for LLM output + API response |
| SCRN-03 | Screenplay CRUD API under `/api/productions/:id/screenplay` with per-component update endpoints | New `screenplay.py` route file registered in `app.py`; mirrors `sequences.py` pattern |
| SCRN-04 | Screenplay editor UI with tabs: Logline, Treatment, Scene Breakdown, Script, Shot List — each editable with independent Regenerate button | React component with tab state; MarkdownEditorModal reuse viable for long-text fields |
| SCRN-05 | Screenplay status field (DRAFT/IN_REVIEW/LOCKED); LOCKED prevents regeneration | Status enum stored as `String(20)` in ORM; API returns 409 if LOCKED |
| SCRN-06 | Scene Breakdown entries link to Production Bible Characters, Sets, and Props | Store IDs as UUIDs in the JSON breakdown structure; resolve via existing asset/production-bible APIs |
| SCRN-07 | Screenwriter agent with sequential generation chain: logline → treatment → character_breakdowns → scene_breakdown → script (uses existing LLM adapter, not CrewAI) | Use `get_adapter(model_id, user_settings)` from `vidpipe.services.llm`; class-based ScreenwriterService |
| SCRN-08 | Each Screenwriter generation step updates Screenplay entity incrementally (user sees progress) | Commit after each step; emit via `event_bus` (already used in storyboard.py) |
| SCRN-09 | Each Screenwriter step can be run independently (regenerate only Script without changing Breakdown) | Each component has its own endpoint: `POST /api/productions/:id/screenplay/generate-logline`, etc. |
| SCRN-10 | Production Bible Characters and Sets injected as context into Screenwriter generation prompts | Load via `load_manifest_assets(session, production_bible_id)` + `format_asset_registry()` — same pattern as storyboard.py |
| SCRN-11 | LLM adapter selectable per Production for Screenwriter agent | Store `text_model` on Screenplay (or inherit from Production); pass to `get_adapter()` |
| SCRN-12 | "Generate Scenes from Screenplay" action creates one Scene per SceneBreakdown entry from a locked Screenplay | New endpoint `POST /api/productions/:id/screenplay/generate-scenes`; creates Scene rows with production_id set |
| SCRN-13 | Scene description populated from SceneBreakdown.intent; Shot prompts include Character, Set, Prop prompt_tags from linked breakdown | Enrich `scene.prompt` and inject breakdown context into storyboard system prompt |
| SCRN-14 | Free-form storyboard generation remains as fallback when no Screenplay exists | No change to existing storyboard.py path; just don't add screenplay context when scene.screenplay_breakdown_id is None |
| SCRN-15 | Scenes generated from Screenplay show "Screenplay linked" badge in UI | Add `screenplay_id` field to `SceneListItem` and `SceneDetail` API response schemas |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy 2.0 async | already installed | `Screenplay` ORM model with `Mapped[Type]` annotations | Project-wide pattern; all models use this |
| Pydantic v2 | already installed | LLM structured output schemas + API request/response models | Used throughout; `generate_text(prompt, schema=MySchema)` |
| FastAPI + APIRouter | already installed | New `/api/productions/:id/screenplay` route file | `sequences.py` demonstrates the split pattern |
| `vidpipe.services.llm` | internal | LLM adapter for all generation (VertexAI / Ollama) | `get_adapter(model_id, user_settings)` call pattern |
| React 19 + TypeScript | already installed | Screenplay editor component with tabbed UI | All frontend is React + strict TS |
| Tailwind CSS 4 | already installed | Styling; use existing design tokens from ProductionDetail, SequencedSceneList | Consistency with existing UI |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tenacity | already installed | Retry on LLM failures in ScreenwriterService | Each generation step should retry like storyboard.py |
| `event_bus` | internal (`vidpipe.services.event_bus`) | Emit progress events per step so frontend can poll | Use `event_bus.emit(production_id, "screenplay_step", step=...)` |
| `MarkdownEditorModal.tsx` | existing component | Reuse for long-text screenplay sections (Treatment, Script) | Already handles multi-line rich text editing |

### No New Dependencies Required

All required capabilities already exist in the project. No `pip install` or `npm install` needed.

**Installation:** None — build on existing stack.

## Architecture Patterns

### Recommended Project Structure
```
backend/vidpipe/
├── api/
│   ├── screenplay.py          # NEW: screenplay CRUD + generation endpoints
│   └── app.py                 # MODIFIED: register screenplay_router
├── db/
│   ├── models.py              # MODIFIED: add Screenplay model
│   └── __init__.py            # MODIFIED: add migration for screenplay columns
├── services/
│   └── screenwriter.py        # NEW: ScreenwriterService class
└── schemas/
    └── screenplay.py          # NEW: LLM output schemas (LoglineOutput, TreatmentOutput, etc.)

frontend/src/
├── components/
│   └── ScreenplayEditor.tsx   # NEW: tabbed screenplay editor
├── api/
│   ├── client.ts              # MODIFIED: add screenplay API functions
│   └── types.ts               # MODIFIED: add screenplay TypeScript types
└── App.tsx                    # MODIFIED: route to screenplay editor from ProductionDetail
```

### Pattern 1: ORM Model — Screenplay with JSON sub-structures

**What:** The Screenplay entity is a single table row per Production. All sub-structures (character_breakdowns, scene_breakdown, script, shot_list) are stored as `JSON` columns. This avoids a complex normalized schema for structured narrative content that the LLM generates holistically.

**When to use:** When data is generated/consumed as a unit and cross-entity joins are not needed.

**Example:**
```python
# Source: models.py pattern (Phase 16 Production, Sequence)
class Screenplay(Base):
    __tablename__ = "screenplays"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("productions.id"), unique=True, index=True  # 1:1 via UniqueConstraint
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
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )
```

**The 1:1 constraint:** Use `unique=True` on `production_id` FK column, not a separate `UniqueConstraint`. SQLAlchemy creates the unique index automatically. Upsert pattern: `GET` or create on first access.

### Pattern 2: ScreenwriterService — Sequential LLM Chain

**What:** A class-based service (like `CVAnalysisService`) that calls the LLM adapter for each generation step in sequence. Each step fetches the current screenplay, calls `adapter.generate_text(prompt, schema=StepSchema)`, updates the relevant field, and commits.

**When to use:** Multi-step LLM chain where each output feeds the next prompt.

**Example:**
```python
# Source: CVAnalysisService pattern + storyboard.py LLM call pattern
class ScreenwriterService:
    def __init__(self, adapter: LLMAdapter):
        self._adapter = adapter

    async def generate_logline(
        self, session: AsyncSession, screenplay: Screenplay,
        bible_context: str = "",
    ) -> None:
        """Generate logline from Production description + bible context."""
        if screenplay.status == "LOCKED":
            raise ValueError("Screenplay is LOCKED — regeneration blocked")

        prompt = _build_logline_prompt(screenplay, bible_context)
        result = await self._adapter.generate_text(
            prompt=prompt,
            schema=LoglineOutput,
            temperature=0.8,
            max_retries=3,
        )
        screenplay.logline = result.logline
        screenplay.updated_at = datetime.utcnow()
        await session.commit()
        # Emit progress event for frontend polling
        event_bus.emit(str(screenplay.production_id), "screenplay_step",
                       step="logline", status="complete")

    async def generate_treatment(self, session, screenplay, bible_context=""):
        """Generate treatment from logline (requires logline to exist)."""
        ...

    async def generate_scene_breakdown(self, session, screenplay, bible_context=""):
        """Generate scene breakdown — most structured step; uses scene breakdown schema."""
        ...

    async def generate_full(self, session, screenplay, bible_context=""):
        """Run full chain: logline → treatment → character_breakdowns → scene_breakdown → script."""
        for step in [self.generate_logline, self.generate_treatment,
                     self.generate_character_breakdowns, self.generate_scene_breakdown,
                     self.generate_script]:
            await step(session, screenplay, bible_context)
```

### Pattern 3: API Route File — `screenplay.py`

**What:** A dedicated `APIRouter` registered in `app.py`, following the `sequences.py` domain-split convention from CLAUDE.md. All screenplay endpoints live under `/api/productions/{production_id}/screenplay`.

**Key endpoints:**
```
GET    /api/productions/{production_id}/screenplay           → get or create screenplay
PUT    /api/productions/{production_id}/screenplay           → update fields directly
POST   /api/productions/{production_id}/screenplay/generate         → full chain
POST   /api/productions/{production_id}/screenplay/generate-logline → step
POST   /api/productions/{production_id}/screenplay/generate-treatment
POST   /api/productions/{production_id}/screenplay/generate-character-breakdowns
POST   /api/productions/{production_id}/screenplay/generate-scene-breakdown
POST   /api/productions/{production_id}/screenplay/generate-script
POST   /api/productions/{production_id}/screenplay/generate-scenes  → creates Scenes from locked Screenplay
PATCH  /api/productions/{production_id}/screenplay/status    → lock/unlock
```

**Registration in app.py:**
```python
from vidpipe.api.screenplay import screenplay_router
app.include_router(screenplay_router)
```

### Pattern 4: Scene Creation from Screenplay Breakdown (SCRN-12/13)

**What:** `POST .../screenplay/generate-scenes` iterates over `screenplay.scene_breakdown` entries and creates one `Scene` per entry with `production_id` set. The `scene.prompt` is derived from `breakdown.intent`. A `screenplay_breakdown_index` column on Scene (or stored in `storyboard_raw`) references back to the breakdown entry for enrichment during storyboard generation.

**Enrichment hook in storyboard.py:**
```python
# In generate_storyboard(), after determining use_manifests:
# Check if this Scene was created from a screenplay breakdown
screenplay_context = ""
if scene.screenplay_breakdown_index is not None:
    # Load the screenplay and inject the breakdown as structured context
    screenplay = await session.get(Screenplay, ...)  # via production_id
    if screenplay and screenplay.scene_breakdown:
        breakdown = screenplay.scene_breakdown[scene.screenplay_breakdown_index]
        screenplay_context = _format_breakdown_context(breakdown)

full_prompt = f"{system_prompt}{screenplay_context}\n\nScript: {scene.prompt}"
```

**Alternatively (cleaner):** Store the relevant breakdown as a JSON field `screenplay_context` on the Scene model itself when creating scenes from screenplay. This avoids a DB join during storyboard generation.

### Pattern 5: Scene Breakdown Pydantic Schema for LLM Output

```python
# Source: storyboard_enhanced.py pattern
class SceneBreakdownEntry(BaseModel):
    scene_number: int
    slugline: str  # e.g. "INT. OFFICE - DAY"
    intent: str    # narrative intent; becomes scene.prompt
    emotional_beat: str
    story_state_in: str
    story_state_out: str
    characters_present: list[str]  # asset IDs or names from Production Bible
    set_ref: Optional[str] = None  # set asset ID or name
    props_required: list[str] = []

class SceneBreakdownOutput(BaseModel):
    scene_count: int
    scenes: list[SceneBreakdownEntry]
```

### Anti-Patterns to Avoid

- **Separate table for SceneBreakdownEntry:** Adds schema complexity without benefit. JSON column on Screenplay is the correct approach (see `storyboard_raw` JSON column on Scene, `manifest_json` on ShotManifest).
- **CrewAI orchestration:** Explicitly out-of-scope (REQUIREMENTS.md "Out of Scope" + Phase 1 decision). Use sequential `async def` calls in ScreenwriterService.
- **Blocking LLM calls without await:** All ScreenwriterService methods must be `async def` and use `await adapter.generate_text(...)`. Never call synchronously.
- **Committing after all steps:** Each step must commit independently so the user can see incremental progress via polling.
- **Storing screenplay_id on Scene (not breakdown index):** The Scene needs to reference WHICH breakdown entry it came from (an index into `scene_breakdown` JSON array), not just the screenplay. Use `screenplay_breakdown_index: int` column or denormalize the context.
- **Skipping the LOCKED status check in API and service:** Both the route handler AND the service method should gate on `screenplay.status == "LOCKED"`. Defense in depth.
- **Hardcoding model in ScreenwriterService:** Accept adapter from caller, constructed via `get_adapter(screenplay.text_model or settings.models.storyboard_llm, user_settings)`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM retry logic | Custom retry loop | `adapter.generate_text(..., max_retries=3)` which uses tenacity internally | Already handles exponential backoff, reraise |
| Asset registry context formatting | Custom formatter | `load_manifest_assets()` + `format_asset_registry()` from `manifest_service.py` | Phase 7 battle-tested pattern used in storyboard.py |
| Progress emission | Custom WebSocket | `event_bus.emit(production_id, "screenplay_step", step=...)` | event_bus already used for scene progress in storyboard.py |
| DB session management | Raw connections | `async with async_session() as session` context manager | Project-wide pattern from routes.py and sequences.py |
| Schema migrations | Recreating tables | `ALTER TABLE screenplays ...` in `_run_migrations()` in `db/__init__.py` | Project mandate: never recreate DB |
| Model routing | Manual if/else | `get_adapter(model_id, user_settings)` from `vidpipe.services.llm` | Handles Vertex AI, Ollama, future providers |

**Key insight:** The ScreenwriterService is structurally identical to the storyboard generation path — load context, build prompt, call adapter, validate output schema, persist, commit, emit. Reuse every mechanism already present.

## Common Pitfalls

### Pitfall 1: Forgetting the UniqueConstraint on production_id

**What goes wrong:** Two screenplays created for the same production. GET endpoints become ambiguous. UI shows stale screenplay after regeneration.

**Why it happens:** `unique=True` on the ORM column creates a DB-level unique index, but if migrations only `ADD COLUMN` without `ADD CONSTRAINT`, the uniqueness isn't enforced on existing DBs.

**How to avoid:** Add `unique=True` in the ORM `mapped_column()` definition AND include a migration step `CREATE UNIQUE INDEX IF NOT EXISTS uq_screenplays_production_id ON screenplays(production_id)` in `_run_migrations()`.

**Warning signs:** Duplicate rows appearing for the same production after repeated `POST` calls to generate.

### Pitfall 2: Scene `screenplay_breakdown_index` Not Persisted Before Storyboard Runs

**What goes wrong:** Storyboard runs without screenplay context because the enrichment reference was lost. Scenes look like regular scenes.

**Why it happens:** The "generate scenes from screenplay" endpoint creates Scenes but forgets to store which breakdown index maps to which scene.

**How to avoid:** Either (a) add `screenplay_breakdown_index: Mapped[Optional[int]]` to the `Scene` model and set it during scene creation, or (b) denormalize: store the breakdown JSON blob directly in a `screenplay_context: Mapped[Optional[dict]]` JSON column on Scene. Option (b) avoids a cross-table join in `storyboard.py`.

**Warning signs:** Scenes created from screenplay have no narrative context in their storyboard shots.

### Pitfall 3: LLM Context Window Overflow on Long Scripts

**What goes wrong:** The `generate_script` step fails or produces truncated output because the context (logline + treatment + character_breakdowns + scene_breakdown + bible context) exceeds the model's context window.

**Why it happens:** Gemini flash models have large context windows but Production Bible context + prior screenplay content can accumulate to 50K+ tokens for multi-scene productions.

**How to avoid:** For script generation, pass a summarized form of previous steps (not the full treatment verbatim). Use temperature=0.7 for creative steps. Test with 10+ scene breakdowns.

**Warning signs:** API errors about token limits, or truncated `script` field in output.

### Pitfall 4: LOCKED Status Bypass via Direct Field Update

**What goes wrong:** `PUT /api/productions/:id/screenplay` endpoint allows updating any field, bypassing the LOCKED status check that the generation endpoints enforce.

**Why it happens:** Direct update endpoints are permissive by default.

**How to avoid:** In the `PUT` handler, check `if screenplay.status == "LOCKED"` before applying changes. Allow status transitions FROM locked (unlock) but block content field updates.

### Pitfall 5: Migration Order — Screenplay Table Before Scenes Column

**What goes wrong:** `ALTER TABLE scenes ADD COLUMN screenplay_breakdown_index` references a Screenplay concept but if it runs before the `screenplays` table is created, it may fail.

**Why it happens:** `_run_migrations()` runs AFTER `create_all()` in `init_database()`, so new tables are fine — but `ALTER TABLE` on `scenes` for the new column can fail on PostgreSQL if `screenplays` table is expected as FK target and hasn't been created yet.

**How to avoid:** The `screenplay_breakdown_index` column on Scene should be a plain `INTEGER` (or `JSON` blob), NOT a FK to `screenplays`. This avoids the FK ordering dependency entirely.

### Pitfall 6: `generate-scenes` Endpoint Called Multiple Times Creates Duplicates

**What goes wrong:** Clicking "Generate Scenes" twice creates duplicate Scene rows for the same breakdown entries.

**Why it happens:** No idempotency guard.

**How to avoid:** Before creating scenes, check if any Scene under this Production has `screenplay_breakdown_index` already set (or `screenplay_id` if using that pattern). If so, return existing scenes or require explicit `force=true` parameter to regenerate.

## Code Examples

Verified patterns from existing codebase (HIGH confidence):

### LLM Adapter Usage (from storyboard.py)
```python
# Source: backend/vidpipe/pipeline/storyboard.py lines 306-308, 430-436
from vidpipe.services.llm import get_adapter, LLMAdapter

model_id = scene.text_model or settings.models.storyboard_llm
adapter = get_adapter(model_id, user_settings)

storyboard = await adapter.generate_text(
    prompt=full_prompt,
    schema=EnhancedStoryboardOutput,  # Pydantic model class
    temperature=0.7,
    max_retries=1,
)
```

### Service Class Pattern (from cv_analysis_service.py)
```python
# Source: backend/vidpipe/services/cv_analysis_service.py lines 77-88
class CVAnalysisService:
    def __init__(self, vision_adapter: Optional[LLMAdapter] = None):
        self._vision_adapter = vision_adapter
        # lazy-load child services

    async def analyze_generated_content(self, ...) -> CVAnalysisResult:
        ...  # calls adapter, commits, returns result
```

### Migration Pattern (from db/__init__.py)
```python
# Source: backend/vidpipe/db/__init__.py _run_migrations()
# For new table: create_all() handles it via ORM definition
# For new columns on existing tables:
migrations = [
    "ALTER TABLE scenes ADD COLUMN screenplay_breakdown_index INTEGER",
    "ALTER TABLE scenes ADD COLUMN screenplay_context TEXT",  # JSON as TEXT in SQLite
]
```

### Route File Pattern (from sequences.py)
```python
# Source: backend/vidpipe/api/sequences.py lines 23, and app.py lines 16-17
screenplay_router = APIRouter(prefix="/api")

@screenplay_router.get("/productions/{production_id}/screenplay")
async def get_screenplay(production_id: uuid.UUID): ...

# In app.py:
from vidpipe.api.screenplay import screenplay_router
app.include_router(screenplay_router)
```

### Asset Registry Context for LLM (from storyboard.py)
```python
# Source: backend/vidpipe/pipeline/storyboard.py lines 342-346
from vidpipe.services.manifest_service import load_manifest_assets, format_asset_registry

assets = await load_manifest_assets(session, production_bible_id)
asset_registry_block = format_asset_registry(assets)
# Pass asset_registry_block into system prompt
```

### Event Bus Emission (from storyboard.py)
```python
# Source: backend/vidpipe/pipeline/storyboard.py lines 407-408, 583-585
from vidpipe.services.event_bus import event_bus
event_bus.emit(scene.id, "phase_started", phase="storyboard")
event_bus.emit(scene.id, "phase_completed", phase="storyboard")
event_bus.emit(scene.id, "refresh")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CrewAI agent orchestration | Sequential async LLM calls in service class | Phase 1 decision | Simpler, no external dependency, same LLM adapter interface |
| Manifest-unaware storyboard | Bible-context-injected storyboard | Phase 7 | Pattern to follow for screenplay context injection |
| Free-form scene prompts | Screenplay-derived scene intents | Phase 18 (this phase) | Narrative structure drives Shot generation |

**Deprecated/outdated:**
- `Manifest` alias: keep using `ProductionBible` in all new code. `Manifest` alias exists only for backward compat.
- `manifest_id`/`manifest_version` column names: these were renamed in Phase 16 to `production_bible_id`/`production_bible_version`. Use the new names in all new code.

## Open Questions

1. **Should `screenplay_id` appear in `SceneListItem` (for SCRN-15 badge)?**
   - What we know: `SceneListItem` in routes.py and types.ts has `production_id`, `sequence_id` — same pattern.
   - What's unclear: Whether to store `screenplay_id` directly on Scene, or derive it via a join (Scene → Production → Screenplay).
   - Recommendation: Store `screenplay_breakdown_index` (int) on Scene for breakdown context, but don't add `screenplay_id` FK — instead, the API can derive `screenplay_linked: bool` by checking if `scene.screenplay_breakdown_index is not None`. This avoids a new FK while supporting the badge.

2. **Text model for Screenwriter — inherit from Production or separate Screenplay field?**
   - What we know: `Scene` stores `text_model` per-scene. `Screenplay` is per-Production.
   - What's unclear: Whether user wants different models for screenplay vs video generation.
   - Recommendation: Add `text_model` column to `Screenplay` (nullable, falls back to `settings.models.storyboard_llm`). This matches the Scene pattern and satisfies SCRN-11.

3. **Should Shot List be LLM-generated or auto-derived from Scene Breakdown?**
   - What we know: SCRN-01 lists `shot_list` as a Screenplay field. SCRN-02 describes SceneBreakdown but not shot-level detail within screenplay.
   - What's unclear: Whether the `shot_list` in the Screenplay is a separate LLM step (like a per-shot director's plan) or auto-populated from Scene breakdown outputs.
   - Recommendation: Treat `shot_list` as a JSON blob populated during the `generate-scene-breakdown` step (the LLM outputs both scene breakdown and a coarser shot list). This avoids a separate LLM step while satisfying the schema requirement.

4. **Production Bible context — does Screenwriter need both Characters and Sets?**
   - What we know: SCRN-10 says Characters and Sets. SCRN-02 says `characters_present`, `set_ref`, `props_required` reference Production Bible entities.
   - What's unclear: How to resolve "name" references from LLM output back to Asset IDs.
   - Recommendation: Pass the full asset registry block (same `format_asset_registry()` format used in storyboard.py) into Screenwriter prompts. Have the LLM output asset `manifest_tag` identifiers (e.g., `CHAR_01`) rather than names. Store these tags in `characters_present`, `set_ref`, `props_required` within the `scene_breakdown` JSON.

## Validation Architecture

> Nyquist validation is not enabled in `.planning/config.json` (no `workflow.nyquist_validation` field). Skipping this section.

## Sources

### Primary (HIGH confidence)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/models.py` — full ORM model inventory, Production/Scene/Shot hierarchy, column types, relationship patterns
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/routes.py` — existing Production endpoints, API schema patterns, request/response models
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/sequences.py` — domain-split route file pattern (the template for screenplay.py)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/app.py` — router registration pattern
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/pipeline/storyboard.py` — LLM call pattern, asset registry injection, event_bus usage, gap-filling logic
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/services/llm/base.py` — LLMAdapter ABC interface
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/services/llm/registry.py` — `get_adapter()` routing function
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/services/llm/vertex_adapter.py` — VertexAIAdapter implementation showing generate_text pattern
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/services/cv_analysis_service.py` — class-based service pattern with lazy-loaded children
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/__init__.py` — migration pattern (_run_migrations, _run_rename_migrations, init_database order)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/schemas/storyboard.py` — Pydantic schema pattern for LLM structured output
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/schemas/storyboard_enhanced.py` — enhanced schema pattern with sub-structures
- `/home/ubuntu/work/video-pipeline/frontend/src/api/client.ts` — API client function pattern
- `/home/ubuntu/work/video-pipeline/frontend/src/api/types.ts` — TypeScript type patterns for API responses
- `/home/ubuntu/work/video-pipeline/frontend/src/components/ProductionDetail.tsx` — ProductionDetail component pattern (entry point for Screenplay tab)
- `/home/ubuntu/work/video-pipeline/.planning/REQUIREMENTS.md` — SCRN-01 through SCRN-15 requirement definitions
- `/home/ubuntu/work/video-pipeline/.planning/STATE.md` — decision history, phase conventions

### Secondary (MEDIUM confidence)
- Phase instruction block (additional_context) — confirmed CrewAI exclusion, terminology hierarchy, pipeline wiring intent

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed, patterns verified from code
- Architecture: HIGH — patterns derived directly from existing codebase (sequences.py, storyboard.py, CVAnalysisService)
- Pitfalls: HIGH — derived from Phase 16 migration patterns, Phase 7 asset registry patterns, and existing unique constraint patterns in the ORM

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (stable internal codebase, no external library changes expected)
