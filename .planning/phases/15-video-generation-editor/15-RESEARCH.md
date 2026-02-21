# Phase 15: Video Generation Editor - Research

**Researched:** 2026-02-21
**Domain:** Full-stack unified editor (FastAPI backend + React/TypeScript frontend)
**Confidence:** HIGH

## Summary

Phase 15 replaces the current two-screen flow (GenerateForm -> ProgressView) with a unified VideoGenEditor that merges project creation, live generation monitoring, and editing into one view. The codebase is mature (13 phases complete) with well-established patterns for both backend (FastAPI + SQLAlchemy async) and frontend (React + TypeScript + Tailwind). Most infrastructure this phase needs already exists in pieces -- the challenge is composing existing components into a new lifecycle while adding the "draft" project concept and gap-filling pipeline logic.

The research reveals three significant findings: (1) many of the required backend endpoints already exist (keyframe upload, clip upload, scene text editing, stop/resume), reducing the backend scope primarily to the new `POST /api/projects` draft creation endpoint and the gap-filling `POST /api/projects/{id}/generate` endpoint; (2) the Scene model uses non-nullable text fields (`Mapped[str]`, not `Optional[str]`), requiring empty-string defaults for draft scenes rather than NULL; (3) the existing `EditModeOverlay` + `SceneEditorCard` components contain 90%+ of the editing UI needed for the VideoGenEditor, meaning the frontend work is primarily composition and lifecycle management rather than building new UI primitives.

**Primary recommendation:** Structure implementation as backend-first (draft status + gap-filling endpoint), then frontend VideoGenEditor component, then pipeline modifications, then navigation integration -- matching the design doc's phased approach but compressed into a single phase.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VGED-01 | `POST /api/projects` creates draft project with empty Scene rows without starting the pipeline | New endpoint needed. Scene model accepts empty strings for text fields (non-nullable `Mapped[str]`). Project model needs `prompt` relaxed to accept empty string for drafts. Add `"draft"` to status enum. |
| VGED-02 | `POST /api/projects/{id}/generate` inspects existing assets and only generates what's missing (gap-filling mode) | New endpoint needed. Builds on existing `run_pipeline` orchestrator + `_check_completed_steps`. Gap-filling requires per-scene checks in each pipeline stage (storyboard, keyframes, video_gen, stitcher). |
| VGED-03 | `"draft"` added to project status enum; draft projects appear in list with Draft badge | Add to `PIPELINE_STATES` in `state.py`, `RESUMABLE_STATES`, `StatusBadge` frontend, `ProjectList` filter chips. `TERMINAL_STATUSES` should NOT include "draft". |
| VGED-04 | Scene-level upload endpoints: start-keyframe, end-keyframe, clip, final-video | Start-keyframe, end-keyframe, and clip upload endpoints ALREADY EXIST at `PUT /api/projects/{id}/scenes/{idx}/keyframes/{position}` and `PUT /api/projects/{id}/scenes/{idx}/clip`. Only `PUT /api/projects/{id}/final-video` is new. |
| VGED-05 | `generation_status` column on Scene model tracks per-scene pipeline progress | New nullable column `generation_status: Mapped[Optional[str]]` on Scene. Values: None, "generating_text", "generating_start_kf", "generating_end_kf", "generating_clip", "complete", "failed". Pipeline stages set/clear this before/after each operation. |
| VGED-06 | Pipeline stages skip scenes/assets that already exist (gap-filling) | Modify `storyboard.py` (skip scenes with non-empty description), `keyframes.py` (skip scenes with existing Keyframe rows), `video_gen.py` (skip scenes with existing VideoClip), `stitcher.py` (skip if output exists). Pass existing scenes as context to LLM for narrative continuity. |
| VGED-07 | `VideoGenEditor` component replaces GenerateForm + ProgressView with unified experience | New ~600-line component. Reuses `SceneEditorCard` (1062 lines, already handles full lifecycle), model selection patterns from `GenerateForm`, polling from `useProjectStatus`/`usePolling`. Three zones: config bar, generation controls, scene grid. |
| VGED-08 | Generate Through slider controls pipeline stop point | Replace 4-button group with `<input type="range" min={0} max={4} step={1}>`. Maps to run_through values: 0=storyboard, 1=keyframes, 2=video, 3=stitch, 4=null(all). Existing `run_through` field on Project model supports this. |
| VGED-09 | Scene cards render through full lifecycle: empty -> partial -> generating -> complete | `SceneEditorCard` already handles empty slots (dashed border), populated scenes, and generating state (`isGeneratingAssets` prop). Add `generation_status` rendering from VGED-05 for per-asset spinners. |
| VGED-10 | Real-time asset population via polling with visual feedback | Existing `usePolling` hook + `getProjectDetail` polling pattern from `ProgressView` (3s interval). Add diff detection via `useRef` storing previous scene state. CSS transition: `ring-2 ring-green-400` fading via `setTimeout`. |
| VGED-11 | Pause/resume at per-scene granularity | Existing `POST /api/projects/{id}/stop` sets status to "stopped". Enhance pipeline stages to check stop flag inside per-scene loops (currently only checked between stages). `POST /api/projects/{id}/generate` acts as resume for gap-filling. |
| VGED-12 | App.tsx navigation merges "generate" + "progress" views into single "editor" view | Add "editor" to `View` type union in `Layout.tsx`. Replace `GenerateForm` rendering with `VideoGenEditor`. Update `handleGenerated` to navigate to "editor" instead of "progress". Keep "progress" temporarily for backward compat. |
</phase_requirements>

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 18.x | UI framework | Already used throughout frontend |
| TypeScript | 5.x | Type safety | Already used throughout frontend |
| Tailwind CSS | 3.x | Styling | Already used throughout frontend |
| FastAPI | 0.100+ | HTTP API | Already used for all backend routes |
| SQLAlchemy | 2.0 | ORM with async | Already used for all models |
| Pydantic | 2.0 | Request/response schemas | Already used for all API schemas |
| clsx | Latest | Conditional CSS classes | Already used in all components |

### Supporting (Already in Project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `usePolling` hook | Custom | Interval-based polling | Real-time asset population (VGED-10) |
| `useProjectStatus` hook | Custom | Status polling with backoff | Editor lifecycle management |
| `SceneEditorCard` | Custom | Full scene editing card | Scene grid in VideoGenEditor |
| `ManifestSelector` | Custom | Manifest picker | Config bar manifest selection |
| `MarkdownEditorModal` | Custom | Rich text editing | Prompt editing in config bar |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Polling (current) | WebSocket/SSE | Better real-time UX but adds infrastructure complexity; polling is established pattern in this project |
| Single VideoGenEditor | Keep GenerateForm + add editing | Less disruption but duplicates config UI and doesn't solve the fragmented workflow |
| New scene card component | Extend SceneEditorCard | New component risks diverging; SceneEditorCard already handles all states |

**Installation:** No new dependencies needed. All libraries already in project.

## Architecture Patterns

### Recommended Component Structure
```
frontend/src/components/
├── VideoGenEditor.tsx         # NEW: Main unified editor (VGED-07)
├── GenerateThroughSlider.tsx  # NEW: Pipeline stage slider (VGED-08)
├── ProjectConfigBar.tsx       # NEW: Collapsible config section (extracted)
├── SceneEditorCard.tsx        # EXISTING: Extended with generation_status
├── GenerateForm.tsx           # KEEP: Deprecated but functional fallback
├── ProgressView.tsx           # KEEP: Deprecated but functional fallback
├── EditModeOverlay.tsx        # KEEP: Used from ProjectDetail initially
└── Layout.tsx                 # MODIFIED: Add "editor" to View type

backend/vidpipe/
├── api/routes.py              # MODIFIED: Add POST /projects, POST /projects/{id}/generate
├── db/models.py               # MODIFIED: Scene.generation_status, Project.prompt nullable
├── orchestrator/
│   ├── state.py               # MODIFIED: Add "draft" to PIPELINE_STATES
│   └── pipeline.py            # MODIFIED: Gap-filling orchestration
├── pipeline/
│   ├── storyboard.py          # MODIFIED: Skip scenes with text
│   ├── keyframes.py           # MODIFIED: Skip scenes with keyframes
│   ├── video_gen.py           # MODIFIED: Skip scenes with clips
│   └── stitcher.py            # MODIFIED: Skip if final video exists
```

### Pattern 1: Editor State Machine
**What:** The VideoGenEditor manages a three-state lifecycle: DRAFTING -> RUNNING -> EDITING (cyclic)
**When to use:** The core state drives button visibility, polling, and card rendering
**Example:**
```typescript
// Derived from projectId + project status + isGenerating
type EditorMode = "drafting" | "running" | "editing";

function getEditorMode(projectId: string | null, status: string, isGenerating: boolean): EditorMode {
  if (!projectId) return "drafting";
  if (isGenerating || !TERMINAL_STATUSES.has(status)) return "running";
  return "editing";
}
```

### Pattern 2: Gap-Filling Pipeline
**What:** Pipeline stages inspect existing assets and skip scenes that already have content
**When to use:** When `POST /api/projects/{id}/generate` is called on a project with pre-populated assets
**Example:**
```python
# In storyboard.py
scenes_needing_text = [s for s in scenes if not s.scene_description.strip()]
if not scenes_needing_text:
    logger.info("All scenes have text, skipping storyboard")
    project.status = "keyframing"
    await session.commit()
    return
```

### Pattern 3: Per-Scene Status Updates for Real-Time UI
**What:** Pipeline sets `scene.generation_status` before/after each operation for frontend spinners
**When to use:** During pipeline execution, committed immediately for next poll cycle
**Example:**
```python
# In keyframes.py, before generating start KF:
scene.generation_status = "generating_start_kf"
await session.commit()
# ... generate ...
scene.generation_status = None
await session.commit()
```

### Pattern 4: Collapsible Config with Summary
**What:** Project config bar auto-collapses during generation, shows compact summary
**When to use:** When editor transitions from DRAFTING to RUNNING mode
**Example:**
```typescript
// Auto-collapse on generation start
useEffect(() => {
  if (editorMode === "running" && configExpanded) {
    setConfigExpanded(false);
  }
}, [editorMode]);
```

### Anti-Patterns to Avoid
- **Don't create a new scene card component:** SceneEditorCard already handles empty, partial, generating, and complete states. Extend it rather than duplicating.
- **Don't bypass the existing pipeline orchestrator:** The gap-filling logic should modify the existing `run_pipeline` function (or create a parallel `run_gap_fill_pipeline`), not implement a separate pipeline flow.
- **Don't make Scene text fields nullable:** The existing `Mapped[str]` pattern works with empty strings. Making them `Optional[str]` would break existing code that assumes non-null strings.
- **Don't poll from multiple places simultaneously:** The VideoGenEditor should have one polling source (like ProgressView does) and distribute updates to children via props.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Polling infrastructure | Custom setTimeout loops | Existing `usePolling` hook | Already battle-tested, handles cleanup on unmount |
| Scene editing UI | New scene editing component | Existing `SceneEditorCard` | 1062 lines of battle-tested editing with upload, regen, staleness, lightbox |
| Model selection UI | New model picker | Extract from GenerateForm/EditModeOverlay | Both already have filtered model lists with Ollama integration |
| Cost estimation | New cost calculator | Existing `estimatePartialCost` from constants.ts | Already handles per-stage cost with all model pricing |
| Pipeline state machine | New orchestration | Existing `run_pipeline` + `state.py` | Add gap-filling to existing flow rather than parallel implementation |
| Import/export | New serialization | Existing `buildSchema`/`handleImportSchema` from EditModeOverlay | Already handles project + scene text serialization |

**Key insight:** This phase is 70% composition of existing pieces and 30% new logic (draft creation, gap-filling, per-scene status). The risk is in rewriting what already works rather than composing it.

## Common Pitfalls

### Pitfall 1: Non-Nullable Scene Text Fields
**What goes wrong:** Creating empty Scene rows fails because `scene_description`, `start_frame_prompt`, etc. are `Mapped[str]` (non-nullable)
**Why it happens:** The design doc says "empty Scene rows" but the DB schema requires strings
**How to avoid:** Use empty strings (`""`) for all text fields when creating draft scenes. The gap-filling storyboard stage checks `if not s.scene_description.strip()` to detect scenes needing text generation.
**Warning signs:** `IntegrityError: NOT NULL constraint failed` when creating draft projects

### Pitfall 2: Project.prompt is Required Non-Nullable
**What goes wrong:** The design doc allows `prompt: str = ""` for manual projects, but `Project.prompt` is `Mapped[str]` (non-nullable) AND the existing `POST /generate` requires a non-empty prompt
**Why it happens:** The original design assumed prompt is always provided upfront
**How to avoid:** For `POST /api/projects` (draft creation), accept empty string prompt. For `POST /api/projects/{id}/generate`, validate that either prompt is non-empty OR all scenes have text (user-provided content). The Project model already accepts empty strings -- only the API validation needs to change.
**Warning signs:** Draft projects with empty prompts that fail when generation starts without scene text

### Pitfall 3: Polling Race Conditions
**What goes wrong:** Multiple polling sources (status polling + detail polling) create stale data conflicts
**Why it happens:** ProgressView polls both `/status` (2-5s) and `/projects/{id}` (3s). If VideoGenEditor does the same, updates can arrive out of order.
**How to avoid:** Single polling source in VideoGenEditor: poll `GET /projects/{id}` at 2s during generation. Derive status from the full detail response. Don't use `useProjectStatus` hook separately.
**Warning signs:** Scene cards flickering between states, status badge showing stale status

### Pitfall 4: Stop Flag Not Checked Within Stages
**What goes wrong:** User clicks Pause but pipeline continues for minutes until the next stage boundary
**Why it happens:** Current `_check_stopped` is only called between stages in `run_pipeline`. Individual pipeline stages (storyboard, keyframes, video_gen) don't check the stop flag inside their per-scene loops.
**How to avoid:** Add `await session.refresh(project); if project.status == "stopped": return` inside each stage's per-scene loop. The `PipelineStopped` exception pattern already exists.
**Warning signs:** User clicks Pause, sees "Stopping..." for 5+ minutes while video_gen processes remaining scenes

### Pitfall 5: Storyboard LLM Context for Partial Fills
**What goes wrong:** When 3 of 8 scenes have user text and 5 need generation, the LLM generates 5 scenes without narrative context from the existing 3
**Why it happens:** Current storyboard generation sends a clean prompt. Gap-filling needs to pass existing scenes as context.
**How to avoid:** When generating for partial scenes, include existing scene text in the LLM prompt as fixed context: "Scene 1 (provided by user): [text]. Scene 2 (GENERATE): ..."
**Warning signs:** Generated scenes that don't flow narratively from user-provided scenes

### Pitfall 6: Frontend State Drift Between Editor and Backend
**What goes wrong:** User edits scene text in the editor while generation is running, but the next poll overwrites their edits
**Why it happens:** Polling replaces the entire scene data, including text fields the user may have edited locally
**How to avoid:** Track local edits separately (like `sceneEdits` in EditModeOverlay). On poll, only update fields the user hasn't edited. Compare incoming data against the pre-edit baseline, not the current local state.
**Warning signs:** User types in a scene description, it briefly appears, then reverts on next poll

## Code Examples

### Backend: Create Draft Project Endpoint
```python
# In routes.py
class CreateProjectRequest(BaseModel):
    prompt: str = ""
    title: str = ""
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    clip_duration: int = 6
    scene_count: int = 3
    text_model: str = "gemini-2.5-flash"
    image_model: str = "gemini-2.5-flash-image"
    video_model: str = "veo-3.1-fast-generate-001"
    enable_audio: bool = True
    manifest_id: Optional[str] = None
    quality_mode: bool = False
    candidate_count: int = 1
    vision_model: Optional[str] = None

@router.post("/projects", status_code=201)
async def create_draft_project(request: CreateProjectRequest):
    async with async_session() as session:
        project = Project(
            title=request.title or None,
            prompt=request.prompt,  # Can be empty for drafts
            style=request.style,
            aspect_ratio=request.aspect_ratio,
            target_clip_duration=request.clip_duration,
            target_scene_count=request.scene_count,
            text_model=request.text_model,
            image_model=request.image_model,
            video_model=request.video_model,
            audio_enabled=request.enable_audio,
            vision_model=request.vision_model,
            quality_mode=request.quality_mode,
            candidate_count=request.candidate_count,
            status="draft",
        )
        session.add(project)
        await session.flush()

        # Create empty scene rows
        for i in range(request.scene_count):
            scene = Scene(
                project_id=project.id,
                scene_index=i,
                scene_description="",
                start_frame_prompt="",
                end_frame_prompt="",
                video_motion_prompt="",
                transition_notes="",
                status="pending",
            )
            session.add(scene)

        # Handle manifest if provided
        if request.manifest_id:
            # ... same manifest snapshot logic as POST /generate ...
            pass

        await session.commit()
        await session.refresh(project)

        return {"project_id": str(project.id), "status": "draft", "scene_count": request.scene_count}
```

### Backend: Start Generation (Gap-Fill) Endpoint
```python
class StartGenerationRequest(BaseModel):
    run_through: Optional[str] = None
    text_model: Optional[str] = None
    image_model: Optional[str] = None
    video_model: Optional[str] = None
    vision_model: Optional[str] = None
    clip_duration: Optional[int] = None
    enable_audio: Optional[bool] = None

@router.post("/projects/{project_id}/generate", status_code=202)
async def start_generation(
    project_id: uuid.UUID,
    request: StartGenerationRequest,
    background_tasks: BackgroundTasks,
):
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        if project.status not in ("draft", "stopped", "staged", "failed", "complete"):
            raise HTTPException(409, f"Cannot start generation from status '{project.status}'")

        # Apply overrides
        if request.run_through is not None:
            project.run_through = request.run_through if request.run_through != "all" else None
        if request.text_model: project.text_model = request.text_model
        if request.image_model: project.image_model = request.image_model
        if request.video_model: project.video_model = request.video_model
        if request.vision_model is not None: project.vision_model = request.vision_model or None
        if request.clip_duration: project.target_clip_duration = request.clip_duration
        if request.enable_audio is not None: project.audio_enabled = request.enable_audio

        # Transition from draft to pending
        if project.status == "draft":
            project.status = "pending"

        await session.commit()

    background_tasks.add_task(run_pipeline_background, project_id)
    return {"project_id": str(project_id), "status": "pending"}
```

### Backend: Per-Scene Generation Status
```python
# In models.py - add to Scene class
generation_status: Mapped[Optional[str]] = mapped_column(
    String(32), nullable=True, default=None
)

# In keyframes.py - before generating start keyframe:
scene.generation_status = "generating_start_kf"
await session.commit()
try:
    # ... generate keyframe ...
    scene.generation_status = None
    await session.commit()
except Exception:
    scene.generation_status = "failed"
    await session.commit()
    raise
```

### Frontend: VideoGenEditor State Shape
```typescript
interface VideoGenEditorState {
  // Project identity
  projectId: string | null;
  editorMode: "drafting" | "running" | "editing";

  // Project config (mirrors GenerateForm state)
  title: string;
  prompt: string;
  style: string;
  aspectRatio: string;
  clipDuration: number;
  sceneCount: number;
  textModel: string;
  imageModel: string;
  videoModel: string;
  visionModel: string;
  enableAudio: boolean;
  manifestId: string | null;
  qualityMode: boolean;
  candidateCount: number;

  // Generation control
  generateThrough: number; // 0-4 slider value
  configExpanded: boolean;

  // Project detail (populated after creation)
  detail: ProjectDetail | null;
  sceneEdits: Record<number, Record<string, string>>;

  // UI state
  error: string | null;
  submitting: boolean;
}
```

### Frontend: Generate Through Slider
```typescript
function GenerateThroughSlider({
  value, onChange, stageCompletion
}: {
  value: number;
  onChange: (v: number) => void;
  stageCompletion: boolean[]; // which stages have completed content
}) {
  const labels = ["Storyboard", "Keyframes", "Video", "Stitch", "All"];
  const runThroughMap = ["storyboard", "keyframes", "video", null, null];

  return (
    <div>
      <input type="range" min={0} max={4} step={1} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-cyan-500" />
      <div className="flex justify-between mt-1">
        {labels.map((label, i) => (
          <span key={label} className={clsx(
            "text-[10px]",
            i <= value ? "text-cyan-400" : "text-gray-600",
            stageCompletion[i] && "text-green-400",
          )}>{label}</span>
        ))}
      </div>
    </div>
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GenerateForm -> ProgressView (two screens) | VideoGenEditor (single screen) | Phase 15 | Eliminates context switching; users see scene cards from moment of creation |
| Pipeline overwrites all content | Gap-filling pipeline | Phase 15 | Users can pre-populate any field; AI only fills gaps |
| No draft projects | Draft status with empty scenes | Phase 15 | Projects exist before generation starts; enables manual asset upload before AI |
| Stop only between stages | Per-scene stop checking | Phase 15 | Pause resolves in seconds instead of minutes |

## Open Questions

1. **Scene Text Field Nullability**
   - What we know: Scene model uses `Mapped[str]` (non-nullable) for all text fields. Empty strings work as "no content" sentinel.
   - What's unclear: Should we change to `Mapped[Optional[str]]` for cleaner null semantics, or keep empty strings for backward compat?
   - Recommendation: Keep `Mapped[str]` with empty strings. Changing nullability would require a migration and could break existing code that assumes non-null. The gap-filling check `if not s.scene_description.strip()` works with empty strings.

2. **Draft Auto-Save**
   - What we know: The design doc recommends auto-save with 2s debounce.
   - What's unclear: Whether this should be in Phase 15 scope or deferred.
   - Recommendation: Defer auto-save. For Phase 15, save draft state via explicit "Save Draft" action or auto-save on significant events (scene count change, model change). This avoids creating many trivial DB updates.

3. **Parallel vs Sequential Storyboard Generation for Partial Fills**
   - What we know: When some scenes have user text and others need generation, the LLM should see existing text for context.
   - What's unclear: Should missing scenes be generated in one LLM call or individually?
   - Recommendation: One LLM call with existing scenes as context, matching the current batch storyboard pattern. Return all generated scenes in one structured response.

4. **Final Video Upload Endpoint**
   - What we know: All scene-level upload endpoints exist. No `PUT /api/projects/{id}/final-video` exists.
   - What's unclear: Should uploading a final video mark the project as "complete" automatically?
   - Recommendation: Upload sets `project.output_path` to the uploaded file. Don't auto-mark complete -- let the user click "Generate" (which skips stitching since output exists) to explicitly complete the project.

5. **EditModeOverlay Consolidation Timing**
   - What we know: The design doc recommends keeping EditModeOverlay initially and only using VideoGenEditor for new projects.
   - What's unclear: Whether to consolidate in this phase or defer.
   - Recommendation: Defer consolidation. Phase 15 adds VideoGenEditor for new projects. Existing projects continue to use ProjectDetail + EditModeOverlay. This minimizes the blast radius and allows iterating on the new editor before committing to it for all workflows.

## Existing Infrastructure Inventory

### Endpoints That Already Exist (No Work Needed)
| Endpoint | Status | Used By |
|----------|--------|---------|
| `PUT /api/projects/{id}/scenes/{idx}/keyframes/{position}` | Complete | SceneEditorCard upload |
| `PUT /api/projects/{id}/scenes/{idx}/clip` | Complete | SceneEditorCard upload |
| `DELETE /api/projects/{id}/scenes/{idx}/clip` | Complete | SceneEditorCard delete |
| `DELETE /api/projects/{id}/scenes/{idx}/keyframes/{position}` | Complete | SceneEditorCard delete |
| `POST /api/projects/{id}/stop` | Complete | ProgressView stop |
| `POST /api/projects/{id}/resume` | Complete | ProgressView/ContinuePanel resume |
| `PATCH /api/projects/{id}/edit` | Complete | EditModeOverlay commit |
| `POST /api/projects/{id}/scenes/{idx}/regenerate` | Complete | SceneEditorCard regen |
| `POST /api/projects/{id}/scenes/{idx}/regenerate-text` | Complete | SceneEditorCard text regen |
| `POST /api/projects/{id}/generate-scene-fields` | Complete | SceneEditorCard text-only gen |
| `POST /api/projects/{id}/generate-new-scene` | Complete | SceneEditorCard full scene gen |

### Endpoints That Need Creation
| Endpoint | Purpose | Complexity |
|----------|---------|------------|
| `POST /api/projects` | Create draft project with empty scenes | Medium (new pattern) |
| `POST /api/projects/{id}/generate` | Start/resume gap-filling generation | High (gap-fill logic) |
| `PUT /api/projects/{id}/final-video` | Upload final stitched video | Low (follows keyframe upload pattern) |

### Frontend Components Reusable As-Is
| Component | Lines | Reuse Strategy |
|-----------|-------|----------------|
| `SceneEditorCard` | 1062 | Direct reuse in VideoGenEditor scene grid |
| `ManifestSelector` | ~100 | Direct reuse in config bar |
| `MarkdownEditorModal` | ~150 | Direct reuse for prompt editing |
| `CopyButton` | ~30 | Direct reuse |
| `StatusBadge` | ~40 | Extended with "draft" status |
| `PipelineStepper` | ~80 | Reuse for generation progress display |

### Frontend Patterns to Extract from Existing Components
| Pattern | Source | Target |
|---------|--------|--------|
| Model selection + filtering + Ollama merge | `GenerateForm` lines 46-120 | `VideoGenEditor` config bar |
| Cost estimation display | `GenerateForm` lines 150-165 | `VideoGenEditor` controls |
| Import/export JSON schema | `EditModeOverlay` lines 18-285 | `VideoGenEditor` import/export |
| Scene edit tracking + dirty detection | `EditModeOverlay` lines 394-498 | `VideoGenEditor` scene edits |
| Background operation polling | `EditModeOverlay` lines 92-118 | `VideoGenEditor` generation polling |

## Sources

### Primary (HIGH confidence)
- `/home/ubuntu/work/video-pipeline/docs/vidgeneditor.md` - Comprehensive design document for Phase 15
- `/home/ubuntu/work/video-pipeline/.planning/REQUIREMENTS.md` - VGED-01 through VGED-12 requirement definitions
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/models.py` - Current database schema (Scene, Project, Keyframe, VideoClip)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/routes.py` - All existing API endpoints (4900+ lines)
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/orchestrator/pipeline.py` - Current pipeline orchestrator with state machine
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/orchestrator/state.py` - Pipeline state constants and resume logic
- `/home/ubuntu/work/video-pipeline/frontend/src/components/GenerateForm.tsx` - Current project creation form (654 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/components/ProgressView.tsx` - Current generation monitoring (241 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/components/EditModeOverlay.tsx` - Current project editing overlay (1100 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/components/SceneEditorCard.tsx` - Full scene editing card (1062 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/App.tsx` - Current view routing (111 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/components/Layout.tsx` - Navigation and view types (53 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/api/client.ts` - All API client functions (529 lines)
- `/home/ubuntu/work/video-pipeline/frontend/src/api/types.ts` - All TypeScript API types (529 lines)

### Secondary (MEDIUM confidence)
- `/home/ubuntu/work/video-pipeline/.planning/STATE.md` - Project decisions and architecture history

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project, no new dependencies
- Architecture: HIGH - Direct investigation of existing codebase reveals clear composition strategy
- Pitfalls: HIGH - Identified from actual code analysis (non-nullable fields, polling patterns, stop flag gaps)
- Backend scope: HIGH - Verified which endpoints exist vs need creation by examining routes.py
- Frontend scope: HIGH - Verified SceneEditorCard and EditModeOverlay cover 90%+ of needed UI

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable domain, internal project)
