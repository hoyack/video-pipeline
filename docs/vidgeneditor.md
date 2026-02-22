# Video Generation Editor — Unified Create/Edit Experience

## Vision

Replace the current two-screen flow (GenerateForm → ProgressView) with a single **Video Generation Editor** that merges project creation, live generation monitoring, and editing into one view. Users can:

- Create a project and immediately see empty scene cards
- Pre-populate any field (text, keyframes, clips, final video) by hand or upload
- Launch generation that only fills in what's missing
- Watch assets appear on scene cards in real time as the pipeline runs
- Pause/resume generation at any point
- Edit and regenerate individual parts without leaving the view

This turns the tool from "submit a prompt, wait for a video" into a **composable project workspace** where AI fills gaps and users retain full control.

---

## Current Architecture (What Exists Today)

### Frontend Flow

```
ProjectList → [+ New] → GenerateForm → [Submit] → ProgressView → [View Details] → ProjectDetail → [Edit] → EditModeOverlay
```

| Component | Role | Key Limitation |
|---|---|---|
| `GenerateForm` | Project creation form. Fields: prompt, style, models, duration, "Generate Through" buttons. No scene cards. | Blind submission — no preview, no partial fill |
| `ProgressView` | Status tracker + scene cards during generation. Polls `/status` (2-5s) and `/projects/{id}` (3s). | Read-only — can only stop, not edit or upload |
| `ProjectDetail` | View completed project. Scene cards are read-only `SceneCard`. | Must click "Edit" to enter editing mode |
| `EditModeOverlay` | Rich editing: scene count slider, per-scene text editing, keyframe/clip upload, regen, staleness tracking, export/import JSON. Uses `SceneEditorCard`. | Only available on terminal-status projects |
| `SceneEditorCard` | Full editor card: upload keyframes/clips, regen per-asset, text editing, staleness badges, empty slot generation. | Not used during creation or generation |

### Backend Flow

```
POST /api/generate → project created → background run_pipeline()
  → storyboarding (creates Scene rows, populates text fields)
  → keyframing (creates Keyframe rows, saves PNGs)
  → video_gen (creates VideoClip rows, saves MP4s)
  → stitching (creates final MP4)
```

- Status communicated via polling (`GET /status` + `GET /projects/{id}`)
- No WebSocket or SSE
- `run_through` field controls early stopping at stage boundaries
- Resume via `POST /projects/{id}/resume`

---

## Design: The Video Generation Editor

### Core Concept

A single component (`VideoGenEditor`) replaces `GenerateForm` + `ProgressView` and reuses the rich editing infrastructure from `EditModeOverlay` / `SceneEditorCard`. The view has three zones:

```
┌─────────────────────────────────────────────────────────┐
│  PROJECT CONFIG BAR                                      │
│  [Title] [Prompt] [Style] [Models] [Manifest] [Audio]   │
├─────────────────────────────────────────────────────────┤
│  GENERATION CONTROLS                                     │
│  ◄━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►            │
│  Storyboard   Keyframes   Video   Stitch   All          │
│                                                          │
│  Duration: ◄━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━►  45s       │
│  Scenes: 8    Scene Length: [4s] [6s] [8s]               │
│                                                          │
│  [ ▶ Generate ]  [ ⏸ Pause ]  [Cost: ~$2.40]           │
├─────────────────────────────────────────────────────────┤
│  SCENE CARDS (scrollable grid)                           │
│                                                          │
│  ┌─Scene 1──────┐ ┌─Scene 2──────┐ ┌─Scene 3──────┐   │
│  │ ┌──┐   ┌──┐  │ │ ⟳ Generating │ │ ┌──┐   ╌╌╌╌  │   │
│  │ │KF│   │KF│  │ │   storyboard │ │ │KF│  empty   │   │
│  │ └──┘   └──┘  │ │              │ │ └──┘  end KF  │   │
│  │ ▶ clip.mp4   │ │              │ │ no clip yet    │   │
│  │ description…  │ │              │ │ description…   │   │
│  └──────────────┘ └──────────────┘ └───────────────┘   │
│                                                          │
│  ┌─Scene 4──────┐ ┌─Scene 5──────┐ ...                 │
│  │  ╌╌╌╌  ╌╌╌╌  │ │  ╌╌╌╌  ╌╌╌╌  │                    │
│  │  empty  empty │ │  empty  empty │                    │
│  │  no clip      │ │  no clip      │                    │
│  │  (empty slot) │ │  (empty slot) │                    │
│  └──────────────┘ └───────────────┘                     │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  FINAL VIDEO                                             │
│  [no video yet]  [Upload Final Video]                    │
└─────────────────────────────────────────────────────────┘
```

### State Machine

The editor manages a single project through its lifecycle:

```
         ┌──────────┐
         │ DRAFTING  │  ← initial state, no project_id yet
         └────┬─────┘
              │ user clicks Generate (or has filled everything)
              ▼
         ┌──────────┐
         │ RUNNING   │  ← pipeline active, polling for updates
         └────┬─────┘
              │ pipeline completes / user pauses / error
              ▼
         ┌──────────┐
         │ EDITING   │  ← project exists, user can edit/regen/upload
         └──────────┘
              │ user clicks Generate again (fill remaining gaps)
              ▼
         ┌──────────┐
         │ RUNNING   │  ← resumes from where it left off
         └──────────┘
```

Key: there is no hard boundary between states. The user can upload assets while the pipeline is running on other scenes. The state primarily controls which buttons are enabled and whether polling is active.

---

## Implementation Plan

### Phase 1: Backend — Granular Project Creation & Smart Resume

#### 1.1 New endpoint: `POST /api/projects` (create without generating)

Create a project row + empty Scene rows without starting the pipeline. This lets the frontend create the project, show scene cards, and let the user fill things in before (optionally) launching generation.

**Request body** — same fields as `GenerateRequest` plus:

```python
class CreateProjectRequest(BaseModel):
    prompt: str = ""                          # can be empty for manual projects
    title: str = ""
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    clip_duration: int = 6
    scene_count: int = 3                      # explicit scene count
    text_model: str = "gemini-2.5-flash"
    image_model: str = "gemini-2.5-flash-image"
    video_model: str = "veo-3.1-fast-generate-001"
    enable_audio: bool = True
    manifest_id: str | None = None
    quality_mode: bool = False
    candidate_count: int = 1
    vision_model: str | None = None
    # NO run_through — that goes on the generate call
```

**Response:** `{ project_id, scenes: [...] }` — returns the project with its empty scene stubs.

**Behavior:**
- Creates `Project` with `status = "draft"` (new status)
- Creates `scene_count` empty `Scene` rows with sequential indices
- If `manifest_id` provided, snapshot and link it
- Does NOT start the pipeline
- Scene rows have empty text fields and no keyframes/clips

#### 1.2 New endpoint: `POST /api/projects/{id}/generate` (start/resume with smart fill)

Replaces the implicit "generate everything" approach. This endpoint inspects what already exists and only generates what's missing, up to the requested stage.

**Request body:**

```python
class StartGenerationRequest(BaseModel):
    run_through: str | None = None            # "storyboard"|"keyframes"|"video"|None (all)
    # Optional model overrides for this run
    text_model: str | None = None
    image_model: str | None = None
    video_model: str | None = None
    vision_model: str | None = None
    clip_duration: int | None = None
    enable_audio: bool | None = None
```

**Smart fill logic in the pipeline orchestrator:**

```
For each stage up to run_through:
  STORYBOARD:
    - For each scene: if description is empty → generate via LLM
    - Scenes with user-provided text are SKIPPED
    - Commit scenes to DB, emit status update

  KEYFRAMES:
    - For each scene: if start_keyframe missing → generate
    - For each scene: if end_keyframe missing → generate (conditioned on start)
    - Scenes with uploaded keyframes are SKIPPED
    - KEYF-03 continuity: if scene N has a start KF but scene N-1 has no end KF,
      generate N-1's end KF first (dependency resolution)

  VIDEO_GEN:
    - For each scene: if clip missing → generate from keyframes
    - Scenes with uploaded clips are SKIPPED

  STITCHER:
    - If final video missing → stitch from clips
    - If final video uploaded → SKIP entirely
```

This is the key behavioral change: the pipeline becomes **gap-filling** rather than **overwriting**.

#### 1.3 Add `"draft"` to the project status enum

New status for projects that exist but haven't had any generation started. The state machine becomes:

```
draft → pending → storyboarding → keyframing → video_gen → stitching → complete
                                                                      → staged
                                                                      → stopped
                                                                      → failed
```

`draft` projects:
- Appear in the project list with a "Draft" badge
- Can be edited freely (no checkpoint needed)
- Can be deleted without confirmation
- Don't show in the active/running filters

#### 1.4 Scene-level upload endpoints

Some of these likely exist already; ensure full coverage:

| Endpoint | Purpose |
|---|---|
| `PUT /api/projects/{id}/scenes/{idx}/start-keyframe` | Upload start keyframe image |
| `PUT /api/projects/{id}/scenes/{idx}/end-keyframe` | Upload end keyframe image |
| `PUT /api/projects/{id}/scenes/{idx}/clip` | Upload video clip |
| `PUT /api/projects/{id}/final-video` | Upload final stitched video |
| `PATCH /api/projects/{id}/scenes/{idx}` | Update scene text fields |

Each upload endpoint should:
- Accept multipart file upload
- Validate file type (image/video as appropriate)
- Save to the correct file path (matching the pipeline's naming convention)
- Create/update the corresponding DB row (`Keyframe`, `VideoClip`)
- If the scene has a manifest, maintain asset associations
- Return the updated scene data

#### 1.5 Pause/stop during generation

The existing `POST /api/projects/{id}/stop` endpoint sets a flag. Enhance to:
- Be checked between individual scene operations (not just between stages)
- Return the project to `stopped` status with partial results preserved
- Enable the frontend to resume from exactly where it stopped

Add a cancellation check inside each stage's per-scene loop:

```python
# In storyboard.py, keyframes.py, video_gen.py — inside scene loop:
await session.refresh(project)
if project.status == "stopped":
    logger.info(f"Pipeline stopped by user at scene {scene.scene_index}")
    return  # partial results already committed
```

#### 1.6 Per-scene status field

Add a `generation_status` column to the `Scene` model:

```python
generation_status: Mapped[Optional[str]] = mapped_column(
    String(32), nullable=True, default=None
)
# Values: None, "generating_text", "generating_start_kf", "generating_end_kf",
#         "generating_clip", "complete", "failed"
```

This allows the frontend to show per-scene spinners without guessing from asset presence. Updated by the pipeline before/after each operation.

---

### Phase 2: Frontend — VideoGenEditor Component

#### 2.1 New component: `VideoGenEditor.tsx`

Replaces `GenerateForm` in `App.tsx` navigation. Manages the full lifecycle from project creation through generation monitoring to editing.

**State:**

```typescript
interface VideoGenEditorState {
  // Project config (pre-creation and editable)
  projectId: string | null;        // null = draft not yet saved
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
  generateThrough: number;        // 0=storyboard, 1=keyframes, 2=video, 3=stitch, 4=all
  isGenerating: boolean;
  isPaused: boolean;

  // Scene data (mirrors ProjectDetail.scenes)
  scenes: SceneDetail[];
  sceneEdits: Record<number, Record<string, string>>;

  // Upload state
  finalVideoFile: File | null;
  finalVideoUrl: string | null;
}
```

**Component structure:**

```
VideoGenEditor
├── ProjectConfigBar          (collapsible top section)
│   ├── TitleInput
│   ├── PromptEditor          (with MarkdownEditorModal)
│   ├── ManifestSelector
│   ├── StyleSelector
│   ├── ModelSelectors        (text, image, video, vision)
│   └── AudioToggle
├── GenerationControls
│   ├── GenerateThroughSlider (new — replaces button group)
│   ├── DurationControls      (total duration slider + scene length buttons)
│   ├── SceneCountDisplay
│   ├── GenerateButton        (▶ Generate / ⏸ Pause / ▶ Resume)
│   └── CostEstimate
├── SceneGrid                 (reuses SceneEditorCard)
│   ├── SceneEditorCard[0]    (may be populated, generating, or empty)
│   ├── SceneEditorCard[1]
│   └── ...
├── FinalVideoSection
│   ├── VideoPlayer | UploadDropzone
│   └── ReStitchButton
└── ImportExportBar
    ├── ImportJSON
    └── ExportJSON
```

#### 2.2 Generate Through Slider

Replace the current 4-button group with a continuous slider showing pipeline stages:

```
◄━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►
Storyboard    Keyframes      Video Clips     Stitch    All

     ▲ current position: "through Keyframes"
```

Implementation: `<input type="range" min={0} max={4} step={1}>` with custom tick marks and labels. Value maps to:

| Slider Value | `run_through` | Label |
|---|---|---|
| 0 | `"storyboard"` | Through Storyboard |
| 1 | `"keyframes"` | Through Keyframes |
| 2 | `"video"` | Through Video Clips |
| 3 | `"stitch"` | Through Stitching |
| 4 | `null` (all) | Complete Video |

The slider should visually indicate which stages have completed content (green ticks) vs which are pending (gray ticks). When the user has uploaded all assets for a stage, that tick turns green even without generation.

#### 2.3 Scene Cards — Lifecycle Rendering

Each scene card in the grid uses `SceneEditorCard` (or an extended version) throughout the entire lifecycle. The card adapts its rendering based on scene state:

**Empty (draft, no content):**
- Dashed border (existing empty slot style)
- All text fields editable and empty
- Upload buttons for keyframes and clip
- "Generate Scene" and "Text Only" buttons

**Partially filled (user uploaded some assets):**
- Solid border
- Uploaded assets shown (keyframe thumbnails, clip player)
- Missing assets shown as dashed placeholders with upload buttons
- Text fields editable

**Generating (pipeline is working on this scene):**
- Spinner/pulse animation on the asset currently being generated
- `generation_status` from backend drives which spinner is shown:
  - `generating_text` → spinner on description field
  - `generating_start_kf` → spinner on start keyframe slot
  - `generating_end_kf` → spinner on end keyframe slot
  - `generating_clip` → spinner on clip slot
- Already-generated assets render normally as they arrive
- Text fields become editable once generated (but editing during generation is deferred)

**Complete (all assets present):**
- Full rendering — keyframe images, clip player, all text
- Edit/regen/upload/delete all available
- Staleness badges if applicable

#### 2.4 Real-Time Asset Population

When the pipeline is running, the editor polls `GET /api/projects/{id}` at a fast interval (2s) and diffs the response against current state. When new data appears:

- **Storyboard text arrives:** Scene card's description field animates in (brief highlight flash). Start/end frame prompts, motion prompt populate.
- **Keyframe arrives:** Thumbnail fades in where the dashed placeholder was. Brief green border pulse.
- **Clip arrives:** Video player appears. Brief green border pulse.
- **Final video:** Player appears in the Final Video section.

Implementation uses the existing `usePolling` hook. The diff logic compares `scene.has_start_keyframe`, `has_end_keyframe`, `has_clip`, and text field emptiness against the previous poll result. A `useRef` stores the previous state for comparison.

Visual feedback: newly-arrived assets get a brief CSS transition (e.g., `ring-2 ring-green-400` that fades after 2s via `setTimeout`).

#### 2.5 Collapsible Project Config

The config bar (prompt, models, style, etc.) should be collapsible to save space during generation/editing:

- **Expanded by default** when creating a new project (no `projectId` yet)
- **Auto-collapses** when generation starts (scene cards are the focus)
- **Click to expand/collapse** at any time
- Shows a compact summary when collapsed: `"Cinematic | Gemini 2.5 Flash | Veo 3.1 | 8 scenes"`

#### 2.6 Import / Export

Reuse the existing JSON import/export from `EditModeOverlay` but available from the start:

- **Export:** Downloads the current state as JSON (even before generation)
- **Import:** Loads a JSON file and populates all fields + scene text. This allows templating workflows — export a good project structure, tweak the prompt, re-import.

Extend the schema to support importing binary asset references (URLs or base64) in a future version. For now, JSON import covers text fields only.

#### 2.7 Upload Final Video

A drop zone in the Final Video section accepts an MP4 upload. When a final video is uploaded:

- Calls `PUT /api/projects/{id}/final-video`
- The Generate Through slider automatically notes that stitching is already done
- If all scenes also have clips (uploaded or generated), the project can be marked complete
- The user can still regenerate the final video via Re-stitch if they change clips later

---

### Phase 3: Pipeline Modifications — Gap-Filling Mode

#### 3.1 Modify `storyboard.py` — Skip scenes with existing text

```python
async def generate_storyboard(session, project):
    scenes = await get_scenes(session, project.id)

    # Determine which scenes need text generation
    scenes_needing_text = [
        s for s in scenes
        if not s.description  # empty = needs generation
    ]

    if not scenes_needing_text:
        logger.info("All scenes have text, skipping storyboard stage")
        project.status = "keyframing"
        await session.commit()
        return

    # Generate text only for scenes missing it
    # Use existing scene indices to maintain continuity context
    ...
```

The storyboard LLM call must be aware of existing scene text so it can maintain narrative continuity. Pass filled scenes as context: "Scene 1 (provided): ... Scene 2 (generate): ..."

#### 3.2 Modify `keyframes.py` — Skip scenes with existing keyframes

Similar pattern. For each scene:
- If `has_start_keyframe` and user-uploaded → skip start KF generation
- If `has_end_keyframe` and user-uploaded → skip end KF generation
- KEYF-03 continuity: if scene N's start KF was uploaded but scene N-1 has no end KF, use scene N's start KF as the conditioning target (reverse the usual direction) or generate N-1's end KF to match

#### 3.3 Modify `video_gen.py` — Skip scenes with existing clips

For each scene:
- If `has_clip` and user-uploaded → skip clip generation
- If keyframes are user-uploaded but prompts are empty → generate a basic motion prompt from the keyframe descriptions

#### 3.4 Modify `stitcher.py` — Skip if final video exists

If `project.final_video_url` is already set (user uploaded), skip stitching entirely. Still mark complete.

#### 3.5 Per-scene status updates

Before each operation, set `scene.generation_status` to the active stage. After completion, set to `None` (or `"complete"` / `"failed"`). These updates are committed immediately so the next poll picks them up.

```python
# In keyframes.py, before generating start KF:
scene.generation_status = "generating_start_kf"
await session.commit()

# ... generate ...

scene.generation_status = None
await session.commit()
```

#### 3.6 Manifest association for uploaded assets

When a user uploads a keyframe or clip to a scene that has a manifest placement, the system should:
1. Maintain the existing `SceneManifest` associations
2. Run face embedding on uploaded keyframes (like Fix 1 from the manifesting engine)
3. Store `selected_reference_tags` based on what characters appear in the scene manifest
4. Allow the prompt rewriter to still enrich video prompts using manifest asset descriptions

This ensures uploaded assets integrate cleanly with the manifest system rather than bypassing it.

---

### Phase 4: Navigation & Integration

#### 4.1 App.tsx routing changes

Replace the current view state machine:

```typescript
// Before:
type View = "list" | "generate" | "progress" | "detail" | ...

// After:
type View = "list" | "editor" | "detail" | ...
```

- `"generate"` and `"progress"` merge into `"editor"`
- `ProjectList` "+ New" → navigates to `"editor"` with no `projectId`
- Existing projects can open in `"editor"` mode (replaces the Edit button flow)
- `"detail"` remains for read-only viewing

#### 4.2 ProjectDetail "Edit" button

The "Edit" button on `ProjectDetail` now navigates to `"editor"` with the existing `projectId`, pre-loading all project data. This replaces the inline `EditModeOverlay`.

Alternatively, keep `EditModeOverlay` as-is for the initial release and only use `VideoGenEditor` for new project creation. Merge later to avoid a massive PR.

**Recommended approach: phased rollout.**
- Phase 1-3: `VideoGenEditor` for new projects only
- Phase 4+: Replace `EditModeOverlay` with `VideoGenEditor` for existing projects too

#### 4.3 Draft projects in ProjectList

Add a "Draft" status filter chip. Draft projects show with a pencil icon and "Continue editing" action (opens the editor). Drafts that are older than 7 days could show a cleanup prompt.

---

### Phase 5: Pause/Resume UX

#### 5.1 Pause button behavior

When the user clicks Pause during generation:
1. Frontend calls `POST /api/projects/{id}/stop`
2. The pipeline checks the stop flag at the next scene boundary
3. All completed assets are preserved
4. Status transitions to `stopped`
5. Frontend updates: Generate button becomes "Resume", scene cards show completed assets, remaining scenes stay in their current state (empty or partial)

#### 5.2 Resume behavior

When the user clicks Resume:
1. Frontend calls `POST /api/projects/{id}/generate` (the new smart-fill endpoint)
2. Pipeline inspects what exists and resumes from the gap
3. Polling resumes, scene cards continue updating

The user can also change the Generate Through slider before resuming. Example: generated through keyframes, paused, now slide to "Video Clips" and resume — only video generation runs.

#### 5.3 Edit while paused

While paused, the user can:
- Edit any scene text
- Upload/replace keyframes or clips
- Add or remove scenes
- Change models (for subsequent generation)
- Re-order scenes (future)

When they resume, the pipeline respects all changes.

---

## Data Model Changes

### New/Modified Models

```python
# Project model additions
class Project:
    status: str  # Add "draft" to allowed values

# Scene model additions
class Scene:
    generation_status: Optional[str]  # Per-scene pipeline status
    # Values: None, "generating_text", "generating_start_kf",
    #         "generating_end_kf", "generating_clip", "complete", "failed"
    source: Optional[str]  # "generated" | "uploaded" | "mixed"
```

### New API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/projects` | Create draft project with empty scenes |
| `POST` | `/api/projects/{id}/generate` | Start/resume gap-filling generation |
| `PUT` | `/api/projects/{id}/scenes/{idx}/start-keyframe` | Upload start keyframe |
| `PUT` | `/api/projects/{id}/scenes/{idx}/end-keyframe` | Upload end keyframe |
| `PUT` | `/api/projects/{id}/scenes/{idx}/clip` | Upload video clip |
| `PUT` | `/api/projects/{id}/final-video` | Upload final video |
| `PATCH` | `/api/projects/{id}/scenes/{idx}` | Update scene text fields |

### Modified Endpoints

| Method | Path | Change |
|---|---|---|
| `GET` | `/api/projects/{id}` | Include `generation_status` per scene |
| `POST` | `/api/projects/{id}/stop` | Check between scenes, not just stages |

---

## Migration Strategy

### Backward Compatibility

- Existing `POST /api/generate` endpoint remains functional (creates + starts pipeline in one call, as today)
- Existing projects with status `pending`→`complete` continue to work
- `ProgressView` remains available as a fallback (not immediately deleted)
- `EditModeOverlay` remains for existing project editing initially

### Rollout Order

1. **Backend Phase 1:** Add `draft` status, `POST /api/projects`, per-scene status, upload endpoints
2. **Backend Phase 3:** Gap-filling pipeline modifications
3. **Frontend Phase 2:** Build `VideoGenEditor`, wire to new endpoints
4. **Frontend Phase 4:** Navigation integration, replace GenerateForm
5. **Frontend Phase 5:** Pause/resume UX polish
6. **Cleanup:** Deprecate `GenerateForm`, `ProgressView` once stable

---

## Open Questions

1. **Scene reordering** — Should the editor support drag-and-drop scene reordering? This affects KEYF-03 continuity (start KF = prev end KF). Defer to a future phase?

2. **Collaborative editing** — If we eventually want multiple users editing the same project, we'd need WebSocket for real-time sync. Current polling approach works for single-user. Worth considering in the architecture?

3. **Partial storyboard generation** — When 3 of 8 scenes have user text and 5 need generation, should the LLM generate all 5 in one call (better narrative arc) or one at a time (faster feedback)? Recommend: one call with existing scenes as context, stream results back scene by scene.

4. **Asset format validation** — What image/video formats should uploads accept? Currently the pipeline produces PNG keyframes and MP4 clips. Should we accept JPEG/WebP keyframes and auto-convert? (Fix 4's `_detect_image_mime` already handles mixed formats for Gemini.)

5. **Draft auto-save** — Should drafts auto-save to the backend as the user types, or only save on explicit action? Auto-save prevents data loss but creates potentially many draft projects. Recommend: auto-save with debounce (2s after last edit), with a "Discard Draft" cleanup option.
