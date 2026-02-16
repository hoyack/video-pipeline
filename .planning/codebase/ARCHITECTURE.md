# Architecture

**Analysis Date:** 2026-02-16

## Pattern Overview

**Overall:** Tiered pipeline orchestration with async-first design

**Key Characteristics:**
- Stateful pipeline with resumable state machine (pending → storyboarding → keyframing → video_gen → stitching → complete)
- Idempotent step execution enabling recovery from failures at any stage
- Async-first architecture using FastAPI backend + React/TypeScript frontend
- Database-driven state persistence with SQLAlchemy ORM + SQLite WAL mode
- Separation of concerns: API layer, orchestration layer, pipeline tasks, services

## Layers

**API Layer:**
- Purpose: HTTP endpoint handling and request validation
- Location: `backend/vidpipe/api/`
- Contains: FastAPI app setup, route handlers, Pydantic request/response schemas
- Depends on: Database models, orchestrator, services
- Used by: Frontend, external clients
- Key files: `app.py` (FastAPI lifespan, CORS, exception handlers), `routes.py` (endpoint implementations)

**Orchestration Layer:**
- Purpose: Pipeline state machine, step sequencing, error recovery
- Location: `backend/vidpipe/orchestrator/`
- Contains: Pipeline execution logic, state transitions, resume point calculation
- Depends on: Database models, pipeline task modules
- Used by: API (background tasks)
- Key files: `pipeline.py` (main orchestration loop), `state.py` (state machine constants)

**Pipeline Task Layer:**
- Purpose: Execute individual pipeline steps (storyboard, keyframes, video_gen, stitching)
- Location: `backend/vidpipe/pipeline/`
- Contains: Gemini/Imagen/Veo API calls, image/video generation, file I/O
- Depends on: Database models, services, config
- Used by: Orchestrator
- Key files: `storyboard.py`, `keyframes.py`, `video_gen.py`, `stitcher.py`

**Data Access Layer:**
- Purpose: ORM models and database configuration
- Location: `backend/vidpipe/db/`
- Contains: SQLAlchemy models, async session factory, PRAGMA configuration
- Depends on: Config
- Used by: All other layers
- Key files: `models.py` (Project, Scene, Keyframe, VideoClip, PipelineRun), `engine.py` (async SQLAlchemy setup)

**Services Layer:**
- Purpose: Cross-cutting concerns (file management, external API clients)
- Location: `backend/vidpipe/services/`
- Contains: FileManager (filesystem I/O), Vertex client (GCP API factory)
- Depends on: Config
- Used by: Pipeline tasks

**Configuration Layer:**
- Purpose: Settings management with YAML + environment variable support
- Location: `backend/vidpipe/config.py`
- Contains: Pydantic Settings classes, YAML source loader, singleton instance
- Depends on: Nothing
- Used by: All backend layers

**Frontend Layer:**
- Purpose: User-facing UI for generation, progress tracking, project management
- Location: `frontend/src/`
- Contains: React components, API client, custom hooks
- Depends on: Backend API
- Used by: End users

## Data Flow

**Video Generation Flow:**

1. **Request Phase** (API Handler)
   - Frontend sends `GenerateRequest` to `POST /api/generate`
   - API validates models, aspect ratio, clip duration constraints
   - Creates `Project` record with status="pending", derives scene_count from total_duration/clip_duration
   - Spawns background task with `run_pipeline(project_id)`
   - Returns 202 Accepted with project_id

2. **Storyboard Phase** (Orchestrator → Storyboard Task)
   - Orchestrator loads project, determines resume point
   - `generate_storyboard()` calls Gemini with structured output (storyboard system prompt)
   - Gemini returns `StoryboardOutput` with scenes, style_guide, character details
   - For each scene, creates `Scene` record with:
     - scene_index, scene_description
     - start_frame_prompt, end_frame_prompt
     - video_motion_prompt, transition_notes
     - status="pending"
   - Stores storyboard_raw (full Gemini response) in project
   - Updates project.status to "keyframing"

3. **Keyframe Phase** (Orchestrator → Keyframes Task)
   - For each Scene in order (seq, no parallelization):
     - **Scene 0 start keyframe:** Call Imagen with text prompt (start_frame_prompt)
     - **All other start frames:** Use previous end frame as visual reference
     - **End keyframes:** Call image-conditioned Imagen with motion context
     - Rate limit between calls (default 3s delay)
     - Save PNG files via `FileManager.save_keyframe()`
     - Create `Keyframe` records with position="start"/"end", source="generated"
   - Update each scene status to "keyframes_done"
   - Update project.status to "video_gen"

4. **Video Generation Phase** (Orchestrator → Video Gen Task)
   - For each Scene with completed keyframes:
     - Load start/end keyframes from disk
     - Submit Veo job with:
       - first_frame_image (start keyframe PNG)
       - last_frame_image (end keyframe PNG)
       - video_motion_prompt (motion description)
       - aspect_ratio (validated at API time)
       - duration (validated at API time)
     - Store operation_name (long-running operation ID)
     - Poll with exponential backoff (15s interval, max 40 polls = ~10min timeout)
     - On completion: download MP4, save via `FileManager.save_clip()`
     - Create `VideoClip` record with source="generated", status="complete"
   - Handle Veo content policy filtering:
     - Level 0: Retry with original prompt
     - Level 1: Prepend safety language to motion prompt, regenerate
     - Level 2: Regenerate end keyframe with safety prompt, retry video
   - Update project.status to "stitching"

5. **Stitching Phase** (Orchestrator → Stitcher Task)
   - Load all VideoClips in scene order
   - Use ffmpeg to concatenate MP4s with crossfade transitions (0s default)
   - Write final MP4 to `FileManager.get_output_path()`
   - Create final output_path reference in project
   - Update project.status to "complete"

**Resume/Fork Flows:**

- **Resume (POST /api/projects/{id}/resume):**
  - API validates project is in resumable state (failed, stopped, or active)
  - Orchestrator calls `get_resume_step()` which examines database completion state
  - Re-enters pipeline at appropriate step (pending, keyframing, video_gen, or stitching)
  - Skips completed steps, regenerates from failure point

- **Fork (POST /api/projects/{id}/fork):**
  - API validates source project is terminal (complete, failed, stopped)
  - Computes invalidation point based on edits:
    - Prompt/style/text_model changes → restart from pending
    - Image_model/aspect_ratio changes → restart from keyframing
    - Video_model/audio_enabled/clip_duration changes → restart from video_gen
    - Pure expansion (more scenes) → continue from keyframing
    - Scene count decrease → restart from pending
  - Copies Scene, Keyframe, VideoClip records up to boundary with source="inherited"
  - For expansions, generates new scenes via Gemini (using existing storyboard as context)
  - Spawns background task to continue pipeline from resume_from status

**State Management:**

- **Project-level state:** Stored in `Project.status` (single state machine)
- **Scene-level state:** Stored in `Scene.status` (tracks progression within project)
- **Artifact state:** Stored in `Keyframe.source` (generated/inherited) and `VideoClip.source` (generated/inherited)
- **Error persistence:** `Project.error_message` captures failure reason
- **Timing metadata:** `PipelineRun` tracks start, completion, duration, step logs

## Key Abstractions

**Project:**
- Purpose: Top-level container for a video generation request
- Examples: `backend/vidpipe/db/models.py` (Project class)
- Pattern: Domain model with embedded config (prompt, style, models, aspect_ratio, durations)
- Lifecycle: pending → complete/failed/stopped

**Scene:**
- Purpose: Single unit of video (one storyboard entry → one video clip)
- Examples: `backend/vidpipe/db/models.py` (Scene class)
- Pattern: Ordered sequence within project, tracks start/end keyframe prompts and motion
- Relationships: One project has multiple scenes; one scene has one or two keyframes (start/end) and one video clip

**Keyframe:**
- Purpose: Static image serving as anchor for video interpolation
- Examples: `backend/vidpipe/db/models.py` (Keyframe class)
- Pattern: PNG file with prompt used for generation and source tracking (generated/inherited)
- Position: "start" or "end" within scene

**VideoClip:**
- Purpose: Generated MP4 segment (interpolation between keyframes)
- Examples: `backend/vidpipe/db/models.py` (VideoClip class)
- Pattern: MP4 file with status tracking, operation_name for polling, safety regen count
- Metadata: duration_seconds, poll_count, veo_submission_count, safety_regen_count

**Pipeline State Machine:**
- Purpose: Enforce ordered execution with resumability
- Examples: `backend/vidpipe/orchestrator/state.py`
- Pattern: Dict-based states (PIPELINE_STATES, STEP_TRANSITIONS)
- Resumable states: pending, failed, stopped, storyboarding, keyframing, video_gen, stitching

**FileManager:**
- Purpose: Structured file I/O with path traversal protection
- Examples: `backend/vidpipe/services/file_manager.py`
- Pattern: Directory hierarchy per project (project_id/keyframes/, project_id/clips/, project_id/output/)
- Safety: Validates paths are within base_dir using `is_relative_to()`

## Entry Points

**Backend:**
- Location: `backend/vidpipe/api/__main__.py`
- Triggers: `python -m vidpipe.api`
- Responsibilities: Start FastAPI server on configured host:port, initialize database

**API Endpoints (Primary):**
- `POST /api/generate` - Create new video generation project (returns 202)
- `GET /api/projects/{id}/status` - Poll lightweight project status
- `GET /api/projects/{id}` - Get full project detail with scene breakdown
- `GET /api/projects` - List all projects
- `POST /api/projects/{id}/resume` - Resume failed/stopped project
- `POST /api/projects/{id}/stop` - Stop active pipeline
- `POST /api/projects/{id}/fork` - Create variant with edits
- `GET /api/projects/{id}/download` - Download final MP4
- `GET /api/keyframes/{id}` - Serve keyframe PNG
- `GET /api/clips/{id}` - Serve video clip MP4
- `GET /api/metrics` - Aggregate statistics

**Frontend:**
- Location: `frontend/src/main.tsx`
- Triggers: `npm run dev` (Vite dev server)
- Responsibilities: Serve SPA, poll API for status, render project list/detail/progress

## Error Handling

**Strategy:** Failure-driven state machine with automatic retry and user recovery options

**Patterns:**

- **Transient Errors:** Use tenacity retry decorator with exponential backoff
  - `@retry(stop=stop_after_attempt(7), wait=wait_exponential(...))`
  - Server errors (5xx), rate limits (429), connection timeouts retried up to 7 times
  - Example: `backend/vidpipe/pipeline/keyframes.py` uses `_is_retriable()` to classify exceptions

- **Persistent Failures:** Persist to database, set project.status="failed"
  - API handler catches exception, stores error_message with step name
  - User can call `POST /api/projects/{id}/resume` to retry from failure point
  - Example: `backend/vidpipe/orchestrator/pipeline.py` lines 252-280

- **Content Policy Failures (Veo):** Escalating remediation
  - Level 0: Retry with original prompt (1 attempt)
  - Level 1: Prepend safety language to video_motion_prompt (1 attempt)
  - Level 2: Regenerate end keyframe with safety prompt, retry video (1 attempt)
  - Max 3 escalation levels per clip
  - Example: `backend/vidpipe/pipeline/video_gen.py` implements levels

- **User Stops Pipeline:** Set project.status="stopped", exit gracefully
  - Orchestrator checks stop flag between steps: `await _check_stopped(session, project_id)`
  - Partial work preserved; can resume with `POST /api/projects/{id}/resume`
  - Example: `backend/vidpipe/orchestrator/pipeline.py` raises `PipelineStopped` exception

## Cross-Cutting Concerns

**Logging:**
- Framework: Python stdlib `logging` module
- Pattern: `logger = logging.getLogger(__name__)` in each module
- Levels: INFO for step transitions, ERROR for failures, DEBUG for detailed traces
- Examples: `backend/vidpipe/orchestrator/pipeline.py` line 27

**Validation:**
- Framework: Pydantic (request/response schemas) + custom validation
- Location: `backend/vidpipe/api/routes.py` (model ID allowlists, aspect ratio, clip duration constraints)
- Pattern: Raise HTTPException(status_code=422) for invalid input

**Authentication:**
- Not implemented; relies on Google Cloud ADC (Application Default Credentials)
- Veo, Imagen, Gemini auth handled via `google.genai` SDK
- Frontend uses unauthenticated API (suitable for development/demo)

**Database Transactions:**
- Pattern: Async context manager (`async with async_session() as session`)
- Commit on success, automatic rollback on exception
- WAL mode (Write-Ahead Logging) for crash safety and concurrency
- Example: `backend/vidpipe/api/routes.py` line 401 (session creation)

**Async/Await:**
- Framework: asyncio + SQLAlchemy async driver (aiosqlite)
- Pattern: All database ops await async session methods
- Caveat: Never share session across async boundaries; create fresh session per background task
- Example: `backend/vidpipe/api/routes.py` line 346-357 (run_pipeline_background)

---

*Architecture analysis: 2026-02-16*
