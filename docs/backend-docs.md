# Backend Documentation

This document describes the backend implementation in `backend/vidpipe` as currently implemented.

## 1. Purpose and Scope

The backend provides:

- A FastAPI HTTP API (`/api/*`) used by the frontend.
- A Typer CLI (`python -m vidpipe ...`).
- A multi-step video generation pipeline:
  1. Storyboard generation
  2. Keyframe generation
  3. Video clip generation
  4. Final stitching
- A manifest/asset system for reference-driven generation.
- CV analysis and candidate scoring for continuity/quality.
- Checkpoint/versioning workflows (PipeSVN-style snapshots).

Core code lives in:

- `backend/vidpipe/api/`
- `backend/vidpipe/orchestrator/`
- `backend/vidpipe/pipeline/`
- `backend/vidpipe/services/`
- `backend/vidpipe/db/`

## 2. Runtime Architecture

### 2.1 Application Bootstrap

`backend/vidpipe/api/app.py` creates the FastAPI app and lifecycle hooks.

Startup:

- Validates `ffmpeg` (`validate_dependencies()` from `vidpipe.__init__`).
- Initializes DB schema and migrations (`init_database()`).

Shutdown:

- Closes ComfyUI singleton client (`close_comfyui_client()`).
- Disposes SQLAlchemy engine (`db.shutdown()`).

Other app-level behavior:

- CORS is enabled for `http://localhost:5173`.
- Router mounted at `/api`.
- If `frontend/dist` exists, static frontend is mounted at `/`.
- A global exception handler returns a JSON `500` payload (`error`, `detail`).

### 2.2 Execution Model

Pipeline and heavy operations run in background tasks (API `BackgroundTasks` and async workers), while state is persisted to SQLite for resume/recovery.

Important session rule:

- Background pipeline entrypoint `run_pipeline_background()` always creates a fresh DB session to avoid sharing session state across async boundaries.

### 2.3 High-Level Request Flow

Typical generation flow:

1. `POST /api/generate` creates a `Project` row (`pending`) and schedules pipeline.
2. Orchestrator (`run_pipeline`) advances status and runs each step.
3. Progress is observable via:
   - `GET /api/projects/{id}/status`
   - `GET /api/projects/{id}`
4. Final output is downloadable via `GET /api/projects/{id}/download`.

## 3. Configuration and Settings

### 3.1 Configuration Sources

`backend/vidpipe/config.py` (`Settings`) loads config in this precedence order:

1. Environment variables (`VIDPIPE_` prefix, `__` nesting).
2. `.env` file.
3. `config.yaml`.
4. Field defaults.

### 3.2 Main Config Models

- `google_cloud`: `project_id`, `location`, `use_vertex_ai`.
- `models`: `storyboard_llm`, `image_gen`, `video_gen`.
- `pipeline`: clip timing, polling, retry/backoff, crossfade, delays.
- `storage`: `database_url`, `tmp_dir`.
- `server`: `host`, `port`.
- `cv_analysis`: thresholds and max limits for CV workflows.

### 3.3 Current Repo Defaults

`config.yaml` sets defaults such as:

- `models.storyboard_llm = gemini-2.5-flash`
- `models.image_gen = gemini-2.5-flash-image`
- `models.video_gen = veo-3.1-fast-generate-001`
- `storage.database_url = sqlite+aiosqlite:///vidpipe.db`
- `storage.tmp_dir = ./tmp`

### 3.4 Required Environment Inputs

From `.env.example`, minimally:

- `GOOGLE_APPLICATION_CREDENTIALS`
- `VIDPIPE_GOOGLE_CLOUD__PROJECT_ID`

## 4. Database Layer

### 4.1 Engine and Safety Settings

`backend/vidpipe/db/engine.py` configures async SQLAlchemy + SQLite with:

- `journal_mode=WAL`
- `synchronous=FULL`
- `foreign_keys=ON`
- `busy_timeout=5000`

### 4.2 Initialization, Migrations, and Seeding

`backend/vidpipe/db/__init__.py`:

- Runs `Base.metadata.create_all()`.
- Runs idempotent `ALTER TABLE` migrations in `_run_migrations()`.
- Seeds default single user/settings rows (`DEFAULT_USER_ID`).

### 4.3 Core Tables

Main entities in `backend/vidpipe/db/models.py`:

- Generation:
  - `projects`
  - `scenes`
  - `keyframes`
  - `video_clips`
  - `pipeline_runs`
- Manifest/asset system:
  - `manifests`
  - `assets`
  - `manifest_snapshots`
  - `scene_manifests`
  - `scene_audio_manifests`
  - `asset_clean_references`
  - `asset_appearances`
- Quality mode:
  - `generation_candidates`
- User/settings:
  - `users`
  - `user_settings`
- Versioning:
  - `project_checkpoints`

### 4.4 Status Fields

Project statuses (observed across routes/orchestrator):

- `pending`
- `storyboarding`
- `keyframing`
- `video_gen`
- `stitching`
- `complete`
- `failed`
- `stopped`
- `staged`

Scene statuses (observed in pipeline/routes):

- `pending`
- `keyframes_done`
- `video_done`
- `failed`
- `timed_out`
- `removed`

Video clip statuses (observed in code paths):

- `polling`
- `complete`
- `failed`
- `timed_out`
- legacy checks also reference `completed` / `rai_filtered`

## 5. Orchestrator and Pipeline

### 5.1 State Machine and Resume Logic

`backend/vidpipe/orchestrator/state.py` defines resumable states and resume-point logic:

- `can_resume(status)` allows resume from active + interrupted states.
- `get_resume_step(status, completed_steps)` derives where to restart using persisted artifacts.

### 5.2 Pipeline Orchestrator

`backend/vidpipe/orchestrator/pipeline.py` (`run_pipeline`) coordinates execution:

1. Creates `PipelineRun`.
2. Computes completed step markers from DB (`_check_completed_steps`).
3. Runs step modules in order.
4. Persists timing/log metadata.
5. Handles:
   - user stop (`PipelineStopped`)
   - stage pausing (`PipelineStaged`)
   - failure capture (`project.status=failed`, `error_message`)
6. Auto-creates initial checkpoint when run reaches `complete` and no `head_sha`.

### 5.3 Selective Stage Execution (`run_through`)

Project-level `run_through` can stop execution at stage boundaries:

- `storyboard` → stage after storyboard
- `keyframes` → stage after keyframes
- `video` → stage after video generation

Reached boundary status becomes `staged`.

### 5.4 Step Modules

#### Storyboard (`pipeline/storyboard.py`)

- Uses LLM adapter abstraction (`get_adapter`).
- Supports:
  - standard `StoryboardOutput`
  - manifest-aware `EnhancedStoryboardOutput` (scene/audio manifests)
- Retries schema/JSON failures with decreasing temperature.
- Persists:
  - `project.style_guide`
  - `project.storyboard_raw`
  - `Scene` rows
  - optional `SceneManifest`/`SceneAudioManifest` rows
- Includes deterministic remapping for unrecognized `CHAR_*` tags.

#### Keyframes (`pipeline/keyframes.py`)

- Sequential only (continuity-first).
- Scene 0 start frame is generated from text.
- Scene N start frame inherits previous scene end frame.
- End frame is image-conditioned on start frame.
- Supports ComfyUI path for `qwen-fast`.
- Manifest-aware enhancements:
  - prompt rewriting
  - reference selection
  - face verification loop with escalating identity prompts
- Commits per scene for crash safety.
- Step function sets project status to `generating_video`; orchestrator normalizes it to `video_gen`.

#### Video Generation (`pipeline/video_gen.py`)

- Supports Veo and ComfyUI video models.
- Veo path:
  - persists operation ID before polling (idempotent resume)
  - handles content policy, transient operation errors, timeout/failure
  - multi-level policy remediation:
    - safety-prefixed prompt
    - end-keyframe safety regeneration + retry
  - transient retries with exponential backoff
- Quality mode:
  - multiple candidates (`candidate_count` up to 4)
  - scoring via `CandidateScoringService`
  - persists `GenerationCandidate`
  - winner updates `VideoClip.local_path`
- Post-generation CV analysis:
  - frame sampling + YOLO + ArcFace + CLIP + optional vision analysis
  - persists `AssetAppearance`, continuity metadata, and possible new entities
  - non-fatal on analysis failure

#### Stitcher (`pipeline/stitcher.py`)

- Gathers `complete` clips in scene order.
- If `crossfade_seconds == 0`: ffmpeg concat demuxer with stream copy.
- If `crossfade_seconds > 0`: ffmpeg `xfade` graph with re-encode.
- Writes `tmp/{project_id}/output/final.mp4`.
- Updates project to `complete` on success; stores failure reason otherwise.

## 6. API Surface

All routes are under `/api` in `backend/vidpipe/api/routes.py`.

### 6.1 Generation and Project Lifecycle

- `POST /api/generate`
- `GET /api/projects/{project_id}/status`
- `GET /api/projects/{project_id}`
- `GET /api/projects`
- `POST /api/projects/{project_id}/resume`
- `POST /api/projects/{project_id}/stop`
- `PATCH /api/projects/{project_id}` (title update)
- `DELETE /api/projects/{project_id}` (soft delete + disk cleanup)
- `GET /api/projects/{project_id}/download`
- `GET /api/keyframes/{keyframe_id}`
- `GET /api/clips/{clip_id}`
- `GET /api/metrics`
- `GET /api/health`

Notable validation and behavior:

- Aspect ratio limited to `16:9` / `9:16`.
- Model allowlists enforced server-side.
- Clip duration validated per selected video model.
- `run_through` validated (`storyboard`, `keyframes`, `video`).
- Audio toggles only allowed for audio-capable models.
- Cost estimation in metrics uses artifact-based accounting.

### 6.2 Forking and In-Place Editing

- `POST /api/projects/{project_id}/fork`
- `PATCH /api/projects/{project_id}/edit`

Forking features:

- Selective scene deletion/edits.
- Model and duration overrides.
- Keyframe clearing.
- Asset inheritance/copy for manifest projects.
- Invalidation logic to resume from correct stage.

Edit features:

- Terminal-state guard.
- Optional optimistic concurrency via `expected_sha`.
- Checkpoint creation with change metadata.

### 6.3 Manifest and Asset Management

- `POST /api/manifests`
- `POST /api/manifests/from-project`
- `GET /api/manifests`
- `GET /api/manifests/{manifest_id}`
- `PUT /api/manifests/{manifest_id}`
- `DELETE /api/manifests/{manifest_id}`
- `POST /api/manifests/{manifest_id}/duplicate`
- `POST /api/manifests/{manifest_id}/assets`
- `GET /api/manifests/{manifest_id}/assets`
- `PUT /api/assets/{asset_id}`
- `DELETE /api/assets/{asset_id}`
- `POST /api/assets/{asset_id}/upload`
- `GET /api/assets/{asset_id}/image`

Processing/extraction:

- `POST /api/manifests/{manifest_id}/process`
- `GET /api/manifests/{manifest_id}/progress`
- `POST /api/manifests/{manifest_id}/upload-video`
- `GET /api/manifests/{manifest_id}/extraction-progress`
- `POST /api/assets/{asset_id}/reprocess`

### 6.4 Candidate Quality Mode

- `GET /api/projects/{project_id}/scenes/{scene_idx}/candidates`
- `PUT /api/projects/{project_id}/scenes/{scene_idx}/candidates/{candidate_id}/select`

Manual selection endpoint updates both candidate selection flags and `VideoClip.local_path` so stitcher uses the selected candidate.

### 6.5 Settings

- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/settings/models`

Settings include enabled/default models, GCP values, ComfyUI config, and Ollama config.

### 6.6 PipeSVN / Checkpoints

- `GET /api/projects/{project_id}/checkpoints`
- `GET /api/projects/{project_id}/checkpoints/{sha}`
- `GET /api/projects/{project_id}/checkpoints/{sha}/diff`
- `POST /api/projects/{project_id}/checkpoints`
- `DELETE /api/projects/{project_id}/checkpoints/{sha}`
- `POST /api/projects/{project_id}/revert`

### 6.7 Regeneration and Edit-Mode Utilities

- `POST /api/projects/{project_id}/scenes/{scene_idx}/regenerate`
- `POST /api/projects/{project_id}/scenes/{scene_idx}/regenerate-text`
- `POST /api/projects/{project_id}/generate-scene-fields`
- `POST /api/projects/{project_id}/generate-new-scene`
- `POST /api/projects/{project_id}/regenerate`

### 6.8 Manual Asset Replace/Delete

- `PUT /api/projects/{project_id}/scenes/{scene_idx}/keyframes/{position}`
- `PUT /api/projects/{project_id}/scenes/{scene_idx}/clip`
- `DELETE /api/projects/{project_id}/scenes/{scene_idx}/clip`
- `DELETE /api/projects/{project_id}/scenes/{scene_idx}/keyframes/{position}`

Each replacement/deletion operation creates a checkpoint.

## 7. Manifesting and CV Workflows

### 7.1 Manifesting Engine

`backend/vidpipe/services/manifesting_engine.py` pipeline stages:

1. `contact_sheet`
2. `yolo_detection`
3. `face_matching`
4. `reverse_prompting`
5. `finalizing`

Finalization:

- Reassigns manifest tags by type.
- Sets manifest `status = READY`.
- Updates asset count and progress summary.

### 7.2 Worker Task Tracking

`backend/vidpipe/workers/processing_tasks.py` uses an in-memory `TASK_STATUS` map:

- `manifest_{manifest_id}` for manifest processing
- `extract_{manifest_id}` for video frame extraction

Progress endpoints read this memory map first, then DB status fallback.

### 7.3 CV Analysis and Entity Enrichment

`CVAnalysisService` composes:

- Frame sampling (`frame_sampler.py`)
- YOLO detection (`cv_detection.py`)
- ArcFace matching (`face_matching.py`)
- CLIP embeddings (`clip_embedding_service.py`)
- Optional LLM semantic assessment

Outputs are used to:

- Persist `AssetAppearance` rows.
- Update `SceneManifest.cv_analysis_json` and continuity score.
- Detect/register new assets via `entity_extraction.py`.

### 7.4 Reference Selection and Prompt Rewriting

- `reference_selection.py` chooses up to 3 references based on shot type/roles.
- `prompt_rewriter.py` adapts keyframe/video prompts using manifests, continuity context, and assets.

## 8. Storage Layout

### 8.1 Project Artifacts (`FileManager`)

Per project:

- `tmp/{project_id}/keyframes/`
- `tmp/{project_id}/clips/`
- `tmp/{project_id}/output/`

Naming:

- Standard pipeline writes deterministic names (`scene_{idx}_start.png`, `scene_{idx}.mp4`).
- Edit/regeneration/upload helpers often write versioned names with short UUID suffixes.

### 8.2 Manifest Artifacts

Per manifest:

- `tmp/manifests/{manifest_id}/uploads/`
- `tmp/manifests/{manifest_id}/crops/`
- `tmp/manifests/{manifest_id}/clean_sheets/` (optional clean-sheet flow)
- `tmp/manifests/{manifest_id}/source_video.*`
- `tmp/manifests/{manifest_id}/contact_sheet.jpg`

## 9. LLM/Provider Abstraction

`backend/vidpipe/services/llm/` provides adapter routing:

- `gemini-*` and default fallback -> `VertexAIAdapter`
- `ollama/*` -> `OllamaAdapter`

`vertex_client.py` caches clients per location and routes some Gemini 3 models to `global`.

## 10. CLI

Commands in `backend/vidpipe/cli/commands.py`:

- `generate`
- `resume`
- `status`
- `list`
- `stitch`

CLI reuses orchestrator and core pipeline modules, including model validation and DB-backed status tracking.

## 11. Operational Notes

### 11.1 Authentication Model

- No user auth/multi-tenant auth layer.
- Backend assumes single-user mode with one seeded default user/settings row.

### 11.2 Dependency Reality vs Declared Requirements

`backend/pyproject.toml` and `backend/requirements.txt` include only core packages.

Code paths additionally import optional/heavy dependencies, including:

- `tenacity`
- `numpy`
- `opencv-python` (`cv2`)
- `ultralytics`
- `insightface`
- `transformers`
- `torch`
- `rembg`
- `ollama`

For full feature coverage (manifesting/CV/quality/Ollama/clean-sheets), these must be installed.

### 11.3 Reliability Characteristics

- DB commits happen at step boundaries and often per scene.
- Video operation IDs are persisted before poll loops.
- Resume logic is artifact-aware and avoids redoing completed work.
- Stopping is cooperative (status checks between long operations/polls).

## 12. Known Caveats in Current Implementation

1. `routes.py` is very large and monolithic (~4.9k lines), making ownership and change-risk high.
2. Regeneration helper code contains legacy references:
   - `app_settings.models.keyframe_image`
   - `app_settings.models.video_generator`
   - `from vidpipe.services.vertex_client import get_client`
   These names/functions do not exist in current config/client modules and can break certain edit/regeneration paths.
3. Resume completion checks in orchestrator currently look for clip statuses including `completed`/`rai_filtered`, while primary generation writes `complete`; this mismatch can affect resume inference in edge cases.

## 13. Tests and Coverage

Current tests in `backend/tests/` are primarily integration/smoke style:

- `test_ollama_storyboard_e2e.py`
- `comfyui/test_txt2img.py`
- `comfyui/test_i2v.py`
- `comfyui/test_flf2v.py`

There is limited unit coverage for core orchestrator/state/DB/service logic.

## 14. Recommended Reading Order for Contributors

1. `backend/vidpipe/api/app.py`
2. `backend/vidpipe/api/routes.py`
3. `backend/vidpipe/orchestrator/pipeline.py`
4. `backend/vidpipe/pipeline/*`
5. `backend/vidpipe/db/models.py`
6. `backend/vidpipe/services/*` (manifesting, CV, llm adapters, checkpoints)
