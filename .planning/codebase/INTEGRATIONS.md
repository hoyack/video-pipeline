# External Integrations

**Analysis Date:** 2026-02-16

## APIs & External Services

**Google Vertex AI (LLM/Generative Models):**
- **Storyboard Generation** - Gemini LLM for scene breakdown
  - Models: `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3-pro-preview`
  - SDK: `google-genai` (Python)
  - Location: `vidpipe/services/vertex_client.py`, `vidpipe/pipeline/storyboard.py`
  - Auth: Via Application Default Credentials (ADC), path set by `GOOGLE_APPLICATION_CREDENTIALS` env var
  - Client method: `get_vertex_client(location)` returns cached `genai.Client` instance

- **Image Generation** - Imagen and Gemini models for keyframe images
  - Imagen models: `imagen-3.0-generate-002`, `imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001`, `imagen-4.0-ultra-generate-001`
  - Gemini models: `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`
  - Location: `vidpipe/pipeline/keyframes.py`
  - API: `client.models.generate_images()` for Imagen, `client.models.generate_content()` for Gemini
  - Retry policy: Exponential backoff with max 7 attempts (Keyframe generation)

- **Image-Conditioned Generation** - Gemini models for end-frame generation using start-frame reference
  - Models: `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`
  - Location: `vidpipe/pipeline/keyframes.py`
  - Maps Imagen usage to Gemini conditioning models via `IMAGE_CONDITIONED_MAP`

- **Video Generation** - Veo models for clip generation
  - Models: `veo-2.0-generate-001`, `veo-3.0-generate-001`, `veo-3.0-fast-generate-001`, `veo-3.1-generate-preview`, `veo-3.1-generate-001`, `veo-3.1-fast-generate-preview`, `veo-3.1-fast-generate-001`
  - Location: `vidpipe/pipeline/video_gen.py`
  - API: Long-running operations via `client.models.generate_videos()` and polling
  - Poll strategy: Configurable interval (default 15s) and max polls (default 40)
  - Audio support: All models except `veo-2.0-generate-001` support `generate_audio: bool`
  - Constraints: Model-specific clip durations (Veo 2: [5,6,7,8]s; Veo 3/3.1: [4,6,8]s)

## Data Storage

**Databases:**
- **SQLite with async support**
  - Type: SQLite local file database
  - Connection: `sqlite+aiosqlite:///vidpipe.db` (configurable via `VIDPIPE_STORAGE__DATABASE_URL`)
  - ORM: SQLAlchemy 2.0+ with async sessions
  - Client: `aiosqlite` for async operations
  - Models: `vidpipe/db/models.py` - Project, Scene, Keyframe, VideoClip
  - Configuration: WAL mode, FULL synchronous mode for crash safety, foreign keys enabled, 5s busy timeout
  - Session management: `async_session` factory with `expire_on_commit=False` to prevent greenlet errors

**File Storage:**
- **Local Filesystem**
  - Keyframes: `{tmp_dir}/{project_id}/keyframes/` - PNG images for scene start/end frames
  - Video Clips: `{tmp_dir}/{project_id}/clips/` - MP4 video files per scene
  - Output: `{tmp_dir}/{project_id}/output/` - Final assembled video
  - Manager: `vidpipe/services/file_manager.py::FileManager` class
  - Default tmp_dir: `./tmp` (configurable via `VIDPIPE_STORAGE__TMP_DIR`)
  - Security: Path traversal protection on project_id resolution

**Cloud Storage (Optional):**
- **Google Cloud Storage**
  - Veo API returns video clips with `gcs_uri` field (gs:// URLs)
  - Download helper: `_download_from_gcs()` in `vidpipe/pipeline/video_gen.py`
  - Conversion: `gs://bucket/path` → `https://storage.googleapis.com/bucket/path` for HTTP download
  - Usage: Optional; clips can be stored locally instead via `local_path`

**Caching:**
- Client caching: Per-location Vertex AI client cache in `_clients` dict (module-level singleton)
- Database: No explicit caching layer; SQLAlchemy session caching

## Authentication & Identity

**Auth Provider:**
- **Google Cloud ADC (Application Default Credentials)**
  - Implementation: `google-genai` SDK handles ADC automatically
  - Credential source: `GOOGLE_APPLICATION_CREDENTIALS` env var (path to service account JSON)
  - Location: `vidpipe/services/vertex_client.py::get_vertex_client()`
  - Setup: Environment variables set by `load_dotenv()` before client creation
  - Project ID: `VIDPIPE_GOOGLE_CLOUD__PROJECT_ID` (required)

**API Security:**
- CORS configured for Vite dev server: `http://localhost:5173`
- Location: `vidpipe/api/app.py::CORSMiddleware`

## Monitoring & Observability

**Error Tracking:**
- None detected - Errors logged to application logs

**Logs:**
- Approach: Python `logging` module with per-module loggers
- Key loggers: `vidpipe.api.app`, `vidpipe.pipeline.storyboard`, `vidpipe.pipeline.keyframes`, `vidpipe.pipeline.video_gen`, `vidpipe.pipeline.stitcher`
- Logging in critical areas: Startup validation, API exceptions, retry attempts, video polling, file operations

**Telemetry:**
- None detected

## CI/CD & Deployment

**Hosting:**
- Not detected in codebase (no Dockerfile, Cloud Run configs, etc.)
- Expected: Manual or GitHub Actions (if present elsewhere)

**CI Pipeline:**
- Not detected - No GitHub Actions or other CI config found

**Deployment Targets:**
- Development: Local (`python -m vidpipe.api` runs uvicorn on 0.0.0.0:8000)
- Database: Local SQLite file
- Frontend: Served statically from `frontend/dist` when built, or via Vite dev server on localhost:5173

## Environment Configuration

**Required Environment Variables:**
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account JSON file
- `VIDPIPE_GOOGLE_CLOUD__PROJECT_ID` - GCP project ID (no default)

**Optional Environment Variables:**
- `VIDPIPE_GOOGLE_CLOUD__LOCATION` - GCP region (default: `us-central1`)
- `VIDPIPE_STORAGE__DATABASE_URL` - SQLite connection string (default: `sqlite+aiosqlite:///vidpipe.db`)
- `VIDPIPE_STORAGE__TMP_DIR` - Temp directory for artifacts (default: `./tmp`)
- `VIDPIPE_PIPELINE__*` - Override pipeline settings (max_scenes, retry_attempts, polling intervals, etc.)
- `VIDPIPE_SERVER__*` - Override server settings (host, port)

**Configuration Sources (Priority Order):**
1. Environment variables (prefix: `VIDPIPE_`)
2. `.env` file
3. `config.yaml` YAML file
4. Field defaults in pydantic models

**Secrets Location:**
- `.env` file (git-ignored) - Contains `GOOGLE_APPLICATION_CREDENTIALS` path
- GCP service account JSON (referenced by env var, not in repo)

## Webhooks & Callbacks

**Incoming:**
- None detected - API is request/response only

**Outgoing:**
- None detected - No webhook notifications to external services

## Model Selection & Fallback

**Per-Project Model Override:**
- Backend allows per-project model selection via `Project` table columns:
  - `text_model` - LLM for storyboard (overrides `settings.models.storyboard_llm`)
  - `image_model` - Image generation model (overrides `settings.models.image_gen`)
  - `video_model` - Video generation model (overrides `settings.models.video_gen`)
  - `audio_enabled` - Optional audio in video generation

**Default Models (from `config.yaml`):**
```yaml
models:
  storyboard_llm: "gemini-2.5-flash"
  image_gen: "imagen-4.0-fast-generate-001"
  image_conditioned: "gemini-2.5-flash-image"
  video_gen: "veo-3.1-fast-generate-001"
```

**Model Location Routing:**
- Some models require global endpoint: `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-3-pro-image-preview`
- Function: `location_for_model(model_id)` in `vidpipe/services/vertex_client.py` handles routing
- Default location: `settings.google_cloud.location` (e.g., `us-central1`)

## Retry & Resilience

**Video Generation (Veo):**
- Long-running operations with polling
- Poll interval: 15s (configurable)
- Max polls: 40 (configurable)
- Timeout detection: After max polls, operation marked as timed out
- Retry: Escalating content-policy safety prompts on policy violations

**Image Generation:**
- Exponential backoff: 2^attempt * 2-120s + random 0-5s
- Max attempts: 7
- Retriable errors: ServerError (5xx), ClientError with code 429 (rate limit), connection/timeout errors

**Storyboard Generation:**
- Retry: max_attempts: 5, base_delay: 2s (exponential)
- Triggered by: Invalid JSON output, pydantic validation errors

## Payment & Quotas

**Pricing (from MEMORY.md):**
- Veo 2: $0.35/s (silent)
- Veo 3: $0.40/s (silent)
- Veo 3 Fast: $0.15/s (silent)
- Veo 3.1: $0.40/s (silent), $0.40/s (audio)
- Veo 3.1 Fast: $0.10/s (silent), $0.15/s (audio)

**Quotas:**
- Not explicitly enforced in code; reliant on GCP quotas and billing

---

*Integration audit: 2026-02-16*
