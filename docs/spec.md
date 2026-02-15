# Viral Video Generation Pipeline — Project Specification

## 1. Overview

A local Python CLI pipeline that transforms a text prompt into a stitched, multi-scene AI-generated video. The system uses Google Vertex AI / Gemini APIs for all generative work: **Gemini 3 Pro** for LLM text (storyboarding), **Nano Banana Pro** (`gemini-3-pro-image-preview`) for keyframe image generation, and **Veo 3.1** (`veo-3.1-generate-001`) for 4-second video clip generation with first/last frame control. Completed clips are stitched locally via **ffmpeg** into a single output video. All state lives in a local **SQLite** database via **SQLAlchemy**, and binary assets (images, clips, final video) are stored in a local `tmp/` artifacts directory.

The project also exposes a lightweight **FastAPI** server so the pipeline can be triggered and monitored via HTTP, making it embeddable in larger automation workflows (n8n, Make, etc.).

---

## 2. Goals & Non-Goals

### Goals

- Accept a text prompt (or script) and produce a cohesive multi-scene short video (target: 15–60 seconds).
- Storyboard the script into discrete scenes with structured output (scene descriptions, keyframe prompts, motion prompts).
- Generate start/end keyframe images per scene with visual continuity (scene N's end frame = scene N+1's start frame).
- Generate 4–8 second video clips per scene using Veo 3.1's first-frame + last-frame interpolation.
- Poll long-running Veo operations with backoff and timeout handling.
- Stitch all clips into a single MP4 using ffmpeg (with optional crossfade transitions).
- Persist all pipeline state, metadata, and asset references in a local SQLite database.
- Save all binary artifacts (keyframe PNGs, clip MP4s, final MP4) to a local `tmp/` directory.
- Provide both a CLI interface (`python -m vidpipe generate "prompt"`) and a FastAPI endpoint (`POST /generate`).
- Support retry/resume — if the pipeline crashes mid-run, restarting should pick up from the last completed step.

### Non-Goals (v1)

- Cloud storage upload (GCS, S3) — future phase.
- Audio narration / TTS overlay — future phase.
- Real-time streaming / WebSocket progress — future phase.
- Multi-user / auth on the FastAPI server.
- CrewAI orchestration — noted as a future option for structured output, but v1 uses Gemini structured output directly.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / FastAPI                        │
│  python -m vidpipe generate "prompt"                        │
│  POST /generate  { "prompt": "...", "style": "cinematic" }  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                     │
│  Manages state machine:                                     │
│  STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE    │
└──────┬──────────┬──────────┬───────────┬────────────────────┘
       │          │          │           │
       ▼          ▼          ▼           ▼
  ┌─────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐
  │ Gemini  │ │ Nano   │ │ Veo 3.1 │ │ ffmpeg  │
  │ 3 Pro   │ │ Banana │ │ Video   │ │ Stitch  │
  │ (LLM)   │ │ Pro    │ │ Gen     │ │         │
  │         │ │(Images)│ │         │ │         │
  └────┬────┘ └───┬────┘ └────┬────┘ └────┬────┘
       │          │           │            │
       ▼          ▼           ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                   SQLAlchemy + SQLite                        │
│  projects, scenes, keyframes, video_clips, pipeline_runs    │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Local Filesystem: tmp/{project_id}/            │
│  keyframes/  clips/  output/                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Data Model (SQLAlchemy)

### 4.1 `Project`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Auto-generated project ID |
| `prompt` | Text | Original user prompt / script |
| `style` | String(50) | Visual style (cinematic, anime, documentary, etc.) |
| `aspect_ratio` | String(10) | `16:9`, `9:16`, `1:1` |
| `target_clip_duration` | Integer | Seconds per clip (default: 4) |
| `status` | Enum | `pending`, `storyboarding`, `keyframing`, `generating_video`, `stitching`, `complete`, `failed` |
| `style_guide` | JSON | LLM-generated style guide (color palette, camera style, etc.) |
| `storyboard_raw` | JSON | Raw storyboard JSON from LLM |
| `output_path` | String | Path to final stitched video |
| `error_message` | Text | Last error if `status=failed` |
| `created_at` | DateTime | |
| `updated_at` | DateTime | |

### 4.2 `Scene`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | |
| `project_id` | UUID (FK→Project) | |
| `scene_index` | Integer | Ordered position (0-based) |
| `scene_description` | Text | What happens in this scene |
| `start_frame_prompt` | Text | Image gen prompt for opening keyframe |
| `end_frame_prompt` | Text | Image gen prompt for closing keyframe |
| `video_motion_prompt` | Text | Describes motion/action for Veo |
| `transition_notes` | Text | How this scene connects to next |
| `status` | Enum | `pending`, `keyframes_generating`, `keyframes_done`, `video_generating`, `video_done`, `failed` |

### 4.3 `Keyframe`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | |
| `scene_id` | UUID (FK→Scene) | |
| `position` | Enum | `start` or `end` |
| `prompt_used` | Text | The prompt that was sent |
| `file_path` | String | Local path to saved PNG |
| `mime_type` | String(20) | `image/png` |
| `source` | Enum | `generated` or `inherited` (reused from previous scene's end frame) |
| `created_at` | DateTime | |

### 4.4 `VideoClip`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | |
| `scene_id` | UUID (FK→Scene) | |
| `operation_name` | String | Vertex AI long-running operation ID |
| `status` | Enum | `submitted`, `polling`, `complete`, `rai_filtered`, `failed`, `timed_out` |
| `gcs_uri` | String | GCS URI returned by Veo (if applicable) |
| `local_path` | String | Local path to downloaded MP4 |
| `duration_seconds` | Float | Actual clip duration |
| `poll_count` | Integer | Number of status polls performed |
| `error_message` | Text | |
| `created_at` | DateTime | |
| `completed_at` | DateTime | |

### 4.5 `PipelineRun`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | |
| `project_id` | UUID (FK→Project) | |
| `started_at` | DateTime | |
| `completed_at` | DateTime | |
| `total_duration_seconds` | Float | Wall-clock pipeline time |
| `total_api_cost_estimate` | Float | Estimated cost based on token/second pricing |
| `log` | JSON | Array of `{step, timestamp, message}` entries |

---

## 5. Pipeline Steps (Detail)

### Step 1: Storyboard Generation

**Model:** `gemini-3-pro` via Vertex AI (or Gemini API `generateContent`)

**Input:** User prompt/script, style, aspect ratio

**Approach — Structured Output:**
- Use Gemini's `responseMimeType: "application/json"` with a JSON schema constraint to force structured output directly from the model.
- Alternatively, if more complex orchestration is needed later, this can be swapped for a CrewAI agent with a Pydantic output parser.

**System Prompt (condensed):**
```
You are a cinematic storyboard director. Given a script, break it into
visual scenes (each = {target_clip_duration} seconds).

For each scene provide:
- scene_description: what happens
- start_frame_prompt: detailed image prompt (include style, aspect ratio)
- end_frame_prompt: closing frame prompt (must show progression)
- video_motion_prompt: motion/action description for video generation
- transition_notes: continuity bridge to next scene

Rules:
- Start frame of scene N+1 MUST match end frame of scene N
- Consistent visual style across all scenes
- Include camera direction (pan, zoom, dolly, static, etc.)
- NO text overlays in visuals
```

**Output Schema:**
```json
{
  "style_guide": {
    "visual_style": "string",
    "color_palette": "string",
    "camera_style": "string"
  },
  "scenes": [
    {
      "scene_index": 0,
      "scene_description": "string",
      "start_frame_prompt": "string",
      "end_frame_prompt": "string",
      "video_motion_prompt": "string",
      "transition_notes": "string"
    }
  ]
}
```

**DB writes:** Create `Project`, parse scenes into `Scene` rows.

---

### Step 2: Keyframe Generation (Sequential Loop)

**Model:** `gemini-3-pro-image-preview` (Nano Banana Pro) via Vertex AI

**Logic:**
```
for scene in scenes (ordered by scene_index):
    if scene_index == 0:
        generate start keyframe from start_frame_prompt
    else:
        reuse previous scene's end keyframe as this scene's start keyframe

    generate end keyframe:
        send start keyframe image + end_frame_prompt to model
        ("Using this image as reference, generate the same scene 4 seconds
         later. {end_frame_prompt}. Maintain visual style, lighting, composition.")

    save both keyframes to tmp/{project_id}/keyframes/
    create Keyframe DB records
    update Scene status
```

**API Call Pattern (image generation):**
```python
from google import genai
from google.genai import types

client = genai.Client()  # configured for Vertex AI

# Text-only generation (first scene start frame)
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[start_frame_prompt],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    ),
)

# Image-conditioned generation (end frames, subsequent start frames)
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        types.Part.from_bytes(data=start_image_bytes, mime_type="image/png"),
        types.Part.from_text(text=conditioning_prompt),
    ],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    ),
)
```

**Rate Limiting:** Implement exponential backoff with max 5 retries per image. Nano Banana Pro has rate limits (~10 req/min on free tier). Add a configurable delay between calls (default: 3 seconds).

**File Storage:** `tmp/{project_id}/keyframes/scene_{idx}_start.png`, `scene_{idx}_end.png`

---

### Step 3: Video Clip Generation (Parallel-capable with Polling)

**Model:** `veo-3.1-generate-001` via Vertex AI `predictLongRunning`

**For each scene**, submit a video generation job with first frame + last frame:

```python
from google import genai
from google.genai.types import GenerateVideosConfig, Image

client = genai.Client()  # configured for Vertex AI

operation = client.models.generate_videos(
    model="veo-3.1-generate-001",
    prompt=scene.video_motion_prompt,
    image=Image(
        image_bytes=start_frame_bytes,
        mime_type="image/png",
    ),
    config=GenerateVideosConfig(
        aspect_ratio="16:9",
        last_frame=Image(
            image_bytes=end_frame_bytes,
            mime_type="image/png",
        ),
        # output_gcs_uri=optional_gcs_bucket,  # omit to get bytes in response
    ),
)
```

**Polling Strategy:**
```
poll_interval = 15 seconds
max_polls = 40 (~10 minutes timeout per clip)

while not operation.done:
    sleep(poll_interval)
    operation = client.operations.get(operation)
    poll_count += 1
    if poll_count > max_polls:
        mark as timed_out
        break

if operation.response:
    video = operation.result.generated_videos[0]
    # Download video bytes and save locally
```

**Concurrency Option:** Since Veo jobs are async, multiple scenes can be submitted in parallel. Use `asyncio.gather()` or a thread pool to submit all jobs, then poll all concurrently. However, start with sequential processing in v1 for simplicity and to respect rate limits.

**RAI Filtering:** Veo may filter content for safety. If `raiMediaFilteredCount > 0` and no videos returned, mark the clip as `rai_filtered` and log the issue. The pipeline should continue with remaining scenes and note the gap.

**File Storage:** `tmp/{project_id}/clips/scene_{idx}.mp4`

**DB writes:** Create `VideoClip` record on submission, update during polling, finalize on completion.

---

### Step 4: Video Stitching (ffmpeg)

**Once all clips are complete (or all non-filtered clips):**

**Approach: ffmpeg concat demuxer** (preferred since all clips share the same codec from Veo):

```python
import subprocess

def stitch_clips(clip_paths: list[str], output_path: str, crossfade_seconds: float = 0.0):
    """Concatenate video clips using ffmpeg."""

    if crossfade_seconds == 0:
        # Simple concat demuxer (no re-encode, fastest)
        list_file = write_concat_list(clip_paths)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ], check=True)
    else:
        # Concat filter with crossfade (requires re-encode)
        filter_parts = []
        inputs = []
        for i, path in enumerate(clip_paths):
            inputs.extend(["-i", path])

        # Build xfade filter chain
        # [0:v][1:v]xfade=transition=fade:duration=0.5:offset=3.5[v01];
        # [v01][2:v]xfade=...
        prev = "0:v"
        for i in range(1, len(clip_paths)):
            out_label = f"v{i:02d}"
            offset = (i * target_duration) - (crossfade_seconds * i)
            filter_parts.append(
                f"[{prev}][{i}:v]xfade=transition=fade:"
                f"duration={crossfade_seconds}:offset={offset}[{out_label}]"
            )
            prev = out_label

        filter_complex = ";".join(filter_parts)
        subprocess.run([
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{prev}]",
            "-vsync", "vfr",
            output_path
        ], check=True)
```

**Audio handling:** Veo 3.1 generates native audio. When using concat demuxer, audio streams are preserved. When using crossfade filter, audio crossfading needs additional filter chains (`acrossfade`).

**Output:** `tmp/{project_id}/output/final.mp4`

---

## 6. API Surface

### 6.1 CLI

```bash
# Generate from prompt
python -m vidpipe generate "A 30-second explainer about quantum computing" \
    --style cinematic \
    --aspect-ratio 16:9 \
    --clip-duration 4

# Resume a failed/incomplete project
python -m vidpipe resume <project_id>

# Check status
python -m vidpipe status <project_id>

# List all projects
python -m vidpipe list

# Re-stitch with crossfade
python -m vidpipe stitch <project_id> --crossfade 0.5
```

### 6.2 FastAPI Endpoints

```
POST   /api/generate          — Start new pipeline run
  Body: { "prompt": "...", "style": "cinematic", "aspect_ratio": "16:9", "clip_duration": 4 }
  Returns: { "project_id": "uuid", "status": "storyboarding" }

GET    /api/projects           — List all projects
GET    /api/projects/{id}      — Get project detail + scenes + clips
GET    /api/projects/{id}/status — Lightweight status poll
POST   /api/projects/{id}/resume — Resume failed pipeline
GET    /api/projects/{id}/download — Download final MP4

GET    /api/health             — Health check
```

The FastAPI server runs the pipeline in a background task (`BackgroundTasks` or a task queue). The generate endpoint returns immediately with the project ID so the caller can poll for status.

---

## 7. Configuration

```yaml
# config.yaml (or .env)
google_cloud:
  project_id: "hoyack-1577568661630"
  location: "us-central1"
  use_vertex_ai: true
  # credentials: path to service account JSON or use ADC

models:
  storyboard_llm: "gemini-3-pro"
  image_gen: "gemini-3-pro-image-preview"
  video_gen: "veo-3.1-generate-001"

pipeline:
  default_style: "cinematic"
  default_aspect_ratio: "16:9"
  default_clip_duration: 4        # seconds
  max_scenes: 15                  # safety cap
  image_gen_delay: 3              # seconds between image API calls
  video_poll_interval: 15         # seconds
  video_poll_max: 40              # max polls before timeout
  video_gen_concurrency: 1        # parallel Veo jobs (1=sequential)
  crossfade_seconds: 0.0          # 0 = hard cuts
  retry_max_attempts: 5
  retry_base_delay: 2             # seconds (exponential backoff)

storage:
  database_url: "sqlite:///vidpipe.db"
  tmp_dir: "./tmp"

server:
  host: "0.0.0.0"
  port: 8000
```

---

## 8. Project Structure

```
vidpipe/
├── __main__.py              # CLI entry point (click or typer)
├── cli.py                   # CLI commands
├── server.py                # FastAPI app
├── config.py                # Pydantic settings model, loads config
├── db/
│   ├── __init__.py
│   ├── engine.py            # SQLAlchemy engine & session factory
│   ├── models.py            # ORM models (Project, Scene, Keyframe, VideoClip, PipelineRun)
│   └── migrations/          # Alembic migrations (optional for v1)
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py      # Main pipeline state machine
│   ├── storyboard.py        # Step 1: LLM storyboard generation
│   ├── keyframes.py         # Step 2: Image generation loop
│   ├── video_gen.py         # Step 3: Veo job submission + polling
│   └── stitcher.py          # Step 4: ffmpeg concatenation
├── services/
│   ├── __init__.py
│   ├── vertex_client.py     # Google GenAI SDK wrapper (Vertex mode)
│   └── file_manager.py      # Local file I/O helpers
├── schemas/
│   ├── __init__.py
│   ├── storyboard.py        # Pydantic models for storyboard structured output
│   ├── api.py               # FastAPI request/response schemas
│   └── enums.py             # Status enums shared across DB + API
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── Dockerfile               # Optional containerized deployment
└── README.md
```

---

## 9. Dependencies

```
# Core
fastapi>=0.115.0
uvicorn>=0.30.0
sqlalchemy>=2.0
pydantic>=2.0
pydantic-settings>=2.0

# Google AI
google-genai>=1.0.0          # Unified Google GenAI SDK (supports Vertex AI mode)

# CLI
typer>=0.12.0
rich>=13.0                   # Pretty terminal output

# Image handling
Pillow>=10.0

# HTTP (fallback if not using SDK for some calls)
httpx>=0.27.0

# Utils
python-dotenv>=1.0
pyyaml>=6.0
uuid7>=0.1                   # Optional: time-ordered UUIDs
```

**System Dependencies:**
- `ffmpeg` (must be on PATH)
- Python 3.11+

---

## 10. Key Design Decisions

### 10.1 Why Not CrewAI for v1?

The n8n workflow used direct LLM calls with JSON output. Gemini 3 Pro natively supports `responseMimeType: "application/json"` which gives us structured output without an orchestration framework. CrewAI adds complexity and is better suited for multi-agent workflows. **For v2**, if we want a "director" agent that iterates on storyboard quality, critiques keyframes, and retries with feedback, CrewAI becomes valuable. The `storyboard.py` module is designed to be swappable.

### 10.2 Why Sequential Keyframe Generation?

Visual continuity requires that scene N's end frame informs scene N+1's start frame. This is inherently sequential. The n8n workflow used a manual recursive loop for exactly this reason. We maintain the same pattern: a simple `for` loop with state accumulation.

### 10.3 Why ffmpeg Over MoviePy?

MoviePy wraps ffmpeg but adds Python overhead and re-encoding by default. Since all Veo clips share the same codec/resolution, `ffmpeg -f concat` with stream copy is nearly instant and lossless. We only use the filter path when crossfades are requested.

### 10.4 Vertex AI vs Gemini API

The n8n workflow used Vertex AI endpoints (project-scoped URLs with service account auth). We continue with Vertex AI via the `google-genai` SDK in Vertex mode (`GOOGLE_GENAI_USE_VERTEXAI=True`). This provides consistent auth via ADC/service accounts and avoids API key management.

### 10.5 Resume / Idempotency

Every step checks the DB before executing. If keyframes already exist for a scene, skip generation. If a Veo operation was submitted but not polled to completion, resume polling. This makes the pipeline crash-safe and restartable.

---

## 11. Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| Storyboard LLM returns invalid JSON | Retry up to 3 times with temperature adjustment. If still invalid, fail project with error. |
| Image generation returns no image | Retry with exponential backoff (max 5). If persistent, fail the scene. |
| Image generation returns `thought` blocks only | Filter for `inlineData` parts, skip `thought` parts (matching n8n logic). |
| Veo RAI filter blocks a clip | Mark clip as `rai_filtered`, continue pipeline, note gap in final manifest. Optionally retry with softened prompt. |
| Veo poll timeout (>10 min) | Mark as `timed_out`, continue with other clips. |
| ffmpeg not found | Fail early at startup with clear error message. |
| Disk space low | Check available space before pipeline start. Warn if <2GB free. |
| Partial completion (3/5 scenes done, crash) | `resume` command picks up from first incomplete scene. |

---

## 12. Cost Estimation

| Component | Price | Per-Scene Cost | 5-Scene Project |
|-----------|-------|---------------|-----------------|
| Gemini 3 Pro (storyboard) | ~$0.01/1K tokens | ~$0.02 | $0.02 |
| Nano Banana Pro (keyframes) | $0.039/image | ~$0.078 (2 images) | $0.35 (9 images*) |
| Veo 3.1 (4s clip) | $0.75/second | $3.00 | $15.00 |
| **Total** | | | **~$15.37** |

*Scene 0 generates 2 images; scenes 1–4 generate 1 each (start is inherited) = 2 + 4 = 6, but end frames use image-conditioned generation which still counts = 9 total image gen calls.

---

## 13. Future Enhancements (v2+)

- **CrewAI Orchestration:** Director agent reviews storyboard quality, keyframe consistency, and video output — retries with feedback loops.
- **Audio Narration:** TTS overlay (Google Cloud TTS or ElevenLabs) synced to scenes.
- **Background Music:** AI-generated or library music mixing.
- **GCS / S3 Upload:** Push final assets to cloud storage for distribution.
- **Supabase Integration:** Match the n8n TODO notes — store records and assets in Supabase for web frontend access.
- **Parallel Video Generation:** Submit all Veo jobs concurrently once keyframes are complete.
- **Scene Extension:** Use Veo 3.1's "extend" feature to create longer scenes from the same seed clip.
- **Reference Images:** Use Veo 3.1's multi-reference-image feature for stronger character/style consistency.
- **Web Dashboard:** React frontend showing pipeline progress, previewing keyframes, playing clips.
- **Webhook Callbacks:** Notify external systems (n8n, Slack) when pipeline completes.
- **Prompt Templates:** Library of proven viral video prompt structures (hook → story → CTA).

---

## 14. Development Phases

### Phase 1: Foundation (MVP)
- [ ] Project scaffolding, config, DB models
- [ ] Vertex AI client wrapper (auth, retries)
- [ ] Storyboard generation with structured output
- [ ] CLI `generate` command (storyboard only)
- [ ] Unit tests for DB models and config

### Phase 2: Keyframe Loop
- [ ] Image generation service (Nano Banana Pro)
- [ ] Sequential keyframe loop with frame inheritance
- [ ] File manager (save/load images from tmp/)
- [ ] Resume logic for partial keyframe completion
- [ ] Integration test with real API

### Phase 3: Video Generation
- [ ] Veo 3.1 job submission service
- [ ] Polling loop with backoff and timeout
- [ ] Video download and local storage
- [ ] RAI filter handling
- [ ] Integration test with real API

### Phase 4: Stitching & Output
- [ ] ffmpeg concat (hard cuts)
- [ ] ffmpeg crossfade (optional)
- [ ] Output manifest generation
- [ ] CLI `stitch` command
- [ ] End-to-end integration test

### Phase 5: API & Polish
- [ ] FastAPI server with background tasks
- [ ] Status polling endpoint
- [ ] Download endpoint
- [ ] `resume` and `list` CLI commands
- [ ] README, Docker support
- [ ] Cost tracking in PipelineRun