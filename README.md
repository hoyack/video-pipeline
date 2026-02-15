# vidpipe

AI-powered multi-scene video generation pipeline. Takes a text prompt and produces a cohesive short video with visual continuity across scenes — fully automated, crash-safe, and resumable.

Built on Google Vertex AI (Gemini, Imagen, Veo) with SQLite state tracking for crash recovery.

## How it works

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. STORYBOARD (Gemini 2.5 Flash)                           │
│     Prompt → structured scene breakdown with style guide,   │
│     keyframe prompts, motion descriptions, transitions      │
├─────────────────────────────────────────────────────────────┤
│  2. KEYFRAMES (Imagen 3.0 + Gemini Flash Image)             │
│     Scene 0 start frame from text (Imagen)                  │
│     End frames via image-conditioned generation (Gemini)    │
│     Scene N+1 start = Scene N end (visual continuity)       │
├─────────────────────────────────────────────────────────────┤
│  3. VIDEO GENERATION (Veo 2.0)                              │
│     Start frame + end frame → interpolated video clip       │
│     Long-running operations polled with crash-safe resume   │
├─────────────────────────────────────────────────────────────┤
│  4. STITCHING (ffmpeg)                                      │
│     Concatenate clips → final MP4                           │
│     Optional crossfade transitions                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  final.mp4
```

Each step persists state to SQLite before proceeding. If the process crashes at any point, resume picks up where it left off — no wasted API calls.

## Prerequisites

- **Python 3.11+**
- **ffmpeg** — `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)
- **Google Cloud service account** with Vertex AI API enabled

## Setup

```bash
# Clone and install
git clone <repo-url> && cd video-pipeline
pip install -e .

# Configure credentials
# Place your GCP service account JSON in the project root, then:
echo 'GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json' > .env

# Verify config
cat config.yaml  # check project_id matches your GCP project
```

The `config.yaml` file controls all settings:

```yaml
google_cloud:
  project_id: "your-gcp-project-id"
  location: "us-central1"
  use_vertex_ai: true

models:
  storyboard_llm: "gemini-2.5-flash"
  image_gen: "imagen-3.0-generate-001"
  image_conditioned: "gemini-2.5-flash-image"
  video_gen: "veo-2.0-generate-001"

pipeline:
  default_style: "cinematic"
  default_aspect_ratio: "16:9"
  default_clip_duration: 4        # seconds per clip
  max_scenes: 15
  image_gen_delay: 3              # seconds between image API calls
  video_poll_interval: 15         # seconds between Veo poll checks
  video_poll_max: 40              # max polls (~10 min timeout)
  crossfade_seconds: 0.0          # 0 = hard cuts, >0 = crossfade
```

Environment variables override YAML values using `VIDPIPE_` prefix with `__` for nesting:
```bash
export VIDPIPE_PIPELINE__MAX_SCENES=10
export VIDPIPE_GOOGLE_CLOUD__PROJECT_ID=my-project
```

## CLI Usage

### Generate a video

```bash
python -m vidpipe generate "A cat exploring a neon-lit Tokyo alley at night"

# With options
python -m vidpipe generate "Ocean waves at sunset" \
  --style cinematic \
  --aspect-ratio 16:9 \
  --clip-duration 5
```

### Check status

```bash
python -m vidpipe status <project-id>
```

### List all projects

```bash
python -m vidpipe list
```

### Resume a failed run

```bash
python -m vidpipe resume <project-id>
```

### Re-stitch with crossfade

```bash
python -m vidpipe stitch <project-id> --crossfade 0.5
```

## HTTP API

Start the server:

```bash
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Start new video generation (returns immediately) |
| `GET` | `/api/projects/{id}/status` | Poll project status |
| `GET` | `/api/projects/{id}` | Full project details with scenes |
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects/{id}/resume` | Resume failed project |
| `GET` | `/api/projects/{id}/download` | Download final MP4 |
| `GET` | `/api/health` | Health check |

### Example

```bash
# Start generation
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat exploring Tokyo at night", "style": "cinematic"}'

# Poll status
curl http://localhost:8000/api/projects/<project-id>/status

# Download result
curl -o video.mp4 http://localhost:8000/api/projects/<project-id>/download
```

## Project Structure

```
vidpipe/
├── __init__.py          # ffmpeg validation
├── __main__.py          # CLI entry point
├── config.py            # Settings (YAML + env vars)
├── cli/
│   └── commands.py      # 5 Typer CLI commands
├── api/
│   ├── app.py           # FastAPI app with lifespan
│   └── routes.py        # 7 API endpoints
├── orchestrator/
│   ├── pipeline.py      # State machine coordinator
│   └── state.py         # Status transitions & resume logic
├── pipeline/
│   ├── storyboard.py    # Gemini structured output → scenes
│   ├── keyframes.py     # Imagen + Gemini → start/end frame PNGs
│   ├── video_gen.py     # Veo → MP4 clips with polling
│   └── stitcher.py      # ffmpeg concat/crossfade → final MP4
├── schemas/
│   └── storyboard.py    # Pydantic models for storyboard output
├── services/
│   ├── vertex_client.py # Singleton Vertex AI client
│   └── file_manager.py  # Artifact storage with path safety
└── db/
    ├── models.py         # SQLAlchemy 2.0 ORM models
    ├── engine.py         # Async engine with WAL mode
    └── __init__.py       # DB init and session factory
```

## Output Structure

```
tmp/<project-id>/
├── keyframes/
│   ├── scene_0_start.png
│   ├── scene_0_end.png
│   ├── scene_1_start.png
│   └── ...
├── clips/
│   ├── scene_0.mp4
│   ├── scene_1.mp4
│   └── ...
└── output/
    └── final.mp4
```

## Crash Recovery

Every step commits to SQLite (WAL mode) before proceeding:

- **Storyboard** — scenes saved to DB before keyframe generation starts
- **Keyframes** — committed after each scene, not at the end
- **Video generation** — Veo operation ID saved before polling begins; resume continues polling from last count
- **Stitching** — idempotent, can be re-run with different crossfade settings

If anything fails, `python -m vidpipe resume <project-id>` skips completed steps and picks up from the failure point.

## Cost Estimate

Veo video generation is the primary cost driver. Approximate per-project:

| Scenes | Clip Duration | Estimated Cost |
|--------|--------------|----------------|
| 3 | 5s | ~$9 |
| 5 | 5s | ~$15 |
| 5 | 8s | ~$15 |

Imagen and Gemini calls are minimal cost by comparison.

## License

MIT
