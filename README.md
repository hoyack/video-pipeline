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

## Repository Structure

```
video-pipeline/
├── backend/            # Python API + pipeline (see backend/README.md)
│   ├── vidpipe/        # Python package
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/           # React SPA (see frontend/README.md)
│   ├── src/
│   └── package.json
├── config.yaml         # Pipeline + model configuration
├── .env                # Credentials (not committed)
├── .env.example
└── docs/
```

## Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **ffmpeg** — `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)
- **Google Cloud service account** with Vertex AI API enabled

## Quick Start

```bash
# Clone
git clone <repo-url> && cd video-pipeline

# Configure credentials
echo 'GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json' > .env

# Verify config
cat config.yaml  # check project_id matches your GCP project

# Install backend
pip install -e backend/

# Install frontend
cd frontend && npm install && cd ..

# Start backend
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000

# Start frontend (separate terminal)
cd frontend && npm run dev
```

The frontend dev server runs on `http://localhost:5173` and proxies `/api` requests to the backend on port 8000.

## Configuration

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

```bash
# Generate a video
python -m vidpipe generate "A cat exploring a neon-lit Tokyo alley at night"

# With options
python -m vidpipe generate "Ocean waves at sunset" \
  --style cinematic \
  --aspect-ratio 16:9 \
  --clip-duration 5

# Check status
python -m vidpipe status <project-id>

# List all projects
python -m vidpipe list

# Resume a failed run
python -m vidpipe resume <project-id>

# Re-stitch with crossfade
python -m vidpipe stitch <project-id> --crossfade 0.5
```

## HTTP API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Start new video generation |
| `GET` | `/api/projects/{id}/status` | Poll project status |
| `GET` | `/api/projects/{id}` | Full project details with scenes |
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects/{id}/resume` | Resume failed project |
| `GET` | `/api/projects/{id}/download` | Download final MP4 |
| `GET` | `/api/health` | Health check |

## Crash Recovery

Every step commits to SQLite (WAL mode) before proceeding:

- **Storyboard** — scenes saved to DB before keyframe generation starts
- **Keyframes** — committed after each scene, not at the end
- **Video generation** — Veo operation ID saved before polling begins; resume continues polling from last count
- **Stitching** — idempotent, can be re-run with different crossfade settings

If anything fails, `python -m vidpipe resume <project-id>` skips completed steps and picks up from the failure point.

## Cost Estimate

| Scenes | Clip Duration | Estimated Cost |
|--------|--------------|----------------|
| 3 | 5s | ~$9 |
| 5 | 5s | ~$15 |
| 5 | 8s | ~$15 |

Imagen and Gemini calls are minimal cost by comparison.

## License

MIT
