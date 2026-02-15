# vidpipe

AI-powered multi-scene video generation pipeline. Takes a text prompt and produces a cohesive short video with visual continuity across scenes — fully automated, crash-safe, and resumable.

Built on Google Vertex AI (Gemini, Imagen, Veo) with SQLite state tracking for crash recovery.

## How it works

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. STORYBOARD (Gemini LLM)                                 │
│     Prompt → structured scene breakdown with style guide,   │
│     character bible, keyframe prompts, motion descriptions  │
├─────────────────────────────────────────────────────────────┤
│  2. KEYFRAMES (Imagen / Gemini Image)                       │
│     Scene 0 start frame from text                           │
│     End frames via image-conditioned generation             │
│     Scene N+1 start = Scene N end (visual continuity)       │
├─────────────────────────────────────────────────────────────┤
│  3. VIDEO GENERATION (Veo)                                   │
│     Start frame + end frame → interpolated video clip       │
│     Optional audio generation (Veo 3+)                      │
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
├── .env                # Credentials + GCP project (not committed)
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

# Configure credentials and GCP project
cp .env.example .env
# Edit .env:
#   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
#   VIDPIPE_GOOGLE_CLOUD__PROJECT_ID=your-gcp-project-id

# Install backend
pip install -e backend/

# Install frontend
cd frontend && npm install && npm run build && cd ..

# Start backend (serves both API and frontend)
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to use the app.

For frontend development with hot reload, run `npm run dev` in the `frontend/` directory (proxies to backend on port 8000).

## Configuration

### Credentials (`.env`)

Google Cloud credentials and project ID live in `.env` (never committed):

```bash
# Required
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
VIDPIPE_GOOGLE_CLOUD__PROJECT_ID=your-gcp-project-id

# Optional (defaults to us-central1)
# VIDPIPE_GOOGLE_CLOUD__LOCATION=us-central1
```

### Pipeline settings (`config.yaml`)

```yaml
models:
  storyboard_llm: "gemini-2.5-flash"
  image_gen: "imagen-4.0-fast-generate-001"
  image_conditioned: "gemini-2.5-flash-image"
  video_gen: "veo-3.1-fast-generate-001"

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

Any setting can be overridden via environment variables with `VIDPIPE_` prefix and `__` for nesting:
```bash
export VIDPIPE_PIPELINE__MAX_SCENES=10
export VIDPIPE_MODELS__VIDEO_GEN=veo-3.0-generate-001
```

## Supported Models

All models can be selected per-project in the UI.

**Text (storyboard):** Gemini 2.5 Flash, Flash Lite, Pro | Gemini 3 Flash, Pro

**Image (keyframes):** Imagen 3, 4, 4 Fast, 4 Ultra | Gemini Flash Image, 3 Pro Image

**Video:** Veo 2 | Veo 3, 3 Fast | Veo 3.1, 3.1 GA, 3.1 Fast, 3.1 Fast GA

Audio generation is supported on Veo 3+ models and can be toggled per-project.

Some preview models (Gemini 3 Flash/Pro, Gemini 3 Pro Image) are automatically routed to the `global` Vertex AI endpoint.

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

# Resume a failed or stopped run
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
| `POST` | `/api/projects/{id}/resume` | Resume failed/stopped project |
| `POST` | `/api/projects/{id}/stop` | Stop a running pipeline |
| `GET` | `/api/projects/{id}/download` | Download final MP4 |
| `GET` | `/api/health` | Health check |

## Pipeline States

```
pending → storyboarding → keyframing → video_gen → stitching → complete
                                                                   │
                          ┌──── stopped (user) ◄───────────────────┤
                          │                                        │
                          └──── failed (error) ◄───────────────────┘
                                    │
                                    └──► resume picks up from last checkpoint
```

Stopped and failed projects can be resumed — the pipeline skips completed steps and picks up from the failure/stop point.

## Crash Recovery

Every step commits to SQLite (WAL mode) before proceeding:

- **Storyboard** — scenes saved to DB before keyframe generation starts
- **Keyframes** — committed after each scene, not at the end
- **Video generation** — Veo operation ID saved before polling begins; resume continues polling from last count
- **Stitching** — idempotent, can be re-run with different crossfade settings

If anything fails, `python -m vidpipe resume <project-id>` skips completed steps and picks up from the failure point.

## Cost Estimate

Costs vary by model selection. Examples using default models (Imagen 4 Fast + Veo 3.1 Fast):

| Scenes | Clip Duration | Estimated Cost |
|--------|--------------|----------------|
| 3 | 4s | ~$2.50 |
| 5 | 6s | ~$5.50 |
| 5 | 8s | ~$7.00 |

Higher-tier models (Veo 3.1 GA, Imagen 4 Ultra) cost more. The UI shows a cost estimate before generation starts.

## License

MIT
