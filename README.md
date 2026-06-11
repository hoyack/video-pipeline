# vidpipe

**An AI film studio in a box.** vidpipe turns a text prompt into a finished, multi-scene video — screenplay, consistent characters, generated footage, narration, and sound effects, all assembled into a single master MP4. Fully automated, crash-safe, and resumable at every stage.

![Scenes dashboard](docs/scenes-view.png)

The Scenes dashboard gives a per-scene view of the pipeline — status badges, keyframe thumbnails, and clip previews — with filtering and editing controls for each shot.

## What it does

- **Screenplay generation** — LLM-driven writers' room: logline → treatment → character breakdowns → scene breakdown → script → shot list, all editable and lockable before production begins
- **Production Bible** — a reusable asset library that keeps your film consistent: cast (actors with reference images, ElevenLabs voice profiles, and wardrobe looks), sets, props, and sound assets, bound to productions by tag
- **Scene pipeline with visual continuity** — storyboard → keyframes → video clips → stitched scene. Each shot's end frame seeds the next shot's start frame, so the camera never "jumps"
- **Multi-model, per-scene** — Vertex AI (Gemini, Imagen, Veo), ComfyUI (WAN 2.2), and Ollama-hosted text models; mix and match per scene from the UI
- **Narration & dialogue** — voice script generated from the screenplay, speaker tags resolved to cast voices, TTS via ElevenLabs, optional lip-sync, mixed into per-scene voice stems
- **Sound design** — an editable Sound Deck of SFX/ambience/foley cues generated from the timeline, rendered with ElevenLabs, mixed into per-scene SFX stems
- **Master render** — scene videos concatenated with voice and SFX stems mixed at timeline positions into one final MP4
- **Crash-safe & resumable** — every stage checkpoints to the database before proceeding; failed or stopped runs resume from the last checkpoint with no wasted API spend
- **Editing & iteration** — regenerate individual shots or clips, score and select among generation candidates, fork scenes, revert to checkpoints
- **Cost awareness** — cost estimates shown before generation starts

## How it works

Each scene runs through a four-stage pipeline:

```
Scene Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. STORYBOARD (Gemini LLM)                                 │
│     Prompt → structured shot breakdown with style guide,    │
│     character bible, keyframe prompts, motion descriptions  │
├─────────────────────────────────────────────────────────────┤
│  2. KEYFRAMES (Imagen / Gemini Image)                       │
│     Shot 0 start frame from text                            │
│     End frames via image-conditioned generation             │
│     Shot N+1 start = Shot N end (visual continuity)         │
├─────────────────────────────────────────────────────────────┤
│  3. VIDEO GENERATION (Veo / ComfyUI WAN)                    │
│     Start frame + end frame → interpolated video clip       │
│     Optional native audio (Veo 3+)                          │
│     Long-running operations polled with crash-safe resume   │
├─────────────────────────────────────────────────────────────┤
│  4. STITCHING (ffmpeg)                                      │
│     Concatenate clips → scene MP4                           │
│     Optional crossfade transitions                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  scene.mp4
```

On top of scenes, a **production** assembles a complete film:

```
Production
  ├── Production Bible ── cast (actors + voices + wardrobe), sets, props, sound assets
  ├── Screenplay ──────── logline → treatment → breakdowns → script → shot list (LLM)
  ├── Scenes ──────────── generated from the screenplay breakdown (pipeline above)
  ├── Voice Script ────── narration/dialogue lines → ElevenLabs TTS → per-scene voice stems
  ├── Sound Deck ──────── SFX/ambience cues → ElevenLabs SFX → per-scene SFX stems
  └── Master Render ───── scene videos + voice stems + SFX stems → master.mp4
```

## Documentation

| Doc | What's in it |
|-----|--------------|
| **[API: Creating a Full Production](docs/api-production-workflow.md)** | Step-by-step API guide from empty database to finished master MP4 — bible, screenplay, scenes, narration, SFX, master render, with curl examples and a full endpoint reference |
| [Backend docs](docs/backend-docs.md) | Backend architecture, services, and pipeline internals |
| [Frontend docs](docs/frontend-docs.md) | React SPA structure, components, and API client |
| [Backend README](backend/README.md) | Backend setup, CLI usage, and development |
| [Frontend README](frontend/README.md) | Frontend setup and development |
| [Video generation pipeline](docs/video-generation-pipeline.md) | Deep dive on the immediate vs. deferred generation workflows |
| [NASA documentary E2E](docs/nasa-documentary-e2e.md) | The end-to-end production test plan — a working reference for the full feature set |

## Setup (Docker — recommended)

The app runs as two containers (FastAPI backend + nginx-served React frontend) and **requires a self-hosted [Supabase](https://github.com/supabase/supabase) stack** for PostgreSQL (via the Supavisor pooler) and S3-compatible object storage (Supabase Storage backed by MinIO).

### 1. Prerequisites

- **Docker** with Compose v2 (and the NVIDIA container toolkit if you plan to use GPU features)
- **Self-hosted Supabase** running as a sibling project (e.g. `../supabase-project/`)
- **Google Cloud service account** JSON with the Vertex AI API enabled
- **ElevenLabs API key** (optional) — needed for narration TTS and SFX generation
- **ComfyUI** (optional) — needed only for WAN video models (`wan-2.2-i2v`)

### 2. Start Supabase with the S3 overlay

Supabase Storage **must** run with the S3 compose overlay so it talks to MinIO over HTTP. Without it, storage defaults to `STORAGE_BACKEND=file` and every asset request fails with a 500 (`EISDIR`):

```bash
cd ../supabase-project

# Correct — includes the S3 overlay
docker compose -f docker-compose.yml -f docker-compose.s3.yml up -d

# WRONG — storage defaults to file backend, all asset GETs return 500
# docker compose up -d
```

Remember the overlay every time you restart Supabase. Verify with:

```bash
docker exec supabase-storage env | grep STORAGE_BACKEND
# Should show: STORAGE_BACKEND=s3
```

Create a storage bucket for vidpipe (default name: `vidpipe-master`) in Supabase Studio if it doesn't exist.

### 3. Configure `.env`

```bash
cp .env.example .env
```

Fill in:

```bash
# Host ports
VIDPIPE_PORT=8100                # backend (8000 is typically taken by Supabase Kong)
VIDPIPE_FRONTEND_PORT=80         # frontend

# PostgreSQL via the Supavisor pooler (tenant ID in username, port 6543)
VIDPIPE_STORAGE__DATABASE_URL=postgresql+asyncpg://postgres.your-tenant-id:your-password@supabase-pooler:6543/postgres

# Supabase service role JWT (for Storage bucket access)
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-jwt
VIDPIPE_STORAGE__S3_SERVICE_KEY=${SUPABASE_SERVICE_ROLE_KEY}

# Google Cloud
VIDPIPE_GOOGLE_CLOUD__PROJECT_ID=your-gcp-project-id
VIDPIPE_GOOGLE_CLOUD__LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json

# ComfyUI (optional — only for WAN video models; can also be set in the Settings UI)
# COMFY_UI_HOST=https://cloud.comfy.org
# COMFY_UI_KEY=comfyui-your-api-key
```

The service account JSON is bind-mounted into the backend container; the backend joins the external `supabase_default` Docker network to reach the pooler and storage.

### 4. Build and start

```bash
docker compose up -d --build
```

Open `http://localhost` (frontend) — the backend API is at `http://localhost:8100`.

Runtime settings — model selection, ComfyUI endpoint, Ollama, and the ElevenLabs API key — are configured in the **Settings UI** (stored in the database), not in `.env`.

### Rebuilding after code changes

Production images copy source code at build time — **code changes are not live until you rebuild**:

```bash
docker compose up -d --build backend frontend   # or just the one you changed
```

For iterative development, use the dev compose file instead — volume-mounted source with backend `--reload` and Vite HMR, running on SQLite (no Supabase needed):

```bash
docker compose -f docker-compose.dev.yml up -d --build
# frontend: http://localhost:5173, backend: http://localhost:8000
```

### ⚠️ Security note

vidpipe currently has **no authentication** ([#28](https://github.com/hoyack/video-pipeline/issues/28)) — anyone who can reach the ports can trigger paid generation jobs and read stored API keys. Keep deployments on localhost or behind a VPN/authenticating reverse proxy.

## Setup (without Docker)

For local development against SQLite:

```bash
pip install -e backend/
cd frontend && npm install && npm run build && cd ..

# .env: VIDPIPE_STORAGE__DATABASE_URL=sqlite+aiosqlite:///vidpipe.db
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000
```

Requires Python 3.11+, Node.js 18+, and ffmpeg (`apt-get install ffmpeg` / `brew install ffmpeg`). See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for development details, tests, and CLI usage.

## Supported models

All models are selectable per scene in the UI.

- **Text (screenplay/storyboard):** Gemini 2.5 Flash / Flash Lite / Pro, Gemini 3 Flash / Pro, Ollama-hosted models (e.g. Kimi K2.5 cloud)
- **Image (keyframes):** Gemini Flash Image (Nano Banana), Gemini 3 Pro Image; via ComfyUI Cloud: Qwen (txt2img, Image Edit, **Edit 2509 multi-ref**), Flux.1 Dev (+LoRA/refs), **FLUX.2 Klein** (up to 4 refs)
- **Video:** Veo 2, Veo 3 / 3 Fast, Veo 3.1 / 3.1 Fast (+ GA variants); via ComfyUI Cloud: WAN 2.2 (i2v and **start+end keyframe FLF2V**), **LTX 2.3 FLF2V** (25fps, native audio), **Seedance 2.0 FLF2V** (paid, 4–15s, native audio)
- **Voice & SFX:** ElevenLabs TTS and sound effects

Native clip audio is supported on Veo 3+, LTX 2.3, and Seedance 2.0; WAN clips are silent, with narration and SFX mixed in at the production level. Reference-image keyframe generation (character/set identity from the Production Bible) works on Nano Banana, Qwen Edit 2509, and FLUX.2 Klein. Preview models route automatically to the `global` Vertex AI endpoint.

## Troubleshooting

**All images/assets return 500; backend logs show `FileNotFoundError: S3 GET failed: 500`.**
Supabase Storage is running with the file backend instead of S3. Restart it with the overlay:

```bash
cd ../supabase-project
docker compose -f docker-compose.yml -f docker-compose.s3.yml up -d storage
```

`docker compose down` preserves volumes and bind-mounted data (MinIO `./volumes/storage/`, PostgreSQL `./volumes/db/data/`) — just remember the S3 overlay when bringing services back up.

## License

MIT
