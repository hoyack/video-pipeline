# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vidpipe is an AI-powered multi-scene video generation pipeline. It takes a text prompt and produces a cohesive short video with visual continuity across scenes — fully automated, crash-safe, and resumable. The pipeline flows through four stages: storyboard (Gemini LLM) → keyframes (Imagen/Gemini image) → video clips (Veo/ComfyUI) → final MP4 (ffmpeg), with every stage checkpointed to SQLite for crash recovery.

## Tech Stack

**Backend:** Python 3.11+ — FastAPI, async SQLAlchemy + aiosqlite (SQLite WAL mode), Pydantic Settings, google-genai (Vertex AI), Typer CLI, httpx, Pillow, tenacity

**Frontend:** React 19, TypeScript (strict mode), Vite 7, Tailwind CSS 4, wouter (routing), @dnd-kit (drag-and-drop)

**External services:** Google Vertex AI (Gemini, Imagen, Veo), ComfyUI (WAN models), ffmpeg (stitching)

**Dev tools:** pytest + pytest-asyncio, ruff, black, ESLint

## Architecture Overview

### Pipeline State Machine
```
pending → storyboarding → keyframing → video_gen → stitching → complete
                                                                  ↓
                          failed/stopped ←────────────────────────┘
                               └──► resume picks up from last checkpoint
```
Every stage commits to SQLite before proceeding. On Veo content safety (RAI) rejection, the pipeline auto-escalates with progressively sanitized prompts (Level 0 → 1 → 2) before failing. All pipeline failures must persist to `Scene.error_message`.

### Backend (`backend/vidpipe/`)

| Directory | Role |
|-----------|------|
| `api/` | FastAPI app (`app.py`), route handlers (`routes.py`). New endpoints should be split into separate route files by domain. |
| `orchestrator/` | Pipeline state machine (`pipeline.py`), resume logic (`state.py`) |
| `pipeline/` | Stage implementations: `storyboard.py`, `keyframes.py`, `video_gen.py`, `stitcher.py` |
| `db/` | Async SQLAlchemy ORM models (`models.py`), engine with WAL mode (`engine.py`) |
| `services/` | Vertex AI client, file manager, manifest service, CV analysis, ComfyUI adapter, candidate scoring |
| `schemas/` | Pydantic models for Gemini structured output |
| `config.py` | Settings singleton — loads from env vars (`VIDPIPE_` prefix, `__` nesting) → `.env` → `config.yaml` → defaults |

**Design pattern:** Singletons for external clients (Vertex AI, ComfyUI). Dependency injection for business logic services.

### Frontend (`frontend/src/`)

| Directory | Role |
|-----------|------|
| `api/` | Fetch-based API client (`client.ts`), TypeScript types (`types.ts`) |
| `components/` | GenerateForm, ProgressView, ProjectList, ProjectDetail, SceneCard, etc. |
| `hooks/` | useProjectStatus (polling with backoff), usePolling |

Routing via wouter. State via React hooks (no Redux). Vite dev server proxies `/api` to the backend port configured in `.env`.

### Key Constraints

- **Veo aspect ratios:** Only `16:9` and `9:16` (no `1:1`)
- **Veo clip duration:** Discrete per model — Veo 2: `[5,6,7,8]`, Veo 3/3.1: `[4,6,8]`
- **Audio:** Veo 3+ only (`generate_audio` flag). Not Veo 2 or WAN models.
- **Mutual exclusion:** `image`/`last_frame` and `reference_images` cannot coexist. We prioritize keyframes.
- **Adding new models:** Update `ALLOWED_TEXT_MODELS`, `ALLOWED_IMAGE_MODELS`, or `ALLOWED_VIDEO_MODELS` in `routes.py`. This is the single gatekeeper.
- **Preview models** (e.g. `gemini-3-*-preview`) route to the `global` Vertex AI endpoint.
- **Project IDs** are stored without dashes (e.g. `23c95d88900f41fba2c50ff2cd475772`).

## ComfyUI Routing (CRITICAL)

`COMFYUI_VIDEO_MODELS` in `video_gen.py` is the router (`wan-2.2-i2v`, `wan-2.2-flf2v`, `ltx-2.3-flf2v`, `seedance-2.0-flf2v`). **Both the main pipeline AND `_regenerate_clip` in `routes.py` must check this set.** Missing the check in either place causes silent routing failures — clips will be sent to Veo instead of ComfyUI (or vice versa) with no error, just wrong output. ComfyUI is actively used in production.

Per-model ComfyUI behavior (fps, end-frame/audio support, workflow builder) lives in `COMFY_VIDEO_SPECS` in `services/comfyui_adapter.py` — it must stay in sync with `COMFYUI_VIDEO_MODELS` (a unit test pins this). FLF2V models degrade to first-frame-only generation when the end keyframe is missing. Audio on the ComfyUI path: `ltx-2.3-flf2v` and `seedance-2.0-flf2v` support `generate_audio`; WAN models do not. Seedance is a paid ByteDance partner node and requires Comfy Org account auth beyond the API key.

## Database (Per Worktree)

SQLite via async SQLAlchemy + aiosqlite. WAL mode, `expire_on_commit=False` (prevents greenlet errors). Tables auto-created at startup via `init_database()`.

Core models: `Scene` → `Shot` → `Keyframe` + `VideoClip`. Also: `Production`, `Manifest`, `Asset`, `GenerationCandidate`.

Each worktree has its own `vidpipe.db` at the path configured in `.env` (`VIDPIPE_STORAGE__DATABASE_URL`). **Schema changes require ALTER TABLE migrations or conditional column adds — never recreate the DB.** Do not delete `vidpipe.db` without asking the user first.

## Conventions

**Git:** Feature branches + PRs via git worktree. Use conventional commits: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, etc.

**Python:** snake_case for files, functions, variables. Type hints on all signatures. Async-first — all I/O uses `async def` + `await`. Sessions via `async_session()` context manager.

**TypeScript:** PascalCase for components, camelCase for functions/variables. Strict mode. Explicit `import type { ... }` for type-only imports.

**Route organization:** Split new API endpoints into separate route files by domain rather than adding to the existing `routes.py`.

**Docs:** Ignore `docs/` unless explicitly told to reference it — code is the source of truth.

## Testing

```bash
pytest backend/                          # All tests
pytest backend/path/to/test.py::test_fn  # Single test
npm run lint                             # Frontend lint (from frontend/)
ruff check backend/                      # Backend lint
```

Write tests for critical paths: pipeline logic, API endpoints, database operations. UI components and glue code don't require tests unless explicitly requested. Test coverage is not comprehensive — for bug fixes and user-facing changes, also verify via the running API/UI.

## Worktree Awareness

This project uses **git worktrees** for parallel development. Agents working in a worktree must:
- Only modify files within their worktree's working directory
- Never modify files in the main working tree or other worktrees
- Create feature branches scoped to their task
- Use PRs to merge back to master

### Worktree Environment

Each worktree runs on an isolated port. Check your local `.env` for:
- `VIDPIPE_SERVER__PORT` — backend port (e.g. 8001, 8002). **Never assume port 8000.**
- `VIDPIPE_STORAGE__DATABASE_URL` — path to this worktree's SQLite DB
- `VIDPIPE_STORAGE__TMP_DIR` — artifact output directory

Always read from config — never hardcode ports or paths.

## Decision Log

Architectural decision records live in `.planning/decisions/`. When making non-trivial architectural choices (new dependencies, schema changes, service patterns, API design), write an ADR documenting the context, decision, and consequences.

## Task Board

All task tracking uses **GitHub Issues**. No file-based task board — Issues are the single source of truth.

```bash
gh issue list --assignee @me             # Check your current assignments
gh issue edit <n> --add-assignee @me     # Claim a task before starting
gh issue create --title "..." --body "..." # Create issues for discovered work
```

When done, create a PR referencing the issue (e.g. `Closes #12`). Do not close issues manually — let the PR merge close them.

## Docker (Primary Runtime)

The app runs in Docker containers. **Code changes are not live until containers are rebuilt.**

### Container Architecture

| Container | Image | Ports | Notes |
|-----------|-------|-------|-------|
| `backend` | Python 3.11 + FastAPI | `${VIDPIPE_PORT:-8100}` → 8000 | Connects to Supabase PostgreSQL via `supabase_default` network |
| `frontend` | nginx (production) or Node (dev) | `${VIDPIPE_FRONTEND_PORT:-80}` → 80 | Serves built Vite bundle, proxies `/api` to backend |

### Compose Files

- **`docker-compose.yml`** — Production: static frontend build (nginx), PostgreSQL via Supabase pooler (port 6543), S3 storage
- **`docker-compose.dev.yml`** — Development: Vite dev server with HMR, backend `--reload`, SQLite, volume-mounted source code (changes auto-reload)

### Supabase Dependency & Safe Startup (CRITICAL)

The production stack depends on an **external self-hosted Supabase** project (sibling dir, e.g. `../supabase-project/`) for PostgreSQL and object storage. Prefer **`./scripts/start-stack.sh`** to bring everything up — it starts Supabase + app in order and auto-repairs the WSL2 failure modes below. Do not `docker compose down -v` the Supabase stack.

- **Always run Supabase with BOTH compose files:** `docker compose -f docker-compose.yml -f docker-compose.s3.yml ...` (for `up` and `down`). The base file sets storage `STORAGE_BACKEND=file` and omits the `minio` service; the `docker-compose.s3.yml` overlay switches storage to `s3 → http://minio:9000`. Objects live in **MinIO**; the backend reaches it via `S3_ENDPOINT=http://supabase-kong:8000/storage/v1`. Bringing the stack up with the base file alone makes all media GETs fail with `EISDIR`/500.
- **Recreate, don't restart, on stale-mount crashes:** after a Docker Desktop / WSL restart, containers may `Exit(127)` with `OCI runtime create failed ... not a directory` — a stale Docker Desktop WSL bind-mount cache, not a config error. `docker restart` reuses the broken spec and loops; `down` + `up` (with both files) re-resolves the mounts.
- **MinIO can format an empty pool:** MinIO (`/data`) and `supabase-storage` (`/var/lib/storage`) bind-mount the *same* host dir. If MinIO's cached view goes empty after a restart it logs `Formatting 1st pool` and serves a blank bucket (media 400s). The objects are NOT lost — recover with `docker compose -f ... -f docker-compose.s3.yml up -d --force-recreate --no-deps minio`. Confirm with `du -sh /data` inside the minio container vs `/var/lib/storage` inside storage.
- `supabase-storage` reporting `(unhealthy)` is cosmetic (healthcheck probes IPv6 `localhost`; service is on IPv4). It still serves.

### Rebuilding After Code Changes (CRITICAL)

**Production mode (`docker-compose.yml`):** Source code is COPIED into images at build time. After modifying backend Python or frontend TypeScript/React code, you **must rebuild** for changes to take effect:

```bash
# Rebuild and restart specific services (fastest)
docker compose up -d --build backend frontend

# Rebuild just backend (Python changes only)
docker compose up -d --build backend

# Rebuild just frontend (TypeScript/React changes only)
docker compose up -d --build frontend
```

**Development mode (`docker-compose.dev.yml`):** Backend source is volume-mounted with `--reload`, so Python changes auto-reload. Frontend runs Vite dev server with HMR. No rebuild needed for code changes, but dependency changes (`pyproject.toml`, `package.json`) still require rebuild.

### After Making Fixes

When code changes are ready to test in the browser, always rebuild the affected containers before verifying. A common failure mode is fixing code but forgetting to rebuild — the running containers still serve the old code.

## Build & Run (Without Docker)

```bash
# Backend
pip install -e backend/
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port $VIDPIPE_SERVER__PORT

# Frontend (dev)
cd frontend && npm install && npm run dev

# Frontend (production build, served by backend)
cd frontend && npm run build
```

Requires: Python 3.11+, Node.js 18+, ffmpeg, Google Cloud service account with Vertex AI API enabled. Credentials in `.env` (see `.env.example`).
