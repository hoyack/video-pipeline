# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

vidpipe — an AI-powered multi-scene video generation pipeline. Text prompt → storyboard (Gemini LLM) → keyframes (Imagen/Gemini image) → video clips (Veo) → final MP4 (ffmpeg). Full-stack: Python FastAPI backend + React TypeScript frontend, with SQLite state tracking for crash recovery.

## Commands

### Backend
```bash
pip install -e backend/                  # Install (editable)
pip install -e "backend/[dev]"           # Install with dev deps (pytest, black, ruff)
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000  # Run server
python -m vidpipe generate "prompt"      # CLI generate
python -m vidpipe status <scene-id>      # CLI status check
pytest backend/                          # Run all tests
pytest backend/path/to/test.py::test_fn  # Run single test
ruff check backend/                      # Lint
black backend/                           # Format
```

### Frontend
```bash
cd frontend
npm install          # Install deps
npm run dev          # Dev server (port 5173, proxies /api → :8000)
npm run build        # Production build (tsc + vite, output: frontend/dist/)
npm run lint         # ESLint
```

The backend serves `frontend/dist/` as static files with SPA fallback, so `npm run build` is needed for production.

## Workflow Conventions

**Git:** Feature branches + PRs via git worktree. Use conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, etc.).

**Validation:** Test coverage is not comprehensive. For bug fixes and user-facing changes, verify via the running API/UI in addition to pytest. Refactors and internal changes can rely on tests alone.

**Database schema changes:** Preserve the existing database. Write ALTER TABLE migrations or conditional column adds for breaking changes. Never delete `vidpipe.db` without asking first.

**Error handling:** All pipeline failures must be persisted to `Scene.error_message` so the user can see what happened. Transient retries are fine, but the final error state must always be recorded.

**Docs:** The specs in `docs/` are mostly accurate and useful for understanding intent, but the code is the source of truth when they diverge.

## Architecture

### Pipeline Stages (Resumable State Machine)
```
pending → storyboarding → keyframing → video_gen → stitching → complete
                                                                  ↓
                          failed/stopped ←────────────────────────┘
                               └──► resume picks up from last checkpoint
```

Every stage commits to SQLite (WAL mode) before proceeding. Resume skips completed steps. The orchestrator is in `backend/vidpipe/orchestrator/pipeline.py`, with resume logic in `orchestrator/state.py`.

On Veo content safety (RAI) rejection, the pipeline auto-escalates with progressively sanitized prompts (Level 0 → 1 → 2) before failing.

### Backend Layout (`backend/vidpipe/`)
- **`api/`** — FastAPI app (`app.py`), route handlers (`routes.py`). Split new endpoints into separate route files by domain rather than adding to `routes.py`.
- **`orchestrator/`** — State machine (`pipeline.py` runs the full pipeline loop), resume point calculation (`state.py`)
- **`pipeline/`** — Individual stage implementations: `storyboard.py` (Gemini → scene breakdown), `keyframes.py` (Imagen/Gemini → PNGs), `video_gen.py` (Veo → MP4 clips with polling + safety escalation), `stitcher.py` (ffmpeg concat/crossfade)
- **`db/`** — Async SQLAlchemy ORM models (`models.py`), engine setup with WAL mode (`engine.py`)
- **`services/`** — Vertex AI client singleton (`vertex_client.py`), file I/O with path traversal protection (`file_manager.py`), manifest/asset CRUD (`manifest_service.py`), CV analysis, ComfyUI adapter, candidate scoring, etc.
- **`schemas/`** — Pydantic models for Gemini structured output
- **`config.py`** — Settings singleton, YAML + env var loader

**Design pattern:** Use singletons for external clients (Vertex AI, ComfyUI). Use dependency injection (pass instances) for business logic services.

### Frontend Layout (`frontend/src/`)
- React 19 + TypeScript + Tailwind CSS + Vite
- Routing: wouter. State: React hooks (no Redux). DnD: @dnd-kit.
- `api/client.ts` — fetch-based API client; `api/types.ts` — TypeScript types
- `components/` — GenerateForm, ProgressView, ProjectList, ProjectDetail, SceneCard, etc.
- `hooks/` — useProjectStatus (polling with backoff), usePolling

### Configuration Hierarchy (highest priority first)
1. Environment variables (`VIDPIPE_` prefix, `__` nesting: `VIDPIPE_PIPELINE__MAX_SHOTS=10`)
2. `.env` file
3. `config.yaml`
4. Pydantic field defaults

Key config: `config.yaml` (models, pipeline params, storage, CV thresholds). Credentials in `.env` (never committed).

### Database
SQLite via async SQLAlchemy + aiosqlite. WAL mode, `expire_on_commit=False` (prevents greenlet errors). Tables auto-created at startup via `init_database()`.

Core models: `Scene` (top-level request) → `Shot` (individual shots) → `Keyframe` (start/end PNGs) + `VideoClip` (generated MP4s). Also: `Production`, `Manifest`, `Asset`, `GenerationCandidate`.

## Key Constraints

**Veo API:**
- Aspect ratios: only `16:9` and `9:16` (no `1:1`)
- Clip duration is discrete per model — Veo 2: `[5,6,7,8]`, Veo 3/3.1: `[4,6,8]`
- Audio: Veo 3+ only (`generate_audio` flag). Not available on Veo 2 or WAN models.

**Mutual exclusion:** `image`/`last_frame` (frame interpolation) and `reference_images` cannot be used together. We prioritize keyframes over reference_images.

**ComfyUI routing:** Models in `COMFYUI_VIDEO_MODELS` set (defined in `video_gen.py`: `wan-2.2-ref-i2v`, `wan-2.2-i2v`) route to ComfyUI instead of Veo. Both the main pipeline AND `_regenerate_clip` in `routes.py` must check this set. ComfyUI is actively used — not experimental.

**Adding new models:** Update `ALLOWED_TEXT_MODELS`, `ALLOWED_IMAGE_MODELS`, or `ALLOWED_VIDEO_MODELS` in `routes.py`. This is the single gatekeeper — no other files need updating for basic model support.

**Preview models** (e.g. `gemini-3-*-preview`) route to the `global` Vertex AI endpoint instead of the configured location.

**Project IDs** are stored without dashes in the database (e.g. `23c95d88900f41fba2c50ff2cd475772`).

## System Dependencies

- Python 3.11+, Node.js 18+
- **ffmpeg** — required at runtime, validated at startup (`__init__.py:validate_dependencies`)
- Google Cloud service account with Vertex AI API enabled
