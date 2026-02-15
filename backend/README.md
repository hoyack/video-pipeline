# vidpipe — Backend

Python backend for the video pipeline. Provides a CLI, HTTP API, and the full generation pipeline (storyboard, keyframes, video generation, stitching).

## Setup

```bash
# From the repo root
pip install -e backend/

# Or with dev dependencies
pip install -e "backend/[dev]"
```

Requires `config.yaml` and `.env` at the repo root (not inside `backend/`).

## Running

```bash
# API server
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000

# CLI
python -m vidpipe generate "A cat exploring Tokyo at night"
python -m vidpipe status <project-id>
python -m vidpipe list
python -m vidpipe resume <project-id>
python -m vidpipe stitch <project-id> --crossfade 0.5
```

## Package Structure

```
vidpipe/
├── __init__.py          # ffmpeg validation
├── __main__.py          # CLI entry point
├── config.py            # Settings (YAML + env vars)
├── cli/
│   └── commands.py      # Typer CLI commands
├── api/
│   ├── app.py           # FastAPI app with lifespan
│   └── routes.py        # API endpoints
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

## Dependencies

Key runtime dependencies (see `pyproject.toml` for full list):

- **FastAPI + Uvicorn** — HTTP API
- **SQLAlchemy 2.0 + aiosqlite** — async SQLite with WAL mode
- **Typer + Rich** — CLI
- **google-genai** — Vertex AI (Gemini, Imagen, Veo)
- **Pydantic + pydantic-settings** — config and validation
- **Pillow** — image handling
- **PyYAML** — config file parsing

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Start new video generation |
| `GET` | `/api/projects/{id}/status` | Poll project status |
| `GET` | `/api/projects/{id}` | Full project details with scenes |
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects/{id}/resume` | Resume failed project |
| `GET` | `/api/projects/{id}/download` | Download final MP4 |
| `GET` | `/api/health` | Health check |

## Testing

```bash
pytest backend/
```
