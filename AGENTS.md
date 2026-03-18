# Repository Guidelines

## Project Structure & Module Organization
`backend/vidpipe/` contains the backend app: `api/`, `orchestrator/`, `pipeline/`, `services/`, `db/`, and `schemas/`. Keep new FastAPI endpoints in domain route files instead of expanding `routes.py`. `frontend/src/` contains the React app in `components/`, `hooks/`, `api/`, and `lib/`; `frontend/dist/` is the build output. Tests live under `backend/tests/`, including `comfyui/` cases. Treat code as source of truth; consult `docs/` only when needed.

## Build, Test, and Development Commands
- `pip install -e "backend/[dev]"`: install backend dev dependencies.
- `uvicorn vidpipe.api.app:app --host 0.0.0.0 --port $VIDPIPE_SERVER__PORT`: run backend outside Docker; read port from `.env`.
- `python -m vidpipe list` and `python -m vidpipe resume <project-id>`: CLI entry points.
- `cd frontend && npm install && npm run dev`: start Vite dev server.
- `cd frontend && npm run build`: run TypeScript checks and bundle the frontend.
- `cd frontend && npm run lint`: run ESLint.
- `pytest backend/` and `ruff check backend/`: run backend tests and lint.
- `docker compose up -d --build backend frontend`: rebuild production containers after code changes. `docker-compose.dev.yml` auto-reloads code, but dependency changes still require rebuilds.

## Coding Style & Naming Conventions
Python uses 4-space indentation, snake_case names, type hints on all signatures, and async-first I/O with `async def` and `await`. Follow existing service and route boundaries, and use migrations or column adds for schema changes rather than recreating the database. TypeScript uses strict mode: PascalCase for components, camelCase for helpers, and `import type` for type-only imports. Keep frontend files focused and match existing hook patterns.

## Testing Guidelines
Name backend tests `test_<feature>.py` and focus coverage on pipeline logic, API endpoints, database behavior, and service adapters. Run focused tests with commands like `pytest backend/tests/test_elevenlabs_adapter.py -q`. No frontend test runner is configured, so UI changes should at minimum pass `npm run lint` and `npm run build`, then be checked in the UI/API. Document Ollama, ComfyUI, or cloud prerequisites.

## Commit, PR, and Environment Rules
Use conventional commits such as `feat:`, `fix:`, and `docs:`. This repo uses git worktrees, so only edit files in the current worktree and never delete or recreate `vidpipe.db` without approval. Read `.env` for `VIDPIPE_SERVER__PORT`, `VIDPIPE_STORAGE__DATABASE_URL`, and temp/output paths instead of hardcoding defaults. PRs should include the problem, verification steps, linked issues, config or migration impact, and screenshots for UI changes. Add an ADR under `.planning/decisions/` for non-trivial architectural or schema decisions, and never commit `.env`, service-account JSON, or generated media.
