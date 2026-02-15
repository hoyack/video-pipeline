---
phase: 03-orchestration-interfaces
plan: 03
subsystem: api
tags: [fastapi, http-api, async-request-reply, background-tasks, file-download]

dependency_graph:
  requires:
    - vidpipe.orchestrator.pipeline.run_pipeline
    - vidpipe.orchestrator.state.can_resume
    - vidpipe.db.models (Project, Scene, Keyframe, VideoClip)
    - vidpipe.db (async_session, init_database, shutdown)
    - vidpipe.config.settings
    - vidpipe.validate_dependencies
  provides:
    - vidpipe.api.app (FastAPI application)
    - vidpipe.api.routes (7 endpoint handlers)
    - python -m vidpipe.api (server entry point)
  affects:
    - Future API clients (can now interact programmatically)
    - Deployment environments (need uvicorn and exposed port)

tech_stack:
  added:
    - FastAPI (web framework)
    - Uvicorn (ASGI server)
  patterns:
    - Async request-reply pattern with 202 status codes
    - Background task execution with fresh session creation
    - Lifespan context manager (not deprecated on_event)
    - Pydantic response schemas for type safety
    - FileResponse for video file downloads

key_files:
  created:
    - vidpipe/api/__init__.py
    - vidpipe/api/app.py
    - vidpipe/api/routes.py
    - vidpipe/api/__main__.py
  modified: []

decisions:
  - Use asynccontextmanager lifespan instead of deprecated @app.on_event
  - Create fresh async_session() in background tasks (never share request session)
  - APIRouter with /api prefix for route organization
  - Direct session creation instead of Depends injection for simplicity
  - UUID validation via FastAPI path parameter type hints
  - Exclude output_path from response schemas (security via obscurity layer)
  - FileResponse with Content-Disposition attachment header for downloads
  - Generic exception handler returns 500 with detail (prevents stack trace leakage)

metrics:
  duration_seconds: 114
  tasks_completed: 2
  files_created: 4
  commits: 2
  completed_at: "2026-02-15T02:08:12Z"
---

# Phase 03 Plan 03: HTTP API Implementation Summary

**One-liner:** Implemented FastAPI HTTP API with async request-reply pattern, 7 RESTful endpoints, background task execution, and MP4 file downloads.

## What Was Built

Created `vidpipe/api/` module with complete HTTP API for programmatic video generation.

**FastAPI Application (`app.py`):**
- FastAPI instance with title "Video Pipeline API" v0.1.0
- Lifespan context manager for startup/shutdown:
  - Startup: validate_dependencies() and init_database()
  - Shutdown: database connection cleanup
- Generic exception handler preventing stack traces in responses
- Router inclusion from routes module

**7 API Endpoints (`routes.py`):**

1. **POST /api/generate** - Start pipeline in background
   - Accepts GenerateRequest with prompt, style, aspect_ratio, clip_duration
   - Creates Project with status="pending"
   - Adds run_pipeline_background to BackgroundTasks
   - Returns 202 with project_id and status_url

2. **GET /api/projects/{id}/status** - Lightweight status polling
   - Returns StatusResponse with project-level status only
   - Includes created_at, updated_at, error_message
   - No scene details (that's the detail endpoint)

3. **GET /api/projects/{id}** - Full project detail
   - Returns ProjectDetail with scene breakdown
   - SceneDetail includes keyframe/clip existence flags
   - Shows per-scene status and clip_status

4. **GET /api/projects** - List all projects
   - Returns list[ProjectListItem] ordered by created_at desc
   - Lightweight listing for project browsing

5. **POST /api/projects/{id}/resume** - Resume failed/interrupted pipeline
   - Checks can_resume(project.status)
   - Returns 409 if not resumable (including "complete")
   - Adds run_pipeline_background to BackgroundTasks
   - Returns 202 with status_url

6. **GET /api/projects/{id}/download** - Download final MP4
   - Checks status == "complete", returns 409 if not ready
   - Validates output_path exists on disk, returns 404 if missing
   - Returns FileResponse with media_type="video/mp4"
   - Content-Disposition attachment header with filename

7. **GET /api/health** - Health check
   - Returns {"status": "ok", "version": "0.1.0"}

**Background Task Wrapper:**
- `run_pipeline_background()` creates fresh async_session()
- Never shares request session across async boundaries (research Pitfall 2)
- Error handling via orchestrator persistence

**Server Entry Point (`__main__.py`):**
- Enables `python -m vidpipe.api` server startup
- Uses settings.server.host and settings.server.port from config
- String import path "vidpipe.api.app:app" for reload support

## Implementation Details

**Async Request-Reply Pattern:**
All long-running operations (generate, resume) follow the pattern:
1. Create/validate resource in database
2. Return 202 Accepted immediately
3. Execute pipeline in background task
4. Client polls GET /status for completion

**Session Management:**
- Request handlers use `async with async_session() as session:`
- Background tasks create their own fresh session
- No session sharing across async boundaries

**Error Handling:**
- 404 for not found resources
- 409 for state conflicts (can't resume complete project, download not ready)
- 500 with generic exception handler (prevents stack traces)
- Error messages persisted to database by orchestrator

**Security Considerations:**
- output_path excluded from response schemas
- Internal file paths never exposed in API responses
- Path used only internally for FileResponse download

## Deviations from Plan

None - plan executed exactly as written.

All 7 endpoints implemented according to specification:
- POST /generate uses BackgroundTasks with 202 response
- GET /status provides lightweight polling
- GET /detail includes scene breakdown with keyframe/clip flags
- GET /projects lists all projects
- POST /resume validates with can_resume() and returns 409 for invalid states
- GET /download serves MP4 via FileResponse with proper headers
- GET /health returns static response

Response schemas exclude internal fields as required.
Background task wrapper creates fresh session as required.
Server entry point uses configured host/port as required.

## Testing & Verification

All verification checks passed:
- `from vidpipe.api.app import app; print(app.title)` → "Video Pipeline API"
- `from vidpipe.api.routes import router; print(len(router.routes))` → 7
- All 7 endpoint paths present: /api/generate, /api/projects/{id}/status, /api/projects/{id}, /api/projects, /api/projects/{id}/resume, /api/projects/{id}/download, /api/health
- ProjectDetail schema excludes output_path field
- run_pipeline_background creates fresh async_session()
- __main__.py imports successfully without starting server

## Dependencies Ready for Next Plans

The HTTP API is now ready for:
- **Integration testing:** Can test full pipeline via HTTP requests
- **Deployment:** Server starts via `python -m vidpipe.api`
- **Client applications:** Can integrate video generation into workflows
- **Monitoring:** Health check endpoint for uptime monitoring

## Files Created

1. **vidpipe/api/__init__.py** (1 line)
   - Module docstring

2. **vidpipe/api/app.py** (62 lines)
   - FastAPI application instance
   - Lifespan context manager
   - Router inclusion
   - Generic exception handler

3. **vidpipe/api/routes.py** (341 lines)
   - 7 Pydantic response schemas
   - Background task wrapper
   - 7 endpoint handlers

4. **vidpipe/api/__main__.py** (11 lines)
   - Server entry point for python -m vidpipe.api

## Key Decisions

1. **Lifespan over on_event:** Used @asynccontextmanager lifespan pattern per FastAPI latest best practices (on_event deprecated).

2. **Fresh sessions in background tasks:** Background task wrapper creates new async_session() to avoid sharing sessions across async boundaries (research Pitfall 2).

3. **APIRouter organization:** Used APIRouter with /api prefix for clean route organization and future versioning.

4. **Direct session creation:** Used `async with async_session()` directly instead of Depends injection for simplicity (fewer abstractions).

5. **UUID path validation:** FastAPI automatically validates and converts project_id: uuid.UUID path parameters.

6. **Security via exclusion:** Response schemas explicitly exclude output_path and other internal fields from API responses.

7. **FileResponse for downloads:** Used FastAPI FileResponse with proper media_type and Content-Disposition headers for MP4 downloads.

8. **Generic exception handler:** Catch-all handler returns 500 with detail message instead of exposing stack traces to API clients.

## Self-Check: PASSED

**Created files exist:**
```bash
FOUND: vidpipe/api/__init__.py
FOUND: vidpipe/api/app.py
FOUND: vidpipe/api/routes.py
FOUND: vidpipe/api/__main__.py
```

**Commits exist:**
```bash
FOUND: 06547f1 (Task 1: FastAPI app with 7 endpoints)
FOUND: 946bc2b (Task 2: Server entry point)
```

**Verification results:**
- App title: "Video Pipeline API"
- Route count: 7
- All endpoint paths present
- Response schemas exclude internal fields
- Background task creates fresh session
- Server entry point imports successfully
