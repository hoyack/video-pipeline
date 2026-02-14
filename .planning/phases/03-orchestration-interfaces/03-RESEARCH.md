# Phase 3: Orchestration & Interfaces - Research

**Researched:** 2026-02-14
**Domain:** CLI/API orchestration with async state machines and crash recovery
**Confidence:** HIGH

## Summary

Phase 3 requires building two user-facing interfaces (CLI and HTTP API) around a resumable pipeline orchestrator. The core challenge is implementing a database-backed state machine that tracks pipeline progress and enables crash recovery by skipping already-completed steps. Python's async ecosystem provides mature solutions for all requirements: Typer for CLI (already installed), FastAPI's BackgroundTasks for async job management (already installed), and idempotent step design patterns with SQLAlchemy state tracking.

The research reveals that external task queues (Celery/Redis) are unnecessary for this phase since pipelines run sequentially per project and FastAPI's built-in BackgroundTasks suffices for single-process async execution. The critical pattern is idempotency: each pipeline step must check database state before executing and record completion atomically. Rich library (already installed) provides production-ready progress visualization for CLI output.

**Primary recommendation:** Implement a simple orchestrator that wraps existing pipeline functions with database state checks. Use FastAPI BackgroundTasks for async execution (no external queue needed). Typer commands share session management via `asyncio.run()` wrapper pattern. Status tracking via polling endpoint following RFC-style async request-reply pattern.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Typer | 0.12+ | CLI framework | Official recommendation from FastAPI creator, built on Click with automatic help generation and type hint support |
| FastAPI | 0.115+ | HTTP API | Already installed, built-in BackgroundTasks sufficient for single-process async jobs |
| Rich | 13.0+ | Terminal output | Industry standard for Python CLI formatting, automatic error formatting, progress bars, and spinners |
| SQLAlchemy | 2.0+ | State persistence | Already in use, async session support for both CLI and API |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| asyncio | stdlib | Async coordination | Bridge between sync Typer commands and async database operations |
| python-statemachine | 2.0+ (optional) | State machine formalization | Only if state transitions become complex; overkill for linear 5-step pipeline |
| structlog | 25.x (optional) | Structured logging | If JSON logging needed for observability; Python stdlib logging sufficient for MVP |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastAPI BackgroundTasks | Celery + Redis | Adds infrastructure complexity for zero benefit (single-process sequential pipelines don't need distributed queue) |
| Manual state tracking | python-statemachine library | Library adds formalism but unnecessary for simple linear state machine (STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE) |
| Rich Progress | yaspin/tqdm | Rich already installed, superior API, better Typer integration |

**Installation:**
All core dependencies already installed per `pyproject.toml`. No additional packages required.

## Architecture Patterns

### Recommended Project Structure
```
vidpipe/
├── cli/
│   ├── __init__.py
│   ├── __main__.py      # Entry point: python -m vidpipe.cli
│   └── commands.py       # Typer app with command handlers
├── api/
│   ├── __init__.py
│   ├── __main__.py       # Entry point: python -m vidpipe.api
│   ├── app.py            # FastAPI app instance
│   ├── routes.py         # API endpoints
│   └── background.py     # Background task wrappers
├── orchestrator/
│   ├── __init__.py
│   ├── pipeline.py       # Main orchestrator: run_pipeline()
│   └── state.py          # State machine logic and transitions
├── pipeline/             # Existing step implementations
│   ├── storyboard.py
│   ├── keyframes.py
│   ├── video_gen.py
│   └── stitcher.py
└── db/                   # Existing database models
    ├── models.py
    └── engine.py
```

### Pattern 1: Idempotent Pipeline Steps with Database State Checks

**What:** Each pipeline step checks database before executing and skips if work already completed. Atomic state updates prevent race conditions during crash recovery.

**When to use:** Every orchestrator step that modifies database state or generates artifacts.

**Example:**
```python
# Source: Idempotent pipeline pattern from https://medium.com/geekculture/idempotent-data-pipeline-ba4c962d8d8c
async def run_storyboard_step(session: AsyncSession, project: Project) -> None:
    """Generate storyboard with idempotency - skip if already completed."""

    # Check current state
    if project.status not in ["pending", "failed"]:
        logger.info(f"Storyboard already completed for {project.id}, skipping")
        return

    # Atomic state transition: mark in-progress
    project.status = "storyboarding"
    await session.commit()

    try:
        # Execute actual work
        await generate_storyboard(session, project)
        # State already updated to "keyframing" by generate_storyboard()

    except Exception as e:
        # Mark failure but preserve partial progress
        project.status = "failed"
        project.error_message = str(e)
        await session.commit()
        raise
```

### Pattern 2: Async CLI Commands with Database Sessions

**What:** Typer commands are synchronous by default but can call async functions via `asyncio.run()`. Share database session factory between CLI and API.

**When to use:** All CLI commands that need database access.

**Example:**
```python
# Source: Async pattern from https://github.com/fastapi/typer/issues/88
import asyncio
import typer
from vidpipe.db import get_session
from vidpipe.orchestrator.pipeline import run_pipeline

app = typer.Typer()

@app.command()
def generate(
    prompt: str,
    style: str = "cinematic",
    aspect_ratio: str = "16:9",
    clip_duration: int = 5
):
    """Generate video from text prompt."""
    asyncio.run(_generate_async(prompt, style, aspect_ratio, clip_duration))

async def _generate_async(prompt: str, style: str, aspect_ratio: str, duration: int):
    """Async implementation of generate command."""
    async with get_session() as session:
        project = await create_project(session, prompt, style, aspect_ratio, duration)
        await run_pipeline(session, project.id)
```

### Pattern 3: FastAPI Background Tasks for Async Pipeline Execution

**What:** Use FastAPI's built-in `BackgroundTasks` to run pipelines asynchronously after returning 202 Accepted response. No external task queue needed for single-process sequential execution.

**When to use:** API endpoints that trigger long-running pipeline operations.

**Example:**
```python
# Source: FastAPI background tasks https://fastapi.tiangolo.com/tutorial/background-tasks/
from fastapi import BackgroundTasks, FastAPI
from vidpipe.orchestrator.pipeline import run_pipeline

@app.post("/api/generate", status_code=202)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """Start video generation in background, return project_id immediately."""
    async with get_session() as session:
        project = await create_project(session, request.prompt, ...)
        project_id = project.id
        await session.commit()

    # Add pipeline execution to background tasks
    background_tasks.add_task(run_pipeline_background, project_id)

    return {
        "project_id": str(project_id),
        "status_url": f"/api/projects/{project_id}/status"
    }

async def run_pipeline_background(project_id: uuid.UUID):
    """Background task wrapper for pipeline execution."""
    async with get_session() as session:
        await run_pipeline(session, project_id)
```

### Pattern 4: Async Request-Reply Status Polling

**What:** Return 202 Accepted with Location header pointing to status endpoint. Client polls status endpoint until completion. Status endpoint returns lightweight progress info.

**When to use:** All long-running async API operations.

**Example:**
```python
# Source: Async request-reply pattern https://learn.microsoft.com/en-us/azure/architecture/patterns/async-request-reply
@app.get("/api/projects/{project_id}/status")
async def get_project_status(project_id: uuid.UUID):
    """Lightweight status endpoint optimized for polling."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404)

        return {
            "project_id": str(project.id),
            "status": project.status,  # pending|storyboarding|keyframing|video_gen|stitching|complete|failed
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "error_message": project.error_message
        }
```

### Pattern 5: Rich Progress Display for CLI

**What:** Use Rich's `Progress` API to show real-time pipeline progress with spinners and status updates. Use `console.status()` for indeterminate operations.

**When to use:** All CLI commands that run long operations.

**Example:**
```python
# Source: Rich progress API https://rich.readthedocs.io/en/stable/progress.html
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

async def run_pipeline_with_progress(session: AsyncSession, project_id: uuid.UUID):
    """Run pipeline with CLI progress display."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Generating storyboard...", total=None)
        await run_storyboard_step(session, project)

        progress.update(task, description="[cyan]Generating keyframes...")
        await run_keyframe_step(session, project)

        progress.update(task, description="[cyan]Generating videos...")
        await run_video_step(session, project)

        progress.update(task, description="[cyan]Stitching final video...")
        await run_stitch_step(session, project)

        progress.update(task, description="[green]✓ Complete!")
```

### Pattern 6: FileResponse for Video Downloads

**What:** Use FastAPI's `FileResponse` to serve final MP4 files with proper Content-Disposition headers. FileResponse automatically streams large files efficiently.

**When to use:** API endpoint serving final video downloads.

**Example:**
```python
# Source: FastAPI FileResponse https://oneuptime.com/blog/post/2026-02-03-fastapi-file-downloads/view
from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/api/projects/{project_id}/download")
async def download_video(project_id: uuid.UUID):
    """Download final video file."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404)
        if project.status != "complete":
            raise HTTPException(status_code=409, detail="Video not ready")
        if not project.output_path or not Path(project.output_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        return FileResponse(
            path=project.output_path,
            media_type="video/mp4",
            filename=f"video_{project_id}.mp4",
            headers={"Content-Disposition": f'attachment; filename="video_{project_id}.mp4"'}
        )
```

### Anti-Patterns to Avoid

- **Running BackgroundTasks without session management:** Background tasks don't share request session. Always create new session inside background task function.
- **Using `python -m vidpipe` without `__main__.py`:** Create `vidpipe/cli/__main__.py` for CLI and `vidpipe/api/__main__.py` for API to enable `-m` module execution.
- **Mixing sync and async session usage:** CLI and API must both use async sessions. Never mix `Session` and `AsyncSession`.
- **Not setting transient=True on CLI progress:** Progress bars persist in terminal after completion unless `transient=True` specified.
- **Returning 200 OK for async operations:** Long-running operations must return 202 Accepted per HTTP spec, not 200 OK.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Task queue system | Custom background job manager with Redis | FastAPI BackgroundTasks | Built-in, zero config, sufficient for single-process sequential pipelines. Celery adds infrastructure overhead without benefit. |
| Progress bar rendering | Manual ANSI escape sequences | Rich Progress API | Handles terminal resizing, cursor management, multi-line updates, and edge cases (pipes, non-TTY). 1000+ lines of complexity. |
| CLI argument parsing | Manual sys.argv parsing | Typer decorators | Automatic type validation, help generation, error messages, and shell completion. Click foundation battle-tested. |
| HTTP error responses | Manual JSON error formatting | FastAPI HTTPException | Automatic status codes, OpenAPI schema generation, consistent error structure. Integrates with FastAPI exception handlers. |
| State machine validation | Manual if/else transition checks | Status field + transition rules | Simple linear pipeline doesn't need formal state machine library. Direct database status checks sufficient. |

**Key insight:** This phase glues together existing, well-tested libraries rather than implementing low-level primitives. Every "build it yourself" urge has a mature library solution already installed in `pyproject.toml`.

## Common Pitfalls

### Pitfall 1: BackgroundTasks Running After Server Shutdown

**What goes wrong:** FastAPI's BackgroundTasks are tied to the request lifecycle. If the server shuts down before tasks complete, tasks are lost without error.

**Why it happens:** BackgroundTasks run in the same process as the web server. `uvicorn` doesn't wait for background tasks on shutdown.

**How to avoid:** For Phase 3 MVP, accept this limitation (restart server = resume pipeline manually). Document that in-progress pipelines must be resumed via CLI/API after server restart. For production hardening (future phase), switch to persistent task queue (Celery).

**Warning signs:** Server restart during pipeline execution leaves project in "storyboarding" or "video_gen" status indefinitely. Resume command required to continue.

### Pitfall 2: Shared Database Session Across Async Boundaries

**What goes wrong:** Passing a SQLAlchemy `AsyncSession` instance to background tasks causes "session is already closed" errors or dangling transaction state.

**Why it happens:** FastAPI's dependency injection closes sessions after request completes. Background tasks run after response sent = after session closed.

**How to avoid:** Never pass `AsyncSession` to background tasks. Always create fresh session inside background task using `async with get_session()`. Session lifecycle must be fully contained within background task.

**Warning signs:** `InvalidRequestError: Session is already closed` or stale data reads in background tasks.

### Pitfall 3: Non-Idempotent Resume Logic

**What goes wrong:** Resuming a failed pipeline re-executes completed steps, wasting API calls and potentially corrupting data (e.g., re-generating keyframes with different results).

**Why it happens:** Orchestrator doesn't check database state before executing each step.

**How to avoid:** Every step must start with status check: `if project.status not in ["pending", "failed"]: return`. Each step transitions to next state atomically: `project.status = "next_step"; await session.commit()`.

**Warning signs:** Resume command takes as long as initial run. Duplicate artifacts in filesystem. Inconsistent scene counts between database and storyboard.

### Pitfall 4: Missing Error Context in Status Responses

**What goes wrong:** API returns `"status": "failed"` with no actionable error information. Users can't diagnose issues.

**Why it happens:** Exception handling updates `project.status = "failed"` but doesn't capture `error_message` or stack trace.

**How to avoid:** Every try/except in orchestrator must write to `project.error_message`: `project.error_message = f"{type(e).__name__}: {str(e)}"`. Status endpoint returns this field. Log full stack trace server-side for debugging.

**Warning signs:** Users report "generation failed" with no details. Support team can't debug issues from status API alone.

### Pitfall 5: Typer Command Blocking Without Feedback

**What goes wrong:** CLI command runs for minutes without output. User assumes it's frozen and kills process, interrupting pipeline.

**Why it happens:** Orchestrator runs in background without progress updates. Rich progress bar not integrated with async pipeline steps.

**How to avoid:** Use Rich's `console.status()` or `Progress` API with context manager. Emit status message before each step. For long steps (video generation), update spinner message periodically.

**Warning signs:** Users report "CLI froze" or frequently Ctrl+C during generation. High rate of incomplete projects.

### Pitfall 6: Race Condition in Resume Command

**What goes wrong:** Two simultaneous resume commands (CLI + API) both start executing same failed pipeline, causing duplicate work or state corruption.

**Why it happens:** No lock mechanism prevents concurrent pipeline execution for same project.

**How to avoid:** For MVP, document that concurrent resume not supported. For future hardening, add `pipeline_runs.status = "running"` row-level lock check before starting orchestrator. Second caller gets 409 Conflict.

**Warning signs:** Duplicate video generation operations logged. Unexpected API costs. Project status flipping between states rapidly.

### Pitfall 7: File Path Leakage in API Responses

**What goes wrong:** Status endpoint returns `"output_path": "/tmp/vidpipe/abc-123/final.mp4"` exposing server filesystem paths to clients.

**Why it happens:** Directly serializing `Project` model to JSON without filtering internal fields.

**How to avoid:** Create Pydantic response schemas that exclude internal fields. Status response should only include project_id, status, timestamps, error_message. Actual file path only used internally for download endpoint.

**Warning signs:** Security scanner flags information disclosure. API responses expose implementation details like temp directories.

## Code Examples

Verified patterns from official sources:

### Orchestrator Main Loop (Idempotent Step Execution)

```python
# Pattern: Idempotent pipeline with database state tracking
# Sources: https://medium.com/geekculture/idempotent-data-pipeline-ba4c962d8d8c
#          https://www.startdataengineering.com/post/why-how-idempotent-data-pipeline/

from sqlalchemy.ext.asyncio import AsyncSession
from vidpipe.db.models import Project
from vidpipe.pipeline.storyboard import generate_storyboard
from vidpipe.pipeline.keyframes import generate_keyframes
from vidpipe.pipeline.video_gen import generate_videos
from vidpipe.pipeline.stitcher import stitch_videos
import logging

logger = logging.getLogger(__name__)

async def run_pipeline(session: AsyncSession, project_id: uuid.UUID) -> None:
    """
    Execute full video generation pipeline with crash recovery.

    Each step is idempotent - checks current state and skips if already completed.
    Can be called multiple times safely (initial run, resume after crash, etc.)
    """
    # Load project
    result = await session.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one()

    # State machine: STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE

    # Step 1: Storyboard generation
    if project.status == "pending":
        logger.info(f"Starting storyboard for {project_id}")
        project.status = "storyboarding"
        await session.commit()

        try:
            await generate_storyboard(session, project)
            # generate_storyboard() sets status to "keyframing"
        except Exception as e:
            project.status = "failed"
            project.error_message = f"Storyboard failed: {str(e)}"
            await session.commit()
            raise

    # Step 2: Keyframe generation
    if project.status == "keyframing":
        logger.info(f"Starting keyframes for {project_id}")
        try:
            await generate_keyframes(session, project)
            project.status = "video_gen"
            await session.commit()
        except Exception as e:
            project.status = "failed"
            project.error_message = f"Keyframe generation failed: {str(e)}"
            await session.commit()
            raise

    # Step 3: Video generation
    if project.status == "video_gen":
        logger.info(f"Starting video generation for {project_id}")
        try:
            await generate_videos(session, project)
            project.status = "stitching"
            await session.commit()
        except Exception as e:
            project.status = "failed"
            project.error_message = f"Video generation failed: {str(e)}"
            await session.commit()
            raise

    # Step 4: Video stitching
    if project.status == "stitching":
        logger.info(f"Starting stitching for {project_id}")
        try:
            await stitch_videos(session, project)
            project.status = "complete"
            await session.commit()
        except Exception as e:
            project.status = "failed"
            project.error_message = f"Stitching failed: {str(e)}"
            await session.commit()
            raise

    logger.info(f"Pipeline complete for {project_id}")
```

### CLI Entry Point with Typer and Async

```python
# Pattern: Typer CLI with async database operations
# Sources: https://typer.tiangolo.com/
#          https://github.com/fastapi/typer/issues/88
#          https://pypi.org/project/async-typer/

# vidpipe/cli/__main__.py
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from vidpipe.db import get_session, init_database
from vidpipe.db.models import Project
from vidpipe.orchestrator.pipeline import run_pipeline
from sqlalchemy import select
import uuid

app = typer.Typer(name="vidpipe", help="AI-powered video generation pipeline")
console = Console()

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for video generation"),
    style: str = typer.Option("cinematic", "--style", "-s", help="Visual style"),
    aspect_ratio: str = typer.Option("16:9", "--aspect-ratio", "-a", help="Aspect ratio"),
    clip_duration: int = typer.Option(5, "--clip-duration", "-d", help="Target clip duration in seconds")
):
    """Generate video from text prompt."""
    asyncio.run(_generate_async(prompt, style, aspect_ratio, clip_duration))

async def _generate_async(prompt: str, style: str, aspect_ratio: str, duration: int):
    """Async implementation of generate command."""
    await init_database()

    async with get_session() as session:
        # Create project
        project = Project(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            target_clip_duration=duration,
            status="pending"
        )
        session.add(project)
        await session.commit()

        console.print(f"[green]Created project: {project.id}")

        # Run pipeline with progress display
        with console.status("[cyan]Running pipeline...") as status:
            try:
                await run_pipeline(session, project.id)
                console.print(f"[green]✓ Video generation complete!")
                console.print(f"[cyan]Output: {project.output_path}")
            except Exception as e:
                console.print(f"[red]✗ Pipeline failed: {e}")
                raise typer.Exit(code=1)

@app.command()
def status(
    project_id: str = typer.Argument(..., help="Project UUID")
):
    """Check status of a project."""
    asyncio.run(_status_async(uuid.UUID(project_id)))

async def _status_async(project_id: uuid.UUID):
    """Async implementation of status command."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            console.print(f"[red]Project not found: {project_id}")
            raise typer.Exit(code=1)

        console.print(f"[cyan]Project: {project.id}")
        console.print(f"[cyan]Status: {project.status}")
        console.print(f"[cyan]Created: {project.created_at}")
        if project.error_message:
            console.print(f"[red]Error: {project.error_message}")

@app.command()
def list():
    """List all projects."""
    asyncio.run(_list_async())

async def _list_async():
    """Async implementation of list command."""
    async with get_session() as session:
        result = await session.execute(select(Project).order_by(Project.created_at.desc()))
        projects = result.scalars().all()

        if not projects:
            console.print("[yellow]No projects found")
            return

        table = Table(title="Projects")
        table.add_column("ID", style="cyan")
        table.add_column("Prompt", style="white")
        table.add_column("Status", style="green")
        table.add_column("Created", style="blue")

        for project in projects:
            table.add_row(
                str(project.id)[:8] + "...",
                project.prompt[:50] + "..." if len(project.prompt) > 50 else project.prompt,
                project.status,
                project.created_at.strftime("%Y-%m-%d %H:%M")
            )

        console.print(table)

@app.command()
def resume(
    project_id: str = typer.Argument(..., help="Project UUID to resume")
):
    """Resume a failed or incomplete project."""
    asyncio.run(_resume_async(uuid.UUID(project_id)))

async def _resume_async(project_id: uuid.UUID):
    """Async implementation of resume command."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            console.print(f"[red]Project not found: {project_id}")
            raise typer.Exit(code=1)

        if project.status == "complete":
            console.print(f"[yellow]Project already complete")
            return

        console.print(f"[cyan]Resuming project from status: {project.status}")

        with console.status("[cyan]Running pipeline...") as status:
            try:
                await run_pipeline(session, project.id)
                console.print(f"[green]✓ Pipeline complete!")
            except Exception as e:
                console.print(f"[red]✗ Pipeline failed: {e}")
                raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
```

### FastAPI with Background Tasks

```python
# Pattern: FastAPI with background tasks and async request-reply
# Sources: https://fastapi.tiangolo.com/tutorial/background-tasks/
#          https://learn.microsoft.com/en-us/azure/architecture/patterns/async-request-reply

# vidpipe/api/app.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from vidpipe.db import get_session, init_database
from vidpipe.db.models import Project
from vidpipe.orchestrator.pipeline import run_pipeline
from pathlib import Path
import uuid
from typing import Optional
from sqlalchemy import select

app = FastAPI(title="Video Pipeline API", version="0.1.0")

# Request/Response schemas
class GenerateRequest(BaseModel):
    prompt: str
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    clip_duration: int = 5

class GenerateResponse(BaseModel):
    project_id: str
    status: str
    status_url: str

class StatusResponse(BaseModel):
    project_id: str
    status: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None

class ProjectDetail(BaseModel):
    project_id: str
    prompt: str
    style: str
    aspect_ratio: str
    status: str
    created_at: str
    scene_count: int
    error_message: Optional[str] = None

class ProjectListItem(BaseModel):
    project_id: str
    prompt: str
    status: str
    created_at: str

# Background task wrapper
async def run_pipeline_background(project_id: uuid.UUID):
    """Background task for running pipeline - creates its own session."""
    async with get_session() as session:
        try:
            await run_pipeline(session, project_id)
        except Exception as e:
            # Error already logged and saved to database by orchestrator
            pass

# Endpoints
@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    await init_database()

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/api/generate", status_code=202, response_model=GenerateResponse)
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start video generation in background.

    Returns 202 Accepted with project_id and status URL immediately.
    Client should poll status URL for completion.
    """
    async with get_session() as session:
        # Create project
        project = Project(
            prompt=request.prompt,
            style=request.style,
            aspect_ratio=request.aspect_ratio,
            target_clip_duration=request.clip_duration,
            status="pending"
        )
        session.add(project)
        await session.commit()
        project_id = project.id

    # Start pipeline in background
    background_tasks.add_task(run_pipeline_background, project_id)

    return GenerateResponse(
        project_id=str(project_id),
        status="pending",
        status_url=f"/api/projects/{project_id}/status"
    )

@app.get("/api/projects/{project_id}/status", response_model=StatusResponse)
async def get_status(project_id: uuid.UUID):
    """
    Lightweight status endpoint optimized for polling.

    Returns current project status. Client should poll this endpoint
    until status is 'complete' or 'failed'.
    """
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return StatusResponse(
            project_id=str(project.id),
            status=project.status,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            error_message=project.error_message
        )

@app.get("/api/projects/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: uuid.UUID):
    """Get full project details including scenes."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Count scenes
        from vidpipe.db.models import Scene
        result = await session.execute(
            select(Scene).where(Scene.project_id == project_id)
        )
        scenes = result.scalars().all()

        return ProjectDetail(
            project_id=str(project.id),
            prompt=project.prompt,
            style=project.style,
            aspect_ratio=project.aspect_ratio,
            status=project.status,
            created_at=project.created_at.isoformat(),
            scene_count=len(scenes),
            error_message=project.error_message
        )

@app.get("/api/projects", response_model=list[ProjectListItem])
async def list_projects():
    """List all projects."""
    async with get_session() as session:
        result = await session.execute(
            select(Project).order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()

        return [
            ProjectListItem(
                project_id=str(p.id),
                prompt=p.prompt,
                status=p.status,
                created_at=p.created_at.isoformat()
            )
            for p in projects
        ]

@app.post("/api/projects/{project_id}/resume", status_code=202)
async def resume_project(project_id: uuid.UUID, background_tasks: BackgroundTasks):
    """Resume a failed or incomplete project."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.status == "complete":
            raise HTTPException(status_code=409, detail="Project already complete")

        # Start pipeline in background
        background_tasks.add_task(run_pipeline_background, project_id)

        return {
            "project_id": str(project_id),
            "status_url": f"/api/projects/{project_id}/status"
        }

@app.get("/api/projects/{project_id}/download")
async def download_video(project_id: uuid.UUID):
    """Download final video file."""
    async with get_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.status != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Video not ready (status: {project.status})"
            )

        if not project.output_path:
            raise HTTPException(status_code=404, detail="Video file path not set")

        video_path = Path(project.output_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on disk")

        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"video_{project_id}.mp4",
            headers={
                "Content-Disposition": f'attachment; filename="video_{project_id}.mp4"'
            }
        )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Celery required for async tasks | FastAPI BackgroundTasks sufficient for simple cases | FastAPI 0.60+ (2020) | Small projects don't need Redis/RabbitMQ infrastructure. BackgroundTasks handles single-process async jobs. |
| Manual argparse CLI | Typer with type hints | Typer 1.0 (2021) | Automatic validation, help generation, and completion from type hints. 10x less boilerplate. |
| print() for CLI output | Rich library | Rich stable (2020+) | Production-grade progress bars, tables, and formatting. Handles edge cases (non-TTY, pipes, resizing). |
| Custom state machine libraries | Status field + transition logic | Ongoing | Simple linear pipelines don't need formal state machine. Direct database checks more readable. |
| SQLAlchemy 1.x sync | SQLAlchemy 2.0 async | SQLAlchemy 2.0 (2023) | Unified async API for CLI and web. No thread pool workarounds. Native asyncio support. |

**Deprecated/outdated:**
- **Click directly:** Typer is higher-level wrapper around Click from same author (FastAPI creator). Use Typer unless you need Click's advanced features.
- **Starlette HTTPException:** FastAPI's HTTPException is superset with JSON detail support. Always import from `fastapi`.
- **Session scoped to request via middleware:** FastAPI's dependency injection is preferred pattern. Middleware-based session management error-prone.

## Open Questions

1. **Should we track pipeline run metrics (API cost, duration) in PipelineRun table?**
   - What we know: ORCH-03 requires "Pipeline run metadata tracked (start time, duration, cost estimate, step log)"
   - What's unclear: Cost estimation formula for Vertex AI calls not defined in existing code
   - Recommendation: Track start_time and duration in PipelineRun. Cost estimation deferred to future phase (requires Vertex AI pricing lookup).

2. **Should CLI commands support streaming logs vs. spinner?**
   - What we know: Rich supports both `console.status()` (spinner) and `Progress` (bars)
   - What's unclear: User preference for verbosity level
   - Recommendation: Use spinner by default (`console.status()`). Add `--verbose` flag for streaming logs if users request it.

3. **Should status endpoint return scene-level progress?**
   - What we know: API-02 specifies "lightweight status for polling"
   - What's unclear: Whether scene count/progress is "lightweight" enough
   - Recommendation: Status endpoint returns only project-level status. Full detail endpoint (API-03) returns scene list.

4. **Should we validate ffmpeg at API startup or only CLI?**
   - What we know: `validate_dependencies()` exists in `vidpipe/__init__.py` for ffmpeg check
   - What's unclear: Whether API should fail startup if ffmpeg missing (API might only serve metadata)
   - Recommendation: Call `validate_dependencies()` in both CLI and API startup. Fail fast with clear error.

## Sources

### Primary (HIGH confidence)
- [Typer Official Documentation](https://typer.tiangolo.com/) - CLI framework features and patterns
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) - Official background task guide
- [Rich Progress API](https://rich.readthedocs.io/en/stable/progress.html) - Progress bars and status spinners
- [SQLAlchemy Asyncio](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) - Async session patterns
- [FastAPI Custom Response](https://fastapi.tiangolo.com/advanced/custom-response/) - FileResponse documentation
- [Python __main__ module](https://docs.python.org/3/library/__main__.html) - Entry point patterns

### Secondary (MEDIUM confidence)
- [Building CLI Tools with Typer and Rich](https://dasroot.net/posts/2026/01/building-cli-tools-with-typer-and-rich/) - 2026 integration patterns
- [How to Implement Background Tasks in FastAPI](https://oneuptime.com/blog/post/2026-02-02-fastapi-background-tasks/view) - 2026 best practices
- [Idempotent Data Pipeline](https://medium.com/geekculture/idempotent-data-pipeline-ba4c962d8d8c) - Idempotency patterns verified across multiple sources
- [Async Request-Reply Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/async-request-reply) - Microsoft Azure official pattern
- [7 Best Practices for Polling API Endpoints](https://www.merge.dev/blog/api-polling-best-practices) - Industry polling patterns
- [FastAPI Complete Guide 2026](https://devtoolbox.dedyn.io/blog/fastapi-complete-guide) - Recent async SQLAlchemy integration
- [Structlog ContextVars: Python Async Logging 2026](https://johal.in/structlog-contextvars-python-async-logging-2026/) - Modern async logging (optional)
- [Python State Machine](https://python-statemachine.readthedocs.io/en/1.0.2/readme.html) - State machine library comparison
- [Python Packaging: Entry Points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) - Official entry point spec

### Tertiary (LOW confidence)
- [Saga Orchestration Pattern](https://github.com/cdddg/py-saga-orchestration) - Complex orchestration (overkill for this phase)
- [sqlalchemy-fsm](https://pypi.org/project/sqlalchemy-fsm/) - State machine library (couldn't verify features from PyPI page)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already installed and officially documented
- Architecture: HIGH - Patterns verified from official FastAPI/Typer docs and 2026 blog posts
- Pitfalls: MEDIUM - Based on common issues in FastAPI/SQLAlchemy projects, some inferred from architecture constraints

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (30 days - stable ecosystem, slow-moving standards)
