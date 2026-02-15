---
phase: 03-orchestration-interfaces
plan: 02
subsystem: cli
tags: [cli, typer, rich, user-interface, commands, progress-display]

dependency_graph:
  requires:
    - vidpipe.orchestrator.pipeline.run_pipeline
    - vidpipe.orchestrator.state.can_resume
    - vidpipe.pipeline.stitcher.stitch_videos
    - vidpipe.db (init_database, async_session)
    - vidpipe.db.models (Project, Scene, Keyframe, VideoClip, PipelineRun)
    - vidpipe.validate_dependencies
  provides:
    - python -m vidpipe (primary CLI entry point)
    - python -m vidpipe.cli (alternate CLI entry point)
    - CLI commands: generate, resume, status, list, stitch
  affects:
    - End users (primary interface for video generation)
    - Testing workflows (manual CLI testing for full pipeline)

tech_stack:
  added:
    - typer: CLI framework with automatic help generation
    - rich: Terminal formatting with colors, tables, panels, spinners
  patterns:
    - Async-over-sync pattern (asyncio.run in Typer commands)
    - Progress callback interface for decoupled UI updates
    - Rich status spinners for long-running operations
    - Color-coded status display for quick visual scanning

key_files:
  created:
    - vidpipe/cli/__init__.py
    - vidpipe/cli/commands.py
    - vidpipe/cli/__main__.py
    - vidpipe/__main__.py
  modified: []

decisions:
  - Use asyncio.run() wrapper pattern for Typer + async database operations
  - Implement progress_callback wrapper to update Rich status spinners from orchestrator
  - Temporarily override settings.pipeline.crossfade_seconds in stitch command for per-invocation control
  - Display cost warning ($15 per 5-scene project) before starting generation
  - Handle KeyboardInterrupt gracefully with resume instructions
  - Color-code all status displays for consistency (green/red/yellow/dim)

metrics:
  duration_seconds: 116
  tasks_completed: 2
  files_created: 4
  commits: 2
  completed_at: "2026-02-15T02:08:12Z"
---

# Phase 03 Plan 02: CLI Interface with Typer Summary

**One-liner:** Implemented full Typer CLI with 5 commands (generate, resume, status, list, stitch) using Rich formatting and async database operations.

## What Was Built

Created `vidpipe/cli/` module with complete CLI interface for video generation pipeline.

**CLI Commands (`commands.py`):**
- **generate**: Creates new project and runs full pipeline with Rich progress spinner and cost warning
- **resume**: Resumes failed/interrupted projects with status validation and progress display
- **status**: Shows detailed project info in Rich Panel with color-coded status and metadata
- **list**: Displays all projects in Rich Table with truncated IDs/prompts and formatted dates
- **stitch**: Re-stitches completed projects with configurable crossfade via --crossfade flag

**Entry Points:**
- `vidpipe/__main__.py`: Primary entry point for `python -m vidpipe`
- `vidpipe/cli/__main__.py`: Alternate entry point for `python -m vidpipe.cli`
- Both import and run the same Typer app instance

**Key Features:**
- Async database operations via `asyncio.run()` wrapper pattern for Typer compatibility
- Progress callbacks update Rich status spinners during pipeline execution
- Fail-fast ffmpeg validation before generation/resume/stitch commands
- Graceful KeyboardInterrupt handling with resume instructions
- Color-coded status throughout: green=complete, red=failed, yellow=in-progress, dim=pending
- Cost estimation warning ($15 per 5-scene project) displayed before generation starts

## Implementation Details

**Async-over-sync Pattern:**
Each Typer command is synchronous but calls an async implementation via `asyncio.run()`:
```python
@app.command()
def generate(...):
    asyncio.run(_generate_async(...))
```

This pattern allows async database operations while maintaining Typer's synchronous command interface.

**Progress Callback Integration:**
Commands pass wrapper functions to `run_pipeline()` that update Rich status spinners:
```python
with console.status("[bold green]Starting pipeline...") as status:
    def callback_wrapper(msg: str):
        status.update(f"[bold green]{msg}")

    await run_pipeline(session, project_id, progress_callback=callback_wrapper)
```

**Crossfade Override in Stitch Command:**
Temporarily modifies settings to allow per-invocation crossfade control:
```python
original_crossfade = settings.pipeline.crossfade_seconds
settings.pipeline.crossfade_seconds = crossfade
try:
    await stitch_videos(session, project)
finally:
    settings.pipeline.crossfade_seconds = original_crossfade
```

**Status Color Mapping:**
Helper function `_get_status_color()` provides consistent color coding:
- complete → green
- failed → red
- storyboarding/keyframing/video_gen/stitching → yellow
- pending → dim

## Deviations from Plan

None - plan executed exactly as written.

## Testing & Verification

All verification checks passed:

**Entry Points:**
- `python -m vidpipe --help` shows all 5 commands with descriptions
- `python -m vidpipe.cli --help` shows identical output

**Command Help Text:**
- `python -m vidpipe generate --help` shows prompt argument and --style, --aspect-ratio, --clip-duration options
- `python -m vidpipe resume --help` shows project_id argument
- `python -m vidpipe status --help` shows project_id argument
- `python -m vidpipe list --help` works (no arguments)
- `python -m vidpipe stitch --help` shows project_id argument and --crossfade option

**Typer App Validation:**
- `from vidpipe.cli.commands import app; print(type(app))` confirms Typer instance

## Dependencies Ready for Next Plan

The CLI is now fully functional and ready for:
- **Plan 03-03 (API):** Will provide REST API alongside CLI for programmatic access
- **End users:** Can generate videos via `python -m vidpipe generate "prompt"`

Both interfaces (CLI and upcoming API) leverage the same orchestrator logic from Plan 03-01.

## Files Created

1. **vidpipe/cli/__init__.py** (1 line)
   - Module docstring for CLI package

2. **vidpipe/cli/commands.py** (437 lines)
   - Typer app instance with 5 commands
   - All async implementations (_generate_async, _resume_async, etc.)
   - Helper function _get_status_color() for consistent status display

3. **vidpipe/cli/__main__.py** (5 lines)
   - Entry point for `python -m vidpipe.cli`

4. **vidpipe/__main__.py** (5 lines)
   - Entry point for `python -m vidpipe` (primary invocation pattern)

## Key Decisions

1. **Use asyncio.run() wrapper pattern:** Enables async database operations within Typer's synchronous command interface without complex event loop management.

2. **Progress callback wrapper for Rich status:** Decouples orchestrator from Rich library while enabling live progress updates in CLI.

3. **Temporary settings override in stitch command:** Allows per-invocation crossfade control without modifying config.yaml or database state.

4. **Display cost warning before generation:** Sets user expectations about Vertex AI API costs (~$15 per 5-scene project with Veo).

5. **Graceful KeyboardInterrupt handling:** Provides clear resume instructions instead of confusing stack traces.

6. **Consistent color-coded status:** Improves UX by enabling quick visual scanning of project states in list/status commands.

## Self-Check: PASSED

**Created files exist:**
```bash
FOUND: vidpipe/cli/__init__.py
FOUND: vidpipe/cli/commands.py
FOUND: vidpipe/cli/__main__.py
FOUND: vidpipe/__main__.py
```

**Commits exist:**
```bash
FOUND: 476c349 (Task 1: CLI commands)
FOUND: e5b83e7 (Task 2: Entry points)
```

**CLI verification:**
All help commands show correct arguments and options. Both entry points work identically.
