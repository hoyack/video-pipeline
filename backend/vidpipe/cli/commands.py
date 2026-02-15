"""CLI commands for vidpipe using Typer and Rich.

Implements all 5 CLI commands:
- generate: Create and run new video generation project
- resume: Resume a failed project
- status: Show detailed project information
- list: List all projects in a table
- stitch: Re-stitch video with configurable crossfade
"""

import asyncio
import uuid
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sqlalchemy import select, func

from vidpipe import validate_dependencies
from vidpipe.db import init_database, async_session
from vidpipe.db.models import Project, Scene, Keyframe, VideoClip, PipelineRun
from vidpipe.orchestrator.pipeline import run_pipeline
from vidpipe.orchestrator.state import can_resume
from vidpipe.pipeline.stitcher import stitch_videos
from vidpipe.config import settings

app = typer.Typer(name="vidpipe", help="AI-powered multi-scene video generation pipeline")
console = Console()


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for video generation"),
    style: str = typer.Option("cinematic", "--style", "-s", help="Visual style"),
    aspect_ratio: str = typer.Option("16:9", "--aspect-ratio", "-a", help="Video aspect ratio"),
    clip_duration: int = typer.Option(5, "--clip-duration", "-d", help="Target clip duration in seconds"),
):
    """Generate a new video from a text prompt.

    Creates a new project and runs the full pipeline: storyboard generation,
    keyframe generation, video clip generation, and stitching.
    """
    # Fail-fast dependency validation
    try:
        validate_dependencies()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

    # Cost warning before starting
    console.print("[yellow]Estimated cost:[/yellow] ~$15 per 5-scene project (Veo video generation)")
    console.print()

    asyncio.run(_generate_async(prompt, style, aspect_ratio, clip_duration))


async def _generate_async(prompt: str, style: str, aspect_ratio: str, clip_duration: int):
    """Async implementation of generate command."""
    # Initialize database
    await init_database()

    # Create project
    async with async_session() as session:
        project = Project(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            target_clip_duration=clip_duration,
            status="pending",
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)

        console.print(f"[green]Created project:[/green] {project.id}")
        console.print()

        # Progress callback for status updates
        status_obj = {"status": None}

        def progress_callback(message: str):
            # Update the status message
            status_obj["status"] = message

        # Run pipeline with progress display
        try:
            with console.status("[bold green]Starting pipeline...") as status:
                # Wrapper to update rich status from callback
                def callback_wrapper(msg: str):
                    status.update(f"[bold green]{msg}")
                    progress_callback(msg)

                await run_pipeline(session, project.id, progress_callback=callback_wrapper)

            # Success - refresh project to get output path
            await session.refresh(project)
            console.print(f"[green]\u2713[/green] Video generation complete!")
            console.print(f"[green]Output:[/green] {project.output_path}")

        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Pipeline interrupted. You can resume this project later with:[/yellow]")
            console.print(f"  python -m vidpipe resume {project.id}")
            raise typer.Exit(code=130)

        except Exception as e:
            console.print()
            console.print(f"[red]\u2717 Pipeline failed:[/red] {str(e)}")
            console.print(f"[yellow]You can retry with:[/yellow] python -m vidpipe resume {project.id}")
            raise typer.Exit(code=1)


@app.command()
def resume(
    project_id: str = typer.Argument(..., help="Project UUID to resume"),
):
    """Resume a failed or interrupted project.

    Loads the project from the database and resumes from the last incomplete step.
    """
    # Fail-fast dependency validation
    try:
        validate_dependencies()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

    asyncio.run(_resume_async(project_id))


async def _resume_async(project_id_str: str):
    """Async implementation of resume command."""
    # Parse UUID
    try:
        project_uuid = uuid.UUID(project_id_str)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid project UUID: {project_id_str}")
        raise typer.Exit(code=1)

    # Initialize database
    await init_database()

    # Load project
    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_uuid))
        project = result.scalar_one_or_none()

        if not project:
            console.print(f"[red]Error:[/red] Project not found: {project_uuid}")
            raise typer.Exit(code=1)

        # Check if already complete
        if project.status == "complete":
            console.print(f"[green]Project already complete![/green]")
            console.print(f"[green]Output:[/green] {project.output_path}")
            return

        # Check if resumable
        if not can_resume(project.status):
            console.print(f"[red]Error:[/red] Project status '{project.status}' cannot be resumed")
            raise typer.Exit(code=1)

        console.print(f"[yellow]Resuming project:[/yellow] {project.id}")
        console.print(f"[yellow]Current status:[/yellow] {project.status}")
        console.print()

        # Run pipeline with progress display
        try:
            with console.status("[bold green]Resuming pipeline...") as status:
                def callback_wrapper(msg: str):
                    status.update(f"[bold green]{msg}")

                await run_pipeline(session, project.id, progress_callback=callback_wrapper)

            # Success
            await session.refresh(project)
            console.print(f"[green]\u2713[/green] Pipeline resumed and completed!")
            console.print(f"[green]Output:[/green] {project.output_path}")

        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Pipeline interrupted. You can resume again with:[/yellow]")
            console.print(f"  python -m vidpipe resume {project.id}")
            raise typer.Exit(code=130)

        except Exception as e:
            console.print()
            console.print(f"[red]\u2717 Pipeline failed:[/red] {str(e)}")
            console.print(f"[yellow]You can retry with:[/yellow] python -m vidpipe resume {project.id}")
            raise typer.Exit(code=1)


@app.command()
def status(
    project_id: str = typer.Argument(..., help="Project UUID"),
):
    """Show detailed project status and information."""
    asyncio.run(_status_async(project_id))


async def _status_async(project_id_str: str):
    """Async implementation of status command."""
    # Parse UUID
    try:
        project_uuid = uuid.UUID(project_id_str)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid project UUID: {project_id_str}")
        raise typer.Exit(code=1)

    # Initialize database
    await init_database()

    # Load project
    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_uuid))
        project = result.scalar_one_or_none()

        if not project:
            console.print(f"[red]Error:[/red] Project not found: {project_uuid}")
            raise typer.Exit(code=1)

        # Query scene count
        scene_count_result = await session.execute(
            select(func.count(Scene.id)).where(Scene.project_id == project.id)
        )
        scene_count = scene_count_result.scalar()

        # Query latest pipeline run
        run_result = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.project_id == project.id)
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        )
        latest_run = run_result.scalar_one_or_none()

        # Color-code status
        status_color = _get_status_color(project.status)
        status_display = f"[{status_color}]{project.status}[/{status_color}]"

        # Truncate prompt for display
        prompt_display = project.prompt if len(project.prompt) <= 80 else project.prompt[:77] + "..."

        # Build info text
        info_lines = [
            f"[bold]ID:[/bold] {project.id}",
            f"[bold]Prompt:[/bold] {prompt_display}",
            f"[bold]Status:[/bold] {status_display}",
            f"[bold]Style:[/bold] {project.style}",
            f"[bold]Aspect Ratio:[/bold] {project.aspect_ratio}",
            f"[bold]Clip Duration:[/bold] {project.target_clip_duration}s",
            f"[bold]Scenes:[/bold] {scene_count}",
            f"[bold]Created:[/bold] {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"[bold]Updated:[/bold] {project.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # Add output path if complete
        if project.status == "complete" and project.output_path:
            info_lines.append(f"[bold]Output:[/bold] [green]{project.output_path}[/green]")

        # Add error message if failed
        if project.status == "failed" and project.error_message:
            info_lines.append(f"[bold]Error:[/bold] [red]{project.error_message}[/red]")

        # Add latest run duration if available
        if latest_run and latest_run.total_duration_seconds:
            duration = latest_run.total_duration_seconds
            if duration < 60:
                duration_str = f"{duration:.1f}s"
            else:
                mins = int(duration // 60)
                secs = duration % 60
                duration_str = f"{mins}m {secs:.1f}s"
            info_lines.append(f"[bold]Last Run Duration:[/bold] {duration_str}")

        # Display panel
        panel = Panel(
            "\n".join(info_lines),
            title="[bold]Project Status[/bold]",
            border_style="blue",
        )
        console.print(panel)


@app.command(name="list")
def list_projects():
    """List all video generation projects."""
    asyncio.run(_list_async())


async def _list_async():
    """Async implementation of list command."""
    # Initialize database
    await init_database()

    # Query all projects
    async with async_session() as session:
        result = await session.execute(
            select(Project).order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()

        if not projects:
            console.print("[yellow]No projects found[/yellow]")
            return

        # Create table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("ID", style="dim")
        table.add_column("Prompt")
        table.add_column("Status")
        table.add_column("Created")

        for project in projects:
            # Truncate ID to first 8 chars + "..."
            id_display = str(project.id)[:8] + "..."

            # Truncate prompt to 50 chars
            prompt_display = project.prompt if len(project.prompt) <= 50 else project.prompt[:47] + "..."

            # Color-code status
            status_color = _get_status_color(project.status)
            status_display = f"[{status_color}]{project.status}[/{status_color}]"

            # Format created date
            created_display = project.created_at.strftime("%Y-%m-%d %H:%M")

            table.add_row(id_display, prompt_display, status_display, created_display)

        console.print(table)


@app.command()
def stitch(
    project_id: str = typer.Argument(..., help="Project UUID to re-stitch"),
    crossfade: float = typer.Option(0.0, "--crossfade", "-c", help="Crossfade duration in seconds"),
):
    """Re-stitch a completed project with configurable crossfade.

    Useful for trying different crossfade durations without regenerating video clips.
    """
    # Fail-fast dependency validation
    try:
        validate_dependencies()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

    asyncio.run(_stitch_async(project_id, crossfade))


async def _stitch_async(project_id_str: str, crossfade: float):
    """Async implementation of stitch command."""
    # Parse UUID
    try:
        project_uuid = uuid.UUID(project_id_str)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid project UUID: {project_id_str}")
        raise typer.Exit(code=1)

    # Initialize database
    await init_database()

    # Load project
    async with async_session() as session:
        result = await session.execute(select(Project).where(Project.id == project_uuid))
        project = result.scalar_one_or_none()

        if not project:
            console.print(f"[red]Error:[/red] Project not found: {project_uuid}")
            raise typer.Exit(code=1)

        # Check for completed clips
        clip_count_result = await session.execute(
            select(func.count(VideoClip.id))
            .join(Scene, Scene.id == VideoClip.scene_id)
            .where(Scene.project_id == project.id)
            .where(VideoClip.status == "completed")
        )
        clip_count = clip_count_result.scalar()

        if clip_count == 0:
            console.print(f"[red]Error:[/red] No completed video clips found for project {project_uuid}")
            console.print("[yellow]Project must have completed video generation before stitching.[/yellow]")
            raise typer.Exit(code=1)

        console.print(f"[yellow]Re-stitching project:[/yellow] {project.id}")
        console.print(f"[yellow]Clips:[/yellow] {clip_count}")
        console.print(f"[yellow]Crossfade:[/yellow] {crossfade}s")
        console.print()

        # Temporarily override crossfade setting
        original_crossfade = settings.pipeline.crossfade_seconds
        settings.pipeline.crossfade_seconds = crossfade

        try:
            with console.status("[bold green]Stitching video..."):
                await stitch_videos(session, project)

            # Success
            await session.refresh(project)
            console.print(f"[green]\u2713[/green] Stitching complete!")
            console.print(f"[green]Output:[/green] {project.output_path}")

        except Exception as e:
            console.print()
            console.print(f"[red]\u2717 Stitching failed:[/red] {str(e)}")
            raise typer.Exit(code=1)

        finally:
            # Restore original crossfade setting
            settings.pipeline.crossfade_seconds = original_crossfade


def _get_status_color(status: str) -> str:
    """Get Rich color for a project status.

    Color coding:
    - complete: green
    - failed: red
    - in-progress states: yellow
    - pending: dim
    """
    if status == "complete":
        return "green"
    elif status == "failed":
        return "red"
    elif status in ["storyboarding", "keyframing", "video_gen", "stitching"]:
        return "yellow"
    elif status == "pending":
        return "dim"
    else:
        return "white"
