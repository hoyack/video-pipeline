"""CLI commands for vidpipe using Typer and Rich.

Implements all 5 CLI commands:
- generate: Create and run new video generation scene
- resume: Resume a failed scene
- status: Show detailed scene information
- list: List all scenes in a table
- stitch: Re-stitch video with configurable crossfade
"""

import asyncio
import math
import random
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
from vidpipe.db.models import Scene, Shot, Keyframe, VideoClip, PipelineRun
from vidpipe.orchestrator.pipeline import run_pipeline
from vidpipe.orchestrator.state import can_resume
from vidpipe.pipeline.stitcher import stitch_videos
from vidpipe.config import settings
from vidpipe.api.routes import ALLOWED_TEXT_MODELS, ALLOWED_IMAGE_MODELS, ALLOWED_VIDEO_MODELS

# Per-model cost constants for CLI cost estimate (silent / base price)
_VIDEO_COST_PER_SECOND = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-001": 0.10,
    "veo-3.1-lite-generate-001": 0.10,
}
_VIDEO_COST_PER_SECOND_AUDIO = {
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-001": 0.15,
    "veo-3.1-lite-generate-001": 0.15,
}
_IMAGE_COST_PER_IMAGE = {
    "gemini-2.5-flash-image": 0.04,
    "gemini-3-pro-image-preview": 0.13,
}
_TEXT_COST_PER_CALL = {
    "gemini-2.5-flash": 0.006,
    "gemini-2.5-flash-lite": 0.001,
    "gemini-2.5-pro": 0.023,
    "gemini-3-flash-preview": 0.007,
    "gemini-3.1-pro-preview": 0.028,
    "gemini-3.5-flash": 0.007,
}
_ALLOWED_DURATIONS: dict[str, list[int]] = {
    "veo-2.0-generate-001": [5, 6, 7, 8],
    "veo-3.0-generate-001": [4, 6, 8],
    "veo-3.0-fast-generate-001": [4, 6, 8],
    "veo-3.1-generate-001": [4, 6, 8],
    "veo-3.1-fast-generate-001": [4, 6, 8],
    "veo-3.1-lite-generate-001": [4, 6, 8],
}

app = typer.Typer(name="vidpipe", help="AI-powered multi-shot video generation pipeline")
console = Console()


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for video generation"),
    style: str = typer.Option("cinematic", "--style", "-s", help="Visual style"),
    aspect_ratio: str = typer.Option("16:9", "--aspect-ratio", "-a", help="Video aspect ratio"),
    clip_duration: int = typer.Option(6, "--clip-duration", "-d", help="Target clip duration in seconds"),
    total_duration: int = typer.Option(30, "--total-duration", "-t", help="Total video duration in seconds"),
    text_model: str = typer.Option("gemini-3.5-flash", "--text-model", help="Text/storyboard model"),
    image_model: str = typer.Option("gemini-2.5-flash-image", "--image-model", help="Image generation model"),
    video_model: str = typer.Option("veo-3.1-fast-generate-001", "--video-model", help="Video generation model"),
    enable_audio: bool = typer.Option(True, "--enable-audio/--no-audio", help="Enable audio generation (Veo 3+ only)"),
):
    """Generate a new video from a text prompt.

    Creates a new scene and runs the full pipeline: storyboard generation,
    keyframe generation, video clip generation, and stitching.
    """
    # Fail-fast dependency validation
    try:
        validate_dependencies()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

    # Validate model IDs
    if text_model not in ALLOWED_TEXT_MODELS:
        console.print(f"[red]Error:[/red] Invalid text model: {text_model}")
        console.print(f"Allowed: {', '.join(sorted(ALLOWED_TEXT_MODELS))}")
        raise typer.Exit(code=1)
    if image_model not in ALLOWED_IMAGE_MODELS:
        console.print(f"[red]Error:[/red] Invalid image model: {image_model}")
        console.print(f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_MODELS))}")
        raise typer.Exit(code=1)
    if video_model not in ALLOWED_VIDEO_MODELS:
        console.print(f"[red]Error:[/red] Invalid video model: {video_model}")
        console.print(f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_MODELS))}")
        raise typer.Exit(code=1)

    # Validate audio + model compatibility
    from vidpipe.api.routes import AUDIO_CAPABLE_MODELS
    if enable_audio and video_model not in AUDIO_CAPABLE_MODELS:
        console.print(f"[red]Error:[/red] Audio not supported for {video_model}")
        raise typer.Exit(code=1)

    # Validate clip duration per video model
    allowed_durs = _ALLOWED_DURATIONS.get(video_model, [5, 6, 7, 8])
    if clip_duration not in allowed_durs:
        console.print(f"[red]Error:[/red] clip_duration {clip_duration} not supported for {video_model}")
        console.print(f"Allowed: {allowed_durs}")
        raise typer.Exit(code=1)

    # Dynamic cost estimate
    shot_count = math.ceil(total_duration / clip_duration)
    if enable_audio and video_model in _VIDEO_COST_PER_SECOND_AUDIO:
        vid_rate = _VIDEO_COST_PER_SECOND_AUDIO[video_model]
    else:
        vid_rate = _VIDEO_COST_PER_SECOND.get(video_model, 0.40)
    vid_cost = shot_count * clip_duration * vid_rate
    img_cost = (shot_count + 1) * _IMAGE_COST_PER_IMAGE.get(image_model, 0.04)
    txt_cost = _TEXT_COST_PER_CALL.get(text_model, 0.01)
    est_cost = vid_cost + img_cost + txt_cost

    audio_label = " +audio" if enable_audio else ""
    console.print(f"[yellow]Estimated cost:[/yellow] ~${est_cost:.2f} ({shot_count} shots, {video_model}{audio_label})")
    console.print()

    asyncio.run(_generate_async(
        prompt, style, aspect_ratio, clip_duration,
        total_duration, text_model, image_model, video_model,
        enable_audio,
    ))


async def _generate_async(
    prompt: str, style: str, aspect_ratio: str, clip_duration: int,
    total_duration: int, text_model: str, image_model: str, video_model: str,
    enable_audio: bool,
):
    """Async implementation of generate command."""
    # Initialize database
    await init_database()

    shot_count = math.ceil(total_duration / clip_duration)

    # Create scene
    async with async_session() as session:
        scene = Scene(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            target_clip_duration=clip_duration,
            target_shot_count=shot_count,
            total_duration=total_duration,
            text_model=text_model,
            image_model=image_model,
            video_model=video_model,
            audio_enabled=enable_audio,
            seed=random.randint(0, 2**32 - 1),
            status="pending",
        )
        session.add(scene)
        await session.commit()
        await session.refresh(scene)

        console.print(f"[green]Created scene:[/green] {scene.id}")
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

                await run_pipeline(session, scene.id, progress_callback=callback_wrapper)

            # Success - refresh scene to get output path
            await session.refresh(scene)
            console.print(f"[green]\u2713[/green] Video generation complete!")
            console.print(f"[green]Output:[/green] {scene.output_path}")

        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Pipeline interrupted. You can resume this scene later with:[/yellow]")
            console.print(f"  python -m vidpipe resume {scene.id}")
            raise typer.Exit(code=130)

        except Exception as e:
            console.print()
            console.print(f"[red]\u2717 Pipeline failed:[/red] {str(e)}")
            console.print(f"[yellow]You can retry with:[/yellow] python -m vidpipe resume {scene.id}")
            raise typer.Exit(code=1)


@app.command()
def resume(
    scene_id: str = typer.Argument(..., help="Scene UUID to resume"),
):
    """Resume a failed or interrupted scene.

    Loads the scene from the database and resumes from the last incomplete step.
    """
    # Fail-fast dependency validation
    try:
        validate_dependencies()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

    asyncio.run(_resume_async(scene_id))


async def _resume_async(scene_id_str: str):
    """Async implementation of resume command."""
    # Parse UUID
    try:
        scene_uuid = uuid.UUID(scene_id_str)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid scene UUID: {scene_id_str}")
        raise typer.Exit(code=1)

    # Initialize database
    await init_database()

    # Load scene
    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_uuid))
        scene = result.scalar_one_or_none()

        if not scene:
            console.print(f"[red]Error:[/red] Scene not found: {scene_uuid}")
            raise typer.Exit(code=1)

        # Check if already complete
        if scene.status == "complete":
            console.print(f"[green]Scene already complete![/green]")
            console.print(f"[green]Output:[/green] {scene.output_path}")
            return

        # Check if resumable
        if not can_resume(scene.status):
            console.print(f"[red]Error:[/red] Scene status '{scene.status}' cannot be resumed")
            raise typer.Exit(code=1)

        console.print(f"[yellow]Resuming scene:[/yellow] {scene.id}")
        console.print(f"[yellow]Current status:[/yellow] {scene.status}")
        console.print()

        # Run pipeline with progress display
        try:
            with console.status("[bold green]Resuming pipeline...") as status:
                def callback_wrapper(msg: str):
                    status.update(f"[bold green]{msg}")

                await run_pipeline(session, scene.id, progress_callback=callback_wrapper)

            # Success
            await session.refresh(scene)
            console.print(f"[green]\u2713[/green] Pipeline resumed and completed!")
            console.print(f"[green]Output:[/green] {scene.output_path}")

        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Pipeline interrupted. You can resume again with:[/yellow]")
            console.print(f"  python -m vidpipe resume {scene.id}")
            raise typer.Exit(code=130)

        except Exception as e:
            console.print()
            console.print(f"[red]\u2717 Pipeline failed:[/red] {str(e)}")
            console.print(f"[yellow]You can retry with:[/yellow] python -m vidpipe resume {scene.id}")
            raise typer.Exit(code=1)


@app.command()
def status(
    scene_id: str = typer.Argument(..., help="Scene UUID"),
):
    """Show detailed scene status and information."""
    asyncio.run(_status_async(scene_id))


async def _status_async(scene_id_str: str):
    """Async implementation of status command."""
    # Parse UUID
    try:
        scene_uuid = uuid.UUID(scene_id_str)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid scene UUID: {scene_id_str}")
        raise typer.Exit(code=1)

    # Initialize database
    await init_database()

    # Load scene
    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_uuid))
        scene = result.scalar_one_or_none()

        if not scene:
            console.print(f"[red]Error:[/red] Scene not found: {scene_uuid}")
            raise typer.Exit(code=1)

        # Query shot count
        shot_count_result = await session.execute(
            select(func.count(Shot.id)).where(Shot.scene_id == scene.id)
        )
        shot_count = shot_count_result.scalar()

        # Query latest pipeline run
        run_result = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.scene_id == scene.id)
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        )
        latest_run = run_result.scalar_one_or_none()

        # Color-code status
        status_color = _get_status_color(scene.status)
        status_display = f"[{status_color}]{scene.status}[/{status_color}]"

        # Truncate prompt for display
        prompt_display = scene.prompt if len(scene.prompt) <= 80 else scene.prompt[:77] + "..."

        # Build info text
        info_lines = [
            f"[bold]ID:[/bold] {scene.id}",
            f"[bold]Prompt:[/bold] {prompt_display}",
            f"[bold]Status:[/bold] {status_display}",
            f"[bold]Style:[/bold] {scene.style}",
            f"[bold]Aspect Ratio:[/bold] {scene.aspect_ratio}",
            f"[bold]Clip Duration:[/bold] {scene.target_clip_duration}s",
            f"[bold]Shots:[/bold] {shot_count}",
            f"[bold]Created:[/bold] {scene.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"[bold]Updated:[/bold] {scene.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # Add model info if available
        if scene.total_duration:
            info_lines.append(f"[bold]Total Duration:[/bold] {scene.total_duration}s")
        if scene.text_model:
            info_lines.append(f"[bold]Text Model:[/bold] {scene.text_model}")
        if scene.image_model:
            info_lines.append(f"[bold]Image Model:[/bold] {scene.image_model}")
        if scene.video_model:
            info_lines.append(f"[bold]Video Model:[/bold] {scene.video_model}")

        # Add output path if complete
        if scene.status == "complete" and scene.output_path:
            info_lines.append(f"[bold]Output:[/bold] [green]{scene.output_path}[/green]")

        # Add error message if failed
        if scene.status == "failed" and scene.error_message:
            info_lines.append(f"[bold]Error:[/bold] [red]{scene.error_message}[/red]")

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
            title="[bold]Scene Status[/bold]",
            border_style="blue",
        )
        console.print(panel)


@app.command(name="list")
def list_scenes():
    """List all video generation scenes."""
    asyncio.run(_list_async())


async def _list_async():
    """Async implementation of list command."""
    # Initialize database
    await init_database()

    # Query all scenes
    async with async_session() as session:
        result = await session.execute(
            select(Scene).order_by(Scene.created_at.desc())
        )
        scenes = result.scalars().all()

        if not scenes:
            console.print("[yellow]No scenes found[/yellow]")
            return

        # Create table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("ID", style="dim")
        table.add_column("Prompt")
        table.add_column("Status")
        table.add_column("Created")

        for scene in scenes:
            # Truncate ID to first 8 chars + "..."
            id_display = str(scene.id)[:8] + "..."

            # Truncate prompt to 50 chars
            prompt_display = scene.prompt if len(scene.prompt) <= 50 else scene.prompt[:47] + "..."

            # Color-code status
            status_color = _get_status_color(scene.status)
            status_display = f"[{status_color}]{scene.status}[/{status_color}]"

            # Format created date
            created_display = scene.created_at.strftime("%Y-%m-%d %H:%M")

            table.add_row(id_display, prompt_display, status_display, created_display)

        console.print(table)


@app.command()
def stitch(
    scene_id: str = typer.Argument(..., help="Scene UUID to re-stitch"),
    crossfade: float = typer.Option(0.0, "--crossfade", "-c", help="Crossfade duration in seconds"),
):
    """Re-stitch a completed scene with configurable crossfade.

    Useful for trying different crossfade durations without regenerating video clips.
    """
    # Fail-fast dependency validation
    try:
        validate_dependencies()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

    asyncio.run(_stitch_async(scene_id, crossfade))


async def _stitch_async(scene_id_str: str, crossfade: float):
    """Async implementation of stitch command."""
    # Parse UUID
    try:
        scene_uuid = uuid.UUID(scene_id_str)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid scene UUID: {scene_id_str}")
        raise typer.Exit(code=1)

    # Initialize database
    await init_database()

    # Load scene
    async with async_session() as session:
        result = await session.execute(select(Scene).where(Scene.id == scene_uuid))
        scene = result.scalar_one_or_none()

        if not scene:
            console.print(f"[red]Error:[/red] Scene not found: {scene_uuid}")
            raise typer.Exit(code=1)

        # Check for completed clips
        clip_count_result = await session.execute(
            select(func.count(VideoClip.id))
            .join(Shot, Shot.id == VideoClip.shot_id)
            .where(Shot.scene_id == scene.id)
            .where(VideoClip.status == "completed")
        )
        clip_count = clip_count_result.scalar()

        if clip_count == 0:
            console.print(f"[red]Error:[/red] No completed video clips found for scene {scene_uuid}")
            console.print("[yellow]Scene must have completed video generation before stitching.[/yellow]")
            raise typer.Exit(code=1)

        console.print(f"[yellow]Re-stitching scene:[/yellow] {scene.id}")
        console.print(f"[yellow]Clips:[/yellow] {clip_count}")
        console.print(f"[yellow]Crossfade:[/yellow] {crossfade}s")
        console.print()

        # Temporarily override crossfade setting
        original_crossfade = settings.pipeline.crossfade_seconds
        settings.pipeline.crossfade_seconds = crossfade

        try:
            with console.status("[bold green]Stitching video..."):
                await stitch_videos(session, scene)

            # Success
            await session.refresh(scene)
            console.print(f"[green]\u2713[/green] Stitching complete!")
            console.print(f"[green]Output:[/green] {scene.output_path}")

        except Exception as e:
            console.print()
            console.print(f"[red]\u2717 Stitching failed:[/red] {str(e)}")
            raise typer.Exit(code=1)

        finally:
            # Restore original crossfade setting
            settings.pipeline.crossfade_seconds = original_crossfade


def _get_status_color(status: str) -> str:
    """Get Rich color for a scene status.

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
