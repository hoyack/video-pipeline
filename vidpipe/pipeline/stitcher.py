"""Video stitching with ffmpeg concat demuxer and crossfade support.

Concatenates completed video clips into final output using:
- concat demuxer for hard cuts (crossfade_seconds=0.0)
- xfade filter for smooth crossfade transitions (crossfade_seconds>0.0)

Implements STCH-01 through STCH-04 requirements from plan spec.
"""

import asyncio
import logging
import subprocess
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.config import settings
from vidpipe.db.models import Project, Scene, VideoClip
from vidpipe.services.file_manager import FileManager

logger = logging.getLogger(__name__)


async def stitch_videos(session: AsyncSession, project: Project) -> None:
    """Stitch completed video clips into final output.

    Queries all completed clips for the project, concatenates them in scene order
    using either concat demuxer (hard cuts) or xfade filter (crossfades),
    and saves the result to tmp/{project_id}/output/final.mp4.

    Updates project status to 'complete' on success or 'failed' on error.

    Args:
        session: Async database session
        project: Project object to stitch videos for

    Requirements:
        - STCH-01: Concat demuxer for hard cuts when crossfade_seconds=0.0
        - STCH-02: xfade filter for crossfades when crossfade_seconds>0.0
        - STCH-03: Audio streams preserved during concatenation
        - STCH-04: Output saved to tmp/{project_id}/output/final.mp4
        - STCH-05: ffmpeg validated at startup (handled by validate_dependencies)
    """
    file_mgr = FileManager()

    # Query completed clips in scene order (STCH-04)
    result = await session.execute(
        select(VideoClip)
        .join(Scene)
        .where(Scene.project_id == project.id)
        .where(VideoClip.status == "complete")
        .order_by(Scene.scene_index)
    )
    clips = result.scalars().all()

    # Handle case with no completed clips
    if not clips:
        logger.error(f"Project {project.id}: No completed clips available for stitching")
        project.status = "failed"
        project.error_message = "No completed video clips available for stitching"
        await session.commit()
        return

    # Get clip paths
    clip_paths = [Path(clip.local_path) for clip in clips]
    logger.info(f"Project {project.id}: Stitching {len(clip_paths)} clips")

    # Verify all clip files exist
    missing_clips = [p for p in clip_paths if not p.exists()]
    if missing_clips:
        logger.error(f"Project {project.id}: Missing clip files: {missing_clips}")
        project.status = "failed"
        project.error_message = f"Missing clip files: {[str(p) for p in missing_clips]}"
        await session.commit()
        return

    # Get output path (STCH-04)
    output_path = file_mgr.get_output_path(project.id, "final.mp4")

    # Choose stitching mode based on crossfade setting
    try:
        if settings.pipeline.crossfade_seconds == 0.0:
            # Hard cuts using concat demuxer (STCH-01)
            logger.info(f"Project {project.id}: Using concat demuxer (hard cuts)")
            await asyncio.to_thread(_stitch_concat_demuxer, clip_paths, output_path)
        else:
            # Crossfade transitions using xfade filter (STCH-02)
            logger.info(
                f"Project {project.id}: Using xfade filter "
                f"(crossfade={settings.pipeline.crossfade_seconds}s)"
            )
            await asyncio.to_thread(
                _stitch_with_crossfade,
                clip_paths,
                output_path,
                settings.pipeline.crossfade_seconds,
                project.target_clip_duration,
            )

        # Update project on success
        project.output_path = str(output_path)
        project.status = "complete"
        logger.info(f"Project {project.id}: Stitching complete -> {output_path}")
        await session.commit()

    except subprocess.CalledProcessError as e:
        # ffmpeg error
        stderr = e.stderr.decode() if e.stderr else "No error output"
        logger.error(f"Project {project.id}: ffmpeg error: {stderr}")
        project.status = "failed"
        project.error_message = f"Video stitching failed: {stderr[:500]}"
        await session.commit()
        raise

    except Exception as e:
        # Unexpected error
        logger.error(f"Project {project.id}: Stitching error: {e}")
        project.status = "failed"
        project.error_message = f"Stitching error: {str(e)[:500]}"
        await session.commit()
        raise


def _stitch_concat_demuxer(clip_paths: list[Path], output_path: Path) -> None:
    """Stitch videos using ffmpeg concat demuxer (hard cuts).

    Creates a concat list file and uses ffmpeg concat demuxer with stream copy
    to preserve original video/audio quality without re-encoding.

    Args:
        clip_paths: List of paths to video clips in order
        output_path: Path for final output video

    Implements:
        - STCH-01: Concat demuxer for hard cuts
        - STCH-03: Stream copy preserves audio
        - Pitfall 7: -safe 0 flag allows absolute paths
    """
    # Create concat list file (STCH-04 pattern)
    list_file = output_path.parent / "concat_list.txt"

    try:
        with open(list_file, "w") as f:
            for clip_path in clip_paths:
                # Use absolute paths for reliability (Pitfall 7)
                f.write(f"file '{clip_path.resolve()}'\n")

        # Run ffmpeg concat demuxer (STCH-01, STCH-03)
        # -safe 0: Allow absolute paths (critical for Pitfall 7)
        # -c copy: Stream copy preserves audio/video without re-encoding (STCH-03)
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-f",
                "concat",  # Use concat demuxer
                "-safe",
                "0",  # CRITICAL: Allow absolute paths
                "-i",
                str(list_file),
                "-c",
                "copy",  # Stream copy (no re-encoding, preserves audio)
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

        logger.info(f"Concat demuxer stitching complete: {output_path}")

    finally:
        # Clean up concat list file
        if list_file.exists():
            list_file.unlink()


def _stitch_with_crossfade(
    clip_paths: list[Path],
    output_path: Path,
    crossfade_duration: float,
    clip_duration: int,
) -> None:
    """Stitch videos with crossfade transitions using xfade filter.

    Creates smooth crossfade transitions between clips. Requires re-encoding
    as the xfade filter cannot work with stream copy.

    Args:
        clip_paths: List of paths to video clips in order
        output_path: Path for final output video
        crossfade_duration: Duration of crossfade in seconds
        clip_duration: Target duration of each clip in seconds

    Implements:
        - STCH-02: xfade filter for crossfade transitions
        - Pattern: Handle single clip edge case
    """
    # Handle single clip case - no crossfade needed
    if len(clip_paths) == 1:
        logger.info("Single clip detected, copying without crossfade")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(clip_paths[0]),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return

    # Build input arguments for all clips
    inputs = []
    for clip_path in clip_paths:
        inputs.extend(["-i", str(clip_path)])

    # Build xfade filter chain (STCH-02)
    # Each xfade takes two inputs and produces one output
    # Chain: [0:v][1:v]xfade[v01] ; [v01][2:v]xfade[v02] ; ...
    filter_parts = []
    prev_label = "0:v"

    for i in range(1, len(clip_paths)):
        out_label = f"v{i:02d}"
        # Calculate offset: where the transition should start
        # offset = (clip_duration * i) - (crossfade_duration * i)
        offset = (clip_duration * i) - (crossfade_duration * i)

        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:"
            f"duration={crossfade_duration}:offset={offset}[{out_label}]"
        )
        prev_label = out_label

    filter_complex = ";".join(filter_parts)

    # Run ffmpeg with xfade filter
    # Note: xfade requires re-encoding, cannot use -c copy
    # -vsync vfr: Variable frame rate to handle timing correctly
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{prev_label}]",  # Map final video output
            "-vsync",
            "vfr",  # Variable frame rate
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Crossfade stitching complete: {output_path}")
