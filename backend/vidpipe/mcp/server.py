"""FastMCP server exposing Vidpipe production workflows."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from vidpipe.mcp.workflow import (
    ProductionSpec,
    SceneSpec,
    ShotSpec,
    get_project_status,
    preflight,
    produce_project,
)


mcp = FastMCP(
    "vidpipe",
    instructions=(
        "Drive Vidpipe end-to-end video productions through the existing Vidpipe API. "
        "Use preflight before paid generation. For best narrative control, pass explicit "
        "scene_breakdown and shot_list entries; otherwise the server creates a seeded "
        "story structure from the story brief."
    ),
)


def _api_base(api_base: str | None = None) -> str:
    return api_base or os.getenv("VIDPIPE_MCP_API_BASE", "http://localhost:8100")


async def _ctx_progress(ctx: Context, message: str) -> None:
    await ctx.info(message)


@mcp.tool()
async def vidpipe_preflight(api_base: str | None = None) -> dict[str, Any]:
    """Check Vidpipe API connectivity and provider prerequisites."""

    return await preflight(_api_base(api_base))


@mcp.tool()
async def vidpipe_project_status(
    production_id: str,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Inspect a Vidpipe production, including screenplay, scenes, sound, and master."""

    return await get_project_status(production_id, _api_base(api_base))


@mcp.tool()
async def vidpipe_produce_project(
    title: str,
    story_brief: str,
    ctx: Context,
    genre: str = "Cinematic short",
    source_production_id: str | None = None,
    production_bible_id: str | None = None,
    tags: list[str] | None = None,
    scene_count: int = 3,
    shots_per_scene: int = 2,
    clip_duration: int = 8,
    style: str = "cinematic",
    aspect_ratio: str = "16:9",
    text_model: str = "gemini-3.5-flash",
    vision_model: str | None = "gemini-3.5-flash",
    image_model: str = "flux-2-klein",
    video_model: str = "ltx-2.3-flf2v",
    audio_enabled: bool = True,
    generate_voice: bool = True,
    generate_sound: bool = True,
    render_master: bool = True,
    create_default_narrator_bible: bool = True,
    scene_breakdown: list[dict[str, Any]] | None = None,
    shot_list: list[dict[str, Any]] | None = None,
    character_breakdowns: list[dict[str, Any]] | None = None,
    script: str | None = None,
    logline: str | None = None,
    treatment: str | None = None,
    poll_interval_seconds: float = 20.0,
    scene_timeout_seconds: int = 3600,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Create and fully produce a Vidpipe project from a story brief or explicit screenplay.

    This is the high-level E2E tool. It creates a production, seeds/locks the
    screenplay, creates scenes, runs storyboard/keyframe/video generation for
    each scene, optionally generates narration and SFX, and optionally renders a
    final master MP4.
    """

    parsed_scene_breakdown = [
        SceneSpec.model_validate(scene)
        for scene in (scene_breakdown or [])
    ] or None
    parsed_shot_list = [
        ShotSpec.model_validate(shot)
        for shot in (shot_list or [])
    ] or None
    spec = ProductionSpec(
        title=title,
        story_brief=story_brief,
        genre=genre,
        logline=logline,
        treatment=treatment,
        character_breakdowns=character_breakdowns,
        scene_breakdown=parsed_scene_breakdown,
        shot_list=parsed_shot_list,
        script=script,
        source_production_id=source_production_id,
        production_bible_id=production_bible_id,
        tags=tags or [],
        scene_count=scene_count,
        shots_per_scene=shots_per_scene,
        clip_duration=clip_duration,
        style=style,
        aspect_ratio=aspect_ratio,
        text_model=text_model,
        vision_model=vision_model,
        image_model=image_model,
        video_model=video_model,
        audio_enabled=audio_enabled,
        generate_voice=generate_voice,
        generate_sound=generate_sound,
        render_master=render_master,
        create_default_narrator_bible=create_default_narrator_bible,
        poll_interval_seconds=poll_interval_seconds,
        scene_timeout_seconds=scene_timeout_seconds,
    )
    return await produce_project(spec, _api_base(api_base), progress=lambda msg: _ctx_progress(ctx, msg))


@mcp.tool()
async def vidpipe_continue_production(
    source_production_id: str,
    sequel_title: str,
    continuation_brief: str,
    ctx: Context,
    tags: list[str] | None = None,
    scene_count: int = 3,
    shots_per_scene: int = 2,
    clip_duration: int = 8,
    text_model: str = "gemini-3.5-flash",
    vision_model: str | None = "gemini-3.5-flash",
    image_model: str = "flux-2-klein",
    video_model: str = "ltx-2.3-flf2v",
    api_base: str | None = None,
) -> dict[str, Any]:
    """Create a sequel production using the source production's Production Bible."""

    source_status = await get_project_status(source_production_id, _api_base(api_base))
    source_screenplay = source_status.get("screenplay") or {}
    story_brief = (
        f"Continue this source production: {source_screenplay.get('title') or source_production_id}. "
        f"Prior logline: {source_screenplay.get('logline') or 'unknown'}. "
        f"Continuation: {continuation_brief}"
    )
    spec = ProductionSpec(
        title=sequel_title,
        story_brief=story_brief,
        genre=source_screenplay.get("genre") or "Cinematic short",
        source_production_id=source_production_id,
        tags=tags or ["sequel"],
        scene_count=scene_count,
        shots_per_scene=shots_per_scene,
        clip_duration=clip_duration,
        text_model=text_model,
        vision_model=vision_model,
        image_model=image_model,
        video_model=video_model,
    )
    return await produce_project(spec, _api_base(api_base), progress=lambda msg: _ctx_progress(ctx, msg))


def main() -> None:
    """Run the MCP server over stdio by default."""

    transport = os.getenv("VIDPIPE_MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)

