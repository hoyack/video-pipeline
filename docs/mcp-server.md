# Vidpipe MCP Server

Vidpipe includes a Model Context Protocol server that lets an MCP client drive
the same production workflow exposed by the web UI and REST API.

## What It Can Do

- Check runtime prerequisites before paid generation.
- Create a production from a story brief or explicit screenplay structure.
- Reuse an existing production bible, or continue from a source production.
- Generate screenplay-linked scenes.
- Run storyboard, keyframe, and video generation scene by scene.
- Generate and mix narration and sound effects.
- Render a final production master MP4.

The server calls the existing Vidpipe HTTP API rather than using private
internals. Keep the app stack running before connecting an MCP client.

## Run It

Install backend dependencies after pulling this change:

```bash
pip install -e "backend/[dev]"
```

Start Vidpipe normally:

```bash
./scripts/start-stack.sh --build
```

Run the MCP server over stdio:

```bash
VIDPIPE_MCP_API_BASE=http://localhost:8100 python -m vidpipe.mcp
```

For an HTTP MCP transport instead:

```bash
VIDPIPE_MCP_TRANSPORT=streamable-http \
VIDPIPE_MCP_API_BASE=http://localhost:8100 \
python -m vidpipe.mcp
```

## Codex Configuration

Add a server entry that runs the stdio command above. A typical local config
looks like:

```toml
[mcp_servers.vidpipe]
command = "python"
args = ["-m", "vidpipe.mcp"]
env = { VIDPIPE_MCP_API_BASE = "http://localhost:8100" }
```

Use `http://localhost:8180` for `VIDPIPE_MCP_API_BASE` if you want requests to
go through the frontend nginx proxy instead of the backend port directly.

## Tools

### `vidpipe_preflight`

Checks API reachability and provider configuration. Run this before any tool
that may spend provider credits.

### `vidpipe_project_status`

Returns production metadata, screenplay, scenes, sound deck, master metadata,
and browser URLs for a production.

### `vidpipe_produce_project`

Creates and fully produces a project. It accepts:

- `title`
- `story_brief`
- optional explicit `scene_breakdown`
- optional explicit `shot_list`
- optional `script`, `logline`, `treatment`, and `character_breakdowns`
- model settings and length controls

For best narrative control, pass explicit `scene_breakdown` and `shot_list`.
If they are omitted, the server creates a seeded scene structure from the story
brief and requested `scene_count`.

### `vidpipe_continue_production`

Creates a sequel production from a source production. It reuses the source
production bible when available, includes source screenplay context, and then
runs the same E2E production workflow.

## Safety

These tools can trigger paid provider calls. Keep Vidpipe on localhost or behind
auth, and prefer `vidpipe_preflight` plus small `scene_count` / `shots_per_scene`
values when testing a new MCP client.
