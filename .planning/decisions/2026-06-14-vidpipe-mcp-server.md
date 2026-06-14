# ADR: Vidpipe MCP server drives production workflows through the public API

**Date:** 2026-06-14
**Status:** Accepted

## Context

Vidpipe now has enough production-level API surface to let an agent create a
story, seed a screenplay, generate scenes, produce clips, mix audio, render a
master, and inspect the result. Codex needs a stable integration point that can
drive those workflows without depending on database internals or Python service
objects whose signatures change more freely.

The existing browser E2E path already exercises the HTTP API. Using that same
surface for MCP keeps agent-driven runs aligned with UI behavior and avoids a
second orchestration contract.

## Decisions

### 1. FastMCP server packaged under `vidpipe.mcp`

The MCP server is a backend package module, runnable with:

```bash
python -m vidpipe.mcp
```

It uses stdio by default for local Codex integration and supports the SDK's
streamable HTTP transport via `VIDPIPE_MCP_TRANSPORT`.

### 2. Public HTTP API is the only production boundary

The MCP workflow calls `/api/*` endpoints through a thin async HTTP client.
It does not import DB models, call orchestrator internals, or mutate storage
directly. This makes MCP runs behave like UI and REST clients, including the
same validation, status transitions, recovery behavior, and provider settings.

### 3. High-level tools cover safe setup, production, continuation, and status

The server exposes:

- `vidpipe_preflight` for API/provider readiness checks before paid work.
- `vidpipe_project_status` for production, screenplay, scene, audio, and
  master inspection.
- `vidpipe_produce_project` for creating and fully producing a project from a
  story brief or explicit screenplay structure.
- `vidpipe_continue_production` for sequel runs that reuse a source
  production's Production Bible when available.

### 4. Generated defaults are story-seeded but conservative

When a caller omits explicit screenplay structure, the workflow creates a
deterministic scene breakdown, shot list, treatment, and narrator script from
the story brief and requested length. Voice-only narrators are not marked as
on-screen characters. Source productions may contribute visible character tags
when continuing an existing story.

## Consequences

- MCP clients can drive complete productions without new API endpoints.
- Very long productions run synchronously from the MCP client's point of view;
  progress is emitted through MCP context logging, and status can be inspected
  separately with `vidpipe_project_status`.
- Provider-spending tools are explicit and documented; `vidpipe_preflight` is
  the recommended first call.
- Any future background-job version of MCP production should still use the
  HTTP API unless the API itself gains a durable production-run abstraction.

## Verification

- `pytest backend/tests/test_mcp_workflow.py -q`
- `ruff check backend/vidpipe/mcp backend/tests/test_mcp_workflow.py`
- `python -m compileall backend/vidpipe/mcp`
- Stdio MCP smoke test initialized `python -m vidpipe.mcp`, listed all four
  tools, and read production `06238bb4-8665-4b01-b097-4dce61a7e2aa` through
  `vidpipe_project_status`.
