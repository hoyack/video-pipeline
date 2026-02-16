# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accept a text prompt and produce a cohesive, multi-scene short video with visual continuity — fully automated, crash-safe, and resumable.
**Current focus:** Phase 4: Manifest System Foundation (V2 pipeline evolution)

## Current Position

Phase: 4 of 12 (Manifest System Foundation)
Plan: 1 of ? in current phase
Status: In progress
Last activity: 2026-02-16 — Completed 04-01: Manifest System Backend Foundation

Progress: [███░░░░░░░] 25% (3 of 12 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 2.3 min
- Total execution time: 0.44 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 3 | 7.0 min | 2.3 min |
| 02-generation-pipeline | 4 | 8.0 min | 2.0 min |
| 03-orchestration-interfaces | 3 | 6.0 min | 2.0 min |
| 04-manifest-system-foundation | 1 | 5.4 min | 5.4 min |

**Recent Trend:**
- Last 5 plans: 03-01 (2.2min), 03-02 (1.9min), 03-03 (1.9min), 04-01 (5.4min)
- Trend: Phase 4 in progress - building V2 manifest system

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Phase 1: Gemini structured output over CrewAI for simpler v1 implementation
- Phase 1: SQLite over cloud DB for local-first tool with no multi-user needs
- Phase 2: Sequential keyframes required for visual continuity across scenes
- Phase 2: Vertex AI over Gemini API for consistent ADC auth
- **01-01:** Used SQLAlchemy 2.0 Mapped[Type] annotations for type safety
- **01-01:** Defined foreign key relationships at database level for referential integrity
- **01-01:** Created modular package structure with db/, services/, pipeline/, schemas/ subdirectories
- **01-02:** Used YamlConfigSettingsSource custom source for YAML loading instead of dotenv approach
- **01-02:** Nested config models inherit from BaseModel (not BaseSettings) per pydantic-settings best practices
- **01-02:** Environment variables use __ delimiter for nested config (VIDPIPE_PIPELINE__MAX_SCENES)
- **01-02:** Hardcoded config.yaml path in YamlConfigSettingsSource for simplicity
- **01-03:** Use engine.sync_engine for event listener to support aiosqlite wrapper
- **01-03:** Set expire_on_commit=False to prevent greenlet errors in async context
- **01-03:** Use synchronous=FULL for maximum crash safety per FOUND-04 requirement
- **01-03:** Implement path traversal protection using is_relative_to() method
- **01-03:** Use metadata.create_all() instead of Alembic for v1 simplicity
- **02-01:** Used google-genai SDK in Vertex AI mode with ADC for unified authentication
- **02-01:** Implemented tenacity retry with temperature reduction (0.7 → 0.55 → 0.4) on JSON failures
- **02-01:** Corrected model names: gemini-2.0-flash-exp, imagen-3.0-generate-001, veo-2.0-generate-001
- **02-01:** Applied singleton pattern to vertex_client to avoid repeated client initialization
- **02-02:** Used image-conditioned generation for end frames to maintain visual style and composition
- **02-02:** Commit after each scene (not at end) for crash recovery and resumability
- **02-02:** Applied jitter to retry backoff to prevent thundering herd on rate limit errors
- **02-02:** Scene 0 start frame from text alone, all other start frames inherited from previous end frame
- **02-03:** Persist operation_name to database BEFORE polling starts for crash recovery
- **02-03:** Use async sleep in polling loop to avoid blocking event loop
- **02-03:** Mark RAI-filtered clips and continue pipeline rather than crashing
- **02-03:** Resume polling from clip.poll_count for idempotent crash recovery
- **02-04:** Used concat demuxer with -safe 0 flag for absolute path support in concat list
- **02-04:** Stream copy (-c copy) for concat demuxer to preserve audio quality without re-encoding
- **02-04:** Wrapped subprocess.run() in asyncio.to_thread() to prevent event loop blocking
- **02-04:** Validate ffmpeg at startup rather than during pipeline execution for fail-fast error handling
- [Phase 03-01]: Use completed_steps dict from database queries for failed state resume logic
- [Phase 03-01]: Fix status mismatch between generate_keyframes (generating_video) and state machine (video_gen) in orchestrator
- [Phase 03-02]: Use asyncio.run() wrapper pattern for Typer + async database operations
- [Phase 03-02]: Implement progress_callback wrapper to update Rich status spinners from orchestrator
- [Phase 03-02]: Temporarily override settings.pipeline.crossfade_seconds in stitch command for per-invocation control
- [Phase 03-03]: Use asynccontextmanager lifespan instead of deprecated @app.on_event
- [Phase 03-03]: Create fresh async_session() in background tasks (never share request session)
- [Phase 03-03]: APIRouter with /api prefix for route organization
- [Phase 03-03]: Exclude output_path from response schemas for security
- [Phase 03-03]: FileResponse with Content-Disposition attachment header for MP4 downloads
- [Phase 03-03]: Generic exception handler returns 500 with detail (prevents stack trace leakage)
- **04-01:** Assets belong to manifests only (no project_id column) per V2 architecture
- **04-01:** Soft delete for manifests (deleted_at column) prevents data loss
- **04-01:** Auto-generate manifest_tag on asset creation (CHAR_01, CHAR_02, OBJ_01)
- **04-01:** Explicit index on Asset.manifest_id for query performance on SQLite
- **04-01:** Image upload saves to tmp/manifests/{manifest_id}/uploads/ directory structure
- **04-01:** Return 409 Conflict when deleting manifest referenced by projects

### Roadmap Evolution

- Phases 4-12 added: V2 studio-grade pipeline (manifest system, manifesting engine, CV analysis, adaptive prompts, multi-candidate scoring, fork integration)
- Reference docs: `docs/v2-manifest.md`, `docs/v2-pipe-optimization.md`

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1:**
- ~~SQLite WAL mode must be enabled from first migration to prevent database corruption during crashes~~ ✓ Resolved in 01-03

**Phase 2:**
- Rate limiting on Vertex AI free tier may require quota increase or billing enablement for production use
- ADC authentication requires GOOGLE_APPLICATION_CREDENTIALS environment variable in production
- ffmpeg must be installed on deployment environment (validated at startup with clear error message)

**Phase 3:**
- ~~Cost estimation ($15 per 5-scene project) should be communicated to users before generation starts~~ ✓ Resolved in 03-02 (cost warning in CLI)

## Session Continuity

Last session: 2026-02-16 (plan execution)
Stopped at: Completed 04-01-PLAN.md
Resume file: None
