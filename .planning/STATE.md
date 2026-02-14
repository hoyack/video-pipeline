# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accept a text prompt and produce a cohesive, multi-scene short video with visual continuity — fully automated, crash-safe, and resumable.
**Current focus:** Phase 2: Generation Pipeline

## Current Position

Phase: 2 of 3 (Generation Pipeline)
Plan: 2 of 4 in current phase
Status: In Progress
Last activity: 2026-02-14 — Completed plan 02-02 (Keyframe Generation)

Progress: [████░░░░░░] 33%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: 2.4 min
- Total execution time: 0.20 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 3 | 7.0 min | 2.3 min |
| 02-generation-pipeline | 2 | 5.3 min | 2.7 min |

**Recent Trend:**
- Last 5 plans: 01-02 (2min), 01-03 (2.5min), 02-01 (3.6min), 02-02 (1.7min)
- Trend: Faster completion for focused implementation tasks

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

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1:**
- ~~SQLite WAL mode must be enabled from first migration to prevent database corruption during crashes~~ ✓ Resolved in 01-03

**Phase 2:**
- Rate limiting on Vertex AI free tier may require quota increase or billing enablement for production use
- FFmpeg must be validated at startup to provide clear error before any generation work
- ADC authentication requires GOOGLE_APPLICATION_CREDENTIALS environment variable in production

**Phase 3:**
- Cost estimation ($15 per 5-scene project) should be communicated to users before generation starts

## Session Continuity

Last session: 2026-02-14 (plan execution)
Stopped at: Completed 02-02-PLAN.md - Keyframe Generation
Resume file: None
