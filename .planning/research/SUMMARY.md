# Project Research Summary

**Project:** AI Video Generation Pipeline
**Domain:** AI-powered multi-scene video generation using Google Vertex AI
**Researched:** 2026-02-14
**Confidence:** MEDIUM-HIGH

## Executive Summary

AI video generation pipelines in 2026 follow a **state-driven, multi-stage architecture** where long text prompts are transformed into coherent multi-scene videos through orchestrated LLM storyboarding, sequential keyframe generation, and frame-guided video synthesis. The recommended approach uses **Python 3.12 + FastAPI + SQLAlchemy 2.0** for the application layer, **Google Vertex AI** (Gemini 3 Pro for storyboarding, Veo 3.1 for video generation), and **FFmpeg** for stitching, with SQLite providing crash-safe state persistence. This is a **long-running async pipeline** (5-15 minutes per multi-scene video) requiring careful state management, rate limiting, and cost control.

The critical architectural decision is implementing a **state machine with persistent checkpointing** from day one. Video generation is expensive ($2-3 per scene), time-consuming (30-120s per clip), and prone to both transient failures (API timeouts, rate limits) and permanent failures (content filter rejections). Without crash recovery, a single failure wastes minutes of generation time and dollars in API costs. The recommended stack supports async-first patterns with SQLAlchemy 2.0's async support, aiosqlite for non-blocking database operations, and google-genai SDK (the official replacement for deprecated Vertex AI libraries).

Key risks center on **cost control** (naive polling burns API quota), **visual continuity** (independently-generated scenes look disjointed without keyframe management), and **error handling** (RAI content rejections must be distinguished from transient failures). Mitigation strategies include exponential backoff polling (10-60s intervals), sequential keyframe generation with continuity context, idempotent operation tracking to prevent duplicate charges, and FFmpeg validation to catch format mismatches before stitching failures.

## Key Findings

### Recommended Stack

Python 3.12 provides the best balance of ecosystem stability and performance for 2026 projects, with security support through 2028. The stack centers on **async-first patterns** essential for long-running video generation operations.

**Core technologies:**
- **Python 3.12**: Runtime with maximum ecosystem compatibility and active security support through 2028
- **FastAPI 0.129.0+**: Async HTTP API server with Pydantic v2 integration (5-50x faster than v1), auto-generated OpenAPI docs
- **SQLAlchemy 2.0.46+**: ORM with first-class async support, essential for non-blocking database operations in async pipeline
- **aiosqlite 0.22.1+**: Async SQLite driver enabling database operations on AsyncIO event loop without blocking
- **google-genai 1.63.0+**: Official GA SDK for Vertex AI (replaces deprecated vertexai.generative_models and google-generativeai)
- **FFmpeg 6.0+**: Industry-standard video processing binary, called via subprocess (ffmpeg-python is unmaintained since 2019)
- **pydantic-settings 2.12.0+**: Type-safe configuration management with automatic validation from environment variables
- **structlog 25.5.0+**: Structured JSON logging for production, context variables for request tracing through async pipeline flows

**Critical architectural choices:**
- Use **subprocess module directly** for FFmpeg calls, not unmaintained wrappers
- Enable **SQLite WAL mode** with `PRAGMA synchronous=FULL` for crash safety
- Deploy with **gunicorn + uvicorn workers** (workers = CPU cores, not 2N+1 formula for async)
- Set **expire_on_commit=False** in SQLAlchemy async sessions to avoid implicit queries

### Expected Features

AI video generation pipelines must handle **long-running operations** (5-15 minutes), **expensive API calls** ($2-3 per scene), and **multi-stage dependencies** (storyboard → keyframes → videos → stitching).

**Must have (table stakes):**
- **Text-to-video with multi-scene storyboarding** — Core value proposition; transforms prompt into coherent narrative structure
- **Async job queue + progress tracking** — Essential for usability; blocking CLI unacceptable for minute-long operations
- **Crash recovery via SQLite state** — Prevents wasted API costs; resume from last successful stage
- **Scene stitching with transitions** — Users expect single playable video file, not individual clips
- **First/last frame keyframe control** — Ensures visual continuity between scenes; Veo 3.1 native capability
- **1080p output (MP4)** — Table stakes resolution; single format sufficient for launch
- **Error handling with clear messages** — Users must understand API failures, quota exhaustion, RAI content rejections

**Should have (competitive advantage):**
- **Batch/parallel scene generation** — Generate multiple scenes simultaneously for 3-5x speedup
- **Cost estimation before generation** — Calculate based on scene count/duration/resolution to prevent surprise bills
- **Preview/dry-run mode** — Review storyboard before committing expensive compute
- **Prompt template library** — Reusable JSON-based prompt components for consistent style
- **Comprehensive logging/observability** — Full trace IDs and span tracking essential for debugging multi-stage pipelines

**Defer (v2+):**
- **Character consistency with reference images** — High complexity; requires reference anchoring and cross-frame tracking (industry-wide unsolved challenge)
- **Intelligent scene planning (RAG-based)** — Multi-agent workflow like ViMax for auto-segmenting long narratives
- **Audio generation with visual sync** — Veo 3.1 supports but not critical for MVP validation
- **Web UI** — CLI + HTTP API sufficient for early adopters; defer frontend until demand proven

### Architecture Approach

The standard pattern is a **state machine pipeline** with explicit transitions (PENDING → STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE), where each stage persists artifacts to filesystem and updates database state atomically. This enables crash recovery by resuming from the last successful checkpoint.

**Major components:**
1. **State Machine Controller** — Validates state transitions, persists job state to SQLite, implements retry logic with exponential backoff
2. **Storyboard Generator** — Uses Gemini 3 Pro LLM to parse text prompt into structured JSON (scenes, transitions, characters, settings)
3. **Keyframe Generator** — Generates images for each scene **sequentially** (not parallel) using Nano Banana Pro, passing previous keyframe as continuity context
4. **Video Generator** — Converts keyframe pairs to video clips using Veo 3.1 frame-guided generation, potentially parallelized (5 concurrent max)
5. **Stitch Engine** — Concatenates clips with FFmpeg, validates format consistency (resolution, framerate, codec) before concatenation
6. **Artifact Store** — Manages filesystem storage with job-scoped directories (`artifacts/jobs/{job_id}/`), stores paths + metadata in database (not BLOBs)

**Critical patterns:**
- **Async API with polling** — Submit to Veo, receive operation ID, poll with exponential backoff (10-60s intervals), never block on sync waits
- **Sequential keyframe generation** — Generate scene N, extract final frame, use as reference for scene N+1 to maintain visual continuity (slower but coherent)
- **Artifact path references in database** — Store 50MB videos on filesystem, track paths in DB to keep queries fast and enable cloud storage migration
- **Retry with exponential backoff** — Use tenacity library (max 5 retries, 1s → 60s backoff) to handle transient API failures vs permanent RAI rejections

### Critical Pitfalls

1. **Naive polling causes cost explosions** — Polling Veo every 5 seconds for 60-second operations burns API quota and hits rate limits. **Fix:** Exponential backoff starting at 10s, doubling to 60s max, store next_poll_time in SQLite to prevent duplicate requests.

2. **No idempotency = duplicate video charges** — Retry logic submits new generation requests instead of checking if original operation is still running, wasting $2-3 per duplicate. **Fix:** Store operation_id in SQLite immediately after submission, check for in-progress operations before creating new ones.

3. **Synchronous FFmpeg blocks entire pipeline** — Running FFmpeg in main thread freezes pipeline for 10-30+ seconds per stitch. **Fix:** Run FFmpeg in ThreadPoolExecutor or asyncio subprocess, track stitching status in database for crash recovery.

4. **Not handling RAI content rejections** — Veo's Responsible AI filters reject 10-30% of prompts containing prohibited content (violence, public figures, sensitive topics), causing pipeline crashes or incomplete videos. **Fix:** Parse error responses to distinguish RAI rejections from transient failures, implement automatic prompt rewriting with max retry limit.

5. **SQLite database corruption from crashes** — Concurrent writes during crash (Ctrl+C, OOM) corrupt database, losing all state. **Fix:** Enable WAL mode (`PRAGMA journal_mode=WAL`), set `synchronous=FULL` for disk sync on commits, wrap updates in explicit transactions.

6. **FFmpeg format mismatches = silent frame loss** — Stitching clips with different codecs/framerates/resolutions causes FFmpeg to drop frames silently. **Fix:** Validate every clip before concatenation (ffprobe), normalize to identical specs (1280x720, 24fps, libx264) with re-encoding.

7. **No visual continuity strategy** — Independent scene generation causes characters to change appearance, locations to shift randomly, lighting to vary wildly. **Fix:** Sequential keyframe generation with continuity context, maintain style guide in prompts, extract final frame from scene N as reference for scene N+1.

8. **Hard-coded prompt limits cause silent truncation** — Veo has undocumented 500-1000 character limits; exceeding causes cryptic errors or dropped context. **Fix:** Validate prompt length before submission (cap at 500 chars), prioritize essential elements, log original vs. truncated prompts.

## Implications for Roadmap

Based on research, suggested phase structure prioritizes **crash-safe state management** and **cost control** before feature expansion.

### Phase 1: Foundation (Database + State Machine)
**Rationale:** Everything depends on state persistence. Build foundation before generators to enable crash recovery from day one and prevent wasted API costs.

**Delivers:**
- SQLAlchemy models (Job, Artifact) with state machine validation
- SQLite database with WAL mode and crash-safe configuration
- ArtifactStore for filesystem + database integration
- Job-scoped directory structure preventing parallel job collisions

**Addresses Features:**
- Crash recovery via SQLite state (table stakes)
- Error handling with state persistence (table stakes)

**Avoids Pitfalls:**
- SQLite database corruption (enable WAL + synchronous=FULL from start)
- No idempotency (operation tracking in database prevents duplicate charges)

**Research flags:** Standard patterns, skip phase research.

### Phase 2: Service Clients (Vertex AI Integration)
**Rationale:** Abstract external APIs early to enable testing with mocks and implement retry/polling logic consistently across all generators.

**Delivers:**
- Vertex AI base client with authentication, exponential backoff, polling patterns
- Gemini client for LLM-based storyboard generation
- Nano Banana client for keyframe image generation
- Veo client for frame-guided video generation

**Uses Stack:**
- google-genai 1.63.0+ (official GA SDK)
- tenacity for exponential backoff retry logic
- httpx for async HTTP if needed beyond google-genai

**Avoids Pitfalls:**
- Naive polling (exponential backoff 10-60s built into clients)
- No retry logic (tenacity wraps all API calls with max 5 retries)
- Hard-coded prompt limits (validation before submission)

**Research flags:** May need deeper research on Vertex AI rate limits, quota management, and operation polling patterns.

### Phase 3: Generators (Pipeline Stages)
**Rationale:** Build generators in pipeline order (storyboard → keyframes → video → stitch) to test each stage independently before integration. Sequential keyframe generation is critical for visual continuity.

**Delivers:**
- StoryboardGenerator (Gemini integration, structured JSON output)
- KeyframeGenerator (Nano Banana + sequential generation with continuity context)
- VideoGenerator (Veo + frame-guided generation, potentially parallel)
- StitchEngine (FFmpeg integration with format validation)

**Implements Architecture:**
- Sequential keyframe generation with previous frame as reference
- Async API polling with operation tracking
- Artifact path references (save to filesystem, store paths in DB)

**Avoids Pitfalls:**
- No visual continuity (sequential generation with style guide prompts)
- FFmpeg format mismatches (ffprobe validation + normalization before concat)
- Synchronous FFmpeg blocking (run in ThreadPoolExecutor)
- RAI content rejections (error classification, prompt rewriting)

**Research flags:** Needs phase research for FFmpeg stitching patterns, transition effects, and format normalization strategies.

### Phase 4: Orchestration (State Machine + Resume Logic)
**Rationale:** Orchestrator ties all generators together, coordinating state transitions and implementing crash recovery. Build after generators exist to test with real components.

**Delivers:**
- Orchestrator coordinating state machine + generator execution
- Resume logic detecting completed work and skipping stages
- Progress tracking with per-scene status updates
- Cost tracking and quota management

**Uses:**
- State machine from Phase 1
- Service clients from Phase 2
- Generators from Phase 3

**Avoids Pitfalls:**
- No crash recovery (resume from last successful state checkpoint)
- Cost tracking missing (calculate and check before each operation)

**Research flags:** Standard orchestration patterns, skip phase research.

### Phase 5: Interfaces (CLI + HTTP API)
**Rationale:** Interfaces are thin wrappers around orchestrator. Build last to focus on core pipeline functionality first.

**Delivers:**
- Typer CLI commands (start, resume, status, cancel)
- Rich progress bars with per-scene status
- FastAPI HTTP endpoints (optional, enables web frontends)
- Error message formatting (RAI rejections → user-friendly guidance)

**Uses Stack:**
- Typer 0.23.1+ for CLI
- Rich 14.3.2+ for progress visualization
- FastAPI 0.129.0+ for optional HTTP API

**Addresses Features:**
- Progress tracking (table stakes)
- Clear error messages (table stakes)
- CLI interface (table stakes for v1)
- HTTP API (competitive advantage, v1.x)

**Research flags:** Standard patterns, skip phase research.

### Phase 6: Enhancements (Cost Control + Templates)
**Rationale:** After core pipeline works, add user-facing features that improve usability and reduce iteration friction.

**Delivers:**
- Cost estimation before generation (scene count × duration × resolution)
- Preview/dry-run mode (review storyboard, skip video generation)
- Prompt template library (JSON-based reusable modules)
- Batch/parallel scene generation (up to 5 concurrent Veo operations)

**Addresses Features:**
- Cost estimation (competitive advantage)
- Preview mode (competitive advantage)
- Prompt templates (competitive advantage)
- Batch processing (competitive advantage)

**Research flags:** Standard patterns, skip phase research.

### Phase Ordering Rationale

- **Phase 1 before all others:** Crash recovery is non-negotiable due to expensive API costs ($2-3 per scene). Database corruption risk is highest during development when crashes are common.

- **Service clients before generators:** Abstracting Vertex AI enables testing generators with mocks, avoiding API costs during development. Retry and polling logic must be consistent across all API calls.

- **Generators in pipeline order:** Storyboard must complete before keyframes, keyframes before videos, videos before stitching. Testing each stage independently validates output before passing to next stage.

- **Orchestration after generators:** Can't build state machine coordinator until generators exist to coordinate. Resume logic requires real artifacts to detect completion.

- **Interfaces last:** CLI/API are thin layers over orchestrator. Building early wastes time on UI polish before core pipeline validates.

- **Enhancements deferred:** Cost estimation, preview, templates improve UX but aren't required to validate core hypothesis: "Can we generate coherent multi-scene videos from text prompts?"

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 2 (Service Clients):** Vertex AI rate limits, quota management, operation state polling, GCS URL expiry handling
- **Phase 3 (Generators - Stitching):** FFmpeg transition effects (xfade filter syntax), format normalization strategies, audio handling

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Foundation):** SQLAlchemy state machines, SQLite WAL configuration well-documented
- **Phase 4 (Orchestration):** Standard async pipeline orchestration
- **Phase 5 (Interfaces):** Typer CLI and FastAPI patterns well-established
- **Phase 6 (Enhancements):** Cost calculation, template systems are straightforward

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All recommendations from official PyPI pages and documentation. Version numbers verified as of 2026-02-14. google-genai GA status confirmed. |
| Features | MEDIUM | Based on competitor analysis (Runway, Luma, Pika) and domain research. MVP definition derived from multiple sources but not validated with users. Character consistency flagged as high-complexity across all sources. |
| Architecture | MEDIUM-HIGH | Multi-stage state machine pattern confirmed across multiple AI video generation systems. Async polling, sequential keyframe generation, and artifact storage patterns verified in academic papers (ViMax, STAGE, DreamFactory). |
| Pitfalls | MEDIUM | Based on Veo documentation, Vertex AI best practices, and community reports. Specific error rates (10-30% RAI rejection) are estimates. Cost numbers ($2-3 per scene) verified from official pricing. |

**Overall confidence:** MEDIUM-HIGH

Stack and architecture patterns have strong documentation. Features and pitfalls are based on secondary sources (competitor analysis, community experience) with some official documentation (Vertex AI long-running operations, Veo API).

### Gaps to Address

- **Veo 3.1 actual generation times:** Documentation doesn't specify typical latencies; estimates (30-120s) based on predecessor models and community reports. **Action:** Time first generations during Phase 2 to calibrate polling intervals.

- **RAI rejection rate:** 10-30% estimate from community sources, not official data. **Action:** Track rejection rate in production, adjust prompt validation thresholds based on real data.

- **Character consistency techniques:** Research shows this is unsolved industry-wide. Sequential keyframe generation with style guides is best available approach, not guaranteed solution. **Action:** Defer to v2+, validate feasibility with small-scale tests before committing to roadmap.

- **Optimal scene count per video:** Research doesn't specify sweet spot for coherence vs. cost. **Action:** Test with 3-scene, 5-scene, and 10-scene videos during Phase 3 to determine practical limits.

- **FFmpeg transition quality:** Concat vs. xfade filter trade-offs not quantified. **Action:** Phase 3 research should include FFmpeg transition testing with Veo-generated clips to assess visual quality.

- **SQLite concurrency limits:** While WAL mode supports concurrent reads, multi-worker write contention point is unclear. **Action:** If scaling beyond 10 concurrent jobs, monitor "database locked" errors and plan PostgreSQL migration.

## Sources

### Primary (HIGH confidence)
- [FastAPI PyPI](https://pypi.org/project/fastapi/) — Version 0.129.0, Python 3.10+ support, Pydantic integration
- [SQLAlchemy PyPI](https://pypi.org/project/sqlalchemy/) — Version 2.0.46, async features
- [google-genai PyPI](https://pypi.org/project/google-genai/) — Version 1.63.0, GA status, migration timeline
- [Veo on Vertex AI Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation) — Official API reference, long-running operations
- [SQLite Locking Documentation](https://sqlite.org/lockingv3.html) — WAL mode, concurrency, crash safety

### Secondary (MEDIUM confidence)
- [ViMax: Agentic Video Generation (GitHub)](https://github.com/HKUDS/ViMax) — Multi-scene architecture patterns
- [STAGE: Storyboard-Anchored Generation (arXiv)](https://arxiv.org/html/2512.12372v1) — Sequential keyframe generation
- [DreamFactory: Multi-Scene Long Video (arXiv)](https://arxiv.org/html/2408.11788) — Visual continuity techniques
- [Runway vs Luma vs Pika Comparison](https://skywork.ai/blog/veo-3-1-vs-runway-vs-pika-vs-luma-2025-comparison/) — Competitor feature analysis
- [Common Veo 3.1 Mistakes (Vmake AI)](https://vmake.ai/blog/common-mistakes-when-using-veo-3-1-how-to-get-the-best-results) — Pitfall identification
- [Vertex AI Long-Running Operations](https://cloud.google.com/vertex-ai/docs/general/long-running-operations) — Polling patterns

### Tertiary (LOW confidence, needs validation)
- Cost estimates ($2-3 per scene) — Derived from pricing pages, needs validation with actual usage
- RAI rejection rates (10-30%) — Community reports, not official statistics
- Generation time estimates (30-120s) — Inferred from predecessor models, needs measurement

---

*Research completed: 2026-02-14*
*Ready for roadmap: YES*
