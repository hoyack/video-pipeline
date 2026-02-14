# Technology Stack

**Project:** AI Video Generation Pipeline
**Researched:** 2026-02-14
**Confidence:** HIGH

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.12+ | Runtime environment | 3.12 recommended for new projects in 2026, with performance improvements over 3.11, active security support until 2028, and broad library compatibility. 3.13 also viable but 3.12 has maximum ecosystem stability. |
| FastAPI | 0.129.0+ | HTTP API server | Industry standard for async Python APIs in 2026, supports Python 3.10-3.14, integrates seamlessly with Pydantic v2 (5-50x faster than v1), excellent async performance, auto-generated OpenAPI docs. |
| Typer | 0.23.1+ | CLI framework | Built by FastAPI author with same type-hint philosophy, integrates with Rich for beautiful terminal output, automatic help generation, minimal boilerplate. |
| Rich | 14.3.2+ | Terminal UI | Production-stable library for progress bars, tables, syntax highlighting, and beautiful CLI output. Essential for video pipeline status tracking. |

### Google Vertex AI Integration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| google-genai | 1.63.0+ | Gemini & Vertex AI SDK | Official GA SDK as of May 2025. Replaces deprecated `google-generativeai` (sunset Nov 2025) and `vertexai.generative_models` (deprecated June 2026). Supports Gemini 3 Pro, Imagen, and Veo 3.1 with async support via `[aiohttp]` extra. Uses Pydantic for type safety. |

### Database & Persistence

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| SQLAlchemy | 2.0.46+ | ORM & database toolkit | SQLAlchemy 2.0 offers first-class async support for modern async frameworks. Provides declarative ORM, connection pooling, and transparent persistence with identity map and unit of work patterns. |
| aiosqlite | 0.22.1+ | Async SQLite driver | Official async bridge to sqlite3 module. Allows SQLite operations on AsyncIO event loop without blocking. Uses single shared thread per connection to prevent overlapping actions. Essential for async FastAPI integration. |

### Video Processing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| ffmpeg | 6.0+ (binary) | Video stitching & processing | Industry-standard video processing tool. Required as system binary - Python wrappers are just interfaces. |
| subprocess | stdlib | FFmpeg interface | Use Python's subprocess module directly for FFmpeg calls. ffmpeg-python unmaintained since 2019. Direct subprocess calls provide full control, no maintenance risk, and avoid wrapper abstraction overhead. |

### Configuration & Environment

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pydantic-settings | 2.12.0+ | Settings management | Type-safe config with automatic validation. Loads from environment variables, .env files, TOML, JSON, YAML. Integrates seamlessly with FastAPI/Pydantic stack. Replaces manual python-dotenv usage. |

### HTTP Client

| Technology | Version | Purpose | When to Use |
|------------|---------|---------|-------------|
| httpx | 0.28.1+ | Async HTTP client | For async HTTP calls (if needed for external APIs). Supports HTTP/1.1 and HTTP/2, both sync and async interfaces, requests-compatible API. |

### Development Tools

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| pytest | 8.0+ | Testing framework | Industry standard Python testing |
| pytest-asyncio | 0.24+ | Async test support | Required for testing async FastAPI endpoints and SQLAlchemy async operations. Use `@pytest.mark.anyio` pattern recommended in 2026. |
| uvicorn | 0.30+ | ASGI server (dev) | Development server with auto-reload |
| gunicorn | 22+ | Production server | Run with `uvicorn.workers.UvicornWorker`, set workers = CPU cores (not 2N+1 for async), behind Nginx for SSL/load balancing |
| structlog | 25.5.0+ | Structured logging | JSON logging for production, context variables for request tracing, processor chains for log enrichment. Essential for debugging async pipeline flows. |

## Installation

```bash
# Core dependencies
pip install fastapi==0.129.0
pip install "uvicorn[standard]==0.30.0"
pip install typer==0.23.1
pip install rich==14.3.2

# Google Vertex AI
pip install "google-genai[aiohttp]==1.63.0"

# Database
pip install sqlalchemy==2.0.46
pip install aiosqlite==0.22.1

# Configuration
pip install pydantic-settings==2.12.0

# HTTP client (if needed)
pip install httpx==0.28.1

# Logging
pip install structlog==25.5.0

# Development dependencies
pip install pytest==8.0+
pip install pytest-asyncio==0.24+
pip install gunicorn==22+
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Python Version | 3.12 | 3.13 | 3.13 has JIT and free-threading (experimental), but 3.12 has better ecosystem compatibility and stability. Use 3.13 for cutting-edge projects only. |
| Vertex AI SDK | google-genai | google-cloud-aiplatform | The old SDK's generative_models module is deprecated (sunset June 2026). google-genai is the official GA replacement with better Pydantic integration. |
| Vertex AI SDK | google-genai | google-generativeai | Old Gemini API SDK, completely sunset November 2025. Do not use. |
| FFmpeg Interface | subprocess | ffmpeg-python | ffmpeg-python unmaintained since 2019. Direct subprocess calls are more maintainable and avoid abstraction overhead. |
| FFmpeg Interface | subprocess | moviepy | MoviePy is slower due to heavier data import/export (converts to numpy arrays). Use for complex editing, not for simple stitching. |
| Config Management | pydantic-settings | python-dotenv | pydantic-settings provides type safety, validation, and multi-format support. python-dotenv only loads .env files without validation. |
| Async SQLite | aiosqlite | raw sqlite3 | sqlite3 is synchronous and blocks the event loop. aiosqlite wraps it properly for async contexts. |
| Web Framework | FastAPI | Flask/Django | FastAPI built for async from ground up, has automatic OpenAPI docs, and Pydantic v2 integration. Flask/Django require async adapters. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| google-generativeai | Permanently sunset November 30, 2025 | google-genai |
| vertexai.generative_models | Deprecated June 2025, removal June 2026 | google-genai |
| ffmpeg-python | Unmaintained since 2019, maintenance risk | subprocess module with direct ffmpeg calls |
| moviepy | Slow for simple stitching (numpy conversion overhead) | subprocess + ffmpeg directly |
| python-dotenv alone | No type validation, manual parsing | pydantic-settings |
| SQLAlchemy 1.x | Legacy API, no first-class async support | SQLAlchemy 2.0+ |
| Synchronous sqlite3 | Blocks event loop in async contexts | aiosqlite |
| Python 3.10 or earlier | FastAPI supports 3.10+, but 3.11+ has significant performance improvements | Python 3.12 |

## Stack Patterns by Variant

**If building CLI-only (no HTTP server):**
- Skip FastAPI, uvicorn, gunicorn
- Use Typer + Rich for CLI
- Still use async patterns with SQLAlchemy/aiosqlite
- Run async CLI commands with `asyncio.run()`

**If adding real-time progress updates:**
- Use Rich progress bars in CLI
- Use Server-Sent Events (SSE) with FastAPI for web clients
- Consider WebSockets for bidirectional communication

**If scaling to multiple workers:**
- Use gunicorn with `workers = CPU_count` (not 2N+1 for async)
- Share SQLite database via file (careful with write contention)
- Consider PostgreSQL + asyncpg for multi-writer scenarios

**If running in container:**
- Pin exact versions in requirements.txt
- Install ffmpeg in container: `apt-get install ffmpeg`
- Use multi-stage build to minimize image size
- Run gunicorn + uvicorn workers, not development uvicorn

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| FastAPI 0.129.0 | Pydantic 2.7.0+ | Pydantic v1 support dropped. Use Pydantic 2.12.5+ for best results. |
| pydantic-settings 2.12.0 | Pydantic 2.12.5 | Version synchronized with Pydantic core |
| SQLAlchemy 2.0.46 | aiosqlite 0.22.1+ | Use AsyncEngine and AsyncSession for async patterns |
| pytest-asyncio 0.24+ | pytest 8.0+ | Use `@pytest.mark.anyio` for async tests in 2026 |
| google-genai 1.63.0 | Python 3.10+ | Requires `[aiohttp]` extra for async support |
| Typer 0.23.1 | Rich 14.3.2+ | Rich bundled as dependency, use `rich_markup_mode='rich'` |

## Critical Configuration Notes

**Google Vertex AI Authentication:**
- Set `GOOGLE_APPLICATION_CREDENTIALS` to service account JSON path
- Use Application Default Credentials (ADC) in production
- google-genai handles both Vertex AI and Gemini API with same codebase

**SQLAlchemy Async Best Practices:**
- Use `AsyncEngine` and `AsyncSession`
- Set `expire_on_commit=False` to avoid implicit queries after commit
- Use `selectinload()` or `joinedload()` for eager loading relationships
- Avoid lazy loading in async contexts (causes implicit blocking queries)

**FastAPI Production Deployment:**
- Run behind Nginx for SSL termination, rate limiting, static files
- Use gunicorn with uvicorn workers: `gunicorn -k uvicorn.workers.UvicornWorker -w 4`
- Set workers to CPU core count (not 2N+1 formula - that's for sync workers)
- Enable access logs only in development; use structured logging in production

**FFmpeg Subprocess Best Practices:**
- Use `subprocess.run()` with `capture_output=True` for safety
- Set `timeout` parameter to prevent hanging on corrupted input
- Check `returncode` and handle errors explicitly
- Log stderr output for debugging failed stitches

**Structlog Production Setup:**
- Output JSON in production for log aggregators
- Use `structlog.contextvars` for request-scoped context (safe in async)
- Log to stdout unbuffered, let systemd/container runtime handle persistence
- Include correlation IDs for tracing requests through pipeline

## Sources

**High Confidence (Official Documentation):**
- [FastAPI PyPI](https://pypi.org/project/fastapi/) - Version 0.129.0, Python 3.10+ support
- [SQLAlchemy PyPI](https://pypi.org/project/sqlalchemy/) - Version 2.0.46, async features
- [Typer PyPI](https://pypi.org/project/typer/) - Version 0.23.1, Rich integration
- [Pydantic PyPI](https://pypi.org/project/pydantic/) - Version 2.12.5, Python 3.9-3.14
- [google-genai PyPI](https://pypi.org/project/google-genai/) - Version 1.63.0, GA status
- [aiosqlite PyPI](https://pypi.org/project/aiosqlite/) - Version 0.22.1, Python 3.9+
- [Rich PyPI](https://pypi.org/project/rich/) - Version 14.3.2, production stable
- [httpx PyPI](https://pypi.org/project/httpx/) - Version 0.28.1, HTTP/2 support
- [pydantic-settings PyPI](https://pypi.org/project/pydantic-settings/) - Version 2.12.0
- [Vertex AI Python SDK Documentation](https://docs.cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk) - Deprecation notices

**Medium Confidence (Web Search + Multiple Sources):**
- [Google Gen AI SDK Migration Guide](https://medium.com/google-cloud/migrating-to-the-new-google-gen-ai-sdk-python-074d583c2350) - Migration timeline, google-genai GA status
- [FastAPI Production Deployment Best Practices](https://render.com/articles/fastapi-production-deployment-best-practices) - Gunicorn + Uvicorn patterns
- [Building High-Performance Async APIs with FastAPI, SQLAlchemy 2.0, and Asyncpg](https://leapcell.io/blog/building-high-performance-async-apis-with-fastapi-sqlalchemy-2-0-and-asyncpg) - Async patterns, expire_on_commit
- [SQLAlchemy Asynchronous I/O](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) - Async best practices
- [Structlog Logging Best Practices](https://www.structlog.org/en/stable/logging-best-practices.html) - Production JSON output, contextvars
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - Type-safe config management
- [Python 3.12 vs 3.13 Performance](https://releaserun.com/python-3-12-vs-3-13-vs-3-14-comparison/) - Version recommendations for 2026
- [FFmpeg Python Wrappers Comparison](https://www.gumlet.com/learn/ffmpeg-python/) - Subprocess vs ffmpeg-python vs moviepy
- [Uvicorn Production Deployment](https://uvicorn.dev/deployment/) - Gunicorn configuration, worker count
- [aiosqlite Documentation](https://aiosqlite.omnilib.dev/en/latest/) - Async SQLite patterns

---
*Stack research for: AI Video Generation Pipeline using Google Vertex AI*
*Researched: 2026-02-14*
