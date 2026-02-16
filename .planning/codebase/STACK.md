# Technology Stack

**Analysis Date:** 2026-02-16

## Languages

**Primary:**
- Python 3.11+ - Backend API and pipeline orchestration
- TypeScript 5.9.3 - Frontend application
- JavaScript/JSX - React components

**Secondary:**
- SQL - SQLite schema and queries

## Runtime

**Environment:**
- Python 3.11+ (backend)
- Node.js (frontend build and development)

**Package Managers:**
- pip (Python) - Primary package manager for backend
- npm (Node.js) - Frontend dependencies
- Lockfiles: `package-lock.json` (frontend); pip uses `requirements.txt`

## Frameworks

**Backend:**
- FastAPI 0.115.0+ - REST API framework, async-first
- SQLAlchemy 2.0+ - ORM with async support
- uvicorn 0.30.0+ - ASGI server
- Pydantic 2.0+ - Data validation
- Pydantic Settings 2.12.0+ - Configuration management with YAML/env support
- Typer 0.12.0+ - CLI framework for command-line tools

**Frontend:**
- React 19.2.0 - UI library
- React DOM 19.2.0 - React web rendering
- Vite 7.3.1 - Build tool and dev server
- TypeScript 5.9.3 - Type safety

**Testing:**
- pytest 7.0+ (backend, optional dependency)
- pytest-asyncio 0.21+ (backend async test support, optional)

**Build/Dev:**
- Tailwind CSS 4.1.18 - Utility-first CSS framework
- @tailwindcss/vite 4.1.18 - Vite plugin for Tailwind
- ESLint 9.39.1 - JavaScript linting
- @vitejs/plugin-react 5.1.1 - React support for Vite
- typescript-eslint 8.48.0 - TypeScript linting

## Key Dependencies

**Critical - Backend APIs:**
- google-genai 1.0.0+ - Google Generative AI SDK for Vertex AI (LLMs, image generation, video generation)
- httpx 0.27.0+ - Async HTTP client for external service calls

**Critical - Database:**
- SQLAlchemy[asyncio] 2.0+ - Async ORM for SQLite
- aiosqlite 0.22.1+ - Async SQLite driver

**Critical - Infrastructure:**
- Pillow 10.0+ - Image processing and manipulation
- pyyaml 6.0+ - YAML configuration parsing
- python-dotenv 1.0+ - Environment variable management

**Development:**
- black 23.0+ - Code formatter (optional)
- ruff 0.1.0+ - Linter (optional)

**Frontend Dependencies:**
- clsx 2.1.1 - Conditional class name utility
- globals 16.5.0 - Global variable definitions
- eslint-plugin-react-hooks 7.0.1 - React Hooks linting rules
- eslint-plugin-react-refresh 0.4.24 - React Fast Refresh linting

## Configuration

**Environment:**
- `.env` file (location: `/home/ubuntu/work/video-pipeline/.env`)
- Environment variable prefix: `VIDPIPE_` with `__` as nested delimiter
- Example: `VIDPIPE_GOOGLE_CLOUD__PROJECT_ID` maps to `settings.google_cloud.project_id`
- Required: `GOOGLE_APPLICATION_CREDENTIALS` path to GCP service account JSON
- Required: `VIDPIPE_GOOGLE_CLOUD__PROJECT_ID` for GCP project

**Build Configuration:**
- `config.yaml` - Main application configuration (models, pipeline parameters, storage, server)
  - Located: `/home/ubuntu/work/video-pipeline/config.yaml`
  - Defines: LLM models, image models, video models, pipeline settings, database URL, temp directory
- `vite.config.ts` - Frontend build and dev server proxy
- `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json` - TypeScript configurations
- `eslint.config.js` - Linting rules
- `pyproject.toml` - Backend project metadata and dependencies
- `.env.example` - Template for required environment variables

## Platform Requirements

**Development:**
- Python 3.11+
- Node.js (for frontend development)
- ffmpeg - Required system dependency for video stitching (validated at startup)
- GCP project with Vertex AI enabled
- GCP service account with Generative AI APIs enabled

**Production:**
- Python 3.11+ runtime
- ffmpeg binary on PATH
- SQLite database (local file-based: `vidpipe.db`)
- GCP Vertex AI APIs accessible
- Network access to Google Cloud services

## System Dependencies

**Required:**
- ffmpeg - For video concatenation and crossfade transitions
  - Validation happens at API startup in `vidpipe/__init__.py::validate_dependencies()`
  - Used by: `vidpipe/pipeline/stitcher.py` for video stitching via subprocess

## External Service Integrations

**Google Cloud Platform:**
- Vertex AI API - LLM, image, and video generation
- Google Cloud Storage (optional) - Video clip storage (gcs_uri field in database)
- Application Default Credentials (ADC) - For GCP authentication

---

*Stack analysis: 2026-02-16*
