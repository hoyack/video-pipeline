"""FastAPI application setup with lifespan and exception handlers."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vidpipe import validate_dependencies
from vidpipe.db import init_database, shutdown
from vidpipe.services.comfyui_client import close_comfyui_client
from vidpipe.api.routes import router

# Configure root logger so application logs (pipeline stages, S3, etc.) reach stdout
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Startup:
        - Validate system dependencies (ffmpeg)
        - Initialize database schema

    Shutdown:
        - Close database connections
    """
    # Startup
    logger.info("Starting Video Pipeline API...")
    validate_dependencies()
    await init_database()
    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Video Pipeline API...")
    await close_comfyui_client()
    await shutdown()
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Video Pipeline API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router with all endpoints
app.include_router(router)

# Serve frontend static files in production (after API routes take priority)
_frontend_dist = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    # Mount /assets separately so JS/CSS bundles are served directly
    _assets_dir = _frontend_dist / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    # SPA catch-all: serve the file if it exists, otherwise index.html
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        file = _frontend_dist / full_path
        if full_path and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(_frontend_dist / "index.html"))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler to prevent stack traces in API responses."""
    logger.error(f"Unhandled exception in {request.method} {request.url.path}: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        }
    )
