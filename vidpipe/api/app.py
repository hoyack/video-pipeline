"""FastAPI application setup with lifespan and exception handlers."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vidpipe import validate_dependencies
from vidpipe.db import init_database, shutdown
from vidpipe.api.routes import router

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
    await shutdown()
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Video Pipeline API",
    version="0.1.0",
    lifespan=lifespan,
)

# Include router with all endpoints
app.include_router(router)


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
