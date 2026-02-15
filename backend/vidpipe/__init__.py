"""Video Pipeline - AI-powered multi-scene video generation.

This module provides startup validation functions to ensure required
dependencies are available before pipeline execution begins.
Call validate_dependencies() during application startup.
"""

import logging
import subprocess

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def validate_dependencies() -> None:
    """Validate required system dependencies are available.

    This function should be called during application startup to fail fast
    with clear installation instructions if required dependencies are missing.

    Raises:
        RuntimeError: If ffmpeg is not found or not functional.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True,
            text=True
        )
        version_line = result.stdout.split('\n')[0]
        logger.info(f"ffmpeg validated: {version_line}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to use video generation pipeline.\n"
            "Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "macOS: brew install ffmpeg\n"
            "Windows: https://ffmpeg.org/download.html"
        ) from e
