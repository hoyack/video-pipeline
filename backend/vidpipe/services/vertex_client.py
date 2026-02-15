"""Vertex AI client wrapper using google-genai SDK.

This module provides location-aware clients for Google Generative AI using Vertex AI mode.
Authentication is handled automatically via Application Default Credentials (ADC).

Usage:
    from vidpipe.services.vertex_client import get_vertex_client

    client = get_vertex_client()                    # default location
    client = get_vertex_client(location="global")   # global endpoint
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from vidpipe.config import settings

# Load .env for GOOGLE_APPLICATION_CREDENTIALS (ADC)
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# Per-location client cache
_clients: dict[str, genai.Client] = {}

# Models that must use the global endpoint
GLOBAL_REGION_MODELS = {
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
}


def location_for_model(model_id: str) -> str:
    """Return the Vertex AI location needed for a given model ID."""
    if model_id in GLOBAL_REGION_MODELS:
        return "global"
    return settings.google_cloud.location


def get_vertex_client(location: str | None = None) -> genai.Client:
    """Get or create a Vertex AI client for the given location.

    Clients are cached per location so repeated calls are cheap.

    Args:
        location: GCP region (e.g., "us-central1", "global").
                  Defaults to settings.google_cloud.location.

    Returns:
        genai.Client: Configured client instance for Vertex AI
    """
    loc = location or settings.google_cloud.location

    if loc not in _clients:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud.project_id

        _clients[loc] = genai.Client(
            vertexai=True,
            project=settings.google_cloud.project_id,
            location=loc,
        )

    return _clients[loc]
