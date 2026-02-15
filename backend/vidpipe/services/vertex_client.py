"""Vertex AI client wrapper using google-genai SDK.

This module provides a singleton client for Google Generative AI using Vertex AI mode.
Authentication is handled automatically via Application Default Credentials (ADC).

Usage:
    from vidpipe.services.vertex_client import get_vertex_client

    client = get_vertex_client()
    response = await client.aio.models.generate_content(...)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from vidpipe.config import settings

# Load .env for GOOGLE_APPLICATION_CREDENTIALS (ADC)
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# Global singleton instance
_vertex_client = None


def get_vertex_client() -> genai.Client:
    """Get or create singleton Vertex AI client instance.

    Configures the google-genai SDK in Vertex AI mode using:
    - Project ID from settings.google_cloud.project_id
    - Location from settings.google_cloud.location
    - ADC authentication (automatically via GOOGLE_APPLICATION_CREDENTIALS)

    Environment variables set:
    - GOOGLE_GENAI_USE_VERTEXAI: Enables Vertex AI mode
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    - GOOGLE_CLOUD_LOCATION: GCP region (e.g., us-central1)

    Returns:
        genai.Client: Configured client instance for Vertex AI
    """
    global _vertex_client

    if _vertex_client is None:
        # Set environment variables for Vertex AI mode
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud.project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud.location

        # Create client in Vertex AI mode
        _vertex_client = genai.Client(
            vertexai=True,
            project=settings.google_cloud.project_id,
            location=settings.google_cloud.location
        )

    return _vertex_client
