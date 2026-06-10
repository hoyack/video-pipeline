"""Vertex AI client wrapper using google-genai SDK.

This module provides location-aware clients for Google Generative AI using Vertex AI mode.
Authentication uses DB-stored service account credentials (preferred) with fallback to
Application Default Credentials (ADC) from the environment.

Usage:
    from vidpipe.services.vertex_client import get_vertex_client

    client = get_vertex_client()                    # default location
    client = get_vertex_client(location="global")   # global endpoint
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai

from vidpipe.config import settings
from vidpipe.services.model_catalog import GLOBAL_REGION_MODELS, canonical_model_id

logger = logging.getLogger(__name__)

# Load .env for GOOGLE_APPLICATION_CREDENTIALS (ADC) — fallback only
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# Per-location client cache
_clients: dict[str, genai.Client] = {}

# DB-loaded credential cache
_db_project_id: Optional[str] = None
_db_location: Optional[str] = None
_db_credentials: Optional[object] = None  # google.oauth2.service_account.Credentials

async def load_vertex_credentials_from_db() -> None:
    """Load Vertex AI credentials from user_settings DB table.

    Caches project_id, location, and service_account.Credentials at module level.
    Clears existing client cache so new clients pick up the new credentials.
    """
    global _db_project_id, _db_location, _db_credentials, _clients

    from vidpipe.db import async_session
    from vidpipe.db.models import UserSettings, DEFAULT_USER_ID
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        us = result.scalar_one_or_none()
        if not us:
            return

        # Cache project/location from DB
        _db_project_id = us.gcp_project_id or None
        _db_location = us.gcp_location or None

        # Parse service account JSON if present
        if us.gcp_service_account_json:
            try:
                from google.oauth2 import service_account
                info = json.loads(us.gcp_service_account_json)
                _db_credentials = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                # Auto-extract project_id from service account if not set in DB
                if not _db_project_id and info.get("project_id"):
                    _db_project_id = info["project_id"]
                logger.info("Loaded Vertex AI service account credentials from DB")
            except Exception as e:
                logger.warning("Failed to parse GCP service account JSON from DB: %s", e)
                _db_credentials = None
        else:
            _db_credentials = None

    # Clear client cache so next get_vertex_client() uses new creds
    _clients.clear()


def invalidate_vertex_clients() -> None:
    """Clear all cached Vertex AI clients and DB credential cache."""
    global _db_project_id, _db_location, _db_credentials, _clients
    _db_project_id = None
    _db_location = None
    _db_credentials = None
    _clients.clear()


def _effective_project_id() -> str:
    """Return project ID: DB → config → empty."""
    return _db_project_id or settings.google_cloud.project_id


def _effective_location() -> str:
    """Return location: DB → config."""
    return _db_location or settings.google_cloud.location


def location_for_model(model_id: str) -> str:
    """Return the Vertex AI location needed for a given model ID."""
    canonical = canonical_model_id(model_id) or model_id
    if canonical in GLOBAL_REGION_MODELS:
        return "global"
    return _effective_location()


def get_vertex_client(location: str | None = None) -> genai.Client:
    """Get or create a Vertex AI client for the given location.

    Uses DB-stored service account credentials if available, otherwise
    falls back to Application Default Credentials (ADC).

    Args:
        location: GCP region (e.g., "us-central1", "global").
                  Defaults to effective location (DB → config).

    Returns:
        genai.Client: Configured client instance for Vertex AI
    """
    loc = location or _effective_location()
    project = _effective_project_id()

    if loc not in _clients:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        if project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project

        kwargs: dict = {
            "vertexai": True,
            "project": project,
            "location": loc,
        }
        if _db_credentials is not None:
            kwargs["credentials"] = _db_credentials

        _clients[loc] = genai.Client(**kwargs)

    return _clients[loc]
