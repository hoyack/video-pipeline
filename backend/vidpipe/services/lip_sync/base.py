"""Lip-sync adapter contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel


class LipSyncRequest(BaseModel):
    """Input media for one lip-sync operation."""

    input_video_path: Path
    input_audio_path: Path
    output_video_path: Path
    speaker_tag: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class LipSyncResult(BaseModel):
    """Result from a lip-sync adapter."""

    output_video_path: Path
    metrics: dict = {}

    model_config = {"arbitrary_types_allowed": True}


class LipSyncAdapter(ABC):
    """Abstract base class for post-video lip-sync providers."""

    @abstractmethod
    async def sync(self, request: LipSyncRequest) -> LipSyncResult:
        """Generate a lip-synced video clip."""
        ...
