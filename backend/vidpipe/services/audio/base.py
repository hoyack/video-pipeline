"""Abstract base class for audio provider adapters.

Defines the consistent async interface that all audio adapters must implement,
supporting TTS voice generation, SFX generation, and voice listing.
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class VoiceProfileInfo(BaseModel):
    """Lightweight voice info returned from provider APIs."""

    voice_id: str
    name: str
    labels: dict[str, str] = {}
    preview_url: Optional[str] = None


class AudioAdapterError(Exception):
    """Base error for audio adapter failures."""


class AudioAuthError(AudioAdapterError):
    """Raised when the API key is invalid or missing."""


class AudioQuotaError(AudioAdapterError):
    """Raised when the API quota/rate limit is exceeded."""


class AudioAdapter(ABC):
    """Abstract base class for audio provider adapters."""

    @abstractmethod
    async def generate_voice(
        self,
        voice_id: str,
        text: str,
        *,
        style_notes: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> bytes:
        """Generate TTS audio from text using the specified voice.

        Args:
            voice_id: Provider-specific voice identifier.
            text: Text to convert to speech.
            style_notes: Optional style direction for the voice.
            model_id: Optional model override (provider-specific).

        Returns:
            Raw audio bytes (MP3 format).
        """
        ...

    @abstractmethod
    async def generate_sfx(
        self,
        prompt: str,
        *,
        duration_seconds: Optional[float] = None,
    ) -> bytes:
        """Generate sound effects from a text prompt.

        Args:
            prompt: Description of the desired sound effect.
            duration_seconds: Optional target duration in seconds.

        Returns:
            Raw audio bytes (MP3 format).
        """
        ...

    @abstractmethod
    async def list_voices(
        self,
        *,
        search: Optional[str] = None,
    ) -> list[VoiceProfileInfo]:
        """List available voices, optionally filtered by search query.

        Args:
            search: Optional name/label search filter.

        Returns:
            List of available voice profiles.
        """
        ...

    @abstractmethod
    async def get_voice(self, voice_id: str) -> VoiceProfileInfo:
        """Get details for a specific voice by ID.

        Args:
            voice_id: Provider-specific voice identifier.

        Returns:
            Voice profile info.
        """
        ...
