"""ElevenLabs audio adapter implementation.

Uses the official ElevenLabs Python SDK (AsyncElevenLabs) for TTS,
sound effects generation, and voice listing.
"""

from __future__ import annotations

import logging
import inspect
from typing import Optional

import httpx

from vidpipe.services.audio.base import (
    AudioAdapter,
    AudioAdapterError,
    AudioAuthError,
    AudioQuotaError,
    VoiceProfileInfo,
)

logger = logging.getLogger(__name__)


class ElevenLabsAdapter(AudioAdapter):
    """ElevenLabs audio adapter using the official async SDK."""

    def __init__(self, api_key: str) -> None:
        from elevenlabs import AsyncElevenLabs

        self._client = AsyncElevenLabs(api_key=api_key)

    async def generate_voice(
        self,
        voice_id: str,
        text: str,
        *,
        style_notes: Optional[str] = None,
        model_id: Optional[str] = None,
        previous_text: Optional[str] = None,
        next_text: Optional[str] = None,
        output_format: Optional[str] = None,
        voice_settings: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """Generate TTS audio via ElevenLabs text-to-speech API."""
        try:
            kwargs: dict = {
                "voice_id": voice_id,
                "text": text,
                "model_id": model_id or "eleven_multilingual_v2",
            }
            if previous_text:
                kwargs["previous_text"] = previous_text
            if next_text:
                kwargs["next_text"] = next_text
            if output_format:
                kwargs["output_format"] = output_format
            if voice_settings:
                kwargs["voice_settings"] = voice_settings
            if seed is not None:
                kwargs["seed"] = seed

            response = self._client.text_to_speech.convert(**kwargs)
            if inspect.isawaitable(response):
                response = await response
            # SDK v2.x returns an async iterator of bytes chunks
            chunks: list[bytes] = []
            async for chunk in response:
                chunks.append(chunk)
            return b"".join(chunks)
        except httpx.HTTPStatusError as e:
            _raise_domain_error(e)
        except Exception as e:
            raise AudioAdapterError(f"ElevenLabs TTS failed: {e}") from e

    async def generate_sfx(
        self,
        prompt: str,
        *,
        duration_seconds: Optional[float] = None,
    ) -> bytes:
        """Generate sound effects via ElevenLabs sound effects API."""
        try:
            kwargs: dict = {"text": prompt}
            if duration_seconds is not None:
                kwargs["duration_seconds"] = duration_seconds
            response = await self._client.text_to_sound_effects.convert(**kwargs)
            chunks: list[bytes] = []
            async for chunk in response:
                chunks.append(chunk)
            return b"".join(chunks)
        except httpx.HTTPStatusError as e:
            _raise_domain_error(e)
        except Exception as e:
            raise AudioAdapterError(f"ElevenLabs SFX generation failed: {e}") from e

    async def list_voices(
        self,
        *,
        search: Optional[str] = None,
    ) -> list[VoiceProfileInfo]:
        """List available ElevenLabs voices with optional name search."""
        try:
            if search:
                result = await self._client.voices.search(search=search)
            else:
                result = await self._client.voices.search()
            voices = result.voices if hasattr(result, "voices") else result
            return [
                VoiceProfileInfo(
                    voice_id=v.voice_id,
                    name=v.name,
                    labels=v.labels if isinstance(v.labels, dict) else {},
                    preview_url=v.preview_url if hasattr(v, "preview_url") else None,
                )
                for v in voices
            ]
        except httpx.HTTPStatusError as e:
            _raise_domain_error(e)
        except Exception as e:
            raise AudioAdapterError(f"ElevenLabs voice list failed: {e}") from e

    async def get_voice(self, voice_id: str) -> VoiceProfileInfo:
        """Get details for a specific ElevenLabs voice."""
        try:
            v = await self._client.voices.get(voice_id=voice_id)
            return VoiceProfileInfo(
                voice_id=v.voice_id,
                name=v.name,
                labels=v.labels if isinstance(v.labels, dict) else {},
                preview_url=v.preview_url if hasattr(v, "preview_url") else None,
            )
        except httpx.HTTPStatusError as e:
            _raise_domain_error(e)
        except Exception as e:
            raise AudioAdapterError(f"ElevenLabs voice get failed: {e}") from e


def _raise_domain_error(e: httpx.HTTPStatusError) -> None:
    """Convert HTTP errors to domain-specific exceptions."""
    status = e.response.status_code
    detail = e.response.text[:200] if e.response.text else str(e)
    if status == 401:
        raise AudioAuthError(f"Invalid ElevenLabs API key: {detail}") from e
    if status == 429:
        raise AudioQuotaError(f"ElevenLabs rate limit exceeded: {detail}") from e
    raise AudioAdapterError(f"ElevenLabs API error ({status}): {detail}") from e
