"""Tests for the ElevenLabs audio adapter.

Mocks the ElevenLabs SDK to verify TTS, SFX, voice list, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vidpipe.services.audio.base import (
    AudioAdapterError,
    AudioAuthError,
    AudioQuotaError,
    VoiceProfileInfo,
)
from vidpipe.services.audio.elevenlabs_adapter import ElevenLabsAdapter
from vidpipe.services.audio.registry import get_audio_adapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Create a mock AsyncElevenLabs client."""
    client = MagicMock()
    client.text_to_speech = MagicMock()
    client.text_to_sound_effects = MagicMock()
    client.voices = MagicMock()
    return client


@pytest.fixture
def adapter(mock_client):
    """Create an ElevenLabsAdapter with a mocked client."""
    a = ElevenLabsAdapter.__new__(ElevenLabsAdapter)
    a._client = mock_client
    return a


# ---------------------------------------------------------------------------
# Helper: async iterator from bytes
# ---------------------------------------------------------------------------


async def _async_iter(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


class FakeElevenLabsApiError(Exception):
    def __init__(self, status_code: int, body: dict):
        super().__init__("headers: noisy-provider-headers")
        self.status_code = status_code
        self.body = body


# ---------------------------------------------------------------------------
# generate_voice tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_voice_returns_bytes(adapter, mock_client):
    """TTS returns concatenated bytes from async iterator."""
    mock_client.text_to_speech.convert = AsyncMock(
        return_value=_async_iter([b"hello ", b"world"])
    )

    result = await adapter.generate_voice("voice123", "test text")

    assert result == b"hello world"
    mock_client.text_to_speech.convert.assert_called_once_with(
        voice_id="voice123",
        text="test text",
        model_id="eleven_multilingual_v2",
    )


@pytest.mark.asyncio
async def test_generate_voice_custom_model(adapter, mock_client):
    """TTS respects model_id override."""
    mock_client.text_to_speech.convert = AsyncMock(
        return_value=_async_iter([b"data"])
    )

    await adapter.generate_voice("v1", "hi", model_id="eleven_turbo_v2")

    mock_client.text_to_speech.convert.assert_called_once_with(
        voice_id="v1",
        text="hi",
        model_id="eleven_turbo_v2",
    )


@pytest.mark.asyncio
async def test_generate_voice_auth_error(adapter, mock_client):
    """TTS raises AudioAuthError on 401."""
    response = MagicMock()
    response.status_code = 401
    response.text = "Unauthorized"
    mock_client.text_to_speech.convert = AsyncMock(
        side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=response)
    )

    with pytest.raises(AudioAuthError, match="Invalid ElevenLabs API key"):
        await adapter.generate_voice("v1", "text")


@pytest.mark.asyncio
async def test_generate_voice_quota_error(adapter, mock_client):
    """TTS raises AudioQuotaError on 429."""
    response = MagicMock()
    response.status_code = 429
    response.text = "Rate limited"
    mock_client.text_to_speech.convert = AsyncMock(
        side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=response)
    )

    with pytest.raises(AudioQuotaError, match="rate limit"):
        await adapter.generate_voice("v1", "text")


@pytest.mark.asyncio
async def test_generate_voice_sdk_api_error_uses_provider_message(adapter, mock_client):
    """TTS SDK ApiError body is surfaced without dumping response headers."""
    mock_client.text_to_speech.convert = AsyncMock(
        side_effect=FakeElevenLabsApiError(
            400,
            {
                "detail": {
                    "message": "An invalid ID has been received: 'bad-voice'.",
                }
            },
        )
    )

    with pytest.raises(AudioAdapterError) as exc_info:
        await adapter.generate_voice("bad-voice", "text")

    message = str(exc_info.value)
    assert "ElevenLabs API error (400)" in message
    assert "An invalid ID has been received" in message
    assert "headers:" not in message


# ---------------------------------------------------------------------------
# generate_sfx tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_sfx_returns_bytes(adapter, mock_client):
    """SFX generation returns concatenated bytes."""
    mock_client.text_to_sound_effects.convert = AsyncMock(
        return_value=_async_iter([b"sfx_data"])
    )

    result = await adapter.generate_sfx("explosion sound")

    assert result == b"sfx_data"
    mock_client.text_to_sound_effects.convert.assert_called_once_with(
        text="explosion sound"
    )


@pytest.mark.asyncio
async def test_generate_sfx_with_duration(adapter, mock_client):
    """SFX generation passes duration_seconds when provided."""
    mock_client.text_to_sound_effects.convert = AsyncMock(
        return_value=_async_iter([b"data"])
    )

    await adapter.generate_sfx("rain", duration_seconds=5.0)

    mock_client.text_to_sound_effects.convert.assert_called_once_with(
        text="rain", duration_seconds=5.0
    )


@pytest.mark.asyncio
async def test_generate_sfx_generic_error(adapter, mock_client):
    """SFX wraps unknown errors in AudioAdapterError."""
    mock_client.text_to_sound_effects.convert = AsyncMock(
        side_effect=RuntimeError("network failure")
    )

    with pytest.raises(AudioAdapterError, match="SFX generation failed"):
        await adapter.generate_sfx("boom")


# ---------------------------------------------------------------------------
# list_voices tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_voices_returns_profiles(adapter, mock_client):
    """list_voices maps SDK response to VoiceProfileInfo list."""
    voice = MagicMock()
    voice.voice_id = "abc123"
    voice.name = "Rachel"
    voice.labels = {"accent": "american", "gender": "female"}
    voice.preview_url = "https://example.com/preview.mp3"

    search_result = MagicMock()
    search_result.voices = [voice]
    mock_client.voices.search = AsyncMock(return_value=search_result)

    result = await adapter.list_voices()

    assert len(result) == 1
    assert isinstance(result[0], VoiceProfileInfo)
    assert result[0].voice_id == "abc123"
    assert result[0].name == "Rachel"
    assert result[0].labels == {"accent": "american", "gender": "female"}
    assert result[0].preview_url == "https://example.com/preview.mp3"


@pytest.mark.asyncio
async def test_list_voices_with_search(adapter, mock_client):
    """list_voices passes search param to SDK."""
    search_result = MagicMock()
    search_result.voices = []
    mock_client.voices.search = AsyncMock(return_value=search_result)

    await adapter.list_voices(search="Rachel")

    mock_client.voices.search.assert_called_once_with(search="Rachel")


# ---------------------------------------------------------------------------
# get_voice tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_voice_returns_profile(adapter, mock_client):
    """get_voice maps single voice to VoiceProfileInfo."""
    voice = MagicMock()
    voice.voice_id = "xyz789"
    voice.name = "Adam"
    voice.labels = {"gender": "male"}
    voice.preview_url = None

    mock_client.voices.get = AsyncMock(return_value=voice)

    result = await adapter.get_voice("xyz789")

    assert isinstance(result, VoiceProfileInfo)
    assert result.voice_id == "xyz789"
    assert result.name == "Adam"
    mock_client.voices.get.assert_called_once_with(voice_id="xyz789")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_returns_elevenlabs():
    """get_audio_adapter returns ElevenLabsAdapter for ELEVENLABS type."""
    with patch("elevenlabs.AsyncElevenLabs"):
        adapter = get_audio_adapter("ELEVENLABS", "test-key")
        assert isinstance(adapter, ElevenLabsAdapter)


def test_registry_unknown_type():
    """get_audio_adapter raises ValueError for unknown adapter type."""
    with pytest.raises(ValueError, match="Unknown audio adapter"):
        get_audio_adapter("UNKNOWN", "key")
