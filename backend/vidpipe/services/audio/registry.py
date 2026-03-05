"""Provider registry for audio adapters.

Routes adapter_type strings to the correct adapter implementation.
"""

from __future__ import annotations

import logging

from vidpipe.services.audio.base import AudioAdapter

logger = logging.getLogger(__name__)


def get_audio_adapter(adapter_type: str, api_key: str) -> AudioAdapter:
    """Return the appropriate audio adapter for the given adapter type.

    Args:
        adapter_type: Adapter identifier (e.g., "ELEVENLABS").
        api_key: API key for the audio provider.

    Returns:
        Configured AudioAdapter instance ready for use.

    Raises:
        ValueError: If the adapter type is unknown.
    """
    if adapter_type == "ELEVENLABS":
        from vidpipe.services.audio.elevenlabs_adapter import ElevenLabsAdapter

        logger.debug("Routing to ElevenLabsAdapter")
        return ElevenLabsAdapter(api_key=api_key)

    raise ValueError(f"Unknown audio adapter: {adapter_type}")
