"""Audio adapter abstraction layer.

Provides a unified async interface for TTS (text-to-speech) and SFX
(sound effects) generation across audio providers (ElevenLabs, future adapters).

Usage:
    from vidpipe.services.audio import get_audio_adapter, AudioAdapter

    adapter = get_audio_adapter("ELEVENLABS", api_key="sk-...")
    audio_bytes = await adapter.generate_voice(voice_id, "Hello world")
"""

from vidpipe.services.audio.base import AudioAdapter, VoiceProfileInfo
from vidpipe.services.audio.registry import get_audio_adapter

__all__ = ["AudioAdapter", "VoiceProfileInfo", "get_audio_adapter"]
