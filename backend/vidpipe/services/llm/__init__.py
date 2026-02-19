"""LLM provider abstraction layer.

Provides a unified async interface for text generation and image analysis
across multiple LLM providers (Vertex AI, Ollama, and future providers).

Usage:
    from vidpipe.services.llm import get_adapter, LLMAdapter

    adapter = get_adapter("gemini-2.5-flash")
    result = await adapter.generate_text(prompt, MySchema)

    adapter = get_adapter("ollama/llama3.1", user_settings)
    result = await adapter.generate_text(prompt, MySchema)
"""

from vidpipe.services.llm.base import LLMAdapter
from vidpipe.services.llm.registry import get_adapter

__all__ = ["LLMAdapter", "get_adapter"]
