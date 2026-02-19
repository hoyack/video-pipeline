"""Provider registry for LLM adapters.

Routes model IDs to the correct adapter implementation based on the model
ID prefix. Supports Vertex AI (gemini- prefix) and Ollama (ollama/ prefix).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from vidpipe.services.llm.base import LLMAdapter

if TYPE_CHECKING:
    from vidpipe.db.models import UserSettings

logger = logging.getLogger(__name__)


def _is_ollama_model(model_id: str) -> bool:
    """Return True if the model ID uses the ollama/ prefix."""
    return model_id.startswith("ollama/")


def _is_gemini_model(model_id: str) -> bool:
    """Return True if the model ID uses the gemini- prefix."""
    return model_id.startswith("gemini-")


def get_adapter(
    model_id: str,
    user_settings: Optional["UserSettings"] = None,
) -> LLMAdapter:
    """Return the appropriate LLM adapter for the given model ID.

    Routing logic:
    - "ollama/*"  → OllamaAdapter (cloud if user_settings.ollama_use_cloud,
                    else localhost:11434 or custom endpoint)
    - "gemini-*"  → VertexAIAdapter
    - anything else → VertexAIAdapter (fallback)

    Args:
        model_id: Model identifier string (e.g., "gemini-2.5-flash",
                  "ollama/llama3.1").
        user_settings: Optional UserSettings ORM instance for Ollama config.

    Returns:
        Configured LLMAdapter instance ready for use.
    """
    if _is_ollama_model(model_id):
        from vidpipe.services.llm.ollama_adapter import OllamaAdapter

        if user_settings is not None and user_settings.ollama_use_cloud:
            # Cloud Ollama (e.g., ollama.com or custom cloud endpoint)
            base_url = user_settings.ollama_endpoint or "https://ollama.com"
            api_key = user_settings.ollama_api_key
        else:
            # Local Ollama or custom non-cloud endpoint
            if user_settings is not None and user_settings.ollama_endpoint:
                base_url = user_settings.ollama_endpoint
            else:
                base_url = "http://localhost:11434"
            api_key = None

        logger.debug(
            "Routing %s to OllamaAdapter (base_url=%s, has_key=%s)",
            model_id,
            base_url,
            bool(api_key),
        )
        return OllamaAdapter(model_id=model_id, base_url=base_url, api_key=api_key)

    # Default: Vertex AI (handles gemini- models and anything else)
    from vidpipe.services.llm.vertex_adapter import VertexAIAdapter

    logger.debug("Routing %s to VertexAIAdapter", model_id)
    return VertexAIAdapter(model_id=model_id)
