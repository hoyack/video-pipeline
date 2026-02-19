"""Ollama adapter for the LLM abstraction layer.

Connects via ollama.AsyncClient with optional auth headers, structured JSON
output via model_json_schema(), and vision support via base64-encoded images.
"""

import base64
import logging
from typing import Optional, Type

from ollama import AsyncClient
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from vidpipe.services.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMAdapter):
    """LLM adapter backed by a local or cloud Ollama instance.

    Strips the "ollama/" prefix from model IDs before passing to the ollama
    library. Always passes stream=False to avoid async generator responses.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize adapter for the given Ollama model.

        Args:
            model_id: Model identifier, optionally prefixed with "ollama/"
                      (e.g., "ollama/llama3.1" or "llama3.1").
            base_url: Base URL of the Ollama server.
            api_key: Optional API key for authentication (cloud deployments).
        """
        # Strip ollama/ prefix — the library uses bare model names
        self._ollama_model = model_id.removeprefix("ollama/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = AsyncClient(host=base_url, headers=headers)

    async def generate_text(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> BaseModel:
        """Generate structured text using an Ollama model.

        Args:
            prompt: User prompt to send.
            schema: Pydantic model class for structured JSON output.
            temperature: Sampling temperature.
            system_prompt: Optional system instruction.
            max_retries: Retry attempts on failure.

        Returns:
            Validated Pydantic model instance.
        """
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call() -> BaseModel:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self._client.chat(
                model=self._ollama_model,
                messages=messages,
                format=schema.model_json_schema(),
                options={"temperature": temperature},
                stream=False,
            )
            return schema.model_validate_json(response.message.content)

        return await _call()

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        schema: Type[BaseModel],
        *,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> BaseModel:
        """Analyze an image using an Ollama vision model.

        Encodes image as base64 and passes it in the user message. Only
        works with vision-capable Ollama models (e.g., llava, moondream).

        Args:
            image_bytes: Raw image bytes.
            prompt: Analysis prompt.
            schema: Pydantic model class for structured JSON output.
            mime_type: Image MIME type (informational — Ollama uses raw base64).
            temperature: Sampling temperature.
            max_retries: Retry attempts on failure.

        Returns:
            Validated Pydantic model instance.
        """
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call() -> BaseModel:
            image_b64 = base64.b64encode(image_bytes).decode()
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ]
            response = await self._client.chat(
                model=self._ollama_model,
                messages=messages,
                format=schema.model_json_schema(),
                options={"temperature": temperature},
                stream=False,
            )
            return schema.model_validate_json(response.message.content)

        return await _call()
