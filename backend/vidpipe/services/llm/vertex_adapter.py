"""Vertex AI adapter for the LLM abstraction layer.

Wraps google-genai client with location-aware routing and structured output.
Uses tenacity for retry logic with configurable max_retries.
"""

import logging
from typing import Optional, Type

from google.genai import types as genai_types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from vidpipe.services.llm.base import LLMAdapter
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

logger = logging.getLogger(__name__)


class VertexAIAdapter(LLMAdapter):
    """LLM adapter backed by Google Vertex AI (google-genai SDK).

    Supports structured JSON output via response_schema and uses the
    location-aware client singleton from vertex_client.py.
    """

    def __init__(self, model_id: str) -> None:
        """Initialize adapter for the given Vertex AI model.

        Args:
            model_id: Vertex AI model identifier (e.g., "gemini-2.5-flash").
        """
        self._model_id = model_id

    async def generate_text(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> BaseModel:
        """Generate structured text using Vertex AI.

        Args:
            prompt: User prompt to send.
            schema: Pydantic model class for structured output.
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
            client = get_vertex_client(location=location_for_model(self._model_id))
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            )
            if system_prompt:
                config = genai_types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                    system_instruction=system_prompt,
                )
            response = await client.aio.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            )
            return schema.model_validate_json(response.text)

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
        """Analyze an image using Vertex AI vision capabilities.

        Args:
            image_bytes: Raw image bytes.
            prompt: Analysis prompt.
            schema: Pydantic model class for structured output.
            mime_type: Image MIME type.
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
            client = get_vertex_client(location=location_for_model(self._model_id))
            image_part = genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = await client.aio.models.generate_content(
                model=self._model_id,
                contents=[image_part, prompt],
                config=config,
            )
            return schema.model_validate_json(response.text)

        return await _call()
