"""Ollama adapter for the LLM abstraction layer.

Connects via ollama.AsyncClient with optional auth headers, structured JSON
output via format='json' with schema instructions, and vision support via
base64-encoded images.

Note: We use format='json' instead of format=schema_dict because Ollama Cloud
does not reliably enforce JSON schema constraints — it sometimes returns
markdown or bare values when given a full JSON schema dict.  Instead, we
append a concise schema description to the system prompt and rely on
format='json' to guarantee valid JSON output.
"""

import base64
import json
import logging
from typing import Optional, Type

from ollama import AsyncClient
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from vidpipe.services.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


def _schema_instruction(schema: Type[BaseModel]) -> str:
    """Build a concise JSON schema instruction to append to the system prompt.

    Produces a compact representation of the expected output structure that
    LLMs follow more reliably than a raw JSON Schema dict passed via the
    format parameter.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "\n\nIMPORTANT: You MUST respond with a single JSON object (no markdown, "
        "no commentary, no code fences). The JSON must conform to this schema:\n"
        f"```json\n{schema_json}\n```\n"
        "All string fields must be strings (not arrays). Return ONLY the JSON object."
    )


class OllamaAdapter(LLMAdapter):
    """LLM adapter backed by a local or cloud Ollama instance.

    Strips the "ollama/" prefix from model IDs before passing to the ollama
    library. Always passes stream=False to avoid async generator responses.
    Uses format='json' with schema instructions in the prompt for reliable
    structured output across local and cloud Ollama deployments.
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
        schema_suffix = _schema_instruction(schema)

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call() -> BaseModel:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt + schema_suffix})
            else:
                messages.append({"role": "system", "content": schema_suffix.lstrip()})
            messages.append({"role": "user", "content": prompt})

            response = await self._client.chat(
                model=self._ollama_model,
                messages=messages,
                format="json",
                options={"temperature": temperature},
                stream=False,
            )

            raw = response.message.content
            # Some models wrap JSON in markdown code fences — strip them
            stripped = raw.strip()
            if stripped.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = stripped.index("\n")
                stripped = stripped[first_newline + 1:]
                if stripped.endswith("```"):
                    stripped = stripped[:-3].rstrip()
                raw = stripped

            return schema.model_validate_json(raw)

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
        schema_suffix = _schema_instruction(schema)

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call() -> BaseModel:
            image_b64 = base64.b64encode(image_bytes).decode()
            messages = [
                {"role": "system", "content": schema_suffix.lstrip()},
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                },
            ]
            response = await self._client.chat(
                model=self._ollama_model,
                messages=messages,
                format="json",
                options={"temperature": temperature},
                stream=False,
            )

            raw = response.message.content
            stripped = raw.strip()
            if stripped.startswith("```"):
                first_newline = stripped.index("\n")
                stripped = stripped[first_newline + 1:]
                if stripped.endswith("```"):
                    stripped = stripped[:-3].rstrip()
                raw = stripped

            return schema.model_validate_json(raw)

        return await _call()
