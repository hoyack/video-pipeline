"""Abstract base class for LLM provider adapters.

Defines the consistent async interface that all adapters must implement,
supporting both text generation and image analysis with structured output.
"""

from abc import ABC, abstractmethod
from typing import Optional, Type

from pydantic import BaseModel


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    All adapters must implement generate_text() and analyze_image() with
    consistent async signatures. Both methods return validated Pydantic
    model instances using the caller-supplied schema class.
    """

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> BaseModel:
        """Generate structured text output from a prompt.

        Args:
            prompt: The user prompt to send to the model.
            schema: Pydantic model class defining the expected output structure.
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
            system_prompt: Optional system/instruction prompt.
            max_retries: Maximum number of retry attempts on failure.

        Returns:
            Validated instance of the supplied schema class.
        """
        ...

    @abstractmethod
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
        """Analyze an image and return structured output.

        Args:
            image_bytes: Raw bytes of the image to analyze.
            prompt: The analysis prompt describing what to extract.
            schema: Pydantic model class defining the expected output structure.
            mime_type: MIME type of the image (e.g., "image/jpeg", "image/png").
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
            max_retries: Maximum number of retry attempts on failure.

        Returns:
            Validated instance of the supplied schema class.
        """
        ...
