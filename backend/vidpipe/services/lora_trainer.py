"""LoRA training service with pluggable backend abstraction.

Provides dataset preparation (resize + VLM caption + zip packaging),
training dispatch via Replicate (or other backends), status polling,
and weight download/storage.

All Replicate SDK calls are wrapped in asyncio.to_thread() to avoid
blocking the async event loop.

Spec reference: Phase 25 - LORA-01, LORA-02
"""

import asyncio
import io
import logging
import uuid
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Actor, ActorRef
from vidpipe.services.file_manager import FileManager
from vidpipe.services.storage_backend import StorageBackend

if TYPE_CHECKING:
    from vidpipe.services.llm.base import LLMAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TrainingJob:
    """Represents the state of a LoRA training job."""

    job_id: str
    status: str  # QUEUED, TRAINING, COMPLETED, FAILED
    progress: Optional[float] = None  # 0.0-1.0
    weights_url: Optional[str] = None  # URL to download .safetensors
    error: Optional[str] = None
    logs: Optional[str] = None


class ImageCaption(BaseModel):
    """Pydantic schema for VLM captioning output."""

    caption: str


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class LoRATrainingBackend(ABC):
    """Abstract base class for LoRA training providers."""

    @abstractmethod
    async def dispatch(
        self,
        dataset_url: str,
        trigger_word: str,
        *,
        steps: int = 1000,
        lora_rank: int = 32,
        learning_rate: float = 0.0004,
    ) -> TrainingJob:
        """Submit a training job and return initial TrainingJob with QUEUED status."""
        ...

    @abstractmethod
    async def poll_status(self, job_id: str) -> TrainingJob:
        """Poll the current status of a training job."""
        ...

    @abstractmethod
    async def get_result(self, job_id: str) -> bytes:
        """Download the trained weights as raw bytes."""
        ...


# ---------------------------------------------------------------------------
# Replicate implementation
# ---------------------------------------------------------------------------

# Replicate status → internal status mapping
_REPLICATE_STATUS_MAP = {
    "starting": "QUEUED",
    "processing": "TRAINING",
    "succeeded": "COMPLETED",
    "failed": "FAILED",
    "canceled": "FAILED",
}


class ReplicateBackend(LoRATrainingBackend):
    """LoRA training backend using Replicate's flux-dev-lora-trainer.

    All SDK calls are wrapped in asyncio.to_thread() because the replicate
    SDK is synchronous and would block the event loop otherwise.
    """

    TRAINER_MODEL = "ostris/flux-dev-lora-trainer"
    TRAINER_VERSION = "e440909d2a14b8af4478fcf6fba62a8a7e258e90bca1aec56cdf52c81ce29e30"

    def __init__(self, api_token: str, username: str = "vidpipe") -> None:
        import replicate

        self._client = replicate.Client(api_token=api_token)
        self._username = username

    async def dispatch(
        self,
        dataset_url: str,
        trigger_word: str,
        *,
        steps: int = 1000,
        lora_rank: int = 32,
        learning_rate: float = 0.0004,
    ) -> TrainingJob:
        """Submit a LoRA training job to Replicate."""
        destination = f"{self._username}/vidpipe-lora-{trigger_word.lower()}"

        def _create() -> object:
            return self._client.trainings.create(
                version=self.TRAINER_VERSION,
                input={
                    "input_images": dataset_url,
                    "trigger_word": trigger_word,
                    "steps": steps,
                    "lora_rank": lora_rank,
                    "learning_rate": learning_rate,
                    "autocaption": False,
                    "resolution": "512,768,1024",
                },
                destination=destination,
            )

        training = await asyncio.to_thread(_create)
        logger.info(
            "Dispatched LoRA training job %s for trigger '%s' -> %s",
            training.id, trigger_word, destination,
        )
        return TrainingJob(job_id=training.id, status="QUEUED")

    async def poll_status(self, job_id: str) -> TrainingJob:
        """Poll Replicate for training job status."""

        def _get() -> object:
            return self._client.trainings.get(job_id)

        training = await asyncio.to_thread(_get)
        status = _REPLICATE_STATUS_MAP.get(training.status, "QUEUED")

        weights_url = None
        if status == "COMPLETED" and training.output:
            # training.output may contain a 'weights' key with the URL
            if isinstance(training.output, dict):
                weights_url = training.output.get("weights")
            elif isinstance(training.output, str):
                weights_url = training.output

        progress = None
        if hasattr(training, "metrics") and training.metrics:
            progress = training.metrics.get("predict_time_share")

        return TrainingJob(
            job_id=job_id,
            status=status,
            progress=progress,
            weights_url=weights_url,
            error=str(training.error) if training.error else None,
            logs=training.logs if hasattr(training, "logs") else None,
        )

    async def get_result(self, job_id: str) -> bytes:
        """Download trained weights from Replicate output URL."""
        job = await self.poll_status(job_id)
        if not job.weights_url:
            raise ValueError(f"Training job {job_id} has no weights URL (status: {job.status})")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(job.weights_url)
            response.raise_for_status()
            return response.content


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _resize_with_padding(image_bytes: bytes, target_size: int = 1024) -> bytes:
    """Resize an image to target_size x target_size with white padding.

    Uses PIL.ImageOps.pad() to maintain aspect ratio and fill with white.
    Output format is PNG.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    padded = ImageOps.pad(img, (target_size, target_size), color=(255, 255, 255))
    buf = io.BytesIO()
    padded.save(buf, format="PNG")
    return buf.getvalue()


async def _generate_caption(
    adapter: "LLMAdapter",
    image_bytes: bytes,
    trigger_word: str,
) -> str:
    """Generate a detailed appearance caption for an image using VLM.

    The caption starts with the trigger word for LoRA training association.
    Uses low temperature (0.3) for consistent descriptive output.
    """
    prompt = (
        f"Describe the appearance of the subject in this image in detail. "
        f"Do not use proper names. Focus on physical features, clothing, "
        f"expression, and pose. Start the description with the word '{trigger_word}'."
    )
    result = await adapter.analyze_image(
        image_bytes=image_bytes,
        prompt=prompt,
        schema=ImageCaption,
        mime_type="image/png",
        temperature=0.3,
    )
    return result.caption


async def prepare_dataset(
    actor_id: uuid.UUID,
    session: AsyncSession,
    vision_adapter: "LLMAdapter",
    file_mgr: FileManager,
) -> tuple[bytes, str]:
    """Prepare a training dataset zip from an Actor's reference images.

    Downloads reference images, resizes to 1024x1024 with padding, generates
    captions via VLM, and packages everything into a zip file with matching
    {index:03d}.png and {index:03d}.txt filenames.

    Args:
        actor_id: UUID of the Actor to build the dataset for
        session: Async SQLAlchemy session
        vision_adapter: LLM adapter with vision capability for captioning
        file_mgr: FileManager for reading reference images

    Returns:
        Tuple of (zip_bytes, trigger_word)

    Raises:
        ValueError: If actor not found or has no reference images
    """
    # Load actor
    actor = await session.get(Actor, actor_id)
    if actor is None:
        raise ValueError(f"Actor {actor_id} not found")

    # Load reference images
    ref_result = await session.execute(
        select(ActorRef).where(ActorRef.actor_id == actor_id)
    )
    refs = ref_result.scalars().all()
    if not refs:
        raise ValueError(f"Actor {actor_id} has no reference images")

    # Build trigger word from actor name
    trigger_word = f"ACTOR_{actor.name.upper().replace(' ', '_')}"

    # Build zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, ref in enumerate(refs):
            logger.info(
                "Processing ref %d/%d for actor '%s': %s",
                i + 1, len(refs), actor.name, ref.image_url,
            )

            # Read image bytes via file manager (handles S3 keys and local paths)
            image_bytes = await file_mgr.read_bytes(ref.image_url)

            # Resize with padding to 1024x1024
            resized = _resize_with_padding(image_bytes, target_size=1024)

            # Generate caption via VLM
            caption = await _generate_caption(vision_adapter, resized, trigger_word)

            # Add to zip
            filename_base = f"{i:03d}"
            zf.writestr(f"{filename_base}.png", resized)
            zf.writestr(f"{filename_base}.txt", caption)

    zip_bytes = zip_buffer.getvalue()
    logger.info(
        "Prepared dataset for actor '%s': %d images, %.1f KB zip",
        actor.name, len(refs), len(zip_bytes) / 1024,
    )
    return zip_bytes, trigger_word


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------


async def download_and_store_weights(
    weights_url: str,
    actor_id: str,
    storage: StorageBackend,
) -> str:
    """Download trained LoRA weights and store them via the storage backend.

    Downloads .safetensors from Replicate output URL, stores at
    asset-library/actors/{actor_id}/lora/model.safetensors.

    Args:
        weights_url: URL to download the .safetensors file from
        actor_id: Actor UUID (string) for storage key namespacing
        storage: StorageBackend instance (S3 or local)

    Returns:
        Storage key for the stored weights file
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(weights_url)
        response.raise_for_status()
        weights_bytes = response.content

    key = f"asset-library/actors/{actor_id}/lora/model.safetensors"
    await storage.put(key, weights_bytes, "application/octet-stream")
    logger.info("Stored LoRA weights at %s (%.1f MB)", key, len(weights_bytes) / (1024 * 1024))
    return key
