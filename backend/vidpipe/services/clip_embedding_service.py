"""CLIP visual similarity embedding service.

Uses openai/clip-vit-base-patch32 model (512-dim vectors) for computing
visual similarity between images. All models are lazy-loaded on first use
following the Phase 5 lazy-loading pattern.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CLIPEmbeddingService:
    """CLIP visual similarity embeddings with lazy model loading.

    Uses openai/clip-vit-base-patch32 (512-dim, ~15ms/image).
    Follows Phase 5 lazy-loading pattern (load on first use, not import time).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """Initialize service.

        Args:
            model_name: HuggingFace model name to use for CLIP embeddings.
            device: Device for inference ("cuda", "cpu"). If None, auto-detects.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy-load CLIPProcessor and CLIPModel on first use.

        Raises:
            RuntimeError: If model loading fails (network error, corrupt weights, etc.)
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            # Auto-detect device if not specified
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"Loading CLIP model ({self.model_name}) on device={self.device}..."
            )
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(
                f"Failed to load CLIP model: {e}. On first run, models (~600MB) are "
                "auto-downloaded from HuggingFace. Ensure internet connectivity and write "
                "access to the HuggingFace cache directory (~/.cache/huggingface/). "
                "Install dependencies with: pip install transformers torch Pillow"
            )

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate a 512-dim normalized CLIP embedding for an image.

        Args:
            image_path: Path to the image file.

        Returns:
            512-dimensional normalized numpy array (unit vector).
        """
        import torch
        from PIL import Image

        self._load_model()

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")

        # Move inputs to device if GPU
        if self.device and self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.get_image_features(**inputs)

        # transformers 5.x returns BaseModelOutputWithPooling instead of tensor
        if isinstance(output, torch.Tensor):
            features = output
        else:
            features = output.pooler_output if hasattr(output, "pooler_output") else output[0]

        # Normalize to unit vector
        embedding = features.cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two CLIP embeddings.

        Args:
            emb1: First 512-dim embedding (unit vector).
            emb2: Second 512-dim embedding (unit vector).

        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        return float(np.dot(emb1, emb2))
