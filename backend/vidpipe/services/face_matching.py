"""Face matching service using InsightFace ArcFace embeddings.

This module provides face embedding generation and cross-matching capabilities
using the InsightFace library with ArcFace models.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class FaceMatchingService:
    """ArcFace face embedding and cross-matching service with lazy model loading."""

    def __init__(self):
        """Initialize service. Model is loaded on first use."""
        self._app = None

    def _load_model(self):
        """Lazy-load InsightFace model on first use.

        Raises:
            RuntimeError: If model initialization fails
        """
        if self._app is not None:
            return

        try:
            from insightface.app import FaceAnalysis

            logger.info("Loading InsightFace buffalo_l model...")
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise RuntimeError(
                f"Failed to load InsightFace model: {e}. On first run, the buffalo_l "
                "model (~200MB) is auto-downloaded from insightface servers. Ensure "
                "internet connectivity and sufficient disk space. If CUDA is unavailable, "
                "install onnxruntime (CPU) as fallback."
            )

    def generate_embedding(self, face_crop_path: str) -> np.ndarray:
        """Generate 512-dim ArcFace embedding for face crop.

        Args:
            face_crop_path: Path to face crop image

        Returns:
            Normalized 512-dim embedding as numpy array

        Raises:
            ValueError: If no face detected in image
        """
        self._load_model()

        import cv2

        img = cv2.imread(face_crop_path)
        faces = self._app.get(img)

        if not faces:
            raise ValueError(f"No face detected in {face_crop_path}")

        # Get embedding from first detected face and normalize
        embedding = faces[0].embedding
        normalized = embedding / np.linalg.norm(embedding)

        return normalized

    def cross_match_faces(
        self, face_data: List[dict], similarity_threshold: float = 0.6
    ) -> List[List[int]]:
        """Cross-match all faces and return groups of same person.

        Args:
            face_data: List of dicts with "embedding" key (already-normalized np.ndarray)
            similarity_threshold: Cosine similarity threshold (0.6 = same person)

        Returns:
            List of groups (indices that are same person)
            Example: [[0, 3, 5], [1], [2, 4]] = faces 0,3,5 are same person
        """
        if not face_data:
            return []

        embeddings = np.array([f["embedding"] for f in face_data])
        n = len(embeddings)

        # Compute cosine similarity matrix
        # Since embeddings are already normalized, similarity = dot product
        similarity_matrix = embeddings @ embeddings.T

        # Group faces by similarity using transitive closure
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue

            # Find all faces similar to face i (including transitively)
            group = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j in visited:
                    continue

                if similarity_matrix[i, j] > similarity_threshold:
                    group.append(j)
                    visited.add(j)

            groups.append(group)

        return groups

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding (should be normalized)
            emb2: Second embedding (should be normalized)

        Returns:
            Cosine similarity (-1 to 1)
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
