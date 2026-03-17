"""Reference image prequalification for character identity preservation.

Screens ActorRef images for face detectability using ArcFace (InsightFace).
Filters out references where no face can be extracted, preventing
multi-reference confusion in Gemini image generation.
"""

import logging
import uuid
from dataclasses import dataclass

import numpy as np

from vidpipe.services.face_matching import FaceMatchingService
from vidpipe.services.file_manager import FileManager

logger = logging.getLogger(__name__)


@dataclass
class QualifiedRef:
    """A reference image that passed face detectability screening."""

    image_url: str
    image_bytes: bytes
    face_embedding: np.ndarray  # 512-dim normalized ArcFace
    detection_score: float  # InsightFace detection score


async def prequalify_refs(
    ref_urls: list[str],
    file_mgr: FileManager,
    face_svc: FaceMatchingService | None = None,
) -> list[QualifiedRef]:
    """Screen reference images for face detectability.

    For each ref URL:
      1. Read bytes via file_mgr.read_bytes()
      2. Try ArcFace generate_embedding_from_bytes()
      3. If face found -> include; if ValueError -> drop with warning

    Args:
        ref_urls: List of storage keys / paths for reference images.
        file_mgr: FileManager for reading image bytes.
        face_svc: Optional FaceMatchingService instance (created if None).

    Returns:
        List of QualifiedRef sorted by detection score (highest first).
        Refs with no detectable face are excluded.
    """
    if face_svc is None:
        face_svc = FaceMatchingService()

    qualified: list[QualifiedRef] = []

    for i, url in enumerate(ref_urls):
        try:
            img_bytes = await file_mgr.read_bytes(url)
        except Exception as e:
            logger.warning("Ref %d: failed to read %s — %s", i, url, e)
            continue

        try:
            embedding, det_score = _extract_embedding_with_score(face_svc, img_bytes)
            qualified.append(QualifiedRef(
                image_url=url,
                image_bytes=img_bytes,
                face_embedding=embedding,
                detection_score=det_score,
            ))
            logger.info("Ref %d: qualified (det_score=%.3f) — %s", i, det_score, url)
        except ValueError:
            logger.warning("Ref %d: no face detected, dropping — %s", i, url)

    # Sort by detection score descending (best face first)
    qualified.sort(key=lambda r: r.detection_score, reverse=True)

    logger.info(
        "Prequalification: %d / %d refs qualified",
        len(qualified), len(ref_urls),
    )
    return qualified


def _extract_embedding_with_score(
    face_svc: FaceMatchingService,
    image_bytes: bytes,
) -> tuple[np.ndarray, float]:
    """Extract ArcFace embedding and detection score from image bytes.

    Uses InsightFace's face analysis directly to get both embedding
    and detection confidence in a single pass.

    Returns:
        (normalized_embedding, detection_score)

    Raises:
        ValueError: If no face detected.
    """
    import io
    import cv2
    from PIL import Image

    face_svc._load_model()

    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    faces = face_svc._app.get(img_bgr)
    if not faces:
        raise ValueError("No face detected in image bytes")

    # Use the face with highest detection score
    best_face = max(faces, key=lambda f: f.det_score)
    embedding = best_face.embedding
    normalized = embedding / np.linalg.norm(embedding)

    return normalized, float(best_face.det_score)
