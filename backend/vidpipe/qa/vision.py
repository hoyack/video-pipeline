"""Identity / story-alignment judges.

Two independent signals, per the QA design:

1. Face-embedding similarity (InsightFace) — objective image-vs-image identity
   score between a candidate keyframe and the character's Bible reference photo.
   Degrades gracefully: returns None when the lib/model is unavailable or no
   face is detected (itself a reportable condition for character shots).

2. Vision-LLM semantic judge — uses the production's *selected* vision model
   (Gemini or Ollama) via ``get_adapter``; checks the candidate against the
   canonical Bible appearance and the scene intent. Prompted adversarially:
   default to a low score when uncertain.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Face-embedding judge (optional dependency) ---------------------------
_FACE_APP = None
_FACE_TRIED = False


def _face_app():
    global _FACE_APP, _FACE_TRIED
    if _FACE_TRIED:
        return _FACE_APP
    _FACE_TRIED = True
    try:
        from insightface.app import FaceAnalysis  # type: ignore

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
        logger.info("InsightFace ready (buffalo_l, CPU).")
    except Exception as e:  # noqa: BLE001 - any failure → embeddings unavailable
        logger.warning("Face-embedding judge unavailable: %s", e)
        _FACE_APP = None
    return _FACE_APP


def _largest_embedding(img_bytes: bytes):
    app = _face_app()
    if app is None:
        return None
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return faces[-1].normed_embedding
    except Exception as e:  # noqa: BLE001
        logger.warning("Embedding extraction failed: %s", e)
        return None


def embedding_similarity(ref_bytes: bytes, cand_bytes: bytes) -> Optional[float]:
    """Cosine similarity of largest faces, or None if either has no detectable face."""
    e_ref = _largest_embedding(ref_bytes)
    e_cand = _largest_embedding(cand_bytes)
    if e_ref is None or e_cand is None:
        return None
    import numpy as np  # type: ignore

    return float(np.dot(e_ref, e_cand))  # both are L2-normalized


def embeddings_available() -> bool:
    return _face_app() is not None


# --- Vision-LLM judge -----------------------------------------------------
class IdentityVerdict(BaseModel):
    identity_match: float = Field(..., ge=0.0, le=1.0,
        description="0..1 confidence the depicted person matches the canonical character description.")
    is_on_model: bool = Field(..., description="True only if all key, hard-to-change features match.")
    observed_subject: str = Field(..., description="Brief description of who/what is actually shown.")
    discrepancies: list[str] = Field(default_factory=list,
        description="Specific mismatches vs the canonical description (e.g. 'has hair; should be bald').")
    scene_alignment: float = Field(..., ge=0.0, le=1.0,
        description="0..1 how well the image depicts the described scene intent/setting.")
    scene_notes: str = Field("", description="Notes on scene/setting alignment.")


def _identity_prompt(character_name: str, appearance: str, scene_intent: str) -> str:
    return (
        "You are a strict, adversarial film continuity QA reviewer. Your job is to "
        "CATCH identity errors, not to be charitable.\n\n"
        f"CANONICAL CHARACTER — {character_name}:\n{appearance}\n\n"
        f"INTENDED SCENE:\n{scene_intent}\n\n"
        "Examine the attached frame. Judge whether the most prominent person IS this "
        "exact character, focusing on hard-to-change features (hairline/baldness, "
        "face shape, glasses style, facial hair, distinguishing marks). Wardrobe and "
        "lighting may vary; identity must not.\n"
        "Rules: if NO clearly visible person, set identity_match=0 and note it. If "
        "uncertain, score LOW. List concrete discrepancies. Also rate how well the "
        "image depicts the intended scene (setting/action), independent of identity."
    )


async def judge_identity(adapter, image_bytes: bytes, *, character_name: str,
                         appearance: str, scene_intent: str,
                         mime_type: str = "image/png") -> Optional[IdentityVerdict]:
    """Run the selected vision LLM. Returns None on adapter error."""
    try:
        result = await adapter.analyze_image(
            image_bytes,
            _identity_prompt(character_name, appearance, scene_intent),
            IdentityVerdict,
            mime_type=mime_type,
            temperature=0.1,
        )
        return result  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        logger.warning("Vision-LLM judge failed: %s", e)
        return None
