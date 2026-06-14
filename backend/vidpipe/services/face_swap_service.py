"""Post-generation face-swap service (InsightFace inswapper).

Pastes a character's *real* reference face onto a generated keyframe so identity
is exact regardless of which image model produced the keyframe. The swap uses
the ``inswapper_128.onnx`` model plus the ``buffalo_l`` detector/recogniser that
the rest of the pipeline already relies on (see ``qa/vision.py`` and
``services/face_matching.py``).

Runs CPU-only to match the GPU-disabled backend container (a 128px swap is fast
on CPU). The ``inswapper_128.onnx`` weight is expected under
``~/.insightface/models/`` — in Docker this is the volume-mounted
``./models/insightface`` dir. When the weight is missing the service degrades
gracefully (``available()`` returns False) so the pipeline proceeds with
un-swapped keyframes rather than failing.

Acquisition note: ``inswapper_128.onnx`` is gated/licensed — source it
deliberately and confirm usage terms for the project's context.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

INSWAPPER_FILENAME = "inswapper_128.onnx"


def _inswapper_path() -> str:
    """Resolve the inswapper weight path (volume-mounted insightface model dir)."""
    return os.path.join(
        os.path.expanduser("~"), ".insightface", "models", INSWAPPER_FILENAME
    )


class FaceSwapService:
    """Lazy-loaded InsightFace face swapper (CPU). Singleton via accessor below."""

    def __init__(self) -> None:
        self._app = None  # buffalo_l FaceAnalysis (detect + recognise)
        self._swapper = None  # inswapper_128 INSwapper
        self._tried = False
        self._ok = False

    def _load(self) -> bool:
        """Lazy-init detector + swapper. Returns True if both are ready."""
        if self._tried:
            return self._ok
        self._tried = True
        try:
            model_path = _inswapper_path()
            if not os.path.exists(model_path):
                logger.warning(
                    "Face-swap unavailable: %s not found (mount ./models/insightface "
                    "to /root/.insightface/models).",
                    model_path,
                )
                self._ok = False
                return False

            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
            self._swapper = insightface.model_zoo.get_model(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._ok = True
            logger.info("FaceSwapService ready (inswapper_128, buffalo_l, CPU).")
        except Exception as e:  # noqa: BLE001 - any failure → swap unavailable
            logger.warning("Face-swap unavailable: %s", e)
            self._ok = False
        return self._ok

    def available(self) -> bool:
        """True when both models loaded (lazy-loads on first call)."""
        return self._load()

    @staticmethod
    def _decode(data: bytes):
        """PNG/JPEG bytes → BGR ndarray (or None if undecodable)."""
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def pick_clearest(self, candidate_pngs: list[bytes]) -> Optional[bytes]:
        """Return the candidate whose best face has the highest detection score.

        Reuses the same det_score ranking as reference prequalification so the
        clearest real reference is chosen as the swap source. Returns None when
        no candidate has a detectable face.
        """
        if not self._load():
            return None
        best_bytes: Optional[bytes] = None
        best_score = -1.0
        for png in candidate_pngs:
            try:
                img = self._decode(png)
                if img is None:
                    continue
                faces = self._app.get(img)
                if not faces:
                    continue
                score = max(float(f.det_score) for f in faces)
                if score > best_score:
                    best_score = score
                    best_bytes = png
            except Exception as e:  # noqa: BLE001
                logger.warning("Face-swap pick_clearest failed on a candidate: %s", e)
        return best_bytes

    def swap_face_with_score(
        self, target_png: bytes, source_png: bytes
    ) -> tuple[Optional[bytes], float]:
        """Swap the source identity onto the largest qualifying face in target.

        Returns ``(swapped_png_bytes, source_vs_swapped_similarity)``. Returns
        ``(None, 0.0)`` when the source has no detectable face, or the target has
        no face passing ``settings.face_swap.min_det_score`` (e.g. heavily
        stylised renders where detection misses) — the caller then keeps the
        generated frame and skips gracefully.
        """
        if not self._load():
            return None, 0.0
        try:
            import cv2  # type: ignore

            from vidpipe.config import settings

            src_bgr = self._decode(source_png)
            tgt_bgr = self._decode(target_png)
            if src_bgr is None or tgt_bgr is None:
                return None, 0.0

            src_faces = self._app.get(src_bgr)
            if not src_faces:
                return None, 0.0
            src_face = max(src_faces, key=lambda f: float(f.det_score))

            tgt_faces = self._app.get(tgt_bgr)
            min_det = settings.face_swap.min_det_score
            qualified = [f for f in tgt_faces if float(f.det_score) >= min_det]
            if not qualified:
                return None, 0.0
            # Largest qualifying face = the shot's primary subject.
            tgt_face = max(
                qualified,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            result_bgr = self._swapper.get(
                tgt_bgr, tgt_face, src_face, paste_back=True
            )
            ok, buf = cv2.imencode(".png", result_bgr)
            if not ok:
                return None, 0.0
            swapped_png = buf.tobytes()

            # Observability: the swapped face IS the real face, so similarity to
            # the source should be high (POC moved 0.17 -> 0.87). A low value here
            # is a red flag worth recording.
            from vidpipe.qa.vision import embedding_similarity

            sim = embedding_similarity(source_png, swapped_png)
            return swapped_png, float(sim or 0.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("Face-swap failed: %s", e)
            return None, 0.0

    def swap_face(self, target_png: bytes, source_png: bytes) -> Optional[bytes]:
        """Swap source identity onto target; None when no usable face. Thin wrapper."""
        out, _ = self.swap_face_with_score(target_png, source_png)
        return out


_SERVICE: Optional[FaceSwapService] = None


def get_face_swap_service() -> FaceSwapService:
    """Module-singleton accessor for the face-swap service."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = FaceSwapService()
    return _SERVICE
