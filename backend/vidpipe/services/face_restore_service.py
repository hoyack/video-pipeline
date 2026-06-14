"""GPU face restoration (CodeFormer) + super-resolution (RealESRGAN) via spandrel.

The inswapper swap is only 128px, so on a high-res keyframe the face comes back
soft. This service re-renders the swapped face at 512px with realistic detail
(CodeFormer) and pastes it back aligned, and can 4x-upscale a full frame
(RealESRGAN, tiled to fit 12GB VRAM).

Both are PyTorch models run via ``spandrel`` on the GPU (``torch.cuda``) — this
sidesteps the onnxruntime CPU/GPU package conflict entirely; only the cheap
128px swap stays on onnxruntime/CPU. Degrades gracefully (``available()`` False)
when torch/spandrel or the weights are missing, so the pipeline still runs.

Validated 2026-06-13 on an RTX 4070 Ti: CodeFormer restore ~0.75s/face,
RealESRGAN 4x ~3.8s/frame (tiled). CodeFormer ``weight`` is the fidelity knob —
0.9 keeps Brandon's identity (~0.81 vs a held-out real ref) while sharpening;
lower weights drift identity.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# FFHQ-512 five-point template — the standard GFPGAN/CodeFormer face alignment.
_FFHQ_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.19360], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]],
    dtype=np.float32,
)


class FaceRestoreService:
    """Lazy-loaded CodeFormer + RealESRGAN (GPU). Singleton via accessor below."""

    def __init__(self) -> None:
        self._tried = False
        self._cf = None       # CodeFormer descriptor
        self._esr = None      # RealESRGAN descriptor
        self._app = None      # buffalo_l detector (for align)
        self._dev = "cpu"

    def _load(self) -> bool:
        if self._tried:
            return self._cf is not None or self._esr is not None
        self._tried = True
        try:
            import torch
            from spandrel import ModelLoader, MAIN_REGISTRY

            try:
                from spandrel_extra_arches import EXTRA_REGISTRY

                MAIN_REGISTRY.add(*EXTRA_REGISTRY)
            except Exception as e:  # noqa: BLE001 - CodeFormer arch lives in extra_arches
                logger.warning("spandrel_extra_arches unavailable (CodeFormer may not load): %s", e)

            from vidpipe.config import settings

            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            mdir = settings.face_swap.restore_model_dir
            loader = ModelLoader(device=self._dev)

            cf_path = os.path.join(mdir, "codeformer.pth")
            esr_path = os.path.join(mdir, "RealESRGAN_x4plus.pth")
            if os.path.exists(cf_path):
                self._cf = loader.load_from_file(cf_path).eval()
            else:
                logger.warning("Face-restore: CodeFormer weight missing at %s", cf_path)
            if os.path.exists(esr_path):
                self._esr = loader.load_from_file(esr_path).eval()
            else:
                logger.warning("Face-restore: RealESRGAN weight missing at %s", esr_path)

            if self._cf is not None or self._esr is not None:
                from insightface.app import FaceAnalysis

                self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                self._app.prepare(ctx_id=-1, det_size=(640, 640))
                logger.info(
                    "FaceRestoreService ready (device=%s, codeformer=%s, realesrgan=%s).",
                    self._dev, self._cf is not None, self._esr is not None,
                )
        except Exception as e:  # noqa: BLE001 - any failure → restore unavailable
            logger.warning("Face-restore unavailable: %s", e)
            self._cf = self._esr = None
        return self._cf is not None or self._esr is not None

    def available(self) -> bool:
        return self._load()

    def has_restore(self) -> bool:
        return self._load() and self._cf is not None

    def has_upscale(self) -> bool:
        return self._load() and self._esr is not None

    # --- image helpers --------------------------------------------------
    @staticmethod
    def _decode(data: bytes):
        import cv2

        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _encode(bgr) -> bytes:
        import cv2

        ok, buf = cv2.imencode(".png", bgr)
        return buf.tobytes() if ok else b""

    def _to_dev(self, bgr):
        import cv2
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._dev)

    def _to_bgr(self, t):
        import cv2

        a = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255.0
        return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # --- public ops -----------------------------------------------------
    def restore_primary_face(self, png: bytes) -> Optional[bytes]:
        """CodeFormer-restore the largest face and paste it back. None if unavailable/no face."""
        if not self.has_restore():
            return None
        try:
            import cv2
            import torch

            from vidpipe.config import settings

            weight = float(settings.face_swap.restore_weight)
            img = self._decode(png)
            if img is None:
                return None
            faces = self._app.get(img)
            if not faces:
                return None
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            M, _ = cv2.estimateAffinePartial2D(
                f.kps.astype(np.float32), _FFHQ_512, method=cv2.LMEDS
            )
            if M is None:
                return None
            h, w = img.shape[:2]
            aligned = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_LINEAR)
            with torch.no_grad():
                out = self._cf.model(self._to_dev(aligned), weight=weight)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            restored = self._to_bgr(out)

            m_inv = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(restored, m_inv, (w, h))
            # Blend with a soft FACE-SHAPED (elliptical) mask built in the aligned
            # 512 space and feathered heavily, so the restored region fades as an
            # oval that dies out well inside the crop — no square crop boundary /
            # box-outline seam where restored meets original.
            mask512 = np.zeros((512, 512), np.float32)
            cv2.ellipse(mask512, (256, 286), (158, 198), 0, 0, 360, 1.0, -1)
            mask512 = cv2.GaussianBlur(mask512, (0, 0), 26)
            mask = np.clip(cv2.warpAffine(mask512, m_inv, (w, h)), 0.0, 1.0)[..., None]
            full = (img * (1.0 - mask) + back * mask).astype(np.uint8)
            return self._encode(full)
        except Exception as e:  # noqa: BLE001
            logger.warning("Face restore failed: %s", e)
            return None

    def upscale(self, png: bytes, scale: int = 4, tile: int = 512, overlap: int = 32) -> Optional[bytes]:
        """RealESRGAN super-resolution, tiled to stay within VRAM. None if unavailable."""
        if not self.has_upscale():
            return None
        try:
            import torch

            img = self._decode(png)
            if img is None:
                return None
            s = int(getattr(self._esr, "scale", 4) or 4)
            h, w = img.shape[:2]
            out = np.zeros((h * s, w * s, 3), np.uint8)
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    y2, x2 = min(y + tile, h), min(x + tile, w)
                    ys, xs = max(0, y - overlap), max(0, x - overlap)
                    ye, xe = min(h, y2 + overlap), min(w, x2 + overlap)
                    with torch.no_grad():
                        up = self._to_bgr(self._esr(self._to_dev(img[ys:ye, xs:xe])))
                    ty, tx = (y - ys) * s, (x - xs) * s
                    out[y * s:y2 * s, x * s:x2 * s] = up[ty:ty + (y2 - y) * s, tx:tx + (x2 - x) * s]
            return self._encode(out)
        except Exception as e:  # noqa: BLE001
            logger.warning("Upscale failed: %s", e)
            return None


_SERVICE: Optional[FaceRestoreService] = None


def get_face_restore_service() -> FaceRestoreService:
    """Module-singleton accessor for the face-restore/upscale service."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = FaceRestoreService()
    return _SERVICE
