"""Tests for the post-generation face-swap service (vidpipe.services.face_swap_service).

The graceful-degrade path (inswapper weight absent) runs everywhere and is the
safety net that keeps the pipeline working when the model isn't installed. The
real-swap assertions are gated on the weight existing locally (it is gitignored
and not present in CI) plus a sample face image supplied via env var — mirroring
how the heavier QA tests opt in (see test_video_qa.py).

    # Run the gated real-swap tests against a local model + face images:
    VIDPIPE_FACESWAP_TEST_IMAGE=/path/to/source_face.png \
    VIDPIPE_FACESWAP_TARGET_IMAGE=/path/to/other_face.png \
    pytest backend/tests/test_face_swap_service.py -q
"""

import os

import pytest

from vidpipe.config import FaceSwapConfig
from vidpipe.services.face_swap_service import (
    FaceSwapService,
    _inswapper_path,
    get_face_swap_service,
)

# Cheap file-existence gate (does NOT load the ~800MB models at collection time).
_HAVE_MODEL = os.path.exists(_inswapper_path())
requires_model = pytest.mark.skipif(
    not _HAVE_MODEL, reason="inswapper_128.onnx not installed locally."
)


# --- Always-on: config defaults + graceful degrade -------------------------
def test_config_defaults_opt_in_off():
    """Face-swap is opt-in; restoration deferred; det-score gate set."""
    c = FaceSwapConfig()
    assert c.enabled is False
    assert c.restore is False
    assert c.min_det_score == pytest.approx(0.50)


def test_degrades_gracefully_when_model_missing(monkeypatch, tmp_path):
    """No inswapper weight → service unavailable and swaps no-op (never raises)."""
    missing = str(tmp_path / "nope" / "inswapper_128.onnx")
    monkeypatch.setattr(
        "vidpipe.services.face_swap_service._inswapper_path", lambda: missing
    )
    svc = FaceSwapService()  # fresh instance — bypass the module singleton cache
    assert svc.available() is False
    # Bad/empty inputs must not raise when the service is unavailable.
    assert svc.swap_face_with_score(b"not-an-image", b"also-not") == (None, 0.0)
    assert svc.swap_face(b"x", b"y") is None
    assert svc.pick_clearest([b"x", b"y"]) is None


def test_singleton_accessor_returns_same_instance():
    assert get_face_swap_service() is get_face_swap_service()


# --- Gated on the model actually being installed locally --------------------
def _load_svc():
    svc = get_face_swap_service()
    if not svc.available():
        pytest.skip("face-swap service failed to load despite weight present.")
    return svc


@requires_model
def test_pick_clearest_prefers_detectable_face():
    """A real face outranks a noise image (which has no detectable face)."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    sample = os.getenv("VIDPIPE_FACESWAP_TEST_IMAGE")
    if not sample or not os.path.exists(sample):
        pytest.skip("Set VIDPIPE_FACESWAP_TEST_IMAGE to a real face image.")
    svc = _load_svc()

    with open(sample, "rb") as f:
        face_png = f.read()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", noise)
    noise_png = buf.tobytes()

    assert svc.pick_clearest([noise_png, face_png]) == face_png
    assert svc.pick_clearest([noise_png]) is None


@requires_model
def test_swap_moves_identity_toward_source():
    """Swapping source onto a target makes the result resemble the source.

    With two distinct faces this is the POC assertion (0.17 -> 0.87): the
    swapped frame's similarity to the source exceeds the original target's
    similarity to the source. With a single image it degenerates to a self-swap
    smoke test (high similarity, swap machinery works end-to-end).
    """
    from vidpipe.qa.vision import embedding_similarity

    source = os.getenv("VIDPIPE_FACESWAP_TEST_IMAGE")
    if not source or not os.path.exists(source):
        pytest.skip("Set VIDPIPE_FACESWAP_TEST_IMAGE to a real face image.")
    target = os.getenv("VIDPIPE_FACESWAP_TARGET_IMAGE") or source
    svc = _load_svc()

    with open(source, "rb") as f:
        source_png = f.read()
    with open(target, "rb") as f:
        target_png = f.read()

    swapped, sim = svc.swap_face_with_score(target_png, source_png)
    assert swapped is not None, "expected a detectable face in the target"
    assert sim >= 0.5, f"swapped face barely resembles source (sim={sim:.3f})"

    if target != source:
        before = embedding_similarity(source_png, target_png) or 0.0
        assert sim > before, (
            f"swap did not move identity toward source "
            f"(before={before:.3f}, after={sim:.3f})"
        )
