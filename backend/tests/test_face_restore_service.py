"""Tests for the GPU face-restore / upscale service.

Always-on tests cover config defaults and graceful degradation (missing
torch/spandrel/weights → service simply unavailable, never raises). The real
GPU restore/upscale is exercised by the live e2e run, not unit tests (it needs
a CUDA GPU + ~430MB of weights not present in CI).
"""

import pytest

from vidpipe.config import FaceSwapConfig
from vidpipe.services.face_restore_service import (
    FaceRestoreService,
    get_face_restore_service,
)


def test_restore_config_defaults():
    c = FaceSwapConfig()
    assert c.restore is False            # opt-in
    assert c.upscale_keyframes is False  # opt-in
    assert c.restore_weight == pytest.approx(0.9)   # identity-preserving default
    assert c.upscale_scale == 4


def test_degrades_gracefully_when_models_missing(monkeypatch, tmp_path):
    """No weights (or no torch/spandrel) → unavailable, ops return None, no raise."""
    from vidpipe.config import settings

    # point at an empty dir so the weight files are absent
    monkeypatch.setattr(settings.face_swap, "restore_model_dir", str(tmp_path), raising=False)
    svc = FaceRestoreService()  # fresh instance, bypass singleton cache
    # available() may be False due to missing weights and/or missing torch/spandrel
    assert svc.available() is False
    assert svc.has_restore() is False
    assert svc.has_upscale() is False
    assert svc.restore_primary_face(b"not-an-image") is None
    assert svc.upscale(b"not-an-image", 4) is None


def test_singleton_accessor():
    assert get_face_restore_service() is get_face_restore_service()
