"""Tests for degenerate-clip detection (vidpipe.services.clip_validation).

Reproduces the observed ComfyUI Cloud failure mode: an MP4 whose first and
last frames match the FLF2V conditioning keyframes but whose middle frames
are noise. Synthetic clips are built with cv2 so tests are self-contained.
"""

import io

import numpy as np
import pytest
from PIL import Image

from vidpipe.services.clip_validation import validate_clip_integrity

cv2 = pytest.importorskip("cv2")

W, H, FPS, N_FRAMES = 320, 192, 16, 48
RNG = np.random.default_rng(42)


def _scene_frame(t: float) -> np.ndarray:
    """A structured 'real' frame: gradient background + moving square."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(40, 200, W, dtype=np.uint8)
    frame[:, :, 0] = xs[None, :]
    frame[:, :, 1] = np.linspace(60, 180, H, dtype=np.uint8)[:, None]
    # Slowly moving square (continuous motion between frames)
    x = int(20 + t * 150) % (W - 60)
    y = int(30 + t * 60) % (H - 60)
    frame[y : y + 50, x : x + 50] = (220, 90, 40)
    return frame


def _noise_frame() -> np.ndarray:
    return RNG.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


def _encode(frames: list[np.ndarray], tmp_path) -> bytes:
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    with open(path, "rb") as fh:
        return fh.read()


def _png(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def good_frames() -> list[np.ndarray]:
    return [_scene_frame(i / FPS) for i in range(N_FRAMES)]


def test_good_clip_passes(tmp_path, good_frames):
    video = _encode(good_frames, tmp_path)
    result = validate_clip_integrity(
        video,
        start_frame_bytes=_png(good_frames[0]),
        end_frame_bytes=_png(good_frames[-1]),
    )
    assert result.ok, result.detail


def test_noise_middle_with_good_conditioning_frames_fails(tmp_path, good_frames):
    """The exact observed corruption: pinned first/last frames, noise between."""
    frames = (
        [good_frames[0]]
        + [_noise_frame() for _ in range(N_FRAMES - 2)]
        + [good_frames[-1]]
    )
    video = _encode(frames, tmp_path)
    result = validate_clip_integrity(
        video,
        start_frame_bytes=_png(good_frames[0]),
        end_frame_bytes=_png(good_frames[-1]),
    )
    assert not result.ok
    assert "noise_frames" in result.detail


def test_all_noise_clip_fails(tmp_path, good_frames):
    frames = [_noise_frame() for _ in range(N_FRAMES)]
    video = _encode(frames, tmp_path)
    result = validate_clip_integrity(
        video, start_frame_bytes=_png(good_frames[0])
    )
    assert not result.ok


def test_wrong_video_fails_start_anchor(tmp_path, good_frames):
    """A coherent clip that is not the one we asked for."""
    # Inverted scene — continuous, but unrelated to the keyframe
    frames = [255 - _scene_frame(0.5 + i / FPS) for i in range(N_FRAMES)]
    video = _encode(frames, tmp_path)
    result = validate_clip_integrity(
        video, start_frame_bytes=_png(good_frames[0])
    )
    assert not result.ok
    assert "start_frame_mismatch" in result.detail


def test_brief_dark_occlusion_passes(tmp_path, good_frames):
    """A near-black moment (object crossing the lens) is not corruption."""
    frames = list(good_frames)
    for i in range(20, 26):
        # Dark, slightly textured frame — like a real occlusion
        dark = (good_frames[i].astype(np.float64) * 0.06).astype(np.uint8)
        frames[i] = dark
    video = _encode(frames, tmp_path)
    result = validate_clip_integrity(
        video,
        start_frame_bytes=_png(good_frames[0]),
        end_frame_bytes=_png(good_frames[-1]),
    )
    assert result.ok, result.detail


def test_good_clip_without_keyframes_passes(tmp_path, good_frames):
    video = _encode(good_frames, tmp_path)
    result = validate_clip_integrity(video)
    assert result.ok, result.detail


def test_undecodable_video_fails():
    result = validate_clip_integrity(b"not a video at all")
    assert not result.ok
    assert "undecodable" in result.detail


def test_moving_clip_reports_more_motion_than_frozen(tmp_path, good_frames):
    """A moving subject reports clearly higher motion than a held frame.

    (Absolute scale is lower for synthetic clips than real footage; the
    real-world clip_min_motion threshold is calibrated separately. What
    matters here is that moving > frozen.)
    """
    moving_dir = tmp_path / "moving"
    frozen_dir = tmp_path / "frozen"
    moving_dir.mkdir()
    frozen_dir.mkdir()
    moving = validate_clip_integrity(_encode(good_frames, moving_dir))
    frozen = validate_clip_integrity(
        _encode([good_frames[0]] * N_FRAMES, frozen_dir)
    )
    assert moving.ok and frozen.ok
    assert moving.motion_mean > frozen.motion_mean


def test_frozen_clip_reports_near_zero_motion_but_still_ok(tmp_path, good_frames):
    """A held single frame is near-zero motion — reported, NOT failed.

    Motion is a quality signal; the integrity gate must still pass so the
    caller (video_gen) decides whether to escalate, not clip_validation.
    """
    frozen = [good_frames[0]] * N_FRAMES
    video = _encode(frozen, tmp_path)
    result = validate_clip_integrity(video)
    assert result.ok, result.detail
    assert result.motion_mean < 1.0, result.motion_mean


def test_motion_metric_orders_clips(tmp_path, good_frames):
    """Faster subject motion yields a strictly higher motion_mean."""
    slow_dir = tmp_path / "slow"
    fast_dir = tmp_path / "fast"
    slow_dir.mkdir()
    fast_dir.mkdir()
    slow = [_scene_frame(i / FPS * 0.2) for i in range(N_FRAMES)]
    fast = [_scene_frame(i / FPS * 3.0) for i in range(N_FRAMES)]
    slow_motion = validate_clip_integrity(_encode(slow, slow_dir)).motion_mean
    fast_motion = validate_clip_integrity(_encode(fast, fast_dir)).motion_mean
    assert fast_motion > slow_motion
