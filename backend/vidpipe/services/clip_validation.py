"""Degenerate-clip detection for generated video clips.

ComfyUI Cloud occasionally returns a structurally valid MP4 whose decoded
frames are garbage (observed: WAN 2.2 FLF2V jobs where the server-side QC
frames were clean but the encoded video contained a static noise pattern for
every frame between the two pinned conditioning frames). The job reports
success, so the only way to catch it is to inspect the downloaded video.

Detection strategy (validated against real corrupted + good clips):

1. Per-frame noise signature — the ratio of mid-frequency to low-frequency
   band energy. Natural frames follow a ~1/f spectrum (measured <= 0.30,
   including near-black occlusion frames); the observed corruption is
   mid-band texture with no large-scale structure (measured >= 3.8).
2. Start/end keyframe anchors — the first frame must correlate with the start
   keyframe (catches wrong-video downloads); the last frame must correlate
   with the end keyframe when one conditioned the generation (FLF2V).

Temporal continuity between adjacent samples is computed for diagnostics but
is NOT a hard gate: legitimate clips can contain near-discontinuities (e.g.
an object sweeping across the lens was observed in a good LTX clip).

All comparisons use Pearson correlation of downscaled grayscale frames.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Downscale target for frame-vs-frame / frame-vs-keyframe comparison.
_COMPARE_SIZE = (208, 120)

# Seconds between adjacent samples in the continuity chain.
_HOP_SECONDS = 0.25

# Thresholds — all lenient relative to measured margins.
_START_ANCHOR_THRESHOLD = 0.45  # good >= 0.92
_END_ANCHOR_THRESHOLD = 0.35    # good >= 0.89
_NOISE_BAND_RATIO = 1.0         # natural <= 0.30; noise >= 3.8
_NOISE_BAND_MIN_ENERGY = 3.0    # ignore ratio on near-uniform frames
_FLAT_FRAME_STD = 15.0          # occlusion frame low-band std 11.5; good >= 38


@dataclass(frozen=True)
class ClipValidationResult:
    ok: bool
    detail: str
    # Mean absolute frame-to-frame difference across sampled frames (grayscale,
    # downscaled). A quality signal — NOT a corruption gate. Calibrated values:
    # frozen/near-static clips measure < 5; lively clips measure 7-37.
    # 0.0 when fewer than two frames decoded.
    motion_mean: float = 0.0


@dataclass(frozen=True)
class _SampledFrame:
    compare: np.ndarray   # grayscale at _COMPARE_SIZE
    band_mid: float       # band energy between 1/4 and 1/16 scale
    band_low: float       # large-scale structure (1/16 scale std)


def _gray_from_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    return np.asarray(img.convert("L").resize(_COMPARE_SIZE), dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _analyze_frame(gray_full: Image.Image) -> _SampledFrame:
    """Compute comparison image + frequency band energies for one frame."""
    w, h = gray_full.size
    d4 = gray_full.resize((max(1, w // 4), max(1, h // 4)))
    d16 = gray_full.resize((max(1, w // 16), max(1, h // 16)))
    a4 = np.asarray(d4, dtype=np.float64)
    u4 = np.asarray(d16.resize(d4.size), dtype=np.float64)
    band_mid = float((a4 - u4).std())
    band_low = float(np.asarray(d16, dtype=np.float64).std())
    compare = np.asarray(gray_full.resize(_COMPARE_SIZE), dtype=np.float64)
    return _SampledFrame(compare=compare, band_mid=band_mid, band_low=band_low)


def _is_noise(frame: _SampledFrame) -> bool:
    if frame.band_mid < _NOISE_BAND_MIN_ENERGY:
        return False
    return frame.band_mid / max(frame.band_low, 0.1) > _NOISE_BAND_RATIO


def _compute_motion_mean(frames: list[_SampledFrame]) -> float:
    """Mean absolute frame-to-frame difference across sampled frames.

    Reuses the already-decoded ``compare`` grayscale arrays (zero extra
    decode cost). Pairs where either frame is genuinely flat (e.g. a dark
    occlusion frame) are skipped, mirroring the continuity diagnostic, so a
    brief blackout does not inflate or deflate the score.
    """
    diffs: list[float] = []
    for i in range(len(frames) - 1):
        if _is_flat(frames[i]) or _is_flat(frames[i + 1]):
            continue
        diffs.append(float(np.abs(frames[i + 1].compare - frames[i].compare).mean()))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _is_flat(frame: _SampledFrame) -> bool:
    return frame.band_low < _FLAT_FRAME_STD


def _sample_frames(video_path: str, hop_seconds: float) -> list[_SampledFrame]:
    """Decode the clip and analyze frames every hop.

    Always includes the first and last decodable frame.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
        hop_frames = max(1, int(round(fps * hop_seconds)))

        samples: list[_SampledFrame] = []
        last_frame: _SampledFrame | None = None
        index = 0
        last_sampled_index = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            )
            analyzed = _analyze_frame(gray)
            last_frame = analyzed
            if index % hop_frames == 0:
                samples.append(analyzed)
                last_sampled_index = index
            index += 1

        # Ensure the true final frame participates in the checks
        if last_frame is not None and last_sampled_index != index - 1:
            samples.append(last_frame)
        return samples
    finally:
        cap.release()


def validate_clip_integrity(
    video_bytes: bytes,
    *,
    start_frame_bytes: bytes | None = None,
    end_frame_bytes: bytes | None = None,
) -> ClipValidationResult:
    """Check a downloaded clip for degenerate/corrupted content.

    Args:
        video_bytes: The full video file as bytes.
        start_frame_bytes: Start keyframe that conditioned generation, if any.
        end_frame_bytes: End keyframe that conditioned generation (FLF2V), if any.

    Returns:
        ClipValidationResult — ``ok=False`` means the clip should not be
        accepted (corrupted, or not the video we asked for).
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        frames = _sample_frames(tmp_path, _HOP_SECONDS)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if len(frames) < 2:
        return ClipValidationResult(
            ok=False, detail=f"undecodable_video: only {len(frames)} frame(s) decoded"
        )

    # 1. Per-frame noise signature
    noise_count = sum(1 for f in frames if _is_noise(f))
    if noise_count > 0:
        return ClipValidationResult(
            ok=False,
            detail=(
                f"noise_frames: {noise_count} of {len(frames)} sampled frames "
                "have a mid-frequency noise spectrum (no large-scale structure)"
            ),
        )

    motion_mean = _compute_motion_mean(frames)

    # Diagnostic only — legitimate clips can contain near-discontinuities
    min_hop = 1.0
    for i in range(len(frames) - 1):
        if _is_flat(frames[i]) or _is_flat(frames[i + 1]):
            continue
        min_hop = min(min_hop, _pearson(frames[i].compare, frames[i + 1].compare))

    checks = [f"continuity_min={min_hop:.3f}", f"motion={motion_mean:.2f}"]

    # 2. Start-frame anchor
    if start_frame_bytes is not None and not _is_flat(frames[0]):
        corr = _pearson(frames[0].compare, _gray_from_bytes(start_frame_bytes))
        if corr < _START_ANCHOR_THRESHOLD:
            return ClipValidationResult(
                ok=False,
                detail=(
                    f"start_frame_mismatch: correlation {corr:.3f} < "
                    f"{_START_ANCHOR_THRESHOLD} — clip does not begin at the "
                    "start keyframe"
                ),
            )
        checks.append(f"start_anchor={corr:.3f}")

    # 3. End-frame anchor (only when an end keyframe conditioned generation)
    if end_frame_bytes is not None and not _is_flat(frames[-1]):
        corr = _pearson(frames[-1].compare, _gray_from_bytes(end_frame_bytes))
        if corr < _END_ANCHOR_THRESHOLD:
            return ClipValidationResult(
                ok=False,
                detail=(
                    f"end_frame_mismatch: correlation {corr:.3f} < "
                    f"{_END_ANCHOR_THRESHOLD} — clip does not end at the "
                    "end keyframe"
                ),
            )
        checks.append(f"end_anchor={corr:.3f}")

    return ClipValidationResult(
        ok=True, detail=" ".join(checks), motion_mean=motion_mean
    )
