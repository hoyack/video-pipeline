"""FFmpeg-based audio-integrity analysis.

Detects the failure modes the pipeline does not gate: garbled/broadband-noise
audio (e.g. LTX native-audio babble), digital clipping, mid-shot dropouts, and
DC offset. Spectral flatness is the primary garble signal — tonal speech/music
sits low (~0.1-0.3); white-noise garble pushes toward 1.0.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from vidpipe.qa import criteria as C
from vidpipe.qa.criteria import Finding, Severity


@dataclass
class AudioMetrics:
    label: str
    has_audio: bool
    duration: float = 0.0
    mean_db: Optional[float] = None
    max_db: Optional[float] = None
    peak_db: Optional[float] = None
    flat_factor: Optional[float] = None
    dc_offset: Optional[float] = None
    spectral_flatness: Optional[float] = None      # mean over the whole segment, 0..1
    flatness_median: Optional[float] = None        # median of 2s-window means
    garble_windows: list[dict] = None              # outlier windows: {start,end,mean,max}
    mid_dropout_s: float = 0.0                       # longest silence gap not at the edges

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return (p.stderr or "") + (p.stdout or "")


def _f(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def probe_has_audio(path: str) -> bool:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0", path])
    return "audio" in out


def _flatness_windows(path: str, win: float = 2.0) -> tuple[Optional[float], Optional[float], list[dict]]:
    """Per-window spectral flatness via aspectralstats metadata.

    Returns (overall_mean, median_of_window_means, garble_windows). Garble is
    detected as a window that is an OUTLIER vs the track's own median (robust to
    naturally noisy beds), with an absolute floor — tonal speech sits ~0.05-0.15,
    broadband-noise garble spikes toward 1.0.
    """
    out = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-af", "aspectralstats=measure=flatness,ametadata=print:file=-",
                "-f", "null", "-"])
    t = None
    buckets: dict[int, list[float]] = {}
    allvals: list[float] = []
    for line in out.splitlines():
        mt = re.search(r"pts_time:([0-9.]+)", line)
        if mt:
            t = float(mt.group(1))
        mf = re.search(r"flatness=([0-9.eE+-]+)", line)
        if mf and t is not None:
            v = float(mf.group(1))
            allvals.append(v)
            buckets.setdefault(int(t // win * win), []).append(v)
    if not allvals:
        return None, None, []
    overall = sum(allvals) / len(allvals)
    win_means = {s: sum(v) / len(v) for s, v in buckets.items()}
    ordered = sorted(win_means.values())
    median = ordered[len(ordered) // 2]
    garble = []
    for s in sorted(buckets):
        wm = win_means[s]
        wmax = max(buckets[s])
        # Outlier vs own median AND absolute floor AND a genuinely noisy peak.
        if wm >= max(C.AUDIO_SPECTRAL_FLATNESS_MAX * 0.4, 2.5 * median) and wmax >= 0.5:
            garble.append({"start": float(s), "end": float(s + win),
                           "mean": round(wm, 3), "max": round(wmax, 3)})
    return overall, median, garble


def analyze(path: str, label: str) -> AudioMetrics:
    if not probe_has_audio(path):
        return AudioMetrics(label=label, has_audio=False)
    dur = _f(r"Duration: (\d+):(\d+):", "") or 0.0
    durtxt = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "csv=p=0", path])
    try:
        dur = float(durtxt.strip().split("\n")[0])
    except Exception:
        dur = 0.0
    vol = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-af", "volumedetect", "-f", "null", "-"])
    ast = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-af", "astats=metadata=1", "-f", "null", "-"])
    sil = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-af", "silencedetect=noise=-50dB:d=0.4", "-f", "null", "-"])

    overall_flat, median_flat, garble = _flatness_windows(path)
    m = AudioMetrics(
        label=label, has_audio=True, duration=dur,
        mean_db=_f(r"mean_volume:\s*(-?[0-9.]+) dB", vol),
        max_db=_f(r"max_volume:\s*(-?[0-9.]+) dB", vol),
        peak_db=_f(r"Peak level dB:\s*(-?[0-9.inf]+)", ast),
        flat_factor=_f(r"Flat factor:\s*([0-9.]+)", ast),
        dc_offset=_f(r"DC offset:\s*(-?[0-9.]+)", ast),
        spectral_flatness=overall_flat,
        flatness_median=median_flat,
        garble_windows=garble,
    )

    # Longest mid-clip silence gap (dropouts), excluding leading/trailing silence.
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[0-9.]+)", sil)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", sil)]
    longest = 0.0
    for s, e in zip(starts, ends):
        if s > 0.3 and e < dur - 0.3:
            longest = max(longest, e - s)
    m.mid_dropout_s = longest
    return m


def findings(m: AudioMetrics, *, expect_audio: bool) -> list[Finding]:
    out: list[Finding] = []
    if not m.has_audio:
        if expect_audio:
            out.append(Finding("audio.missing", Severity.CRITICAL, m.label,
                               "Audio expected (audio_enabled) but no audio stream present."))
        return out
    for w in (m.garble_windows or []):
        out.append(Finding(
            "audio.garble", Severity.CRITICAL, f"{m.label} {w['start']:.0f}-{w['end']:.0f}s",
            f"Broadband-noise/garble burst: window spectral flatness {w['mean']:.2f} "
            f"(peak {w['max']:.2f}) vs track median {m.flatness_median:.2f} — "
            f"tonal speech/music sits ~0.05-0.15.",
            {"window": w, "track_median": m.flatness_median}))
    if m.peak_db is not None and m.peak_db > C.AUDIO_TRUE_PEAK_MAX_DB:
        out.append(Finding("audio.clipping", Severity.WARNING, m.label,
                           f"Peak level {m.peak_db:.2f} dB risks clipping.",
                           {"peak_db": m.peak_db}))
    if m.mid_dropout_s > C.AUDIO_MID_DROPOUT_MAX_S:
        out.append(Finding("audio.dropout", Severity.WARNING, m.label,
                           f"Mid-shot silence gap {m.mid_dropout_s:.2f}s.",
                           {"mid_dropout_s": m.mid_dropout_s}))
    if m.dc_offset is not None and abs(m.dc_offset) > C.AUDIO_DC_OFFSET_MAX:
        out.append(Finding("audio.dc_offset", Severity.WARNING, m.label,
                           f"DC offset {m.dc_offset:.3f}.", {"dc_offset": m.dc_offset}))
    return out


def render_spectrogram(path: str, out_png: str) -> bool:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", path,
         "-lavfi", "showspectrumpic=s=1200x400:legend=1", out_png],
        capture_output=True, text=True)
    return res.returncode == 0
