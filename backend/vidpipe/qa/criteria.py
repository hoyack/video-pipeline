"""Quality criteria, severities, and the finding/verdict data model.

Thresholds live here so the standard is explicit and tunable in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"      # below ideal, does not by itself fail the audit
    CRITICAL = "critical"    # fails the audit → triggers re-generation


# ---- Identity / vision thresholds ----------------------------------------
# Vision-LLM identity_match is 0..1. Face-embedding cosine similarity is 0..1.
IDENTITY_VISION_MIN = 0.70          # establishing keyframe must clearly be the character
IDENTITY_VISION_WARN = 0.85         # below this on any sampled frame → warning
FACE_EMBED_MIN = 0.35               # InsightFace cosine; stylized art runs low, so advisory-ish
# A scene's *establishing* keyframe (shot 0 start) is the strictest gate.

# ---- Audio thresholds ----------------------------------------------------
AUDIO_TRUE_PEAK_MAX_DB = -0.1       # above this → clipping risk
AUDIO_SPECTRAL_FLATNESS_MAX = 0.45  # broadband-noise/garble heuristic (0=tonal,1=white noise)
AUDIO_MID_DROPOUT_MAX_S = 0.6       # silence gap inside a shot longer than this → dropout
AUDIO_DC_OFFSET_MAX = 0.02


@dataclass
class Finding:
    code: str                       # e.g. "identity.establishing_miss"
    severity: Severity
    where: str                      # e.g. "S1 shot 0 start keyframe" / "master 52-58s"
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "where": self.where,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class Verdict:
    production_id: str
    findings: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def passed(self) -> bool:
        return len(self.critical) == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "production_id": self.production_id,
            "passed": self.passed,
            "n_critical": len(self.critical),
            "n_warning": len(self.warnings),
            "findings": [f.as_dict() for f in self.findings],
            "meta": self.meta,
        }
