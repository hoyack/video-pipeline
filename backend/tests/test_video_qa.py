"""Adversarial video-QA gate (opt-in).

Audits a finished production and FAILS if any CRITICAL quality criterion is
violated (off-model character identity, audio garble/corruption, etc.).

Run against a live API + a finished production:

    VIDPIPE_QA_PRODUCTION_ID=<id> \
    VIDPIPE_QA_API_BASE=http://localhost:8000 \
    pytest backend/tests/test_video_qa.py -q -s

Skips by default so the normal suite stays offline/fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PID = os.getenv("VIDPIPE_QA_PRODUCTION_ID")

pytestmark = pytest.mark.skipif(
    not PID, reason="Set VIDPIPE_QA_PRODUCTION_ID to audit a finished production.")


@pytest.mark.asyncio
async def test_production_meets_quality_bar(tmp_path):
    from vidpipe.qa.audit import run_audit

    api = os.getenv("VIDPIPE_QA_API_BASE", "http://localhost:8000")
    out = Path(os.getenv("VIDPIPE_QA_OUT", str(tmp_path)))
    verdict = await run_audit(
        api, PID, out,
        sample_every=int(os.getenv("VIDPIPE_QA_SAMPLE_EVERY", "10")),
        do_embed=os.getenv("VIDPIPE_QA_NO_EMBED") != "1",
        do_vision=os.getenv("VIDPIPE_QA_NO_VISION") != "1",
    )
    crit = verdict.critical
    assert not crit, (
        f"{len(crit)} CRITICAL QA finding(s):\n"
        + "\n".join(f"  [{f.code}] {f.where} — {f.message}" for f in crit)
        + f"\nFull report: {out}/report.md"
    )
