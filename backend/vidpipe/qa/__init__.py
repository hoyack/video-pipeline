"""Adversarial video-QA harness.

Audits a finished production against quality criteria the generation pipeline
itself does not hard-gate: character identity alignment to the Production Bible,
story/scene alignment, and audio integrity (corruption, garble, clipping,
dropouts). Produces a scored PASS/FAIL report and is wired into a pytest gate.

Run:
    python -m vidpipe.qa.audit <production_id> [--out DIR] [--no-embeddings]

The vision judge uses the production's *selected* vision LLM (Gemini or Ollama)
via ``vidpipe.services.llm.registry.get_adapter`` — it is not hard-coded to any
provider. A face-embedding (InsightFace) check runs alongside it when available.
"""

from vidpipe.qa.criteria import Finding, Severity, Verdict  # noqa: F401
