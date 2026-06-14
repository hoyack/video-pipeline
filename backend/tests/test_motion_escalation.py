"""Tests for motion-prompt escalation (vidpipe.services.comfyui_adapter).

When a generated LTX/ComfyUI clip measures near-static, the pipeline rewrites
the motion prompt to force visible movement and resubmits. escalate_motion_prompt
is the pure rewrite; these tests pin its behavior.
"""

from vidpipe.services.comfyui_adapter import escalate_motion_prompt

# The exact frozen-clip prompt from production 89d98b1b scene 1 shot 0.
FROZEN_PROMPT = (
    "The camera slowly tracks backward in a smooth, steady motion, "
    "maintaining a wide-to-medium framing on the subject as he walks forward "
    "through the oncoming crowd."
)

FREEZE_WORDS = [
    "slowly",
    "smooth, steady",
    "remains static",
    "gently",
    "subtle",
    "motionless",
]


def test_prepends_high_motion_override():
    out = escalate_motion_prompt(FROZEN_PROMPT)
    assert out.startswith("HIGH MOTION:")


def test_removes_freeze_words_from_original_text():
    out = escalate_motion_prompt(FROZEN_PROMPT)
    # Strip the override prefix; only inspect the rewritten original.
    body = out.split("Avoid any static or frozen moment. ", 1)[1]
    assert "slowly" not in body.lower()
    assert "smooth, steady" not in body.lower()


def test_substitutions_are_case_insensitive():
    out = escalate_motion_prompt("The camera Remains Static. Subtle breathing.")
    body = out.split("Avoid any static or frozen moment. ", 1)[1]
    assert "remains static" not in body.lower()
    assert "subtle" not in body.lower()
    assert "moves dynamically" in body
    assert "pronounced" in body


def test_preserves_non_freeze_content():
    out = escalate_motion_prompt(FROZEN_PROMPT)
    assert "crowd" in out
    assert "subject" in out


def test_idempotent_on_already_dynamic_prompt():
    dynamic = "The subject strides briskly through the market as the camera tracks."
    out = escalate_motion_prompt(dynamic)
    assert out.startswith("HIGH MOTION:")
    assert "strides briskly" in out
