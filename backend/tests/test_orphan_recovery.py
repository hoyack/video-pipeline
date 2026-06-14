"""Tests for restart-orphan recovery: resumability of in-flight statuses.

A server restart mid-pipeline (e.g. container rebuild) used to strand scenes
in "generating_video" — not resumable, not stoppable, invisible to recovery.
"""

from vidpipe.orchestrator.recovery import IN_FLIGHT_STATUSES
from vidpipe.orchestrator.state import RESUMABLE_STATES, can_resume, get_resume_step


def test_generating_video_is_resumable():
    assert can_resume("generating_video")


def test_generating_video_resumes_at_video_gen():
    completed = {"has_storyboard": True, "has_keyframes": True, "has_clips": False}
    assert get_resume_step("generating_video", completed) == "video_gen"


def test_all_in_flight_statuses_are_resumable():
    """Every status startup recovery targets must be accepted by resume."""
    for status in IN_FLIGHT_STATUSES:
        assert status in RESUMABLE_STATES, (
            f"startup recovery would resume '{status}' but it is not resumable"
        )


def test_terminal_statuses_are_not_recovered():
    for status in ("complete", "failed", "stopped", "draft", "staged"):
        assert status not in IN_FLIGHT_STATUSES
