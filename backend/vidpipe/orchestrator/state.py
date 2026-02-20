"""State machine constants and transition logic for pipeline orchestrator.

Defines the ordered state machine that governs pipeline execution with
idempotent resume capability from any interrupted or failed state.
"""

from typing import Dict

# Pipeline states in execution order
PIPELINE_STATES = {
    "pending": "Initial state after project creation",
    "storyboarding": "Generating storyboard from prompt",
    "keyframing": "Generating sequential keyframe images",
    "video_gen": "Generating video clips via Veo",
    "stitching": "Concatenating clips into final MP4",
    "complete": "Pipeline finished successfully",
    "failed": "Pipeline encountered unrecoverable error",
    "stopped": "Pipeline stopped by user",
    "staged": "Pipeline paused at user-requested stage boundary",
}

# State transitions for active pipeline steps
STEP_TRANSITIONS = {
    "pending": "storyboarding",
    "storyboarding": "keyframing",
    "keyframing": "video_gen",
    "video_gen": "stitching",
    "stitching": "complete",
}

# States from which pipeline can resume
RESUMABLE_STATES = {
    "pending",
    "failed",
    "stopped",
    "staged",
    "storyboarding",
    "keyframing",
    "video_gen",
    "stitching",
}


def can_resume(status: str) -> bool:
    """Check if pipeline can resume from given status.

    Args:
        status: Current project status

    Returns:
        True if status is resumable, False otherwise
    """
    return status in RESUMABLE_STATES


def get_resume_step(status: str, completed_steps: Dict[str, bool]) -> str:
    """Determine which pipeline step to resume from.

    Uses database state (completed_steps) to find the correct re-entry point,
    especially important for 'failed' status where work may have been partially
    completed.

    Args:
        status: Current project status
        completed_steps: Dict with keys:
            - has_storyboard: project has scenes (count > 0)
            - has_keyframes: all scenes have both start and end keyframes
            - has_clips: all scenes have completed video clips

    Returns:
        Status string representing the step to resume from

    Examples:
        >>> get_resume_step("pending", {"has_storyboard": False, ...})
        'pending'
        >>> get_resume_step("failed", {"has_storyboard": True, "has_keyframes": False, ...})
        'keyframing'
    """
    # If failed/stopped/staged, use completed_steps to determine resume point
    if status in ("failed", "stopped", "staged"):
        if not completed_steps.get("has_storyboard", False):
            return "pending"
        elif not completed_steps.get("has_keyframes", False):
            return "keyframing"
        elif not completed_steps.get("has_clips", False):
            return "video_gen"
        else:
            return "stitching"

    # For non-failed states, resume from current state
    if status in ["pending", "storyboarding"]:
        return "pending"
    elif status == "keyframing":
        return "keyframing"
    elif status == "video_gen":
        return "video_gen"
    elif status == "stitching":
        return "stitching"

    # Should not reach here for resumable states
    return "pending"
