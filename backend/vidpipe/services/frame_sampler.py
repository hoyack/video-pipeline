"""Video frame extraction and motion delta detection.

Provides strategic frame sampling for CV analysis: base frames at fixed
intervals plus additional motion-delta frames where significant pixel change
is detected between consecutive frames.

Uses opencv-python for video I/O. cv2 is imported inside functions to avoid
import failures if opencv is not installed.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sample_video_frames(
    clip_path: str,
    duration: int = 8,
    fps: int = 24,
    motion_threshold: float = 0.15,
    max_frames: int = 8,
) -> list[int]:
    """Sample 5-8 key frames from a video clip.

    Computes base frame indices at fixed intervals (first, 2s, 4s, 6s, last)
    plus additional frames detected via motion delta analysis.

    Args:
        clip_path: Path to video file.
        duration: Expected clip duration in seconds (used for base frame positions).
        fps: Expected frames per second (used for base frame positions).
        motion_threshold: Ratio of changed pixels threshold for motion detection.
        max_frames: Maximum number of frames to return.

    Returns:
        Sorted, deduplicated list of frame indices, capped at max_frames.
    """
    import cv2

    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()

    if total_frames <= 0:
        logger.warning(f"Could not read frame count from {clip_path}, using defaults")
        total_frames = duration * fps

    # Base frames: first, 2s, 4s, 6s, last
    base_indices = [
        0,
        int(actual_fps * 2),
        int(actual_fps * 4),
        int(actual_fps * 6),
        total_frames - 1,
    ]
    # Clamp to valid range
    base_indices = [max(0, min(i, total_frames - 1)) for i in base_indices]

    # Get additional frames from motion detection
    motion_indices = detect_motion_deltas(clip_path, threshold=motion_threshold)
    logger.info(
        f"Frame sampling: {len(base_indices)} base frames + "
        f"{len(motion_indices)} motion delta frames from {clip_path}"
    )

    # Combine, deduplicate, sort, cap
    combined = sorted(set(base_indices + motion_indices))
    return combined[:max_frames]


def detect_motion_deltas(clip_path: str, threshold: float = 0.15) -> list[int]:
    """Detect frames with significant motion relative to the previous frame.

    Converts frames to grayscale, computes absolute difference between consecutive
    frames, thresholds at 30 pixel difference per channel, and counts the ratio of
    changed pixels. Frames where this ratio exceeds threshold are returned.

    Args:
        clip_path: Path to video file.
        threshold: Ratio of changed pixels to trigger inclusion (0.0-1.0).

    Returns:
        List of frame indices with significant motion.
    """
    import cv2

    cap = cv2.VideoCapture(clip_path)
    motion_frames = []
    prev_gray = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            # Count pixels with change > 30 intensity units
            changed = (diff > 30).sum()
            ratio = changed / diff.size
            if ratio > threshold:
                motion_frames.append(frame_index)

        prev_gray = gray
        frame_index += 1

    cap.release()
    logger.info(
        f"Motion delta detection: {len(motion_frames)} frames exceeded "
        f"threshold={threshold} in {clip_path}"
    )
    return motion_frames


def extract_frame(clip_path: str, frame_index: int) -> str:
    """Extract a single frame from a video and save as JPEG.

    Args:
        clip_path: Path to video file.
        frame_index: Zero-based frame index to extract.

    Returns:
        Path to the saved JPEG file (tmp/cv_analysis/frame_{frame_index}.jpg).
    """
    import cv2

    output_path = f"tmp/cv_analysis/frame_{frame_index}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(
            f"Could not read frame {frame_index} from {clip_path}"
        )

    cv2.imwrite(output_path, frame)
    return output_path


def extract_frames(
    clip_path: str, frame_indices: list[int], output_dir: str
) -> list[str]:
    """Batch extract multiple frames from a video efficiently.

    Reads frames sequentially, saving only frames at target indices.
    More efficient than calling extract_frame() repeatedly for each index.

    Args:
        clip_path: Path to video file.
        frame_indices: Sorted list of zero-based frame indices to extract.
        output_dir: Directory to save extracted JPEG frames.

    Returns:
        List of file paths for successfully saved frames.
    """
    import cv2

    if not frame_indices:
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    target_set = set(frame_indices)
    saved_paths = []

    cap = cv2.VideoCapture(clip_path)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index in target_set:
            output_path = str(Path(output_dir) / f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_paths.append(output_path)

            # Early exit if we've collected all target frames
            if len(saved_paths) == len(frame_indices):
                break

        frame_index += 1

    cap.release()
    logger.info(
        f"Extracted {len(saved_paths)}/{len(frame_indices)} frames to {output_dir}"
    )
    return saved_paths
