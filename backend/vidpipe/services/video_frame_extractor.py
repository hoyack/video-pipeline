"""Video frame extraction with CLIP-based deduplication.

Extracts frames from uploaded videos at an adaptive sampling rate,
deduplicates visually similar frames using CLIP embeddings, and
returns paths to unique frames for manifest asset creation.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract and deduplicate video frames for manifest creation.

    Adaptive sampling rates:
    - <= 30s: 2 fps
    - 30-120s: 1 fps
    - > 120s: 0.5 fps

    CLIP dedup uses greedy forward scan against kept frames.
    """

    def __init__(self, dedup_threshold: float = 0.90):
        self.dedup_threshold = dedup_threshold
        self.progress: dict = {
            "status": "extracting",
            "current_step": "initializing",
            "progress": {
                "candidate_frames": 0,
                "unique_frames": 0,
            },
        }

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata using cv2.

        Args:
            video_path: Path to video file.

        Returns:
            Dict with duration, fps, frame_count, width, height.

        Raises:
            ValueError: If video cannot be opened or exceeds 300s.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")

        duration = frame_count / fps

        if duration > 300:
            raise ValueError(
                f"Video duration {duration:.1f}s exceeds maximum of 300 seconds"
            )

        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }

    def compute_sample_indices(self, duration: float, fps: float) -> list[int]:
        """Compute frame indices to sample at adaptive rate.

        Args:
            duration: Video duration in seconds.
            fps: Video frame rate.

        Returns:
            Sorted list of frame indices to extract.
        """
        if duration <= 30:
            sample_rate = 2.0
        elif duration <= 120:
            sample_rate = 1.0
        else:
            sample_rate = 0.5

        total_frames = int(duration * fps)
        interval = fps / sample_rate  # frames between samples

        indices = []
        pos = 0.0
        while int(pos) < total_frames:
            indices.append(int(pos))
            pos += interval

        return indices

    def extract_candidate_frames(
        self, video_path: str, frame_indices: list[int], output_dir: str
    ) -> list[str]:
        """Extract frames at specified indices using sequential read.

        Reuses the efficient sequential-read pattern from frame_sampler.

        Args:
            video_path: Path to video file.
            frame_indices: Sorted list of frame indices to extract.
            output_dir: Directory to save JPEG frames.

        Returns:
            List of saved frame file paths.
        """
        import cv2

        if not frame_indices:
            return []

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        target_set = set(frame_indices)
        saved_paths = []

        cap = cv2.VideoCapture(video_path)
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index in target_set:
                output_path = str(
                    Path(output_dir) / f"frame_{frame_index:06d}.jpg"
                )
                cv2.imwrite(output_path, frame)
                saved_paths.append(output_path)

                if len(saved_paths) == len(frame_indices):
                    break

            frame_index += 1

        cap.release()
        logger.info(
            f"Extracted {len(saved_paths)}/{len(frame_indices)} candidate frames"
        )
        return saved_paths

    def deduplicate_with_clip(
        self, frame_paths: list[str], threshold: float | None = None
    ) -> list[str]:
        """Deduplicate frames using CLIP embedding similarity.

        Greedy forward scan: keep first frame, then for each subsequent
        frame compute max similarity against all kept frames. Keep if
        below threshold.

        Args:
            frame_paths: List of frame image paths.
            threshold: Similarity threshold (default: self.dedup_threshold).

        Returns:
            List of unique frame paths (discarded files are deleted).
        """
        from vidpipe.services.clip_embedding_service import CLIPEmbeddingService

        if not frame_paths:
            return []

        if threshold is None:
            threshold = self.dedup_threshold

        clip_service = CLIPEmbeddingService()

        # Compute all embeddings
        embeddings: list[np.ndarray] = []
        for i, path in enumerate(frame_paths):
            emb = clip_service.generate_embedding(path)
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                logger.info(f"Computed CLIP embedding {i + 1}/{len(frame_paths)}")

        # Greedy dedup
        kept_indices = [0]
        kept_embeddings = [embeddings[0]]

        for i in range(1, len(embeddings)):
            # Compute max similarity against all kept frames
            max_sim = max(
                CLIPEmbeddingService.compute_similarity(embeddings[i], kept_emb)
                for kept_emb in kept_embeddings
            )

            if max_sim < threshold:
                kept_indices.append(i)
                kept_embeddings.append(embeddings[i])

        # Delete discarded frame files
        kept_set = set(kept_indices)
        for i, path in enumerate(frame_paths):
            if i not in kept_set:
                try:
                    Path(path).unlink()
                except OSError:
                    pass

        kept_paths = [frame_paths[i] for i in kept_indices]
        logger.info(
            f"CLIP dedup: {len(frame_paths)} → {len(kept_paths)} unique frames "
            f"(threshold={threshold})"
        )
        return kept_paths

    def extract_unique_frames(
        self, video_path: str, output_dir: str
    ) -> tuple[list[str], dict]:
        """Orchestrate full extraction pipeline.

        Steps: get_video_info → compute_sample_indices → extract_candidate_frames
        → deduplicate_with_clip.

        Args:
            video_path: Path to source video.
            output_dir: Directory for extracted frames.

        Returns:
            Tuple of (unique_frame_paths, video_info_dict).
        """
        # Step 1: Get video info
        self.progress["current_step"] = "analyzing"
        video_info = self.get_video_info(video_path)
        logger.info(
            f"Video: {video_info['duration']:.1f}s, "
            f"{video_info['fps']:.1f}fps, "
            f"{video_info['width']}x{video_info['height']}"
        )

        # Step 2: Compute sample indices
        self.progress["current_step"] = "sampling"
        indices = self.compute_sample_indices(
            video_info["duration"], video_info["fps"]
        )
        self.progress["progress"]["candidate_frames"] = len(indices)
        logger.info(f"Will sample {len(indices)} frames")

        # Step 3: Extract candidate frames
        frames_dir = str(Path(output_dir) / "video_frames")
        candidate_paths = self.extract_candidate_frames(
            video_path, indices, frames_dir
        )
        self.progress["progress"]["candidate_frames"] = len(candidate_paths)

        # Step 4: CLIP dedup
        self.progress["current_step"] = "deduplicating"
        unique_paths = self.deduplicate_with_clip(candidate_paths)
        self.progress["progress"]["unique_frames"] = len(unique_paths)

        # Done
        self.progress["current_step"] = "saving"
        logger.info(
            f"Extraction complete: {len(unique_paths)} unique frames "
            f"from {video_info['duration']:.1f}s video"
        )

        return unique_paths, video_info
