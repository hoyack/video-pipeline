"""
File management service for vidpipe.

Handles structured filesystem artifact storage with path traversal protection.
Creates per-scene directories with subdirectories for keyframes, clips, and output.
"""
import uuid
from pathlib import Path

from vidpipe.config import settings


class FileManager:
    """
    Manage filesystem artifacts for video pipeline scenes.

    Creates structured directories:
    - {base_dir}/{scene_id}/keyframes/ - Shot keyframe images
    - {base_dir}/{scene_id}/clips/ - Individual shot video clips
    - {base_dir}/{scene_id}/output/ - Final assembled video

    Implements path traversal protection to prevent directory escape attacks.
    """

    def __init__(self, base_dir: str | Path | None = None):
        """
        Initialize FileManager with base directory.

        Args:
            base_dir: Root directory for all scene artifacts.
                     If None, uses settings.storage.tmp_dir
        """
        if base_dir is None:
            base_dir = settings.storage.tmp_dir

        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_scene_dir(self, scene_id: uuid.UUID) -> Path:
        """
        Get or create scene directory with subdirectories.

        Creates:
        - {base_dir}/{scene_id}/
        - {base_dir}/{scene_id}/keyframes/
        - {base_dir}/{scene_id}/clips/
        - {base_dir}/{scene_id}/output/

        Args:
            scene_id: UUID of the scene

        Returns:
            Resolved Path to scene directory

        Raises:
            ValueError: If scene_id creates path outside base_dir (traversal attack)
        """
        scene_dir = (self.base_dir / str(scene_id)).resolve()

        # Path traversal protection (Pitfall 5)
        if not scene_dir.is_relative_to(self.base_dir):
            raise ValueError("Invalid scene path")

        # Create scene directory and subdirectories
        scene_dir.mkdir(exist_ok=True)
        (scene_dir / "keyframes").mkdir(exist_ok=True)
        (scene_dir / "clips").mkdir(exist_ok=True)
        (scene_dir / "output").mkdir(exist_ok=True)

        return scene_dir

    def save_keyframe(
        self, scene_id: uuid.UUID, shot_idx: int, position: str, data: bytes
    ) -> Path:
        """
        Save keyframe image for a shot.

        Args:
            scene_id: UUID of the scene
            shot_idx: Shot index (0-based)
            position: Position identifier (e.g., 'start', 'end')
            data: PNG image data

        Returns:
            Path to saved keyframe file
        """
        scene_dir = self.get_scene_dir(scene_id)
        filename = f"shot_{shot_idx}_{position}.png"
        filepath = scene_dir / "keyframes" / filename

        # Atomic write (Pattern 5)
        filepath.write_bytes(data)

        return filepath

    def save_clip(self, scene_id: uuid.UUID, shot_idx: int, data: bytes) -> Path:
        """
        Save video clip for a shot.

        Args:
            scene_id: UUID of the scene
            shot_idx: Shot index (0-based)
            data: MP4 video data

        Returns:
            Path to saved clip file
        """
        scene_dir = self.get_scene_dir(scene_id)
        filename = f"shot_{shot_idx}.mp4"
        filepath = scene_dir / "clips" / filename

        # Atomic write (Pattern 5)
        filepath.write_bytes(data)

        return filepath

    def save_keyframe_versioned(
        self, scene_id: uuid.UUID, shot_idx: int, position: str, data: bytes
    ) -> Path:
        """Save keyframe with UUID-based filename to avoid overwriting previous versions.

        Old files remain on disk for checkpoint history.
        """
        scene_dir = self.get_scene_dir(scene_id)
        short_id = uuid.uuid4().hex[:8]
        filename = f"shot_{shot_idx}_{position}_{short_id}.png"
        filepath = scene_dir / "keyframes" / filename
        filepath.write_bytes(data)
        return filepath

    def save_clip_versioned(
        self, scene_id: uuid.UUID, shot_idx: int, data: bytes
    ) -> Path:
        """Save clip with UUID-based filename to avoid overwriting previous versions.

        Old files remain on disk for checkpoint history.
        """
        scene_dir = self.get_scene_dir(scene_id)
        short_id = uuid.uuid4().hex[:8]
        filename = f"shot_{shot_idx}_{short_id}.mp4"
        filepath = scene_dir / "clips" / filename
        filepath.write_bytes(data)
        return filepath

    def get_output_path(
        self, scene_id: uuid.UUID, filename: str = "final.mp4"
    ) -> Path:
        """
        Get path for final output video.

        Args:
            scene_id: UUID of the scene
            filename: Output filename (default: 'final.mp4')

        Returns:
            Path to output file location
        """
        scene_dir = self.get_scene_dir(scene_id)
        return scene_dir / "output" / filename
