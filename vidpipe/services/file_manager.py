"""
File management service for vidpipe.

Handles structured filesystem artifact storage with path traversal protection.
Creates per-project directories with subdirectories for keyframes, clips, and output.
"""
import uuid
from pathlib import Path

from vidpipe.config import settings


class FileManager:
    """
    Manage filesystem artifacts for video pipeline projects.

    Creates structured directories:
    - {base_dir}/{project_id}/keyframes/ - Scene keyframe images
    - {base_dir}/{project_id}/clips/ - Individual scene video clips
    - {base_dir}/{project_id}/output/ - Final assembled video

    Implements path traversal protection to prevent directory escape attacks.
    """

    def __init__(self, base_dir: str | Path | None = None):
        """
        Initialize FileManager with base directory.

        Args:
            base_dir: Root directory for all project artifacts.
                     If None, uses settings.storage.tmp_dir
        """
        if base_dir is None:
            base_dir = settings.storage.tmp_dir

        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_project_dir(self, project_id: uuid.UUID) -> Path:
        """
        Get or create project directory with subdirectories.

        Creates:
        - {base_dir}/{project_id}/
        - {base_dir}/{project_id}/keyframes/
        - {base_dir}/{project_id}/clips/
        - {base_dir}/{project_id}/output/

        Args:
            project_id: UUID of the project

        Returns:
            Resolved Path to project directory

        Raises:
            ValueError: If project_id creates path outside base_dir (traversal attack)
        """
        project_dir = (self.base_dir / str(project_id)).resolve()

        # Path traversal protection (Pitfall 5)
        if not project_dir.is_relative_to(self.base_dir):
            raise ValueError("Invalid project path")

        # Create project directory and subdirectories
        project_dir.mkdir(exist_ok=True)
        (project_dir / "keyframes").mkdir(exist_ok=True)
        (project_dir / "clips").mkdir(exist_ok=True)
        (project_dir / "output").mkdir(exist_ok=True)

        return project_dir

    def save_keyframe(
        self, project_id: uuid.UUID, scene_idx: int, position: str, data: bytes
    ) -> Path:
        """
        Save keyframe image for a scene.

        Args:
            project_id: UUID of the project
            scene_idx: Scene index (0-based)
            position: Position identifier (e.g., 'start', 'end')
            data: PNG image data

        Returns:
            Path to saved keyframe file
        """
        project_dir = self.get_project_dir(project_id)
        filename = f"scene_{scene_idx}_{position}.png"
        filepath = project_dir / "keyframes" / filename

        # Atomic write (Pattern 5)
        filepath.write_bytes(data)

        return filepath

    def save_clip(self, project_id: uuid.UUID, scene_idx: int, data: bytes) -> Path:
        """
        Save video clip for a scene.

        Args:
            project_id: UUID of the project
            scene_idx: Scene index (0-based)
            data: MP4 video data

        Returns:
            Path to saved clip file
        """
        project_dir = self.get_project_dir(project_id)
        filename = f"scene_{scene_idx}.mp4"
        filepath = project_dir / "clips" / filename

        # Atomic write (Pattern 5)
        filepath.write_bytes(data)

        return filepath

    def get_output_path(
        self, project_id: uuid.UUID, filename: str = "final.mp4"
    ) -> Path:
        """
        Get path for final output video.

        Args:
            project_id: UUID of the project
            filename: Output filename (default: 'final.mp4')

        Returns:
            Path to output file location
        """
        project_dir = self.get_project_dir(project_id)
        return project_dir / "output" / filename
