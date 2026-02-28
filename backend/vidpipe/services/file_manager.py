"""
File management service for vidpipe.

Handles structured artifact storage with path traversal protection.
Delegates to StorageBackend (local filesystem or S3) for actual I/O.

Keys follow the pattern:
- {scene_id}/keyframes/shot_{idx}_{position}.png
- {scene_id}/clips/shot_{idx}.mp4
- {scene_id}/output/final.mp4
- manifests/{manifest_id}/uploads/{asset_id}_{filename}
"""

import uuid
from pathlib import Path

from vidpipe.services.storage_backend import StorageBackend, LocalStorageBackend, get_storage_backend


class FileManager:
    """
    Manage artifacts for video pipeline scenes.

    Creates structured key namespaces:
    - {scene_id}/keyframes/ - Shot keyframe images
    - {scene_id}/clips/ - Individual shot video clips
    - {scene_id}/output/ - Final assembled video

    For local backend, also creates directories on disk.
    """

    def __init__(self, base_dir: str | Path | None = None, backend: StorageBackend | None = None):
        """
        Initialize FileManager.

        Args:
            base_dir: Legacy parameter for local storage root. If None and backend
                     is local, uses settings.storage.tmp_dir.
            backend: StorageBackend to use. If None, uses the singleton from config.
        """
        self.backend = backend or get_storage_backend()

        # For local backend, ensure base_dir is set for backwards compatibility
        if isinstance(self.backend, LocalStorageBackend):
            if base_dir is not None:
                self.base_dir = Path(base_dir).resolve()
            else:
                self.base_dir = self.backend.base_dir
        else:
            # S3 backend — base_dir is only used for temp files (stitcher)
            from vidpipe.config import settings
            self.base_dir = Path(base_dir).resolve() if base_dir else settings.storage.tmp_dir.resolve()

    def _ensure_local_dirs(self, scene_id: uuid.UUID) -> None:
        """Create local directories for scene (local backend only)."""
        if isinstance(self.backend, LocalStorageBackend):
            scene_dir = self.base_dir / str(scene_id)
            scene_dir.mkdir(exist_ok=True)
            (scene_dir / "keyframes").mkdir(exist_ok=True)
            (scene_dir / "clips").mkdir(exist_ok=True)
            (scene_dir / "output").mkdir(exist_ok=True)

    def get_scene_dir(self, scene_id: uuid.UUID) -> Path:
        """
        Get or create scene directory (local backend compatibility).

        For S3 backend, returns a local temp directory for intermediate files.
        """
        scene_dir = (self.base_dir / str(scene_id)).resolve()

        # Path traversal protection
        if not scene_dir.is_relative_to(self.base_dir):
            raise ValueError("Invalid scene path")

        # Create directories
        scene_dir.mkdir(exist_ok=True)
        (scene_dir / "keyframes").mkdir(exist_ok=True)
        (scene_dir / "clips").mkdir(exist_ok=True)
        (scene_dir / "output").mkdir(exist_ok=True)

        return scene_dir

    async def save_keyframe_async(
        self, scene_id: uuid.UUID, shot_idx: int, position: str, data: bytes
    ) -> str:
        """Save keyframe image and return storage key.

        For S3 backend, also writes a local copy so pipeline stages (CV, ffmpeg)
        can access the file without downloading from S3.
        """
        key = f"{scene_id}/keyframes/shot_{shot_idx}_{position}.png"
        if not isinstance(self.backend, LocalStorageBackend):
            # Keep local copy for pipeline reads
            local_path = self.base_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
        await self.backend.put(key, data, "image/png")
        return key

    def save_keyframe(
        self, scene_id: uuid.UUID, shot_idx: int, position: str, data: bytes
    ) -> Path:
        """
        Save keyframe image (sync, local backend only — legacy interface).

        Returns Path to saved file for backwards compatibility with pipeline code
        that stores str(path) in the database.
        """
        self._ensure_local_dirs(scene_id)
        key = f"{scene_id}/keyframes/shot_{shot_idx}_{position}.png"

        if isinstance(self.backend, LocalStorageBackend):
            filepath = self.backend.base_dir / key
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(data)
            return filepath
        else:
            # For S3, write locally as temp AND schedule async upload
            # But sync callers can't await — this path shouldn't be used in S3 mode
            # Pipeline code should use save_keyframe_async instead
            raise RuntimeError(
                "Sync save_keyframe not supported with S3 backend. "
                "Use save_keyframe_async instead."
            )

    async def save_clip_async(
        self, scene_id: uuid.UUID, shot_idx: int, data: bytes
    ) -> str:
        """Save video clip and return storage key.

        For S3 backend, also writes a local copy for ffmpeg/CV access.
        """
        key = f"{scene_id}/clips/shot_{shot_idx}.mp4"
        if not isinstance(self.backend, LocalStorageBackend):
            local_path = self.base_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
        await self.backend.put(key, data, "video/mp4")
        return key

    def save_clip(self, scene_id: uuid.UUID, shot_idx: int, data: bytes) -> Path:
        """Save video clip (sync, local backend only — legacy interface)."""
        self._ensure_local_dirs(scene_id)
        key = f"{scene_id}/clips/shot_{shot_idx}.mp4"

        if isinstance(self.backend, LocalStorageBackend):
            filepath = self.backend.base_dir / key
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(data)
            return filepath
        else:
            raise RuntimeError(
                "Sync save_clip not supported with S3 backend. "
                "Use save_clip_async instead."
            )

    async def save_keyframe_versioned_async(
        self, scene_id: uuid.UUID, shot_idx: int, position: str, data: bytes
    ) -> str:
        """Save keyframe with UUID suffix to avoid overwrites. Returns key."""
        short_id = uuid.uuid4().hex[:8]
        key = f"{scene_id}/keyframes/shot_{shot_idx}_{position}_{short_id}.png"
        if not isinstance(self.backend, LocalStorageBackend):
            local_path = self.base_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
        await self.backend.put(key, data, "image/png")
        return key

    def save_keyframe_versioned(
        self, scene_id: uuid.UUID, shot_idx: int, position: str, data: bytes
    ) -> Path:
        """Save keyframe with UUID suffix (sync, local backend only)."""
        self._ensure_local_dirs(scene_id)
        short_id = uuid.uuid4().hex[:8]
        key = f"{scene_id}/keyframes/shot_{shot_idx}_{position}_{short_id}.png"

        if isinstance(self.backend, LocalStorageBackend):
            filepath = self.backend.base_dir / key
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(data)
            return filepath
        else:
            raise RuntimeError(
                "Sync save_keyframe_versioned not supported with S3 backend."
            )

    async def save_clip_versioned_async(
        self, scene_id: uuid.UUID, shot_idx: int, data: bytes
    ) -> str:
        """Save clip with UUID suffix to avoid overwrites. Returns key."""
        short_id = uuid.uuid4().hex[:8]
        key = f"{scene_id}/clips/shot_{shot_idx}_{short_id}.mp4"
        if not isinstance(self.backend, LocalStorageBackend):
            local_path = self.base_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
        await self.backend.put(key, data, "video/mp4")
        return key

    def save_clip_versioned(
        self, scene_id: uuid.UUID, shot_idx: int, data: bytes
    ) -> Path:
        """Save clip with UUID suffix (sync, local backend only)."""
        self._ensure_local_dirs(scene_id)
        short_id = uuid.uuid4().hex[:8]
        key = f"{scene_id}/clips/shot_{shot_idx}_{short_id}.mp4"

        if isinstance(self.backend, LocalStorageBackend):
            filepath = self.backend.base_dir / key
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(data)
            return filepath
        else:
            raise RuntimeError(
                "Sync save_clip_versioned not supported with S3 backend."
            )

    def get_output_path(
        self, scene_id: uuid.UUID, filename: str = "final.mp4"
    ) -> Path:
        """
        Get local path for final output video.

        Always returns a local path — the stitcher needs local files for ffmpeg.
        For S3 backend, caller should upload the result after stitching.
        """
        scene_dir = self.get_scene_dir(scene_id)
        return scene_dir / "output" / filename

    async def save_output_async(
        self, scene_id: uuid.UUID, filename: str = "final.mp4"
    ) -> str:
        """Upload the locally-stitched output to storage. Returns key."""
        local_path = self.get_output_path(scene_id, filename)
        if not local_path.exists():
            raise FileNotFoundError(f"Output file not found: {local_path}")

        key = f"{scene_id}/output/{filename}"
        data = local_path.read_bytes()
        await self.backend.put(key, data, "video/mp4")
        return key

    async def read_bytes(self, path_or_key: str) -> bytes:
        """Read bytes from storage — handles both absolute paths (legacy) and keys.

        For local backend with absolute paths, reads directly from disk.
        For S3 or relative keys, reads via the backend.
        """
        # Absolute path — legacy local file
        if path_or_key.startswith("/"):
            p = Path(path_or_key)
            if p.exists():
                return p.read_bytes()
            # Might be a relocated file — try stripping to a key
            # e.g. /app/tmp/scene_id/keyframes/shot_0_start.png -> scene_id/keyframes/...
            if isinstance(self.backend, LocalStorageBackend):
                raise FileNotFoundError(f"Local file not found: {path_or_key}")
            # For S3, try to extract a relative key
            for prefix in ("/app/tmp/", str(self.base_dir) + "/"):
                if path_or_key.startswith(prefix):
                    key = path_or_key[len(prefix):]
                    return await self.backend.get(key)
            raise FileNotFoundError(f"Cannot resolve path to S3 key: {path_or_key}")

        # Relative key — read from backend
        return await self.backend.get(path_or_key)

    async def get_url(self, path_or_key: str) -> str:
        """Get a serveable URL for a stored asset.

        For local backend: returns the FileResponse path.
        For S3 backend: returns the public Supabase Storage URL.
        """
        if isinstance(self.backend, LocalStorageBackend):
            if path_or_key.startswith("/"):
                # Absolute local path — return as-is for FileResponse
                return path_or_key
            return str(self.backend.base_dir / path_or_key)
        else:
            # S3 — strip any absolute prefix to get the key
            key = path_or_key
            for prefix in ("/app/tmp/", str(self.base_dir) + "/"):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            return await self.backend.get_url(key)
