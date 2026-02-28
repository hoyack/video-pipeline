"""Storage backend abstraction for local filesystem and Supabase Storage (S3).

Provides a unified interface for storing and retrieving pipeline artifacts
(keyframes, video clips, stitched output, manifest uploads). Two implementations:

- LocalStorageBackend: wraps current filesystem logic under tmp_dir
- S3StorageBackend: uses Supabase Storage REST API (backed by MinIO S3)

Usage:
    from vidpipe.services.storage_backend import get_storage_backend

    storage = get_storage_backend()
    key = await storage.put("scene_id/keyframes/shot_0_start.png", data, "image/png")
    url = await storage.get_url(key)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract interface for artifact storage."""

    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str) -> str:
        """Store data at key. Returns the key."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve data by key."""

    @abstractmethod
    async def get_url(self, key: str) -> str:
        """Get a public URL for serving the asset."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from storage."""

    @abstractmethod
    def is_local(self) -> bool:
        """Return True if this backend serves files from local disk."""


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage under tmp_dir.

    Keys are relative paths (e.g. "scene_id/keyframes/shot_0_start.png").
    Files are stored at {base_dir}/{key}.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        filepath = self.base_dir / key
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(data)
        return key

    async def get(self, key: str) -> bytes:
        filepath = self.base_dir / key
        if not filepath.exists():
            raise FileNotFoundError(f"Local file not found: {filepath}")
        return filepath.read_bytes()

    async def get_url(self, key: str) -> str:
        # Served by the API via FileResponse
        return f"/api/storage/{key}"

    async def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()

    async def delete(self, key: str) -> None:
        filepath = self.base_dir / key
        if filepath.exists():
            filepath.unlink()

    def is_local(self) -> bool:
        return True

    def resolve_local_path(self, key: str) -> Path:
        """Resolve a key to an absolute local filesystem path."""
        return self.base_dir / key


class S3StorageBackend(StorageBackend):
    """Supabase Storage REST API backend (backed by MinIO S3).

    Uses httpx to talk to the Supabase Storage API through Kong.
    Endpoint pattern: {s3_endpoint}/object/{bucket}/{key}
    Auth: service_role JWT in Authorization header.
    """

    def __init__(self, endpoint: str, bucket: str, service_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
        self.service_key = service_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.service_key}",
                },
            )
        return self._client

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        client = self._get_client()
        url = f"{self.endpoint}/object/{self.bucket}/{key}"

        # Supabase Storage uses POST for new objects, PUT for upsert
        resp = await client.post(
            url,
            content=data,
            headers={
                "Content-Type": content_type,
                "x-upsert": "true",
            },
        )
        if resp.status_code not in (200, 201):
            logger.error(
                f"S3 PUT failed: {resp.status_code} {resp.text} for key={key}"
            )
            resp.raise_for_status()

        logger.debug(f"S3 PUT ok: {key} ({len(data)} bytes)")
        return key

    async def get(self, key: str) -> bytes:
        client = self._get_client()
        url = f"{self.endpoint}/object/{self.bucket}/{key}"
        resp = await client.get(url)
        if resp.status_code != 200:
            raise FileNotFoundError(
                f"S3 GET failed: {resp.status_code} for key={key}"
            )
        return resp.content

    async def get_url(self, key: str) -> str:
        # Public URL via Supabase Storage
        return f"{self.endpoint}/object/public/{self.bucket}/{key}"

    async def exists(self, key: str) -> bool:
        client = self._get_client()
        url = f"{self.endpoint}/object/{self.bucket}/{key}"
        resp = await client.head(url)
        return resp.status_code == 200

    async def delete(self, key: str) -> None:
        client = self._get_client()
        url = f"{self.endpoint}/object/{self.bucket}"
        resp = await client.delete(
            url,
            json={"prefixes": [key]},
        )
        if resp.status_code not in (200, 204):
            logger.warning(f"S3 DELETE failed: {resp.status_code} for key={key}")

    def is_local(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Get the singleton StorageBackend instance, configured from settings."""
    global _backend
    if _backend is not None:
        return _backend

    from vidpipe.config import settings

    if settings.storage.storage_backend == "s3":
        _backend = S3StorageBackend(
            endpoint=settings.storage.s3_endpoint,
            bucket=settings.storage.s3_bucket,
            service_key=settings.storage.s3_service_key,
        )
        logger.info(
            f"Storage backend: S3 (bucket={settings.storage.s3_bucket}, "
            f"endpoint={settings.storage.s3_endpoint})"
        )
    else:
        _backend = LocalStorageBackend(settings.storage.tmp_dir)
        logger.info(f"Storage backend: local ({settings.storage.tmp_dir})")

    return _backend


def reset_storage_backend() -> None:
    """Reset the singleton (for testing)."""
    global _backend
    _backend = None
