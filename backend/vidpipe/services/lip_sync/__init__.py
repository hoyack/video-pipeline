"""Lip-sync adapter registry."""

from vidpipe.services.lip_sync.base import LipSyncAdapter, LipSyncRequest, LipSyncResult
from vidpipe.services.lip_sync.fake import FakeLipSyncAdapter


def get_lip_sync_adapter(adapter_type: str = "FAKE") -> LipSyncAdapter:
    """Return a lip-sync adapter by provider name."""
    if adapter_type.upper() == "FAKE":
        return FakeLipSyncAdapter()
    raise ValueError(f"Unsupported lip-sync adapter: {adapter_type}")


__all__ = [
    "FakeLipSyncAdapter",
    "LipSyncAdapter",
    "LipSyncRequest",
    "LipSyncResult",
    "get_lip_sync_adapter",
]
