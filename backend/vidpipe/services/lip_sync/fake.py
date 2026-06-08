"""Local fake lip-sync adapter used for tests and pipeline wiring."""

from __future__ import annotations

import shutil

from vidpipe.services.lip_sync.base import LipSyncAdapter, LipSyncRequest, LipSyncResult


class FakeLipSyncAdapter(LipSyncAdapter):
    """Copies the input clip to the output path without modifying pixels."""

    async def sync(self, request: LipSyncRequest) -> LipSyncResult:
        request.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(request.input_video_path, request.output_video_path)
        return LipSyncResult(
            output_video_path=request.output_video_path,
            metrics={"provider": "fake", "copied": True},
        )
