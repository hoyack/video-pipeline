"""Opt-in 4K video upscaling (GPU, background).

Upscales a scene's finished ``final.mp4`` to 4K with RealESRGAN (tiled, on the
4070 Ti), frame by frame, re-muxing the original audio. This is deliberately a
manual trigger, not part of the pipeline — it costs ~9 min of GPU per 6s of
video. The result is saved as ``final_4k.mp4`` alongside the original.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from vidpipe.db import async_session
from vidpipe.db.models import Scene
from vidpipe.services.face_restore_service import get_face_restore_service
from vidpipe.services.file_manager import FileManager

logger = logging.getLogger("vidpipe.upscale")

upscale_router = APIRouter(prefix="/api", tags=["upscale"])


def _process_video(src_path: str, out_path: str, scale: int) -> int:
    """Decode → RealESRGAN upscale each frame → encode (libx264) + mux audio.

    Streams raw frames straight to ffmpeg (no temp PNGs). Returns frame count.
    Runs in a worker thread (blocking GPU + ffmpeg work).
    """
    import cv2
    import numpy as np

    rsvc = get_face_restore_service()
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    proc = None
    i = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok_enc, buf = cv2.imencode(".png", frame)
            up_png = rsvc.upscale(buf.tobytes(), scale) if ok_enc else None
            if up_png:
                up = cv2.imdecode(np.frombuffer(up_png, np.uint8), cv2.IMREAD_COLOR)
            else:
                up = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            if proc is None:
                h, w = up.shape[:2]
                proc = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                        "-i", src_path,
                        "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        out_path,
                    ],
                    stdin=subprocess.PIPE,
                )
            proc.stdin.write(up.tobytes())
            i += 1
            if i % 12 == 0:
                logger.info("upscale: %d/%d frames (%.1fs/frame)", i, total, (time.time() - t0) / i)
    finally:
        cap.release()
        if proc is not None:
            proc.stdin.close()
            proc.wait()
    logger.info("upscale: encoded %d frames in %.1fs", i, time.time() - t0)
    return i


async def _upscale_video_job(scene_id: uuid.UUID, scale: int) -> None:
    file_mgr = FileManager()
    src = file_mgr.get_output_path(scene_id, "final.mp4")
    out = file_mgr.get_output_path(scene_id, "final_4k.mp4")
    try:
        if not src.exists():
            # Materialize from storage (S3) if the local copy is gone.
            data = await file_mgr.read_bytes(f"{scene_id}/output/final.mp4")
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(data)

        n = await asyncio.to_thread(_process_video, str(src), str(out), scale)
        if n == 0 or not out.exists():
            logger.error("upscale: produced no output for scene %s", scene_id)
            return
        key = f"{scene_id}/output/final_4k.mp4"
        await file_mgr.backend.put(key, out.read_bytes(), "video/mp4")
        logger.info("upscale: scene %s done → %s (%d frames)", scene_id, key, n)
    except Exception as e:  # noqa: BLE001
        logger.error("upscale: scene %s failed: %s", scene_id, e)


@upscale_router.post("/scenes/{scene_id}/upscale-video", status_code=202)
async def upscale_scene_video(scene_id: uuid.UUID, background_tasks: BackgroundTasks):
    """Kick off a background 4K upscale of the scene's final video (opt-in, GPU)."""
    async with async_session() as session:
        scene = await session.get(Scene, scene_id)
        if scene is None:
            raise HTTPException(status_code=404, detail="Scene not found")

    rsvc = get_face_restore_service()
    if not await asyncio.to_thread(rsvc.has_upscale):
        raise HTTPException(
            status_code=503,
            detail="Upscaler unavailable (GPU or RealESRGAN weight missing).",
        )

    file_mgr = FileManager()
    has_local = file_mgr.get_output_path(scene_id, "final.mp4").exists()
    if not has_local:
        try:
            await file_mgr.read_bytes(f"{scene_id}/output/final.mp4")
        except Exception:
            raise HTTPException(status_code=409, detail="Scene has no final video to upscale.")

    from vidpipe.config import settings

    background_tasks.add_task(_upscale_video_job, scene_id, settings.face_swap.upscale_scale)
    return {
        "scene_id": str(scene_id),
        "status": "upscaling_started",
        "note": "4K upscale runs in the background on the GPU (~9 min per 6s of video). "
                "Result saved as final_4k.mp4 when complete.",
    }
