"""ComfyUI Wan 2.2 first-last-frame-to-video smoke test.

Generates synthetic solid-color start/end PNGs, uploads them, submits the
Wan FLF2V workflow to ComfyUI Cloud, polls for completion, downloads the
output video, and tests the pipeline's video output extractor.

Prerequisites:
    - COMFY_UI_HOST and COMFY_UI_KEY set in .env at repo root
    - Workflow template at docs/video_wan2_2_14B_i2v.json
    - Backend importable (run from repo root)

Usage:
    cd <repo-root>
    python backend/tests/comfyui/test_flf2v.py
"""

import asyncio
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Resolve repo root (three levels up from this file)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Load .env from repo root
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(REPO_ROOT / "backend"))

from vidpipe.services.comfyui_client import build_wan22_flf2v_workflow
from vidpipe.services.comfyui_adapter import find_video_output

import httpx

OUTPUT_DIR = REPO_ROOT / "tmp" / "test_outputs"
OUTPUT_FILE = OUTPUT_DIR / "test_flf2v_output.mp4"

COMFY_HOST = os.environ["COMFY_UI_HOST"].rstrip("/")
COMFY_KEY = os.environ["COMFY_UI_KEY"]


def make_png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    """Generate a minimal solid-color PNG in memory."""
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


async def main():
    client = httpx.AsyncClient(
        base_url=COMFY_HOST,
        headers={"X-API-Key": COMFY_KEY},
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )

    try:
        # 1. Upload synthetic start/end keyframes
        r = await client.post(
            "/api/upload/image",
            files={"image": ("start.png", make_png(832, 480, (200, 50, 50)), "image/png")},
        )
        sfn = r.json()["name"]
        r = await client.post(
            "/api/upload/image",
            files={"image": ("end.png", make_png(832, 480, (50, 50, 200)), "image/png")},
        )
        efn = r.json()["name"]
        print(f"Uploaded: start={sfn}, end={efn}")

        # 2. Build + queue workflow
        wf = build_wan22_flf2v_workflow(
            prompt="Slow dissolve from red to blue.",
            start_keyframe_filename=sfn,
            end_keyframe_filename=efn,
            width=832,
            height=480,
            length=81,
            seed=42,
        )
        r = await client.post("/api/prompt", json={"prompt": wf})
        pid = r.json()["prompt_id"]
        print(f"Queued: {pid}")

        # 3. Poll for completion
        t0 = time.time()
        for i in range(120):
            r = await client.get(f"/api/job/{pid}/status")
            data = r.json()
            s = data.get("status", "???")
            elapsed = time.time() - t0
            print(f"[{elapsed:5.0f}s] status={s}")
            if s in ("completed", "success", "done"):
                break
            elif s in ("failed", "error", "cancelled"):
                print(f"FAILED: {json.dumps(data, indent=2)}")
                return
            await asyncio.sleep(15)
        else:
            print("TIMED OUT")
            return

        # 4. Get history + test extractor
        for ep in [f"/api/history_v2/{pid}", f"/api/history/{pid}"]:
            print(f"\n=== {ep} ===")
            r = await client.get(ep)
            print(f"HTTP {r.status_code}")
            if r.status_code == 200:
                h = r.json()
                s = json.dumps(h, indent=2, default=str)
                print(s[:6000])
                if len(s) > 6000:
                    print(f"... ({len(s)} total)")

                result = find_video_output(h, pid)
                print(f"\nfind_video_output => {result}")

                if result:
                    fn, sf = result
                    print(f"\nDownloading {fn} (subfolder={sf!r})...")
                    r = await client.get(
                        "/api/view",
                        params={"filename": fn, "subfolder": sf, "type": "output"},
                    )
                    print(f"Download: HTTP {r.status_code}, {len(r.content)} bytes")
                    if r.status_code == 200:
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        OUTPUT_FILE.write_bytes(r.content)
                        print(f"Saved {OUTPUT_FILE} ({len(r.content)} bytes)")
                break
            else:
                print(r.text[:500])

    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
