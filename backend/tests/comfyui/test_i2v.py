"""ComfyUI Wan 2.2 image-to-video smoke test.

Uploads an input image, submits the Wan I2V workflow to ComfyUI Cloud,
polls for completion, and downloads the output video.

Prerequisites:
    - COMFY_UI_HOST and COMFY_UI_KEY set in .env at repo root
    - Workflow template at docs/wan-img-to-video.json
    - Input image at tmp/test_outputs/test_output.png
      (run test_txt2img.py first to generate it)

Usage:
    cd <repo-root>
    python backend/tests/comfyui/test_i2v.py
"""

import asyncio
import json
import os
import sys
import time
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

import httpx

# -- Config ------------------------------------------------------------------
WORKFLOW_PATH = REPO_ROOT / "docs" / "wan-img-to-video.json"
INPUT_IMAGE = REPO_ROOT / "tmp" / "test_outputs" / "test_output.png"
TEST_PROMPT = "The red fox slowly turns its head, breath visible in the cold air, soft snow falling gently around it"
SEED = 42
WIDTH = 640
HEIGHT = 640
LENGTH = 81  # ~5s at 16fps
OUTPUT_DIR = REPO_ROOT / "tmp" / "test_outputs"
OUTPUT_FILE = OUTPUT_DIR / "test_output_video.mp4"

COMFY_HOST = os.environ["COMFY_UI_HOST"].rstrip("/")
COMFY_KEY = os.environ["COMFY_UI_KEY"]

POLL_INTERVAL = 15  # seconds between status checks (video gen is slow)
POLL_TIMEOUT = 900  # 15 min max


# -- Workflow builder --------------------------------------------------------
def build_i2v_workflow(
    prompt: str,
    image_filename: str,
    seed: int = 0,
    width: int = 640,
    height: int = 640,
    length: int = 81,
) -> dict:
    """Load Wan I2V workflow and inject runtime parameters."""
    wf = json.loads(WORKFLOW_PATH.read_text())

    wf["93"]["inputs"]["text"] = prompt       # positive prompt
    wf["97"]["inputs"]["image"] = image_filename  # start image
    wf["86"]["inputs"]["noise_seed"] = seed   # KSamplerAdvanced seed
    wf["98"]["inputs"]["width"] = width       # WanImageToVideo dims
    wf["98"]["inputs"]["height"] = height
    wf["98"]["inputs"]["length"] = length

    return wf


# -- Video output extractor --------------------------------------------------
def find_video_output(history: dict, prompt_id: str):
    """Extract video filename + subfolder from ComfyUI history.

    Returns (filename, subfolder) or None.
    """
    prompt_data = history.get(prompt_id, history)
    outputs = prompt_data.get("outputs", prompt_data)

    for node_id in ["108"] + [k for k in outputs if k != "108"]:
        node_out = outputs.get(node_id, {})
        if not isinstance(node_out, dict):
            continue
        for key in ("videos", "gifs", "images"):
            items = node_out.get(key, [])
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    return item["filename"], item.get("subfolder", "")

    return None


# -- Main --------------------------------------------------------------------
async def main():
    if not INPUT_IMAGE.exists():
        print(f"ERROR: Input image not found: {INPUT_IMAGE}")
        print("Run test_txt2img.py first to generate it.")
        return

    client = httpx.AsyncClient(
        base_url=COMFY_HOST,
        headers={"X-API-Key": COMFY_KEY},
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )

    try:
        # 1. Upload input image
        image_bytes = INPUT_IMAGE.read_bytes()
        print(f"Uploading {INPUT_IMAGE.name} ({len(image_bytes):,} bytes)...")
        r = await client.post(
            "/api/upload/image",
            files={"image": (INPUT_IMAGE.name, image_bytes, "image/png")},
        )
        if r.status_code != 200:
            print(f"ERROR uploading image: HTTP {r.status_code}")
            print(r.text[:1000])
            return
        upload_data = r.json()
        uploaded_filename = upload_data.get("name", INPUT_IMAGE.name)
        print(f"Uploaded as: {uploaded_filename}")

        # 2. Build workflow
        wf = build_i2v_workflow(
            prompt=TEST_PROMPT,
            image_filename=uploaded_filename,
            seed=SEED,
            width=WIDTH,
            height=HEIGHT,
            length=LENGTH,
        )
        print(f"\nWorkflow loaded ({len(wf)} nodes)")
        print(f"Prompt: {TEST_PROMPT}")
        print(f"Seed:   {SEED}")
        print(f"Dims:   {WIDTH}x{HEIGHT}, {LENGTH} frames")

        # 3. Submit job
        print("\nSubmitting job...")
        r = await client.post("/api/prompt", json={"prompt": wf})
        if r.status_code != 200:
            print(f"ERROR submitting job: HTTP {r.status_code}")
            print(r.text[:1000])
            return
        data = r.json()
        prompt_id = data["prompt_id"]
        print(f"Job queued: prompt_id={prompt_id}")

        # 4. Poll for completion
        print(f"\nPolling (interval={POLL_INTERVAL}s, timeout={POLL_TIMEOUT}s)...")
        t0 = time.time()
        final_status = None

        while True:
            elapsed = time.time() - t0
            if elapsed > POLL_TIMEOUT:
                print(f"TIMED OUT after {elapsed:.0f}s")
                return

            r = await client.get(f"/api/job/{prompt_id}/status")
            if r.status_code != 200:
                print(f"  [{elapsed:5.0f}s] Poll error: HTTP {r.status_code} - {r.text[:200]}")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            status_data = r.json()
            status = status_data.get("status", "unknown")
            print(f"  [{elapsed:5.0f}s] status={status}")

            if status in ("completed", "success", "done"):
                final_status = status
                break
            elif status in ("failed", "error", "cancelled"):
                print(f"\nJOB FAILED:")
                print(json.dumps(status_data, indent=2, default=str))
                return

            await asyncio.sleep(POLL_INTERVAL)

        elapsed = time.time() - t0
        print(f"\nJob completed in {elapsed:.1f}s (status={final_status})")

        # 5. Get history and find output
        history = None
        for ep in [f"/api/history_v2/{prompt_id}", f"/api/history/{prompt_id}"]:
            print(f"\nTrying {ep}...")
            r = await client.get(ep)
            print(f"  HTTP {r.status_code}")
            if r.status_code == 200:
                history = r.json()
                break
            else:
                print(f"  {r.text[:300]}")

        if history is None:
            print("\nERROR: All history endpoints failed.")
            return
        print(f"History received ({len(json.dumps(history))} bytes)")

        result = find_video_output(history, prompt_id)
        if not result:
            print("\nERROR: Could not find video output in history.")
            print("Raw history (first 3000 chars):")
            print(json.dumps(history, indent=2, default=str)[:3000])
            return

        filename, subfolder = result
        print(f"Output found: filename={filename!r}, subfolder={subfolder!r}")

        # 6. Download video
        print(f"\nDownloading {filename}...")
        params = {"filename": filename, "type": "output"}
        if subfolder:
            params["subfolder"] = subfolder
        r = await client.get("/api/view", params=params)

        if r.status_code != 200:
            print(f"Download failed: HTTP {r.status_code}")
            print(r.text[:500])
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_bytes(r.content)
        print(f"Saved: {OUTPUT_FILE} ({len(r.content):,} bytes)")
        print("\nSUCCESS")

    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
