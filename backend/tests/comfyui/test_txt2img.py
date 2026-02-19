"""ComfyUI text-to-image smoke test.

Submits the Qwen 2512 txt2img workflow to ComfyUI Cloud, polls for
completion, and downloads the output image.

Prerequisites:
    - COMFY_UI_HOST and COMFY_UI_KEY set in .env at repo root
    - Workflow template at docs/text-to-img-qwen.json

Usage:
    cd <repo-root>
    python backend/tests/comfyui/test_txt2img.py
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
WORKFLOW_PATH = REPO_ROOT / "docs" / "text-to-img-qwen.json"
TEST_PROMPT = "A red fox sitting in a snowy forest, soft morning light, photorealistic"
SEED = 42
OUTPUT_DIR = REPO_ROOT / "tmp" / "test_outputs"
OUTPUT_FILE = OUTPUT_DIR / "test_output.png"

COMFY_HOST = os.environ["COMFY_UI_HOST"].rstrip("/")
COMFY_KEY = os.environ["COMFY_UI_KEY"]

POLL_INTERVAL = 10  # seconds between status checks
POLL_TIMEOUT = 300  # max seconds to wait


# -- Workflow builder --------------------------------------------------------
def build_txt2img_workflow(prompt: str, seed: int = 0) -> dict:
    """Load the Qwen txt2img workflow and inject prompt + seed."""
    wf = json.loads(WORKFLOW_PATH.read_text())

    # Node 108 = CLIPTextEncode (positive prompt)
    wf["108"]["inputs"]["text"] = prompt
    # Node 106 = KSampler (seed)
    wf["106"]["inputs"]["seed"] = seed

    return wf


# -- Output extractor -------------------------------------------------------
def find_image_output(history: dict, prompt_id: str):
    """Extract image filename + subfolder from ComfyUI history.

    Returns (filename, subfolder) or None.
    """
    prompt_data = history.get(prompt_id, history)
    outputs = prompt_data.get("outputs", prompt_data)

    # Look for SaveImage node 123 first, then scan all nodes
    for node_id in ["123"] + [k for k in outputs if k != "123"]:
        node_out = outputs.get(node_id, {})
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "image", "gifs"):
            items = node_out.get(key, [])
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    return item["filename"], item.get("subfolder", "")

    return None


# -- Main --------------------------------------------------------------------
async def main():
    client = httpx.AsyncClient(
        base_url=COMFY_HOST,
        headers={"X-API-Key": COMFY_KEY},
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )

    try:
        # 1. Build workflow
        wf = build_txt2img_workflow(TEST_PROMPT, seed=SEED)
        print(f"Workflow loaded ({len(wf)} nodes)")
        print(f"Prompt: {TEST_PROMPT}")
        print(f"Seed:   {SEED}")

        # 2. Submit job
        print("\nSubmitting job...")
        r = await client.post("/api/prompt", json={"prompt": wf})
        if r.status_code != 200:
            print(f"ERROR submitting job: HTTP {r.status_code}")
            print(r.text[:1000])
            return
        data = r.json()
        prompt_id = data["prompt_id"]
        print(f"Job queued: prompt_id={prompt_id}")

        # 3. Poll for completion
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

        # 4. Get history and find output
        r = await client.get(f"/api/job/{prompt_id}/status")
        print(f"\nFull status response:")
        print(json.dumps(r.json(), indent=2, default=str)[:3000])

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

        result = find_image_output(history, prompt_id)
        if not result:
            print("\nERROR: Could not find image output in history.")
            print("Raw history (first 3000 chars):")
            print(json.dumps(history, indent=2, default=str)[:3000])
            return

        filename, subfolder = result
        print(f"Output found: filename={filename!r}, subfolder={subfolder!r}")

        # 5. Download image
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
