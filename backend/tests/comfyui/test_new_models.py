"""Live ComfyUI Cloud smoke tests for the new model workflows.

Submits each new model's production workflow (built by the real
vidpipe builders) to ComfyUI Cloud, polls, and downloads the output.

Prerequisites:
    - COMFY_UI_HOST and COMFY_UI_KEY set in .env at repo root
    - vidpipe importable (pip install -e backend/ or PYTHONPATH=backend)
    - Seedance is a PAID partner node: it only runs with
      RUN_PAID_COMFY_TESTS=1 in the environment

Usage:
    cd <repo-root>
    python backend/tests/comfyui/test_new_models.py qwen-2509
    python backend/tests/comfyui/test_new_models.py flux2-klein
    python backend/tests/comfyui/test_new_models.py ltx
    RUN_PAID_COMFY_TESTS=1 python backend/tests/comfyui/test_new_models.py seedance
    python backend/tests/comfyui/test_new_models.py all
"""

import asyncio
import io
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

from PIL import Image, ImageDraw  # noqa: E402

from vidpipe.services.comfyui_client import (  # noqa: E402
    ComfyUIClient,
    build_flux2_klein_workflow,
    build_ltx23_flf2v_workflow,
    build_qwen_edit_2509_workflow,
    build_seedance2_flf2v_workflow,
    find_comfyui_image_output,
    ltx_frames_for_duration,
)
from vidpipe.services.comfyui_adapter import find_video_output  # noqa: E402

OUTPUT_DIR = Path(
    os.environ.get("VIDPIPE_TEST_OUTPUT_DIR", "/tmp/vidpipe_test_outputs")
)
POLL_INTERVAL = 10
POLL_TIMEOUT = 900


def _test_image(color: tuple[int, int, int], label: str, size=(832, 480)) -> bytes:
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.ellipse([size[0] // 3, size[1] // 4, 2 * size[0] // 3, 3 * size[1] // 4],
                 fill=(240, 220, 60))
    draw.text((20, 20), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _run_job(client: ComfyUIClient, workflow: dict, label: str,
                   is_video: bool) -> bool:
    prompt_id = await client.queue_prompt(workflow)
    print(f"[{label}] queued: {prompt_id}")

    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed > POLL_TIMEOUT:
            print(f"[{label}] TIMED OUT after {elapsed:.0f}s")
            return False
        status, error = await client.poll_status(prompt_id)
        print(f"[{label}] [{elapsed:5.0f}s] status={status}")
        if status in ("completed", "success", "done"):
            break
        if status in ("failed", "error", "cancelled"):
            print(f"[{label}] FAILED: {error}")
            return False
        await asyncio.sleep(POLL_INTERVAL)

    history = await client.get_history(prompt_id)
    if is_video:
        result = find_video_output(history, prompt_id)
        ext = "mp4"
    else:
        result = find_comfyui_image_output(history, prompt_id)
        ext = "png"
    if not result:
        print(f"[{label}] ERROR: no output found in history")
        return False
    filename, subfolder = result
    data = await client.download_output(filename, subfolder=subfolder)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"new_model_{label}.{ext}"
    out.write_bytes(data)
    print(f"[{label}] SUCCESS → {out} ({len(data):,} bytes)")
    return True


async def test_qwen_2509(client: ComfyUIClient) -> bool:
    ref1 = await client.upload_image(_test_image((40, 60, 120), "REF1"), "t2509_r1.png")
    ref2 = await client.upload_image(_test_image((120, 40, 60), "REF2"), "t2509_r2.png")
    wf = build_qwen_edit_2509_workflow(
        prompt="Combine the subjects from both images into one scene on a beach at sunset",
        image_filenames=[ref1, ref2],
        seed=42,
        output_width=1664,
        output_height=928,
    )
    return await _run_job(client, wf, "qwen-2509", is_video=False)


async def test_flux2_klein(client: ComfyUIClient) -> bool:
    ref = await client.upload_image(_test_image((20, 100, 60), "REF"), "tklein_r1.png")
    wf = build_flux2_klein_workflow(
        prompt="The same yellow shape floating above a city skyline at night",
        width=1344,
        height=768,
        seed=42,
        reference_image_filenames=[ref],
    )
    return await _run_job(client, wf, "flux2-klein", is_video=False)


async def test_ltx(client: ComfyUIClient) -> bool:
    start = await client.upload_image(
        _test_image((30, 30, 80), "START", size=(1280, 720)), "tltx_start.png")
    end = await client.upload_image(
        _test_image((80, 30, 30), "END", size=(1280, 720)), "tltx_end.png")
    wf = build_ltx23_flf2v_workflow(
        prompt="The yellow circle drifts slowly to the right as the light shifts from blue to red",
        negative_prompt="blurry, distorted",
        start_keyframe_filename=start,
        end_keyframe_filename=end,
        width=1280,
        height=720,
        frames=ltx_frames_for_duration(4),
        seed=42,
        generate_audio=True,
    )
    return await _run_job(client, wf, "ltx", is_video=True)


async def test_seedance(client: ComfyUIClient) -> bool:
    if os.environ.get("RUN_PAID_COMFY_TESTS") != "1":
        print("[seedance] SKIPPED — set RUN_PAID_COMFY_TESTS=1 to run (bills real credits)")
        return True
    start = await client.upload_image(
        _test_image((30, 30, 80), "START"), "tsd_start.png")
    end = await client.upload_image(
        _test_image((80, 30, 30), "END"), "tsd_end.png")
    wf = build_seedance2_flf2v_workflow(
        prompt="The yellow circle drifts to the right; gentle ambient sound",
        first_frame_filename=start,
        last_frame_filename=end,
        duration=4,           # shortest duration to bound cost
        resolution="480p",    # cheapest resolution
        aspect_ratio="16:9",
        seed=42,
        generate_audio=True,
    )
    return await _run_job(client, wf, "seedance", is_video=True)


TESTS = {
    "qwen-2509": test_qwen_2509,
    "flux2-klein": test_flux2_klein,
    "ltx": test_ltx,
    "seedance": test_seedance,
}


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = list(TESTS) if which == "all" else [which]
    if any(t not in TESTS for t in targets):
        print(f"Unknown test {which!r}. Options: {list(TESTS)} or 'all'")
        sys.exit(2)

    client = ComfyUIClient(
        host=os.environ["COMFY_UI_HOST"].rstrip("/"),
        api_key=os.environ["COMFY_UI_KEY"],
    )
    failures = []
    try:
        for name in targets:
            ok = await TESTS[name](client)
            if not ok:
                failures.append(name)
    finally:
        await client.close()

    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print("\nALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
