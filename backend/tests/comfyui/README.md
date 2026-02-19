# ComfyUI Smoke Tests

Standalone integration tests for the ComfyUI Cloud API. These are **not** unit
tests — they make real API calls to `cloud.comfy.org` and cost real credits.

## Prerequisites

1. `COMFY_UI_HOST` and `COMFY_UI_KEY` set in `.env` at the repo root.
2. Workflow JSON templates in `docs/` (committed to the repo).
3. Run from the repo root directory.

## Tests

### `test_txt2img.py` — Qwen 2512 Text-to-Image

Submits a text prompt to the Qwen txt2img workflow and downloads the generated
PNG. Quick sanity check that the ComfyUI Cloud connection, prompt submission,
polling, and download all work.

```bash
python backend/tests/comfyui/test_txt2img.py
```

- **Workflow:** `docs/text-to-img-qwen.json`
- **Output:** `tmp/test_outputs/test_output.png`
- **Time:** ~15-30s

### `test_i2v.py` — Wan 2.2 Image-to-Video

Uploads a source image, submits the Wan 2.2 I2V workflow, and downloads the
generated video. Tests image upload, I2V workflow injection (prompt, seed,
dimensions), and video output retrieval.

```bash
python backend/tests/comfyui/test_i2v.py
```

- **Workflow:** `docs/wan-img-to-video.json`
- **Input:** `tmp/test_outputs/test_output.png` (run `test_txt2img.py` first)
- **Output:** `tmp/test_outputs/test_output_video.mp4`
- **Time:** ~30-90s

### `test_flf2v.py` — Wan 2.2 First-Last-Frame-to-Video

Generates synthetic solid-color start/end keyframes, submits the Wan 2.2 FLF2V
workflow, and downloads the result. Tests the `build_wan22_flf2v_workflow`
builder and the pipeline's `find_video_output` history extractor.

```bash
python backend/tests/comfyui/test_flf2v.py
```

- **Workflow:** `docs/video_wan2_2_14B_i2v.json`
- **Input:** Synthetic PNGs (generated in-memory)
- **Output:** `tmp/test_outputs/test_flf2v_output.mp4`
- **Time:** ~30-90s

## Output Location

All test outputs are saved to `tmp/test_outputs/` which is gitignored.

## Running in Sequence

To run the full chain (txt2img generates the input for i2v):

```bash
python backend/tests/comfyui/test_txt2img.py && python backend/tests/comfyui/test_i2v.py
```
