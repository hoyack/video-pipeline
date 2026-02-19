"""ComfyUI video generation adapter for the pipeline.

Provides a clean interface between the video pipeline and ComfyUI Cloud,
handling workflow building, status normalization, and result extraction.

The adapter abstracts away ComfyUI-specific details (image upload, workflow
templates, status endpoint quirks, history parsing) so the pipeline only
deals with: submit → poll → download.

Usage:
    adapter = ComfyUIVideoAdapter(comfy_client)
    op_id = await adapter.submit(video_prompt=..., start_frame_bytes=..., ...)
    status, err = await adapter.poll(op_id)
    if status == "completed":
        video_bytes, duration = await adapter.download(op_id)
"""

import logging
from typing import Optional

from vidpipe.services.comfyui_client import (
    ComfyUIClient,
    build_wan22_flf2v_workflow,
    build_wan22_i2v_workflow,
    wan_resolution,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants moved from video_gen.py (ComfyUI-specific)
# ---------------------------------------------------------------------------

# English equivalent of the Chinese negative prompt in the Wan I2V template
WAN_I2V_NEGATIVE_PROMPT = (
    "blurry, low quality, overexposed, static, fuzzy details, subtitles, "
    "watermark, painting, still image, gray tones, worst quality, "
    "JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, mutated, disfigured, malformed limbs, fused fingers, "
    "motionless scene, cluttered background, three legs, crowded background, "
    "walking backwards"
)

_VIDEO_OUTPUT_KEYS = ("videos", "video", "gifs", "images")
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

# Status normalization sets
_COMPLETED_STATUSES = frozenset({"completed", "success", "done"})
_FAILED_STATUSES = frozenset({"failed", "error", "cancelled"})


# ---------------------------------------------------------------------------
# Video output extraction from ComfyUI history
# ---------------------------------------------------------------------------

def find_video_output(
    history: dict, prompt_id: str
) -> Optional[tuple[str, str]]:
    """Extract the video output filename from ComfyUI history response.

    Looks for SaveVideo node 108's output in the history data.
    The history format varies — try common patterns.

    Returns:
        (filename, subfolder) tuple, or None if not found.
    """
    # Unwrap: history may be keyed by prompt_id or flat
    prompt_data = history.get(prompt_id, history)
    outputs = prompt_data.get("outputs", prompt_data)

    logger.debug("ComfyUI history outputs for %s: %s", prompt_id, outputs)

    def _extract_video(node_output: dict) -> Optional[tuple[str, str]]:
        """Try each video output key and return (filename, subfolder)."""
        for key in _VIDEO_OUTPUT_KEYS:
            items = node_output.get(key, [])
            if isinstance(items, dict):
                items = [items]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("filename"):
                        return (
                            item["filename"],
                            item.get("subfolder", ""),
                        )
        return None

    # Pattern 1: look for SaveVideo node 108 specifically
    node_108 = outputs.get("108", {})
    if isinstance(node_108, dict):
        result = _extract_video(node_108)
        if result:
            return result

    # Pattern 2: scan all nodes for any video output
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        result = _extract_video(node_output)
        if result:
            fn = result[0]
            if any(fn.endswith(ext) for ext in _VIDEO_EXTENSIONS):
                return result

    logger.warning(
        "Could not find video output in ComfyUI history for %s. "
        "Available output nodes: %s",
        prompt_id,
        list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__,
    )
    return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ComfyUIVideoAdapter:
    """Pipeline-facing adapter for ComfyUI video generation.

    Encapsulates all ComfyUI-specific logic: image upload, workflow building,
    prompt queueing, status normalization, and result download.

    The pipeline interacts through three methods:
      submit()   — upload assets + queue workflow → operation_id
      poll()     — check job status → normalized (status, error_msg)
      download() — retrieve completed video → (bytes, duration)
    """

    def __init__(self, client: ComfyUIClient):
        self.client = client

    async def submit(
        self,
        *,
        video_prompt: str,
        start_frame_bytes: bytes,
        end_frame_bytes: Optional[bytes],
        char_ref_bytes: list[bytes],
        aspect_ratio: str,
        seed: int,
        scene_index: int,
        video_model: str,
    ) -> str:
        """Upload assets, build workflow, and queue prompt.

        Args:
            video_prompt: Motion/scene prompt text.
            start_frame_bytes: PNG bytes for the start keyframe.
            end_frame_bytes: PNG bytes for the end keyframe (None for I2V).
            char_ref_bytes: List of character reference image bytes (0-2).
            aspect_ratio: "16:9" or "9:16".
            seed: Random seed for reproducibility.
            scene_index: Scene number (for filename prefixes).
            video_model: Model ID (e.g. "wan-2.2-i2v" or "wan-2.2-ref-i2v").

        Returns:
            Operation ID in format "comfyui:{prompt_id}".
        """
        is_i2v = video_model == "wan-2.2-i2v"

        # Upload start keyframe
        start_fn = await self.client.upload_image(
            start_frame_bytes, f"scene_{scene_index}_start.png"
        )
        logger.info("Scene %d: uploaded start keyframe as %s", scene_index, start_fn)

        width, height = wan_resolution(aspect_ratio)

        if is_i2v:
            workflow = build_wan22_i2v_workflow(
                prompt=video_prompt,
                negative_prompt=WAN_I2V_NEGATIVE_PROMPT,
                image_filename=start_fn,
                width=width,
                height=height,
                length=81,  # 81 frames @ 16fps ≈ 5s
                seed=seed,
            )
        else:
            # FLF2V: upload end keyframe + optional character references
            end_fn = await self.client.upload_image(
                end_frame_bytes, f"scene_{scene_index}_end.png"
            )
            logger.info("Scene %d: uploaded end keyframe as %s", scene_index, end_fn)

            char_ref_fns: list[str] = []
            for i, ref_bytes in enumerate(char_ref_bytes[:2]):
                fn = await self.client.upload_image(
                    ref_bytes, f"scene_{scene_index}_charref{i + 1:02d}.png"
                )
                char_ref_fns.append(fn)

            workflow = build_wan22_flf2v_workflow(
                prompt=video_prompt,
                start_keyframe_filename=start_fn,
                end_keyframe_filename=end_fn,
                char_ref_01_filename=char_ref_fns[0] if len(char_ref_fns) > 0 else None,
                char_ref_02_filename=char_ref_fns[1] if len(char_ref_fns) > 1 else None,
                width=width,
                height=height,
                length=81,
                seed=seed,
            )

        # Log what we injected into the workflow for diagnostics
        _log_workflow_injection(workflow, scene_index, video_model)

        prompt_id = await self.client.queue_prompt(workflow)
        logger.info("Scene %d: ComfyUI prompt queued: %s", scene_index, prompt_id)
        return f"comfyui:{prompt_id}"

    async def poll(self, operation_id: str) -> tuple[str, Optional[str]]:
        """Check job status with normalized status values.

        Returns:
            (status, error_message) where status is one of:
            "completed", "running", "failed".
        """
        prompt_id = operation_id.removeprefix("comfyui:")
        raw_status, error_msg = await self.client.poll_status(prompt_id)

        if raw_status in _COMPLETED_STATUSES:
            logger.info(
                "ComfyUI %s: raw_status=%r → completed", prompt_id, raw_status,
            )
            return "completed", None
        elif raw_status in _FAILED_STATUSES:
            logger.warning(
                "ComfyUI %s: raw_status=%r → failed (error=%s)",
                prompt_id, raw_status, error_msg,
            )
            return "failed", error_msg or f"ComfyUI job {raw_status}"
        else:
            logger.debug(
                "ComfyUI %s: raw_status=%r → running", prompt_id, raw_status,
            )
            return "running", None

    async def download(self, operation_id: str) -> tuple[bytes, float]:
        """Download the completed video from ComfyUI.

        Fetches execution history, locates the video output file,
        and downloads it.

        Returns:
            (video_bytes, duration_seconds)

        Raises:
            ValueError: If no video output found in history.
        """
        prompt_id = operation_id.removeprefix("comfyui:")

        history = await self.client.get_history(prompt_id)
        video_result = find_video_output(history, prompt_id)
        if not video_result:
            raise ValueError(
                f"No video output found in ComfyUI history for {prompt_id}"
            )

        filename, subfolder = video_result
        logger.info(
            "Downloading video output %r (subfolder=%r)", filename, subfolder
        )

        video_bytes = await self.client.download_output(
            filename, subfolder=subfolder,
        )

        duration = 81 / 16.0  # 81 frames @ 16fps ≈ 5.06s
        logger.info(
            "ComfyUI %s: downloaded %d bytes (%.1fs duration)",
            prompt_id, len(video_bytes), duration,
        )
        return video_bytes, duration

    async def inspect_job(self, operation_id: str) -> dict:
        """Read-only inspection of a ComfyUI job for debugging.

        Returns a summary dict with status, submitted workflow details,
        and output info — without triggering any new workflows.

        Safe to call at any time; only uses GET endpoints.
        """
        prompt_id = operation_id.removeprefix("comfyui:")
        info: dict = {"prompt_id": prompt_id}

        # Status
        try:
            raw_status, error_msg = await self.client.poll_status(prompt_id)
            info["raw_status"] = raw_status
            info["error_message"] = error_msg
            if raw_status in _COMPLETED_STATUSES:
                info["normalized_status"] = "completed"
            elif raw_status in _FAILED_STATUSES:
                info["normalized_status"] = "failed"
            else:
                info["normalized_status"] = "running"
        except Exception as e:
            info["status_error"] = str(e)

        # History
        try:
            history = await self.client.get_history(prompt_id)
            prompt_data = history.get(prompt_id, history)

            # Output info
            outputs = prompt_data.get("outputs", {})
            info["output_nodes"] = {}
            for node_id, node_out in outputs.items():
                if isinstance(node_out, dict):
                    info["output_nodes"][node_id] = list(node_out.keys())

            # Video output detection
            video_result = find_video_output(history, prompt_id)
            info["video_output"] = (
                {"filename": video_result[0], "subfolder": video_result[1]}
                if video_result else None
            )

            # Submitted workflow summary
            prompt_wf = prompt_data.get("prompt", {}).get("prompt", {})
            if prompt_wf:
                # Prompt text (node 93)
                n93 = prompt_wf.get("93", {}).get("inputs", {})
                info["positive_prompt"] = n93.get("text", "(not found)")

                # Negative prompt (node 89)
                n89 = prompt_wf.get("89", {}).get("inputs", {})
                info["negative_prompt"] = n89.get("text", "(not found)")

                # Start image (node 97)
                n97 = prompt_wf.get("97", {}).get("inputs", {})
                info["start_image"] = n97.get("image", "(not found)")

                # End image (node 200, FLF2V only)
                n200 = prompt_wf.get("200", {}).get("inputs", {})
                if n200:
                    info["end_image"] = n200.get("image", "(not found)")

                # Dimensions (node 98)
                n98 = prompt_wf.get("98", {}).get("inputs", {})
                info["dimensions"] = {
                    "width": n98.get("width"),
                    "height": n98.get("height"),
                    "length": n98.get("length"),
                }

                # Seed (node 86)
                n86 = prompt_wf.get("86", {}).get("inputs", {})
                info["seed"] = n86.get("noise_seed")
        except Exception as e:
            info["history_error"] = str(e)

        return info


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_workflow_injection(
    workflow: dict, scene_index: int, video_model: str
) -> None:
    """Log key parameters injected into a ComfyUI workflow before submission."""
    # Positive prompt (node 93)
    n93_text = workflow.get("93", {}).get("inputs", {}).get("text", "")
    # Negative prompt (node 89)
    n89_text = workflow.get("89", {}).get("inputs", {}).get("text", "")
    # Start image (node 97)
    n97_image = workflow.get("97", {}).get("inputs", {}).get("image", "")
    # End image (node 200, FLF2V only)
    n200_image = workflow.get("200", {}).get("inputs", {}).get("image", "")
    # Dimensions (node 98)
    n98 = workflow.get("98", {}).get("inputs", {})
    # Seed (node 86)
    n86_seed = workflow.get("86", {}).get("inputs", {}).get("noise_seed", "")

    logger.info(
        "Scene %d [%s] workflow injection:\n"
        "  positive_prompt: %.120s%s\n"
        "  negative_prompt: %.80s%s\n"
        "  start_image: %s\n"
        "  end_image: %s\n"
        "  dimensions: %dx%d, %d frames\n"
        "  seed: %s",
        scene_index, video_model,
        n93_text, "..." if len(n93_text) > 120 else "",
        n89_text, "..." if len(n89_text) > 80 else "",
        n97_image or "(none)",
        n200_image or "(none — I2V mode)",
        n98.get("width", 0), n98.get("height", 0), n98.get("length", 0),
        n86_seed,
    )
