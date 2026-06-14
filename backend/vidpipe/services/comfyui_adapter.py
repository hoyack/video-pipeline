"""ComfyUI video generation adapter for the pipeline.

Provides a clean interface between the video pipeline and ComfyUI Cloud,
handling workflow building, status normalization, and result extraction.

The adapter abstracts away ComfyUI-specific details (image upload, workflow
templates, status endpoint quirks, history parsing) so the pipeline only
deals with: submit → poll → download.

Model-specific behavior (workflow builder, fps, end-frame/audio support)
lives in COMFY_VIDEO_SPECS — add new ComfyUI video models there.

Usage:
    adapter = ComfyUIVideoAdapter(comfy_client)
    op_id = await adapter.submit(video_prompt=..., start_frame_bytes=..., ...)
    status, err = await adapter.poll(op_id)
    if status == "completed":
        video_bytes, duration = await adapter.download(
            op_id, video_model=..., duration_seconds=...)
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from vidpipe.services.comfyui_client import (
    ComfyUIClient,
    build_ltx23_flf2v_workflow,
    build_seedance2_flf2v_workflow,
    build_wan22_flf2v_workflow,
    build_wan22_i2v_workflow,
    ltx_frames_for_duration,
    ltx_resolution,
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
    "walking backwards, head cut off, face out of frame, subject partially visible, "
    "inconsistent face, cropped head, decapitated framing"
)

# Generic video negative prompt for LTX. The static/motionless terms matter:
# LTX FLF2V interprets gentle motion prompts very literally and can produce a
# near-still clip (measured frame-diff 0.79 vs 30+ for the same keyframes/seed
# with motion-forward prompting).
LTX_NEGATIVE_PROMPT = (
    "static, still image, frozen, motionless scene, no movement, "
    "blurry, out of focus, overexposed, underexposed, low contrast, "
    "washed out colors, excessive noise, grainy texture, poor lighting, "
    "flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, extra limbs, disfigured hands, "
    "inconsistent perspective, camera shake, color banding, "
    "cartoonish rendering, uncanny valley effect, jittery movement, "
    "unnatural transitions, tilted camera, AI artifacts"
)

# Motion-forward prefix for LTX — without it, FLF2V conditioning eases in so
# conservatively that the first seconds of a clip can be effectively frozen.
_LTX_MOTION_PREFIX = (
    "Continuous visible motion from the very first frame, with the subject "
    "and environment in clear movement throughout the clip. "
)

# Framing-safety prefix for models without reference-image support
_FRAMING_SAFETY_PREFIX = (
    "Keep the subject's face and full head visible in frame throughout the entire clip. "
    "Maintain consistent character appearance and proportions. "
)

# Freeze-word substitutions for motion escalation. Order matters: longer,
# multi-word phrases first so they win over single-word replacements.
_MOTION_FREEZE_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("remains static", "moves dynamically"),
    ("remains still", "keeps moving"),
    ("stays still", "keeps moving"),
    ("static camera", "actively moving camera"),
    ("smooth, steady", "fast, energetic"),
    ("smooth and steady", "fast and energetic"),
    ("slow dolly", "fast dolly"),
    ("slowly", "continuously and energetically"),
    ("gently", "vigorously"),
    ("subtle", "pronounced"),
    ("steady", "sweeping"),
    ("motionless", "in constant motion"),
    ("barely moving", "moving briskly"),
)

# Strong override sentence prepended on motion escalation.
_MOTION_ESCALATION_PREFIX = (
    "HIGH MOTION: every part of the frame must be in obvious, continuous "
    "movement from the very first frame — the subject moving briskly, the "
    "camera tracking dynamically, and the environment (crowd, rain, steam, "
    "lights, traffic) all visibly animated. Avoid any static or frozen moment. "
)


def escalate_motion_prompt(prompt: str) -> str:
    """Rewrite a motion prompt to force visible motion from an LTX/ComfyUI model.

    Used when a generated clip measured near-static: replaces freeze-inducing
    language ("slowly", "steady", "remains static", ...) with energetic
    equivalents and prepends a strong high-motion override. Pure function so it
    can be unit-tested and reused by the regen path. Case-insensitive matching;
    replacements use the energetic form's casing.
    """
    import re

    rewritten = prompt
    for needle, replacement in _MOTION_FREEZE_SUBSTITUTIONS:
        rewritten = re.sub(re.escape(needle), replacement, rewritten, flags=re.IGNORECASE)
    return _MOTION_ESCALATION_PREFIX + rewritten

_VIDEO_OUTPUT_KEYS = ("videos", "video", "gifs", "images")
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

# Status normalization sets
_COMPLETED_STATUSES = frozenset({"completed", "success", "done"})
_FAILED_STATUSES = frozenset({"failed", "error", "cancelled"})
# Comfy Cloud holds jobs in queue states (observed: "queued_limited" when the
# account concurrency limit is hit) before any execution starts. Time spent
# queued must not count against execution timeouts.
QUEUED_STATUSES = frozenset({"queued", "queued_limited", "pending"})


# ---------------------------------------------------------------------------
# Per-model specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComfyVideoModelSpec:
    """Behavioral metadata for one ComfyUI video model.

    Attributes:
        fps: Output frame rate.
        fixed_duration: Actual clip duration in seconds when the model ignores
            the requested duration (WAN: always 81 frames / 16 fps). None means
            the clip honors the requested duration.
        supports_end_frame: Model can condition on an end keyframe.
        supports_audio: Model can generate audio.
        supports_char_refs: Workflow accepts character reference passthroughs.
        prompt_prefix: Text prepended to the motion prompt at submit time
            (framing safety for models without reference images).
    """

    fps: float
    fixed_duration: Optional[float]
    supports_end_frame: bool
    supports_audio: bool
    supports_char_refs: bool
    prompt_prefix: Optional[str]

    def clip_duration(self, requested_seconds: int) -> float:
        """Actual output duration for a requested clip length."""
        if self.fixed_duration is not None:
            return self.fixed_duration
        return float(requested_seconds)


COMFY_VIDEO_SPECS: dict[str, ComfyVideoModelSpec] = {
    "wan-2.2-i2v": ComfyVideoModelSpec(
        fps=16,
        fixed_duration=81 / 16.0,
        supports_end_frame=False,
        supports_audio=False,
        supports_char_refs=False,
        prompt_prefix=_FRAMING_SAFETY_PREFIX,
    ),
    "wan-2.2-flf2v": ComfyVideoModelSpec(
        fps=16,
        fixed_duration=81 / 16.0,
        supports_end_frame=True,
        supports_audio=False,
        supports_char_refs=True,
        prompt_prefix=_FRAMING_SAFETY_PREFIX,
    ),
    "ltx-2.3-flf2v": ComfyVideoModelSpec(
        fps=25,
        fixed_duration=None,
        supports_end_frame=True,
        supports_audio=True,
        supports_char_refs=False,
        prompt_prefix=_LTX_MOTION_PREFIX,
    ),
    "seedance-2.0-flf2v": ComfyVideoModelSpec(
        fps=24,
        fixed_duration=None,
        supports_end_frame=True,
        supports_audio=True,
        supports_char_refs=False,
        prompt_prefix=None,
    ),
}


def _resize_image_bytes(image_bytes: bytes, width: int, height: int) -> bytes:
    """Resize an image to exact dimensions, returning PNG bytes.

    LTX guide images must match the latent dimensions; the official template
    does this with ResizeImageMaskNode ("scale dimensions"), which we replicate
    Python-side to keep the committed workflow graph minimal.
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.size != (width, height):
        img = img.convert("RGB").resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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

    # Pattern 1: look for SaveVideo node 108 specifically (WAN template)
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
        shot_index: int,
        video_model: str,
        duration_seconds: int = 5,
        audio_enabled: bool = False,
    ) -> str:
        """Upload assets, build the model-specific workflow, and queue prompt.

        Args:
            video_prompt: Motion/shot prompt text.
            start_frame_bytes: PNG bytes for the start keyframe.
            end_frame_bytes: PNG bytes for the end keyframe; None falls back
                to first-frame-only generation (even on FLF2V models).
            char_ref_bytes: Character reference image bytes (0-2). Only wired
                for models with supports_char_refs (QC passthrough on
                wan-2.2-flf2v).
            aspect_ratio: "16:9" or "9:16".
            seed: Random seed for reproducibility.
            shot_index: Shot number (for filename prefixes).
            video_model: Model ID; must be a key of COMFY_VIDEO_SPECS.
            duration_seconds: Requested clip duration (ignored by WAN models,
                which always produce ~5s).
            audio_enabled: Generate audio (models with supports_audio only).

        Returns:
            Operation ID in format "comfyui:{prompt_id}".
        """
        spec = COMFY_VIDEO_SPECS.get(video_model)
        if spec is None:
            raise ValueError(
                f"Unknown ComfyUI video model {video_model!r}. "
                f"Known: {sorted(COMFY_VIDEO_SPECS)}"
            )

        if spec.prompt_prefix:
            video_prompt = spec.prompt_prefix + video_prompt

        use_end_frame = end_frame_bytes is not None and spec.supports_end_frame
        workflow = await self._build_workflow(
            spec=spec,
            video_model=video_model,
            video_prompt=video_prompt,
            start_frame_bytes=start_frame_bytes,
            end_frame_bytes=end_frame_bytes if use_end_frame else None,
            char_ref_bytes=char_ref_bytes if spec.supports_char_refs else [],
            aspect_ratio=aspect_ratio,
            seed=seed,
            shot_index=shot_index,
            duration_seconds=duration_seconds,
            audio_enabled=audio_enabled and spec.supports_audio,
        )

        # Log what we injected into the workflow for diagnostics
        _log_workflow_injection(workflow, shot_index, video_model)

        prompt_id = await self.client.queue_prompt(workflow)
        logger.info("Shot %d: ComfyUI prompt queued: %s", shot_index, prompt_id)
        return f"comfyui:{prompt_id}"

    async def _build_workflow(
        self,
        *,
        spec: ComfyVideoModelSpec,
        video_model: str,
        video_prompt: str,
        start_frame_bytes: bytes,
        end_frame_bytes: Optional[bytes],
        char_ref_bytes: list[bytes],
        aspect_ratio: str,
        seed: int,
        shot_index: int,
        duration_seconds: int,
        audio_enabled: bool,
    ) -> dict:
        """Upload inputs and build the API workflow for one model."""
        if video_model in ("wan-2.2-i2v", "wan-2.2-flf2v"):
            width, height = wan_resolution(aspect_ratio)
            start_fn = await self.client.upload_image(
                start_frame_bytes, f"shot_{shot_index}_start.png"
            )
            logger.info("Shot %d: uploaded start keyframe as %s", shot_index, start_fn)

            if video_model == "wan-2.2-flf2v" and end_frame_bytes is not None:
                end_fn = await self.client.upload_image(
                    end_frame_bytes, f"shot_{shot_index}_end.png"
                )
                char_fns: list[Optional[str]] = [None, None]
                for i, ref in enumerate(char_ref_bytes[:2]):
                    char_fns[i] = await self.client.upload_image(
                        ref, f"shot_{shot_index}_charref_{i + 1}.png"
                    )
                return build_wan22_flf2v_workflow(
                    prompt=video_prompt,
                    start_keyframe_filename=start_fn,
                    end_keyframe_filename=end_fn,
                    char_ref_01_filename=char_fns[0],
                    char_ref_02_filename=char_fns[1],
                    width=width,
                    height=height,
                    length=81,
                    seed=seed,
                )

            if video_model == "wan-2.2-flf2v":
                logger.warning(
                    "Shot %d: wan-2.2-flf2v has no end keyframe — "
                    "falling back to I2V workflow",
                    shot_index,
                )
            return build_wan22_i2v_workflow(
                prompt=video_prompt,
                negative_prompt=WAN_I2V_NEGATIVE_PROMPT,
                image_filename=start_fn,
                width=width,
                height=height,
                length=81,
                seed=seed,
            )

        if video_model == "ltx-2.3-flf2v":
            width, height = ltx_resolution(aspect_ratio)
            # LTX guide images must match the latent dimensions
            start_fn = await self.client.upload_image(
                _resize_image_bytes(start_frame_bytes, width, height),
                f"shot_{shot_index}_start.png",
            )
            end_fn = None
            if end_frame_bytes is not None:
                end_fn = await self.client.upload_image(
                    _resize_image_bytes(end_frame_bytes, width, height),
                    f"shot_{shot_index}_end.png",
                )
            else:
                logger.warning(
                    "Shot %d: ltx-2.3-flf2v has no end keyframe — "
                    "using first-frame guide only",
                    shot_index,
                )
            return build_ltx23_flf2v_workflow(
                prompt=video_prompt,
                negative_prompt=LTX_NEGATIVE_PROMPT,
                start_keyframe_filename=start_fn,
                end_keyframe_filename=end_fn,
                width=width,
                height=height,
                frames=ltx_frames_for_duration(duration_seconds),
                seed=seed,
                generate_audio=audio_enabled,
            )

        if video_model == "seedance-2.0-flf2v":
            start_fn = await self.client.upload_image(
                start_frame_bytes, f"shot_{shot_index}_start.png"
            )
            end_fn = None
            if end_frame_bytes is not None:
                end_fn = await self.client.upload_image(
                    end_frame_bytes, f"shot_{shot_index}_end.png"
                )
            else:
                logger.warning(
                    "Shot %d: seedance-2.0-flf2v has no end keyframe — "
                    "first-frame-only generation",
                    shot_index,
                )
            return build_seedance2_flf2v_workflow(
                prompt=video_prompt,
                first_frame_filename=start_fn,
                last_frame_filename=end_fn,
                duration=max(4, min(15, duration_seconds)),
                resolution="720p",
                aspect_ratio=aspect_ratio,
                seed=seed,
                generate_audio=audio_enabled,
            )

        raise ValueError(f"No workflow builder for ComfyUI model {video_model!r}")

    async def poll(self, operation_id: str) -> tuple[str, Optional[str]]:
        """Check job status with normalized status values.

        Returns:
            (status, error_message) where status is one of:
            "completed", "running", "queued", "failed".
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
        elif raw_status in QUEUED_STATUSES:
            logger.debug(
                "ComfyUI %s: raw_status=%r → queued", prompt_id, raw_status,
            )
            return "queued", None
        else:
            logger.debug(
                "ComfyUI %s: raw_status=%r → running", prompt_id, raw_status,
            )
            return "running", None

    async def download(
        self,
        operation_id: str,
        *,
        video_model: str = "wan-2.2-i2v",
        duration_seconds: int = 5,
    ) -> tuple[bytes, float]:
        """Download the completed video from ComfyUI.

        Fetches execution history, locates the video output file,
        and downloads it.

        Args:
            operation_id: "comfyui:{prompt_id}" operation ID.
            video_model: Model that produced the job (drives the reported
                duration). Defaults preserve pre-spec behavior for in-flight
                jobs submitted before a deploy.
            duration_seconds: Requested clip duration.

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

        spec = COMFY_VIDEO_SPECS.get(video_model)
        duration = (
            spec.clip_duration(duration_seconds) if spec else 81 / 16.0
        )
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

            # Submitted workflow summary (WAN node IDs; other models degrade
            # gracefully to "(not found)" placeholders)
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

def _summarize_workflow(workflow: dict) -> dict:
    """Extract loggable key parameters from any of our video workflows."""
    summary: dict = {}
    for node_id, node in workflow.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ct == "CLIPTextEncode" or ct == "TextEncodeQwenImageEditPlus":
            key = "positive_prompt" if "positive_prompt" not in summary else "negative_prompt"
            summary[key] = str(inputs.get("text", inputs.get("prompt", "")))
        elif ct == "LoadImage":
            summary.setdefault("images", []).append(inputs.get("image", ""))
        elif ct in ("WanFirstLastFrameToVideo", "WanImageToVideo", "EmptyLTXVLatentVideo"):
            summary["dimensions"] = (
                f"{inputs.get('width', 0)}x{inputs.get('height', 0)}, "
                f"{inputs.get('length', 0)} frames"
            )
        elif ct == "ByteDance2FirstLastFrameNode":
            summary["positive_prompt"] = str(inputs.get("model.prompt", ""))
            summary["dimensions"] = (
                f"{inputs.get('model.resolution')} {inputs.get('model.ratio')} "
                f"{inputs.get('model.duration')}s audio={inputs.get('model.generate_audio')}"
            )
            summary["seed"] = inputs.get("seed")
        elif ct in ("KSamplerAdvanced", "RandomNoise"):
            summary["seed"] = inputs.get("noise_seed", summary.get("seed"))
    return summary


def _log_workflow_injection(
    workflow: dict, shot_index: int, video_model: str
) -> None:
    """Log key parameters injected into a ComfyUI workflow before submission."""
    s = _summarize_workflow(workflow)
    pos = s.get("positive_prompt", "")
    logger.info(
        "Shot %d [%s] workflow injection:\n"
        "  positive_prompt: %.120s%s\n"
        "  images: %s\n"
        "  dimensions: %s\n"
        "  seed: %s",
        shot_index, video_model,
        pos, "..." if len(pos) > 120 else "",
        s.get("images", []),
        s.get("dimensions", "(n/a)"),
        s.get("seed", "(n/a)"),
    )
