"""ComfyUI Cloud API client for Wan 2.2 FLF2V video generation.

Provides:
- Workflow template builder for Wan 2.2 14B First-Last-Frame-to-Video
- Async API client for cloud.comfy.org (upload, queue, poll, download)
- Resolution mapping for supported aspect ratios

Usage:
    from vidpipe.services.comfyui_client import get_comfyui_client

    client = get_comfyui_client()
    filename = await client.upload_image(image_bytes, "start.png")
    workflow = build_wan22_flf2v_workflow(prompt="...", start_keyframe_filename=filename, ...)
    prompt_id = await client.queue_prompt(workflow)
    ...
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolution mapping
# ---------------------------------------------------------------------------

_WAN_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (832, 480),
    "9:16": (480, 832),
}


def wan_resolution(aspect_ratio: str) -> tuple[int, int]:
    """Map aspect ratio string to Wan 2.2 native resolution (width, height)."""
    if aspect_ratio not in _WAN_RESOLUTIONS:
        raise ValueError(
            f"Unsupported aspect ratio for Wan 2.2: {aspect_ratio}. "
            f"Supported: {list(_WAN_RESOLUTIONS.keys())}"
        )
    return _WAN_RESOLUTIONS[aspect_ratio]


# ---------------------------------------------------------------------------
# Workflow template: app → API format converter
# ---------------------------------------------------------------------------

_WORKFLOW_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "docs"
    / "video_wan2_2_14B_i2v.json"
)

# Widget name mapping for each ComfyUI node type used in this workflow.
# Positional order must match the widgets_values array in the app-format JSON.
_WIDGET_NAMES: dict[str, list[str]] = {
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "ModelSamplingSD3": ["shift"],
    "CLIPTextEncode": ["text"],
    "LoadImage": ["image", "upload"],
    "WanFirstLastFrameToVideo": ["width", "height", "length", "batch_size"],
    "KSamplerAdvanced": [
        "add_noise", "noise_seed", "control_after_generate",
        "steps", "cfg", "sampler_name", "scheduler",
        "start_at_step", "end_at_step", "return_with_leftover_noise",
    ],
    "VAEDecode": [],
    "CreateVideo": ["frame_rate"],
    "SaveVideo": ["filename_prefix", "format", "codec"],
    "ImageFromBatch": ["batch_index", "length"],
    "SaveImage": ["filename_prefix"],
}


def _convert_app_to_api(app_workflow: dict) -> dict:
    """Convert ComfyUI app-format workflow to API format.

    App format uses a nodes array + links array + widgets_values.
    API format uses ``{node_id: {class_type, inputs}}`` with links resolved
    to ``[str(source_node_id), source_slot_index]`` references.
    """
    # Build link lookup: link_id → (str(source_node_id), source_slot_index)
    link_lookup: dict[int, tuple[str, int]] = {}
    for link in app_workflow.get("links", []):
        link_id, src_node, src_slot = link[0], link[1], link[2]
        link_lookup[link_id] = (str(src_node), src_slot)

    api_workflow: dict[str, dict] = {}
    for node in app_workflow.get("nodes", []):
        node_id = str(node["id"])
        class_type = node["type"]
        inputs: dict = {}

        # Resolve linked inputs (skip unconnected optional slots)
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None:
                src_node_id, src_slot = link_lookup[link_id]
                inputs[inp["name"]] = [src_node_id, src_slot]

        # Map positional widget values to named inputs
        widget_names = _WIDGET_NAMES.get(class_type, [])
        widget_values = node.get("widgets_values", [])
        for i, name in enumerate(widget_names):
            if i < len(widget_values):
                inputs[name] = widget_values[i]

        api_workflow[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

    return api_workflow


# Module-level cache for the converted API template
_cached_api_template: Optional[dict] = None


def _load_api_template() -> dict:
    """Load and convert the workflow JSON template, caching the result."""
    global _cached_api_template
    if _cached_api_template is None:
        with open(_WORKFLOW_TEMPLATE_PATH) as f:
            app_workflow = json.load(f)
        _cached_api_template = _convert_app_to_api(app_workflow)
        logger.info(
            "Loaded and converted ComfyUI workflow template from %s",
            _WORKFLOW_TEMPLATE_PATH,
        )
    return _cached_api_template


def build_wan22_flf2v_workflow(
    *,
    prompt: str,
    start_keyframe_filename: str,
    end_keyframe_filename: str,
    char_ref_01_filename: Optional[str] = None,
    char_ref_02_filename: Optional[str] = None,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    seed: int = 0,
) -> dict:
    """Build ComfyUI API-format workflow dict for Wan 2.2 FLF2V.

    Loads the workflow from the JSON template (app format), converts to
    API format, and injects runtime parameters.

    Args:
        prompt: Motion/scene prompt for CLIPTextEncode node 93
        start_keyframe_filename: Uploaded filename for node 97 (start frame)
        end_keyframe_filename: Uploaded filename for node 200 (end frame)
        char_ref_01_filename: Uploaded filename for node 201 (char ref 1, QC passthrough).
            Omitted from workflow when None.
        char_ref_02_filename: Uploaded filename for node 202 (char ref 2, QC passthrough).
            Omitted from workflow when None.
        width: Video width (default 832 for 16:9)
        height: Video height (default 480 for 16:9)
        length: Frame count (default 81 = ~5s at 16fps)
        seed: Random seed for KSampler node 86

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config)
    """
    workflow = copy.deepcopy(_load_api_template())

    # Inject runtime parameters
    workflow["93"]["inputs"]["text"] = prompt
    workflow["97"]["inputs"]["image"] = start_keyframe_filename
    workflow["200"]["inputs"]["image"] = end_keyframe_filename
    workflow["98"]["inputs"]["width"] = width
    workflow["98"]["inputs"]["height"] = height
    workflow["98"]["inputs"]["length"] = length
    workflow["86"]["inputs"]["noise_seed"] = seed

    # Character reference passthroughs (QC only) —
    # inject filename if present, remove nodes entirely if not.
    if char_ref_01_filename is not None:
        workflow["201"]["inputs"]["image"] = char_ref_01_filename
    else:
        workflow.pop("201", None)
        workflow.pop("225", None)

    if char_ref_02_filename is not None:
        workflow["202"]["inputs"]["image"] = char_ref_02_filename
    else:
        workflow.pop("202", None)
        workflow.pop("226", None)

    return workflow


# ---------------------------------------------------------------------------
# ComfyUI Cloud API client
# ---------------------------------------------------------------------------

class ComfyUIClient:
    """Async client for ComfyUI Cloud API (cloud.comfy.org).

    Handles image upload, prompt queueing, status polling, history
    retrieval, and output download.
    """

    def __init__(self, host: str, api_key: str):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _safe_host(self) -> str:
        """Host string safe for logging (no credentials)."""
        return self.host

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers={"X-API-Key": self.api_key},
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, connect=30.0),
            )
        return self._client

    async def upload_image(self, image_bytes: bytes, filename: str) -> str:
        """Upload an image to ComfyUI Cloud.

        Returns the server-side filename to use in workflow nodes.
        """
        logger.info(
            "POST %s/api/upload/image filename=%s size=%d bytes",
            self._safe_host, filename, len(image_bytes),
        )
        response = await self.client.post(
            "/api/upload/image",
            files={"image": (filename, image_bytes, "image/png")},
        )
        logger.info(
            "  upload response: HTTP %d", response.status_code,
        )
        response.raise_for_status()
        data = response.json()
        server_name = data.get("name", filename)
        logger.info("  server filename: %s", server_name)
        return server_name

    async def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow prompt for execution.

        Returns the prompt_id for status polling.
        """
        node_ids = sorted(workflow.keys())
        logger.info(
            "POST %s/api/prompt — workflow with %d nodes: %s",
            self._safe_host, len(workflow), node_ids,
        )
        response = await self.client.post(
            "/api/prompt",
            json={"prompt": workflow},
        )
        logger.info("  queue response: HTTP %d", response.status_code)
        response.raise_for_status()
        data = response.json()
        prompt_id = data["prompt_id"]
        logger.info("  prompt_id: %s", prompt_id)
        return prompt_id

    async def poll_status(self, prompt_id: str) -> tuple[str, Optional[str]]:
        """Check execution status of a queued prompt.

        Returns (status, error_message) tuple. Raw status from the API
        (e.g. "success", "pending", "error"). Normalization is done by
        the adapter layer.
        """
        response = await self.client.get(f"/api/job/{prompt_id}/status")
        logger.debug(
            "GET %s/api/job/%s/status — HTTP %d",
            self._safe_host, prompt_id, response.status_code,
        )
        response.raise_for_status()
        data = response.json()
        raw_status = data.get("status", "unknown")
        error_msg = data.get("error_message")
        logger.debug("  raw status=%s error=%s", raw_status, error_msg)
        return raw_status, error_msg

    async def get_history(self, prompt_id: str) -> dict:
        """Get execution history with output filenames.

        Returns the full history dict with per-node outputs.
        """
        logger.info(
            "GET %s/api/history_v2/%s", self._safe_host, prompt_id,
        )
        response = await self.client.get(f"/api/history_v2/{prompt_id}")
        logger.info("  history response: HTTP %d", response.status_code)
        response.raise_for_status()
        data = response.json()
        # Log output structure for diagnostics
        prompt_data = data.get(prompt_id, data)
        outputs = prompt_data.get("outputs", {})
        if isinstance(outputs, dict):
            for node_id, node_out in outputs.items():
                if isinstance(node_out, dict):
                    keys = list(node_out.keys())
                    logger.info(
                        "  history node %s: keys=%s", node_id, keys,
                    )
        return data

    async def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output",
    ) -> bytes:
        """Download an output file (video or image) by filename.

        The API returns a 302 redirect to a signed URL; httpx follows it
        automatically with follow_redirects=True.
        """
        params = {"filename": filename, "type": output_type}
        if subfolder:
            params["subfolder"] = subfolder
        logger.info(
            "GET %s/api/view filename=%s subfolder=%r type=%s",
            self._safe_host, filename, subfolder, output_type,
        )
        response = await self.client.get("/api/view", params=params)
        logger.info(
            "  download response: HTTP %d, %d bytes",
            response.status_code, len(response.content),
        )
        response.raise_for_status()
        return response.content

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_comfyui_client: Optional[ComfyUIClient] = None


async def get_comfyui_client(
    host: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ComfyUIClient:
    """Get or create a singleton ComfyUIClient.

    Falls back to environment variables COMFY_UI_HOST and COMFY_UI_KEY
    if host/api_key are not provided.
    """
    global _comfyui_client

    resolved_host = host or os.environ.get("COMFY_UI_HOST", "https://api.comfy.org")
    resolved_key = api_key or os.environ.get("COMFY_UI_KEY", "")

    # Recreate if config changed
    if _comfyui_client is not None:
        if (
            _comfyui_client.host != resolved_host.rstrip("/")
            or _comfyui_client.api_key != resolved_key
        ):
            # Config changed — close old client before replacing
            await _comfyui_client.close()
            _comfyui_client = None

    if _comfyui_client is None:
        if not resolved_key:
            raise ValueError(
                "ComfyUI API key not configured. Set COMFY_UI_KEY env var "
                "or configure in Settings > ComfyUI Configuration."
            )
        _comfyui_client = ComfyUIClient(resolved_host, resolved_key)

    return _comfyui_client


# ---------------------------------------------------------------------------
# Qwen txt2img workflow builder + image output extractor
# ---------------------------------------------------------------------------

_QWEN_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "qwen-txt2img.json"
)

_cached_qwen_template: Optional[dict] = None


def _load_qwen_template() -> dict:
    """Load the Qwen txt2img API-format workflow template, caching the result."""
    global _cached_qwen_template
    if _cached_qwen_template is None:
        with open(_QWEN_TEMPLATE_PATH) as f:
            _cached_qwen_template = json.load(f)
        logger.info("Loaded Qwen txt2img template from %s", _QWEN_TEMPLATE_PATH)
    return _cached_qwen_template


def build_qwen_txt2img_workflow(
    *,
    prompt: str,
    width: int = 1328,
    height: int = 1328,
    seed: int = 0,
) -> dict:
    """Build ComfyUI API-format workflow for Qwen 2512 text-to-image.

    Injects runtime parameters into the cached template:
    - Node 108: positive prompt text
    - Node 106: KSampler seed
    - Node 107: EmptySD3LatentImage dimensions

    Args:
        prompt: Text description for image generation
        width: Image width (default 1328, Qwen native)
        height: Image height (default 1328, Qwen native)
        seed: Random seed for reproducibility

    Returns:
        ComfyUI API-format prompt dict
    """
    workflow = copy.deepcopy(_load_qwen_template())
    workflow["108"]["inputs"]["text"] = prompt
    workflow["106"]["inputs"]["seed"] = seed
    workflow["107"]["inputs"]["width"] = width
    workflow["107"]["inputs"]["height"] = height
    return workflow


def find_comfyui_image_output(history: dict, prompt_id: str) -> tuple[str, str]:
    """Extract image filename and subfolder from ComfyUI history response.

    The history_v2 response nests data under the prompt_id key:
    ``{prompt_id: {outputs: {node_id: {images: [...]}}}}``

    Looks for SaveImage output in node 123 first (Qwen workflow),
    then scans all nodes for any SaveImage output.

    Args:
        history: Full history dict from /api/history_v2/{prompt_id}
        prompt_id: The prompt ID to look up

    Returns:
        (filename, subfolder) tuple

    Raises:
        ValueError: If no image output found in history
    """
    # history_v2 nests under the prompt_id key
    job_data = history.get(prompt_id, history)
    outputs = job_data.get("outputs", {})

    # Try node 123 first (Qwen SaveImage node)
    node_123 = outputs.get("123", {})
    images = node_123.get("images", [])
    if images:
        img = images[0]
        return img["filename"], img.get("subfolder", "")

    # Fallback: scan all nodes for SaveImage output
    for node_id, node_output in outputs.items():
        images = node_output.get("images", [])
        if images:
            img = images[0]
            return img["filename"], img.get("subfolder", "")

    raise ValueError(f"No image output found in history for prompt {prompt_id}")


# ---------------------------------------------------------------------------
# Wan 2.2 I2V workflow builder
# ---------------------------------------------------------------------------

_WAN_I2V_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "wan-i2v.json"
)

_cached_wan_i2v_template: Optional[dict] = None


def _load_wan_i2v_template() -> dict:
    """Load the Wan 2.2 I2V API-format workflow template, caching the result."""
    global _cached_wan_i2v_template
    if _cached_wan_i2v_template is None:
        with open(_WAN_I2V_TEMPLATE_PATH) as f:
            _cached_wan_i2v_template = json.load(f)
        logger.info("Loaded Wan I2V template from %s", _WAN_I2V_TEMPLATE_PATH)
    return _cached_wan_i2v_template


def build_wan22_i2v_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    image_filename: str,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    seed: int = 0,
) -> dict:
    """Build ComfyUI API-format workflow for Wan 2.2 Image-to-Video.

    Injects runtime parameters into the cached template:
    - Node 93: positive prompt (CLIPTextEncode)
    - Node 89: negative prompt (CLIPTextEncode)
    - Node 97: start image (LoadImage)
    - Node 98: dimensions + length (WanImageToVideo)
    - Node 86: noise seed (KSamplerAdvanced)

    Args:
        prompt: Motion/scene prompt for positive CLIP encoding
        negative_prompt: Quality-negative prompt for negative CLIP encoding
        image_filename: Uploaded filename for start image
        width: Video width (default 832 for 16:9)
        height: Video height (default 480 for 16:9)
        length: Frame count (default 81 = ~5s at 16fps)
        seed: Random seed for KSampler node 86

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config)
    """
    workflow = copy.deepcopy(_load_wan_i2v_template())

    workflow["93"]["inputs"]["text"] = prompt
    workflow["89"]["inputs"]["text"] = negative_prompt
    workflow["97"]["inputs"]["image"] = image_filename
    workflow["98"]["inputs"]["width"] = width
    workflow["98"]["inputs"]["height"] = height
    workflow["98"]["inputs"]["length"] = length
    workflow["86"]["inputs"]["noise_seed"] = seed

    return workflow


async def close_comfyui_client() -> None:
    """Close the singleton ComfyUIClient (for app shutdown)."""
    global _comfyui_client
    if _comfyui_client is not None:
        await _comfyui_client.close()
        _comfyui_client = None
