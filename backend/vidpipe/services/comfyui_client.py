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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, wait_random, before_sleep_log

logger = logging.getLogger(__name__)


def _is_retriable_http(exc: BaseException) -> bool:
    """Return True for transient HTTP errors worth retrying (429, 5xx, connection errors)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                        httpx.PoolTimeout, httpx.ConnectTimeout, ConnectionError,
                        TimeoutError, OSError)):
        return True
    return False


_comfyui_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=60) + wait_random(0, 3),
    retry=retry_if_exception(_is_retriable_http),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

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
    "CreateVideo": ["fps"],
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


# Packaged API-format template (pre-converted from docs/video_wan2_2_14B_i2v.json
# — the docs/ copy is NOT shipped in the Docker image, so the converted graph
# is committed alongside the package).
_WAN_FLF2V_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "wan-flf2v.json"
)

# Module-level cache for the API template
_cached_api_template: Optional[dict] = None


def _load_api_template() -> dict:
    """Load the Wan FLF2V API-format workflow template, caching the result.

    Prefers the packaged pre-converted template; falls back to converting the
    app-format docs/ copy for dev checkouts where it exists.
    """
    global _cached_api_template
    if _cached_api_template is None:
        if _WAN_FLF2V_TEMPLATE_PATH.exists():
            with open(_WAN_FLF2V_TEMPLATE_PATH) as f:
                _cached_api_template = json.load(f)
            logger.info(
                "Loaded Wan FLF2V template from %s", _WAN_FLF2V_TEMPLATE_PATH,
            )
        else:
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
        prompt: Motion/shot prompt for CLIPTextEncode node 93
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

    @_comfyui_retry
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

    @_comfyui_retry
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

    @_comfyui_retry
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

    @_comfyui_retry
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

    @_comfyui_retry
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
# Module-level lazy singleton + DB credential cache
# ---------------------------------------------------------------------------

_comfyui_client: Optional[ComfyUIClient] = None
_db_comfyui_host: Optional[str] = None
_db_comfyui_key: Optional[str] = None


async def load_comfyui_credentials_from_db() -> None:
    """Load ComfyUI credentials from user_settings DB table.

    Caches host and api_key at module level. Closes existing client
    so next get_comfyui_client() picks up new credentials.
    """
    global _db_comfyui_host, _db_comfyui_key, _comfyui_client

    from vidpipe.db import async_session
    from vidpipe.db.models import UserSettings, DEFAULT_USER_ID
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
        )
        us = result.scalar_one_or_none()
        if not us:
            return

        _db_comfyui_host = us.comfyui_host or None
        _db_comfyui_key = us.comfyui_api_key or None

    # Close existing client so next call recreates with new creds
    if _comfyui_client is not None:
        await _comfyui_client.close()
        _comfyui_client = None

    if _db_comfyui_host or _db_comfyui_key:
        logger.info("Loaded ComfyUI credentials from DB")


def invalidate_comfyui_client() -> None:
    """Clear cached ComfyUI credentials and client."""
    global _db_comfyui_host, _db_comfyui_key, _comfyui_client
    _db_comfyui_host = None
    _db_comfyui_key = None
    _comfyui_client = None


async def get_comfyui_client(
    host: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ComfyUIClient:
    """Get or create a singleton ComfyUIClient.

    Fallback chain: explicit params → DB cache → environment variables.
    """
    global _comfyui_client

    resolved_host = (
        host
        or _db_comfyui_host
        or os.environ.get("COMFY_UI_HOST", "https://api.comfy.org")
    )
    resolved_key = (
        api_key
        or _db_comfyui_key
        or os.environ.get("COMFY_UI_KEY", "")
    )

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

# Qwen-Image native resolutions (~1.7MP budget) per aspect ratio
_QWEN_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "1:1": (1328, 1328),
}

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
    seed: int | None = 0,
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
    seed_value = int(seed or 0)
    workflow = copy.deepcopy(_load_qwen_template())
    workflow["108"]["inputs"]["text"] = prompt
    workflow["106"]["inputs"]["seed"] = seed_value
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
# Qwen Image Edit workflow builder
# ---------------------------------------------------------------------------

_QWEN_IMAGE_EDIT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "qwen-image-edit.json"
)

_cached_qwen_image_edit_template: Optional[dict] = None


def _load_qwen_image_edit_template() -> dict:
    """Load the Qwen Image Edit API-format workflow template, caching the result."""
    global _cached_qwen_image_edit_template
    if _cached_qwen_image_edit_template is None:
        with open(_QWEN_IMAGE_EDIT_TEMPLATE_PATH) as f:
            _cached_qwen_image_edit_template = json.load(f)
        logger.info("Loaded Qwen Image Edit template from %s", _QWEN_IMAGE_EDIT_TEMPLATE_PATH)
    return _cached_qwen_image_edit_template


def build_qwen_image_edit_workflow(
    *,
    prompt: str,
    input_image_filename: str,
    seed: int | None = 0,
) -> dict:
    """Build ComfyUI API-format workflow for Qwen Image Edit.

    This model takes an input image and an edit instruction prompt,
    then produces an edited version of the image.

    Injects runtime parameters into the cached template:
    - Node 102:76: positive prompt (edit instruction)
    - Node 102:3: KSampler seed
    - Node 78: LoadImage filename

    Args:
        prompt: Edit instruction describing what to change
        input_image_filename: Server-side filename of the uploaded input image
        seed: Random seed for reproducibility

    Returns:
        ComfyUI API-format prompt dict
    """
    seed_value = int(seed or 0)
    workflow = copy.deepcopy(_load_qwen_image_edit_template())
    workflow["102:76"]["inputs"]["prompt"] = prompt
    workflow["102:3"]["inputs"]["seed"] = seed_value
    workflow["78"]["inputs"]["image"] = input_image_filename
    return workflow


# ---------------------------------------------------------------------------
# Flux.1 Dev workflow builders + resolution mapping
# ---------------------------------------------------------------------------

_FLUX_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (1024, 1024),
}


def flux_resolution(aspect_ratio: str) -> tuple[int, int]:
    """Map aspect ratio string to Flux.1 Dev native resolution (width, height)."""
    if aspect_ratio not in _FLUX_RESOLUTIONS:
        raise ValueError(
            f"Unsupported aspect ratio for Flux.1 Dev: {aspect_ratio}. "
            f"Supported: {list(_FLUX_RESOLUTIONS.keys())}"
        )
    return _FLUX_RESOLUTIONS[aspect_ratio]


# -- Template paths + cached loaders --

_FLUX_BASE_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "flux-txt2img-base.json"
)
_FLUX_LORA_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "flux-txt2img-with-lora.json"
)
_FLUX_REFS_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "flux-txt2img-with-refs.json"
)
_FLUX_FULL_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "flux-txt2img-full.json"
)

_cached_flux_base_template: Optional[dict] = None
_cached_flux_lora_template: Optional[dict] = None
_cached_flux_refs_template: Optional[dict] = None
_cached_flux_full_template: Optional[dict] = None


def _load_flux_base_template() -> dict:
    """Load the Flux.1 Dev base txt2img template, caching the result."""
    global _cached_flux_base_template
    if _cached_flux_base_template is None:
        with open(_FLUX_BASE_TEMPLATE_PATH) as f:
            _cached_flux_base_template = json.load(f)
        logger.info("Loaded Flux base template from %s", _FLUX_BASE_TEMPLATE_PATH)
    return _cached_flux_base_template


def _load_flux_lora_template() -> dict:
    """Load the Flux.1 Dev + LoRA txt2img template, caching the result."""
    global _cached_flux_lora_template
    if _cached_flux_lora_template is None:
        with open(_FLUX_LORA_TEMPLATE_PATH) as f:
            _cached_flux_lora_template = json.load(f)
        logger.info("Loaded Flux LoRA template from %s", _FLUX_LORA_TEMPLATE_PATH)
    return _cached_flux_lora_template


def _load_flux_refs_template() -> dict:
    """Load the Flux.1 Dev + reference injection txt2img template, caching the result."""
    global _cached_flux_refs_template
    if _cached_flux_refs_template is None:
        with open(_FLUX_REFS_TEMPLATE_PATH) as f:
            _cached_flux_refs_template = json.load(f)
        logger.info("Loaded Flux refs template from %s", _FLUX_REFS_TEMPLATE_PATH)
    return _cached_flux_refs_template


def _load_flux_full_template() -> dict:
    """Load the Flux.1 Dev full hybrid (LoRA + refs) txt2img template, caching the result."""
    global _cached_flux_full_template
    if _cached_flux_full_template is None:
        with open(_FLUX_FULL_TEMPLATE_PATH) as f:
            _cached_flux_full_template = json.load(f)
        logger.info("Loaded Flux full template from %s", _FLUX_FULL_TEMPLATE_PATH)
    return _cached_flux_full_template


def build_flux_txt2img_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    lora_filename: Optional[str] = None,
    lora_strength: float = 0.8,
    reference_image_filenames: Optional[list[str]] = None,
    reference_strengths: Optional[list[float]] = None,
) -> dict:
    """Build ComfyUI API-format workflow for Flux.1 Dev text-to-image.

    Dynamically selects the correct template based on whether LoRA and/or
    reference images are provided, then injects runtime parameters.

    Template selection:
    - LoRA + refs -> full template
    - LoRA only  -> lora template
    - refs only  -> refs template
    - neither    -> base template

    Args:
        prompt: Positive prompt text for image generation.
        negative_prompt: Negative prompt text (default empty).
        width: Image width in pixels (default 1024).
        height: Image height in pixels (default 1024).
        seed: Random seed for reproducibility.
        lora_filename: LoRA .safetensors filename on the ComfyUI server.
            None means no LoRA.
        lora_strength: LoRA application strength (default 0.8).
        reference_image_filenames: List of uploaded reference image filenames
            on the ComfyUI server (max 3). None means no references.
        reference_strengths: Per-reference conditioning strengths.
            Defaults to 0.65 for each if not provided.

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config).
    """
    has_lora = lora_filename is not None
    has_refs = bool(reference_image_filenames)

    # Select template
    if has_lora and has_refs:
        template = _load_flux_full_template()
    elif has_lora:
        template = _load_flux_lora_template()
    elif has_refs:
        template = _load_flux_refs_template()
    else:
        template = _load_flux_base_template()

    workflow = copy.deepcopy(template)

    # Inject prompt text
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt

    # Inject seed and dimensions
    workflow["3"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height

    # Inject LoRA parameters
    if has_lora:
        workflow["14"]["inputs"]["lora_name"] = lora_filename
        workflow["14"]["inputs"]["strength_model"] = lora_strength

    # Inject reference image filenames and handle unused LoadImage nodes
    if has_refs:
        refs = reference_image_filenames  # type: ignore[assignment]
        strengths = reference_strengths or []
        ref_strength = strengths[0] if strengths else 0.65

        # Inject node 33 unCLIPConditioning strength
        workflow["33"]["inputs"]["strength"] = ref_strength

        # Map reference filenames to LoadImage nodes 20, 21, 22
        ref_nodes = ["20", "21", "22"]
        for i, node_id in enumerate(ref_nodes):
            if i < len(refs):
                workflow[node_id]["inputs"]["image"] = refs[i]
            else:
                # Remove unused LoadImage node
                workflow.pop(node_id, None)

        # Rebuild ImageBatch connections based on actual ref count
        if len(refs) == 1:
            # Single ref: ImageBatch gets only images1
            workflow["30"]["inputs"] = {"images1": ["20", 0]}
        elif len(refs) == 2:
            # Two refs: ImageBatch gets images1 + images2
            workflow["30"]["inputs"] = {
                "images1": ["20", 0],
                "images2": ["21", 0],
            }
        # else: 3 refs, keep all three connections as-is in template

    return workflow


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
        prompt: Motion/shot prompt for positive CLIP encoding
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


# ---------------------------------------------------------------------------
# Qwen Image Edit 2509 (multi-reference) workflow builder
# ---------------------------------------------------------------------------

_QWEN_EDIT_2509_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "qwen-edit-2509.json"
)

_cached_qwen_edit_2509_template: Optional[dict] = None


def _load_qwen_edit_2509_template() -> dict:
    """Load the Qwen Image Edit 2509 API-format workflow template, caching the result."""
    global _cached_qwen_edit_2509_template
    if _cached_qwen_edit_2509_template is None:
        with open(_QWEN_EDIT_2509_TEMPLATE_PATH) as f:
            _cached_qwen_edit_2509_template = json.load(f)
        logger.info("Loaded Qwen Edit 2509 template from %s", _QWEN_EDIT_2509_TEMPLATE_PATH)
    return _cached_qwen_edit_2509_template


def build_qwen_edit_2509_workflow(
    *,
    prompt: str,
    image_filenames: list[str],
    negative_prompt: str = "",
    seed: int | None = 0,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
) -> dict:
    """Build ComfyUI API-format workflow for Qwen Image Edit 2509 (multi-ref).

    Accepts 1-3 input images via TextEncodeQwenImageEditPlus (image1-3).
    Two output-size modes (the template KSampler runs at denoise 1.0, so the
    latent_image input only determines output dimensions, not content):

    - Edit mode (default): output dimensions follow image1 — used for
      end-keyframe generation where image1 is the start frame.
    - Generation mode (output_width/output_height given): output uses an
      EmptySD3LatentImage at the requested dimensions — used for start-frame
      composition from reference images at the scene aspect ratio.

    Injects runtime parameters into the cached template:
    - Node 110/111: positive/negative prompt + image1-3 references
    - Node 10/11/12: LoadImage filenames (unused ref nodes pruned)
    - Node 3: KSampler seed (+ latent_image source per mode)
    - Node 107: EmptySD3LatentImage dimensions (generation mode only)

    Args:
        prompt: Edit/composition instruction.
        image_filenames: 1-3 uploaded server-side filenames. The first is
            image1 (drives output size in edit mode).
        negative_prompt: Negative prompt text (default empty).
        seed: Random seed for reproducibility.
        output_width: Explicit output width (enables generation mode).
        output_height: Explicit output height (enables generation mode).

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config).

    Raises:
        ValueError: If image_filenames is empty or has more than 3 entries.
    """
    if not image_filenames:
        raise ValueError("qwen-image-edit-2509 requires at least one input image")
    if len(image_filenames) > 3:
        raise ValueError(
            f"qwen-image-edit-2509 supports at most 3 input images, got {len(image_filenames)}"
        )

    workflow = copy.deepcopy(_load_qwen_edit_2509_template())

    workflow["110"]["inputs"]["prompt"] = prompt
    workflow["111"]["inputs"]["prompt"] = negative_prompt
    workflow["3"]["inputs"]["seed"] = int(seed or 0)

    # Map filenames to LoadImage nodes 10/11/12; prune unused refs and drop
    # the corresponding imageN inputs from both text-encode nodes.
    ref_nodes = ["10", "11", "12"]
    image_keys = ["image1", "image2", "image3"]
    for i, node_id in enumerate(ref_nodes):
        if i < len(image_filenames):
            workflow[node_id]["inputs"]["image"] = image_filenames[i]
        else:
            workflow.pop(node_id, None)
            workflow["110"]["inputs"].pop(image_keys[i], None)
            workflow["111"]["inputs"].pop(image_keys[i], None)

    generation_mode = output_width is not None and output_height is not None
    if generation_mode:
        workflow["107"]["inputs"]["width"] = output_width
        workflow["107"]["inputs"]["height"] = output_height
        workflow["3"]["inputs"]["latent_image"] = ["107", 0]
    else:
        # Edit mode: latent follows image1 (template default); drop the
        # unused empty-latent node.
        workflow.pop("107", None)

    return workflow


# ---------------------------------------------------------------------------
# FLUX.2 Klein workflow builder + resolution mapping
# ---------------------------------------------------------------------------

_FLUX2_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
}


def flux2_resolution(aspect_ratio: str) -> tuple[int, int]:
    """Map aspect ratio string to FLUX.2 Klein native resolution (width, height)."""
    if aspect_ratio not in _FLUX2_RESOLUTIONS:
        raise ValueError(
            f"Unsupported aspect ratio for FLUX.2 Klein: {aspect_ratio}. "
            f"Supported: {list(_FLUX2_RESOLUTIONS.keys())}"
        )
    return _FLUX2_RESOLUTIONS[aspect_ratio]


_FLUX2_KLEIN_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "flux2-klein.json"
)

_cached_flux2_klein_template: Optional[dict] = None


def _load_flux2_klein_template() -> dict:
    """Load the FLUX.2 Klein API-format workflow template, caching the result."""
    global _cached_flux2_klein_template
    if _cached_flux2_klein_template is None:
        with open(_FLUX2_KLEIN_TEMPLATE_PATH) as f:
            _cached_flux2_klein_template = json.load(f)
        logger.info("Loaded FLUX.2 Klein template from %s", _FLUX2_KLEIN_TEMPLATE_PATH)
    return _cached_flux2_klein_template


# Per-ref-slot node IDs: (LoadImage, ImageScaleToTotalPixels, VAEEncode,
# positive ReferenceLatent, negative ReferenceLatent)
_FLUX2_REF_SLOTS: list[tuple[str, str, str, str, str]] = [
    ("20", "30", "40", "50", "60"),
    ("21", "31", "41", "51", "61"),
    ("22", "32", "42", "52", "62"),
    ("23", "33", "43", "53", "63"),
]


def build_flux2_klein_workflow(
    *,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    reference_image_filenames: Optional[list[str]] = None,
) -> dict:
    """Build ComfyUI API-format workflow for FLUX.2 Klein 4B (distilled).

    Supports 0-4 reference images. References are chained ReferenceLatent
    conditioning on both the positive and negative (zeroed) branches, per the
    official image_flux2_klein_image_edit_4b_distilled template. With zero
    references the graph reduces to plain text-to-image.

    Injects runtime parameters into the cached template:
    - Node 74: positive prompt text
    - Node 73: RandomNoise seed
    - Node 66/67: latent + scheduler dimensions
    - Nodes 20-23: reference LoadImage filenames (unused slots pruned)
    - Node 76: CFGGuider conditioning rewired to the last surviving
      ReferenceLatent in each chain (or directly to the encoders at 0 refs)

    Args:
        prompt: Positive prompt text.
        width: Output width in pixels (multiple of 16).
        height: Output height in pixels (multiple of 16).
        seed: Random seed for reproducibility.
        reference_image_filenames: Up to 4 uploaded server-side filenames.

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config).

    Raises:
        ValueError: If more than 4 reference images are provided.
    """
    refs = reference_image_filenames or []
    if len(refs) > 4:
        raise ValueError(
            f"flux-2-klein supports at most 4 reference images, got {len(refs)}"
        )

    workflow = copy.deepcopy(_load_flux2_klein_template())

    workflow["74"]["inputs"]["text"] = prompt
    workflow["73"]["inputs"]["noise_seed"] = seed
    workflow["66"]["inputs"]["width"] = width
    workflow["66"]["inputs"]["height"] = height
    workflow["67"]["inputs"]["width"] = width
    workflow["67"]["inputs"]["height"] = height

    # Fill used ref slots, prune unused ones.
    for i, (load_id, scale_id, encode_id, pos_rl_id, neg_rl_id) in enumerate(
        _FLUX2_REF_SLOTS
    ):
        if i < len(refs):
            workflow[load_id]["inputs"]["image"] = refs[i]
        else:
            for node_id in (load_id, scale_id, encode_id, pos_rl_id, neg_rl_id):
                workflow.pop(node_id, None)

    # Rewire CFGGuider conditioning to the end of the surviving chains.
    if refs:
        last_slot = _FLUX2_REF_SLOTS[len(refs) - 1]
        workflow["76"]["inputs"]["positive"] = [last_slot[3], 0]
        workflow["76"]["inputs"]["negative"] = [last_slot[4], 0]
    else:
        workflow["76"]["inputs"]["positive"] = ["74", 0]
        workflow["76"]["inputs"]["negative"] = ["82", 0]

    return workflow


# ---------------------------------------------------------------------------
# LTX-2.3 FLF2V workflow builder + resolution mapping
# ---------------------------------------------------------------------------

LTX_FPS = 25

_LTX_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}


def ltx_resolution(aspect_ratio: str) -> tuple[int, int]:
    """Map aspect ratio string to LTX-2.3 native resolution (width, height)."""
    if aspect_ratio not in _LTX_RESOLUTIONS:
        raise ValueError(
            f"Unsupported aspect ratio for LTX-2.3: {aspect_ratio}. "
            f"Supported: {list(_LTX_RESOLUTIONS.keys())}"
        )
    return _LTX_RESOLUTIONS[aspect_ratio]


def ltx_frames_for_duration(seconds: int) -> int:
    """Frame count for an LTX clip of the given duration (template math: s*fps+1)."""
    return seconds * LTX_FPS + 1


_LTX_FLF2V_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "ltx-flf2v.json"
)

_cached_ltx_flf2v_template: Optional[dict] = None


def _load_ltx_flf2v_template() -> dict:
    """Load the LTX-2.3 FLF2V API-format workflow template, caching the result."""
    global _cached_ltx_flf2v_template
    if _cached_ltx_flf2v_template is None:
        with open(_LTX_FLF2V_TEMPLATE_PATH) as f:
            _cached_ltx_flf2v_template = json.load(f)
        logger.info("Loaded LTX FLF2V template from %s", _LTX_FLF2V_TEMPLATE_PATH)
    return _cached_ltx_flf2v_template


def build_ltx23_flf2v_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    start_keyframe_filename: str,
    end_keyframe_filename: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    frames: int = 126,
    seed: int = 0,
    generate_audio: bool = True,
) -> dict:
    """Build ComfyUI API-format workflow for LTX-2.3 first-last-frame to video.

    Based on the official video_ltx2_3_flf2v template (22B distilled
    checkpoint, joint audio+video latent, dual LTXVAddGuide conditioning at
    frame 0 and frame -1, audio muxed via CreateVideo).

    Keyframe images must be pre-resized to (width, height) before upload —
    the committed template omits the ResizeImageMaskNode plumbing.

    Injects runtime parameters into the cached template:
    - Node 4/5: positive/negative prompt
    - Node 10/13: start/end LoadImage filenames
    - Node 16/19: latent dimensions + frame counts
    - Node 21: RandomNoise seed
    - End-frame guide (node 18) spliced out when end_keyframe_filename is None
    - Audio input dropped from CreateVideo when generate_audio is False

    Args:
        prompt: Motion/shot prompt (can include a "Music:"/"Audio:" hint).
        negative_prompt: Negative prompt text.
        start_keyframe_filename: Uploaded filename for the start frame.
        end_keyframe_filename: Uploaded filename for the end frame, or None
            for plain image-to-video.
        width: Video width (default 1280 for 16:9).
        height: Video height (default 720 for 16:9).
        frames: Frame count (use ltx_frames_for_duration; default 126 ≈ 5s @ 25fps).
        seed: Random seed.
        generate_audio: Mux generated audio into the output video.

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config).
    """
    workflow = copy.deepcopy(_load_ltx_flf2v_template())

    workflow["4"]["inputs"]["text"] = prompt
    workflow["5"]["inputs"]["text"] = negative_prompt
    workflow["10"]["inputs"]["image"] = start_keyframe_filename
    workflow["16"]["inputs"]["width"] = width
    workflow["16"]["inputs"]["height"] = height
    workflow["16"]["inputs"]["length"] = frames
    workflow["19"]["inputs"]["frames_number"] = frames
    workflow["21"]["inputs"]["noise_seed"] = seed

    if end_keyframe_filename is not None:
        workflow["13"]["inputs"]["image"] = end_keyframe_filename
    else:
        # Splice out the end-frame guide chain: downstream consumers of the
        # last guide (node 18) rewire to the first guide (node 17).
        workflow.pop("13", None)
        workflow.pop("15", None)
        workflow.pop("18", None)
        workflow["20"]["inputs"]["video_latent"] = ["17", 2]
        workflow["22"]["inputs"]["positive"] = ["17", 0]
        workflow["22"]["inputs"]["negative"] = ["17", 1]
        workflow["27"]["inputs"]["positive"] = ["17", 0]
        workflow["27"]["inputs"]["negative"] = ["17", 1]

    if not generate_audio:
        # Silent output: drop the audio link from CreateVideo and the now
        # unused audio decode node (the joint AV latent itself is structural).
        workflow["30"]["inputs"].pop("audio", None)
        workflow.pop("29", None)

    return workflow


# ---------------------------------------------------------------------------
# Seedance 2.0 FLF2V workflow builder (ByteDance partner API node)
# ---------------------------------------------------------------------------

_SEEDANCE_RATIOS = {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"}

_SEEDANCE_FLF2V_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "comfyui_templates" / "seedance-flf2v.json"
)

_cached_seedance_flf2v_template: Optional[dict] = None


def _load_seedance_flf2v_template() -> dict:
    """Load the Seedance 2.0 FLF2V API-format workflow template, caching the result."""
    global _cached_seedance_flf2v_template
    if _cached_seedance_flf2v_template is None:
        with open(_SEEDANCE_FLF2V_TEMPLATE_PATH) as f:
            _cached_seedance_flf2v_template = json.load(f)
        logger.info(
            "Loaded Seedance FLF2V template from %s", _SEEDANCE_FLF2V_TEMPLATE_PATH
        )
    return _cached_seedance_flf2v_template


def build_seedance2_flf2v_workflow(
    *,
    prompt: str,
    first_frame_filename: str,
    last_frame_filename: Optional[str] = None,
    duration: int = 5,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    seed: int = 0,
    generate_audio: bool = True,
    watermark: bool = False,
) -> dict:
    """Build ComfyUI API-format workflow for Seedance 2.0 first-last-frame video.

    Uses the ByteDance2FirstLastFrameNode partner API node (paid, metered in
    Comfy credits). The node's DynamicCombo "model" input serializes nested
    fields as dotted paths (model.prompt, model.resolution, ...).

    Injects runtime parameters into the cached template:
    - Node 1/2: first/last LoadImage filenames (node 2 pruned when no last frame)
    - Node 3: prompt, resolution, ratio, duration, generate_audio, seed, watermark

    Args:
        prompt: Motion/shot prompt text.
        first_frame_filename: Uploaded filename for the first frame.
        last_frame_filename: Uploaded filename for the last frame, or None
            for first-frame-only generation.
        duration: Clip duration in seconds (4-15).
        resolution: "480p", "720p", or "1080p".
        aspect_ratio: One of 16:9, 4:3, 1:1, 3:4, 9:16, 21:9, adaptive.
        seed: Random seed (0-2147483647; results are non-deterministic anyway).
        generate_audio: Generate synchronized audio.
        watermark: Add a visible watermark.

    Returns:
        ComfyUI API-format prompt dict (node_id -> node_config).

    Raises:
        ValueError: On out-of-range duration or unsupported aspect ratio.
    """
    if not 4 <= duration <= 15:
        raise ValueError(
            f"Seedance 2.0 duration must be 4-15 seconds, got {duration}"
        )
    if aspect_ratio not in _SEEDANCE_RATIOS:
        raise ValueError(
            f"Unsupported aspect ratio for Seedance 2.0: {aspect_ratio}. "
            f"Supported: {sorted(_SEEDANCE_RATIOS)}"
        )

    workflow = copy.deepcopy(_load_seedance_flf2v_template())

    node = workflow["3"]["inputs"]
    node["model.prompt"] = prompt
    node["model.resolution"] = resolution
    node["model.ratio"] = aspect_ratio
    node["model.duration"] = duration
    node["model.generate_audio"] = generate_audio
    node["seed"] = int(seed) % 2147483648
    node["watermark"] = watermark

    workflow["1"]["inputs"]["image"] = first_frame_filename
    if last_frame_filename is not None:
        workflow["2"]["inputs"]["image"] = last_frame_filename
    else:
        workflow.pop("2", None)
        node.pop("last_frame", None)

    return workflow


async def close_comfyui_client() -> None:
    """Close the singleton ComfyUIClient (for app shutdown)."""
    global _comfyui_client
    if _comfyui_client is not None:
        await _comfyui_client.close()
        _comfyui_client = None
