"""Model catalog, pricing, and compatibility mappings."""

from __future__ import annotations

TEXT_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
}

IMAGE_MODELS = {
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "qwen-fast",
    "qwen-image-edit",
    "flux-dev",
    "flux-dev-lora",
    "flux-dev-redux",
    "flux-dev-full",
}

VIDEO_MODELS = {
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-001",
    "veo-3.1-lite-generate-001",
    "wan-2.2-i2v",
}

LEGACY_MODEL_REPLACEMENTS = {
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "veo-3.1-generate-preview": "veo-3.1-generate-001",
    "veo-3.1-fast-generate-preview": "veo-3.1-fast-generate-001",
    "wan-2.2-ref-i2v": "wan-2.2-i2v",
}

GLOBAL_REGION_MODELS = {
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3-pro-image-preview",
}

AUDIO_CAPABLE_VIDEO_MODELS = VIDEO_MODELS - {
    "veo-2.0-generate-001",
    "wan-2.2-i2v",
}

VIDEO_MODEL_DURATIONS: dict[str, list[int]] = {
    "veo-2.0-generate-001": [5, 6, 7, 8],
    "veo-3.0-generate-001": [4, 6, 8],
    "veo-3.0-fast-generate-001": [4, 6, 8],
    "veo-3.1-generate-001": [4, 6, 8],
    "veo-3.1-fast-generate-001": [4, 6, 8],
    "veo-3.1-lite-generate-001": [4, 6, 8],
    "wan-2.2-i2v": [5],
}

TEXT_MODEL_COST: dict[str, float] = {
    "gemini-2.5-flash": 0.006,
    "gemini-2.5-flash-lite": 0.001,
    "gemini-2.5-pro": 0.023,
    "gemini-3-flash-preview": 0.007,
    "gemini-3.1-pro-preview": 0.028,
    "gemini-3.5-flash": 0.007,
}

IMAGE_MODEL_COST: dict[str, float] = {
    "gemini-2.5-flash-image": 0.04,
    "gemini-3-pro-image-preview": 0.13,
    "qwen-fast": 0.0,
    "qwen-image-edit": 0.0,
    "flux-dev": 0.0,
    "flux-dev-lora": 0.0,
    "flux-dev-redux": 0.0,
    "flux-dev-full": 0.0,
}

VIDEO_MODEL_COST_SILENT: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-001": 0.10,
    "veo-3.1-lite-generate-001": 0.10,
    "wan-2.2-i2v": 0.0,
}

VIDEO_MODEL_COST_AUDIO: dict[str, float] = {
    "veo-2.0-generate-001": 0.35,
    "veo-3.0-generate-001": 0.40,
    "veo-3.0-fast-generate-001": 0.15,
    "veo-3.1-generate-001": 0.40,
    "veo-3.1-fast-generate-001": 0.15,
    "veo-3.1-lite-generate-001": 0.15,
    "wan-2.2-i2v": 0.0,
}


def canonical_model_id(model_id: str | None) -> str | None:
    """Return the replacement for a legacy model ID, if known."""
    if model_id is None:
        return None
    return LEGACY_MODEL_REPLACEMENTS.get(model_id, model_id)
