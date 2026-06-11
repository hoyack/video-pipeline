"""Tests for ComfyUIVideoAdapter model dispatch and the video spec registry."""

import io

import pytest
from PIL import Image

from vidpipe.pipeline.video_gen import COMFYUI_VIDEO_MODELS
from vidpipe.services.comfyui_adapter import (
    COMFY_VIDEO_SPECS,
    ComfyUIVideoAdapter,
)


class MockComfyClient:
    """Records uploads and the queued workflow without any network I/O."""

    def __init__(self):
        self.uploads: list[str] = []
        self.queued: dict | None = None

    async def upload_image(self, image_bytes: bytes, filename: str) -> str:
        self.uploads.append(filename)
        return filename

    async def queue_prompt(self, workflow: dict) -> str:
        self.queued = workflow
        return "test-prompt-id"


def _png(width: int = 832, height: int = 480) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 80, 120)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return _png()


def _node_types(workflow: dict) -> set[str]:
    return {node["class_type"] for node in workflow.values()}


def test_every_routed_model_has_a_spec() -> None:
    assert set(COMFY_VIDEO_SPECS) == COMFYUI_VIDEO_MODELS


def test_spec_durations() -> None:
    # WAN ignores the requested duration (always 81 frames @ 16fps)
    assert COMFY_VIDEO_SPECS["wan-2.2-i2v"].clip_duration(8) == pytest.approx(81 / 16)
    assert COMFY_VIDEO_SPECS["wan-2.2-flf2v"].clip_duration(8) == pytest.approx(81 / 16)
    # LTX and Seedance honor it
    assert COMFY_VIDEO_SPECS["ltx-2.3-flf2v"].clip_duration(8) == 8.0
    assert COMFY_VIDEO_SPECS["seedance-2.0-flf2v"].clip_duration(12) == 12.0


@pytest.mark.asyncio
async def test_wan_flf2v_submits_flf_workflow_with_char_refs(png_bytes) -> None:
    client = MockComfyClient()
    adapter = ComfyUIVideoAdapter(client)

    op_id = await adapter.submit(
        video_prompt="orbit the subject",
        start_frame_bytes=png_bytes,
        end_frame_bytes=png_bytes,
        char_ref_bytes=[png_bytes, png_bytes],
        aspect_ratio="16:9",
        seed=11,
        shot_index=2,
        video_model="wan-2.2-flf2v",
        duration_seconds=5,
    )

    assert op_id == "comfyui:test-prompt-id"
    assert "shot_2_start.png" in client.uploads
    assert "shot_2_end.png" in client.uploads
    assert "shot_2_charref_1.png" in client.uploads
    assert "shot_2_charref_2.png" in client.uploads
    assert "WanFirstLastFrameToVideo" in _node_types(client.queued)
    # Framing-safety prefix applied per spec
    assert client.queued["93"]["inputs"]["text"].startswith("Keep the subject")


@pytest.mark.asyncio
async def test_wan_flf2v_missing_end_frame_falls_back_to_i2v(png_bytes) -> None:
    client = MockComfyClient()
    adapter = ComfyUIVideoAdapter(client)

    await adapter.submit(
        video_prompt="pan",
        start_frame_bytes=png_bytes,
        end_frame_bytes=None,
        char_ref_bytes=[],
        aspect_ratio="16:9",
        seed=0,
        shot_index=0,
        video_model="wan-2.2-flf2v",
    )

    types = _node_types(client.queued)
    assert "WanImageToVideo" in types
    assert "WanFirstLastFrameToVideo" not in types


@pytest.mark.asyncio
async def test_wan_i2v_ignores_end_frame(png_bytes) -> None:
    client = MockComfyClient()
    adapter = ComfyUIVideoAdapter(client)

    await adapter.submit(
        video_prompt="pan",
        start_frame_bytes=png_bytes,
        end_frame_bytes=png_bytes,  # provided but unsupported
        char_ref_bytes=[],
        aspect_ratio="16:9",
        seed=0,
        shot_index=0,
        video_model="wan-2.2-i2v",
    )

    assert "WanImageToVideo" in _node_types(client.queued)
    assert not any("end" in fn for fn in client.uploads)


@pytest.mark.asyncio
async def test_ltx_resizes_keyframes_and_sets_frames(png_bytes) -> None:
    client = MockComfyClient()
    adapter = ComfyUIVideoAdapter(client)

    await adapter.submit(
        video_prompt="dolly in",
        start_frame_bytes=png_bytes,
        end_frame_bytes=png_bytes,
        char_ref_bytes=[png_bytes],  # ignored: no char-ref support
        aspect_ratio="16:9",
        seed=3,
        shot_index=1,
        video_model="ltx-2.3-flf2v",
        duration_seconds=6,
        audio_enabled=True,
    )

    wf = client.queued
    assert wf["16"]["inputs"]["width"] == 1280
    assert wf["16"]["inputs"]["height"] == 720
    assert wf["16"]["inputs"]["length"] == 151  # 6s * 25fps + 1
    assert wf["30"]["inputs"]["audio"] == ["29", 0]
    assert not any("charref" in fn for fn in client.uploads)


@pytest.mark.asyncio
async def test_ltx_audio_disabled(png_bytes) -> None:
    client = MockComfyClient()
    adapter = ComfyUIVideoAdapter(client)

    await adapter.submit(
        video_prompt="dolly in",
        start_frame_bytes=png_bytes,
        end_frame_bytes=None,
        char_ref_bytes=[],
        aspect_ratio="9:16",
        seed=3,
        shot_index=1,
        video_model="ltx-2.3-flf2v",
        duration_seconds=4,
        audio_enabled=False,
    )

    wf = client.queued
    assert wf["16"]["inputs"]["width"] == 720
    assert "audio" not in wf["30"]["inputs"]


@pytest.mark.asyncio
async def test_seedance_clamps_duration_and_maps_ratio(png_bytes) -> None:
    client = MockComfyClient()
    adapter = ComfyUIVideoAdapter(client)

    await adapter.submit(
        video_prompt="fly through",
        start_frame_bytes=png_bytes,
        end_frame_bytes=None,
        char_ref_bytes=[],
        aspect_ratio="9:16",
        seed=1,
        shot_index=4,
        video_model="seedance-2.0-flf2v",
        duration_seconds=20,  # above max — clamped to 15
        audio_enabled=True,
    )

    node = client.queued["3"]["inputs"]
    assert node["model.duration"] == 15
    assert node["model.ratio"] == "9:16"
    assert node["model.generate_audio"] is True
    assert "last_frame" not in node


@pytest.mark.asyncio
async def test_unknown_model_raises(png_bytes) -> None:
    adapter = ComfyUIVideoAdapter(MockComfyClient())

    with pytest.raises(ValueError, match="Unknown ComfyUI video model"):
        await adapter.submit(
            video_prompt="x",
            start_frame_bytes=png_bytes,
            end_frame_bytes=None,
            char_ref_bytes=[],
            aspect_ratio="16:9",
            seed=0,
            shot_index=0,
            video_model="not-a-model",
        )
