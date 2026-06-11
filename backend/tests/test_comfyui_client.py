import pytest

from vidpipe.services.comfyui_client import (
    build_flux2_klein_workflow,
    build_ltx23_flf2v_workflow,
    build_qwen_edit_2509_workflow,
    build_qwen_image_edit_workflow,
    build_qwen_txt2img_workflow,
    build_seedance2_flf2v_workflow,
    ltx_frames_for_duration,
)


def test_qwen_txt2img_workflow_defaults_null_seed_to_zero() -> None:
    workflow = build_qwen_txt2img_workflow(prompt="NASA launch pad", seed=None)

    assert workflow["106"]["inputs"]["seed"] == 0


def test_qwen_image_edit_workflow_defaults_null_seed_to_zero() -> None:
    workflow = build_qwen_image_edit_workflow(
        prompt="Add period documentary lighting",
        input_image_filename="input.png",
        seed=None,
    )

    assert workflow["102:3"]["inputs"]["seed"] == 0


# ---------------------------------------------------------------------------
# Qwen Image Edit 2509 (multi-ref)
# ---------------------------------------------------------------------------

def test_qwen_edit_2509_single_image_prunes_unused_refs() -> None:
    wf = build_qwen_edit_2509_workflow(
        prompt="edit", image_filenames=["a.png"], seed=42,
    )

    assert wf["10"]["inputs"]["image"] == "a.png"
    assert "11" not in wf and "12" not in wf
    for encode_node in ("110", "111"):
        assert "image2" not in wf[encode_node]["inputs"]
        assert "image3" not in wf[encode_node]["inputs"]
    # Edit mode: latent follows image1 via VAEEncode, empty latent pruned
    assert wf["3"]["inputs"]["latent_image"] == ["88", 0]
    assert "107" not in wf
    assert wf["3"]["inputs"]["seed"] == 42


def test_qwen_edit_2509_three_images_kept() -> None:
    wf = build_qwen_edit_2509_workflow(
        prompt="compose", image_filenames=["a.png", "b.png", "c.png"], seed=1,
    )

    assert wf["11"]["inputs"]["image"] == "b.png"
    assert wf["12"]["inputs"]["image"] == "c.png"
    assert wf["110"]["inputs"]["image2"] == ["11", 0]
    assert wf["110"]["inputs"]["image3"] == ["12", 0]


def test_qwen_edit_2509_generation_mode_uses_empty_latent() -> None:
    wf = build_qwen_edit_2509_workflow(
        prompt="compose", image_filenames=["a.png", "b.png"], seed=1,
        output_width=1664, output_height=928,
    )

    assert wf["3"]["inputs"]["latent_image"] == ["107", 0]
    assert wf["107"]["inputs"]["width"] == 1664
    assert wf["107"]["inputs"]["height"] == 928


def test_qwen_edit_2509_rejects_empty_and_excess_images() -> None:
    with pytest.raises(ValueError):
        build_qwen_edit_2509_workflow(prompt="x", image_filenames=[], seed=0)
    with pytest.raises(ValueError):
        build_qwen_edit_2509_workflow(
            prompt="x", image_filenames=["1", "2", "3", "4"], seed=0,
        )


# ---------------------------------------------------------------------------
# FLUX.2 Klein
# ---------------------------------------------------------------------------

def test_flux2_klein_zero_refs_is_plain_txt2img() -> None:
    wf = build_flux2_klein_workflow(prompt="a cat", width=1344, height=768, seed=5)

    # CFGGuider wired directly to the encoders
    assert wf["76"]["inputs"]["positive"] == ["74", 0]
    assert wf["76"]["inputs"]["negative"] == ["82", 0]
    # All ref machinery pruned
    for node_id in ("20", "30", "40", "50", "60", "23", "33", "43", "53", "63"):
        assert node_id not in wf
    assert wf["66"]["inputs"]["width"] == 1344
    assert wf["67"]["inputs"]["height"] == 768
    assert wf["73"]["inputs"]["noise_seed"] == 5


def test_flux2_klein_two_refs_rewires_guider_to_chain_end() -> None:
    wf = build_flux2_klein_workflow(
        prompt="a cat", seed=5,
        reference_image_filenames=["r1.png", "r2.png"],
    )

    assert wf["20"]["inputs"]["image"] == "r1.png"
    assert wf["21"]["inputs"]["image"] == "r2.png"
    assert "22" not in wf and "23" not in wf
    # Positive chain ends at ref-2 ReferenceLatent (51), negative at 61
    assert wf["76"]["inputs"]["positive"] == ["51", 0]
    assert wf["76"]["inputs"]["negative"] == ["61", 0]
    # Chain integrity: 51 consumes 50, which consumes the text encoder
    assert wf["51"]["inputs"]["conditioning"] == ["50", 0]
    assert wf["50"]["inputs"]["conditioning"] == ["74", 0]


def test_flux2_klein_rejects_more_than_four_refs() -> None:
    with pytest.raises(ValueError):
        build_flux2_klein_workflow(
            prompt="x", seed=0,
            reference_image_filenames=["1", "2", "3", "4", "5"],
        )


# ---------------------------------------------------------------------------
# LTX-2.3 FLF2V
# ---------------------------------------------------------------------------

def test_ltx_frames_for_duration() -> None:
    assert ltx_frames_for_duration(4) == 101
    assert ltx_frames_for_duration(6) == 151
    assert ltx_frames_for_duration(10) == 251


def test_ltx_flf2v_with_end_frame() -> None:
    wf = build_ltx23_flf2v_workflow(
        prompt="pan", negative_prompt="bad",
        start_keyframe_filename="s.png", end_keyframe_filename="e.png",
        width=1280, height=720, frames=151, seed=7,
    )

    assert wf["4"]["inputs"]["text"] == "pan"
    assert wf["13"]["inputs"]["image"] == "e.png"
    assert wf["16"]["inputs"]["length"] == 151
    assert wf["19"]["inputs"]["frames_number"] == 151
    assert wf["17"]["inputs"]["frame_idx"] == 0
    assert wf["18"]["inputs"]["frame_idx"] == -1
    # Audio muxed by default
    assert wf["30"]["inputs"]["audio"] == ["29", 0]


def test_ltx_flf2v_without_end_frame_splices_guide_chain() -> None:
    wf = build_ltx23_flf2v_workflow(
        prompt="pan", negative_prompt="bad",
        start_keyframe_filename="s.png", seed=7,
    )

    for node_id in ("13", "15", "18"):
        assert node_id not in wf
    # Downstream consumers rewired to the first guide
    assert wf["20"]["inputs"]["video_latent"] == ["17", 2]
    assert wf["22"]["inputs"]["positive"] == ["17", 0]
    assert wf["22"]["inputs"]["negative"] == ["17", 1]
    assert wf["27"]["inputs"]["positive"] == ["17", 0]


def test_ltx_flf2v_audio_disabled_drops_audio_mux() -> None:
    wf = build_ltx23_flf2v_workflow(
        prompt="pan", negative_prompt="bad",
        start_keyframe_filename="s.png", end_keyframe_filename="e.png",
        seed=7, generate_audio=False,
    )

    assert "audio" not in wf["30"]["inputs"]
    assert "29" not in wf


# ---------------------------------------------------------------------------
# Seedance 2.0 FLF2V
# ---------------------------------------------------------------------------

def test_seedance_flf2v_injects_dotted_dynamic_inputs() -> None:
    wf = build_seedance2_flf2v_workflow(
        prompt="drone shot", first_frame_filename="f.png",
        last_frame_filename="l.png", duration=8, resolution="720p",
        aspect_ratio="9:16", seed=9, generate_audio=True,
    )

    node = wf["3"]["inputs"]
    assert node["model"] == "Seedance 2.0"
    assert node["model.prompt"] == "drone shot"
    assert node["model.resolution"] == "720p"
    assert node["model.ratio"] == "9:16"
    assert node["model.duration"] == 8
    assert node["model.generate_audio"] is True
    assert node["seed"] == 9
    assert node["first_frame"] == ["1", 0]
    assert node["last_frame"] == ["2", 0]
    assert wf["2"]["inputs"]["image"] == "l.png"


def test_seedance_flf2v_without_last_frame() -> None:
    wf = build_seedance2_flf2v_workflow(
        prompt="x", first_frame_filename="f.png", duration=4, seed=0,
    )

    assert "2" not in wf
    assert "last_frame" not in wf["3"]["inputs"]


def test_seedance_flf2v_validates_duration_and_ratio() -> None:
    with pytest.raises(ValueError):
        build_seedance2_flf2v_workflow(
            prompt="x", first_frame_filename="f.png", duration=3,
        )
    with pytest.raises(ValueError):
        build_seedance2_flf2v_workflow(
            prompt="x", first_frame_filename="f.png", duration=16,
        )
    with pytest.raises(ValueError):
        build_seedance2_flf2v_workflow(
            prompt="x", first_frame_filename="f.png", duration=5,
            aspect_ratio="2:1",
        )
