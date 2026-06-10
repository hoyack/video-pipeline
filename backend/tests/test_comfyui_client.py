from vidpipe.services.comfyui_client import (
    build_qwen_image_edit_workflow,
    build_qwen_txt2img_workflow,
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
