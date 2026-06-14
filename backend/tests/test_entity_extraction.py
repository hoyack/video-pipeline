"""Tests for post-generation entity extraction wiring."""

import uuid
from types import SimpleNamespace

import pytest

from vidpipe.services.entity_extraction import extract_and_register_new_entities


@pytest.mark.asyncio
async def test_entity_extraction_uses_injected_vision_adapter(monkeypatch):
    captured = {}
    adapter = SimpleNamespace(name="ollama-vision-adapter")

    class FakeReversePromptService:
        def __init__(self, vision_adapter=None):
            captured["vision_adapter"] = vision_adapter

    monkeypatch.setattr(
        "vidpipe.services.entity_extraction.ReversePromptService",
        FakeReversePromptService,
    )

    registered = await extract_and_register_new_entities(
        session=SimpleNamespace(),
        scene_id=uuid.uuid4(),
        manifest_id=uuid.uuid4(),
        shot_index=0,
        new_entities=[],
        vision_adapter=adapter,
    )

    assert registered == []
    assert captured["vision_adapter"] is adapter
