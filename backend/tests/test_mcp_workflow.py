"""Tests for the Vidpipe MCP workflow layer."""

import pytest

from vidpipe.mcp.server import mcp
from vidpipe.mcp.workflow import (
    APIRequestSpec,
    ProductionSpec,
    _default_scene_breakdown,
    _list_scenes_for_production,
    normalize_api_path,
)


def test_default_scene_breakdown_keeps_voice_only_narrator_off_screen():
    spec = ProductionSpec(
        title="Signal",
        story_brief="A courier follows a pirate broadcast.",
        scene_count=2,
    )

    scenes = _default_scene_breakdown(spec, source_context=None)

    assert len(scenes) == 2
    assert all(scene.characters_present == [] for scene in scenes)


def test_default_scene_breakdown_reuses_source_visible_character():
    spec = ProductionSpec(
        title="Signal II",
        story_brief="The courier carries the broadcast into the city core.",
        scene_count=1,
    )
    source_context = {
        "title": "Signal",
        "character_breakdowns": [
            {"tag": "NARRATOR", "name": "Narrator"},
            {"tag": "COURIER", "name": "Courier"},
        ],
    }

    scenes = _default_scene_breakdown(spec, source_context=source_context)

    assert scenes[0].characters_present == ["COURIER"]
    assert "Continue from Signal" in scenes[0].intent


@pytest.mark.asyncio
async def test_list_scenes_for_production_paginates_and_sorts():
    class FakeAPI:
        async def get(self, path, **kwargs):
            assert path == "/scenes"
            page = kwargs["params"]["page"]
            pages = {
                1: {
                    "items": [
                        {
                            "scene_id": "other",
                            "production_id": "different",
                            "scene_order": 0,
                        },
                        {
                            "scene_id": "second",
                            "production_id": "target",
                            "scene_order": 2,
                        },
                    ],
                    "total": 97,
                },
                2: {
                    "items": [
                        {
                            "scene_id": "first",
                            "production_id": "target",
                            "scene_order": 1,
                        },
                    ],
                    "total": 97,
                },
            }
            return pages[page]

    scenes = await _list_scenes_for_production(FakeAPI(), "target")

    assert [scene["scene_id"] for scene in scenes] == ["first", "second"]


@pytest.mark.asyncio
async def test_mcp_tool_schemas_do_not_expose_context_parameter():
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert set(tools) == {
        "vidpipe_api_catalog",
        "vidpipe_api_request",
        "vidpipe_preflight",
        "vidpipe_project_status",
        "vidpipe_produce_project",
        "vidpipe_continue_production",
    }
    produce_schema = tools["vidpipe_produce_project"].inputSchema
    continue_schema = tools["vidpipe_continue_production"].inputSchema

    assert produce_schema["required"] == ["title", "story_brief"]
    assert continue_schema["required"] == [
        "source_production_id",
        "sequel_title",
        "continuation_brief",
    ]
    assert "ctx" not in produce_schema["properties"]
    assert "ctx" not in continue_schema["properties"]


def test_api_request_spec_normalizes_api_paths():
    assert normalize_api_path("settings") == "/settings"
    assert normalize_api_path("/api/settings") == "/settings"
    assert normalize_api_path("/api/scenes/abc/status") == "/scenes/abc/status"


def test_api_request_spec_requires_mutation_acknowledgement():
    with pytest.raises(ValueError, match="allow_mutation=true"):
        APIRequestSpec(method="POST", path="/productions", json_body={"name": "Test"})

    request = APIRequestSpec(
        method="POST",
        path="/api/productions",
        json_body={"name": "Test"},
        allow_mutation=True,
    )

    assert request.method == "POST"
    assert request.path == "/productions"


def test_api_request_spec_rejects_mixed_json_and_multipart_payloads():
    with pytest.raises(ValueError, match="json_body cannot be combined"):
        APIRequestSpec(
            method="POST",
            path="/editor-images",
            json_body={"name": "bad"},
            form_fields={"label": "bad"},
            allow_mutation=True,
        )
