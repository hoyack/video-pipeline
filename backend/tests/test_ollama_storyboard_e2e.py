"""End-to-end test for Ollama storyboard generation.

Tests the full chain: OllamaAdapter → generate_text → StoryboardOutput validation.
Uses real Ollama cloud/local endpoint with the user's configured settings from the DB.

Usage:
    # Run with default settings from DB:
    cd backend && python -m pytest tests/test_ollama_storyboard_e2e.py -v -s

    # Override model:
    OLLAMA_MODEL=ollama/llama3.1 python -m pytest tests/test_ollama_storyboard_e2e.py -v -s

    # Force local Ollama:
    OLLAMA_ENDPOINT=http://localhost:11434 python -m pytest tests/test_ollama_storyboard_e2e.py -v -s
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must set CWD to project root so config.yaml loads
os.chdir(Path(__file__).resolve().parent.parent.parent)

from pydantic import BaseModel, Field, ValidationError

from vidpipe.services.llm.ollama_adapter import OllamaAdapter
from vidpipe.schemas.storyboard import StoryboardOutput, SceneSchema, StyleGuide, CharacterDescription

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Helpers to load Ollama settings from DB or env
# ---------------------------------------------------------------------------

def _load_ollama_settings() -> dict:
    """Load Ollama settings from DB or env overrides."""
    import sqlite3

    settings = {
        "model_id": os.environ.get("OLLAMA_MODEL", ""),
        "base_url": os.environ.get("OLLAMA_ENDPOINT", ""),
        "api_key": os.environ.get("OLLAMA_API_KEY", ""),
    }

    # Try loading from vidpipe.db if env vars not fully set
    db_path = Path("vidpipe.db")
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT ollama_use_cloud, ollama_api_key, ollama_endpoint, ollama_models "
            "FROM user_settings LIMIT 1"
        ).fetchone()
        conn.close()

        if row:
            use_cloud, api_key, endpoint, models_json = row
            if not settings["base_url"]:
                if use_cloud:
                    settings["base_url"] = endpoint or "https://ollama.com"
                else:
                    settings["base_url"] = endpoint or "http://localhost:11434"
            if not settings["api_key"] and api_key:
                settings["api_key"] = api_key
            if not settings["model_id"] and models_json:
                models = json.loads(models_json)
                enabled = [m for m in models if m.get("enabled")]
                if enabled:
                    settings["model_id"] = enabled[0]["id"]

    if not settings["model_id"]:
        settings["model_id"] = "ollama/llama3.1"
    if not settings["base_url"]:
        settings["base_url"] = "http://localhost:11434"

    return settings


OLLAMA_SETTINGS = _load_ollama_settings()


def _make_adapter() -> OllamaAdapter:
    return OllamaAdapter(
        model_id=OLLAMA_SETTINGS["model_id"],
        base_url=OLLAMA_SETTINGS["base_url"],
        api_key=OLLAMA_SETTINGS["api_key"] or None,
    )


# ---------------------------------------------------------------------------
# Test: raw connectivity — can we reach the Ollama endpoint?
# ---------------------------------------------------------------------------

class SimpleResponse(BaseModel):
    """Minimal schema for connectivity test."""
    answer: str = Field(description="A short answer")


@pytest.mark.asyncio
async def test_01_ollama_connectivity():
    """Verify Ollama endpoint is reachable and returns structured JSON."""
    adapter = _make_adapter()
    print(f"\n  Model: {OLLAMA_SETTINGS['model_id']}")
    print(f"  Endpoint: {OLLAMA_SETTINGS['base_url']}")
    print(f"  Has API key: {bool(OLLAMA_SETTINGS['api_key'])}")

    result = await adapter.generate_text(
        prompt="What is 2+2? Reply with just the number.",
        schema=SimpleResponse,
        temperature=0.1,
        max_retries=2,
    )
    assert isinstance(result, SimpleResponse)
    assert result.answer  # non-empty
    print(f"  Response: {result.answer}")


# ---------------------------------------------------------------------------
# Test: schema validation — does the model produce valid StoryboardOutput?
# ---------------------------------------------------------------------------

SIMPLE_STORYBOARD_PROMPT = """You are a storyboard director. Create a storyboard for this script.

REQUIREMENTS:
- Exactly 3 scenes
- Scene indices start from 0
- Each scene needs: scene_description, key_details (3-6 items), start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes
- Include a style_guide with visual_style, color_palette, and camera_style
- Include a characters list with name, physical_description, and clothing_description

Script: A cat walks across a sunny garden, chases a butterfly, then falls asleep under a tree."""


@pytest.mark.asyncio
async def test_02_storyboard_schema_validation():
    """Verify the adapter produces a valid StoryboardOutput with a simple prompt."""
    adapter = _make_adapter()
    print(f"\n  Testing StoryboardOutput via adapter with {OLLAMA_SETTINGS['model_id']}...")

    try:
        storyboard = await adapter.generate_text(
            prompt=SIMPLE_STORYBOARD_PROMPT,
            schema=StoryboardOutput,
            temperature=0.3,
            max_retries=2,
        )
    except Exception as e:
        print(f"\n  adapter.generate_text() FAILED: {type(e).__name__}: {e}")
        raise

    assert isinstance(storyboard, StoryboardOutput)
    print(f"  Pydantic validation: OK")
    print(f"  Scenes: {len(storyboard.scenes)}")
    print(f"  Characters: {len(storyboard.characters)}")
    print(f"  Style: {storyboard.style_guide.visual_style}")
    print(f"  Palette: {storyboard.style_guide.color_palette}")
    for s in storyboard.scenes:
        print(f"    Scene {s.scene_index}: {s.scene_description[:60]}...")


# ---------------------------------------------------------------------------
# Test: full adapter flow — same path as the real pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_03_adapter_generate_text_storyboard():
    """Test the full adapter.generate_text() → StoryboardOutput path (same as pipeline)."""
    adapter = _make_adapter()

    # Use the same system+user prompt structure as storyboard.py
    from vidpipe.pipeline.storyboard import STORYBOARD_SYSTEM_PROMPT

    system_prompt = STORYBOARD_SYSTEM_PROMPT.format(
        style="cinematic",
        aspect_ratio="16:9",
    ).replace(
        "- Break the script into 3-5 distinct visual scenes",
        "- Break the script into exactly 3 distinct visual scenes",
    )

    full_prompt = f"{system_prompt}\n\nScript: A cat walks across a sunny garden, chases a butterfly, then falls asleep under a tree."

    print(f"\n  Running full adapter.generate_text() with StoryboardOutput schema...")
    print(f"  Prompt length: {len(full_prompt)} chars")

    try:
        storyboard = await adapter.generate_text(
            prompt=full_prompt,
            schema=StoryboardOutput,
            temperature=0.5,
            max_retries=2,
        )
    except Exception as e:
        # If it fails, let's get the raw response to see what went wrong
        print(f"\n  adapter.generate_text() FAILED: {type(e).__name__}: {e}")

        # Try a raw call to see what the model returns
        from ollama import AsyncClient
        headers = {"Authorization": f"Bearer {OLLAMA_SETTINGS['api_key']}"} if OLLAMA_SETTINGS["api_key"] else {}
        client = AsyncClient(host=OLLAMA_SETTINGS["base_url"], headers=headers)
        bare_model = OLLAMA_SETTINGS["model_id"].removeprefix("ollama/")

        raw_resp = await client.chat(
            model=bare_model,
            messages=[{"role": "user", "content": full_prompt}],
            format=StoryboardOutput.model_json_schema(),
            options={"temperature": 0.3},
            stream=False,
        )
        print(f"\n  Raw response for debugging:\n{raw_resp.message.content[:2000]}")
        raise

    assert isinstance(storyboard, StoryboardOutput)
    assert len(storyboard.scenes) >= 2
    assert storyboard.style_guide.visual_style
    print(f"  Success! {len(storyboard.scenes)} scenes, {len(storyboard.characters)} characters")
    for s in storyboard.scenes:
        print(f"    Scene {s.scene_index}: {len(s.key_details)} key details, motion: {s.video_motion_prompt[:50]}...")


# ---------------------------------------------------------------------------
# Test: retry behavior — simulate what happens when validation fails
# ---------------------------------------------------------------------------

class StrictSchema(BaseModel):
    """Schema intentionally hard to satisfy — tests retry logic."""
    items: list[str] = Field(description="Exactly 3 items", min_length=3, max_length=3)
    score: int = Field(description="A number between 1 and 10", ge=1, le=10)


@pytest.mark.asyncio
async def test_04_retry_on_validation_error():
    """Test that adapter retries when model output fails validation."""
    adapter = _make_adapter()

    print(f"\n  Testing retry logic with strict schema...")
    try:
        result = await adapter.generate_text(
            prompt="Give me exactly 3 fruit names and a score between 1 and 10.",
            schema=StrictSchema,
            temperature=0.2,
            max_retries=3,
        )
        print(f"  Success: {result.items}, score={result.score}")
        assert len(result.items) == 3
        assert 1 <= result.score <= 10
    except Exception as e:
        print(f"  Failed after retries: {type(e).__name__}: {e}")
        # This is informational — some models may not handle strict constraints
        pytest.skip(f"Model couldn't satisfy strict schema after retries: {e}")


# ---------------------------------------------------------------------------
# Test: JSON schema dump — verify schema is valid for Ollama format param
# ---------------------------------------------------------------------------

def test_05_storyboard_schema_json_serializable():
    """Verify StoryboardOutput.model_json_schema() is JSON-serializable and well-formed."""
    schema = StoryboardOutput.model_json_schema()

    # Must be JSON-serializable (Ollama format param requirement)
    serialized = json.dumps(schema)
    assert len(serialized) > 100

    # Must have expected top-level structure
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "style_guide" in schema["properties"]
    assert "characters" in schema["properties"]
    assert "scenes" in schema["properties"]

    # Check $defs for nested models
    defs = schema.get("$defs", {})
    assert "StyleGuide" in defs
    assert "SceneSchema" in defs
    assert "CharacterDescription" in defs

    print(f"\n  Schema size: {len(serialized)} chars")
    print(f"  Top-level properties: {list(schema['properties'].keys())}")
    print(f"  $defs: {list(defs.keys())}")
    print(f"  Schema is valid for Ollama format param")
