"""Opt-in cloud E2E for a short historical NASA documentary production.

Run only when you intentionally want to spend provider credits:

    VIDPIPE_RUN_NASA_E2E=1 pytest backend/tests/test_nasa_documentary_e2e.py -q -s
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from math import ceil
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("VIDPIPE_RUN_NASA_E2E") != "1",
    reason="Set VIDPIPE_RUN_NASA_E2E=1 to run the paid cloud NASA documentary E2E.",
)

API_BASE = os.getenv("VIDPIPE_E2E_API_BASE", "http://localhost:8100").rstrip("/")
TEXT_MODEL = os.getenv("VIDPIPE_NASA_E2E_TEXT_MODEL", "ollama/kimi-k2.5:cloud")
VISION_MODEL = os.getenv("VIDPIPE_NASA_E2E_VISION_MODEL", "ollama/kimi-k2.5:cloud")
IMAGE_MODEL = os.getenv("VIDPIPE_NASA_E2E_IMAGE_MODEL", "gemini-2.5-flash-image")
VIDEO_MODEL = os.getenv("VIDPIPE_NASA_E2E_VIDEO_MODEL", "wan-2.2-i2v")
SCENE_TIMEOUT_SECONDS = int(os.getenv("VIDPIPE_NASA_E2E_SCENE_TIMEOUT", "3600"))
SCENE_COUNT = int(os.getenv("VIDPIPE_NASA_E2E_SCENE_COUNT", "2"))
CLIP_DURATION_SECONDS = int(os.getenv("VIDPIPE_NASA_E2E_CLIP_DURATION", "5"))
TARGET_DURATION_SECONDS = int(os.getenv("VIDPIPE_NASA_E2E_TARGET_DURATION", "0"))
SHOTS_PER_SCENE = int(
    os.getenv(
        "VIDPIPE_NASA_E2E_SHOTS_PER_SCENE",
        str(max(2, ceil(TARGET_DURATION_SECONDS / max(1, SCENE_COUNT * CLIP_DURATION_SECONDS))))
        if TARGET_DURATION_SECONDS
        else "2",
    )
)

REFERENCE_IMAGES = {
    "katherine_johnson.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Katherine%20Johnson%20at%20NASA,%20in%201966.jpg",
    "john_glenn.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/John-Glenn-NASA-portrait.jpg",
    "wernher_von_braun.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Wernher-von-Braun.jpg",
    "mercury_control.jpg": "https://commons.wikimedia.org/wiki/Special:FilePath/Mercury%20Control.jpg",
}


def _api(path: str) -> str:
    return f"{API_BASE}/api{path}"


def _assert_ok(response: httpx.Response) -> dict:
    assert response.status_code < 400, f"{response.request.method} {response.url}: {response.text[:1000]}"
    if response.content:
        return response.json()
    return {}


def _preflight(client: httpx.Client) -> dict:
    settings = _assert_ok(client.get(_api("/settings")))
    missing = []
    if not settings.get("has_comfyui_key") or not settings.get("comfyui_host"):
        missing.append("ComfyUI host/API key")
    if not settings.get("ollama_use_cloud") or not settings.get("has_ollama_key"):
        missing.append("Ollama cloud/API key")
    if not settings.get("has_elevenlabs_key") or not settings.get("default_voice_id"):
        missing.append("ElevenLabs key/default voice")
    ollama_models = settings.get("ollama_models") or []
    kimi_enabled = any(model.get("id") == TEXT_MODEL and model.get("enabled") for model in ollama_models)
    if not kimi_enabled:
        missing.append(f"enabled Ollama model {TEXT_MODEL}")
    if missing:
        pytest.skip("Missing E2E prerequisites: " + ", ".join(missing))
    return settings


def _download_references(tmp_path: Path) -> dict[str, Path]:
    refs: dict[str, Path] = {}
    headers = {
        "User-Agent": "vidpipe-nasa-documentary-e2e/1.0 (local integration test)",
    }
    with httpx.Client(timeout=120, follow_redirects=True, headers=headers) as client:
        for filename, url in REFERENCE_IMAGES.items():
            path = tmp_path / filename
            response = client.get(url)
            if response.status_code == 403 and shutil.which("curl"):
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        "-sS",
                        "-A",
                        "Mozilla/5.0 vidpipe-nasa-documentary-e2e",
                        "-o",
                        str(path),
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                response.raise_for_status()
                path.write_bytes(response.content)
            assert path.exists() and path.stat().st_size > 1024, f"Downloaded reference is empty: {url}"
            refs[filename] = path
    return refs


def _upload_file(client: httpx.Client, url: str, path: Path) -> dict:
    with path.open("rb") as handle:
        return _assert_ok(
            client.post(
                url,
                files={"file": (path.name, handle, "image/jpeg")},
                timeout=180,
            )
        )


def _create_actor(
    client: httpx.Client,
    *,
    name: str,
    tag: str,
    bible_id: str,
    image_path: Path,
    voice_id: str | None,
    wardrobe: str,
) -> str:
    actor = _assert_ok(client.post(
        _api("/asset-library/actors"),
        json={
            "name": name,
            "description": f"Historical documentary reference for {name}.",
            "base_appearance_prompt": f"Respectful historical likeness of {name}, archival NASA documentary style.",
            "prompt_tags": ["historical", "NASA", "documentary"],
        },
    ))
    actor_id = actor["id"]
    _upload_file(client, _api(f"/asset-library/actors/{actor_id}/refs"), image_path)
    voice_profile_id = None
    if voice_id:
        voice = _assert_ok(client.post(
            _api(f"/asset-library/actors/{actor_id}/voice-profiles"),
            json={
                "voice_id": voice_id,
                "adapter_type": "ELEVENLABS",
                "style_notes": "Measured documentary delivery, clear and historically grounded.",
            },
        ))
        voice_profile_id = voice["id"]
    wardrobe_preset = _assert_ok(client.post(
        _api(f"/asset-library/actors/{actor_id}/wardrobe-presets"),
        json={
            "label": wardrobe,
            "description": f"{wardrobe}; accurate early-1960s NASA documentary wardrobe.",
        },
    ))
    binding = _assert_ok(client.post(
        _api(f"/production-bibles/{bible_id}/cast"),
        json={
            "actor_id": actor_id,
            "tag": tag,
            "character_name": name,
            "character_description": f"{name} as represented in an early NASA documentary.",
            "role": "SUPPORTING" if tag != "NARRATOR" else "NARRATOR",
            "voice_profile_id": voice_profile_id,
            "prompt_tags": [tag, "historical_figure"],
        },
    ))
    _assert_ok(client.post(
        _api(f"/cast-bindings/{binding['id']}/looks"),
        json={
            "wardrobe_preset_id": wardrobe_preset["id"],
            "tag": f"{tag}_LOOK",
            "is_default": True,
        },
    ))
    return binding["id"]


def _seed_screenplay(client: httpx.Client, production_id: str) -> dict:
    _assert_ok(client.get(_api(f"/productions/{production_id}/screenplay")))
    screenplay = _assert_ok(client.put(
        _api(f"/productions/{production_id}/screenplay"),
        json={
            "title": "Before the Moon: NASA Learns to Fly",
            "genre": "Historical documentary",
            "logline": "A concise documentary follows the early NASA teams who turned orbital flight from an engineering gamble into a national program.",
            "treatment": (
                "The film opens inside Mercury Control, where mathematicians and flight controllers "
                "turn telemetry into decisions. It then moves to the launch program, connecting "
                "John Glenn's orbital mission with the engineering culture forming around NASA."
            ),
            "character_breakdowns": [
                {"tag": "NARRATOR", "name": "Narrator", "role": "historical guide"},
                {"tag": "KATHERINE_JOHNSON", "name": "Katherine Johnson", "role": "NASA mathematician"},
                {"tag": "JOHN_GLENN", "name": "John Glenn", "role": "Mercury astronaut"},
                {"tag": "WERNHER_VON_BRAUN", "name": "Wernher von Braun", "role": "rocket engineer"},
            ],
            "scene_breakdown": [
                {
                    "scene_number": 1,
                    "slugline": "INT. MERCURY CONTROL - DAY",
                    "intent": "Mercury Control hums with consoles, tracking charts, and careful human calculation.",
                    "characters_present": ["KATHERINE_JOHNSON", "NARRATOR"],
                    "set_ref": "MERCURY_CONTROL",
                    "props_required": ["MERCURY_CAPSULE"],
                    "emotional_beat": "Precision and confidence under pressure.",
                },
                {
                    "scene_number": 2,
                    "slugline": "EXT. CAPE CANAVERAL LAUNCH COMPLEX - DAWN",
                    "intent": "The Mercury launch stack stands ready while engineers and astronauts prepare for orbital flight.",
                    "characters_present": ["JOHN_GLENN", "WERNHER_VON_BRAUN", "NARRATOR"],
                    "set_ref": "MERCURY_CONTROL",
                    "props_required": ["MERCURY_CAPSULE"],
                    "emotional_beat": "Ambition and disciplined risk at the edge of launch.",
                },
            ],
            "script": (
                "NARRATOR: Before Apollo, NASA learned to trust numbers, teams, and disciplined risk.\n"
                "NARRATOR: In Mercury Control, every console depended on people who could turn uncertainty into a flight decision.\n"
                "NARRATOR: The early launches proved that orbital flight was not a single heroic act, but an industrial collaboration."
            ),
            "shot_list": [
                {"scene_number": 1, "shot_number": 1, "description": "Wide view of Mercury Control consoles and wall map."},
                {"scene_number": 1, "shot_number": 2, "description": "Close documentary insert of calculations and headset operators."},
                {"scene_number": 2, "shot_number": 1, "description": "The launch vehicle on the pad at dawn."},
                {"scene_number": 2, "shot_number": 2, "description": "Engineers and astronaut preparation in archival NASA style."},
            ],
            "text_model": TEXT_MODEL,
        },
    ))
    return screenplay


def _wait_for_scene(client: httpx.Client, scene_id: str) -> dict:
    deadline = time.monotonic() + SCENE_TIMEOUT_SECONDS
    last = {}
    while time.monotonic() < deadline:
        last = _assert_ok(client.get(_api(f"/scenes/{scene_id}"), timeout=60))
        if last["status"] == "complete":
            return last
        if last["status"] in {"failed", "stopped"}:
            raise AssertionError(f"Scene {scene_id} ended as {last['status']}: {last.get('error_message')}")
        time.sleep(20)
    raise AssertionError(f"Timed out waiting for scene {scene_id}; last state: {last}")


def _assert_master_has_audio(path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required to validate the master video")
    streams = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "codec_type=video" in streams
    assert "codec_type=audio" in streams
    volume = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    assert "mean_volume: -inf" not in volume
    assert "max_volume: -inf" not in volume


def test_nasa_documentary_cloud_e2e(tmp_path):
    refs = _download_references(tmp_path)
    with httpx.Client(timeout=httpx.Timeout(60, read=600), follow_redirects=True) as client:
        settings = _preflight(client)
        default_voice_id = settings["default_voice_id"]

        bible = _assert_ok(client.post(
            _api("/production-bibles"),
            json={
                "name": f"NASA Documentary E2E {int(time.time())}",
                "description": "Historical documentary bible for early NASA, seeded by the cloud E2E test.",
                "category": "FULL_PRODUCTION",
                "tags": ["NASA", "Mercury", "historical", "e2e"],
            },
        ))
        bible_id = bible["production_bible_id"]
        production = _assert_ok(client.post(
            _api("/productions"),
            json={
                "name": f"NASA Documentary E2E {int(time.time())}",
                "description": (
                    f"Opt-in cloud E2E production using {TEXT_MODEL}, {IMAGE_MODEL}, "
                    f"{VIDEO_MODEL}, and ElevenLabs."
                ),
                "tags": ["e2e", "nasa", "documentary"],
            },
        ))
        production_id = production["id"]
        _assert_ok(client.put(
            _api(f"/productions/{production_id}"),
            json={"production_bible_id": bible_id},
        ))

        _create_actor(
            client,
            name="Narrator",
            tag="NARRATOR",
            bible_id=bible_id,
            image_path=refs["john_glenn.jpg"],
            voice_id=default_voice_id,
            wardrobe="Neutral documentary narrator",
        )
        _create_actor(
            client,
            name="Katherine Johnson",
            tag="KATHERINE_JOHNSON",
            bible_id=bible_id,
            image_path=refs["katherine_johnson.jpg"],
            voice_id=None,
            wardrobe="1960s NASA office attire",
        )
        _create_actor(
            client,
            name="John Glenn",
            tag="JOHN_GLENN",
            bible_id=bible_id,
            image_path=refs["john_glenn.jpg"],
            voice_id=None,
            wardrobe="Mercury astronaut pressure suit",
        )
        _create_actor(
            client,
            name="Wernher von Braun",
            tag="WERNHER_VON_BRAUN",
            bible_id=bible_id,
            image_path=refs["wernher_von_braun.jpg"],
            voice_id=None,
            wardrobe="Early-1960s engineering suit",
        )

        mission_control = _assert_ok(client.post(
            _api("/asset-library/sets"),
            json={
                "name": "Mercury Control Center",
                "description": "Early-1960s NASA control room with consoles, headsets, wall map, and telemetry.",
                "prompt_tags": ["MERCURY_CONTROL", "NASA_SET"],
                "lighting_notes": "Archival fluorescent interior documentary light.",
            },
        ))
        _upload_file(client, _api(f"/asset-library/sets/{mission_control['id']}/refs"), refs["mercury_control.jpg"])
        _assert_ok(client.post(
            _api(f"/production-bibles/{bible_id}/set-bindings"),
            json={
                "library_set_id": mission_control["id"],
                "tag": "MERCURY_CONTROL",
                "production_name": "Mercury Control Center",
            },
        ))
        capsule = _assert_ok(client.post(
            _api("/asset-library/props"),
            json={
                "name": "Mercury Capsule",
                "description": "Small early NASA orbital spacecraft capsule.",
                "appearance_prompt": "1960s Mercury spacecraft capsule, documentary archival detail.",
                "prompt_tags": ["MERCURY_CAPSULE"],
            },
        ))
        _assert_ok(client.post(
            _api(f"/production-bibles/{bible_id}/prop-bindings"),
            json={
                "library_prop_id": capsule["id"],
                "tag": "MERCURY_CAPSULE",
                "production_name": "Mercury capsule",
            },
        ))
        _assert_ok(client.post(_api(f"/production-bibles/{bible_id}/finalize")))

        screenplay = _seed_screenplay(client, production_id)
        assert screenplay["text_model"] == TEXT_MODEL
        _assert_ok(client.patch(_api(f"/productions/{production_id}/screenplay/status"), json={"status": "LOCKED"}))
        created_scenes = _assert_ok(client.post(
            _api(f"/productions/{production_id}/screenplay/generate-scenes?force=true"),
        ))
        assert len(created_scenes) >= 2

        completed_scenes = []
        for scene in created_scenes[:SCENE_COUNT]:
            scene_id = scene["scene_id"]
            _assert_ok(client.patch(
                _api(f"/scenes/{scene_id}/edit"),
                json={
                    "clip_duration": CLIP_DURATION_SECONDS,
                    "target_shot_count": SHOTS_PER_SCENE,
                    "text_model": TEXT_MODEL,
                    "vision_model": VISION_MODEL,
                    "image_model": IMAGE_MODEL,
                    "video_model": VIDEO_MODEL,
                    "audio_enabled": False,
                    "production_bible_id": bible_id,
                    "commit_message": "Configure NASA E2E models and shot count",
                },
            ))
            _assert_ok(client.post(
                _api(f"/scenes/{scene_id}/regenerate"),
                json={
                    "scope": "all_phases",
                    "text_model": TEXT_MODEL,
                    "image_model": IMAGE_MODEL,
                    "video_model": VIDEO_MODEL,
                },
                timeout=120,
            ))
            completed_scenes.append(_wait_for_scene(client, scene_id))

        assert len(completed_scenes) == SCENE_COUNT
        assert all(scene["shot_count"] >= SHOTS_PER_SCENE for scene in completed_scenes)

        voice_script = _assert_ok(client.post(
            _api(f"/productions/{production_id}/voice-script/generate"),
            json={"text_model": TEXT_MODEL},
            timeout=300,
        ))["voice_script"]
        _assert_ok(client.post(_api(f"/voice-scripts/{voice_script['id']}/resolve-bindings")))
        voice_script = _assert_ok(client.post(
            _api(f"/voice-scripts/{voice_script['id']}/generate-audio"),
            timeout=600,
        ))["voice_script"]
        assert any(line["generation_status"] == "READY" for line in voice_script["lines"])
        voice_script = _assert_ok(client.post(_api(f"/voice-scripts/{voice_script['id']}/mix")))["voice_script"]
        assert any(artifact["status"] == "READY" for artifact in voice_script["mix_artifacts"])

        sound_deck = _assert_ok(client.post(
            _api(f"/productions/{production_id}/sound-deck/generate"),
            json={"text_model": TEXT_MODEL},
            timeout=300,
        ))["sound_deck"]
        assert len(sound_deck["cues"]) >= SCENE_COUNT
        assert all(cue["generation_status"] == "PENDING" for cue in sound_deck["cues"])

        sound_deck = _assert_ok(client.post(
            _api(f"/productions/{production_id}/sound-deck/generate-audio"),
            timeout=900,
        ))["sound_deck"]
        ready_sound_cues = [cue for cue in sound_deck["cues"] if cue["generation_status"] == "READY"]
        assert len(ready_sound_cues) >= SCENE_COUNT
        assert all(cue["audio_url"] for cue in ready_sound_cues)

        sound_deck = _assert_ok(client.post(
            _api(f"/productions/{production_id}/sound-deck/mix"),
            timeout=600,
        ))["sound_deck"]
        ready_sfx_stems = [
            artifact
            for artifact in sound_deck["mix_artifacts"]
            if artifact["artifact_type"] == "SCENE_SFX_STEM" and artifact["status"] == "READY"
        ]
        assert len(ready_sfx_stems) >= SCENE_COUNT
        assert all(artifact["audio_url"] for artifact in ready_sfx_stems)

        master = _assert_ok(client.post(_api(f"/productions/{production_id}/render-master"), timeout=900))
        assert master["scene_count"] >= SCENE_COUNT
        assert master["audio_stem_count"] >= len(ready_sfx_stems) + 1
        if TARGET_DURATION_SECONDS:
            assert master["duration_seconds"] >= TARGET_DURATION_SECONDS - CLIP_DURATION_SECONDS

        video_response = client.get(f"{API_BASE}{master['video_url']}", timeout=900)
        assert video_response.status_code == 200
        master_path = tmp_path / "nasa_documentary_master.mp4"
        master_path.write_bytes(video_response.content)
        _assert_master_has_audio(master_path)
