"""Opt-in cloud E2E: a 1-minute cyberpunk short with Seedance 2.0 + FLUX.2 Klein + Gemini.

Mirrors the NASA documentary E2E flow but:
- Reuses the existing Asset Library actor "Brandon" (migrated from the legacy
  Brandon Manifest production bible) instead of creating actors from downloads.
- Cyberpunk megacity concept (Johnny Mnemonic energy): Brandon as a data
  courier crossing a neon city with stolen corporate data.
- Models: seedance-2.0-flf2v video (native audio ON), flux-2-klein keyframes,
  Gemini text + vision.
- 1-minute target: 3 scenes x 2 shots x 10s.

Run only when you intentionally want to spend provider credits:

    VIDPIPE_RUN_CYBERPUNK_E2E=1 pytest backend/tests/test_cyberpunk_seedance_e2e.py -q -s

Note: seedance-2.0-flf2v is a paid ByteDance partner node — the ComfyUI Cloud
account must have partner-node (API node) access or every video job fails with
"Unauthorized: Please login first to use this node".
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("VIDPIPE_RUN_CYBERPUNK_E2E") != "1",
    reason="Set VIDPIPE_RUN_CYBERPUNK_E2E=1 to run the paid cloud cyberpunk E2E.",
)

API_BASE = os.getenv("VIDPIPE_E2E_API_BASE", "http://localhost:8100").rstrip("/")
TEXT_MODEL = os.getenv("VIDPIPE_CYBERPUNK_E2E_TEXT_MODEL", "gemini-3.5-flash")
VISION_MODEL = os.getenv("VIDPIPE_CYBERPUNK_E2E_VISION_MODEL", "gemini-3.5-flash")
IMAGE_MODEL = os.getenv("VIDPIPE_CYBERPUNK_E2E_IMAGE_MODEL", "flux-2-klein")
VIDEO_MODEL = os.getenv("VIDPIPE_CYBERPUNK_E2E_VIDEO_MODEL", "seedance-2.0-flf2v")
ACTOR_NAME = os.getenv("VIDPIPE_CYBERPUNK_E2E_ACTOR", "Brandon")
SCENE_TIMEOUT_SECONDS = int(os.getenv("VIDPIPE_CYBERPUNK_E2E_SCENE_TIMEOUT", "3600"))
SCENE_COUNT = int(os.getenv("VIDPIPE_CYBERPUNK_E2E_SCENE_COUNT", "3"))
CLIP_DURATION_SECONDS = int(os.getenv("VIDPIPE_CYBERPUNK_E2E_CLIP_DURATION", "10"))
SHOTS_PER_SCENE = int(os.getenv("VIDPIPE_CYBERPUNK_E2E_SHOTS_PER_SCENE", "2"))
AUDIO_ENABLED = os.getenv("VIDPIPE_CYBERPUNK_E2E_AUDIO", "1") == "1"


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
    if not settings.get("has_elevenlabs_key") or not settings.get("default_voice_id"):
        missing.append("ElevenLabs key/default voice")
    assert not missing, f"Preflight failed; configure: {', '.join(missing)}"
    return settings


def _find_actor(client: httpx.Client, name: str) -> dict:
    actors = _assert_ok(client.get(_api("/asset-library/actors")))
    if isinstance(actors, dict):
        actors = actors.get("actors", actors.get("items", []))
    matches = [a for a in actors if a["name"] == name]
    assert matches, f"Actor {name!r} not found in asset library — migrate it first"
    # Newest match wins (the freshly migrated one)
    return matches[-1]


def _bind_actor(
    client: httpx.Client,
    *,
    actor_id: str,
    bible_id: str,
    tag: str,
    character_name: str,
    character_description: str,
    role: str,
    voice_profile_id: str | None,
    wardrobe_label: str,
    wardrobe_description: str,
) -> str:
    wardrobe_preset = _assert_ok(client.post(
        _api(f"/asset-library/actors/{actor_id}/wardrobe-presets"),
        json={"label": wardrobe_label, "description": wardrobe_description},
    ))
    binding = _assert_ok(client.post(
        _api(f"/production-bibles/{bible_id}/cast"),
        json={
            "actor_id": actor_id,
            "tag": tag,
            "character_name": character_name,
            "character_description": character_description,
            "role": role,
            "voice_profile_id": voice_profile_id,
            "prompt_tags": [tag, "cyberpunk"],
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
    return _assert_ok(client.put(
        _api(f"/productions/{production_id}/screenplay"),
        json={
            "title": "Wet Circuit",
            "genre": "Cyberpunk noir short",
            "logline": (
                "A freelance data courier carries three hundred megabytes of stolen "
                "corporate memory across a neon megacity in one rain-soaked night."
            ),
            "treatment": (
                "Night in the megacity. BRANDON, a veteran data courier with a wet-wired "
                "implant behind his ear, threads through a neon street market while corporate "
                "hunter-drones sweep the crowds. He reaches a hacker den buried under a noodle "
                "bar and jacks into a console to verify the cargo before the handoff. With the "
                "data confirmed and the buyers inbound, he climbs to a rooftop above the skyline "
                "as dawn threatens, and lets the city decide whether he gets to keep his head."
            ),
            "character_breakdowns": [
                {"tag": "NARRATOR", "name": "Narrator", "role": "noir voiceover"},
                {"tag": "BRANDON", "name": "Brandon", "role": "data courier protagonist"},
            ],
            "scene_breakdown": [
                {
                    "scene_number": 1,
                    "slugline": "EXT. @NEON_MARKET - NIGHT",
                    "intent": (
                        "Brandon moves through a rain-slick neon street market — holographic "
                        "signage, steam vents, dense crowds — keeping the implant hidden."
                    ),
                    "characters_present": ["BRANDON"],
                    "set_ref": "NEON_MARKET",
                    "props_required": ["CYBERDECK"],
                    "emotional_beat": "Paranoia under sensory overload.",
                },
                {
                    "scene_number": 2,
                    "slugline": "INT. @DATA_DEN - NIGHT",
                    "intent": (
                        "In a cramped hacker den beneath a noodle bar, Brandon jacks the "
                        "implant into a battered cyberdeck and watches the stolen data unfold "
                        "in cascading glyphs."
                    ),
                    "characters_present": ["BRANDON"],
                    "set_ref": "DATA_DEN",
                    "props_required": ["CYBERDECK"],
                    "emotional_beat": "Focus and dread — the cargo is bigger than the job.",
                },
                {
                    "scene_number": 3,
                    "slugline": "EXT. @ROOFTOP_SKYLINE - PRE-DAWN",
                    "intent": (
                        "Brandon stands on a rooftop above the endless megacity as hunter-"
                        "drones trace the streets below; the handoff window opens with the "
                        "first gray light."
                    ),
                    "characters_present": ["BRANDON"],
                    "set_ref": "ROOFTOP_SKYLINE",
                    "props_required": ["CYBERDECK"],
                    "emotional_beat": "Defiance at the edge of escape.",
                },
            ],
            "script": (
                "NARRATOR: Three hundred megabytes of somebody else's memory, riding wet-wired behind my ear.\n"
                "NARRATOR: The market doesn't care what you carry. It only cares that you keep moving.\n"
                "NARRATOR: Down under the noodle bar, the den smelled like ozone and old solder.\n"
                "NARRATOR: The data unfolded like a city of its own — and half of it wasn't supposed to exist.\n"
                "NARRATOR: By rooftop light the drones were still hunting yesterday's ghost.\n"
                "NARRATOR: Come dawn, the city keeps the data. With luck, I keep my head."
            ),
            "shot_list": [
                {"scene_number": 1, "shot_number": 1, "description": "Wide neon market street in rain, Brandon weaving through the crowd under holographic signs."},
                {"scene_number": 1, "shot_number": 2, "description": "Tracking close-up of Brandon's face lit by shifting neon, eyes scanning for drones."},
                {"scene_number": 2, "shot_number": 1, "description": "Brandon descends into the cramped hacker den, cables and CRT glow everywhere."},
                {"scene_number": 2, "shot_number": 2, "description": "Close on Brandon jacked into the cyberdeck, data glyphs reflecting on his glasses."},
                {"scene_number": 3, "shot_number": 1, "description": "Brandon steps onto the rooftop, endless megacity skyline with searchlight drones below."},
                {"scene_number": 3, "shot_number": 2, "description": "Wide hero shot of Brandon at the rooftop edge as pre-dawn light breaks over the towers."},
            ],
            "text_model": TEXT_MODEL,
        },
    ))


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
        ["ffprobe", "-v", "error", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "codec_type=video" in streams
    assert "codec_type=audio" in streams
    volume = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    assert "mean_volume: -inf" not in volume
    assert "max_volume: -inf" not in volume


def test_cyberpunk_seedance_cloud_e2e(tmp_path):
    with httpx.Client(timeout=httpx.Timeout(60, read=600), follow_redirects=True) as client:
        settings = _preflight(client)
        default_voice_id = settings["default_voice_id"]

        actor = _find_actor(client, ACTOR_NAME)
        actor_id = actor["id"]

        # cast_bindings has a unique (production_bible_id, actor_id) constraint,
        # so the narrator needs its own voice-only actor rather than reusing
        # the on-screen Brandon actor.
        narrator_actor = _assert_ok(client.post(
            _api("/asset-library/actors"),
            json={
                "name": "Brandon Narrator (voice)",
                "description": "Voice-only noir narrator for the Wet Circuit cyberpunk short. Never appears on screen.",
                "base_appearance_prompt": "Voice-only narrator; no on-screen appearance.",
                "prompt_tags": ["narrator", "voice-only"],
            },
        ))
        narrator_actor_id = narrator_actor["id"]
        voice = _assert_ok(client.post(
            _api(f"/asset-library/actors/{narrator_actor_id}/voice-profiles"),
            json={
                "voice_id": default_voice_id,
                "adapter_type": "ELEVENLABS",
                "style_notes": "Low, dry cyberpunk noir voiceover. World-weary, deliberate pacing.",
            },
        ))
        voice_profile_id = voice["id"]

        bible = _assert_ok(client.post(
            _api("/production-bibles"),
            json={
                "name": f"Wet Circuit Bible {int(time.time())}",
                "description": "Cyberpunk noir short — Brandon as a data courier in a neon megacity.",
                "category": "FULL_PRODUCTION",
                "tags": ["cyberpunk", "noir", "e2e", "seedance"],
            },
        ))
        bible_id = bible["production_bible_id"]
        production = _assert_ok(client.post(
            _api("/productions"),
            json={
                "name": f"Wet Circuit (Cyberpunk E2E) {int(time.time())}",
                "description": (
                    f"Opt-in cloud E2E production using {TEXT_MODEL}, {IMAGE_MODEL}, "
                    f"{VIDEO_MODEL}, and ElevenLabs narration."
                ),
                "tags": ["e2e", "cyberpunk", "seedance"],
            },
        ))
        production_id = production["id"]
        _assert_ok(client.put(
            _api(f"/productions/{production_id}"),
            json={"production_bible_id": bible_id},
        ))

        _bind_actor(
            client,
            actor_id=actor_id,
            bible_id=bible_id,
            tag="BRANDON",
            character_name="Brandon",
            character_description=(
                "A veteran freelance data courier: bald, rectangular glasses, a subtle "
                "data-implant scar behind the right ear. Calm under pressure, always scanning."
            ),
            role="LEAD",
            voice_profile_id=None,
            wardrobe_label="Courier rig",
            wardrobe_description=(
                "Worn black tactical jacket with high collar over a graphite tech-weave shirt, "
                "matte utility strap across the chest, dark cargo trousers — rain-slick "
                "cyberpunk street courier."
            ),
        )
        _bind_actor(
            client,
            actor_id=narrator_actor_id,
            bible_id=bible_id,
            tag="NARRATOR",
            character_name="Narrator",
            character_description="Brandon's interior noir voiceover. Never appears on screen.",
            role="NARRATOR",
            voice_profile_id=voice_profile_id,
            wardrobe_label="No wardrobe (voice only)",
            wardrobe_description="Voice-only narrator; no on-screen appearance.",
        )

        for set_name, set_tag, set_desc, lighting in [
            (
                "Neon Street Market",
                "NEON_MARKET",
                "Rain-slick cyberpunk street market: dense crowds, noodle stalls, steam vents, "
                "towering holographic advertisements in kanji and hangul, tangled power lines.",
                "Saturated neon signage (magenta/cyan) reflecting off wet asphalt; volumetric steam.",
            ),
            (
                "Underground Data Den",
                "DATA_DEN",
                "Cramped hacker den beneath a noodle bar: racks of salvaged servers, CRT monitors, "
                "cable bundles on every wall, a battered cyberdeck workstation.",
                "Dim practicals — CRT green and amber glow, single hanging bulb, deep shadows.",
            ),
            (
                "Megacity Rooftop",
                "ROOFTOP_SKYLINE",
                "Flat industrial rooftop high above an endless cyberpunk megacity skyline; "
                "antenna clusters, vents, distant arcologies and searchlight drones.",
                "Pre-dawn blue hour with neon haze below; first gray light on the horizon.",
            ),
        ]:
            library_set = _assert_ok(client.post(
                _api("/asset-library/sets"),
                json={
                    "name": set_name,
                    "description": set_desc,
                    "prompt_tags": [set_tag, "cyberpunk"],
                    "lighting_notes": lighting,
                },
            ))
            _assert_ok(client.post(
                _api(f"/production-bibles/{bible_id}/set-bindings"),
                json={
                    "library_set_id": library_set["id"],
                    "tag": set_tag,
                    "production_name": set_name,
                },
            ))

        cyberdeck = _assert_ok(client.post(
            _api("/asset-library/props"),
            json={
                "name": "Cyberdeck",
                "description": "A battered portable cyberdeck — matte black slab with worn keys and a coiled neural jack cable.",
                "appearance_prompt": "Battered matte-black portable cyberdeck console, worn keycaps, coiled neural jack cable, scuffed decals.",
                "prompt_tags": ["CYBERDECK", "cyberpunk"],
            },
        ))
        _assert_ok(client.post(
            _api(f"/production-bibles/{bible_id}/prop-bindings"),
            json={
                "library_prop_id": cyberdeck["id"],
                "tag": "CYBERDECK",
                "production_name": "Cyberdeck",
            },
        ))
        _assert_ok(client.post(_api(f"/production-bibles/{bible_id}/finalize")))

        screenplay = _seed_screenplay(client, production_id)
        assert screenplay["text_model"] == TEXT_MODEL
        _assert_ok(client.patch(_api(f"/productions/{production_id}/screenplay/status"), json={"status": "LOCKED"}))
        created_scenes = _assert_ok(client.post(
            _api(f"/productions/{production_id}/screenplay/generate-scenes?force=true"),
        ))
        assert len(created_scenes) >= SCENE_COUNT

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
                    "audio_enabled": AUDIO_ENABLED,
                    "production_bible_id": bible_id,
                    "commit_message": "Configure cyberpunk E2E models and shot count",
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

        sound_deck = _assert_ok(client.post(
            _api(f"/productions/{production_id}/sound-deck/generate-audio"),
            timeout=900,
        ))["sound_deck"]
        ready_sound_cues = [cue for cue in sound_deck["cues"] if cue["generation_status"] == "READY"]
        assert len(ready_sound_cues) >= SCENE_COUNT

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

        master = _assert_ok(client.post(_api(f"/productions/{production_id}/render-master"), timeout=900))
        assert master["scene_count"] >= SCENE_COUNT

        video_response = client.get(f"{API_BASE}{master['video_url']}", timeout=900)
        assert video_response.status_code == 200
        master_path = tmp_path / "cyberpunk_master.mp4"
        master_path.write_bytes(video_response.content)
        _assert_master_has_audio(master_path)

        print(f"\nPRODUCTION_ID={production_id}")
        print(f"MASTER_BYTES={len(video_response.content)}")
