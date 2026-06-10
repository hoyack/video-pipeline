# NASA Documentary End-to-End Test Plan

This document describes the current production-generation feature set and the
target end-to-end test for creating a short historical documentary about early
NASA. It uses the configured cloud stack:

- Text model: `ollama/kimi-k2.5:cloud`
- Vision model: `ollama/kimi-k2.5:cloud`
- Image model: `gemini-2.5-flash-image` (Nano Banana)
- Video model: `wan-2.2-i2v`
- Voice: ElevenLabs default narrator voice from `/api/settings`

## Feature Map

The project now supports these production-level pieces:

- Production Bible: create, bind, finalize, and attach a reusable bible to a production.
- Asset Library: create actors, actor references, actor voice profiles, wardrobe presets, sets, props, and bindings.
- Screenplay: attach a screenplay to a production, lock it, and create draft scenes from `scene_breakdown`.
- Scene pipeline: edit generated scenes, bind a production bible, generate storyboards, keyframes, video clips, and stitched scene outputs.
- Voice script: generate narration/dialogue from the screenplay, bind speaker tags to cast voices, generate ElevenLabs audio, and create scene voice stems.
- Sound Deck: generate editable SFX/ambience cues from the production timeline, generate ElevenLabs sound-effect audio, and mix scene SFX stems.
- Production master render: render completed scene outputs plus scene voice and SFX stems into one final MP4 and serve it at `/api/productions/{production_id}/master-video`.

## E2E Shape

The NASA E2E creates a short documentary production with multiple scenes and
shots:

1. Preflight settings: verify ComfyUI, Ollama cloud, Kimi, ElevenLabs, and the default voice are configured.
2. Download public-domain historical references:
   - Katherine Johnson, NASA, 1966: `https://commons.wikimedia.org/wiki/File:Katherine_Johnson_at_NASA,_in_1966.jpg`
   - John Glenn NASA portrait: `https://commons.wikimedia.org/wiki/File:John-Glenn-NASA-portrait.jpg`
   - Wernher von Braun: `https://commons.wikimedia.org/wiki/File:Wernher-von-Braun.jpg`
   - Mercury Control: `https://commons.wikimedia.org/wiki/File:Mercury_Control.jpg`
3. Create a Production Bible and production.
4. Create and bind historical actors:
   - `NARRATOR` with the default ElevenLabs voice.
   - `KATHERINE_JOHNSON` with a NASA portrait reference and wardrobe.
   - `JOHN_GLENN` with a NASA portrait reference and wardrobe.
   - `WERNHER_VON_BRAUN` with a NASA reference and wardrobe.
5. Create and bind sets/props:
   - `MERCURY_CONTROL`
   - `MERCURY_CAPSULE`
6. Seed a deterministic two-scene screenplay, lock it, and generate scenes from it.
7. Configure each scene for Kimi, Nano Banana, Wan 2.2, 5-second clips, and disabled video audio.
8. Regenerate each scene through all phases.
9. Generate voice script, resolve bindings, generate ElevenLabs audio, and mix scene stems.
10. Generate Sound Deck cues, generate ElevenLabs SFX audio, and mix scene SFX stems.
11. Render the production master and verify the downloaded MP4 has video and non-silent audio with ffprobe/ffmpeg.

The test seeds the screenplay directly for deterministic cost and scene count.
The screenplay and voice script still exercise the production binding, scene,
image, video, voice, SFX, mix, and master-render integrations. A future
expansion can flip screenplay generation to Kimi end-to-end once cost controls
and scene-count constraints are stricter.

## Test Tiers

Fast local renderer test:

```bash
pytest backend/tests/test_production_master_service.py -q
```

This test creates synthetic silent scene MP4s plus voice and SFX stems with
ffmpeg, then verifies the production master has video and a non-silent audio
curve. `test_sound_deck_service.py` also verifies that delayed SFX cues land in
the intended scene-stem windows instead of collapsing to the start of the mix.

Opt-in paid cloud E2E:

```bash
VIDPIPE_RUN_NASA_E2E=1 pytest backend/tests/test_nasa_documentary_e2e.py -q -s
```

Optional knobs:

```bash
VIDPIPE_E2E_API_BASE=http://localhost:8100
VIDPIPE_NASA_E2E_SHOTS_PER_SCENE=2
VIDPIPE_NASA_E2E_SCENE_TIMEOUT=3600
```

Normal `pytest backend/` does not spend cloud credits because the NASA E2E is
skipped unless `VIDPIPE_RUN_NASA_E2E=1` is set.

## Acceptance Criteria

The E2E passes only if:

- Production Bible bindings exist for historical actors, wardrobe looks, set, and prop.
- At least two screenplay-generated scenes are configured for Kimi, Nano Banana, and Wan 2.2.
- Each selected scene reaches `complete` and has at least the requested shot count.
- Voice generation produces at least one ready ElevenLabs line.
- Voice mixing produces at least one ready scene voice stem.
- Sound Deck generation produces at least one ready SFX cue per selected scene.
- Sound Deck mixing produces at least one ready scene SFX stem per selected scene.
- Production master render includes at least two scenes plus voice and SFX stems.
- The downloaded final MP4 has a video stream, an audio stream, and non-silent ffmpeg `volumedetect` output.

## Known Constraints

- Wan 2.2 currently requires 5-second clips, so the E2E patches screenplay scenes from the default 8-second duration to 5 seconds before generation.
- Wan 2.2 does not provide native clip audio in this pipeline, so narration is mixed in as a production-level post step.
- The master renderer currently writes one stored MP4 at `productions/{production_id}/output/master.mp4`; it does not yet persist master metadata in a DB table.
- The NASA cloud test is intentionally slow and paid. Keep it out of default CI until provider budgets and cleanup policies are defined.
