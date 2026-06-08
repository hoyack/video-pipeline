# Phase 28: Voice Script, TTS, and Lip Sync Layer - Research

**Researched:** 2026-06-07
**Domain:** Screenplay voice scripting, ElevenLabs TTS, ffmpeg audio mixing, and post-video lip-sync adapter design
**Confidence:** HIGH for voice/TTS/mixing, MEDIUM for lip-sync runtime selection

## Summary

The repo already has most primitives needed for the first useful voice slice:

- Screenplays are durable and editable.
- Production Bible cast bindings already include `voice_profile_id`.
- Actor-library and bible-scoped voice profile models already exist.
- ElevenLabs TTS is already wrapped in `AudioAdapter`.
- Audio upload and playback paths already exist.
- ffmpeg is already a validated dependency for stitching.

The missing layer is orchestration and persistence: normalized voice scripts, line-level TTS artifacts, audio mix artifacts, and a post-video lip-sync job boundary.

Recommendation:

1. Ship voice-script persistence and one-line ElevenLabs generation first.
2. Add auto-generation from Screenplay as structured LLM output once the manual line flow exists.
3. Add ffmpeg mix artifacts before lip sync.
4. Add lip sync as an adapter with eligibility and fake/test implementation before selecting a heavyweight default runtime.

## Existing Patterns To Reuse

### Database and Migration

Use `backend/vidpipe/db/models.py` for ORM classes and the existing startup migration approach in `backend/vidpipe/db/__init__.py`. This repo has historically used conditional table/column creation instead of a full Alembic workflow.

### API Shape

Use a domain route file:

- New file: `backend/vidpipe/api/voice_script.py`
- Include router in `backend/vidpipe/api/app.py`
- Keep request/response Pydantic models local or move to `backend/vidpipe/schemas/voice.py` when reused by services/tests.

Follow route style from:

- `backend/vidpipe/api/screenplay.py`
- `backend/vidpipe/api/audio_gen.py`
- `backend/vidpipe/api/bindings.py`

### Audio Provider

`AudioAdapter.generate_voice()` currently accepts `voice_id`, `text`, `style_notes`, and `model_id`. Extend this method conservatively with optional kwargs used by ElevenLabs, while keeping defaults backward-compatible.

ElevenLabs current API shape supports text-to-speech through `POST /v1/text-to-speech/:voice_id`, `text`, optional `model_id`, and output-format query parameters. The current adapter default of `eleven_multilingual_v2` is a safe compatibility default.

### Frontend

`ScreenplayEditor.tsx` already has tab routing, inline regeneration, polling for full generation, status transitions, and JSON/text editing. Add a `Voice` tab rather than creating a separate production page.

Use existing `AudioPlayer.tsx` for line playback.

### ffmpeg

`stitcher.py` already shells out to ffmpeg from async code by wrapping sync subprocess work in `asyncio.to_thread`. Reuse this style for duration probing, line delay/mix, and overlay.

## Proposed Architecture

### Services

Add:

- `backend/vidpipe/services/voice_script_service.py`
  - create/get active script
  - generate structured lines from Screenplay
  - resolve speakers and voice profiles
  - generate one/all line audio
  - stale detection
- `backend/vidpipe/services/voice_mixer.py`
  - duration probing
  - build shot/scene stems
  - overlay voice stem onto video
- `backend/vidpipe/services/lip_sync/base.py`
  - adapter interface
- `backend/vidpipe/services/lip_sync/registry.py`
  - adapter lookup
- `backend/vidpipe/services/lip_sync/fake_adapter.py`
  - deterministic tests
- optional future `external_http_adapter.py` or `wav2lip_adapter.py`

### Data Flow

```
Screenplay + Production Bible bindings
  -> VoiceScriptService.generate_voice_script()
  -> VoiceScript + VoiceLine rows
  -> resolve CastBinding and voice profile
  -> ElevenLabs TTS per line
  -> line MP3 artifacts
  -> ffmpeg voice stem per shot/scene
  -> optional lip-sync replacement clip
  -> final stitch uses replacement when ready
```

## Lip Sync Runtime Notes

Use a provider boundary because runtime choices are unstable and environment-dependent.

Practical baseline:

- Wav2Lip-style local or external wrapper is the lowest-risk first integration because it accepts target video plus speech audio and produces a lip-synced video as a post-process.
- MuseTalk and LatentSync should remain adapters for later because they may offer better quality but have larger GPU/runtime packaging implications.

Eligibility should be deterministic and cheap:

- dialogue only
- one active speaker per shot
- human identity only by default
- close/medium shot preferred
- detectable face required
- original clip is never overwritten

## Risks

| Risk | Mitigation |
|------|------------|
| TTS cost and quota | line-level generation, generate-all confirmation, provider errors per line |
| Generated audio too long | measure duration, show warning, allow edit/regenerate |
| Native Veo audio conflicts | warn and recommend disabling native audio for voice-script scenes |
| Storage path leakage | serve generated audio through API/storage helper routes |
| Lip-sync dependencies are heavy | adapter boundary and fake/external implementation first |
| Ambiguous speaker tags | explicit warnings and manual binding in UI |

## Verification Strategy

Use fake adapters for CI:

- fake LLM for structured voice-script output
- fake audio adapter returning a small valid audio fixture or generated silent MP3/WAV
- fake lip-sync adapter copying input video or writing a fixture output

Run real ElevenLabs only as opt-in smoke with one short line and env-gated credentials.

## PLAN COMPLETE INPUTS

Planner should produce two plans:

1. VoiceScript/VoiceLine foundation, structured generation, ElevenLabs line generation, and Voice tab.
2. Voice mixing, non-destructive video overlay, LipSyncJob model, adapter boundary, and status UI.

---

## RESEARCH COMPLETE
