---
phase: 28-voice-script-tts-lip-sync-layer
plan: 01
subsystem: api
tags: [voice-script, elevenlabs, tts, screenplay, react, sqlalchemy]
requires: []
provides:
  - VoiceScript and VoiceLine persistence
  - VoiceScriptService generation, binding, and TTS rendering
  - Voice script API routes and Screenplay Voice tab
affects: [screenplay, production-bible, audio, ui]
tech-stack:
  added: []
  patterns: [domain API router, service-owned voice workflow, typed frontend API client]
key-files:
  created:
    - backend/vidpipe/api/voice_script.py
    - backend/vidpipe/schemas/voice.py
    - backend/vidpipe/services/voice_script_service.py
    - backend/tests/test_voice_script_service.py
    - frontend/src/components/VoiceScriptTab.tsx
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/api/app.py
    - backend/vidpipe/services/audio/base.py
    - backend/vidpipe/services/audio/elevenlabs_adapter.py
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts
    - frontend/src/components/ScreenplayEditor.tsx
key-decisions:
  - Voice lines store resolved provider voice_id, but LLM output never supplies provider voice IDs.
  - Binding resolution prefers CastBinding.voice_profile_id, then actor voice profile, then legacy character VoiceProfile.
  - Generated voice audio is served through voice-line API endpoints rather than relying on generic storage URLs.
patterns-established:
  - Voice workflow routes delegate all stateful behavior to VoiceScriptService.
  - Frontend action responses return the full VoiceScript state for UI refresh.
requirements-completed: [VOICE-01, VOICE-02, VOICE-03, VOICE-04, VOICE-05, VOICE-08]
duration: 75min
completed: 2026-06-07
---

# Phase 28 Plan 01 Summary

**Editable screenplay voice scripts with deterministic cast voice binding and ElevenLabs-compatible line audio generation**

## Performance

- **Completed:** 2026-06-07T21:29:48-05:00
- **Tasks:** 5
- **Files modified:** 12

## Accomplishments

- Added `VoiceScript` and `VoiceLine` ORM models. Startup `create_all()` picks up the new tables through the existing `vidpipe.db.models` import path.
- Added structured voice-script schemas and `VoiceScriptService` for generation, binding resolution, editable lines, and TTS artifact storage.
- Added `/api/productions/{id}/voice-script`, line audio generation, binding, edit, delete, and audio serving routes.
- Extended the audio adapter interface for contextual TTS parameters while preserving existing ElevenLabs behavior.
- Added a Screenplay `Voice` tab with line editing, generation controls, warnings, status pills, and audio playback.
- Added focused backend service coverage for generation, binding, and TTS storage.

## Verification

- `python -m py_compile backend/vidpipe/api/voice_script.py backend/vidpipe/services/voice_script_service.py backend/vidpipe/services/voice_mixer.py backend/vidpipe/services/lip_sync/base.py backend/vidpipe/services/lip_sync/fake.py backend/tests/test_voice_script_service.py` passed.
- `pytest backend/tests/test_voice_script_service.py backend/tests/test_elevenlabs_adapter.py -q` passed: 15 passed, 1 third-party ElevenLabs/Pydantic warning.
- `ruff check backend/vidpipe/api/voice_script.py backend/vidpipe/services/voice_script_service.py backend/vidpipe/services/voice_mixer.py backend/vidpipe/services/lip_sync backend/tests/test_voice_script_service.py` passed.

## Issues Encountered

- The ElevenLabs adapter needed to support both direct async iterators and awaitable SDK calls; fixed with `inspect.isawaitable`.
- Frontend `npm run build` and `npm run lint` are blocked by existing unrelated TypeScript/ESLint issues in editor and asset components. The failures were not in the new voice files.

## User Setup Required

ElevenLabs API key must be configured in Settings before real TTS generation can run.

## Next Phase Readiness

Plan 02 can build directly on the persisted `audio_path`, `lip_sync_mode`, shot linkage, and full VoiceScript API response shape.
