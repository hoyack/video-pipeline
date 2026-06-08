---
phase: 28-voice-script-tts-lip-sync-layer
plan: 02
subsystem: pipeline
tags: [voice-mix, lip-sync, ffmpeg, stitcher, react, sqlalchemy]
requires:
  - phase: 28-voice-script-tts-lip-sync-layer
    provides: VoiceScript and VoiceLine records with generated audio paths
provides:
  - VoiceMixArtifact and LipSyncJob persistence
  - VoiceMixer stem creation
  - LipSyncAdapter boundary with fake local adapter
  - Stitcher preference for ready lip-synced clips
affects: [stitcher, audio, video, ui]
tech-stack:
  added: []
  patterns: [adapter registry, non-destructive generated media artifacts]
key-files:
  created:
    - backend/vidpipe/services/voice_mixer.py
    - backend/vidpipe/services/lip_sync/__init__.py
    - backend/vidpipe/services/lip_sync/base.py
    - backend/vidpipe/services/lip_sync/fake.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/services/voice_script_service.py
    - backend/vidpipe/api/voice_script.py
    - backend/vidpipe/pipeline/stitcher.py
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts
    - frontend/src/components/VoiceScriptTab.tsx
key-decisions:
  - Initial VoiceMixer concatenates ready line audio into scene-level stems and persists artifacts without mutating source line audio.
  - Fake lip-sync adapter copies input clips to a new output path, giving tests and UI a deterministic READY job path.
  - Stitcher selects ready lip-sync job outputs by input clip, leaving original generated clips untouched.
patterns-established:
  - Lip-sync providers plug in behind `get_lip_sync_adapter`.
  - Mix and lip-sync actions refresh the same VoiceScript response consumed by the UI.
requirements-completed: [VOICE-06, VOICE-07, VOICE-08]
duration: 75min
completed: 2026-06-07
---

# Phase 28 Plan 02 Summary

**Voice stem artifacts and fake lip-sync jobs with non-destructive stitcher integration**

## Performance

- **Completed:** 2026-06-07T21:29:48-05:00
- **Tasks:** 4
- **Files modified:** 10

## Accomplishments

- Added `VoiceMixArtifact` and `LipSyncJob` ORM models.
- Added `VoiceMixer` to persist voice stems from generated line audio.
- Added lip-sync adapter contracts and a deterministic fake adapter for tests and local pipeline wiring.
- Added mix and lip-sync API endpoints and UI controls/status display.
- Updated stitcher to prefer a ready lip-synced clip output when present while preserving the original clip path.
- Added focused backend coverage proving mix artifacts and fake lip-sync jobs land in persistent state.

## Verification

- `pytest backend/tests/test_voice_script_service.py backend/tests/test_elevenlabs_adapter.py -q` passed: 15 passed.
- `ruff check backend/vidpipe/api/voice_script.py backend/vidpipe/services/voice_script_service.py backend/vidpipe/services/voice_mixer.py backend/vidpipe/services/lip_sync backend/tests/test_voice_script_service.py` passed.

## Deviations From Plan

- Implemented `backend/vidpipe/services/lip_sync/__init__.py` as the registry entry point instead of separate `registry.py`; the public function is still `get_lip_sync_adapter`.
- Implemented `fake.py` instead of `fake_adapter.py`; same adapter role and export are present.
- The first mixer slice concatenates generated line audio into stems. Full timed `adelay`/`amix` video overlay remains a follow-up refinement.
- Eligibility currently skips non-dialogue, `NEVER`, missing shot, and missing clip cases. Non-human cast identity and multi-speaker-shot skip policies are not yet fully enforced.

## Issues Encountered

- Frontend global build/lint remain blocked by unrelated existing errors outside this phase; backend verification passed.

## User Setup Required

No additional setup for fake lip-sync. Real lip-sync providers will need a provider adapter and model/runtime configuration.

## Next Phase Readiness

The API/UI now exposes persisted mix artifacts and lip-sync jobs. A future provider can replace the fake adapter behind the registry without changing the Voice tab contract.
