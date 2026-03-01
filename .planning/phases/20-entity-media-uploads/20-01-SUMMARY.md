---
phase: 20-entity-media-uploads
plan: 01
subsystem: api
tags: [fastapi, upload, file-upload, storage-backend, reverse-prompt, audio]

# Dependency graph
requires:
  - phase: 17-production-bible-entity-expansion
    provides: "Character, Wardrobe, VoiceProfile, Set, SonicIdentity, ScoreTheme, SFXItem entity models and CRUD endpoints"
  - phase: 16-production-bible-foundation
    provides: "ProductionBible model, dual storage backend, upload_set_reference pattern"
provides:
  - "POST /characters/:id/actor-refs image upload endpoint"
  - "POST /characters/:id/generate-appearance LLM Vision endpoint"
  - "POST /wardrobes/:id/upload-reference image upload endpoint"
  - "POST /generate-reverse-prompt standalone image analysis endpoint"
  - "POST /score-themes/:id/upload-audio audio upload endpoint"
  - "POST /sfx/:id/upload-audio audio upload endpoint"
  - "POST /sonic-identities/:id/upload-audio audio upload endpoint"
affects: [20-entity-media-uploads]

# Tech tracking
tech-stack:
  added: []
  patterns: [audio-upload-with-20MB-limit, ALLOWED_AUDIO_TYPES-constant, new-list-pattern-for-json-array-mutation]

key-files:
  created: []
  modified:
    - backend/vidpipe/api/characters.py
    - backend/vidpipe/api/sets_props.py
    - backend/vidpipe/api/sound.py

key-decisions:
  - "Audio uploads use 20MB limit (vs 10MB for images) since audio files are typically larger"
  - "generate-appearance returns 500 on LLM failure (not graceful degradation) since user explicitly requested the action"
  - "Standalone generate-reverse-prompt uses OBJECT asset type as generic default"
  - "Sonic identity audio stored under sets/{set_id}/sonic_identity/ path hierarchy matching parent relationship"

patterns-established:
  - "ALLOWED_AUDIO_TYPES tuple for audio MIME validation across sound endpoints"
  - "Helper dict functions (_score_theme_to_dict, _sfx_item_to_dict, _sonic_identity_to_dict) in sound.py for consistent response shapes"

requirements-completed: [PBEX-01, PBEX-02, PBEX-07, PBEX-08, PBEX-16, PBEX-17]

# Metrics
duration: 2min
completed: 2026-03-01
---

# Phase 20 Plan 01: Entity Media Uploads Summary

**7 file upload endpoints across 3 route files: actor refs, wardrobe refs, generate-appearance, standalone reverse-prompt, and audio uploads for ScoreTheme/SFXItem/SonicIdentity**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-01T22:29:36Z
- **Completed:** 2026-03-01T22:31:50Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- 4 image upload endpoints: actor-ref upload (appends to JSON array), generate-appearance (LLM Vision with 422 guard), wardrobe reference upload (appends to JSON array), standalone generate-reverse-prompt
- 3 audio upload endpoints: ScoreTheme reference_audio, SFXItem source_audio, SonicIdentity reference_audio with 20MB limit and audio MIME validation
- All endpoints follow the established dual local/S3 storage pattern from upload_set_reference

## Task Commits

Each task was committed atomically:

1. **Task 1: Character and wardrobe upload endpoints + standalone reverse-prompt** - `ee2e122` (feat)
2. **Task 2: Audio upload endpoints for Sound Department entities** - `7347ec4` (feat)

## Files Created/Modified
- `backend/vidpipe/api/characters.py` - Added upload_actor_ref, generate_appearance, upload_wardrobe_reference endpoints with imports for asyncio, File, UploadFile, storage_backend
- `backend/vidpipe/api/sets_props.py` - Added generate_reverse_prompt standalone endpoint before prop upload
- `backend/vidpipe/api/sound.py` - Added ALLOWED_AUDIO_TYPES constant, helper dict functions, and upload_score_theme_audio, upload_sfx_audio, upload_sonic_identity_audio endpoints with imports for asyncio, File, UploadFile, storage_backend, Set, SonicIdentity

## Decisions Made
- Audio uploads use 20MB limit (vs 10MB for images) since audio files are typically larger
- generate-appearance returns 500 on LLM failure (not graceful degradation) since user explicitly requested the action
- Standalone generate-reverse-prompt uses OBJECT asset type as generic default
- Sonic identity audio stored under sets/{set_id}/sonic_identity/ path hierarchy matching parent relationship

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 7 backend upload endpoints ready for frontend integration in Plan 20-02
- Frontend can wire up file pickers and audio players to these endpoints

---
*Phase: 20-entity-media-uploads*
*Completed: 2026-03-01*
