---
phase: 25-lora-training-infrastructure
plan: 02
subsystem: api, ui
tags: [lora, replicate, training, fastapi, react, typescript, polling]

# Dependency graph
requires:
  - phase: 25-lora-training-infrastructure
    provides: LoRA training service (lora_trainer.py), Actor model with lora columns, UserSettings with replicate columns
  - phase: 22-asset-library-core
    provides: Actor CRUD, ActorRef, asset_library_router, ActorLibraryDetail component
provides:
  - POST /api/asset-library/actors/{id}/train-lora endpoint with ref count and token validation
  - GET /api/asset-library/actors/{id}/lora-status endpoint with Replicate polling and weight download
  - Actor detail API response includes lora_url, lora_trained_at, lora_training_status
  - Frontend trainActorLora() and getActorLoraStatus() client functions
  - Train Identity Model button and LoRA status badge UI in ActorLibraryDetail
affects: [video_gen, keyframes, tag_resolver]

# Tech tracking
tech-stack:
  added: []
  patterns: [asyncio.create_task for background training dispatch, interval polling in React useEffect for training status]

key-files:
  created: []
  modified:
    - backend/vidpipe/api/asset_library.py
    - frontend/src/api/types.ts
    - frontend/src/api/client.ts
    - frontend/src/components/ActorLibraryDetail.tsx

key-decisions:
  - "Background training uses asyncio.create_task with its own async_session per Phase 18 convention"
  - "getattr() used for backward compat with pre-migration Actor objects in serialization"
  - "Frontend polling interval of 10 seconds for QUEUED/TRAINING states avoids excessive API calls"
  - "LoraStatusBadge is a separate inline component for reusability"

patterns-established:
  - "Background task pattern: extract plain values from request session, launch asyncio.create_task with fresh async_session"
  - "Training status polling: React useEffect with setInterval, cleared on terminal state or unmount"

requirements-completed: [LORA-03, LORA-04, LORA-05]

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 25 Plan 02: LoRA Training API & Frontend Summary

**LoRA training dispatch/polling API endpoints with Train Identity Model button, status badges (No Model/Training/Ready/Failed), and React status polling in ActorLibraryDetail**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-14T22:34:51Z
- **Completed:** 2026-03-14T22:39:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- POST /actors/{id}/train-lora validates min 5 refs and Replicate token, dispatches background training, returns 202
- GET /actors/{id}/lora-status polls Replicate for non-terminal states, downloads+stores weights on completion
- Actor detail API response includes lora_url, lora_trained_at, lora_training_status fields
- Frontend Actor interface extended with LoRA fields; trainActorLora/getActorLoraStatus client functions added
- ActorLibraryDetail OverviewTab shows LoRA Identity Model section with Train/Retrain button (disabled when refs < 5), LoraStatusBadge, and 10s polling for in-progress training

## Task Commits

Each task was committed atomically:

1. **Task 1: Add POST train-lora and GET lora-status API endpoints + update actor serialization** - `7b6bd78` (feat)
2. **Task 2: Frontend Actor types, API client functions, and Train Identity Model UI** - `cef54ea` (feat)

## Files Created/Modified
- `backend/vidpipe/api/asset_library.py` - Added train-lora (POST, 202), lora-status (GET) endpoints, updated _actor_detail_to_dict with lora fields
- `frontend/src/api/types.ts` - Extended Actor/ActorListItem with lora fields, added LoraStatusResponse/TrainLoraResponse types
- `frontend/src/api/client.ts` - Added trainActorLora() and getActorLoraStatus() client functions
- `frontend/src/components/ActorLibraryDetail.tsx` - Added LoRA Identity Model section with Train button, LoraStatusBadge, polling, and error display

## Decisions Made
- Background training uses `asyncio.create_task()` with its own `async_session()` per Phase 18 convention (never share request session)
- Used `getattr(actor, 'lora_url', None)` for backward compatibility with pre-migration Actor objects
- Frontend polling interval of 10 seconds for QUEUED/TRAINING states balances responsiveness vs API load
- LoraStatusBadge is a standalone function component for clarity and potential reuse
- Button text changes contextually: "Train Identity Model" (no model/failed), "Retrain Model" (completed), "Training in Progress..." (active)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - Replicate API token is configured via the existing Settings page. The train-lora endpoint validates its presence and returns a user-friendly 422 error if missing.

## Next Phase Readiness
- LoRA training loop is now complete end-to-end: user triggers from Actor detail -> background training -> status polling -> weight storage -> tag resolver feeds lora_url to Flux generation
- Phase 25 (LoRA Training Infrastructure) is fully complete

---
*Phase: 25-lora-training-infrastructure*
*Completed: 2026-03-14*

## Self-Check: PASSED
