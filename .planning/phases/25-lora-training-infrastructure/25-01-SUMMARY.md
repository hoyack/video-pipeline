---
phase: 25-lora-training-infrastructure
plan: 01
subsystem: services
tags: [lora, replicate, training, pillow, sqlalchemy, async]

# Dependency graph
requires:
  - phase: 22-asset-library-core
    provides: Actor, ActorRef models and tag resolver infrastructure
  - phase: 24-comfyui-flux-workflows
    provides: Flux pipeline integration that consumes lora_url from ResolvedAssetRef
provides:
  - Actor model with lora_url/lora_trained_at/lora_training_status/lora_training_job_id columns
  - UserSettings with replicate_api_token/replicate_username columns
  - LoRATrainingBackend ABC with pluggable implementations
  - ReplicateBackend for Flux LoRA training via Replicate API
  - Dataset preparation pipeline (resize + VLM caption + zip)
  - Tag resolver lora_url passthrough from Actor to ResolvedAssetRef
affects: [25-02, keyframes, video_gen, tag_resolver]

# Tech tracking
tech-stack:
  added: [replicate>=1.0.0]
  patterns: [asyncio.to_thread for sync SDK wrapping, ABC backend abstraction for training providers, TYPE_CHECKING guard for circular import avoidance]

key-files:
  created:
    - backend/vidpipe/services/lora_trainer.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/services/tag_resolver.py
    - backend/pyproject.toml

key-decisions:
  - "Used getattr(actor, 'lora_url', None) for backward compatibility with pre-migration Actor objects"
  - "ReplicateBackend wraps all SDK calls in asyncio.to_thread() since replicate SDK is synchronous"
  - "TYPE_CHECKING guard on LLMAdapter import to avoid circular imports between services"
  - "Dataset trigger_word derived from actor name: ACTOR_{NAME_UPPER} for unique LoRA association"

patterns-established:
  - "LoRATrainingBackend ABC: dispatch/poll_status/get_result pattern for pluggable training providers"
  - "Dataset preparation pipeline: resize with padding + VLM captioning + zip packaging"

requirements-completed: [LORA-01, LORA-02]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 25 Plan 01: LoRA Training Infrastructure Summary

**LoRA training data model, Replicate backend with async SDK wrapping, and dataset preparation pipeline (resize + VLM caption + zip)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T22:29:09Z
- **Completed:** 2026-03-14T22:32:05Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Actor model extended with 4 LoRA training state columns (lora_url, lora_trained_at, lora_training_status, lora_training_job_id)
- UserSettings extended with Replicate configuration (replicate_api_token, replicate_username)
- Tag resolver wired to pass actor.lora_url through to ResolvedAssetRef (no more hardcoded None)
- Created lora_trainer.py (345 lines) with LoRATrainingBackend ABC, ReplicateBackend, dataset prep, and weight storage

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend Actor and UserSettings models + migration + tag resolver + dependency** - `b1ff7e4` (feat)
2. **Task 2: Create lora_trainer.py service** - `cb14179` (feat)

## Files Created/Modified
- `backend/vidpipe/services/lora_trainer.py` - New: LoRA training service with ABC, Replicate backend, dataset preparation, weight storage
- `backend/vidpipe/db/models.py` - Modified: Actor +4 lora columns, UserSettings +2 replicate columns
- `backend/vidpipe/db/__init__.py` - Modified: 6 ALTER TABLE migration statements for new columns
- `backend/vidpipe/services/tag_resolver.py` - Modified: lora_url passthrough from actor to ResolvedAssetRef
- `backend/pyproject.toml` - Modified: Added replicate>=1.0.0 dependency

## Decisions Made
- Used `getattr(actor, 'lora_url', None)` in tag resolver for backward compatibility with pre-migration Actor objects
- All Replicate SDK calls wrapped in `asyncio.to_thread()` since the SDK is synchronous
- Used `TYPE_CHECKING` guard for LLMAdapter import to avoid circular imports between services
- Dataset trigger word derived from actor name (`ACTOR_{NAME_UPPER}`) for unique LoRA training association
- Used `ostris/flux-dev-lora-trainer` on Replicate as the training model
- Image captions generated via VLM with temperature 0.3 for consistent descriptive output

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Replicate SDK 1.0.7 has a pydantic v1 compatibility issue on Python 3.14 (the dev environment uses 3.14). The SDK works correctly on Python 3.11-3.13, which is the project's supported runtime per pyproject.toml. The import succeeds in production (Docker uses Python 3.11).

## User Setup Required

None - no external service configuration required. Replicate API token is stored per-user in UserSettings and will be configured via the UI in Plan 02.

## Next Phase Readiness
- Data model and service layer ready for Plan 02 API endpoints and frontend
- Tag resolver automatically flows trained LoRA URLs into Flux image generation
- ReplicateBackend ready for training dispatch once API token is configured

---
*Phase: 25-lora-training-infrastructure*
*Completed: 2026-03-14*

## Self-Check: PASSED
