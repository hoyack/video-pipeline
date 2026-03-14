# Phase 25: LoRA Training Infrastructure - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning
**Source:** PRD Express Path (docs/assets_mapping.md, Section C)

<domain>
## Phase Boundary

This phase adds per-actor LoRA training capability — extending the Actor model with training state fields, implementing a training service with dataset preparation and pluggable backend dispatch, adding API endpoints for triggering and polling training, and providing frontend UI for training status. It does NOT cover frontend @tag autocomplete or tag preview (Phase 26).

Specifically, this phase delivers:
- Actor model extended with `lora_url`, `lora_trained_at`, `lora_training_status` columns + migration
- `lora_trainer.py` service with dataset prep (download refs, resize, VLM caption) and pluggable backend interface
- Initial backend implementation: Replicate API (`lucataco/simpletuner-flux` or similar)
- `POST /api/asset-library/actors/{id}/train-lora` endpoint
- `GET /api/asset-library/actors/{id}/lora-status` endpoint
- Frontend "Train Identity Model" button and status badge on Actor detail view

</domain>

<decisions>
## Implementation Decisions

### Actor Model Extensions
- Three new columns: `lora_url` (nullable str, S3 path to .safetensors), `lora_trained_at` (nullable datetime), `lora_training_status` (nullable str: QUEUED/TRAINING/COMPLETED/FAILED)
- Alembic migration or conditional column add (per project convention for SQLite)
- Similar fields on `LibraryProp` deferred to future — only Actor gets LoRA for now

### Training Service Architecture
- New `lora_trainer.py` in `backend/vidpipe/services/`
- Pluggable backend interface: `LoRATrainingBackend` abstract base with `dispatch()`, `poll_status()`, `get_result()`
- Initial implementation: `ReplicateBackend` using Replicate API
- Service handles: dataset prep → dispatch → status polling → result storage
- Async-first (all I/O uses async def + await)

### Dataset Preparation
- Download all ActorRef images for the actor
- Resize to 512x512 or 768x768 (maintain aspect ratio with padding)
- Generate captions via VLM (existing LLM adapter pattern — use vision_model)
- Caption format: describe appearance without name, add trigger word (`ACTOR_{TAG}`) to subset
- Package as zip, upload to S3 for training worker
- Minimum 5 reference images to enable training (button disabled below 5)

### API Endpoints
- `POST /api/asset-library/actors/{id}/train-lora` — validates min refs, dispatches job, returns job status
- `GET /api/asset-library/actors/{id}/lora-status` — returns current training status, progress, lora_url when complete
- Both in existing `asset_library.py` route file (per project convention: split by domain)

### Replicate API Integration
- Use `replicate` Python package (new dependency)
- Model: `lucataco/simpletuner-flux` or equivalent Flux LoRA training model
- Input: zip of captioned images + trigger word + training config
- Output: .safetensors file URL → download to S3 → update Actor.lora_url
- Polling via Replicate's prediction status API

### Frontend UI
- Actor detail view: "Train Identity Model" button (enabled when refs.length >= 5)
- Status badge: "No Model" / "Training..." / "Model Ready" with training date
- "Regenerate Model" button when actor updated since last training
- Uses existing `ActorLibraryDetail.tsx` component

### Claude's Discretion
- Exact Replicate model version/ID selection
- Training hyperparameters (steps, learning rate, LoRA rank)
- Error handling and retry strategy for failed training jobs
- Background polling mechanism (one-shot check vs continuous poll)
- Whether to store training config/history for debugging

</decisions>

<specifics>
## Specific Ideas

- PRD recommends Replicate API as initial backend (easy, per-job cost, no infrastructure)
- Minimum dataset sizes from PRD: face identity 5-20 images, full body 10-30 images
- Training steps: 1000-1500 for face, 1500-2000 for full body
- Trigger word pattern: `ACTOR_BRANDON` (prefix + tag)
- The `lora_url` field on ResolvedAssetRef (from Phase 23) is already wired into the Flux pipeline (Phase 24) — once this phase populates it, the LoRA will automatically be used in image generation

</specifics>

<deferred>
## Deferred Ideas

- LibraryProp LoRA training (similar pattern, lower priority) → Future
- ComfyUI Flux Trainer custom node as alternative backend → Future
- Local GPU worker backend → Future
- RunPod/Lambda Labs on-demand backend → Future
- LoRA versioning and rollback → Future
- Automatic LoRA invalidation when refs updated → Future
- LoRA merging for multi-character shots → Future

</deferred>

---

*Phase: 25-lora-training-infrastructure*
*Context gathered: 2026-03-14 via PRD Express Path*
