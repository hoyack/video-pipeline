---
phase: 25-lora-training-infrastructure
verified: 2026-03-14T23:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 25: LoRA Training Infrastructure Verification Report

**Phase Goal:** Enable per-actor LoRA training from reference images with dataset preparation, a pluggable training backend (initially Replicate API), job management, and Actor model extensions for tracking training state
**Verified:** 2026-03-14T23:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Actor model has lora_url, lora_trained_at, lora_training_status, lora_training_job_id columns | VERIFIED | `models.py` lines 455-458; Python ORM inspection confirmed all 4 columns |
| 2 | UserSettings has replicate_api_token and replicate_username columns | VERIFIED | `models.py` lines 1087-1088; Python ORM inspection confirmed |
| 3 | Tag resolver passes actor.lora_url through to ResolvedAssetRef.lora_url instead of hardcoded None | VERIFIED | `tag_resolver.py` line 511: `lora_url=getattr(actor, 'lora_url', None)` |
| 4 | LoRATrainingBackend ABC defines dispatch(), poll_status(), get_result() async methods | VERIFIED | `lora_trainer.py` lines 66-91; ABC inspection confirmed all 3 abstract methods |
| 5 | ReplicateBackend wraps replicate SDK calls in asyncio.to_thread() | VERIFIED | `lora_trainer.py` lines 150, 163: both `_create` and `_get` closures wrapped in `asyncio.to_thread()` |
| 6 | Dataset preparation downloads refs, resizes to 1024x1024, captions via VLM, packages as zip | VERIFIED | `lora_trainer.py` lines 243-311; resize test confirmed 1024x1024 output via `_resize_with_padding` |
| 7 | replicate>=1.0.0 is listed in pyproject.toml dependencies | VERIFIED | `pyproject.toml` line 35: `"replicate>=1.0.0"` |
| 8 | POST /api/asset-library/actors/{id}/train-lora validates min 5 refs, checks Replicate token, dispatches training, returns 202 | VERIFIED | `asset_library.py` lines 996-1095; routes confirmed by Python import test |
| 9 | GET /api/asset-library/actors/{id}/lora-status polls Replicate for non-terminal states, updates DB, returns current status | VERIFIED | `asset_library.py` lines 1098-1177 |
| 10 | Frontend Actor interface has lora_url, lora_trained_at, lora_training_status fields | VERIFIED | `types.ts` lines 892-894; ActorListItem also has lora_training_status at line 908 |
| 11 | Train Identity Model button appears on Actor detail when refs >= 5; status badge shows No Model / Training / Model Ready with training date | VERIFIED | `ActorLibraryDetail.tsx` lines 380-407 (button + disable logic), 427-471 (LoraStatusBadge with all 4 states) |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/vidpipe/services/lora_trainer.py` | Training service with ABC, ReplicateBackend, dataset prep | VERIFIED | 345 lines (exceeds 150 minimum); all required exports present |
| `backend/vidpipe/db/models.py` | Actor with 4 lora columns, UserSettings with 2 replicate columns | VERIFIED | Lines 455-458 (Actor), 1087-1088 (UserSettings) |
| `backend/vidpipe/db/__init__.py` | ALTER TABLE migrations for new columns | VERIFIED | Lines 220-225: all 6 migration statements present |
| `backend/vidpipe/services/tag_resolver.py` | LoRA URL passthrough from Actor to ResolvedAssetRef | VERIFIED | Line 511: `lora_url=getattr(actor, 'lora_url', None)` |
| `backend/vidpipe/api/asset_library.py` | POST train-lora and GET lora-status endpoints | VERIFIED | Both routes registered at correct paths and HTTP methods |
| `frontend/src/api/types.ts` | Actor interface with lora fields | VERIFIED | Lines 892-894 (Actor), 908 (ActorListItem), 1137-1148 (LoraStatusResponse, TrainLoraResponse) |
| `frontend/src/api/client.ts` | trainActorLora() and getActorLoraStatus() API client functions | VERIFIED | Lines 1603-1617 |
| `frontend/src/components/ActorLibraryDetail.tsx` | Train Identity Model button and status badge UI | VERIFIED | Lines 361-408 (section), 427-471 (LoraStatusBadge component) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tag_resolver.py` | `db/models.py` | `actor.lora_url` attribute access | WIRED | `lora_url=getattr(actor, 'lora_url', None)` at line 511 |
| `lora_trainer.py` | `replicate` | `replicate.Client` in ReplicateBackend | WIRED | Line 120: `self._client = replicate.Client(api_token=api_token)` (lazy import inside `__init__`) |
| `asset_library.py` | `lora_trainer.py` | `from vidpipe.services.lora_trainer import` | WIRED | Lines 1005 (train-lora) and 1108 (lora-status): both import ReplicateBackend and relevant functions |
| `ActorLibraryDetail.tsx` | `client.ts` | `trainActorLora()` and `getActorLoraStatus()` calls | WIRED | Lines 26-27 (imports), 207 (getActorLoraStatus called in polling), 224 (trainActorLora called in handleTrainLora) |
| `client.ts` | `asset_library.py` | POST/GET to `/api/asset-library/actors/{id}/train-lora` and `.../lora-status` | WIRED | Lines 1604, 1614: fetch calls to correct API paths |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| LORA-01 | 25-01 | Actor model extended with lora_url, lora_trained_at, lora_training_status | SATISFIED | All 4 ORM columns present in Actor; 6 ALTER TABLE migrations in db/__init__.py |
| LORA-02 | 25-01 | lora_trainer.py service with dataset preparation, pluggable backend, job dispatch | SATISFIED | 345-line service with ABC, ReplicateBackend, prepare_dataset, download_and_store_weights |
| LORA-03 | 25-02 | POST /api/asset-library/actors/{id}/train-lora validates min refs and dispatches | SATISFIED | Endpoint at line 996; validates ref_count >= 5 (422 if not), checks replicate_api_token (422 if not), returns 202 |
| LORA-04 | 25-02 | GET /api/asset-library/actors/{id}/lora-status returns training status and LoRA URL | SATISFIED | Endpoint at line 1098; polls Replicate for non-terminal states, downloads weights on completion |
| LORA-05 | 25-02 | Frontend "Train Identity Model" button (enabled when refs >= 5) and status badge | SATISFIED | Button at lines 380-397 with disable logic for refs < 5 / in-progress; LoraStatusBadge at lines 427-471 with all 4 states |

No orphaned requirements — all 5 LORA requirements are claimed by plans and implemented.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None found |

No stubs, placeholders, empty implementations, or TODO/FIXME comments detected in any phase-modified file.

---

### Human Verification Required

#### 1. Replicate API End-to-End Training Flow

**Test:** With a real Replicate API token configured in UserSettings, add 5+ reference images to an Actor, click "Train Identity Model", and observe the status progression through QUEUED → TRAINING → COMPLETED.
**Expected:** Training job appears on Replicate dashboard; status badge on Actor detail polls and updates every 10 seconds; upon completion, `lora_url` is set on Actor with storage key for `.safetensors` weights.
**Why human:** Requires a live Replicate account and real API credentials; involves external async job lifecycle that cannot be verified from static code analysis.

#### 2. Dataset Preparation Quality

**Test:** Trigger training for an Actor with 5+ reference images and inspect the zip file uploaded to storage.
**Expected:** Each image is resized to 1024x1024 with white padding; each `.txt` caption begins with the trigger word (`ACTOR_{NAME}`) and contains a detailed appearance description.
**Why human:** VLM caption quality and resize correctness on real production images requires visual inspection; unit test covers the resize logic but not caption quality.

#### 3. Frontend < 5 Refs Disabled State

**Test:** Open an Actor with fewer than 5 reference images and inspect the "Train Identity Model" button.
**Expected:** Button is visually disabled (gray) and helper text "Need at least 5 reference images (N/5)" is shown below.
**Why human:** Visual/UI behavior requires browser rendering to confirm.

---

### Gaps Summary

No gaps found. All 11 observable truths are verified, all 8 artifacts are substantive and wired, all 5 key links are confirmed, and all 5 requirement IDs (LORA-01 through LORA-05) are satisfied by actual code in the codebase.

The implementation matches the plan specifications exactly: Actor model has 4 LoRA columns with migrations, UserSettings has 2 Replicate configuration columns with migrations, tag resolver passes `actor.lora_url` through via `getattr` for backward compatibility, `lora_trainer.py` implements the ABC and ReplicateBackend with `asyncio.to_thread()` wrapping, dataset preparation (resize + VLM caption + zip) is fully implemented, both API endpoints are registered and wired to the service layer, and the frontend UI includes the Train Identity Model button with status polling.

TypeScript compilation passes with zero errors.

---

_Verified: 2026-03-14T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
