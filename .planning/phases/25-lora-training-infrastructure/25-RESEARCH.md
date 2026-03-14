# Phase 25: LoRA Training Infrastructure - Research

**Researched:** 2026-03-14
**Domain:** LoRA fine-tuning via Replicate API, Actor model extension, dataset preparation, async job management
**Confidence:** HIGH

## Summary

This phase adds per-actor LoRA training capability to the Asset Library. The core work involves: (1) extending the Actor model with three new columns for LoRA tracking, (2) building a dataset preparation pipeline that downloads actor reference images, resizes them, and captions them via VLM, (3) implementing a pluggable training backend abstraction with Replicate API as initial backend, (4) adding two API endpoints for training dispatch and status polling, and (5) a frontend "Train Identity Model" button with status badge on the Actor detail view.

The existing codebase already has the `lora_url` field stubbed as `None` in `ResolvedAssetRef` (tag_resolver.py line 54) and the Flux pipeline already reads `aref.lora_url` for ComfyUI workflow selection (keyframes.py line 845). This means once this phase populates `Actor.lora_url` and the tag resolver passes it through, the LoRA will automatically flow into image generation -- no changes to the Flux pipeline are needed.

**Primary recommendation:** Use `replicate` Python package v1.0.7 with `ostris/flux-dev-lora-trainer` model. Store the Replicate API token on `UserSettings` (same pattern as ElevenLabs/ComfyUI keys). Dataset prep should resize to 1024x1024 (Flux native resolution) and use the existing VLM adapter for auto-captioning. Polling should be one-shot (check on user request) rather than background continuous polling, matching the simpler architecture of this codebase.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Three new columns on Actor: `lora_url` (nullable str, S3 path to .safetensors), `lora_trained_at` (nullable datetime), `lora_training_status` (nullable str: QUEUED/TRAINING/COMPLETED/FAILED)
- Alembic migration or conditional column add (per project convention for SQLite)
- Similar fields on `LibraryProp` deferred to future -- only Actor gets LoRA for now
- New `lora_trainer.py` in `backend/vidpipe/services/`
- Pluggable backend interface: `LoRATrainingBackend` abstract base with `dispatch()`, `poll_status()`, `get_result()`
- Initial implementation: `ReplicateBackend` using Replicate API
- Service handles: dataset prep -> dispatch -> status polling -> result storage
- Async-first (all I/O uses async def + await)
- Download all ActorRef images for the actor
- Resize to 512x512 or 768x768 (maintain aspect ratio with padding)
- Generate captions via VLM (existing LLM adapter pattern -- use vision_model)
- Caption format: describe appearance without name, add trigger word (`ACTOR_{TAG}`) to subset
- Package as zip, upload to S3 for training worker
- Minimum 5 reference images to enable training (button disabled below 5)
- `POST /api/asset-library/actors/{id}/train-lora` -- validates min refs, dispatches job, returns job status
- `GET /api/asset-library/actors/{id}/lora-status` -- returns current training status, progress, lora_url when complete
- Both in existing `asset_library.py` route file (per project convention: split by domain)
- Use `replicate` Python package (new dependency)
- Model: `lucataco/simpletuner-flux` or equivalent Flux LoRA training model
- Input: zip of captioned images + trigger word + training config
- Output: .safetensors file URL -> download to S3 -> update Actor.lora_url
- Polling via Replicate's prediction status API
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

### Deferred Ideas (OUT OF SCOPE)
- LibraryProp LoRA training (similar pattern, lower priority) -> Future
- ComfyUI Flux Trainer custom node as alternative backend -> Future
- Local GPU worker backend -> Future
- RunPod/Lambda Labs on-demand backend -> Future
- LoRA versioning and rollback -> Future
- Automatic LoRA invalidation when refs updated -> Future
- LoRA merging for multi-character shots -> Future
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LORA-01 | Actor model extended with lora_url, lora_trained_at, lora_training_status | Migration pattern documented (ALTER TABLE in _run_migrations), column types verified against existing Actor model |
| LORA-02 | lora_trainer.py service with dataset prep, pluggable backend, job dispatch | Replicate API training lifecycle documented, VLM captioning pattern from generate-metadata endpoint, PIL resize pattern from existing _generate_thumbnail |
| LORA-03 | POST train-lora endpoint validates min refs and dispatches | Endpoint pattern from existing actor endpoints in asset_library.py, UserSettings API key pattern documented |
| LORA-04 | GET lora-status endpoint returns status, progress, lora_url | Replicate training status values documented (starting/processing/succeeded/failed/canceled), one-shot poll pattern recommended |
| LORA-05 | Frontend Train button (refs >= 5) and status badge | Actor interface and ActorLibraryDetail.tsx structure documented, tab layout pattern identified |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `replicate` | 1.0.7 | Replicate API client for training dispatch and polling | Official Python SDK, async support, handles auth/webhooks |
| `Pillow` | >=10.0 | Image resize/padding for dataset prep | Already in project dependencies |
| `httpx` | >=0.27.0 | Download trained weights from Replicate output URL | Already in project dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `zipfile` (stdlib) | - | Package captioned images into training zip | Dataset preparation step |
| `io.BytesIO` (stdlib) | - | In-memory image/zip buffers | Dataset prep without temp files |
| `tempfile` (stdlib) | - | Temporary zip file before S3 upload | If zip exceeds memory budget |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `ostris/flux-dev-lora-trainer` | `replicate/fast-flux-trainer` | Replicate's own trainer is faster (~2 min) but same underlying ostris/ai-toolkit |
| `ostris/flux-dev-lora-trainer` | `lucataco/simpletuner-flux` | CONTEXT.md mentions lucataco; ostris is more widely documented and actively maintained |
| One-shot polling | Background task polling | Background is more responsive but adds complexity; one-shot is simpler and matches existing codebase patterns |

**Installation:**
```bash
pip install replicate>=1.0.0
```

Add to `backend/pyproject.toml` dependencies:
```
"replicate>=1.0.0",
```

## Architecture Patterns

### Recommended Project Structure
```
backend/vidpipe/
├── services/
│   └── lora_trainer.py          # Training service + backends
├── api/
│   └── asset_library.py         # Add train-lora + lora-status endpoints
├── db/
│   ├── models.py                # Actor model: +3 columns
│   └── __init__.py              # Migration: +3 ALTER TABLE statements
└── schemas/
    └── lora_training.py         # Pydantic schemas for captions (optional, can inline)
frontend/src/
├── components/
│   └── ActorLibraryDetail.tsx   # Add LoRA section to overview/refs tab
└── api/
    ├── client.ts                # Add trainActorLora(), getActorLoraStatus()
    └── types.ts                 # Extend Actor interface with lora fields
```

### Pattern 1: Pluggable Training Backend (ABC)
**What:** Abstract base class defining the training backend interface
**When to use:** Dispatch, poll, and retrieve results from any training provider
**Example:**
```python
# Source: Matches project's LLMAdapter pattern in services/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingJob:
    job_id: str
    status: str  # QUEUED, TRAINING, COMPLETED, FAILED
    progress: Optional[float] = None  # 0.0-1.0 if available
    weights_url: Optional[str] = None  # URL to download .safetensors
    error: Optional[str] = None
    logs: Optional[str] = None

class LoRATrainingBackend(ABC):
    @abstractmethod
    async def dispatch(
        self,
        dataset_url: str,
        trigger_word: str,
        *,
        steps: int = 1000,
        lora_rank: int = 32,
        learning_rate: float = 0.0004,
    ) -> TrainingJob:
        """Submit training job. Returns job with job_id and QUEUED status."""
        ...

    @abstractmethod
    async def poll_status(self, job_id: str) -> TrainingJob:
        """Check current status of a training job."""
        ...

    @abstractmethod
    async def get_result(self, job_id: str) -> bytes:
        """Download the trained weights as bytes."""
        ...
```

### Pattern 2: Replicate Backend Implementation
**What:** Concrete backend using replicate Python SDK
**When to use:** Initial and primary training backend
**Example:**
```python
# Source: Replicate HTTP API docs + Python SDK
import replicate

class ReplicateBackend(LoRATrainingBackend):
    TRAINER_VERSION = "ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497"

    def __init__(self, api_token: str):
        self._client = replicate.Client(api_token=api_token)

    async def dispatch(self, dataset_url, trigger_word, *, steps=1000, lora_rank=32, learning_rate=0.0004):
        # Note: replicate SDK trainings.create is sync; wrap in to_thread
        import asyncio
        training = await asyncio.to_thread(
            self._client.trainings.create,
            version=self.TRAINER_VERSION,
            input={
                "input_images": dataset_url,
                "trigger_word": trigger_word,
                "steps": steps,
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "autocaption": False,  # We provide our own captions
                "resolution": "1024",
            },
            destination=f"vidpipe/lora-{trigger_word.lower()}",
        )
        return TrainingJob(job_id=training.id, status="QUEUED")

    async def poll_status(self, job_id):
        import asyncio
        training = await asyncio.to_thread(self._client.trainings.get, job_id)
        status_map = {
            "starting": "QUEUED",
            "processing": "TRAINING",
            "succeeded": "COMPLETED",
            "failed": "FAILED",
            "canceled": "FAILED",
        }
        weights_url = None
        if training.status == "succeeded" and training.output:
            weights_url = getattr(training.output, "weights", None) or training.output.get("weights")
        return TrainingJob(
            job_id=job_id,
            status=status_map.get(training.status, "TRAINING"),
            weights_url=weights_url,
            error=training.error,
            logs=training.logs,
        )
```

### Pattern 3: Dataset Preparation Pipeline
**What:** Download ActorRef images, resize, caption via VLM, package as zip
**When to use:** Before dispatching training job
**Example:**
```python
# Source: Follows existing generate_actor_metadata pattern in asset_library.py
async def prepare_dataset(
    actor_id: uuid.UUID,
    session: AsyncSession,
    vision_adapter: LLMAdapter,
    file_mgr: FileManager,
) -> tuple[bytes, str]:
    """Prepare training dataset. Returns (zip_bytes, trigger_word)."""
    actor = await session.get(Actor, actor_id)
    refs = (await session.execute(
        select(ActorRef).where(ActorRef.actor_id == actor_id)
    )).scalars().all()

    trigger_word = f"ACTOR_{actor.name.upper().replace(' ', '_')}"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, ref in enumerate(refs):
            image_bytes = await file_mgr.read_bytes(ref.image_url)
            resized = await asyncio.to_thread(_resize_with_padding, image_bytes, 1024)
            # Caption via VLM
            caption = await _generate_caption(vision_adapter, resized, trigger_word)
            # Write image + caption with matching filenames
            zf.writestr(f"{i:03d}.png", resized)
            zf.writestr(f"{i:03d}.txt", caption)
    return zip_buffer.getvalue(), trigger_word
```

### Pattern 4: Actor Model Column Addition (Migration)
**What:** Conditional ALTER TABLE for three new columns
**When to use:** Database initialization
**Example:**
```python
# Source: Existing migration pattern in db/__init__.py
# Add to _run_migrations() migrations list:
"ALTER TABLE actors ADD COLUMN lora_url TEXT",
"ALTER TABLE actors ADD COLUMN lora_trained_at TIMESTAMP",
"ALTER TABLE actors ADD COLUMN lora_training_status VARCHAR(20)",
# Also: lora_training_job_id for tracking the Replicate training ID
"ALTER TABLE actors ADD COLUMN lora_training_job_id VARCHAR(200)",
```

### Pattern 5: Tag Resolver Integration
**What:** Pass `actor.lora_url` through to `ResolvedAssetRef.lora_url`
**When to use:** When building CHARACTER resolution in tag_resolver.py
**Example:**
```python
# Source: tag_resolver.py line 511 (currently hardcoded to None)
# Change from:
lora_url=None,
# Change to:
lora_url=actor.lora_url if hasattr(actor, 'lora_url') else None,
```

### Anti-Patterns to Avoid
- **Background polling loop in the API process:** Do NOT start a background asyncio task that continuously polls Replicate. Training takes 2-30 minutes; a continuous poll wastes resources. Use one-shot status check on user request.
- **Storing Replicate API token in config.yaml:** API keys belong in `UserSettings` table (per-user), matching the existing pattern for ElevenLabs and ComfyUI keys.
- **Synchronous Replicate calls blocking the event loop:** The `replicate` Python SDK is synchronous; always wrap in `asyncio.to_thread()`.
- **Re-captioning on every training attempt:** Cache captions alongside refs if training is retried; VLM calls are expensive.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Training dispatch + polling | Custom HTTP client to Replicate | `replicate` Python SDK | Handles auth, retries, file uploads, status parsing |
| Image resize with padding | Custom numpy/cv2 resize | `PIL.Image.resize()` with `ImageOps.pad()` | Correct aspect ratio handling, LANCZOS filter already used in project |
| Zip creation | Custom binary format | `zipfile.ZipFile` (stdlib) | Industry standard, no dependencies, Replicate expects zip |
| VLM captioning | Custom prompt + raw API | Existing `LLMAdapter.analyze_image()` | Already handles retries, structured output, multi-provider |
| Image download from storage | Raw file reads | Existing `FileManager.read_bytes()` | Handles both local paths and S3 keys transparently |

**Key insight:** The dataset preparation pipeline is the most complex part, but every individual operation (image download, resize, VLM caption, zip, S3 upload) already has an established pattern in the codebase. The novelty is composing them together.

## Common Pitfalls

### Pitfall 1: Replicate SDK is Synchronous
**What goes wrong:** Calling `replicate.trainings.create()` or `.get()` directly in an async function blocks the event loop.
**Why it happens:** The `replicate` Python SDK (v1.0.x) uses synchronous HTTP under the hood. The `async_create()` method had bugs (issue #408, fixed Feb 2025) but the sync methods are battle-tested.
**How to avoid:** Always wrap in `asyncio.to_thread()`: `await asyncio.to_thread(client.trainings.create, ...)`.
**Warning signs:** Slow API responses, timeouts on other endpoints while training dispatches.

### Pitfall 2: Replicate Training Requires a Destination Model
**What goes wrong:** Training fails with 422 because no `destination` parameter was provided.
**Why it happens:** Replicate trainings create a new version of a model; you must specify a destination model in `{owner}/{name}` format.
**How to avoid:** Create a destination model on Replicate first (can be done via API), or use a pre-existing model the user owns. The `destination` parameter is required.
**Warning signs:** 422 errors from Replicate API on training create.

### Pitfall 3: Image Format and Resolution Mismatch
**What goes wrong:** Training produces poor results because images were not properly preprocessed.
**Why it happens:** Flux LoRA training expects 1024x1024 (or at least square) images. The CONTEXT.md says 512x512 or 768x768, but Flux native resolution is 1024x1024. Using smaller images means the trainer must upscale, losing detail.
**How to avoid:** Resize to 1024x1024 with center-padding (not stretch). Use `PIL.ImageOps.pad()` with white/black fill. CONTEXT.md says 512-768; recommend 1024 as that's what the trainer expects for optimal Flux results. The ostris trainer has a `resolution` parameter (default "512,768,1024") so even smaller images work, but 1024 is best.
**Warning signs:** Blurry or distorted training outputs.

### Pitfall 4: Trigger Word Collision
**What goes wrong:** LoRA activates unintentionally in prompts because the trigger word is a common English word.
**Why it happens:** Using something like "Brandon" as trigger word means every mention of that word activates the LoRA.
**How to avoid:** Use the pattern `ACTOR_BRANDON` -- all-caps with prefix. This will never appear naturally in a prompt. The VLM captioner must include this trigger word in captions.
**Warning signs:** LoRA style bleeding into unrelated generations.

### Pitfall 5: Replicate Training Status is Not Real-Time
**What goes wrong:** User sees "Training..." badge indefinitely because the status was never checked again.
**Why it happens:** One-shot polling means status only updates when the user explicitly checks. If user navigates away and comes back, status is stale from DB.
**How to avoid:** The `GET /lora-status` endpoint should FIRST check if status is QUEUED/TRAINING (non-terminal), then poll Replicate for latest status, update DB, and return. Only skip the Replicate poll if status is already terminal (COMPLETED/FAILED).
**Warning signs:** Stale "Training..." status that never resolves.

### Pitfall 6: UserSettings Replicate Token Not Configured
**What goes wrong:** Training endpoint returns 500 because there's no Replicate API token.
**Why it happens:** New dependency requires new API key; user hasn't configured it.
**How to avoid:** Check for token early in the endpoint, return clear 422 with message "Replicate API token not configured. Go to Settings to add it." Do NOT attempt training without a token.
**Warning signs:** Cryptic errors from the replicate SDK about authentication.

## Code Examples

Verified patterns from official sources and existing codebase:

### Image Resize with Padding (PIL)
```python
# Source: PIL documentation + existing _generate_thumbnail pattern
from PIL import Image, ImageOps
import io

def _resize_with_padding(image_bytes: bytes, target_size: int = 1024) -> bytes:
    """Resize image to target_size x target_size with white padding."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # pad() maintains aspect ratio and adds fill color
    img = ImageOps.pad(img, (target_size, target_size), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

### VLM Caption Generation
```python
# Source: Existing generate_actor_metadata pattern in asset_library.py
from pydantic import BaseModel

class ImageCaption(BaseModel):
    caption: str

async def _generate_caption(
    adapter: LLMAdapter,
    image_bytes: bytes,
    trigger_word: str,
) -> str:
    """Generate training caption for an image using VLM."""
    prompt = f"""Describe this person's appearance in detail for AI image training.
Include: gender, age range, body type, hair style/color, skin tone, facial features,
clothing, pose, expression. Do NOT include any names.
Start the caption with the token: {trigger_word}
Example: "{trigger_word}, a woman in her 30s with long brown hair..."
Return JSON with a single "caption" field."""

    result = await adapter.analyze_image(
        image_bytes=image_bytes,
        prompt=prompt,
        schema=ImageCaption,
        mime_type="image/png",
        temperature=0.3,
    )
    return result.caption
```

### Replicate Training Create (with asyncio.to_thread)
```python
# Source: Replicate docs + Python SDK
import asyncio
import replicate

async def dispatch_training(api_token: str, dataset_url: str, trigger_word: str):
    client = replicate.Client(api_token=api_token)
    training = await asyncio.to_thread(
        client.trainings.create,
        version="ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
        input={
            "input_images": dataset_url,
            "trigger_word": trigger_word,
            "steps": 1000,
            "lora_rank": 32,
            "learning_rate": 0.0004,
            "autocaption": False,
            "resolution": "512,768,1024",
        },
        destination="your-username/actor-lora",
    )
    return training.id  # str: the training ID for polling
```

### Replicate Training Status Poll
```python
# Source: Replicate HTTP API reference
async def check_training(api_token: str, training_id: str):
    client = replicate.Client(api_token=api_token)
    training = await asyncio.to_thread(client.trainings.get, training_id)
    # training.status: "starting" | "processing" | "succeeded" | "failed" | "canceled"
    # training.output: {"version": "...", "weights": "https://..."} on success
    # training.error: str on failure
    # training.logs: str (training logs)
    return training
```

### Download Weights and Store to S3
```python
# Source: Existing storage_backend pattern
async def download_and_store_weights(
    weights_url: str,
    actor_id: str,
    storage: StorageBackend,
) -> str:
    """Download .safetensors from Replicate and store in S3/local."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(weights_url)
        resp.raise_for_status()
        weights_bytes = resp.content

    key = f"asset-library/actors/{actor_id}/lora/model.safetensors"
    await storage.put(key, weights_bytes, "application/octet-stream")
    return key
```

### Actor Model Columns (ORM)
```python
# Source: Existing Actor class in models.py
# Add after existing columns:
lora_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
lora_trained_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
lora_training_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
lora_training_job_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SDXL LoRA training (DreamBooth) | Flux LoRA training (AI-Toolkit/SimpleTuner) | Mid-2024 | Flux produces significantly better identity preservation with fewer images |
| 20-50+ training images | 5-20 images sufficient | 2024 with Flux | Lower barrier to entry for per-actor LoRA |
| Local GPU training (hours) | Cloud API training (2-30 min) | 2024 Replicate/cloud | No GPU infrastructure needed |
| Manual captioning | Auto-captioning via VLM | 2024 | Eliminates tedious manual labeling |

**Deprecated/outdated:**
- `lucataco/simpletuner-flux`: Still works but `ostris/flux-dev-lora-trainer` is more actively maintained and is the canonical Replicate recommendation
- `replicate.trainings.async_create()`: Had bugs in early 2025 (issue #408), fixed in Feb 2025. Still recommend wrapping sync methods in `to_thread` for reliability.

## Open Questions

1. **Replicate Destination Model**
   - What we know: Replicate trainings require a `destination` parameter in `{owner}/{name}` format. The user needs a Replicate account.
   - What's unclear: Should we create a destination model programmatically per actor, or use a single shared destination model?
   - Recommendation: Store the Replicate `destination` as part of the training config. For simplicity, use a convention like `{replicate_username}/vidpipe-actor-{actor_tag}`. The endpoint can create the model if it doesn't exist (Replicate API supports model creation).

2. **Replicate API Token Storage**
   - What we know: Other API keys (ElevenLabs, ComfyUI) are stored on `UserSettings`.
   - What's unclear: Does Replicate need a separate "destination username" in addition to the token?
   - Recommendation: Store `replicate_api_token` on UserSettings. The destination username can be derived from the Replicate account associated with the token (via `replicate.models.list()` or hardcoded).

3. **Dataset Image Size**
   - What we know: CONTEXT.md says 512x512 or 768x768. Replicate's ostris trainer accepts multi-resolution ("512,768,1024"). Flux native is 1024x1024.
   - What's unclear: What size produces the best results for face identity training?
   - Recommendation: Use 1024x1024 (Flux native resolution) for best quality. The trainer handles it natively. If bandwidth is a concern, 768x768 is a reasonable compromise.

4. **Training Steps for Face vs Full Body**
   - What we know: PRD recommends 1000-1500 for face, 1500-2000 for full body. Community consensus is ~40 steps per image.
   - What's unclear: How to detect whether refs are face-only vs full-body?
   - Recommendation: Default to 1000 steps. Allow override via training config. Do not auto-detect face vs body in v1.

## Sources

### Primary (HIGH confidence)
- [Replicate HTTP API reference](https://replicate.com/docs/reference/http) - Training lifecycle states: starting/processing/succeeded/failed/canceled, create/get/cancel endpoints, output structure with `weights` URL
- [Replicate Python SDK v1.0.7](https://pypi.org/project/replicate/) - Current version, Python 3.8+ requirement, `Client.trainings.create()` and `.get()` methods
- [ostris/flux-dev-lora-trainer](https://replicate.com/ostris/flux-dev-lora-trainer/train) - Training parameters: input_images, trigger_word, steps, lora_rank, learning_rate, autocaption, resolution
- Existing codebase: `tag_resolver.py` (ResolvedAssetRef.lora_url stub), `keyframes.py` (Flux pipeline LoRA consumption), `asset_library.py` (actor CRUD patterns), `db/__init__.py` (migration pattern)

### Secondary (MEDIUM confidence)
- [Replicate fine-tune Flux blog](https://replicate.com/blog/fine-tune-flux) - Training cost ~$1.46, ~2 min for 1000 steps, 10+ images recommended
- [Replicate fine-tune Flux with faces blog](https://replicate.com/blog/fine-tune-flux-with-faces) - Face-specific training tips
- [Replicate fine-tune with API blog](https://replicate.com/blog/fine-tune-flux-with-an-api) - API workflow examples
- [Training a Personal LoRA](https://www.pelayoarbues.com/notes/Training-a-Personal-LoRA-on-Replicate-Using-FLUX.1-dev) - Hyperparameter recommendations: steps=1040, lora_rank=32, learning_rate=0.0004

### Tertiary (LOW confidence)
- [replicate-python issue #408](https://github.com/replicate/replicate-python/issues/408) - async_create bug, reportedly fixed Feb 2025. Still recommend sync+to_thread for safety.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Replicate SDK is well-documented, existing codebase patterns clear
- Architecture: HIGH - Follows established ABC pattern (like LLMAdapter), existing service patterns
- Pitfalls: HIGH - Verified against Replicate docs and real-world training examples
- Dataset prep: MEDIUM - Image size recommendation based on community consensus, not A/B tested
- Training hyperparameters: MEDIUM - Based on community recommendations, optimal values are model-specific

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (Replicate models update frequently; verify trainer version before implementation)
