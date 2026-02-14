# Architecture Research

**Domain:** AI Video Generation Pipeline
**Researched:** 2026-02-14
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

AI video generation pipelines in 2026 follow a **multi-stage, state-driven architecture** with async API orchestration and crash-safe state persistence. The dominant pattern is:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐         │
│  │ CLI/API     │  │ State Machine│  │ Job Queue   │         │
│  │ Interface   │──│ Controller   │──│ Manager     │         │
│  └─────────────┘  └──────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                   GENERATION PIPELINE                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Storyboard│→ │ Keyframe │→ │  Video   │→ │  Stitch  │    │
│  │Generator │  │Generator │  │Generator │  │  Engine  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    PERSISTENCE LAYER                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         State Store (SQLite + SQLAlchemy)           │    │
│  │  • Job state (STORYBOARD → KEYFRAMES → VIDEO_GEN    │    │
│  │              → STITCH → COMPLETE)                   │    │
│  │  • Generation metadata (prompts, params, retries)   │    │
│  │  • Artifact references (filesystem paths)           │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                     ARTIFACT STORAGE                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Storyboard│  │ Keyframe │  │  Video   │  │  Final   │    │
│  │   JSON   │  │  Images  │  │  Clips   │  │  Output  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    EXTERNAL SERVICES                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Google Vertex AI (Async APIs)               │   │
│  │  • Gemini 3 Pro (LLM for storyboarding)              │   │
│  │  • Nano Banana Pro (image generation for keyframes)  │   │
│  │  • Veo 3.1 (video generation from keyframes)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **CLI Interface** | User commands (start, resume, status, cancel), parameter input, progress display | Typer with rich progress bars |
| **API Server** | REST endpoints for job management, async job submission, webhook handling | FastAPI with background tasks |
| **State Machine Controller** | Transition validation, state persistence, crash recovery, retry logic | SQLAlchemy models with state column + transitions library |
| **Storyboard Generator** | Parse text prompt into multi-scene narrative structure, define scene descriptions and transitions | LLM (Gemini) with structured output (JSON) |
| **Keyframe Generator** | Generate images for each scene, maintain visual continuity (character appearance, setting), sequential generation to enforce continuity | Image model (Nano Banana Pro) with continuity prompts |
| **Video Generator** | Convert keyframe pairs to video clips, handle temporal interpolation, potentially parallel generation | Video model (Veo 3.1) with frame-guided generation |
| **Stitch Engine** | Concatenate video clips, handle scene transitions, apply temporal smoothing at boundaries | FFmpeg or MoviePy with transition effects |
| **Artifact Store** | Save/retrieve generated assets, manage filesystem paths, track artifact-to-job relationships | Local filesystem with structured directories + DB references |
| **Job Queue** | Async task management (for API mode), retry with exponential backoff, priority handling | asyncio.Queue or Taskiq for production |

## Recommended Project Structure

```
video-pipeline/
├── src/
│   ├── cli/                  # Typer CLI commands
│   │   ├── __init__.py
│   │   ├── commands.py       # start, resume, status, cancel
│   │   └── display.py        # Rich progress bars, status formatting
│   ├── api/                  # FastAPI server (optional)
│   │   ├── __init__.py
│   │   ├── routes.py         # Job endpoints
│   │   └── middleware.py     # Auth, logging
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state_machine.py  # State transition logic
│   │   ├── orchestrator.py   # Pipeline coordinator
│   │   └── config.py         # Configuration management
│   ├── generators/           # Generation pipeline stages
│   │   ├── __init__.py
│   │   ├── storyboard.py     # LLM-based scene planning
│   │   ├── keyframes.py      # Image generation with continuity
│   │   ├── video.py          # Video clip generation
│   │   └── stitcher.py       # Video concatenation
│   ├── models/               # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── job.py            # Job state, metadata
│   │   ├── artifact.py       # Artifact references
│   │   └── schema.py         # Pydantic schemas
│   ├── services/             # External API clients
│   │   ├── __init__.py
│   │   ├── vertex_ai.py      # Vertex AI SDK wrapper
│   │   ├── gemini.py         # Gemini LLM client
│   │   ├── nano_banana.py    # Image generation client
│   │   └── veo.py            # Video generation client
│   ├── storage/              # Artifact management
│   │   ├── __init__.py
│   │   ├── filesystem.py     # Local file operations
│   │   └── database.py       # DB session management
│   └── utils/
│       ├── __init__.py
│       ├── retry.py          # Exponential backoff
│       ├── validation.py     # Input validation
│       └── logging.py        # Structured logging
├── artifacts/                # Local artifact storage
│   ├── jobs/
│   │   └── {job_id}/
│   │       ├── storyboard.json
│   │       ├── keyframes/
│   │       │   ├── scene_001.png
│   │       │   └── scene_002.png
│   │       ├── videos/
│   │       │   ├── clip_001.mp4
│   │       │   └── clip_002.mp4
│   │       └── output/
│   │           └── final.mp4
├── data/
│   └── pipeline.db           # SQLite database
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .env                      # API keys, config
├── pyproject.toml
└── README.md
```

### Structure Rationale

- **generators/:** Each pipeline stage is isolated. Storyboard → Keyframes → Video → Stitch maps directly to state transitions.
- **services/:** External API clients are abstracted. Easy to mock for testing, swap implementations (Vertex AI → local models).
- **models/:** SQLAlchemy models separate from business logic. Pydantic schemas for validation and API contracts.
- **storage/:** Artifact management centralized. Database tracks metadata, filesystem stores binary assets.
- **artifacts/{job_id}/:** Job-scoped directories prevent collisions, enable atomic cleanup, support parallel job execution.

## Architectural Patterns

### Pattern 1: State Machine with Persistent State

**What:** Each job progresses through explicit states (STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE) with transitions stored in the database.

**When to use:** Pipelines with multiple async stages, long-running jobs requiring resume capability, external API dependencies prone to failure.

**Trade-offs:**
- **Pros:** Crash-safe (resume from last successful state), explicit failure modes (know exactly where it failed), easy to add retry logic per stage
- **Cons:** More complex than linear scripts, requires database for state persistence, state transition validation adds code

**Example:**
```python
from sqlalchemy import Column, String, Integer, DateTime, Enum
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()

class JobState(str, enum.Enum):
    PENDING = "PENDING"
    STORYBOARD = "STORYBOARD"
    KEYFRAMES = "KEYFRAMES"
    VIDEO_GEN = "VIDEO_GEN"
    STITCH = "STITCH"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    state = Column(Enum(JobState), default=JobState.PENDING)
    prompt = Column(String, nullable=False)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    error_message = Column(String, nullable=True)
    retry_count = Column(Integer, default=0)

    # State transition with validation
    def transition_to(self, new_state: JobState):
        valid_transitions = {
            JobState.PENDING: [JobState.STORYBOARD, JobState.FAILED],
            JobState.STORYBOARD: [JobState.KEYFRAMES, JobState.FAILED],
            JobState.KEYFRAMES: [JobState.VIDEO_GEN, JobState.FAILED],
            JobState.VIDEO_GEN: [JobState.STITCH, JobState.FAILED],
            JobState.STITCH: [JobState.COMPLETE, JobState.FAILED],
        }

        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(f"Invalid transition: {self.state} -> {new_state}")

        self.state = new_state
        self.updated_at = datetime.utcnow()
```

### Pattern 2: Async API with Polling

**What:** Submit generation requests to async APIs (Vertex AI Veo), receive job ID, poll until complete. Never block on sync waits.

**When to use:** Video generation (30s-5min), image generation (5-30s), LLM calls with long context (10-60s).

**Trade-offs:**
- **Pros:** Handles API timeouts gracefully, can monitor multiple jobs concurrently, resilient to network issues
- **Cons:** More complex than sync calls, requires polling logic, adds latency (poll interval trade-off)

**Example:**
```python
import asyncio
from typing import Optional

class VeoClient:
    async def submit_video_generation(self, prompt: str, keyframes: dict) -> str:
        """Submit async video generation job, return job_id"""
        response = await self.vertex_ai.generate_video_async(
            prompt=prompt,
            first_frame=keyframes["start"],
            last_frame=keyframes["end"],
        )
        return response["job_id"]

    async def poll_until_complete(
        self,
        job_id: str,
        poll_interval: int = 10,
        max_wait: int = 300
    ) -> Optional[bytes]:
        """Poll video generation job until complete or timeout"""
        elapsed = 0

        while elapsed < max_wait:
            status = await self.vertex_ai.get_job_status(job_id)

            if status["state"] == "COMPLETE":
                return status["video_bytes"]
            elif status["state"] == "FAILED":
                raise Exception(f"Video generation failed: {status['error']}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Video generation exceeded {max_wait}s")
```

### Pattern 3: Sequential Generation with Continuity Context

**What:** Generate keyframes sequentially (not parallel) to maintain visual continuity. Pass previous keyframe as context to next generation.

**When to use:** Multi-scene videos requiring character consistency, setting consistency, narrative coherence.

**Trade-offs:**
- **Pros:** Strong visual continuity, characters/settings remain consistent, narrative coherence maintained
- **Cons:** Slower than parallel generation, one failure blocks downstream scenes, can't parallelize

**Example:**
```python
class KeyframeGenerator:
    async def generate_keyframes_sequential(
        self,
        storyboard: dict
    ) -> list[bytes]:
        """Generate keyframes with continuity by sequential generation"""
        keyframes = []
        previous_keyframe = None

        for scene in storyboard["scenes"]:
            # Build prompt with continuity context from previous scene
            prompt = self._build_continuity_prompt(
                scene_description=scene["description"],
                previous_keyframe=previous_keyframe,
                character_refs=scene["characters"],
                setting=scene["setting"]
            )

            # Generate keyframe (blocks until complete)
            keyframe = await self.nano_banana_client.generate_image(prompt)
            keyframes.append(keyframe)

            # Store for next iteration
            previous_keyframe = keyframe

        return keyframes

    def _build_continuity_prompt(
        self,
        scene_description: str,
        previous_keyframe: Optional[bytes],
        character_refs: list[str],
        setting: str
    ) -> str:
        """Enhance prompt with continuity instructions"""
        continuity_context = []

        if previous_keyframe:
            # Reference previous frame for consistency
            continuity_context.append(
                "Maintain character appearance and style from previous scene."
            )

        if character_refs:
            continuity_context.append(
                f"Characters present: {', '.join(character_refs)}"
            )

        context = " ".join(continuity_context)
        return f"{scene_description}. {context}"
```

### Pattern 4: Artifact Path References in Database

**What:** Store binary artifacts (images, videos) on filesystem, store filesystem paths + metadata in database.

**When to use:** Large binary assets (videos, high-res images), need to track artifact lineage, want DB queries on metadata without loading binaries.

**Trade-offs:**
- **Pros:** DB stays small (no BLOBs), fast queries on metadata, easy to backup/restore separately, supports cloud storage migration
- **Cons:** Two storage systems to manage, path integrity requires validation, cleanup must handle both DB and filesystem

**Example:**
```python
class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    artifact_type = Column(String)  # "storyboard", "keyframe", "video_clip", "final"
    file_path = Column(String, nullable=False)  # artifacts/jobs/{job_id}/...
    file_size = Column(Integer)
    mime_type = Column(String)
    created_at = Column(DateTime)

    def exists(self) -> bool:
        """Verify artifact file exists on filesystem"""
        return os.path.exists(self.file_path)

    def read(self) -> bytes:
        """Load artifact from filesystem"""
        with open(self.file_path, "rb") as f:
            return f.read()

class ArtifactStore:
    def save_artifact(
        self,
        job_id: int,
        artifact_type: str,
        content: bytes,
        filename: str
    ) -> Artifact:
        """Save artifact to filesystem and create DB record"""
        # Filesystem: artifacts/jobs/{job_id}/{artifact_type}/{filename}
        dir_path = f"artifacts/jobs/{job_id}/{artifact_type}"
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # Database: metadata + path reference
        artifact = Artifact(
            job_id=job_id,
            artifact_type=artifact_type,
            file_path=file_path,
            file_size=len(content),
            created_at=datetime.utcnow()
        )

        db.add(artifact)
        db.commit()

        return artifact
```

### Pattern 5: Retry with Exponential Backoff

**What:** Retry failed API calls with increasing delays (1s, 2s, 4s, 8s...) to handle transient failures.

**When to use:** External API calls (rate limits, network issues), cloud service integration, any fallible I/O.

**Trade-offs:**
- **Pros:** Resilient to transient failures, handles rate limits gracefully, avoids thundering herd
- **Cons:** Increases latency on failures, can mask systemic issues, requires tuning (max retries, backoff multiplier)

**Example:**
```python
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class VertexAIClient:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def generate_video(self, prompt: str, **kwargs) -> bytes:
        """Generate video with automatic retry on transient failures"""
        try:
            response = await self.client.generate(prompt, **kwargs)
            return response.video_bytes
        except Exception as e:
            # Log retry attempt
            logger.warning(f"Video generation failed, retrying: {e}")
            raise
```

## Data Flow

### Request Flow

```
[User: "typer start --prompt 'A cat exploring Mars'"]
    ↓
[CLI] Create job record (state=PENDING)
    ↓
[Orchestrator] Load job, transition to STORYBOARD
    ↓
[StoryboardGenerator]
    → Call Gemini LLM (async)
    → Poll until complete
    → Parse structured output (JSON: scenes, transitions, characters)
    → Save artifacts/jobs/{id}/storyboard.json
    → Create Artifact record (type="storyboard")
    ↓
[Orchestrator] Transition job to KEYFRAMES
    ↓
[KeyframeGenerator]
    For each scene (sequential):
        → Build continuity prompt (reference previous keyframe)
        → Call Nano Banana Pro (async)
        → Poll until complete
        → Save artifacts/jobs/{id}/keyframes/scene_{n}.png
        → Create Artifact record (type="keyframe")
    ↓
[Orchestrator] Transition job to VIDEO_GEN
    ↓
[VideoGenerator]
    For each scene pair (potentially parallel):
        → Submit Veo 3.1 request (first_frame, last_frame)
        → Poll until complete
        → Save artifacts/jobs/{id}/videos/clip_{n}.mp4
        → Create Artifact record (type="video_clip")
    ↓
[Orchestrator] Transition job to STITCH
    ↓
[StitchEngine]
    → Load all video clips (ordered)
    → Concatenate with FFmpeg (add transitions)
    → Save artifacts/jobs/{id}/output/final.mp4
    → Create Artifact record (type="final")
    ↓
[Orchestrator] Transition job to COMPLETE
    ↓
[CLI] Display success, output path
```

### Crash Recovery Flow

```
[System crashes during VIDEO_GEN]
    ↓
[User: "typer resume --job-id 123"]
    ↓
[CLI] Load job from DB (state=VIDEO_GEN)
    ↓
[Orchestrator] Detect state=VIDEO_GEN
    ↓
[Orchestrator] Load existing artifacts:
    - Storyboard: EXISTS (artifacts/jobs/123/storyboard.json)
    - Keyframes: EXISTS (all scenes present)
    - Video clips: PARTIAL (some scenes complete, some missing)
    ↓
[VideoGenerator] Resume from missing clips:
    - Skip completed clips (artifacts exist)
    - Generate only missing clips
    ↓
[Continue normal flow from VIDEO_GEN → STITCH → COMPLETE]
```

### State Management

```
[Job State Machine]
    ↓
PENDING (initial) → STORYBOARD (LLM generating)
    ↓
STORYBOARD → KEYFRAMES (images generating sequentially)
    ↓
KEYFRAMES → VIDEO_GEN (videos generating, potentially parallel)
    ↓
VIDEO_GEN → STITCH (concatenating clips)
    ↓
STITCH → COMPLETE (job finished)

[Error paths from any state] → FAILED (with error_message)

[Resume logic]
For each state:
    - Check artifacts table for completed work
    - Skip completed stages
    - Resume from first incomplete stage
```

### Key Data Flows

1. **Prompt → Storyboard:** User text prompt → Gemini LLM → Structured JSON (scenes, transitions, characters, settings)

2. **Storyboard → Keyframes:** For each scene → Nano Banana Pro (with continuity context) → PNG image → Sequential to maintain continuity

3. **Keyframes → Videos:** For each scene pair (start, end keyframes) → Veo 3.1 (frame-guided generation) → MP4 clip → Potentially parallel

4. **Videos → Stitched Output:** All clips → FFmpeg concatenation (with transitions) → Final MP4

5. **State Persistence:** After each stage → Update job.state in DB → Commit transaction → Enable crash recovery

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **1-10 jobs** | SQLite + local filesystem sufficient. Single process handles orchestration. CLI-only interface. |
| **10-100 jobs** | Add FastAPI server for API access. Use asyncio.Queue for job queue (in-memory). Add worker pool (3-5 workers) for parallel video generation. Consider Redis for distributed job queue if multi-machine. |
| **100-1000 jobs** | Migrate to PostgreSQL (better concurrency). Use Celery or Taskiq for distributed task queue. Move artifacts to S3/GCS (filesystem doesn't scale). Add job prioritization (paid users first). Implement rate limiting for Vertex AI calls. |
| **1000+ jobs** | Kubernetes for orchestration (horizontal scaling). Separate services: API gateway, job scheduler, generation workers. Use Cloud SQL + Cloud Storage. Implement job sharding (different workers for different stages). Add monitoring (Prometheus, Grafana). Consider dedicated video processing GPUs. |

### Scaling Priorities

1. **First bottleneck: Sequential keyframe generation.** For 10-scene video, keyframes take 10 × 20s = 200s. **Fix:** Batch generation with continuity constraints (generate 2-3 at a time with shared context). **Alternative:** Pre-generate character reference sheet, use as context for all scenes (enables full parallelization).

2. **Second bottleneck: Vertex AI rate limits.** Default quota: 10 requests/min for Veo. **Fix:** Implement request queue with rate limiting (10/min ceiling). **Alternative:** Request quota increase from Google. **Monitoring:** Track API quota usage, alert before hitting limits.

3. **Third bottleneck: SQLite write concurrency.** SQLite supports 1 writer at a time. With multiple workers updating job state, contention occurs. **Fix:** Migrate to PostgreSQL (supports concurrent writes). **Short-term:** Minimize write frequency (batch state updates).

4. **Fourth bottleneck: Local filesystem for artifacts.** 1000 jobs × 10 scenes × 50MB/video = 500GB+. **Fix:** Migrate to S3/GCS. **Benefit:** Also enables distributed workers (all access same storage). **Consideration:** Update Artifact model with S3 URIs instead of filesystem paths.

## Anti-Patterns

### Anti-Pattern 1: Storing Binary Assets in Database

**What people do:** Store images/videos as BLOBs in SQLite/PostgreSQL for "simplicity."

**Why it's wrong:** DB bloats rapidly (50MB video × 100 jobs = 5GB), queries slow down (DB loads BLOBs into memory), backups become massive, migrations take forever, can't use cloud storage (S3/GCS) later.

**Do this instead:** Store artifacts on filesystem (or S3), store paths + metadata in DB. Artifacts table references files, not embeds them. Enables fast queries, small backups, easy cloud migration.

### Anti-Pattern 2: Synchronous API Calls

**What people do:** Call Vertex AI APIs synchronously, block until response (5 minutes).

**Why it's wrong:** Video generation takes 30s-5min, HTTP timeouts occur, CLI appears frozen, can't monitor multiple jobs, crashes if connection drops.

**Do this instead:** Use async APIs with polling. Submit request, get job ID, poll status every 10-30s. Show progress bar. Resilient to timeouts. Can monitor multiple jobs concurrently.

### Anti-Pattern 3: No State Persistence

**What people do:** Run entire pipeline in single script, no checkpointing. If it crashes, restart from beginning.

**Why it's wrong:** For 10-scene video (storyboard 30s + keyframes 200s + videos 300s + stitch 20s = 550s), a crash at stitch loses 530s of work. Expensive API calls wasted. User frustration.

**Do this instead:** Implement state machine with DB persistence. After each stage, update job state + save artifacts. Resume from last successful state. Only redo failed stage, not entire pipeline.

### Anti-Pattern 4: Parallel Keyframe Generation (Naive)

**What people do:** Generate all keyframes in parallel for speed.

**Why it's wrong:** No continuity context between scenes. Character appearance changes (different clothes, age, style). Settings inconsistent (different lighting, weather). Narrative incoherence (props appear/disappear).

**Do this instead:** Generate keyframes sequentially, pass previous keyframe as context. Slower (200s vs 20s) but consistent. **Alternative:** Generate character reference sheet first, use as context for all scenes (enables parallelization with continuity).

### Anti-Pattern 5: No Retry Logic

**What people do:** Call external APIs without retries. If it fails, job fails.

**Why it's wrong:** Vertex AI has transient failures (rate limits, network issues, service restarts). 5% failure rate × 10 API calls = 40% job failure rate. User sees cryptic errors.

**Do this instead:** Implement exponential backoff with retries. Wrap API calls with tenacity (max 5 retries, exponential backoff). Log failures for debugging. Only fail job after exhausting retries. Differentiate transient (retry) vs permanent (fail fast) errors.

### Anti-Pattern 6: Hardcoded Artifact Paths

**What people do:** Hardcode artifact paths like `f"/tmp/video_{scene_id}.mp4"` in generation code.

**Why it's wrong:** Path collisions with parallel jobs (`/tmp/video_1.mp4` overwritten), cleanup impossible (where are all artifacts?), can't migrate to cloud storage (S3 paths different), testing requires filesystem setup.

**Do this instead:** Centralize artifact path generation in ArtifactStore. Use job-scoped directories (`artifacts/jobs/{job_id}/`). Store paths in DB for tracking. Supports parallel jobs, easy cleanup (delete job directory), cloud migration (swap filesystem backend).

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **Gemini 3 Pro (LLM)** | Vertex AI SDK → async request → poll for completion | For storyboard generation. Structured output (JSON) with scenes, transitions, characters. Typical latency: 10-30s for complex prompts. |
| **Nano Banana Pro (Image)** | Vertex AI SDK → async request → poll for completion | For keyframe generation. Supports prompt + reference image (continuity). Typical latency: 15-30s per image. |
| **Veo 3.1 (Video)** | Vertex AI SDK → async request → poll for completion → download from GCS | For video clip generation. Requires first_frame + last_frame (frame-guided). Outputs to GCS bucket (must download). Typical latency: 30s-5min per clip. |
| **FFmpeg (Video Processing)** | Subprocess call → local binary | For stitching video clips. Concatenation + transitions. Ensure installed (`apt install ffmpeg`). |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **CLI ↔ Orchestrator** | Direct function calls (same process) | CLI invokes orchestrator methods. Orchestrator updates job state, CLI polls for progress. |
| **API ↔ Orchestrator** | Background tasks (FastAPI BackgroundTasks) | API endpoint submits job, returns job_id. Orchestrator runs in background. Client polls status endpoint. |
| **Orchestrator ↔ Generators** | Async function calls | Orchestrator awaits generator methods. Generators return artifacts (bytes or paths). Orchestrator saves to ArtifactStore. |
| **Generators ↔ Services** | Async API clients | Generators call service methods (submit, poll). Services abstract Vertex AI SDK. Generators don't import Vertex AI directly. |
| **All ↔ Database** | SQLAlchemy sessions (context managers) | Use `with get_session() as session:` pattern. Automatic commit/rollback. Thread-safe for CLI (single-threaded) and API (multi-threaded). |
| **All ↔ ArtifactStore** | Shared ArtifactStore instance | Centralized artifact management. Handles filesystem + DB artifact records. Used by all generators and orchestrator. |

## Build Order Recommendations

Based on component dependencies, suggested build order:

### Phase 1: Foundation (Database + State Machine)
1. **SQLAlchemy models** (Job, Artifact) → Define schema first
2. **State machine** (JobState enum, transition validation) → Core logic
3. **Database setup** (migrations, session management) → Persistence layer

**Rationale:** Everything depends on state persistence. Build foundation before generators.

### Phase 2: Artifact Storage
4. **ArtifactStore** (filesystem + DB integration) → Centralized artifact management
5. **Path conventions** (job-scoped directories) → Prevent collisions

**Rationale:** Generators need to store artifacts. Build storage before generation.

### Phase 3: Service Clients
6. **Vertex AI base client** (auth, retry, polling) → Shared infrastructure
7. **Gemini client** (storyboard generation) → First generator dependency
8. **Nano Banana client** (keyframe generation) → Second generator dependency
9. **Veo client** (video generation) → Third generator dependency

**Rationale:** Abstract external APIs early. Enables testing with mocks.

### Phase 4: Generators (Sequential)
10. **StoryboardGenerator** (Gemini integration) → First pipeline stage
11. **KeyframeGenerator** (Nano Banana + continuity logic) → Second stage
12. **VideoGenerator** (Veo + frame-guided generation) → Third stage
13. **StitchEngine** (FFmpeg integration) → Final stage

**Rationale:** Build generators in pipeline order. Test each stage independently before integration.

### Phase 5: Orchestration
14. **Orchestrator** (state machine + generator coordination) → Pipeline coordinator
15. **Resume logic** (detect completed work, skip stages) → Crash recovery

**Rationale:** Orchestrator ties everything together. Build after generators exist.

### Phase 6: Interfaces
16. **CLI commands** (Typer: start, resume, status, cancel) → User interface
17. **Progress display** (Rich progress bars) → UX polish
18. **API routes** (FastAPI: optional) → HTTP interface

**Rationale:** Interfaces are thin wrappers around orchestrator. Build last.

### Dependency Graph

```
Database Models ────┐
                    ├──→ State Machine ──→ Orchestrator ──→ CLI/API
Artifact Store ─────┘                          ↑
                                               │
Service Clients ──→ Generators (Storyboard, Keyframes, Video, Stitch)
```

**Critical path:** Database → State Machine → Service Clients → Generators → Orchestrator → CLI

**Parallelizable:** Service clients (Gemini, Nano Banana, Veo) can be built concurrently. Generators can be built concurrently (with service client mocks).

## Sources

### AI Video Generation Architecture
- [NVIDIA RTX AI Video Generation](https://blogs.nvidia.com/blog/rtx-ai-garage-ces-2026-open-models-video-generation/)
- [How to Build AI Video Generator with APIs (2026)](https://modelslab.com/blog/video-generation/how-to-build-ai-video-generator-api-guide-2026)
- [Text-to-video generators: comprehensive survey](https://link.springer.com/article/10.1186/s40537-025-01314-3)
- [ImproveYourVideos: Architectural Improvements for T2V Pipeline](https://ieeexplore.ieee.org/abstract/document/10815947)

### State Machines & Crash Recovery
- [SQLAlchemy State Management Documentation](https://docs.sqlalchemy.org/en/21/orm/session_state_management.html)
- [python-statemachine PyPI](https://pypi.org/project/python-statemachine/)
- [sqlalchemy-fsm GitHub](https://github.com/presslabs/sqlalchemy-fsm)
- [Disaster Recovery for AI Pipelines](https://www.axrail.ai/post/disaster-recovery-for-ai-pipelines-the-essential-rto-rpo-guide-for-enterprise-resilience)
- [Self-Healing Data Pipeline](https://towardsdatascience.com/building-a-self-healing-data-pipeline-that-fixes-its-own-python-errors/)

### Async Job Queues & Patterns
- [Python asyncio Queue Documentation](https://docs.python.org/3/library/asyncio-queue.html)
- [Developing Asynchronous Task Queue in Python](https://testdriven.io/blog/developing-an-asynchronous-task-queue-in-python/)
- [Taskiq GitHub](https://github.com/taskiq-python/taskiq)
- [Python Async Architecture: Real-World Experience](https://xaviercollantes.dev/articles/python-async)

### Multi-Scene Video & Keyframe Continuity
- [DreamFactory: Multi-Scene Long Video Generation](https://arxiv.org/html/2408.11788)
- [Video diffusion generation: comprehensive review](https://link.springer.com/article/10.1007/s10462-025-11331-6)
- [STAGE: Storyboard-Anchored Generation](https://arxiv.org/html/2512.12372)
- [SceneDecorator: Scene Planning and Consistency](https://arxiv.org/html/2510.22994)

### Google Vertex AI Integration
- [Veo on Vertex AI Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation)
- [Generate videos from first and last frames](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-first-and-last-frames)
- [Introducing Veo 3.1 - Gemini API](https://developers.googleblog.com/introducing-veo-3-1-and-new-creative-capabilities-in-the-gemini-api/)

### FastAPI & Pipeline Architecture
- [Modern FastAPI Architecture Patterns](https://medium.com/algomart/modern-fastapi-architecture-patterns-for-scalable-production-systems-41a87b165a8b)
- [End-to-End ML Deployment with FastAPI](https://liviaerxin.github.io/blog/end-to-end-ml-deployment)
- [FastAPI Best Architecture GitHub](https://github.com/fastapi-practices/fastapi_best_architecture)

### Artifact Management
- [ML Metadata | TensorFlow](https://www.tensorflow.org/tfx/guide/mlmd)
- [MLflow - Storing Artifacts in SQLite](https://medium.com/@moyukh_51433/mlflow-storing-artifacts-in-hdfs-and-in-an-sqlite-db-7be26971b6ab)

### Video Stitching & Transitions
- [End-to-End Online Video Stitching](https://www.mdpi.com/2076-3417/15/11/5987)
- [3D Scene Prompting for Camera-Controllable Video](https://cvlab-kaist.github.io/3DScenePrompt/)

---
*Architecture research for: AI Video Generation Pipeline*
*Researched: 2026-02-14*
