# Phase 2: Generation Pipeline - Research

**Researched:** 2026-02-14
**Domain:** AI video generation pipeline (LLM storyboarding, image generation, video synthesis, video stitching)
**Confidence:** MEDIUM-HIGH

## Summary

Phase 2 implements a multi-stage generative AI pipeline that transforms text prompts into multi-scene videos using Google Vertex AI services. The pipeline orchestrates four sequential steps: (1) LLM-based storyboarding with structured JSON output, (2) sequential keyframe image generation with visual continuity, (3) video clip generation using Veo 3.1 with first/last frame control and long-running operation polling, and (4) local video stitching via ffmpeg.

The technical stack centers on the `google-genai` Python SDK (v1.0+) operating in Vertex AI mode for all generative tasks. The SDK provides native async support via `.aio` interface, structured output via JSON schema constraints, and unified handling of Gemini (LLM), Nano Banana Pro (images), and Veo 3.1 (video) models. Critical implementation concerns include exponential backoff for rate limiting, idempotent operation polling for crash recovery, SQLite WAL mode configuration for async writes, and ffmpeg validation at startup.

The architecture follows an async state machine pattern where each pipeline step persists state to SQLite before advancing, enabling resume-from-failure capability. Visual continuity is achieved through sequential keyframe generation where scene N's end frame becomes scene N+1's start frame via image-conditioned generation.

**Primary recommendation:** Use Tenacity library for all retry logic (API calls and polling), implement async context managers for database sessions, validate ffmpeg availability during application startup, and structure the pipeline as an async state machine with explicit state transitions persisted to the database.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `google-genai` | >=1.0.0 | Unified Google AI SDK for Vertex AI | Official Google SDK (GA as of Feb 2026), supports async, structured output, all three model types (Gemini, Imagen, Veo) |
| `tenacity` | >=8.5.0 | Async retry with exponential backoff | Industry standard for retry logic, decorator-based, async-native, supports jitter and exception filtering |
| `sqlalchemy[asyncio]` | >=2.0 | Async ORM and database management | Official async support in 2.x, context manager patterns, session lifecycle control |
| `aiosqlite` | >=0.22.1 | Async SQLite driver | Required for SQLAlchemy async with SQLite, WAL mode support |
| `ffmpeg` (system) | 6.x/7.x | Video concatenation and transitions | Industry standard for video processing, concat demuxer is lossless and instant |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `httpx` | >=0.27.0 | HTTP client for API calls | Already included; google-genai uses it internally for both sync and async |
| `Pillow` | >=10.0 | Image I/O and format conversion | Converting between bytes/PIL/base64 for API consumption |
| `pydantic` | >=2.0 | JSON schema definition, response validation | Defining structured output schemas for Gemini storyboard generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `google-genai` | Deprecated `vertexai` module | Old vertexai module deprecated; google-genai is the GA replacement with better async support |
| `tenacity` | Manual retry loops | Tenacity provides battle-tested exponential backoff, jitter, exception filtering — don't hand-roll |
| `ffmpeg` | MoviePy | MoviePy wraps ffmpeg but adds re-encoding overhead; ffmpeg concat demuxer is lossless for same-codec clips |

**Installation:**
```bash
pip install google-genai>=1.0.0 tenacity>=8.5.0 sqlalchemy[asyncio]>=2.0 aiosqlite>=0.22.1 httpx>=0.27.0 Pillow>=10.0 pydantic>=2.0
```

**System Dependencies:**
```bash
# ffmpeg must be on PATH
# Validate at startup with: subprocess.run(['ffmpeg', '-version'], check=True)
```

## Architecture Patterns

### Recommended Project Structure
```
vidpipe/
├── pipeline/
│   ├── orchestrator.py      # Main async state machine coordinating all steps
│   ├── storyboard.py        # Step 1: Gemini structured output for scene planning
│   ├── keyframes.py         # Step 2: Sequential image generation with continuity
│   ├── video_gen.py         # Step 3: Veo submission + polling with backoff
│   └── stitcher.py          # Step 4: ffmpeg concat/crossfade
├── services/
│   ├── vertex_client.py     # google-genai Client wrapper with retry decorators
│   └── file_manager.py      # [EXISTING] Local file I/O
├── schemas/
│   ├── storyboard.py        # Pydantic models for structured output
│   └── enums.py             # Status enums for state machine
└── db/
    ├── models.py            # [EXISTING] ORM models
    └── engine.py            # [EXISTING] Async session factory
```

### Pattern 1: Async State Machine with Persistent State

**What:** Pipeline orchestrator as async function that checks DB state before each step, executes step, updates DB state, then advances.

**When to use:** Any multi-step pipeline that must survive crashes and resume from last completed step.

**Example:**
```python
# Source: SQLAlchemy 2.0 async patterns + state machine best practices
from sqlalchemy.ext.asyncio import AsyncSession
from vidpipe.db.engine import async_session

async def run_pipeline(project_id: UUID):
    """
    Orchestrate pipeline with state persistence and resume capability.
    Each step checks current state, executes if needed, updates state.
    """
    async with async_session() as session:
        project = await session.get(Project, project_id)

        # Step 1: Storyboard
        if project.status == "pending":
            await generate_storyboard(session, project)
            project.status = "keyframing"
            await session.commit()

        # Step 2: Keyframes
        if project.status == "keyframing":
            await generate_keyframes(session, project)
            project.status = "generating_video"
            await session.commit()

        # Step 3: Video generation
        if project.status == "generating_video":
            await generate_videos(session, project)
            project.status = "stitching"
            await session.commit()

        # Step 4: Stitch
        if project.status == "stitching":
            await stitch_videos(session, project)
            project.status = "complete"
            await session.commit()
```

### Pattern 2: Tenacity Retry Decorators for API Calls

**What:** Declarative retry logic using decorators with exponential backoff, jitter, and exception filtering.

**When to use:** All external API calls (Google GenAI SDK methods) and any polling loops.

**Example:**
```python
# Source: https://github.com/jd/tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    wait_random
)
from google.genai import errors

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 2),  # Jitter prevents thundering herd
    retry=retry_if_exception_type(errors.APIError),
    reraise=True
)
async def generate_keyframe_with_retry(client, prompt, config):
    """
    Generate keyframe with automatic retry on transient failures.
    Retries only on APIError (rate limits, transient 5xx).
    Max 5 attempts, exponential backoff from 2s to 60s with jitter.
    """
    response = await client.aio.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt],
        config=config
    )
    return response
```

### Pattern 3: Sequential Keyframe Loop with Frame Inheritance

**What:** For loop that generates keyframes in scene order, reusing previous scene's end frame as next scene's start frame.

**When to use:** Maintaining visual continuity across scene transitions.

**Example:**
```python
# Source: Spec section 5.2 (Step 2: Keyframe Generation)
async def generate_keyframes(session: AsyncSession, project: Project):
    """
    Sequential keyframe generation with visual continuity.
    Scene 0: Generate start and end frames from prompts.
    Scene N: Inherit start frame from scene N-1 end, generate new end frame.
    """
    scenes = await session.execute(
        select(Scene).where(Scene.project_id == project.id).order_by(Scene.scene_index)
    )
    scenes = scenes.scalars().all()

    previous_end_frame_bytes = None

    for scene in scenes:
        # Start frame: first scene generates, others inherit
        if scene.scene_index == 0:
            start_frame_bytes = await generate_image(scene.start_frame_prompt)
            start_source = "generated"
        else:
            start_frame_bytes = previous_end_frame_bytes
            start_source = "inherited"

        # Save start keyframe
        file_path = file_manager.save_keyframe(project.id, scene.scene_index, "start", start_frame_bytes)
        session.add(Keyframe(
            scene_id=scene.id,
            position="start",
            file_path=str(file_path),
            source=start_source,
            prompt_used=scene.start_frame_prompt
        ))

        # End frame: image-conditioned generation using start frame
        conditioning_prompt = f"Using this image as reference, show the same scene 4 seconds later. {scene.end_frame_prompt}. Maintain visual style, lighting, composition."
        end_frame_bytes = await generate_image_conditioned(start_frame_bytes, conditioning_prompt)

        # Save end keyframe
        file_path = file_manager.save_keyframe(project.id, scene.scene_index, "end", end_frame_bytes)
        session.add(Keyframe(
            scene_id=scene.id,
            position="end",
            file_path=str(file_path),
            source="generated",
            prompt_used=scene.end_frame_prompt
        ))

        previous_end_frame_bytes = end_frame_bytes
        scene.status = "keyframes_done"
        await session.commit()
```

### Pattern 4: Long-Running Operation Polling with Backoff

**What:** Submit async operation, persist operation ID, poll with configurable interval until completion or timeout.

**When to use:** Veo video generation (and any long-running GCP operations).

**Example:**
```python
# Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-first-and-last-frames
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

async def generate_video_clip(session: AsyncSession, scene: Scene, start_frame_bytes: bytes, end_frame_bytes: bytes):
    """
    Submit Veo job, persist operation ID, poll until completion.
    Idempotent: if operation_name exists, resume polling.
    """
    # Check if already submitted
    clip = await session.execute(
        select(VideoClip).where(VideoClip.scene_id == scene.id)
    )
    clip = clip.scalar_one_or_none()

    if not clip:
        # Submit new job
        operation = await client.aio.models.generate_videos(
            model="veo-3.1-generate-001",
            prompt=scene.video_motion_prompt,
            image=types.Image(image_bytes=start_frame_bytes, mime_type="image/png"),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=4,
                last_frame=types.Image(image_bytes=end_frame_bytes, mime_type="image/png")
            )
        )

        # Persist operation BEFORE polling (idempotent resume)
        clip = VideoClip(
            scene_id=scene.id,
            operation_name=operation.name,
            status="polling",
            poll_count=0
        )
        session.add(clip)
        await session.commit()

    # Poll operation
    operation_name = clip.operation_name
    max_polls = 40  # ~10 minutes at 15s intervals
    poll_interval = 15

    for poll_count in range(clip.poll_count, max_polls):
        operation = await client.aio.operations.get(operation_name)
        clip.poll_count = poll_count + 1

        if operation.done:
            if operation.response:
                video = operation.response.generated_videos[0]
                # Download and save video
                clip.status = "complete"
                clip.local_path = str(file_manager.save_clip(project.id, scene.scene_index, video_bytes))
                break
            elif operation.response.raiMediaFilteredCount > 0:
                clip.status = "rai_filtered"
                clip.error_message = "Content filtered by responsible AI"
                break

        await session.commit()
        await asyncio.sleep(poll_interval)
    else:
        clip.status = "timed_out"
        clip.error_message = f"Operation did not complete after {max_polls} polls"

    await session.commit()
    return clip
```

### Pattern 5: Structured Output with Pydantic Schemas

**What:** Define Pydantic models for expected LLM output, pass as `response_schema` with `response_mime_type="application/json"`.

**When to use:** Storyboard generation (Gemini 3 Pro structured output).

**Example:**
```python
# Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output
from pydantic import BaseModel, Field
from google.genai import types

class StyleGuide(BaseModel):
    visual_style: str = Field(description="Overall visual style (e.g., cinematic, anime, documentary)")
    color_palette: str = Field(description="Dominant color palette description")
    camera_style: str = Field(description="Camera movement style (e.g., static, handheld, drone)")

class SceneSchema(BaseModel):
    scene_index: int
    scene_description: str = Field(description="What happens in this scene")
    start_frame_prompt: str = Field(description="Detailed image prompt for opening keyframe")
    end_frame_prompt: str = Field(description="Detailed image prompt for closing keyframe")
    video_motion_prompt: str = Field(description="Motion/action description for video generation")
    transition_notes: str = Field(description="How this scene connects to the next")

class StoryboardOutput(BaseModel):
    style_guide: StyleGuide
    scenes: list[SceneSchema]

async def generate_storyboard(prompt: str) -> StoryboardOutput:
    """
    Generate storyboard with structured JSON output using Pydantic schema.
    Gemini 3 Pro natively supports response_schema constraint.
    """
    system_prompt = """You are a cinematic storyboard director. Break the script into visual scenes.
    For each scene provide detailed prompts ensuring visual continuity between scenes."""

    response = await client.aio.models.generate_content(
        model="gemini-3-pro",
        contents=[f"{system_prompt}\n\nScript: {prompt}"],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StoryboardOutput  # Pydantic model
        )
    )

    # Parse response as Pydantic model
    storyboard = StoryboardOutput.model_validate_json(response.text)
    return storyboard
```

### Anti-Patterns to Avoid

- **Blocking sleep in async functions:** Use `await asyncio.sleep()`, never `time.sleep()` — blocks event loop
- **Sharing AsyncSession across concurrent tasks:** SQLAlchemy AsyncSession is not thread-safe; use context managers per task
- **Retrying on all exceptions:** Only retry transient errors (429 rate limit, 5xx server errors); never retry 4xx validation errors
- **Hard-coding paths without path traversal protection:** Always use `Path.resolve()` and `is_relative_to()` to prevent directory escape
- **Submitting long operations without persisting operation ID first:** If process crashes before polling, operation is lost — persist immediately after submission

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Retry with exponential backoff | Custom retry loops with manual delay calculation | `tenacity` library | Handles jitter (prevents thundering herd), exception filtering, async support, max attempts, logging — battle-tested for edge cases |
| Video concatenation | Custom video processing with libraries like OpenCV | `ffmpeg` concat demuxer | ffmpeg concat with `-c copy` is lossless and instant for same-codec clips; hand-rolled solutions require re-encoding |
| JSON schema validation | Manual dict validation with `if/else` | Pydantic models with `model_validate_json()` | Pydantic provides validation, type coercion, error messages, serialization — far more robust than manual checks |
| Long-running operation polling | Custom while loops with sleep | Async for loop with Tenacity retry decorator | Proper async sleep, timeout handling, poll count persistence, idempotent resume logic requires careful implementation |
| Async context managers for DB | Manual session creation/cleanup | SQLAlchemy `async with async_session()` pattern | Ensures session cleanup on exception, prevents connection leaks, handles async properly |

**Key insight:** This domain has complex edge cases (rate limits, RAI filtering, concurrent writes, crash recovery). Use battle-tested libraries that have already solved these problems rather than discovering them in production.

## Common Pitfalls

### Pitfall 1: Missing ffmpeg at Runtime

**What goes wrong:** Pipeline fails during stitching step with "FileNotFoundError: ffmpeg not found" after spending minutes/dollars generating all clips.

**Why it happens:** ffmpeg is a system dependency, not a Python package. It's not automatically installed with pip and may not be on PATH.

**How to avoid:**
- Validate ffmpeg availability during application startup (not during pipeline execution)
- Check both `ffmpeg` and get version to ensure it's functional
- Fail fast with clear error message directing user to install ffmpeg

**Warning signs:**
```python
# GOOD: Validate at startup
async def startup():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info(f"ffmpeg validated: {result.stdout.decode()[:50]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg not found on PATH. Install: https://ffmpeg.org/download.html")
```

**Source:** [Gumlet - How to Use FFmpeg with Python](https://www.gumlet.com/learn/ffmpeg-python/), [ffmpeg-python error solutions](https://github.com/Mordekai66/FFmpeg-Python-Error-Solution/)

### Pitfall 2: Rate Limiting Without Exponential Backoff

**What goes wrong:** Pipeline hits Vertex AI rate limits (429 errors), retries immediately, gets throttled again, burns through retry attempts in seconds, then fails permanently.

**Why it happens:** Image generation has tight rate limits (~10 req/min on free tier). Sequential keyframe generation can hit limits quickly. Immediate retries compound the problem.

**How to avoid:**
- Use Tenacity with exponential backoff (min 2s, max 60s)
- Add jitter (`wait_random(0, 2)`) to prevent synchronized retries across multiple processes
- Only retry on 429/5xx errors, never on 4xx validation errors
- Consider fixed delay between successful image generations (3-5s) to stay under rate limits

**Warning signs:** Logs show rapid retry attempts (<1s apart), multiple 429 errors in quick succession

**Source:** [Google Cloud - Handling 429 errors in LLMs](https://cloud.google.com/blog/products/ai-machine-learning/learn-how-to-handle-429-resource-exhausted-errors-in-your-llms), [Vertex AI quotas documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/quotas)

### Pitfall 3: RAI Filtering Without Graceful Degradation

**What goes wrong:** Veo filters a clip due to responsible AI policies, pipeline crashes with KeyError trying to access non-existent video data, entire project marked failed.

**Why it happens:** Veo returns `raiMediaFilteredCount > 0` with no video data when content violates policies. Code assumes `generated_videos[0]` always exists.

**How to avoid:**
- Check `raiMediaFilteredCount` before accessing video data
- Mark clip as `rai_filtered`, log the scene, continue pipeline with remaining scenes
- Store partial results — user can review filtered scenes and retry with modified prompts
- Don't fail entire project over one filtered clip

**Warning signs:**
```python
# BAD: Assumes video always exists
video = operation.response.generated_videos[0]  # IndexError if filtered

# GOOD: Check for filtering
if operation.response.raiMediaFilteredCount > 0:
    clip.status = "rai_filtered"
    clip.error_message = "Content filtered by responsible AI"
    logger.warning(f"Scene {scene.scene_index} filtered by RAI")
else:
    video = operation.response.generated_videos[0]
    # ... process video
```

**Source:** [Veo API documentation - RAI filtering](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-first-and-last-frames)

### Pitfall 4: Invalid JSON from LLM Without Retry Strategy

**What goes wrong:** Gemini returns malformed JSON (extra text before/after JSON, syntax errors), parsing fails, storyboard generation crashes after user waited 30+ seconds.

**Why it happens:** Even with `response_mime_type="application/json"`, LLMs occasionally return invalid JSON, especially with complex schemas or high temperature.

**How to avoid:**
- Wrap JSON parsing in try/except
- Retry with temperature adjustment (reduce by 0.1-0.2 each attempt)
- Max 3 retries for structured output
- Log the raw response for debugging
- Consider using `response_schema` constraint (stricter than just mime type)

**Warning signs:**
```python
# GOOD: Retry with temperature adjustment
@retry(stop=stop_after_attempt(3))
async def generate_storyboard_with_retry(prompt: str, temperature: float = 0.7):
    try:
        response = await client.aio.models.generate_content(
            model="gemini-3-pro",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StoryboardOutput,
                temperature=temperature
            )
        )
        return StoryboardOutput.model_validate_json(response.text)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"Invalid JSON at temp {temperature}: {e}")
        # Tenacity will retry with lower temperature
        return await generate_storyboard_with_retry(prompt, temperature - 0.15)
```

**Source:** Spec section 5.1 (STOR-05 requirement), [Vertex AI structured output docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output)

### Pitfall 5: SQLite Concurrent Write Deadlocks

**What goes wrong:** Multiple async tasks try to write to SQLite simultaneously, get "database is locked" errors, transactions fail, pipeline state becomes inconsistent.

**Why it happens:** SQLite allows only one writer at a time. Even in WAL mode, concurrent writes block. Async code can easily create concurrent write scenarios.

**How to avoid:**
- Use `async with async_session()` context manager for each operation
- Set `busy_timeout` PRAGMA (already configured in engine.py: 5000ms)
- Avoid long-running transactions — commit frequently
- For polling loops: commit after each poll iteration to release lock
- Never share AsyncSession across concurrent tasks

**Warning signs:**
- Logs show "sqlite3.OperationalError: database is locked"
- DetachedInstanceError when accessing ORM objects after commit
- Timeouts during high concurrency (multiple scenes processing)

**Code pattern:**
```python
# GOOD: Short transactions, frequent commits
async with async_session() as session:
    clip = await session.get(VideoClip, clip_id)
    clip.poll_count += 1
    await session.commit()  # Release lock immediately

# BAD: Holding lock during long async operation
async with async_session() as session:
    clip = await session.get(VideoClip, clip_id)
    await asyncio.sleep(15)  # Lock held for 15 seconds!
    clip.poll_count += 1
    await session.commit()
```

**Source:** [SQLite WAL documentation](https://sqlite.org/wal.html), [SQLAlchemy async documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html), [SQLite concurrency best practices](https://www.sqliteforum.com/p/handling-concurrency-in-sqlite-best)

### Pitfall 6: Path Traversal in File Manager

**What goes wrong:** Malicious project_id like `../../etc/passwd` or `../../../tmp/evil` escapes the base directory, reads/writes files outside tmp/, potential security breach.

**Why it happens:** String concatenation of user input into file paths without validation.

**How to avoid:**
- Use `Path.resolve()` to canonicalize paths
- Check with `is_relative_to(base_dir)` before creating files
- FileManager already implements this (see file_manager.py line 58-62)
- Never trust user input in file paths

**Warning signs:** File operations outside expected directory structure, unexpected ValueError from FileManager

**Source:** [Python pathlib path traversal prevention](https://salvatoresecurity.com/preventing-directory-traversal-vulnerabilities-in-python/), [Path traversal remediation in Python](https://osintteam.blog/path-traversal-and-remediation-in-python-0b6e126b4746)

### Pitfall 7: ffmpeg Concat List with Absolute Paths Rejected

**What goes wrong:** ffmpeg concat demuxer rejects file list with "Unsafe file name" error when using absolute paths, stitching fails.

**Why it happens:** ffmpeg's concat demuxer uses `safe=1` by default, which rejects absolute paths as potential security risk.

**How to avoid:**
- Use `-safe 0` flag when invoking ffmpeg concat demuxer
- Quote file paths in list file (single quotes for paths with spaces)
- Format: `file '/absolute/path/to/clip.mp4'` per line

**Warning signs:** ffmpeg error message containing "Unsafe file name" or "Protocol not found"

**Example:**
```python
# Create concat list file
list_path = tmp_dir / "concat_list.txt"
with open(list_path, 'w') as f:
    for clip_path in clip_paths:
        f.write(f"file '{clip_path}'\n")

# Run ffmpeg with -safe 0
subprocess.run([
    'ffmpeg', '-y',
    '-f', 'concat',
    '-safe', '0',  # CRITICAL: Allow absolute paths
    '-i', str(list_path),
    '-c', 'copy',
    str(output_path)
], check=True)
```

**Source:** [FFmpeg concat unsafe file name](https://copyprogramming.com/howto/ffmpeg-concat-unsafe-file-name), [FFmpeg protocols documentation](https://ffmpeg.org/ffmpeg-protocols.html)

### Pitfall 8: Blocking Event Loop with time.sleep() in Async Functions

**What goes wrong:** Using `time.sleep(15)` in polling loop blocks entire async event loop, all other async tasks freeze, application becomes unresponsive.

**Why it happens:** `time.sleep()` is synchronous and blocks the thread. In async context, it prevents event loop from switching to other tasks.

**How to avoid:**
- Always use `await asyncio.sleep()` in async functions
- For CPU-bound operations, use `asyncio.to_thread()` or `loop.run_in_executor()`
- Linters like pylint can catch this (`await-outside-async`)

**Warning signs:** Application freezes during polling, other API endpoints don't respond during video generation, CPU-bound tasks

**Source:** [Python asyncio sleep documentation](https://superfastpython.com/asyncio-sleep/), [FastAPI async best practices](https://fastapi.tiangolo.com/async/)

## Code Examples

Verified patterns from official sources:

### 1. Google GenAI Client Setup for Vertex AI

```python
# Source: https://googleapis.github.io/python-genai/
from google import genai
from google.genai import types
import os

# Set environment variables for Vertex AI mode
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'true'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'hoyack-1577568661630'
os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'

# Create client (uses ADC for authentication)
client = genai.Client(
    vertexai=True,
    project='hoyack-1577568661630',
    location='us-central1'
)

# Async usage
async with genai.Client(vertexai=True).aio as aclient:
    response = await aclient.models.generate_content(...)
```

### 2. Image Generation with Nano Banana Pro

```python
# Source: Spec section 5.2 + https://ai.google.dev/gemini-api/docs/image-generation
from google import genai
from google.genai import types

async def generate_start_frame(prompt: str, aspect_ratio: str = "16:9") -> bytes:
    """
    Generate first keyframe from text prompt using Nano Banana Pro.
    Returns PNG image bytes.
    """
    response = await client.aio.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],  # Text+Image output
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
        )
    )

    # Extract image from response (filter out "thought" parts)
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data  # bytes

    raise ValueError("No image generated in response")


async def generate_end_frame(start_image_bytes: bytes, end_prompt: str, aspect_ratio: str = "16:9") -> bytes:
    """
    Generate end keyframe using start frame as conditioning image.
    Image-conditioned generation maintains visual continuity.
    """
    conditioning_prompt = (
        f"Using this image as reference, show the same scene 4 seconds later. "
        f"{end_prompt}. Maintain visual style, lighting, and composition."
    )

    response = await client.aio.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            types.Part.from_bytes(data=start_image_bytes, mime_type="image/png"),
            types.Part.from_text(text=conditioning_prompt)
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
        )
    )

    # Extract image
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise ValueError("No image generated in response")
```

### 3. Video Generation with Veo 3.1

```python
# Source: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-first-and-last-frames
from google import genai
from google.genai import types
import asyncio

async def submit_veo_job(
    motion_prompt: str,
    start_frame_bytes: bytes,
    end_frame_bytes: bytes,
    aspect_ratio: str = "16:9",
    duration: int = 4
) -> str:
    """
    Submit Veo 3.1 video generation job with first/last frame control.
    Returns operation name for polling.
    """
    operation = await client.aio.models.generate_videos(
        model="veo-3.1-generate-001",
        prompt=motion_prompt,
        image=types.Image(
            image_bytes=start_frame_bytes,
            mime_type="image/png"
        ),
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration,
            last_frame=types.Image(
                image_bytes=end_frame_bytes,
                mime_type="image/png"
            )
            # output_gcs_uri can be omitted to get bytes directly in response
        )
    )

    return operation.name


async def poll_veo_operation(operation_name: str, max_polls: int = 40, interval: int = 15) -> dict:
    """
    Poll long-running Veo operation until completion or timeout.
    Returns video data or error information.
    """
    for poll_count in range(max_polls):
        operation = await client.aio.operations.get(operation_name)

        if operation.done:
            if operation.response:
                # Check for RAI filtering
                if operation.response.raiMediaFilteredCount > 0:
                    return {
                        "status": "rai_filtered",
                        "error": "Content filtered by responsible AI"
                    }

                # Success: extract video
                video = operation.response.generated_videos[0]
                return {
                    "status": "complete",
                    "video_bytes": video.video_bytes,  # or download from gcs_uri
                    "mime_type": video.mime_type
                }
            else:
                return {
                    "status": "failed",
                    "error": operation.error
                }

        await asyncio.sleep(interval)

    # Timeout
    return {
        "status": "timed_out",
        "error": f"Operation did not complete after {max_polls * interval} seconds"
    }
```

### 4. ffmpeg Video Concatenation

```python
# Source: Spec section 5.4 + https://shotstack.io/learn/use-ffmpeg-to-concatenate-video/
import subprocess
from pathlib import Path

def stitch_clips_concat_demuxer(clip_paths: list[Path], output_path: Path):
    """
    Stitch video clips using ffmpeg concat demuxer (lossless, fast).
    All clips must have same codec, resolution, frame rate.
    """
    # Create concat list file
    list_file = output_path.parent / "concat_list.txt"
    with open(list_file, 'w') as f:
        for clip_path in clip_paths:
            # Use single quotes and absolute paths with -safe 0
            f.write(f"file '{clip_path.resolve()}'\n")

    # Run ffmpeg concat
    subprocess.run([
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',  # Allow absolute paths
        '-i', str(list_file),
        '-c', 'copy',  # Stream copy (no re-encoding)
        str(output_path)
    ], check=True, capture_output=True)

    list_file.unlink()  # Clean up


def stitch_clips_with_crossfade(
    clip_paths: list[Path],
    output_path: Path,
    crossfade_duration: float = 0.5,
    clip_duration: int = 4
):
    """
    Stitch clips with crossfade transitions using xfade filter.
    Requires re-encoding (slower than concat demuxer).
    """
    # Build input arguments
    inputs = []
    for clip_path in clip_paths:
        inputs.extend(['-i', str(clip_path)])

    # Build xfade filter chain
    # [0:v][1:v]xfade=transition=fade:duration=0.5:offset=3.5[v01];[v01][2:v]xfade=...
    filter_parts = []
    prev_label = "0:v"

    for i in range(1, len(clip_paths)):
        out_label = f"v{i:02d}"
        offset = (i * clip_duration) - (crossfade_duration * i)
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:"
            f"duration={crossfade_duration}:offset={offset}[{out_label}]"
        )
        prev_label = out_label

    filter_complex = ";".join(filter_parts)

    subprocess.run([
        'ffmpeg', '-y',
        *inputs,
        '-filter_complex', filter_complex,
        '-map', f"[{prev_label}]",
        '-vsync', 'vfr',
        str(output_path)
    ], check=True, capture_output=True)
```

### 5. Async Session Context Manager Pattern

```python
# Source: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
from sqlalchemy.ext.asyncio import AsyncSession
from vidpipe.db.engine import async_session
from contextlib import asynccontextmanager

# Pattern 1: Direct context manager (recommended for most cases)
async def update_project_status(project_id: UUID, status: str):
    async with async_session() as session:
        project = await session.get(Project, project_id)
        project.status = status
        await session.commit()
    # Session automatically closed on exit


# Pattern 2: Dependency injection for FastAPI
from fastapi import Depends

async def get_db_session():
    """FastAPI dependency for database sessions."""
    async with async_session() as session:
        yield session

@app.post("/generate")
async def generate_video(request: GenerateRequest, session: AsyncSession = Depends(get_db_session)):
    project = Project(prompt=request.prompt, status="pending")
    session.add(project)
    await session.commit()
    return {"project_id": project.id}


# Pattern 3: Manual session management (only when context manager doesn't fit)
async def complex_operation():
    session = async_session()
    try:
        # Multiple operations
        project = await session.get(Project, project_id)
        project.status = "processing"
        await session.commit()

        # More operations...

        await session.commit()
    except Exception as e:
        await session.rollback()
        raise
    finally:
        await session.close()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `vertexai` Python module | `google-genai` unified SDK | Jan 2025 | google-genai is now GA; old vertexai module deprecated. Simpler API, better async support, consistent interface across Gemini/Imagen/Veo |
| Manual retry loops | Tenacity library decorators | Standard since 2020 | Declarative retry logic with jitter, exponential backoff, exception filtering — reduces boilerplate |
| MoviePy for video stitching | ffmpeg concat demuxer | Always preferred | ffmpeg concat with stream copy is lossless and instant; MoviePy wraps ffmpeg but forces re-encoding |
| Sync-only Google AI SDKs | Native async support via `.aio` | google-genai 1.0 (Feb 2026) | Async methods for all API calls, proper asyncio integration, no thread pool wrappers |
| String-based status tracking | Enum-based state machines | Python 3.10+ | Type safety, IDE completion, transition validation, python-statemachine library support |

**Deprecated/outdated:**
- **`google.cloud.aiplatform.vertexai` module**: Replaced by `google-genai` SDK (GA Feb 2026)
- **String response schemas**: Use Pydantic models with `response_schema` instead of raw dict definitions
- **Manual base64 encoding for images**: google-genai SDK accepts bytes directly via `types.Part.from_bytes()`

## Open Questions

1. **Image-conditioned generation API details**
   - What we know: Nano Banana Pro supports image + text input for conditioning
   - What's unclear: Exact parameter names in google-genai SDK (does it use `contents=[image, text]` or special conditioning parameter?)
   - Recommendation: Test with spec's approach (image as first Part, prompt as second Part) and verify in integration tests; fallback to official docs if API differs

2. **Veo operation polling: when to use GCS vs. bytes**
   - What we know: Veo can return video bytes directly or write to GCS URI
   - What's unclear: Size limits for bytes response, whether bytes response is GA or preview feature
   - Recommendation: Start with bytes response (simpler, no GCS setup). If clips exceed size limits, add GCS bucket configuration as fallback

3. **SQLite write concurrency under async load**
   - What we know: WAL mode supports concurrent reads, but only one writer. busy_timeout helps but doesn't eliminate all conflicts
   - What's unclear: Real-world performance with 10-15 concurrent scene processing (if we parallelize video generation)
   - Recommendation: Start with sequential processing (video_gen_concurrency=1). If performance demands parallelism, test with 2-3 concurrent Veo jobs and monitor for lock timeouts

4. **Rate limits for Nano Banana Pro image generation**
   - What we know: General rate limits exist (~10 req/min mentioned for free tier)
   - What's unclear: Exact limits for paid tier, whether project has quota increases
   - Recommendation: Implement configurable `image_gen_delay` (default: 3s between calls). Monitor 429 responses and adjust delay as needed

## Sources

### Primary (HIGH confidence)
- [Google GenAI SDK Documentation](https://googleapis.github.io/python-genai/) - Client setup, API methods, async patterns
- [Vertex AI Structured Output Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output) - JSON schema, response_mime_type
- [Vertex AI Veo First/Last Frame Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-first-and-last-frames) - Video API, polling
- [SQLAlchemy 2.0 Asyncio Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) - Async session patterns
- [Tenacity GitHub Repository](https://github.com/jd/tenacity) - Retry patterns, decorators
- [SQLite WAL Mode Documentation](https://sqlite.org/wal.html) - Concurrency model
- [FFmpeg Protocols Documentation](https://ffmpeg.org/ffmpeg-protocols.html) - Concat demuxer, safe mode

### Secondary (MEDIUM confidence)
- [Nano Banana Pro Overview (Google AI)](https://ai.google.dev/gemini-api/docs/image-generation) - Image generation capabilities
- [Veo 3.1 Documentation (Google AI)](https://ai.google.dev/gemini-api/docs/video) - Video generation API
- [Google Cloud Blog - Handling 429 Errors](https://cloud.google.com/blog/products/ai-machine-learning/learn-how-to-handle-429-resource-exhausted-errors-in-your-llms) - Rate limiting strategies
- [Shotstack - FFmpeg Concat Guide](https://shotstack.io/learn/use-ffmpeg-to-concatenate-video/) - Concat demuxer usage
- [WaveSpeed AI - FFmpeg Merge Guide](https://wavespeed.ai/blog/posts/blog-how-to-merge-concatenate-videos-ffmpeg/) - Concat and xfade patterns
- [OTTVerse - FFmpeg xfade Filter](https://ottverse.com/crossfade-between-videos-ffmpeg-xfade-filter/) - Crossfade transitions
- [Mike Salvatore - Path Traversal Prevention](https://salvatoresecurity.com/preventing-directory-traversal-vulnerabilities-in-python/) - Security patterns
- [Retry Mechanisms in Python (Medium)](https://medium.com/@oggy/retry-mechanisms-in-python-practical-guide-with-real-life-examples-ed323e7a8871) - Retry patterns
- [Python Asyncio Sleep (Super Fast Python)](https://superfastpython.com/asyncio-sleep/) - Async sleep patterns

### Tertiary (LOW confidence - needs validation)
- [Python State Machine Libraries Comparison](https://github.com/pytransitions/transitions) - State machine patterns
- [Pillow Image/Base64 Conversion](https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/) - Image I/O patterns
- [SQLite Concurrency Challenges](https://www.slingacademy.com/article/concurrency-challenges-in-sqlite-and-how-to-overcome-them/) - WAL mode best practices

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - google-genai is GA, Tenacity is industry standard, SQLAlchemy 2.0 async is mature
- Architecture: MEDIUM-HIGH - Patterns verified in official docs, but exact SDK methods for image conditioning need testing
- Pitfalls: HIGH - Based on official docs, real GitHub issues, and community experience reports
- Code examples: MEDIUM-HIGH - Structured from official docs but combined for this use case; needs integration testing

**Research date:** 2026-02-14
**Valid until:** Mid-March 2026 (30 days for stable APIs; google-genai SDK is GA but may receive updates)

**Key validation items for integration testing:**
1. Image-conditioned generation exact API syntax
2. Veo operation response format (bytes vs. GCS)
3. Rate limit thresholds for project quota
4. SQLite lock contention under concurrent load
5. RAI filtering response structure
