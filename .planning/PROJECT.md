# Viral Video Generation Pipeline (vidpipe)

## What This Is

A local Python CLI pipeline that transforms a text prompt into a stitched, multi-scene AI-generated video. It uses Google Vertex AI / Gemini APIs for all generative work — Gemini 3 Pro for storyboarding, Nano Banana Pro for keyframe image generation, and Veo 3.1 for 4-second video clip generation with first/last frame control. Completed clips are stitched locally via ffmpeg. The project also exposes a lightweight FastAPI server for HTTP-triggered automation.

## Core Value

Accept a text prompt and produce a cohesive, multi-scene short video (15–60 seconds) with visual continuity between scenes — fully automated, crash-safe, and resumable.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Storyboard generation from text prompt using Gemini 3 Pro structured output
- [ ] Sequential keyframe image generation with visual continuity (scene N end = scene N+1 start)
- [ ] Video clip generation per scene using Veo 3.1 with first/last frame control
- [ ] Polling of long-running Veo operations with backoff and timeout
- [ ] ffmpeg stitching of clips into a single MP4 (hard cuts + optional crossfade)
- [ ] SQLAlchemy + SQLite persistence of all pipeline state and asset references
- [ ] Local filesystem storage of binary artifacts in tmp/{project_id}/
- [ ] CLI interface (generate, resume, status, list, stitch)
- [ ] FastAPI server with background task execution and status polling
- [ ] Retry/resume — crash-safe pipeline that picks up from last completed step
- [ ] Configurable pipeline parameters (style, aspect ratio, clip duration, etc.)
- [ ] Rate limiting with exponential backoff for API calls
- [ ] RAI filter handling for Veo content safety rejections

### Out of Scope

- Cloud storage upload (GCS, S3) — future phase
- Audio narration / TTS overlay — future phase
- Real-time streaming / WebSocket progress — future phase
- Multi-user / auth on FastAPI server — not needed for local tool
- CrewAI orchestration — v1 uses Gemini structured output directly; CrewAI deferred to v2+
- Mobile app — web/CLI first
- Web dashboard — future phase

## Context

- Replaces an existing n8n workflow for video generation with a standalone Python pipeline
- Target platform: local development machine with Google Cloud credentials (ADC)
- Google Cloud project: `hoyack-1577568661630`, region: `us-central1`
- Models: `gemini-3-pro` (LLM), `gemini-3-pro-image-preview` (images), `veo-3.1-generate-001` (video)
- Unified Google GenAI SDK (`google-genai`) in Vertex AI mode
- ffmpeg required as system dependency
- Python 3.11+
- Estimated cost: ~$15 per 5-scene project (dominated by Veo at $0.75/sec)

## Constraints

- **Tech Stack**: Python 3.11+, FastAPI, SQLAlchemy, google-genai SDK — specified in spec
- **API Provider**: Google Vertex AI only (no OpenAI, no Replicate) — existing cloud project and auth
- **Video Model**: Veo 3.1 (`veo-3.1-generate-001`) — only model with first+last frame interpolation
- **Image Model**: Nano Banana Pro (`gemini-3-pro-image-preview`) — required for image-conditioned generation
- **Local Storage**: SQLite + filesystem — no cloud database for v1
- **System Dependency**: ffmpeg must be on PATH — used for video stitching
- **Sequential Keyframes**: Keyframe generation is inherently sequential (continuity constraint)
- **Rate Limits**: Nano Banana Pro ~10 req/min on free tier; configurable delay between calls

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Gemini structured output over CrewAI | Simpler for v1, Gemini natively supports JSON schema output | — Pending |
| ffmpeg over MoviePy | Concat demuxer is instant and lossless for same-codec clips | — Pending |
| Vertex AI over Gemini API | Consistent auth via ADC, matches existing n8n workflow | — Pending |
| Sequential keyframes | Visual continuity requires scene N end → scene N+1 start | — Pending |
| SQLite over cloud DB | Local-first tool, no multi-user needs in v1 | — Pending |
| typer + rich for CLI | Modern Python CLI with pretty output | — Pending |

---
*Last updated: 2026-02-14 after initialization*
