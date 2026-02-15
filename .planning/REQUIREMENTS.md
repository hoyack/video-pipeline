# Requirements: Viral Video Generation Pipeline (vidpipe)

**Defined:** 2026-02-14
**Core Value:** Accept a text prompt and produce a cohesive, multi-scene short video with visual continuity — fully automated, crash-safe, and resumable.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundation

- [ ] **FOUND-01**: Project uses Python 3.11+ with SQLAlchemy 2.0, Pydantic 2.0, and async-first patterns
- [ ] **FOUND-02**: SQLite database with WAL mode stores all pipeline state (projects, scenes, keyframes, clips, runs)
- [ ] **FOUND-03**: Configuration loaded from config.yaml/.env with typed validation via pydantic-settings
- [ ] **FOUND-04**: Local filesystem stores binary artifacts in tmp/{project_id}/ with structured subdirectories

### Storyboard

- [ ] **STOR-01**: User submits text prompt and receives structured storyboard with scenes, keyframe prompts, and motion descriptions
- [ ] **STOR-02**: Storyboard uses Gemini 3 Pro with JSON schema structured output (responseMimeType: application/json)
- [ ] **STOR-03**: Each scene includes scene_description, start_frame_prompt, end_frame_prompt, video_motion_prompt, and transition_notes
- [ ] **STOR-04**: Storyboard generates a style guide (visual_style, color_palette, camera_style) for cross-scene consistency
- [ ] **STOR-05**: Invalid JSON from LLM is retried up to 3 times with temperature adjustment before failing

### Keyframe Generation

- [ ] **KEYF-01**: Start keyframe for scene 0 is generated from start_frame_prompt using Nano Banana Pro
- [ ] **KEYF-02**: End keyframe for each scene is generated using start keyframe image + end_frame_prompt (image-conditioned)
- [ ] **KEYF-03**: Scene N+1's start keyframe is inherited from scene N's end keyframe (visual continuity)
- [ ] **KEYF-04**: Keyframe generation is sequential to maintain continuity across scenes
- [ ] **KEYF-05**: Rate limiting with exponential backoff (max 5 retries, configurable delay between calls)
- [ ] **KEYF-06**: Keyframe images saved as PNG to tmp/{project_id}/keyframes/

### Video Generation

- [ ] **VGEN-01**: Each scene's video clip is generated using Veo 3.1 with first-frame + last-frame interpolation
- [ ] **VGEN-02**: Long-running Veo operations are polled with configurable interval (default 15s) and timeout (default ~10min)
- [ ] **VGEN-03**: Operation ID is persisted to database before polling begins (idempotent resume)
- [ ] **VGEN-04**: RAI-filtered clips are marked as rai_filtered and pipeline continues with remaining scenes
- [ ] **VGEN-05**: Timed-out operations are marked as timed_out after max polls exceeded
- [ ] **VGEN-06**: Video clips saved as MP4 to tmp/{project_id}/clips/

### Stitching

- [ ] **STCH-01**: All completed clips are concatenated into a single MP4 using ffmpeg concat demuxer (hard cuts)
- [ ] **STCH-02**: Optional crossfade transitions supported via ffmpeg xfade filter with configurable duration
- [ ] **STCH-03**: Audio streams from Veo 3.1 are preserved during concatenation
- [ ] **STCH-04**: Final output saved to tmp/{project_id}/output/final.mp4
- [ ] **STCH-05**: ffmpeg availability is validated at startup with clear error if missing

### Pipeline Orchestration

- [ ] **ORCH-01**: Pipeline follows state machine: STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE
- [ ] **ORCH-02**: Each step checks database before executing and skips already-completed work (resume capability)
- [ ] **ORCH-03**: Pipeline run metadata tracked (start time, duration, cost estimate, step log)
- [ ] **ORCH-04**: Failed pipeline can be resumed from last completed step via resume command

### CLI Interface

- [ ] **CLI-01**: User can generate video from prompt via `python -m vidpipe generate "prompt"` with style, aspect-ratio, clip-duration options
- [ ] **CLI-02**: User can resume a failed/incomplete project via `python -m vidpipe resume <project_id>`
- [ ] **CLI-03**: User can check project status via `python -m vidpipe status <project_id>`
- [ ] **CLI-04**: User can list all projects via `python -m vidpipe list`
- [ ] **CLI-05**: User can re-stitch with crossfade via `python -m vidpipe stitch <project_id> --crossfade 0.5`

### HTTP API

- [ ] **API-01**: POST /api/generate starts new pipeline run in background and returns project_id immediately
- [ ] **API-02**: GET /api/projects/{id}/status returns lightweight status for polling
- [ ] **API-03**: GET /api/projects/{id} returns full project detail with scenes and clips
- [ ] **API-04**: GET /api/projects lists all projects
- [ ] **API-05**: POST /api/projects/{id}/resume resumes a failed pipeline
- [ ] **API-06**: GET /api/projects/{id}/download serves final MP4 file
- [ ] **API-07**: GET /api/health returns health check

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Generation

- **ADVG-01**: Parallel video generation — submit all Veo jobs concurrently after keyframes complete
- **ADVG-02**: Cost estimation before generation based on scene count and duration
- **ADVG-03**: Preview/dry-run mode — review storyboard and keyframes before video generation
- **ADVG-04**: Scene extension using Veo 3.1's extend feature for longer scenes

### Content Enhancement

- **CENH-01**: Audio narration / TTS overlay synced to scenes
- **CENH-02**: Background music generation or library mixing
- **CENH-03**: Reference images for stronger character/style consistency across scenes
- **CENH-04**: Prompt template library with proven viral video structures

### Infrastructure

- **INFR-01**: Cloud storage upload (GCS, S3) for distribution
- **INFR-02**: Webhook callbacks to notify external systems on completion
- **INFR-03**: Web dashboard showing pipeline progress and previewing keyframes
- **INFR-04**: Docker containerized deployment

## Out of Scope

| Feature | Reason |
|---------|--------|
| CrewAI orchestration | v1 uses Gemini structured output directly; CrewAI adds complexity without clear benefit for single-pass pipeline |
| Multi-user auth on API | Local tool, single-user; auth adds complexity with no benefit |
| Real-time streaming/WebSocket | Video generation is batch, not real-time; polling is sufficient |
| Mobile app | CLI + HTTP API sufficient for target users |
| Video editing features | Scope creep into full editor; recommend external tools for post-processing |
| Character consistency (v1) | Industry-wide unsolved problem; defer to v2+ after validating core pipeline |
| In-app video player | Users have VLC/browser; standard MP4 output is sufficient |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Complete |
| STOR-01 | Phase 2 | Complete |
| STOR-02 | Phase 2 | Complete |
| STOR-03 | Phase 2 | Complete |
| STOR-04 | Phase 2 | Complete |
| STOR-05 | Phase 2 | Complete |
| KEYF-01 | Phase 2 | Complete |
| KEYF-02 | Phase 2 | Complete |
| KEYF-03 | Phase 2 | Complete |
| KEYF-04 | Phase 2 | Complete |
| KEYF-05 | Phase 2 | Complete |
| KEYF-06 | Phase 2 | Complete |
| VGEN-01 | Phase 2 | Complete |
| VGEN-02 | Phase 2 | Complete |
| VGEN-03 | Phase 2 | Complete |
| VGEN-04 | Phase 2 | Complete |
| VGEN-05 | Phase 2 | Complete |
| VGEN-06 | Phase 2 | Complete |
| STCH-01 | Phase 2 | Complete |
| STCH-02 | Phase 2 | Complete |
| STCH-03 | Phase 2 | Complete |
| STCH-04 | Phase 2 | Complete |
| STCH-05 | Phase 2 | Complete |
| ORCH-01 | Phase 3 | Complete |
| ORCH-02 | Phase 3 | Complete |
| ORCH-03 | Phase 3 | Complete |
| ORCH-04 | Phase 3 | Complete |
| CLI-01 | Phase 3 | Complete |
| CLI-02 | Phase 3 | Complete |
| CLI-03 | Phase 3 | Complete |
| CLI-04 | Phase 3 | Complete |
| CLI-05 | Phase 3 | Complete |
| API-01 | Phase 3 | Complete |
| API-02 | Phase 3 | Complete |
| API-03 | Phase 3 | Complete |
| API-04 | Phase 3 | Complete |
| API-05 | Phase 3 | Complete |
| API-06 | Phase 3 | Complete |
| API-07 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 41 total
- Mapped to phases: 41
- Unmapped: 0

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-02-14 after Phase 3 completion (all v1 requirements complete)*
