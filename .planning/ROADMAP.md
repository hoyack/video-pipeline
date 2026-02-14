# Roadmap: Viral Video Generation Pipeline (vidpipe)

## Overview

This roadmap transforms a text prompt into a multi-scene AI-generated video through three delivery phases: Foundation establishes crash-safe state management and configuration, Generation Pipeline implements all content creation stages from storyboarding through stitching, and Orchestration & Interfaces ties everything together with resume logic and user-facing CLI/API. The architecture prioritizes crash recovery from day one to prevent wasted API costs during expensive Veo video generation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Crash-safe state management, database, config, and artifact storage ✓
- [x] **Phase 2: Generation Pipeline** - Vertex AI integration and all content generators (storyboard, keyframes, video, stitch) ✓
- [ ] **Phase 3: Orchestration & Interfaces** - State machine coordinator, resume logic, CLI, and HTTP API

## Phase Details

### Phase 1: Foundation
**Goal**: Project can persist all pipeline state to crash-safe SQLite database, load validated configuration, and manage filesystem artifacts in structured directories
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04
**Success Criteria** (what must be TRUE):
  1. SQLite database with WAL mode enabled stores all pipeline entities (projects, scenes, keyframes, clips, runs)
  2. Configuration loads from config.yaml and environment variables with type validation
  3. Binary artifacts save to tmp/{project_id}/ with structured subdirectories (keyframes/, clips/, output/)
  4. Database operations survive crashes without corruption (WAL + synchronous=FULL)
**Plans**: 3 plans in 3 waves

Plans:
- [x] 01-01-PLAN.md — Project structure and SQLAlchemy models with Mapped annotations
- [x] 01-02-PLAN.md — Configuration loading with pydantic-settings and YAML source
- [x] 01-03-PLAN.md — Database engine with WAL mode, file manager, and schema initialization

### Phase 2: Generation Pipeline
**Goal**: Pipeline generates storyboards, keyframes, video clips, and stitched output from text prompts using Google Vertex AI APIs
**Depends on**: Phase 1
**Requirements**: STOR-01, STOR-02, STOR-03, STOR-04, STOR-05, KEYF-01, KEYF-02, KEYF-03, KEYF-04, KEYF-05, KEYF-06, VGEN-01, VGEN-02, VGEN-03, VGEN-04, VGEN-05, VGEN-06, STCH-01, STCH-02, STCH-03, STCH-04, STCH-05
**Success Criteria** (what must be TRUE):
  1. User submits text prompt and receives structured storyboard with scenes, keyframe prompts, motion descriptions, and style guide
  2. Keyframes are generated sequentially with visual continuity (scene N end frame becomes scene N+1 start frame)
  3. Video clips are generated using Veo 3.1 with first/last frame control and long-running operations are polled with backoff
  4. RAI-filtered clips are marked and pipeline continues without crashing
  5. All completed clips are concatenated into single MP4 with optional crossfade transitions
  6. ffmpeg is validated at startup with clear error if missing
**Plans**: 4 plans in 4 waves

Plans:
- [x] 02-01-PLAN.md — Storyboard generation with Gemini structured output
- [x] 02-02-PLAN.md — Sequential keyframe generation with visual continuity
- [x] 02-03-PLAN.md — Video generation with Veo polling and error handling
- [x] 02-04-PLAN.md — Video stitching with ffmpeg and startup validation

### Phase 3: Orchestration & Interfaces
**Goal**: Users can generate videos via CLI or HTTP API with full crash recovery, status tracking, and resume capability
**Depends on**: Phase 2
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04, CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, API-01, API-02, API-03, API-04, API-05, API-06, API-07
**Success Criteria** (what must be TRUE):
  1. Pipeline follows state machine transitions (STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE) with database-tracked progress
  2. Failed pipeline can resume from last completed step without redoing completed work
  3. User can generate video via CLI command with configurable style, aspect ratio, and clip duration options
  4. User can check project status, list all projects, resume failed projects, and re-stitch with crossfade via CLI
  5. HTTP API accepts generation requests in background and returns project_id immediately
  6. HTTP API serves status polling, project details, project listing, resume triggers, and final MP4 downloads
**Plans**: TBD

Plans:
- [ ] 03-01: [Brief description]

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 3/3 | ✓ Complete | 2026-02-14 |
| 2. Generation Pipeline | 4/4 | ✓ Complete | 2026-02-14 |
| 3. Orchestration & Interfaces | 0/TBD | Not started | - |
