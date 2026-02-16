# Roadmap: Viral Video Generation Pipeline (vidpipe)

## Overview

This roadmap transforms a text prompt into a multi-scene AI-generated video. Phases 1-3 (V1) established the core pipeline: crash-safe state management, Vertex AI content generation, and CLI/API interfaces. Phases 4-12 (V2) evolve the pipeline into a studio-grade production system built around reusable manifests, an asset registry with reverse-engineered prompts, computer vision analysis, adaptive prompt rewriting, and reference image passthrough to Veo 3.1. The V2 architecture draws from professional VFX pipeline practices — asset management, shot breakdowns, continuity tracking — but replaces human-in-the-loop handoffs with LLM-driven orchestration.

**Reference Docs:** `docs/v2-manifest.md`, `docs/v2-pipe-optimization.md`

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Crash-safe state management, database, config, and artifact storage ✓
- [x] **Phase 2: Generation Pipeline** - Vertex AI integration and all content generators (storyboard, keyframes, video, stitch) ✓
- [x] **Phase 3: Orchestration & Interfaces** - State machine coordinator, resume logic, CLI, and HTTP API ✓
- [ ] **Phase 4: Manifest System Foundation** - Database schema, CRUD API, Manifest Library UI, Creator Stage 1 (upload + tag)
- [ ] **Phase 5: Manifesting Engine** - YOLO detection, ArcFace face matching, Gemini reverse-prompting, Creator Stages 2+3
- [ ] **Phase 6: GenerateForm Integration** - Manifest selector, snapshotting, conditional Phase 0 skip
- [ ] **Phase 7: Manifest-Aware Storyboarding and Audio Manifest** - Enhanced storyboard with asset context, scene manifests, audio direction
- [ ] **Phase 8: Veo Reference Passthrough and Clean Sheets** - 3-reference selection logic, background removal, clean sheet generation
- [ ] **Phase 9: CV Analysis Pipeline and Progressive Enrichment** - Post-generation YOLO + face matching + CLIP, asset extraction from generated content
- [ ] **Phase 10: Adaptive Prompt Rewriting** - Dynamic prompt enrichment with continuity checking and LLM rewriter
- [ ] **Phase 11: Multi-Candidate Quality Mode** - sampleCount configuration, composite scoring pipeline, candidate comparison UI
- [ ] **Phase 12: Fork System Integration with Manifests** - Asset/manifest inheritance, incremental manifesting on fork

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
**Plans**: 3 plans in 2 waves

Plans:
- [x] 03-01-PLAN.md — Pipeline orchestrator with state machine, resume logic, and run metadata tracking
- [x] 03-02-PLAN.md — Typer CLI interface with generate, resume, status, list, and stitch commands
- [x] 03-03-PLAN.md — FastAPI HTTP API with 7 endpoints for async generation, polling, and downloads

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → ... → 12

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 3/3 | ✓ Complete | 2026-02-14 |
| 2. Generation Pipeline | 4/4 | ✓ Complete | 2026-02-14 |
| 3. Orchestration & Interfaces | 3/3 | ✓ Complete | 2026-02-14 |
| 4. Manifest System Foundation | 0/3 | ○ Planned | — |
| 5. Manifesting Engine | 0/? | ○ Not planned | — |
| 6. GenerateForm Integration | 0/? | ○ Not planned | — |
| 7. Manifest-Aware Storyboarding | 0/? | ○ Not planned | — |
| 8. Veo Reference Passthrough | 0/? | ○ Not planned | — |
| 9. CV Analysis Pipeline | 0/? | ○ Not planned | — |
| 10. Adaptive Prompt Rewriting | 0/? | ○ Not planned | — |
| 11. Multi-Candidate Quality Mode | 0/? | ○ Not planned | — |
| 12. Fork System Integration | 0/? | ○ Not planned | — |

### Phase 4: Manifest System Foundation
**Goal**: Manifests exist as standalone, reusable entities with CRUD API, database storage, and a frontend Manifest Library view with filter/sort plus a Manifest Creator that supports Stage 1 (upload + tag, no processing yet)
**Depends on**: Phase 3
**Success Criteria** (what must be TRUE):
  1. `manifests` table stores standalone manifest entities with name, description, category, tags, status (DRAFT/PROCESSING/READY/ERROR), and versioning
  2. `assets` table updated with `manifest_id` foreign key; assets belong to manifests not directly to projects
  3. `projects` table updated with `manifest_id` and `manifest_version` columns
  4. Manifest CRUD API: list, create, get, update, delete endpoints under `/api/manifests`
  5. Manifest Library view displays manifest cards with contact sheet thumbnails, asset counts, category filters, sort options, and card actions (Edit, Duplicate, Delete, View)
  6. Manifest Creator view supports Stage 1: drag-drop image upload with per-image name, type, description, and tag inputs; saves as DRAFT status with no processing
**Plans:** 3 plans in 2 waves

Plans:
- [ ] 04-01-PLAN.md — Database models (Manifest, Asset, Project additions) and CRUD API with service layer
- [ ] 04-02-PLAN.md — Manifest Library frontend with card grid, filters, sort, and navigation
- [ ] 04-03-PLAN.md — Manifest Creator Stage 1 with drag-drop upload and asset tagging

### Phase 5: Manifesting Engine
**Goal**: Manifest Creator processes uploaded images through YOLO object/face detection, ArcFace face embedding and cross-matching, Gemini vision reverse-prompting, contact sheet assembly, and tag assignment — populating the Asset Registry automatically
**Depends on**: Phase 4
**Success Criteria** (what must be TRUE):
  1. YOLO detection sweep runs on each uploaded image, extracting object and face crops with bounding boxes and confidence scores
  2. ArcFace face embeddings are generated for every detected face; cross-matching merges same-person detections across uploads (similarity > 0.6)
  3. Gemini vision reverse-prompting generates `reverse_prompt` (recreation-style prompt text) and `visual_description` (production bible entry) for each crop
  4. Contact sheet assembled via Pillow with numbered grid layout and labels
  5. Manifest tags auto-assigned (CHAR_01, ENV_01, PROP_01, etc.) and Asset Registry populated with all fields
  6. Manifest Creator supports Stages 2 (processing with live progress) and 3 (review and refine: edit prompts, swap images, re-process, remove assets)
  7. Processing progress tracked with status transitions: DRAFT → PROCESSING → READY
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 5 to break down)

### Phase 6: GenerateForm Integration
**Goal**: Users can select an existing manifest from the library or quick-upload inline when generating a video; projects reference manifests with snapshot isolation so in-progress projects are unaffected by manifest edits
**Depends on**: Phase 5
**Success Criteria** (what must be TRUE):
  1. GenerateForm shows manifest selector: "Select Existing Manifest" or "Quick Upload (inline)"
  2. Selecting existing manifest shows manifest card preview with asset summary and key asset thumbnails
  3. Quick Upload creates an auto-manifest behind the scenes (same as inline reference upload)
  4. `manifest_snapshots` table freezes manifest state at generation start; completed projects reference exact snapshot used
  5. Pipeline conditionally skips Phase 0 (manifesting) when a pre-built manifest is selected
  6. Usage tracking: `times_used` and `last_used_at` updated on manifest when selected for a project
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 6 to break down)

### Phase 7: Manifest-Aware Storyboarding and Audio Manifest
**Goal**: Storyboard LLM receives full Asset Registry context and produces scene manifests with manifest-tagged asset placements, plus per-scene audio manifests with dialogue, SFX, ambient, and music direction
**Depends on**: Phase 6
**Success Criteria** (what must be TRUE):
  1. Enhanced storyboard system prompt includes all registered assets with manifest_tags, reverse_prompts, and quality scores
  2. Storyboard output includes per-scene `SceneManifest` with asset placements (tag, role, position, action, expression, wardrobe notes)
  3. Scene manifests include composition metadata (shot_type, camera_movement, focal_point) and continuity notes
  4. `scene_manifests` table stores structured manifest JSON per scene
  5. Per-scene `SceneAudioManifest` generated with dialogue lines mapped to character tags, SFX with timing, ambient layers, and music direction
  6. `scene_audio_manifests` table stores audio manifest per scene
  7. LLM can declare NEW assets not in registry (described textually, generated during keyframe phase)
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 7 to break down)

### Phase 8: Veo Reference Passthrough and Clean Sheets
**Goal**: Video generation passes up to 3 asset reference images per scene to Veo 3.1 for identity consistency, with optional clean sheet generation to optimize reference quality
**Depends on**: Phase 7
**Success Criteria** (what must be TRUE):
  1. Reference selection logic picks 3 most relevant assets per scene based on manifest placements and scene type (character close-up, two-shot, establishing shot, etc.)
  2. Selected references passed as `referenceImages` with `referenceType: "asset"` to Veo 3.1 API
  3. Hybrid approach: first-frame from keyframe daisy-chain (`image` param) + 3 reference images for identity
  4. Clean sheet generation available per asset: background removal (rembg), full clean sheet via Gemini Image, multi-angle sheet
  5. `asset_clean_references` table stores clean reference images with tier, quality score, and face similarity score
  6. SceneCard in frontend shows which 3 references were selected per scene
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 8 to break down)

### Phase 9: CV Analysis Pipeline and Progressive Enrichment
**Goal**: Post-generation CV analysis runs YOLO + face matching + CLIP on generated keyframes and video clips, extracting new assets and progressively enriching the registry so later scenes benefit from earlier extractions
**Depends on**: Phase 8
**Success Criteria** (what must be TRUE):
  1. YOLO object/face detection runs on generated keyframes (local GPU, per-frame)
  2. ArcFace embeddings match detected faces against Asset Registry; new faces registered as new assets
  3. CLIP embeddings generated for general visual similarity matching
  4. Gemini Vision semantic analysis provides scene understanding, continuity assessment, and quality rating
  5. Video clip analysis uses frame sampling strategy (first, 2s, 4s, 6s, last + motion delta frames) — ~5-8 frames per clip
  6. New entities extracted from generated content are reverse-prompted and registered in Asset Registry
  7. `asset_appearances` table tracks where each asset appears across scenes
  8. Progressive enrichment: scene N+1 generation benefits from assets extracted from scenes 1..N
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 9 to break down)

### Phase 10: Adaptive Prompt Rewriting
**Goal**: A dedicated LLM rewriter assembles final generation prompts by injecting asset reverse_prompts, manifest metadata, continuity corrections, and audio direction — replacing static storyboard prompts with dynamically enriched versions
**Depends on**: Phase 9
**Success Criteria** (what must be TRUE):
  1. Prompt assembly pipeline combines: original storyboard prompt + manifest enrichment + asset injection + continuity patch + reference selection
  2. Dedicated Gemini rewriter call produces scene prompts under 500 words following cinematography formula
  3. Rewriter selects which 3 reference images to attach with reasoning
  4. Continuity checking compares scene N-1 end state with scene N start requirements and patches prompts accordingly
  5. Reverse prompts refined based on what models actually produce (not just initial descriptions)
  6. `scene_manifests.rewritten_keyframe_prompt` and `rewritten_video_prompt` stored separately from original prompt
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 10 to break down)

### Phase 11: Multi-Candidate Quality Mode
**Goal**: Users can generate 2-4 candidate clips per scene with composite quality scoring (manifest adherence, visual quality, continuity, prompt adherence) and select the best take manually or automatically
**Depends on**: Phase 10
**Success Criteria** (what must be TRUE):
  1. `sampleCount` configurable per project (1-4, default 1 for standard mode)
  2. `generation_candidates` table stores per-scene candidates with individual scores and composite score
  3. Scoring pipeline evaluates: manifest adherence (0.35 weight), visual quality (0.25), continuity (0.25), prompt adherence (0.15)
  4. Face matching confirms character identity against manifest for adherence scoring
  5. CLIP embedding similarity between scene N-1 last frame and candidate first frame for continuity scoring
  6. Candidate comparison UI shows all candidates with scores; user can manually override auto-selection
  7. Cost impact clearly shown (Quality Mode ~2x video generation cost)
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 11 to break down)

### Phase 12: Fork System Integration with Manifests
**Goal**: Forked projects inherit the parent's full Asset Registry, manifest reference, and scene manifests with proper invalidation rules; users can add new reference uploads, modify assets, or remove assets in the fork with incremental manifesting
**Depends on**: Phase 11
**Success Criteria** (what must be TRUE):
  1. Forked project copies all parent assets with `is_inherited=true` and shared GCS URLs (no re-processing, $0 cost)
  2. Project manifest inherited with `inherited_from_project` tracking
  3. Scene manifests inherited for unchanged scenes; invalidated scenes get blank manifests for regeneration
  4. Users can add new reference uploads in fork triggering incremental manifesting (only new uploads processed)
  5. Modified assets (swapped reference image, edited reverse_prompt) invalidate scenes using that asset from the modification point forward
  6. Face embeddings cross-matched against ALL assets (inherited + new) during incremental manifesting
  7. EditForkPanel shows inherited assets with lock/edit/remove controls and "Add New Reference Images" option
