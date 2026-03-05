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
- [x] **Phase 10: Adaptive Prompt Rewriting** - Dynamic prompt enrichment with continuity checking and LLM rewriter (completed 2026-02-17)
- [x] **Phase 11: Multi-Candidate Quality Mode** - sampleCount configuration, composite scoring pipeline, candidate comparison UI (completed 2026-02-17)
- [x] **Phase 12: Fork System Integration with Manifests** - Asset/manifest inheritance, incremental manifesting on fork (completed 2026-02-17)
- [x] **Phase 13: LLM Provider Abstraction & Ollama Integration** - LLM adapter pattern, Vertex AI adapter extraction, Ollama text/vision adapter, settings UI, model management, pipeline wiring (completed 2026-02-19)
- [x] **Phase 14: Storyboard Manifest Asset Binding Fix** - Defense-in-depth fix for storyboard LLM creating new character tags instead of using existing manifest assets; prompt hardening, post-LLM tag remapping, prompt rewriter fallback, keyframe enforcement fallback (completed 2026-02-20)
- [x] **Phase 15: Video Generation Editor** - Unified create/edit/monitor experience replacing GenerateForm + ProgressView; draft projects, gap-filling pipeline, per-scene uploads, VideoGenEditor component, pause/resume (completed 2026-02-21)
- [x] **Phase 16: Production Bible Foundation** - Rename Manifest → Production Bible across stack, department tab structure, Sequence grouping layer above Scenes (Issues #7, #24) (completed 2026-03-01)
- [x] **Phase 17: Production Bible Entity Expansion** - Full Character, Set, Prop entities with sub-entities (Wardrobe, VoiceProfile, SonicIdentity), Score Themes and SFX Library in Sound Department (Issues #8, #9, #10, #11) (completed 2026-03-01)
- [x] **Phase 18: Screenplay System** - Screenplay data model and editor, Screenwriter CrewAI agent for LLM generation chain, Scene Breakdown → pipeline wiring (Issues #12, #13, #14) (completed 2026-03-01)
- [x] **Phase 19: Bible Context Fix + Code Cleanup** - Fix load_bible_context indirect lookup, remove dead manifest strings/orphan files/dead code (Gap Closure) (completed 2026-03-01)
- [x] **Phase 20: Entity Media Uploads** - Missing upload endpoints and UI for actor refs, wardrobe, audio, props across Production Bible entities (Issues #8, #9, #10, #11) (Gap Closure) (completed 2026-03-01)
- [x] **Phase 21: Sequence UI Polish** - Wire sequence drag-reorder, act field, duration display, within-sequence scene reorder (Issue #24) (Gap Closure) (completed 2026-03-01)
- [ ] **Phase 22: Asset Library & Actor-Character Model** - Global Asset Library (Actors, Sets, Props, Sound Assets), Actor vs Character distinction, binding system for Production Bibles, scene tag resolution

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
Phases execute in numeric order: 1 → 2 → 3 → 4 → ... → 16

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 3/3 | ✓ Complete | 2026-02-14 |
| 2. Generation Pipeline | 4/4 | ✓ Complete | 2026-02-14 |
| 3. Orchestration & Interfaces | 3/3 | ✓ Complete | 2026-02-14 |
| 4. Manifest System Foundation | 3/3 | ✓ Complete | 2026-02-16 |
| 5. Manifesting Engine | 0/3 | ○ Planned | — |
| 6. GenerateForm Integration | 0/2 | ○ Planned | — |
| 7. Manifest-Aware Storyboarding | 0/2 | ○ Planned | — |
| 8. Veo Reference Passthrough | 0/3 | ○ Planned | — |
| 9. CV Analysis Pipeline | 0/3 | ○ Planned | — |
| 10. Adaptive Prompt Rewriting | 0/2 | Complete    | 2026-02-17 |
| 11. Multi-Candidate Quality Mode | 0/3 | Complete    | 2026-02-17 |
| 12. Fork System Integration | 0/3 | Complete    | 2026-02-17 |
| 13. LLM Provider Abstraction | 3/3 | Complete    | 2026-02-19 |
| 14. Storyboard Asset Binding | 1/1 | Complete    | 2026-02-20 |
| 15. Video Generation Editor | 3/3 | Complete    | 2026-02-21 |
| 16. Production Bible Foundation | 4/4 | Complete    | 2026-03-01 |
| 17. PB Entity Expansion | 4/4 | Complete    | 2026-03-01 |
| 18. Screenplay System | 3/3 | Complete    | 2026-03-01 |
| 19. Bible Context Fix + Cleanup | 2/2 | Complete    | 2026-03-01 |
| 20. Entity Media Uploads | 2/2 | Complete    | 2026-03-01 |
| 21. Sequence UI Polish | 2/2 | Complete    | 2026-03-01 |
| 22. Asset Library & Actor-Character | 1/6 | In Progress|  |

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
- [x] 04-01-PLAN.md — Database models (Manifest, Asset, Project additions) and CRUD API with service layer
- [x] 04-02-PLAN.md — Manifest Library frontend with card grid, filters, sort, and navigation
- [x] 04-03-PLAN.md — Manifest Creator Stage 1 with drag-drop upload and asset tagging

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
  7. Processing progress tracked with status transitions: DRAFT -> PROCESSING -> READY
**Plans:** 3 plans in 3 waves

Plans:
- [ ] 05-01-PLAN.md — Asset model migration + CV detection, face matching, and reverse-prompt services
- [ ] 05-02-PLAN.md — ManifestingEngine orchestrator, background task runner, and processing API endpoints
- [ ] 05-03-PLAN.md — Frontend Stages 2 (processing progress) and 3 (review/refine) in ManifestCreator

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
**Plans:** 2 plans in 2 waves

Plans:
- [ ] 06-01-PLAN.md — ManifestSnapshot model, snapshot service, usage tracking, enhanced generate endpoint
- [ ] 06-02-PLAN.md — ManifestSelector component, compact ManifestCard, GenerateForm integration

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
**Plans:** 2 plans in 2 waves

Plans:
- [ ] 07-01-PLAN.md — Enhanced Pydantic schemas and SceneManifest/SceneAudioManifest database models
- [ ] 07-02-PLAN.md — Enhanced storyboard pipeline with manifest context injection and manifest persistence

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
**Plans:** 3 plans in 2 waves

Plans:
- [ ] 08-01-PLAN.md — AssetCleanReference model, SceneManifest column, and reference selection service
- [ ] 08-02-PLAN.md — Video gen pipeline with reference passthrough and clean sheet generation service
- [ ] 08-03-PLAN.md — Frontend SceneCard reference badges and API SceneDetail enhancement

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
**Plans:** 3 plans in 3 waves

Plans:
- [ ] 09-01-PLAN.md — CLIP embedding service, video frame sampler, AssetAppearance model, and cv_analysis config
- [ ] 09-02-PLAN.md — CV analysis orchestrator service and entity extraction/registration service
- [ ] 09-03-PLAN.md — Pipeline integration with per-scene analysis hook and progressive enrichment

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
**Plans:** 2/2 plans complete

Plans:
- [ ] 10-01-PLAN.md — PromptRewriterService, pydantic output schemas, SceneManifest columns, and SQL migration
- [ ] 10-02-PLAN.md — Pipeline integration: keyframes.py and video_gen.py rewriter hooks with LLM reference override

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
**Plans:** 3/3 plans complete

Plans:
- [ ] 11-01-PLAN.md — GenerationCandidate model, Project quality columns, CandidateScoringService
- [ ] 11-02-PLAN.md — Pipeline integration: multi-candidate video_gen.py with scoring and auto-selection
- [ ] 11-03-PLAN.md — Candidate API endpoints, GenerateForm Quality Mode toggle, SceneCard comparison UI

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
**Plans:** 3/3 plans complete

Plans:
- [ ] 12-01-PLAN.md — DB migration, Asset inheritance ORM fields, ForkRequest/AssetChanges schemas, ProjectDetail manifest_id
- [ ] 12-02-PLAN.md — Fork endpoint asset copy, scene manifest inheritance, invalidation extension, incremental manifesting
- [ ] 12-03-PLAN.md — Frontend TypeScript types, API client, EditForkPanel asset management UI

### Phase 13: LLM Provider Abstraction & Ollama Integration
**Goal**: Abstract all LLM text/vision calls behind a provider adapter interface, extract existing Vertex AI/Gemini calls into a Vertex adapter, implement an Ollama adapter for text and vision models, add settings UI for Ollama configuration (API key, cloud/local toggle, endpoint, model management), and wire the pipeline to route through the correct adapter based on selected model provider
**Depends on**: Phase 3 (core pipeline), independent of Phases 4-12 (manifest/CV features)
**Requirements**: LLMA-01, LLMA-02, LLMA-03, LLMA-04, LLMA-05, LLMA-06, LLMA-07
**Success Criteria** (what must be TRUE):
  1. `LLMAdapter` abstract base class defines text generation (with structured JSON output) and vision analysis interfaces
  2. `VertexAIAdapter` wraps existing `get_vertex_client()` calls; all current Gemini text/vision call sites route through it with zero behavior change
  3. `OllamaAdapter` implements text generation and vision analysis using Ollama API (both local and cloud endpoints)
  4. Settings UI shows Ollama section: API key input, cloud/local toggle (local disables API key), endpoint URL, model add/remove/toggle
  5. Custom Ollama models added via input box appear in the text/vision model lists and can be toggled on/off or removed
  6. GenerateForm supports separate text_model and vision_model selection; pipeline routes storyboard through text adapter and image analysis through vision adapter
  7. Existing Vertex AI/Gemini pipeline works identically when Gemini models are selected (no regression)
  8. Provider detection automatic: model ID prefix or provider registry determines which adapter handles each call
**Plans:** 3/3 plans complete

Plans:
- [x] 13-01-PLAN.md — LLM adapter package (base, VertexAI, Ollama, registry), vision schemas, DB schema, settings API
- [x] 13-02-PLAN.md — Call-site migration (storyboard, prompt rewriter, reverse-prompt, CV analysis, candidate scoring), orchestrator wiring, route validation
- [x] 13-03-PLAN.md — Frontend Ollama settings UI, model management, GenerateForm vision_model dropdown

### Phase 14: Storyboard Manifest Asset Binding Fix
**Goal**: Storyboard LLM uses existing manifest CHARACTER tags instead of inventing new ones, with defense-in-depth safety nets ensuring reference images always reach the image adapter and face verification is never silently skipped
**Depends on**: Phase 7, Phase 10 (existing storyboard + prompt rewriter code)
**Requirements**: SBIND-01, SBIND-02, SBIND-03, SBIND-04
**Success Criteria** (what must be TRUE):
  1. Storyboard prompt mandates using existing CHARACTER tags from the asset registry; `new_asset_declarations` restricted to non-CHARACTER types only
  2. Post-storyboard deterministic remapping catches any LLM-invented CHARACTER tags and maps them to existing manifest CHARACTER assets
  3. Prompt rewriter falls back to marking ALL manifest CHARACTER assets as MUST SELECT when scene manifest placements reference non-existent tags
  4. Keyframe enforcement falls back to all manifest CHARACTER assets with reference images when `placed_char_tags` resolves empty
  5. Face verification retry loop fires whenever manifest has CHARACTER assets, regardless of scene manifest tag accuracy
  6. No regression for projects without manifests — original code paths unchanged
**Plans:** 1/1 plans complete

Plans:
- [ ] 14-01-PLAN.md — Defense-in-depth: prompt hardening, tag remapping, rewriter fallback, keyframe enforcement fallback

### Phase 15: Video Generation Editor
**Goal**: Replace the GenerateForm + ProgressView two-screen flow with a unified VideoGenEditor that merges project creation, live generation monitoring, and editing into one view — turning the tool from "submit and wait" into a composable project workspace where AI fills gaps and users retain full control
**Depends on**: Phase 3 (core pipeline), Phase 14 (latest completed phase)
**Requirements**: VGED-01, VGED-02, VGED-03, VGED-04, VGED-05, VGED-06, VGED-07, VGED-08, VGED-09, VGED-10, VGED-11, VGED-12
**Success Criteria** (what must be TRUE):
  1. `POST /api/projects` creates a draft project with empty Scene rows without starting the pipeline
  2. `POST /api/projects/{id}/generate` inspects existing assets and only generates what's missing up to the requested stage (gap-filling)
  3. Pipeline stages (storyboard, keyframes, video_gen, stitcher) skip scenes/assets that already exist
  4. Per-scene `generation_status` field tracks granular pipeline progress visible to frontend polling
  5. Scene-level upload endpoints accept keyframe images, video clips, and final video with proper DB row creation
  6. `VideoGenEditor` component provides unified create → generate → monitor → edit experience with SceneEditorCard lifecycle rendering
  7. Generate Through slider controls pipeline stop point; scene cards show per-asset generation spinners
  8. Pause/resume works at per-scene granularity; user can edit assets while paused and resume fills remaining gaps
  9. App.tsx navigation merges "generate" and "progress" views into single "editor" view
  10. Existing `POST /api/generate` endpoint remains functional (backward compatibility)
**Design Doc**: `docs/vidgeneditor.md`
**Plans:** 3/3 plans complete

Plans:
- [ ] 15-01-PLAN.md — Backend: draft status, create project endpoint, generate endpoint, generation_status column, final-video upload
- [ ] 15-02-PLAN.md — Pipeline: gap-filling storyboard/keyframes/video_gen/stitcher, per-scene stop flag, generation_status updates
- [ ] 15-03-PLAN.md — Frontend: VideoGenEditor component, GenerateThroughSlider, ProjectConfigBar, App.tsx navigation merge

### Phase 16: Production Bible Foundation
**Goal**: Rename the Manifest concept to Production Bible across the entire stack (database, API, frontend), introduce department tab structure in the Production Bible detail view, and add an optional Sequence grouping layer above Scenes for narrative chapter organization
**Depends on**: Phase 15 (latest completed phase)
**Requirements**: PBIB-01, PBIB-02, PBIB-03, PBIB-04, PBIB-05, PBIB-06, SEQ-01, SEQ-02, SEQ-03, SEQ-04
**GitHub Issues**: #7, #24
**Success Criteria** (what must be TRUE):
  1. `production_bibles` table exists with all data migrated from `manifests`; all FK columns renamed to `production_bible_id`
  2. All API endpoints respond at `/api/production-bibles/*` with 301 redirects from legacy `/api/manifests/*` paths
  3. Frontend uses "Production Bible" terminology everywhere; routes updated to `/production-bibles/*`
  4. Production Bible detail view has three department tabs: Casting, Art Department, Sound — with existing assets sorted into correct tabs
  5. `sequences` table stores optional grouping layer with title, description, order, act, and color fields
  6. Scene model has optional `sequence_id` FK; scenes with null sequence_id remain in flat list
  7. Sequence CRUD API under `/api/productions/{id}/sequences` with drag-and-drop reorder support
  8. Frontend renders scenes grouped by sequence when sequences exist, with collapsible sections and drag between sequences
**Plans:** 4/4 plans complete

Plans:
- [ ] 16-01-PLAN.md — Backend: DB rename manifests to production_bibles, API endpoint rename, 301 redirects, service layer updates
- [ ] 16-02-PLAN.md — Backend: Sequence model, migration, and CRUD API endpoints
- [x] 16-03-PLAN.md — Frontend: Component renames, route updates, department tabs (Casting, Art Dept, Sound)
- [ ] 16-04-PLAN.md — Frontend: Sequence types, API client, sequence grouping UI with drag-and-drop

### Phase 17: Production Bible Entity Expansion
**Goal**: Expand the Production Bible with full Character, Set, and Prop entities (each with sub-entities and CRUD APIs), plus Score Themes and SFX Library in the Sound Department — providing the structured data layer that generation pipelines, audio tracks, and crew agents depend on
**Depends on**: Phase 16
**Requirements**: PBEX-01, PBEX-02, PBEX-03, PBEX-04, PBEX-05, PBEX-06, PBEX-07, PBEX-08, PBEX-09, PBEX-10, PBEX-11, PBEX-12, PBEX-13, PBEX-14, PBEX-15, PBEX-16, PBEX-17, PBEX-18, PBEX-19, PBEX-20
**GitHub Issues**: #8, #9, #10, #11
**Success Criteria** (what must be TRUE):
  1. Character entity exists with full schema (name, role, description, arc, actor_refs, base_appearance, wardrobe, voice_profile, prompt_tags) and CRUD API
  2. Wardrobe sub-entity per character supports label, reference_images, scene_context, prompt_descriptor, is_default
  3. VoiceProfile sub-entity per character stores voice_id, adapter_type, style_notes, sample_audio (generation disabled until audio adapter ships)
  4. Set entity exists with reference_image, reverse_prompt (auto-generated via LLM Vision), style_tags, lighting_notes, prompt_tags, and SonicIdentity sub-entity
  5. Prop entity exists under Art Department tab with reference_image, description, associated_characters, prompt_tags
  6. ScoreTheme entity exists with mood_descriptors, tempo/usage notes, reference_audio, generation_prompt
  7. SFXItem entity exists with category filter (IMPACT/MECHANICAL/NATURAL/UI/FOLEY/AMBIENCE), source_audio, generation_prompt
  8. All entities have full CRUD APIs with list/create/get/update/delete endpoints
  9. Casting tab shows Character list with detail view (4 tabs: Overview, Actor References, Wardrobe, Voice Profile)
  10. Art Department tab shows Set list with detail view (2 tabs: Visual, Sonic Identity) and Prop list with detail view
  11. Sound Department tab shows Score Themes section and SFX Library section with category filters
  12. Existing manifest character/background assets migrated to Character/Set entities respectively
  13. Scene.score_theme_id nullable FK added for forward compatibility
  14. prompt-context endpoints for Character and Set return injection strings for generation pipeline
**Plans**: 4 plans in 3 waves

Plans:
- [ ] 17-01-PLAN.md — ORM models (Character, Wardrobe, VoiceProfile, Set, SonicIdentity, Prop, ScoreTheme, SFXItem) + Scene.score_theme_id migration
- [ ] 17-02-PLAN.md — Character + Set + Prop CRUD API routes with prompt-context endpoints and LLM Vision reverse-prompting
- [ ] 17-03-PLAN.md — Sound Department CRUD API (ScoreTheme, SFXItem) + asset-to-entity migration service
- [ ] 17-04-PLAN.md — Frontend: CharacterDetail, SetDetail, SoundDepartment components wired into department tabs

### Phase 18: Screenplay System
**Goal**: Introduce Screenplay as a structured narrative document attached to Productions (1:1), with a Screenwriter service that generates screenplay components via LLM chain using the existing adapter pattern, and wire Scene Breakdown into the scene/shot generation pipeline so Scenes are driven by structured narrative intent rather than free-form prompts
**Depends on**: Phase 17
**Requirements**: SCRN-01, SCRN-02, SCRN-03, SCRN-04, SCRN-05, SCRN-06, SCRN-07, SCRN-08, SCRN-09, SCRN-10, SCRN-11, SCRN-12, SCRN-13, SCRN-14, SCRN-15
**GitHub Issues**: #12, #13, #14
**Success Criteria** (what must be TRUE):
  1. Screenplay entity exists with one-to-one Project relationship, storing logline, treatment, character_breakdowns, scene_breakdown, script, shot_list
  2. Scene Breakdown entries reference Production Bible Characters, Sets, and Props
  3. Screenplay CRUD API allows per-component updates and independent regeneration
  4. Screenplay editor UI has 5 tabs (Logline, Treatment, Scene Breakdown, Script, Shot List) with per-tab Regenerate buttons
  5. Screenplay status (DRAFT/IN_REVIEW/LOCKED) controls regeneration permissions
  6. Screenwriter service generates screenplay via sequential LLM chain (existing adapter pattern, not CrewAI): logline → treatment → character_breakdowns → scene_breakdown → script
  7. Each generation step updates Screenplay incrementally and can be run independently
  8. Production Bible Characters and Sets injected as context into generation prompts
  9. "Generate Scenes from Screenplay" creates one Scene per SceneBreakdown entry under the Production, with prompt_tag injection
  10. Free-form storyboard generation remains as fallback when no Screenplay exists
  11. Scenes from Screenplay show "Screenplay linked" badge in UI
**Plans**: 3 plans in 3 waves

Plans:
- [ ] 18-01-PLAN.md — Screenplay ORM model + Scene columns + Pydantic schemas + ScreenwriterService (6-step LLM chain with entity validation)
- [ ] 18-02-PLAN.md — Screenplay CRUD API (11 endpoints), generate-scenes endpoint, storyboard.py enrichment hook
- [ ] 18-03-PLAN.md — Frontend: TypeScript types, API client, ScreenplayEditor (6 tabs), ProductionDetail integration

### Phase 19: Bible Context Fix + Code Cleanup
**Goal**: Fix the `load_bible_context` indirect lookup so Production Bible context is available to the Screenwriter even before any Scenes exist, and remove dead code/orphan files left over from the manifest→Production Bible rename
**Depends on**: Phase 18
**Requirements**: SCRN-10
**Gap Closure**: Closes integration gap `SCRN-10-bible-context` and flow gap `bible-before-scenes` from v1.0 audit
**Success Criteria** (what must be TRUE):
  1. `load_bible_context` looks up Production Bible directly via `Production.production_bible_id` rather than indirectly through Scene FK; returns bible context even when no scenes exist yet
  2. "Bible → Screenplay → Generate Full" flow provides bible context to all Screenwriter generation steps
  3. User-facing "manifest" strings removed from ShotCard.tsx and EditForkPanel.tsx
  4. Orphan files deleted: ManifestLibrary.tsx, ManifestCreator.tsx, ManifestCard.tsx, ManifestSelector.tsx
  5. Dead `sound_router` try/except guard removed from app.py
**Plans**: 2 plans in 1 wave

Plans:
- [ ] 19-01-PLAN.md — Backend: Production.production_bible_id FK, migration, load_bible_context fix, API response update, sound_router cleanup
- [ ] 19-02-PLAN.md — Frontend: Fix manifest strings in ShotCard/EditForkPanel, delete orphan Manifest*.tsx files

### Phase 20: Entity Media Uploads
**Goal**: Add missing upload endpoints and frontend UI for reference images and audio files across all Production Bible entity types
**Depends on**: Phase 17
**Requirements**: PBEX-01, PBEX-02, PBEX-07, PBEX-08, PBEX-13, PBEX-16, PBEX-17
**GitHub Issues**: #8, #9, #10, #11
**Gap Closure**: Closes 8 tech debt items from v1.0 audit
**Success Criteria** (what must be TRUE):
  1. Actor reference image upload endpoint and UI functional (`POST /api/characters/:id/actor-refs`)
  2. Generate Base Appearance endpoint functional (`POST /api/characters/:id/generate-appearance`)
  3. Wardrobe reference image upload endpoint and UI functional
  4. Standalone generate-reverse-prompt endpoint functional (`POST /api/generate-reverse-prompt`)
  5. SonicIdentity reference audio upload UI functional
  6. Prop reference image upload button in frontend works
  7. Audio upload endpoints for ScoreTheme and SFXItem functional
  8. Inline audio playback component renders for entities with audio files
**Plans**: 2 plans in 2 waves

Plans:
- [ ] 20-01-PLAN.md — Backend: 7 upload endpoints (actor-refs, generate-appearance, wardrobe-ref, generate-reverse-prompt, sonic-identity audio, score-theme audio, SFX audio)
- [ ] 20-02-PLAN.md — Frontend: AudioPlayer component, API client upload functions, upload UI in CharacterDetail/SetDetail/SoundDepartment

### Phase 21: Sequence UI Polish
**Goal**: Wire up the remaining Sequence frontend features so users can fully manage narrative sequences — reorder them, assign acts, see duration, and reorder scenes within sequences
**Depends on**: Phase 16
**GitHub Issues**: #24
**Gap Closure**: Closes 4 tech debt items from v1.0 audit
**Success Criteria** (what must be TRUE):
  1. Sequence drag-and-drop reordering updates `sort_order` via API call
  2. Act field UI allows setting/changing act on sequences
  3. Total duration displayed in sequence header (sum of scene durations)
  4. Within-sequence scene reordering calls API and updates UI
**Plans**: 2 plans in 2 waves

Plans:
- [ ] 21-01-PLAN.md — Backend: bulk scene reorder endpoint, SceneListItem type fix, client function
- [ ] 21-02-PLAN.md — Frontend: sequence DnD reorder, act field setter, duration display, within-sequence scene reorder

### Phase 22: Asset Library & Actor-Character Model
**Goal**: Introduce a global Asset Library with standalone Actor, Set, Prop, and Sound Asset entities that can be manually created, browsed, and bound into Production Bibles via a casting/binding system — replacing the current tightly-coupled asset model with a reusable, composable architecture where Actors are persistent identities cast as Characters in specific productions
**Depends on**: Phase 21
**Requirements**: ALIB-01, ALIB-02, ALIB-03, ALIB-04, ALIB-05, ALIB-06, ALIB-07, ALIB-08, ALIB-09
**PRD**: `docs/issues/production-bible-spec.md`
**Success Criteria** (what must be TRUE):
  1. Actor entity exists as a standalone, reusable identity with name, description, appearance refs, voice profiles, wardrobe presets, and prompt tags — independent of any Production Bible
  2. Character entity is a binding of an Actor into a Production Bible role, with character name, arc, wardrobe overrides, voice profile selection, and behavioral notes
  3. Set, Prop, and Sound Asset entities exist as standalone reusable entities in a global Asset Library
  4. Asset Library is a new top-level navigation section with browsable/searchable listings for Actors, Sets, Props, and Sound Assets
  5. Binding system allows associating standalone assets with Production Bibles (CastBinding, SetBinding, PropBinding, SoundBinding) with production-specific overrides
  6. New Production Bible creation view includes Casting, Art Department, and Sound sections with library pickers
  7. Scene prompts support tag syntax ([CHAR:TAG], [SET:TAG], [PROP:TAG]) with autocomplete and tag resolution at generation time
  8. Existing Production Bible assets can be promoted to the standalone Asset Library
  9. Migration path preserves existing data — no breaking changes to current Production Bible workflow
**Plans:** 1/6 plans executed

Plans:
- [ ] 22-01-PLAN.md — ORM models: standalone Actor, LibrarySet, LibraryProp, SoundAsset + binding tables + promotion columns
- [ ] 22-02-PLAN.md — Asset Library CRUD API (actors, sets, props, sounds) + TypeScript types
- [ ] 22-03-PLAN.md — Binding CRUD API + tag resolver service + frontend API client functions
- [ ] 22-04-PLAN.md — Frontend: AssetLibrary view with tabs, ActorLibraryDetail, routing + navigation
- [ ] 22-05-PLAN.md — Frontend: AssetPicker modal, CastingSection, ProductionBibleCreator binding integration
- [ ] 22-06-PLAN.md — Promote-to-library endpoints + tag resolver pipeline wiring
