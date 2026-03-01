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

## LLM Provider Abstraction

- [ ] **LLMA-01**: Abstract base `LLMAdapter` class with `generate_text(prompt, schema, temperature, retries)` and `analyze_image(image_bytes, prompt, schema, temperature)` methods
- [ ] **LLMA-02**: `VertexAIAdapter` wraps existing Gemini calls — storyboard, prompt rewriting, reverse-prompting, CV semantic analysis, candidate scoring all route through it
- [ ] **LLMA-03**: `OllamaAdapter` implements text generation (with JSON mode for structured output) and vision analysis using Ollama REST API (`/api/generate`, `/api/chat`)
- [ ] **LLMA-04**: Settings UI Ollama section: API key input, cloud vs local toggle (local hides API key, uses `localhost:11434`; cloud uses `ollama.com`), custom endpoint URL override
- [ ] **LLMA-05**: Model management: input box to add custom Ollama model names, toggle models on/off in settings, remove models from list entirely; added models appear in GenerateForm dropdowns
- [ ] **LLMA-06**: GenerateForm supports text_model and vision_model selection; storyboard/prompt-rewriting uses text_model adapter; reverse-prompting/CV-analysis/candidate-scoring uses vision_model adapter
- [ ] **LLMA-07**: Provider routing: model ID → adapter mapping via provider registry; Gemini models → VertexAIAdapter, Ollama models → OllamaAdapter; future providers (Anthropic, OpenAI, Grok) can register without modifying core pipeline

## Video Generation Editor

- [x] **VGED-01**: `POST /api/projects` creates draft project with empty Scene rows without starting the pipeline
- [x] **VGED-02**: `POST /api/projects/{id}/generate` inspects existing assets and only generates what's missing (gap-filling mode)
- [x] **VGED-03**: `"draft"` added to project status enum; draft projects appear in list with Draft badge
- [x] **VGED-04**: Scene-level upload endpoints: start-keyframe, end-keyframe, clip, final-video; each validates file type and creates/updates DB rows
- [x] **VGED-05**: `generation_status` column on Scene model tracks per-scene pipeline progress (generating_text, generating_start_kf, generating_end_kf, generating_clip, complete, failed)
- [x] **VGED-06**: Pipeline stages (storyboard, keyframes, video_gen, stitcher) skip scenes/assets that already exist (gap-filling)
- [x] **VGED-07**: `VideoGenEditor` component replaces GenerateForm + ProgressView with unified create/edit/monitor experience
- [x] **VGED-08**: Generate Through slider controls pipeline stop point (storyboard, keyframes, video, stitch, all)
- [x] **VGED-09**: Scene cards render through full lifecycle: empty (dashed) → partially filled → generating (spinners per asset) → complete
- [x] **VGED-10**: Real-time asset population via polling with visual feedback (highlight flash on new assets)
- [x] **VGED-11**: Pause/resume at per-scene granularity; stop flag checked between individual scene operations
- [x] **VGED-12**: App.tsx navigation merges "generate" + "progress" views into single "editor" view

## Production Bible Entity Expansion

- [x] **PBEX-01**: Character entity with full schema: name, role (PROTAGONIST/ANTAGONIST/SUPPORTING/EXTRA), description, arc, actor_refs images, base_appearance, wardrobe items, voice_profile, prompt_tags
- [x] **PBEX-02**: Wardrobe sub-entity per character: label, reference_images, scene_context, prompt_descriptor, is_default toggle
- [x] **PBEX-03**: VoiceProfile sub-entity per character: voice_id, adapter_type (ELEVENLABS), style_notes, sample_audio
- [x] **PBEX-04**: Character CRUD API under `/api/production-bibles/:id/characters` + `/api/characters/:id` with prompt-context endpoint
- [x] **PBEX-05**: Character detail UI with four tabs: Overview, Actor References, Wardrobe, Voice Profile in Casting tab
- [x] **PBEX-06**: Existing manifest character assets migrated to Character entities on first load
- [x] **PBEX-07**: Set entity with full schema: name, reference_image, reverse_prompt, style_tags, lighting_notes, prompt_tags, sonic_identity
- [x] **PBEX-08**: SonicIdentity sub-entity per set: ambience_description, reference_audio, generation_prompt
- [x] **PBEX-09**: LLM Vision reverse-prompt auto-generation for Sets on reference image upload
- [x] **PBEX-10**: Set CRUD API under `/api/production-bibles/:id/sets` + `/api/sets/:id` with prompt-context endpoint
- [x] **PBEX-11**: Set detail UI with Visual and Sonic Identity tabs in Art Department tab
- [x] **PBEX-12**: Existing background/scene assets migrated to Set entities
- [x] **PBEX-13**: Prop entity: name, reference_image, description, associated_characters, prompt_tags under Art Department tab
- [x] **PBEX-14**: Prop CRUD API under `/api/production-bibles/:id/props` + `/api/props/:id`
- [x] **PBEX-15**: Prop list/detail UI in Art Department tab with thumbnail grid
- [x] **PBEX-16**: ScoreTheme entity: name, mood_descriptors, tempo_notes, usage_notes, reference_audio, generation_prompt, adapter_type
- [x] **PBEX-17**: SFXItem entity: name, category (IMPACT/MECHANICAL/NATURAL/UI/FOLEY/AMBIENCE), source_audio, generation_prompt, tags
- [x] **PBEX-18**: ScoreTheme and SFXItem CRUD API under `/api/production-bibles/:id/score-themes` and `/api/production-bibles/:id/sfx`
- [x] **PBEX-19**: Sound Department tab UI with Score Themes and SFX Library sections with category filters
- [x] **PBEX-20**: Scene.score_theme_id nullable FK for forward compatibility with Director agent

## Screenplay System

- [x] **SCRN-01**: Screenplay entity attached one-to-one to Production with title, genre, status (DRAFT/IN_REVIEW/LOCKED), logline, treatment, character_breakdowns, scene_breakdown, script, shot_list
- [x] **SCRN-02**: Scene Breakdown sub-structure per scene: scene_number, slugline, intent, emotional_beat, story_state_in, story_state_out, characters_present (Character refs), set_ref (Set ref), props_required (Prop refs)
- [x] **SCRN-03**: Screenplay CRUD API under `/api/productions/:id/screenplay` with per-component update endpoints
- [ ] **SCRN-04**: Screenplay editor UI with tabs: Logline, Treatment, Scene Breakdown, Script, Shot List — each editable with independent Regenerate button
- [x] **SCRN-05**: Screenplay status field (DRAFT/IN_REVIEW/LOCKED); LOCKED prevents regeneration
- [x] **SCRN-06**: Scene Breakdown entries link to Production Bible Characters, Sets, and Props
- [x] **SCRN-07**: Screenwriter agent with sequential generation chain: logline → treatment → character_breakdowns → scene_breakdown → script (uses existing LLM adapter, not CrewAI)
- [x] **SCRN-08**: Each Screenwriter generation step updates Screenplay entity incrementally (user sees progress)
- [x] **SCRN-09**: Each Screenwriter step can be run independently (e.g. regenerate only Script without changing Breakdown)
- [x] **SCRN-10**: Production Bible Characters and Sets injected as context into Screenwriter generation prompts
- [x] **SCRN-11**: LLM adapter selectable per Production for Screenwriter agent
- [x] **SCRN-12**: "Generate Scenes from Screenplay" action creates one Scene per SceneBreakdown entry from a locked Screenplay
- [x] **SCRN-13**: Scene description populated from SceneBreakdown.intent; Shot prompts include Character, Set, Prop prompt_tags from linked breakdown
- [x] **SCRN-14**: Free-form storyboard generation remains as fallback when no Screenplay exists
- [ ] **SCRN-15**: Scenes generated from Screenplay show "Screenplay linked" badge in UI

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
| LLMA-01 | Phase 13 | Planned |
| LLMA-02 | Phase 13 | Planned |
| LLMA-03 | Phase 13 | Planned |
| LLMA-04 | Phase 13 | Planned |
| LLMA-05 | Phase 13 | Planned |
| LLMA-06 | Phase 13 | Planned |
| LLMA-07 | Phase 13 | Planned |
| VGED-01 | Phase 15 | Planned |
| VGED-02 | Phase 15 | Planned |
| VGED-03 | Phase 15 | Planned |
| VGED-04 | Phase 15 | Planned |
| VGED-05 | Phase 15 | Planned |
| VGED-06 | Phase 15 | Planned |
| VGED-07 | Phase 15 | Planned |
| VGED-08 | Phase 15 | Planned |
| VGED-09 | Phase 15 | Planned |
| VGED-10 | Phase 15 | Planned |
| VGED-11 | Phase 15 | Planned |
| VGED-12 | Phase 15 | Planned |

**Coverage:**
- v1 requirements: 41 total (all complete)
- v3 requirements (LLM abstraction): 7 total
- Video Generation Editor: 12 total
- Mapped to phases: 60
- Unmapped: 0

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-02-21 after adding Video Generation Editor requirements*
