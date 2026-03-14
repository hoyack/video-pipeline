# Requirements: Viral Video Generation Pipeline (vidpipe)

**Defined:** 2026-02-14
**Core Value:** Accept a text prompt and produce a cohesive, multi-scene short video with visual continuity — fully automated, crash-safe, and resumable.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundation

- [x] **FOUND-01**: Project uses Python 3.11+ with SQLAlchemy 2.0, Pydantic 2.0, and async-first patterns
- [x] **FOUND-02**: SQLite database with WAL mode stores all pipeline state (projects, scenes, keyframes, clips, runs)
- [x] **FOUND-03**: Configuration loaded from config.yaml/.env with typed validation via pydantic-settings
- [x] **FOUND-04**: Local filesystem stores binary artifacts in tmp/{project_id}/ with structured subdirectories

### Storyboard

- [x] **STOR-01**: User submits text prompt and receives structured storyboard with scenes, keyframe prompts, and motion descriptions
- [x] **STOR-02**: Storyboard uses Gemini 3 Pro with JSON schema structured output (responseMimeType: application/json)
- [x] **STOR-03**: Each scene includes scene_description, start_frame_prompt, end_frame_prompt, video_motion_prompt, and transition_notes
- [x] **STOR-04**: Storyboard generates a style guide (visual_style, color_palette, camera_style) for cross-scene consistency
- [x] **STOR-05**: Invalid JSON from LLM is retried up to 3 times with temperature adjustment before failing

### Keyframe Generation

- [x] **KEYF-01**: Start keyframe for scene 0 is generated from start_frame_prompt using Nano Banana Pro
- [x] **KEYF-02**: End keyframe for each scene is generated using start keyframe image + end_frame_prompt (image-conditioned)
- [x] **KEYF-03**: Scene N+1's start keyframe is inherited from scene N's end keyframe (visual continuity)
- [x] **KEYF-04**: Keyframe generation is sequential to maintain continuity across scenes
- [x] **KEYF-05**: Rate limiting with exponential backoff (max 5 retries, configurable delay between calls)
- [x] **KEYF-06**: Keyframe images saved as PNG to tmp/{project_id}/keyframes/

### Video Generation

- [x] **VGEN-01**: Each scene's video clip is generated using Veo 3.1 with first-frame + last-frame interpolation
- [x] **VGEN-02**: Long-running Veo operations are polled with configurable interval (default 15s) and timeout (default ~10min)
- [x] **VGEN-03**: Operation ID is persisted to database before polling begins (idempotent resume)
- [x] **VGEN-04**: RAI-filtered clips are marked as rai_filtered and pipeline continues with remaining scenes
- [x] **VGEN-05**: Timed-out operations are marked as timed_out after max polls exceeded
- [x] **VGEN-06**: Video clips saved as MP4 to tmp/{project_id}/clips/

### Stitching

- [x] **STCH-01**: All completed clips are concatenated into a single MP4 using ffmpeg concat demuxer (hard cuts)
- [x] **STCH-02**: Optional crossfade transitions supported via ffmpeg xfade filter with configurable duration
- [x] **STCH-03**: Audio streams from Veo 3.1 are preserved during concatenation
- [x] **STCH-04**: Final output saved to tmp/{project_id}/output/final.mp4
- [x] **STCH-05**: ffmpeg availability is validated at startup with clear error if missing

### Pipeline Orchestration

- [x] **ORCH-01**: Pipeline follows state machine: STORYBOARD → KEYFRAMES → VIDEO_GEN → STITCH → COMPLETE
- [x] **ORCH-02**: Each step checks database before executing and skips already-completed work (resume capability)
- [x] **ORCH-03**: Pipeline run metadata tracked (start time, duration, cost estimate, step log)
- [x] **ORCH-04**: Failed pipeline can be resumed from last completed step via resume command

### CLI Interface

- [x] **CLI-01**: User can generate video from prompt via `python -m vidpipe generate "prompt"` with style, aspect-ratio, clip-duration options
- [x] **CLI-02**: User can resume a failed/incomplete project via `python -m vidpipe resume <project_id>`
- [x] **CLI-03**: User can check project status via `python -m vidpipe status <project_id>`
- [x] **CLI-04**: User can list all projects via `python -m vidpipe list`
- [x] **CLI-05**: User can re-stitch with crossfade via `python -m vidpipe stitch <project_id> --crossfade 0.5`

### HTTP API

- [x] **API-01**: POST /api/generate starts new pipeline run in background and returns project_id immediately
- [x] **API-02**: GET /api/projects/{id}/status returns lightweight status for polling
- [x] **API-03**: GET /api/projects/{id} returns full project detail with scenes and clips
- [x] **API-04**: GET /api/projects lists all projects
- [x] **API-05**: POST /api/projects/{id}/resume resumes a failed pipeline
- [x] **API-06**: GET /api/projects/{id}/download serves final MP4 file
- [x] **API-07**: GET /api/health returns health check

## LLM Provider Abstraction

- [x] **LLMA-01**: Abstract base `LLMAdapter` class with `generate_text(prompt, schema, temperature, retries)` and `analyze_image(image_bytes, prompt, schema, temperature)` methods
- [x] **LLMA-02**: `VertexAIAdapter` wraps existing Gemini calls — storyboard, prompt rewriting, reverse-prompting, CV semantic analysis, candidate scoring all route through it
- [x] **LLMA-03**: `OllamaAdapter` implements text generation (with JSON mode for structured output) and vision analysis using Ollama REST API (`/api/generate`, `/api/chat`)
- [x] **LLMA-04**: Settings UI Ollama section: API key input, cloud vs local toggle (local hides API key, uses `localhost:11434`; cloud uses `ollama.com`), custom endpoint URL override
- [x] **LLMA-05**: Model management: input box to add custom Ollama model names, toggle models on/off in settings, remove models from list entirely; added models appear in GenerateForm dropdowns
- [x] **LLMA-06**: GenerateForm supports text_model and vision_model selection; storyboard/prompt-rewriting uses text_model adapter; reverse-prompting/CV-analysis/candidate-scoring uses vision_model adapter
- [x] **LLMA-07**: Provider routing: model ID → adapter mapping via provider registry; Gemini models → VertexAIAdapter, Ollama models → OllamaAdapter; future providers (Anthropic, OpenAI, Grok) can register without modifying core pipeline

## Storyboard Asset Binding

- [x] **SBIND-01**: Storyboard prompt mandates using existing CHARACTER tags from asset registry; `new_asset_declarations` restricted to non-CHARACTER types only
- [x] **SBIND-02**: Post-storyboard deterministic remapping catches any LLM-invented CHARACTER tags and maps them to existing manifest CHARACTER assets
- [x] **SBIND-03**: Prompt rewriter falls back to marking ALL manifest CHARACTER assets as MUST SELECT when scene manifest placements reference non-existent tags
- [x] **SBIND-04**: Keyframe enforcement falls back to all manifest CHARACTER assets with reference images when `placed_char_tags` resolves empty

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

## Production Bible Foundation

- [x] **PBIB-01**: `production_bibles` table exists with all data migrated from `manifests`; all FK columns renamed to `production_bible_id`
- [x] **PBIB-02**: All API endpoints respond at `/api/production-bibles/*` with 301 redirects from legacy `/api/manifests/*` paths
- [x] **PBIB-03**: Frontend uses "Production Bible" terminology everywhere; routes updated to `/production-bibles/*`
- [x] **PBIB-04**: Production Bible detail view has three department tabs: Casting, Art Department, Sound — with existing assets sorted into correct tabs
- [x] **PBIB-05**: `sequences` table stores optional grouping layer with title, description, order, act, and color fields
- [x] **PBIB-06**: Scene model has optional `sequence_id` FK; scenes with null sequence_id remain in flat list

## Sequences

- [x] **SEQ-01**: Sequence CRUD API under `/api/productions/{id}/sequences` with create, list, update, delete endpoints
- [x] **SEQ-02**: Sequence drag-and-drop reorder support via PATCH endpoint
- [x] **SEQ-03**: Frontend renders scenes grouped by sequence when sequences exist, with collapsible sections
- [x] **SEQ-04**: Scenes can be assigned to or moved between sequences

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
- [x] **SCRN-04**: Screenplay editor UI with tabs: Logline, Treatment, Scene Breakdown, Script, Shot List — each editable with independent Regenerate button
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
- [x] **SCRN-15**: Scenes generated from Screenplay show "Screenplay linked" badge in UI

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
| LLMA-01 | Phase 13 | Complete |
| LLMA-02 | Phase 13 | Complete |
| LLMA-03 | Phase 13 | Complete |
| LLMA-04 | Phase 13 | Complete |
| LLMA-05 | Phase 13 | Complete |
| LLMA-06 | Phase 13 | Complete |
| LLMA-07 | Phase 13 | Complete |
| SBIND-01 | Phase 14 | Complete |
| SBIND-02 | Phase 14 | Complete |
| SBIND-03 | Phase 14 | Complete |
| SBIND-04 | Phase 14 | Complete |
| VGED-01 | Phase 15 | Complete |
| VGED-02 | Phase 15 | Complete |
| VGED-03 | Phase 15 | Complete |
| VGED-04 | Phase 15 | Complete |
| VGED-05 | Phase 15 | Complete |
| VGED-06 | Phase 15 | Complete |
| VGED-07 | Phase 15 | Complete |
| VGED-08 | Phase 15 | Complete |
| VGED-09 | Phase 15 | Complete |
| VGED-10 | Phase 15 | Complete |
| VGED-11 | Phase 15 | Complete |
| VGED-12 | Phase 15 | Complete |
| PBIB-01 | Phase 16 | Complete |
| PBIB-02 | Phase 16 | Complete |
| PBIB-03 | Phase 16 | Complete |
| PBIB-04 | Phase 16 | Complete |
| PBIB-05 | Phase 16 | Complete |
| PBIB-06 | Phase 16 | Complete |
| SEQ-01 | Phase 16 | Complete |
| SEQ-02 | Phase 16 | Complete |
| SEQ-03 | Phase 16 | Complete |
| SEQ-04 | Phase 16 | Complete |
| PBEX-01 | Phase 17 | Complete |
| PBEX-02 | Phase 17 | Complete |
| PBEX-03 | Phase 17 | Complete |
| PBEX-04 | Phase 17 | Complete |
| PBEX-05 | Phase 17 | Complete |
| PBEX-06 | Phase 17 | Complete |
| PBEX-07 | Phase 17 | Complete |
| PBEX-08 | Phase 17 | Complete |
| PBEX-09 | Phase 17 | Complete |
| PBEX-10 | Phase 17 | Complete |
| PBEX-11 | Phase 17 | Complete |
| PBEX-12 | Phase 17 | Complete |
| PBEX-13 | Phase 17 | Complete |
| PBEX-14 | Phase 17 | Complete |
| PBEX-15 | Phase 17 | Complete |
| PBEX-16 | Phase 17 | Complete |
| PBEX-17 | Phase 17 | Complete |
| PBEX-18 | Phase 17 | Complete |
| PBEX-19 | Phase 17 | Complete |
| PBEX-20 | Phase 17 | Complete |
| SCRN-01 | Phase 18 | Complete |
| SCRN-02 | Phase 18 | Complete |
| SCRN-03 | Phase 18 | Complete |
| SCRN-04 | Phase 18 | Complete |
| SCRN-05 | Phase 18 | Complete |
| SCRN-06 | Phase 18 | Complete |
| SCRN-07 | Phase 18 | Complete |
| SCRN-08 | Phase 18 | Complete |
| SCRN-09 | Phase 18 | Complete |
| SCRN-10 | Phase 18 | Complete |
| SCRN-11 | Phase 18 | Complete |
| SCRN-12 | Phase 18 | Complete |
| SCRN-13 | Phase 18 | Complete |
| SCRN-14 | Phase 18 | Complete |
| SCRN-15 | Phase 18 | Complete |
| ALIB-01 | Phase 22 | In Progress |
| ALIB-02 | Phase 22 | In Progress |
| ALIB-03 | Phase 22 | In Progress |
| ALIB-04 | Phase 22 | In Progress |
| ALIB-05 | Phase 22 | In Progress |
| ALIB-06 | Phase 22 | In Progress |
| ALIB-07 | Phase 22 | In Progress |
| ALIB-08 | Phase 22 | In Progress |
| ALIB-09 | Phase 22 | In Progress |

## Asset Library & Actor-Character Model

- [x] **ALIB-01**: Actor entity exists as standalone identity with name, description, base_appearance_prompt, prompt_tags, appearance_refs (ActorRef[]), voice_profiles (ActorVoiceProfile[]), wardrobe_presets (ActorWardrobePreset[])
- [x] **ALIB-02**: Character is a CastBinding of an Actor into a Production Bible role with character_name, character_arc, role (LEAD/SUPPORTING/EXTRA/NARRATOR), wardrobe_override, voice_profile_id, behavioral_notes, prompt_tags
- [x] **ALIB-03**: Standalone LibrarySet, LibraryProp, and SoundAsset entities exist in global Asset Library independent of Production Bibles
- [x] **ALIB-04**: Asset Library is a new top-level navigation section with browsable/searchable listings for Actors, Sets, Props, and Sound Assets
- [x] **ALIB-05**: Binding system (CastBinding, SetBinding, PropBinding, SoundBinding) connects library assets to Production Bibles with production-specific overrides
- [x] **ALIB-06**: Production Bible creation view includes Casting, Art Department, and Sound sections with library pickers for adding bound assets
- [x] **ALIB-07**: Scene prompts support tag syntax ([CHAR:TAG], [SET:TAG], [PROP:TAG]) with tag resolution at generation time via binding lookup
- [x] **ALIB-08**: Existing bible-scoped Characters, Sets, Props can be promoted to standalone Asset Library entities with auto-created bindings back
- [x] **ALIB-09**: Migration preserves existing data; promoted_to columns track promotion state; existing scenes without tags continue to work

## Tag Syntax & Binding Pipeline Wiring

- [x] **ATAG-01**: Tag resolver supports @tag pattern alongside existing [TYPE:TAG] syntax with cross-type lookup (CastBinding → PropBinding → SetBinding)
- [x] **ATAG-02**: ResolvedAssetRef dataclass carries structured asset metadata (tag, type, description, reference_image_urls, lora_url, wardrobe_override, lighting_notes) for image generation
- [x] **ATAG-03**: resolve_tags_with_assets() function loads asset data from binding tables (CastBinding, SetBinding, PropBinding) via production_bible_id
- [x] **ATAG-04**: format_binding_registry() formats all bound assets from a Production Bible for LLM context injection in storyboard pipeline
- [x] **ATAG-05**: Storyboard pipeline uses format_binding_registry() when scene has production_bible_id with bindings
- [x] **ATAG-06**: GET /api/production-bibles/{id}/bound-assets/summary returns flat list of all bindings with tags, names, types, and primary thumbnails
- [x] **ATAG-07**: Frontend BoundAssetSummary TypeScript type and getBoundAssetsSummary() API client function

## ComfyUI Flux.1 Workflows

- [x] **FLUX-01**: Flux.1 Dev base text-to-image ComfyUI workflow template (flux_txt2img_base.json)
- [x] **FLUX-02**: Flux.1 Dev + dynamic LoRA loader workflow template (flux_txt2img_with_lora.json)
- [x] **FLUX-03**: Flux.1 Dev + UNO/Redux reference injection workflow template for up to 3 reference images (flux_txt2img_with_references.json)
- [x] **FLUX-04**: Full hybrid Flux.1 Dev + LoRA + UNO workflow template (flux_txt2img_full.json)
- [x] **FLUX-05**: build_flux_txt2img_workflow() builder function in comfyui_client.py that dynamically selects template based on available LoRA and reference images
- [x] **FLUX-06**: Flux model IDs (flux-dev, flux-dev-lora, flux-dev-redux, flux-dev-full) added to COMFYUI_IMAGE_MODELS with routing in keyframe pipeline
- [x] **FLUX-07**: Binding-based reference resolution path in keyframes.py categorizes ResolvedAssetRefs by type (CHARACTER → LoRA, PROP/SET → reference images)
- [x] **FLUX-08**: Frontend Flux model options added to IMAGE_MODELS catalog in constants.ts

## LoRA Training Infrastructure

- [x] **LORA-01**: Actor model extended with lora_url (S3 path to .safetensors), lora_trained_at (datetime), lora_training_status (QUEUED/TRAINING/COMPLETED/FAILED)
- [x] **LORA-02**: lora_trainer.py service with dataset preparation (download refs, resize, caption via VLM), pluggable training backend interface, and job dispatch
- [x] **LORA-03**: POST /api/asset-library/actors/{id}/train-lora endpoint validates minimum reference images and dispatches training job
- [x] **LORA-04**: GET /api/asset-library/actors/{id}/lora-status endpoint returns training status, progress, and LoRA URL when complete
- [x] **LORA-05**: Frontend "Train Identity Model" button (enabled when refs >= 5) and status badge (No Model / Training / Model Ready) on Actor detail view

## Asset Tag Frontend Enhancements

- [x] **ATED-01**: CodeMirror @tag autocomplete extension shows dropdown of bound assets when user types @ in scene editor
- [x] **ATED-02**: Tag preview panel in scene editor shows asset reference image, name, and description on hover/click of @tag
- [x] **ATED-03**: Actor detail view shows LoRA training status with Train/Regenerate buttons and training date
- [x] **ATED-04**: Production Bible "Tag Reference Sheet" tab lists all bound assets with @tag syntax, type, and thumbnail

**Coverage:
- v1 requirements: 41 total (all complete)
- LLM Provider Abstraction: 7 total (all complete)
- Storyboard Asset Binding: 4 total (all complete)
- Video Generation Editor: 12 total (all complete)
- Production Bible Foundation: 6 total (all complete)
- Sequences: 4 total (all complete)
- Production Bible Entity Expansion: 20 total (all complete)
- Screenplay System: 15 total (all complete)
- Asset Library & Actor-Character Model: 9 total (complete)
- Tag Syntax & Binding Pipeline Wiring: 7 total (planned)
- ComfyUI Flux.1 Workflows: 8 total (planned)
- LoRA Training Infrastructure: 5 total (planned)
- Asset Tag Frontend Enhancements: 4 total (planned)
- **Total mapped: 142 requirements across 23 phases**
- Unmapped: 0

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-03-14 after Phase 23-26 planning*
