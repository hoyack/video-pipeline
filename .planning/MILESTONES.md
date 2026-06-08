# Milestones

## v1.0 milestone (Shipped: 2026-06-08)

**Phases completed:** 27 phases, 73 plans, 125 tasks

**Close note:** auto-closed with 9 acknowledged deferred items: 5 audit gaps and 4 storyboard performance requirements. See `.planning/STATE.md` Deferred Items.

**Key accomplishments:**

- Python package with SQLAlchemy 2.0 models for all 5 pipeline entities using Mapped annotations and ForeignKey relationships
- Type-safe configuration loading from YAML with environment variable overrides using pydantic-settings and custom YamlConfigSettingsSource
- Crash-safe async SQLite engine with WAL mode, PRAGMA configuration, and structured filesystem artifact storage with path traversal protection.
- Gemini-powered storyboard generation with structured JSON output, scene-by-scene breakdowns, and automatic retry with temperature adjustment for parse errors
- Sequential keyframe generation with visual continuity via frame inheritance and image-conditioned Nano Banana Pro
- Veo 3.1 video generation with first/last frame interpolation, long-running operation polling, and graceful RAI filtering
- ffmpeg-based video stitching with concat demuxer for hard cuts, xfade filter for crossfade transitions, and startup validation ensuring ffmpeg availability
- Implemented state machine orchestrator with idempotent resume, PipelineRun metadata tracking, and progress callback interface.
- Implemented full Typer CLI with 5 commands (generate, resume, status, list, stitch) using Rich formatting and async database operations.
- Implemented FastAPI HTTP API with async request-reply pattern, 7 RESTful endpoints, background task execution, and MP4 file downloads.
- Complete backend for Manifest System with Manifest and Asset ORM models, business logic service layer, and 11 REST API endpoints for full CRUD plus image upload.
- Filterable/sortable card-grid library view for manifests with CRUD actions (edit, duplicate, delete) integrated into top navigation.
- Extended Asset model with CV fields and created YOLO detection, ArcFace face matching, and Gemini reverse-prompting services with lazy loading
- Built ManifestingEngine orchestrator composing CV/AI services into complete pipeline, plus background task runner with progress tracking and API endpoints for triggering and monitoring processing
- ManifestSnapshot model with immutable state capture, usage tracking, and enhanced generate endpoint with optional manifest selection.
- ManifestSelector component with radio toggle, compact ManifestCard mode, and GenerateForm integration for selecting pre-built manifests during video generation.
- Pydantic schemas for asset placement and audio direction with ORM models using composite PKs on (project_id, scene_index)
- Manifest-aware storyboard generation with asset registry injection and structured scene/audio manifest persistence
- AssetCleanReference ORM model, SceneManifest.selected_reference_tags column, and scene-type-aware reference selection service that prioritizes up to 3 assets per scene based on shot composition.
- Video generation pipeline passes up to 3 reference images to Veo 3.1 for identity preservation, with clean sheet generation (Tier 2 rembg, Tier 3 Gemini Image) for reference quality optimization.
- SceneDetail API response includes selected_references array with asset metadata, SceneCard component displays reference badges showing thumbnail, manifest_tag, and quality score for each identity reference used in Veo generation.
- Post-generation CV analysis pipeline composing YOLO + ArcFace + CLIP + Gemini Vision into CVAnalysisService, with entity extraction and quality-gated asset registration.
- Per-scene CV analysis (YOLO + ArcFace + CLIP + Gemini) wired into video_gen loop with progressive asset enrichment — each scene's extracted assets feed into the next scene's reference selection
- PromptRewriterService with Gemini 2.5 Flash structured output, assembling 5 context inputs (original prompt, manifest composition, placed asset reverse_prompts, CV continuity patch, audio direction) into cinematography-formula prompts with LLM-reasoned reference selection
- PromptRewriterService wired into keyframes.py and video_gen.py — manifest projects now use LLM-rewritten cinematography prompts with asset injection, continuity corrections, and audio direction before Imagen and Veo submission.
- GenerationCandidate ORM model with four-dimension composite scoring via CVAnalysisService, CLIP embeddings, and batched Gemini Flash call
- Multi-candidate Veo generation wired into video_gen.py: number_of_videos passed to API, all candidates saved and scored via CandidateScoringService, winner selected by composite_score, VideoClip.local_path updated for stitcher compatibility
- Two candidate REST endpoints, Quality Mode toggle in GenerateForm with cost multiplier display, and scored candidate comparison grid in SceneCard with click-to-override selection
- Asset inheritance schema (3 new ORM columns + SQL migration) and extended fork API payload with AssetChanges, ModifiedAsset, NewUpload Pydantic models plus manifest_id in ProjectDetail
- Fork endpoint extended with asset copy (is_inherited, shared GCS URLs), scene manifest inheritance, asset-modification-driven invalidation boundary tightening, and ManifestingEngine.process_new_uploads for incremental YOLO+face+reverse-prompt processing of new uploads
- EditForkPanel extended with Asset Registry section: inherited assets shown with lock/edit/remove controls, inline reverse_prompt editing, and base64 file picker for new reference uploads — all serialized into ForkRequest.asset_changes on submit
- LLM adapter ABC with VertexAI + Ollama implementations, provider registry, vision Pydantic schemas, and DB/API extensions for Ollama configuration.
- All LLM call sites migrated from direct Gemini SDK to LLMAdapter interface, orchestrator wired to create and pass adapters from project config, enabling Ollama models throughout the pipeline.
- Ollama Settings section with cloud/local toggle, model management (add/toggle vision/enable/remove), and GenerateForm vision_model dropdown that merges Gemini and Ollama models.
- Four-layer defense-in-depth fix preventing storyboard LLM from inventing CHARACTER tags, with prompt hardening, deterministic remapping, rewriter fallback, and keyframe enforcement
- Draft project creation with empty scenes, gap-filling generate endpoint, final-video upload, and Scene.generation_status tracking column
- All four pipeline stages (storyboard, keyframes, video_gen, stitcher) modified for gap-filling mode with per-scene generation_status tracking and per-scene stop flag checking
- Unified VideoGenEditor replacing GenerateForm + ProgressView with drafting/running/editing modes, real-time polling, GenerateThroughSlider, and ProjectConfigBar
- 1. [Rule 1 - Bug] Fixed remaining manifest_id column refs in checkpoint_service.py
- SQLAlchemy Sequence model with production/scene FKs and 7-endpoint CRUD API for narrative chapter organization above the Scene layer
- 6 new React components and 7 API client functions enabling drag-and-drop sequence grouping of scenes within ProductionDetail using @dnd-kit
- 8 ORM models for characters, sets, props, and audio entities with 1:1 sub-entity enforcement and Scene.score_theme_id FK migration
- 29 CRUD endpoints for Character/Wardrobe/VoiceProfile and Set/SonicIdentity/Prop with prompt-context injection and LLM Vision reverse-prompting on set reference uploads
- ScoreTheme + SFXItem CRUD (10 endpoints) with category filter, plus idempotent asset-to-entity migration service converting CHARACTER/ENVIRONMENT assets to Character/Set entities
- Full CRUD entity editors for Characters (4-tab), Sets/Props (dual-view), and Sound (Score Themes + SFX with category filters) wired into Production Bible department tabs
- Screenplay ORM model with 6-step LLM generation chain (logline through script), incremental commits, LOCKED-status guard, and Production Bible entity validation
- 11-endpoint Screenplay API with CRUD, per-step generation, scene creation from breakdown, and conditional storyboard prompt enrichment from screenplay context
- 6-tab ScreenplayEditor with per-tab regeneration, status controls, Generate Scenes action, and Screenplay badge on production scenes
- Direct Production.production_bible_id FK fixes load_bible_context for bible-first screenplay flow, with API schema updates and sound_router cleanup
- 7 file upload endpoints across 3 route files: actor refs, wardrobe refs, generate-appearance, standalone reverse-prompt, and audio uploads for ScoreTheme/SFXItem/SonicIdentity
- Reusable AudioPlayer component, 7 upload client functions, and upload UI wired into CharacterDetail (actor refs + wardrobe), SetDetail (prop + sonic identity audio), and SoundDepartment (score theme + SFX audio)
- Bulk scene reorder endpoint (PUT /api/sequences/{id}/scenes/reorder) with SceneListItem.scene_order field and reorderScenesInSequence client function
- Sequence and scene drag-and-drop reorder, act field setter submenu, and duration badge display in SequenceHeader
- 14 ORM models for Asset Library standalone entities (Actor, LibrarySet, LibraryProp, SoundAsset) with binding tables and promotion tracking on existing Phase 17 entities
- 33-route FastAPI CRUD API for Actors (with refs/voice/wardrobe), LibrarySets, LibraryProps, and SoundAssets with dual-backend file uploads and complete TypeScript type contracts
- 17-endpoint binding CRUD API with tag resolver for [CHAR/SET/PROP:TAG] prompt injection and full frontend API client for asset library operations
- AssetLibrary with 4 tabbed entity listings (Actors, Sets, Props, Sound Assets) and ActorLibraryDetail with 5-tab CRUD interface, wired into top-level navigation
- AssetPicker modal for browsing library assets, CastingSection with add/edit/remove cast bindings, and Art Dept/Sound binding sections integrated into ProductionBibleCreator tabs
- 5 promote endpoints converting bible entities to library entities with auto-bindings, plus tag resolver wired into storyboard pipeline for generation-time prompt enrichment
- Dual-regex tag resolution with @tag cross-type binding lookup, ResolvedAssetRef structured metadata, and batch-loaded resolve_tags_with_assets()
- format_binding_registry() for LLM @tag context injection, storyboard pipeline wiring with legacy fallback, bound-assets summary API endpoint, and frontend TypeScript types
- Four Flux.1 Dev ComfyUI workflow templates with dynamic builder function using unCLIPConditioning for reference injection and LoraLoaderModelOnly for identity LoRA
- Flux.1 Dev models wired into all keyframe generation paths with binding-based LoRA/reference categorization and frontend model selection
- LoRA training data model, Replicate backend with async SDK wrapping, and dataset preparation pipeline (resize + VLM caption + zip)
- LoRA training dispatch/polling API endpoints with Train Identity Model button, status badges (No Model/Training/Ready/Failed), and React status polling in ActorLibraryDetail
- CodeMirror @tag autocomplete with hover tooltip and side preview panel for bound Production Bible assets in the scene editor
- Tag Reference Sheet tab in Production Bible showing all @tags with type badges, thumbnails, and filter input; ATED-03 (LoRA training status) verified pre-satisfied from Phase 25
- CodeMirror click handler extension for @tag preview panel using EditorView.domEventHandlers, closing the ATED-02 hover/click gap

---
