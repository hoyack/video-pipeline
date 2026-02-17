# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** Accept a text prompt and produce a cohesive, multi-scene short video with visual continuity — fully automated, crash-safe, and resumable.
**Current focus:** Phase 5: Manifesting Engine (V2 pipeline evolution)

## Current Position

Phase: 11 of 12 (Multi-Candidate Quality Mode) — IN PROGRESS
Plan: 1 of 3 complete
Status: 11-01 complete — GenerationCandidate model, CandidateScoringService, Project quality columns
Last activity: 2026-02-16 — Completed 11-01 (data layer and scoring engine)

Progress: [████████░░] 89% (10 of 12 phases complete, 27 of 29 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 26
- Average duration: 2.5 min
- Total execution time: 1.08 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 3 | 7.0 min | 2.3 min |
| 02-generation-pipeline | 4 | 8.0 min | 2.0 min |
| 03-orchestration-interfaces | 3 | 6.0 min | 2.0 min |
| 04-manifest-system-foundation | 2 | 9.2 min | 4.6 min |
| 05-manifesting-engine | 2 | 9.1 min | 4.6 min |
| 06-generateform-integration | 2 | 5.0 min | 2.5 min |
| 07-manifest-aware-storyboarding | 2 | 5.1 min | 2.6 min |
| 08-veo-reference-passthrough | 2 | 7.1 min | 3.6 min |
| 09-cv-analysis-pipeline | 3 | 10.0 min | 3.3 min |
| 10-adaptive-prompt-rewriting | 2 | 4.0 min | 2.0 min |
| 11-multi-candidate-quality-mode | 1/3 | 3.0 min | 3.0 min |

**Recent Trend:**
- Last 5 plans: 09-03 (2.0min), 10-01 (2.0min), 10-02 (2.0min), 11-01 (3.0min)
- Trend: Phase 11 started — data layer (GenerationCandidate model + CandidateScoringService) complete in 3 min

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Phase 1: Gemini structured output over CrewAI for simpler v1 implementation
- Phase 1: SQLite over cloud DB for local-first tool with no multi-user needs
- Phase 2: Sequential keyframes required for visual continuity across scenes
- Phase 2: Vertex AI over Gemini API for consistent ADC auth
- **01-01:** Used SQLAlchemy 2.0 Mapped[Type] annotations for type safety
- **01-01:** Defined foreign key relationships at database level for referential integrity
- **01-01:** Created modular package structure with db/, services/, pipeline/, schemas/ subdirectories
- **01-02:** Used YamlConfigSettingsSource custom source for YAML loading instead of dotenv approach
- **01-02:** Nested config models inherit from BaseModel (not BaseSettings) per pydantic-settings best practices
- **01-02:** Environment variables use __ delimiter for nested config (VIDPIPE_PIPELINE__MAX_SCENES)
- **01-02:** Hardcoded config.yaml path in YamlConfigSettingsSource for simplicity
- **01-03:** Use engine.sync_engine for event listener to support aiosqlite wrapper
- **01-03:** Set expire_on_commit=False to prevent greenlet errors in async context
- **01-03:** Use synchronous=FULL for maximum crash safety per FOUND-04 requirement
- **01-03:** Implement path traversal protection using is_relative_to() method
- **01-03:** Use metadata.create_all() instead of Alembic for v1 simplicity
- **02-01:** Used google-genai SDK in Vertex AI mode with ADC for unified authentication
- **02-01:** Implemented tenacity retry with temperature reduction (0.7 → 0.55 → 0.4) on JSON failures
- **02-01:** Corrected model names: gemini-2.0-flash-exp, imagen-3.0-generate-001, veo-2.0-generate-001
- **02-01:** Applied singleton pattern to vertex_client to avoid repeated client initialization
- **02-02:** Used image-conditioned generation for end frames to maintain visual style and composition
- **02-02:** Commit after each scene (not at end) for crash recovery and resumability
- **02-02:** Applied jitter to retry backoff to prevent thundering herd on rate limit errors
- **02-02:** Scene 0 start frame from text alone, all other start frames inherited from previous end frame
- **02-03:** Persist operation_name to database BEFORE polling starts for crash recovery
- **02-03:** Use async sleep in polling loop to avoid blocking event loop
- **02-03:** Mark RAI-filtered clips and continue pipeline rather than crashing
- **02-03:** Resume polling from clip.poll_count for idempotent crash recovery
- **02-04:** Used concat demuxer with -safe 0 flag for absolute path support in concat list
- **02-04:** Stream copy (-c copy) for concat demuxer to preserve audio quality without re-encoding
- **02-04:** Wrapped subprocess.run() in asyncio.to_thread() to prevent event loop blocking
- **02-04:** Validate ffmpeg at startup rather than during pipeline execution for fail-fast error handling
- [Phase 03-01]: Use completed_steps dict from database queries for failed state resume logic
- [Phase 03-01]: Fix status mismatch between generate_keyframes (generating_video) and state machine (video_gen) in orchestrator
- [Phase 03-02]: Use asyncio.run() wrapper pattern for Typer + async database operations
- [Phase 03-02]: Implement progress_callback wrapper to update Rich status spinners from orchestrator
- [Phase 03-02]: Temporarily override settings.pipeline.crossfade_seconds in stitch command for per-invocation control
- [Phase 03-03]: Use asynccontextmanager lifespan instead of deprecated @app.on_event
- [Phase 03-03]: Create fresh async_session() in background tasks (never share request session)
- [Phase 03-03]: APIRouter with /api prefix for route organization
- [Phase 03-03]: Exclude output_path from response schemas for security
- [Phase 03-03]: FileResponse with Content-Disposition attachment header for MP4 downloads
- [Phase 03-03]: Generic exception handler returns 500 with detail (prevents stack trace leakage)
- **04-01:** Assets belong to manifests only (no project_id column) per V2 architecture
- **04-01:** Soft delete for manifests (deleted_at column) prevents data loss
- **04-01:** Auto-generate manifest_tag on asset creation (CHAR_01, CHAR_02, OBJ_01)
- **04-01:** Explicit index on Asset.manifest_id for query performance on SQLite
- **04-01:** Image upload saves to tmp/manifests/{manifest_id}/uploads/ directory structure
- **04-01:** Return 409 Conflict when deleting manifest referenced by projects
- **04-02:** Reuse StatusBadge component for manifest status display (works with arbitrary status strings)
- **04-02:** Category filter as pills rather than dropdown for better visibility
- **04-02:** Delete confirmation modal with "can be undone" messaging (soft delete on backend)
- **04-02:** Duplicate prepends to list for immediate visibility of new manifest
- **05-01:** Store face embeddings as bytes (numpy.tobytes()) not JSON for 10x storage reduction
- **05-01:** Use yolov8m.pt (medium model) for balance of speed and accuracy
- **05-01:** Extract faces from person detections (upper 40% of bbox) until dedicated face model added
- **05-01:** Use gemini-2.0-flash-exp for reverse-prompting (speed over accuracy for 20+ crops)
- **05-01:** Lazy-load all CV models to avoid import-time overhead and allow graceful failure
- **05-01:** Add VEHICLE asset type with VEH prefix for automotive content
- **05-02:** Contact sheet uses 4-column grid with 256px thumbnails and DejaVu Sans font
- **05-02:** Rate limiting: 5 concurrent reverse-prompting requests via asyncio.Semaphore
- **05-02:** Face deduplication keeps highest-confidence crop, marks others in description
- **05-02:** Sequential tag reassignment ordered by sort_order then detection_confidence
- **05-02:** reprocess_asset updates 7 fields: reverse_prompt, visual_description, quality_score, detection_class, detection_confidence, is_face_crop, crop_bbox
- **05-02:** Stage 3 inline editing: UpdateAssetRequest accepts reverse_prompt and visual_description, manifest_service.update_asset allowed_fields includes both
- **06-01:** ManifestSnapshot model serializes full manifest + assets to JSON for immutable state capture
- **06-01:** Usage tracking increments times_used and sets last_used_at with UTC timezone
- **06-01:** Optional manifest_id in GenerateRequest maintains backward compatibility
- **06-01:** Pipeline manifesting skip documented for Phase 7+ integration (implicit in Phase 6)
- **06-02:** ManifestCard compact mode reuses component pattern (smaller text, hidden actions)
- **06-02:** Filter READY manifests client-side until backend status filtering implemented
- **06-02:** Use ?? undefined (not ?? null) to omit manifest_id from JSON when unset
- **06-02:** Limit manifest grid to 6 items with max-h-64 to prevent UI bloat
- **07-01:** EnhancedStoryboardOutput is separate model (not subclass) because scenes field type differs
- **07-01:** Store full manifests as JSON with denormalized fields for efficient querying
- **07-01:** Use JSON columns for arrays (asset_tags, speaker_tags) following Manifest.tags pattern
- **07-01:** Composite PKs (project_id, scene_index) eliminate need for UUID primary keys on scene manifest tables
- **07-02:** load_manifest_assets orders by quality_score desc (not sort_order) for LLM attention prioritization
- **07-02:** Production notes only included for quality >= 7.0 to manage context window size
- **07-02:** Asset tags validated post-generation with warnings (not errors) to allow new asset declarations
- **07-02:** use_manifests flag determines schema, prompt, and persistence path in single function
- **08-01:** Scene-type-aware selection adapts prioritization by shot_type (close_up prioritizes face crops, two_shot ensures 2 unique characters, establishing prioritizes environments)
- **08-01:** Deduplication by manifest_tag prevents same character occupying multiple slots for Veo reference diversity
- **08-02:** Duration forced to 8 seconds when reference_images attached (Veo 3.1 API constraint - non-negotiable)
- **08-02:** Reference images passed on ALL safety escalation levels (identity references independent of content-policy prefixes)
- **08-02:** Tier 3 face validation with 3 attempts and threshold loosening 0.6 → 0.5 (balances quality with success rate)
- **09-01:** Store clip_embedding as bytes (numpy.tobytes()) matching face_embedding pattern for 10x storage reduction vs JSON
- **09-01:** cv2 imported inside frame_sampler functions (not top-level) to avoid ImportError when opencv not installed
- **09-01:** CVAnalysisConfig uses Field(default_factory=CVAnalysisConfig) so cv_analysis section is optional in config.yaml
- **09-01:** extract_frames() reads sequentially and saves only target frames for efficiency (avoids random seeks)
- **09-02:** CVAnalysisService lazy-loads child services via _get_X() getters to avoid import-time model loading
- **09-02:** Semantic analysis is OPTIONAL — Gemini Vision failure returns None and CVAnalysisResult remains valid
- **09-02:** asyncio.to_thread() wraps all CPU-bound inference (YOLO, ArcFace, CLIP) for event loop safety
- **09-02:** extract_and_register_new_entities uses asyncio.Semaphore(3) for Gemini rate-limiting (inline with generation)
- **09-02:** Extracted assets go to Asset Registry only — NOT auto-added to scene manifests (intent vs validation)
- **09-02:** IoU > 0.70 deduplication threshold for overlapping entity detections
- **09-03:** CV analysis runs inline per-scene (not batch at end) to enable progressive enrichment before next scene generates
- **09-03:** scene_manifest_row initialized to None before manifest_id guard so both crash-recovery and escalation paths can access it
- **09-03:** clip_embeddings excluded from cv_analysis_json persistence (model_dump exclude) to avoid large binary in JSON column
- **09-03:** CV analysis failure wrapped in try/except — never escalates to pipeline failure (graceful degradation)
- **10-01:** Separate rewrite_keyframe_prompt and rewrite_video_prompt methods — static-image formula vs motion+audio formula require different system prompts and schemas
- **10-01:** Module-level helper functions (_format_placed_assets, _build_continuity_patch, etc.) not static methods — simplifies testing, follows Python conventions
- **10-01:** _list_available_references shows only assets WITH reference_image_url — LLM cannot select what Veo cannot receive
- **10-01:** Rewriter does not persist results itself — caller (Plan 02: keyframes.py/video_gen.py) stores rewritten prompts in SceneManifest columns
- **10-02:** veo_ref_images built after LLM override — no separate rebuild needed (correct ordering: Phase 8 selection → Phase 10 rewriter → veo_ref_images construction)
- **10-02:** style_prefix kept with rewritten keyframe prompt; character_prefix dropped (rewriter already injects asset reverse_prompts, avoids double-injection)
- **10-02:** Re-raise PipelineStopped inside except Exception block in video rewriter — inherits from Exception so must be caught and re-raised explicitly
- **10-02:** base_video_prompt = None pattern in escalation loop — rewriter sets it, loop checks it, fallback to original scene.video_motion_prompt with style suffix
- **11-01:** GenerationCandidate placed after VideoClip in models.py — logically related video output artifacts
- **11-01:** Composite index idx_candidates_project_scene on (project_id, scene_index) for efficient per-scene queries
- **11-01:** quality_mode and candidate_count have default values (False/1) so existing projects load correctly without migration
- **11-01:** Gemini visual_quality and prompt_adherence batched into single Flash call to minimize cost and latency
- **11-01:** Scene 0 continuity auto-scores 10.0 — no prior scene to compare against
- **11-01:** Scoring failures use neutral 5.0 fallback — never escalate to pipeline failure (graceful degradation)
- **11-01:** candidate_count forced to 1 when quality_mode=False to prevent accidental multi-generation

### Roadmap Evolution

- Phases 4-12 added: V2 studio-grade pipeline (manifest system, manifesting engine, CV analysis, adaptive prompts, multi-candidate scoring, fork integration)
- Reference docs: `docs/v2-manifest.md`, `docs/v2-pipe-optimization.md`

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1:**
- ~~SQLite WAL mode must be enabled from first migration to prevent database corruption during crashes~~ ✓ Resolved in 01-03

**Phase 2:**
- Rate limiting on Vertex AI free tier may require quota increase or billing enablement for production use
- ADC authentication requires GOOGLE_APPLICATION_CREDENTIALS environment variable in production
- ffmpeg must be installed on deployment environment (validated at startup with clear error message)

**Phase 3:**
- ~~Cost estimation ($15 per 5-scene project) should be communicated to users before generation starts~~ ✓ Resolved in 03-02 (cost warning in CLI)

## Session Continuity

Last session: 2026-02-16 (execution)
Stopped at: Completed 11-01-PLAN.md (GenerationCandidate model, CandidateScoringService, Project quality columns)
Resume file: None
