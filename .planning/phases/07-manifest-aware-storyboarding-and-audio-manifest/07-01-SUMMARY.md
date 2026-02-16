---
phase: 07-manifest-aware-storyboarding-and-audio-manifest
plan: 01
subsystem: database
tags: [pydantic, sqlalchemy, schemas, orm, manifest, audio]

# Dependency graph
requires:
  - phase: 02-generation-pipeline
    provides: StoryboardOutput Pydantic schemas for Gemini structured output
  - phase: 04-manifest-system-foundation
    provides: Manifest and Asset ORM models with JSON columns
provides:
  - EnhancedStoryboardOutput Pydantic schema with SceneManifest and SceneAudioManifest
  - SceneManifest and SceneAudioManifest ORM models with composite primary keys
  - 11 Pydantic models defining manifest-aware storyboard structure
affects: [07-02-enhanced-storyboard-pipeline, 08-veo-reference-passthrough, 09-cv-analysis-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Composite primary keys (project_id, scene_index) for scene-level manifest tables"
    - "JSON column storage for complex nested structures with denormalized query fields"
    - "Schema inheritance (EnhancedSceneSchema extends SceneSchema) for backward compatibility"

key-files:
  created:
    - backend/vidpipe/schemas/storyboard_enhanced.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py

key-decisions:
  - "EnhancedStoryboardOutput is separate model (not subclass) because scenes field type differs"
  - "Store full manifests as JSON with denormalized fields for efficient querying"
  - "Use JSON columns for arrays (asset_tags, speaker_tags) following Manifest.tags pattern"
  - "Composite PKs eliminate need for UUID primary keys on scene manifest tables"

patterns-established:
  - "All Field() calls include descriptions for Gemini structured output guidance"
  - "Denormalize frequently-queried fields (shot_type, has_dialogue) from JSON blobs"
  - "Import ORM models in __init__.py to register with Base.metadata for auto-creation"

# Metrics
duration: 2.0min
completed: 2026-02-16
---

# Phase 07 Plan 01: Schemas and Models for Manifest-Aware Storyboarding Summary

**Pydantic schemas for asset placement and audio direction with ORM models using composite PKs on (project_id, scene_index)**

## Performance

- **Duration:** 1min 57s
- **Started:** 2026-02-16T23:34:04Z
- **Completed:** 2026-02-16T23:36:01Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created 11 Pydantic models defining manifest-aware storyboard structure with asset placements, composition, and audio direction
- EnhancedStoryboardOutput schema enables Gemini to produce structured output referencing manifest assets
- SceneManifest and SceneAudioManifest ORM models with composite PKs persist per-scene manifests efficiently
- All schemas include Field descriptions required for Gemini structured output guidance

## Task Commits

Each task was committed atomically:

1. **Task 1: Create enhanced Pydantic schemas for manifest-aware storyboard output** - `19ff878` (feat)
2. **Task 2: Add SceneManifest and SceneAudioManifest ORM models with composite PKs** - `ef91354` (feat)

## Files Created/Modified
- `backend/vidpipe/schemas/storyboard_enhanced.py` - 11 Pydantic models for manifest-aware storyboard: AssetPlacement, SceneComposition, DialogueLine, SFXEntry, AmbientAudio, MusicDirection, AudioContinuity, SceneManifestSchema, SceneAudioManifestSchema, EnhancedSceneSchema, EnhancedStoryboardOutput
- `backend/vidpipe/db/models.py` - Added SceneManifest and SceneAudioManifest ORM models with composite primary keys
- `backend/vidpipe/db/__init__.py` - Import new models to register with Base.metadata

## Decisions Made

**EnhancedStoryboardOutput as separate model (not subclass):**
- EnhancedStoryboardOutput cannot inherit from StoryboardOutput because the `scenes` field type differs (list[EnhancedSceneSchema] vs list[SceneSchema])
- Pydantic does not support field type overriding in inheritance
- Created as parallel model reusing StyleGuide and CharacterDescription imports

**Composite primary keys (project_id, scene_index):**
- Eliminates need for UUID primary keys on scene manifest tables
- Natural composite key provides direct lookup by project and scene number
- Matches semantic relationship: exactly one manifest per (project, scene) pair

**JSON storage with denormalization:**
- Full manifests stored as JSON (manifest_json, dialogue_json, sfx_json, etc.)
- Frequently-queried fields denormalized (composition_shot_type, has_dialogue, speaker_tags)
- Balances storage efficiency with query performance
- Follows established pattern from Manifest.tags (JSON columns storing arrays)

**Field descriptions for Gemini:**
- All Field() calls include description parameter
- Descriptions guide Gemini's structured output generation
- Critical for response_schema parameter effectiveness (established in Phase 2)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Config validation during verification:**
- Direct import of vidpipe.db.models triggers config.yaml loading via engine.py → config.py chain
- Config validation fails without config.yaml present
- Workaround: Used exec() to load models.py directly for verification
- No impact on actual functionality - models will work correctly when config is present

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 02 (Enhanced Storyboard Pipeline):**
- Pydantic schemas ready for Gemini response_schema parameter
- ORM models ready for database persistence
- EnhancedSceneSchema properly inherits SceneSchema fields (scene_index, scene_description, key_details, start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes) plus scene_manifest and audio_manifest
- Tables will auto-create on first init_database() call via metadata.create_all()

**No blockers.**

## Self-Check: PASSED

All files exist:
- FOUND: backend/vidpipe/schemas/storyboard_enhanced.py
- FOUND: backend/vidpipe/db/models.py (modified)
- FOUND: backend/vidpipe/db/__init__.py (modified)

All commits exist:
- FOUND: 19ff878 (Task 1: Pydantic schemas)
- FOUND: ef91354 (Task 2: ORM models)

---
*Phase: 07-manifest-aware-storyboarding-and-audio-manifest*
*Completed: 2026-02-16*
