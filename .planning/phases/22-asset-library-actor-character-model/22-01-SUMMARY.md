---
phase: 22-asset-library-actor-character-model
plan: 01
subsystem: database
tags: [sqlalchemy, orm, asset-library, actor, bindings, uuid, migration]

requires:
  - phase: 17-production-bible-entity-expansion
    provides: Character, Set, Prop, ScoreTheme, SFXItem ORM models

provides:
  - Actor, ActorRef, ActorVoiceProfile, ActorWardrobePreset standalone ORM models
  - LibrarySet, LibrarySetRef, LibrarySonicIdentity standalone ORM models
  - LibraryProp, LibraryPropRef standalone ORM models
  - SoundAsset unified sound entity ORM model
  - CastBinding, SetBinding, PropBinding, SoundBinding bible connector ORM models
  - promoted_to columns on Phase 17 entities for promotion tracking

affects: [22-02, 22-03, 22-04, 22-05, 22-06]

tech-stack:
  added: []
  patterns:
    - "Standalone library entities with separate ref/sub-entity tables"
    - "Binding tables with UniqueConstraint for tag/entity dedup per bible"
    - "promoted_to FK columns for promotion tracking from bible-scoped to library entities"

key-files:
  created: []
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py

key-decisions:
  - "Separate actor_voice_profiles and actor_wardrobe_presets tables from existing voice_profiles/wardrobes to avoid coupling library to bible-scoped entities"
  - "library_sets/library_props naming avoids collision with existing sets/props tables"
  - "SoundAsset unified with category discrimination (SCORE_THEME/SFX/AMBIENCE/FOLEY/UI) rather than separate tables"
  - "CastBinding tag column for tag resolution in prompt injection, plus prompt_tags for additional injection"

patterns-established:
  - "Standalone library entity pattern: entity + ref table + sub-entity tables, no production_bible_id FK"
  - "Binding table pattern: UniqueConstraint on (production_bible_id, entity_id) and (production_bible_id, tag)"
  - "Promotion tracking pattern: nullable FK on bible-scoped entity pointing to library entity"

requirements-completed: [ALIB-01, ALIB-03, ALIB-05, ALIB-09]

duration: 3min
completed: 2026-03-05
---

# Phase 22 Plan 01: Database Schema Foundation Summary

**14 ORM models for Asset Library standalone entities (Actor, LibrarySet, LibraryProp, SoundAsset) with binding tables and promotion tracking on existing Phase 17 entities**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-05T13:53:02Z
- **Completed:** 2026-03-05T13:55:49Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added 14 new ORM model classes covering 4 standalone library entities, their sub-entity/ref tables, and 4 binding tables
- All binding tables have UniqueConstraints preventing duplicate actor/set/prop/sound per bible and duplicate tags per bible
- Added promoted_to FK columns on 5 existing Phase 17 entities with corresponding migration entries

## Task Commits

Each task was committed atomically:

1. **Task 1: Add standalone library entity ORM models and sub-entity tables** - `6f2db5c` (feat)
2. **Task 2: Add migration entries for promotion tracking columns** - `636dd35` (feat)

## Files Created/Modified
- `backend/vidpipe/db/models.py` - 14 new ORM classes (Actor, ActorRef, ActorVoiceProfile, ActorWardrobePreset, LibrarySet, LibrarySetRef, LibrarySonicIdentity, LibraryProp, LibraryPropRef, SoundAsset, CastBinding, SetBinding, PropBinding, SoundBinding) plus promoted_to columns on Character, Set, Prop, ScoreTheme, SFXItem
- `backend/vidpipe/db/__init__.py` - Migration entries for promoted_to columns, imports for new models

## Decisions Made
- Separate actor_voice_profiles/actor_wardrobe_presets tables from existing voice_profiles/wardrobes per Research pitfall guidance
- library_sets/library_props table naming avoids collision with existing sets/props tables
- SoundAsset uses category column (SCORE_THEME/SFX/AMBIENCE/FOLEY/UI) for unified sound entity per Research recommendation
- CastBinding has dedicated tag column for tag resolution plus prompt_tags for additional injection per Research open question 4

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 14 ORM models are importable and verified with correct table names, FKs, and constraints
- Migration entries ready for existing DB upgrades on both SQLite and PostgreSQL
- Schema foundation complete for Plans 02-06 (API endpoints, frontend components)

---
*Phase: 22-asset-library-actor-character-model*
*Completed: 2026-03-05*
