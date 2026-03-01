---
phase: 18-screenplay-system
plan: 01
subsystem: database, services
tags: [sqlalchemy, pydantic, llm, screenplay, structured-output]

# Dependency graph
requires:
  - phase: 17-production-bible-entity-expansion
    provides: Character, Set, Prop entity models with prompt_tags for entity validation
  - phase: 13-llm-provider-abstraction-ollama
    provides: LLMAdapter ABC and get_adapter() registry for model routing
provides:
  - Screenplay ORM model with 1:1 production_id unique constraint
  - Scene screenplay_breakdown_index and screenplay_context columns
  - 8 Pydantic schemas for LLM structured output (LoglineOutput through ScriptOutput)
  - ScreenwriterService with 6 individual generation methods + generate_full chain
  - load_bible_context helper for Production Bible context injection
  - Post-LLM entity validation against Production Bible (SCRN-06)
affects: [18-02 (API routes), 18-03 (frontend UI), storyboard enrichment]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sequential LLM chain with incremental DB commits per step"
    - "generating_step column for progress polling (alternative to event_bus)"
    - "Post-LLM entity tag validation with warning-only logging"

key-files:
  created:
    - backend/vidpipe/schemas/screenplay.py
    - backend/vidpipe/services/screenwriter.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py

key-decisions:
  - "generating_step column on Screenplay for progress tracking instead of event_bus (simpler, consistent with Shot.generation_status pattern)"
  - "screenplay_breakdown_index is plain INTEGER not FK to screenplays (avoids FK ordering issues per Research pitfall 5)"
  - "screenplay_context JSON column on Scene for denormalized breakdown data (avoids cross-table join in storyboard.py)"
  - "Entity validation is warning-only (LLM may invent characters not in bible)"
  - "character_breakdowns stored as list of dicts (not nested dict with 'characters' key) for simpler downstream consumption"

patterns-established:
  - "ScreenwriterService: class-based with LLMAdapter injection, each method checks LOCKED, sets generating_step, calls adapter, commits"
  - "load_bible_context: finds production bible via Scene.production_bible_id under the production"

requirements-completed: [SCRN-01, SCRN-02, SCRN-05, SCRN-07, SCRN-08, SCRN-09, SCRN-10, SCRN-11]

# Metrics
duration: 3min
completed: 2026-03-01
---

# Phase 18 Plan 01: Screenplay Data Model and Generation Service Summary

**Screenplay ORM model with 6-step LLM generation chain (logline through script), incremental commits, LOCKED-status guard, and Production Bible entity validation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-01T15:40:28Z
- **Completed:** 2026-03-01T15:44:04Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Screenplay ORM model with unique production_id FK, all text/JSON fields, status (DRAFT/IN_REVIEW/LOCKED), text_model, and generating_step progress column
- Scene model extended with screenplay_breakdown_index (int) and screenplay_context (JSON) for storyboard enrichment
- 8 Pydantic output schemas covering all 6 generation steps plus sub-entry types
- ScreenwriterService with LOCKED-status guard, incremental commits, and generating_step progress tracking
- Post-LLM entity validation in generate_scene_breakdown validates character/set/prop tags against Production Bible entities
- Dedicated generate_shot_list method independently regeneratable per SCRN-09

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Screenplay ORM model and Scene columns** - `0b25dd0` (feat)
2. **Task 2: Create Pydantic schemas and ScreenwriterService** - `660fa8c` (feat)

## Files Created/Modified
- `backend/vidpipe/db/models.py` - Added Screenplay ORM model (after Sequence, before Scene) and screenplay_breakdown_index/screenplay_context columns on Scene
- `backend/vidpipe/db/__init__.py` - Added Screenplay import and migration entries for Scene columns + unique index
- `backend/vidpipe/schemas/screenplay.py` - 8 Pydantic schemas for LLM structured output (LoglineOutput, TreatmentOutput, CharacterBreakdownsOutput, SceneBreakdownOutput, ShotListOutput, ShotListEntry, SceneBreakdownEntry, ScriptOutput)
- `backend/vidpipe/services/screenwriter.py` - ScreenwriterService with 7 async methods + prompt builders + entity validation + bible context loader

## Decisions Made
- Used generating_step column on Screenplay for progress polling instead of event_bus -- simpler approach, consistent with Shot.generation_status polling pattern already in the codebase
- screenplay_breakdown_index is a plain INTEGER, not a FK to screenplays -- avoids FK ordering issues during migration (per Research pitfall 5)
- screenplay_context is a JSON blob denormalized onto Scene -- avoids cross-table join in storyboard.py during generation
- Entity validation in generate_scene_breakdown is warning-only -- LLM may invent characters not yet in the Production Bible, and that is acceptable
- character_breakdowns stored as flat list of dicts (not nested under a "characters" key) for simpler downstream iteration

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Screenplay ORM model and service layer complete, ready for Plan 02 (API routes)
- ScreenwriterService ready for endpoint wiring via get_adapter() pattern
- Pydantic schemas ready for both LLM output parsing and API response serialization

## Self-Check: PASSED

- FOUND: backend/vidpipe/schemas/screenplay.py
- FOUND: backend/vidpipe/services/screenwriter.py
- FOUND: backend/vidpipe/db/models.py (modified)
- FOUND: backend/vidpipe/db/__init__.py (modified)
- FOUND: commit 0b25dd0 (Task 1)
- FOUND: commit 660fa8c (Task 2)

---
*Phase: 18-screenplay-system*
*Completed: 2026-03-01*
