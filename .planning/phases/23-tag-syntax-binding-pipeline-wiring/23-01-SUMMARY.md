---
phase: 23-tag-syntax-binding-pipeline-wiring
plan: 01
subsystem: api
tags: [regex, dataclass, sqlalchemy, tag-resolution, asset-library]

# Dependency graph
requires:
  - phase: 22-asset-library-actor-character-model
    provides: CastBinding, SetBinding, PropBinding, Actor, ActorRef, LibrarySet, LibrarySetRef, LibraryProp, LibraryPropRef models and tag_resolver.py
provides:
  - AT_TAG_PATTERN regex for @tag syntax detection
  - ResolvedAssetRef dataclass with structured asset metadata (8 fields)
  - has_any_tags() function detecting both @tag and [TYPE:TAG] patterns
  - resolve_tags_with_assets() function with batch-loaded binding lookup
  - Updated resolve_tags() with @tag cross-type resolution and deduplication
affects: [23-02, 24-keyframe-pipeline, storyboard-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [dual-regex-tag-detection, cross-type-binding-lookup, batch-entity-preloading, resolved-asset-ref-metadata]

key-files:
  created: []
  modified:
    - backend/vidpipe/services/tag_resolver.py

key-decisions:
  - "resolve_tags_with_assets() batch-loads ALL bindings for bible in 3 queries + bulk entity loads to avoid N+1"
  - "@tag cross-type lookup order: CastBinding -> PropBinding -> SetBinding, first match wins"
  - "Overlapping @tag and [TYPE:TAG] for same entity deduplicates -- typed tag resolves first, @tag skipped"
  - "resolve_tags() updated to handle both patterns (not just resolve_tags_with_assets) for consistency"
  - "Three _build_*_resolution() helpers share code between typed and @tag paths in batch mode"

patterns-established:
  - "Dual-regex pattern: TAG_PATTERN for [TYPE:TAG], AT_TAG_PATTERN for @tag -- both checked in has_any_tags()"
  - "Cross-type binding lookup: uppercase-normalize tag, check CastBinding->PropBinding->SetBinding priority"
  - "Batch preloading: load all bindings for a bible, then bulk-load entities and refs, resolve from memory dicts"
  - "ResolvedAssetRef as structured carrier for downstream image generation metadata"

requirements-completed: [ATAG-01, ATAG-02, ATAG-03]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 23 Plan 01: Tag Resolver Extension Summary

**Dual-regex tag resolution with @tag cross-type binding lookup, ResolvedAssetRef structured metadata, and batch-loaded resolve_tags_with_assets()**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T21:24:57Z
- **Completed:** 2026-03-14T21:27:58Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Extended tag_resolver.py to support @tag syntax alongside existing [TYPE:TAG] with cross-type CastBinding->PropBinding->SetBinding lookup
- Added ResolvedAssetRef dataclass with 8 fields (tag, asset_type, display_name, text_description, reference_image_urls, lora_url, wardrobe_override, lighting_notes)
- Implemented resolve_tags_with_assets() with batch preloading (3 binding queries + bulk entity/ref queries) to avoid N+1
- Added has_any_tags() for dual-pattern detection, used by resolve_scene_tags()
- Deduplication logic prevents double-resolution when both @brandon and [CHAR:BRANDON] appear in same prompt

## Task Commits

Each task was committed atomically:

1. **Task 1: Add ResolvedAssetRef, AT_TAG_PATTERN, has_any_tags, and @tag resolution** - `38ea20b` (feat)
2. **Task 2: Implement resolve_tags_with_assets()** - included in `38ea20b` (same file, tightly coupled implementation)

**Plan metadata:** pending

## Files Created/Modified
- `backend/vidpipe/services/tag_resolver.py` - Extended with AT_TAG_PATTERN, ResolvedAssetRef, has_any_tags(), resolve_tags_with_assets(), _resolve_at_tag_cross_type(), _build_char_resolution(), _build_set_resolution(), _build_prop_resolution()

## Decisions Made
- Both resolve_tags() and resolve_tags_with_assets() handle @tag patterns -- resolve_tags() does per-tag DB queries (lightweight path), resolve_tags_with_assets() does batch preloading (performance path)
- Three _build_*_resolution() pure functions extract resolution logic shared between typed and @tag paths in batch mode
- Cross-type collision warning logged but first match wins per CONTEXT.md decision
- Deleted entity references treated as partially resolved (name-only substitution, no ResolvedAssetRef)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Tag resolver fully extended with both syntaxes and structured metadata
- resolve_tags_with_assets() ready for storyboard pipeline wiring (Plan 23-02)
- ResolvedAssetRef carries all fields needed by future Phase 24 keyframe pipeline

---
*Phase: 23-tag-syntax-binding-pipeline-wiring*
*Completed: 2026-03-14*
