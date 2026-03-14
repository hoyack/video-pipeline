# Phase 23: Tag Syntax & Binding Pipeline Wiring - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning
**Source:** PRD Express Path (docs/assets_mapping.md, Section A)

<domain>
## Phase Boundary

This phase extends the existing tag resolver and pipeline wiring to support `@tag` syntax, carry structured asset metadata through the generation pipeline, and expose binding data for frontend consumption. It does NOT touch image generation (ComfyUI/Flux), LoRA training, or frontend editor enhancements — those are Phases 24-26.

Specifically, this phase delivers:
- `@tag` pattern matching alongside existing `[TYPE:TAG]` syntax
- `ResolvedAssetRef` dataclass with full asset metadata for downstream image generation
- `resolve_tags_with_assets()` function loading from binding tables
- `format_binding_registry()` for LLM context injection in storyboard pipeline
- Storyboard pipeline wiring to use binding registry
- `GET /api/production-bibles/{id}/bound-assets/summary` API endpoint
- Frontend TypeScript types and API client function for bound asset summaries

</domain>

<decisions>
## Implementation Decisions

### Tag Syntax
- Support BOTH `@tag` and `[TYPE:TAG]` syntax — `@tag` is primary user-facing, `[TYPE:TAG]` retained for backward compatibility and explicit type-scoped resolution
- `@tag` performs cross-type lookup in order: CastBinding → PropBinding → SetBinding; first match wins
- Unresolved `@tags` are left as-is and logged as unresolved
- Case-insensitive matching on binding `tag` field

### Tag Resolver Extension
- Add `AT_TAG_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")` alongside existing `TAG_PATTERN`
- Extend `ResolvedPrompt` with `asset_refs: list[ResolvedAssetRef]` field
- Keep backward-compat fields (`character_refs`, `set_context`) intact
- New `resolve_tags_with_assets()` function loads from CastBinding, SetBinding, PropBinding tables

### ResolvedAssetRef Structure
- Fields: tag, asset_type (CHARACTER/PROP/SET), display_name, text_description, reference_image_urls, lora_url (nullable), wardrobe_override (nullable), lighting_notes (nullable)
- text_description sourced from: Actor.base_appearance_prompt (CHARACTER), LibraryProp.appearance_prompt (PROP), LibrarySet.reverse_prompt (SET)
- reference_image_urls collected from: ActorRef images (CHARACTER), LibraryProp reference images (PROP), LibrarySet reference images (SET)

### Storyboard Pipeline Integration
- New `format_binding_registry()` function in manifest_service.py reads from binding tables
- Used when scene has `production_bible_id` AND bible has bindings
- Injected into LLM system prompt so storyboard LLM knows valid `@tag` references
- Coexists with existing `format_asset_registry()` — uses binding path when bindings exist, falls back to old manifest path otherwise

### API Endpoint
- `GET /api/production-bibles/{id}/bound-assets/summary` returns flat list combining CastBindings, SetBindings, PropBindings
- Each entry includes: tag, name, type, primary_thumbnail_url, description
- Used by frontend for autocomplete data (Phase 26) and general binding overview

### Claude's Discretion
- Error handling strategy for missing/deleted assets referenced by bindings
- Exact SQL query optimization for cross-type binding lookups
- Response pagination for bound-assets summary (if needed)
- Unit test strategy and test file organization

</decisions>

<specifics>
## Specific Ideas

- Tag resolver regex patterns from PRD:
  ```python
  TAG_PATTERN = re.compile(r"\[(CHAR|SET|PROP):([A-Z0-9_]+)\]")
  AT_TAG_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")
  ```
- Cross-type lookup order: CastBinding → PropBinding → SetBinding (character identity is highest priority)
- ResolvedAssetRef dataclass as defined in PRD Section 4.1
- format_binding_registry() query pattern from PRD Section 4.3
- The decision tree for keyframe pipeline path selection (PRD Section 4.2) — implementation deferred to Phase 24, but the data structures built here must support it

</specifics>

<deferred>
## Deferred Ideas

- ComfyUI Flux.1 workflow templates and builder → Phase 24
- Keyframe pipeline updates to use ResolvedAssetRef for image generation → Phase 24
- LoRA training infrastructure → Phase 25
- Frontend @tag autocomplete in scene editor → Phase 26
- Frontend tag preview panel → Phase 26
- Frontend tag reference sheet tab → Phase 26

</deferred>

---

*Phase: 23-tag-syntax-binding-pipeline-wiring*
*Context gathered: 2026-03-14 via PRD Express Path*
