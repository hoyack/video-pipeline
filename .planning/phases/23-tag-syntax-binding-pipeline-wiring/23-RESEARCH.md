# Phase 23: Tag Syntax & Binding Pipeline Wiring - Research

**Researched:** 2026-03-14
**Domain:** Tag resolution, pipeline wiring, binding data aggregation API
**Confidence:** HIGH

## Summary

Phase 23 extends the existing tag resolver in `tag_resolver.py` to support `@tag` syntax alongside the existing `[TYPE:TAG]` pattern, introduces a `ResolvedAssetRef` dataclass that carries structured asset metadata through the pipeline, adds `resolve_tags_with_assets()` for binding-based asset loading, creates `format_binding_registry()` for LLM context injection in the storyboard pipeline, and exposes a summary API endpoint for frontend consumption.

The codebase is well-prepared for this work. The existing tag resolver handles `[CHAR:TAG]`, `[SET:TAG]`, `[PROP:TAG]` with full binding table lookups. The storyboard pipeline already calls `resolve_tags()` and `format_asset_registry()`. The binding tables (CastBinding, SetBinding, PropBinding) are complete with tag uniqueness constraints, and the frontend already has TypeScript types for all binding entities. The primary work is additive: new regex pattern, new dataclass, new functions, and a new API endpoint.

**Primary recommendation:** Extend `tag_resolver.py` with the `AT_TAG_PATTERN` regex and `resolve_tags_with_assets()`, add `format_binding_registry()` to `manifest_service.py`, wire it into `storyboard.py`'s system prompt, add the summary endpoint to `bindings.py`, and add the frontend TS type + API function.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Support BOTH `@tag` and `[TYPE:TAG]` syntax -- `@tag` is primary user-facing, `[TYPE:TAG]` retained for backward compatibility and explicit type-scoped resolution
- `@tag` performs cross-type lookup in order: CastBinding -> PropBinding -> SetBinding; first match wins
- Unresolved `@tags` are left as-is and logged as unresolved
- Case-insensitive matching on binding `tag` field
- Add `AT_TAG_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")` alongside existing `TAG_PATTERN`
- Extend `ResolvedPrompt` with `asset_refs: list[ResolvedAssetRef]` field
- Keep backward-compat fields (`character_refs`, `set_context`) intact
- New `resolve_tags_with_assets()` function loads from CastBinding, SetBinding, PropBinding tables
- ResolvedAssetRef fields: tag, asset_type (CHARACTER/PROP/SET), display_name, text_description, reference_image_urls, lora_url (nullable), wardrobe_override (nullable), lighting_notes (nullable)
- text_description sourced from: Actor.base_appearance_prompt (CHARACTER), LibraryProp.appearance_prompt (PROP), LibrarySet.reverse_prompt (SET)
- reference_image_urls collected from: ActorRef images (CHARACTER), LibraryProp reference images (PROP), LibrarySet reference images (SET)
- New `format_binding_registry()` function in manifest_service.py reads from binding tables
- Used when scene has `production_bible_id` AND bible has bindings
- Injected into LLM system prompt so storyboard LLM knows valid `@tag` references
- Coexists with existing `format_asset_registry()` -- uses binding path when bindings exist, falls back to old manifest path otherwise
- `GET /api/production-bibles/{id}/bound-assets/summary` returns flat list combining CastBindings, SetBindings, PropBindings
- Each entry includes: tag, name, type, primary_thumbnail_url, description
- Frontend BoundAssetSummary TypeScript type and getBoundAssetsSummary() API client function

### Claude's Discretion
- Error handling strategy for missing/deleted assets referenced by bindings
- Exact SQL query optimization for cross-type binding lookups
- Response pagination for bound-assets summary (if needed)
- Unit test strategy and test file organization

### Deferred Ideas (OUT OF SCOPE)
- ComfyUI Flux.1 workflow templates and builder -> Phase 24
- Keyframe pipeline updates to use ResolvedAssetRef for image generation -> Phase 24
- LoRA training infrastructure -> Phase 25
- Frontend @tag autocomplete in scene editor -> Phase 26
- Frontend tag preview panel -> Phase 26
- Frontend tag reference sheet tab -> Phase 26
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ATAG-01 | Tag resolver supports @tag pattern alongside existing [TYPE:TAG] syntax with cross-type lookup | AT_TAG_PATTERN regex, cross-type resolution function with CastBinding->PropBinding->SetBinding priority, case-insensitive matching via `.upper()` comparison |
| ATAG-02 | ResolvedAssetRef dataclass carries structured asset metadata | New dataclass in tag_resolver.py with fields mapped to Actor/LibraryProp/LibrarySet + their Ref tables |
| ATAG-03 | resolve_tags_with_assets() loads asset data from binding tables | New async function performing sequential lookups across binding tables + joined entity loads |
| ATAG-04 | format_binding_registry() formats all bound assets for LLM context injection | New function in manifest_service.py querying CastBinding/SetBinding/PropBinding with joined entities |
| ATAG-05 | Storyboard pipeline uses format_binding_registry() when bindings exist | Conditional logic in storyboard.py: check for bindings, use binding registry, fallback to asset registry |
| ATAG-06 | GET /api/production-bibles/{id}/bound-assets/summary endpoint | New route in bindings.py aggregating all three binding types into flat list |
| ATAG-07 | Frontend BoundAssetSummary type and getBoundAssetsSummary() function | New type in types.ts, new function in client.ts |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `re` | 3.11+ | Regex pattern matching for `@tag` syntax | Already used for `TAG_PATTERN`, no external dep needed |
| Python stdlib `dataclasses` | 3.11+ | `ResolvedAssetRef` dataclass definition | Already used for `ResolvedPrompt` |
| SQLAlchemy async | 2.0+ | Async queries for binding table lookups | Already used throughout project |
| FastAPI | 0.100+ | New API endpoint | Already used for all routes |
| Pydantic | 2.0+ | Request/response schemas (if needed) | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `sqlalchemy.orm.selectinload` | 2.0+ | Eager-load related entities in binding queries | When fetching bindings + their parent entities in batch |

No new dependencies required. This phase is purely additive to existing libraries.

## Architecture Patterns

### Recommended Changes Structure
```
backend/vidpipe/
├── services/
│   ├── tag_resolver.py         # MODIFY: Add AT_TAG_PATTERN, ResolvedAssetRef, resolve_tags_with_assets(), has_any_tags()
│   └── manifest_service.py     # MODIFY: Add format_binding_registry()
├── pipeline/
│   └── storyboard.py           # MODIFY: Use format_binding_registry() when bindings exist
├── api/
│   └── bindings.py             # MODIFY: Add bound-assets/summary endpoint
frontend/src/
├── api/
│   ├── types.ts                # MODIFY: Add BoundAssetSummary type
│   └── client.ts               # MODIFY: Add getBoundAssetsSummary() function
```

### Pattern 1: Dual-Regex Tag Detection
**What:** Support both `@tag` and `[TYPE:TAG]` in a single resolution pass.
**When to use:** All tag resolution entry points.
**Example:**
```python
TAG_PATTERN = re.compile(r"\[(CHAR|SET|PROP):([A-Z0-9_]+)\]")
AT_TAG_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")

def has_any_tags(text: str) -> bool:
    """Return True if text contains @tag or [TYPE:TAG] patterns."""
    if not text:
        return False
    return bool(TAG_PATTERN.search(text) or AT_TAG_PATTERN.search(text))
```

### Pattern 2: Cross-Type Binding Lookup for @tag
**What:** When an `@tag` is found, look it up across all binding tables in priority order.
**When to use:** Resolution of `@tag` patterns (not `[TYPE:TAG]` which already specifies the type).
**Example:**
```python
async def _resolve_at_tag(
    tag_name: str,
    bible_id: uuid.UUID,
    session: AsyncSession,
) -> tuple[str | None, str | None, ResolvedAssetRef | None]:
    """Cross-type lookup: CastBinding -> PropBinding -> SetBinding.

    Returns (replacement_text, asset_type, resolved_asset_ref) or (None, None, None).
    Case-insensitive matching via .upper() on the tag field.
    """
    upper_tag = tag_name.upper()

    # 1. CastBinding first (character identity is highest priority)
    result = await session.execute(
        select(CastBinding).where(
            CastBinding.production_bible_id == bible_id,
            func.upper(CastBinding.tag) == upper_tag,
        )
    )
    binding = result.scalars().first()
    if binding:
        # Load actor + refs, build ResolvedAssetRef
        ...

    # 2. PropBinding second
    result = await session.execute(
        select(PropBinding).where(
            PropBinding.production_bible_id == bible_id,
            func.upper(PropBinding.tag) == upper_tag,
        )
    )
    # ...

    # 3. SetBinding last
    # ...
```

### Pattern 3: Additive ResolvedPrompt Extension
**What:** Add `asset_refs` field to `ResolvedPrompt` while keeping existing fields for backward compat.
**When to use:** All existing callers of `resolve_tags()` continue to work unchanged.
**Example:**
```python
@dataclass
class ResolvedPrompt:
    text: str
    asset_refs: list[ResolvedAssetRef] = field(default_factory=list)  # NEW
    character_refs: list = field(default_factory=list)   # backward compat
    set_context: list = field(default_factory=list)      # backward compat
    unresolved_tags: list = field(default_factory=list)
```

### Pattern 4: Binding Registry vs Asset Registry Fallback
**What:** In storyboard.py, check if bindings exist for the bible before choosing which registry format to inject.
**When to use:** When building the LLM system prompt.
**Example:**
```python
# In storyboard.py generate_storyboard():
if use_manifests:
    # Check if bible has bindings (new path)
    from vidpipe.services.manifest_service import format_binding_registry
    binding_registry = await format_binding_registry(session, scene.production_bible_id)
    if binding_registry:
        asset_registry_block = binding_registry
    else:
        # Fallback to old manifest asset path
        assets = await load_manifest_assets(session, scene.production_bible_id)
        asset_registry_block = format_asset_registry(assets)
```

### Anti-Patterns to Avoid
- **Do NOT modify existing `resolve_tags()` return signature in a breaking way:** Add `asset_refs` as optional with default, keep `character_refs` and `set_context` intact.
- **Do NOT make `@tag` detection break `[TYPE:TAG]`:** Process `[TYPE:TAG]` matches first (they are more specific), then `@tag` matches. Avoid double-resolving if a tag is referenced in both forms.
- **Do NOT query binding tables separately for each tag match:** Batch-load all bindings for the bible once, then resolve tags from the in-memory lookup dict.
- **Do NOT make the summary endpoint paginated initially:** The typical Production Bible has 5-20 bindings. Pagination adds complexity for no benefit at this scale.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Case-insensitive SQL comparison | Python `.upper()` then `==` | `func.upper(CastBinding.tag) == upper_tag` | Works on both SQLite and PostgreSQL, uses SQL-level comparison |
| Tag uniqueness across types | Custom check logic | Rely on existing per-table UniqueConstraints | Tags are already unique within each binding type per bible |
| Asset primary thumbnail lookup | Custom per-asset query | Reuse `is_primary == True` pattern from existing binding list endpoints | Pattern already proven in `bindings.py` list endpoints |

## Common Pitfalls

### Pitfall 1: @tag Matching False Positives
**What goes wrong:** The `@` pattern matches email addresses (`user@example.com`), Twitter handles in text, or other non-tag uses of `@`.
**Why it happens:** `@([a-zA-Z0-9_]+)` is broad.
**How to avoid:** The pattern `r"@([a-zA-Z0-9_]+)"` already requires alphanumeric+underscore (no dots), which excludes emails. For extra safety, only resolve `@tags` when the scene has a `production_bible_id`. Unresolved tags are left as-is (not errors), so false positives just get logged and left in the text.
**Warning signs:** Unresolved tag warnings in logs for non-tag `@` usage.

### Pitfall 2: Case Sensitivity Mismatch Between @tag and DB Tag
**What goes wrong:** User writes `@brandon` but binding stores tag as `BRANDON`. Lookup fails.
**Why it happens:** `[TYPE:TAG]` syntax uses uppercase by convention. `@tag` allows mixed case.
**How to avoid:** Use `func.upper()` in SQL WHERE clause for case-insensitive comparison. The CONTEXT.md decision explicitly requires case-insensitive matching.
**Warning signs:** Tags resolve via `[CHAR:BRANDON]` but not via `@brandon`.

### Pitfall 3: Processing Order -- @tag vs [TYPE:TAG] Overlap
**What goes wrong:** A prompt contains both `[CHAR:BRANDON]` and `@brandon` for the same entity. Both get processed, resulting in duplicate substitutions or a mangled prompt.
**Why it happens:** Both regex patterns match independently.
**How to avoid:** Process `[TYPE:TAG]` patterns first (more specific, explicit type). Track which tags have been resolved. When processing `@tag` matches, skip any that match an already-resolved tag.
**Warning signs:** Doubled character descriptions in resolved prompt text.

### Pitfall 4: Deleted Asset References in Bindings
**What goes wrong:** A binding references an Actor/LibrarySet/LibraryProp that has been deleted. The binding row still exists but the joined entity is None.
**Why it happens:** No cascade delete from library entity to binding (by design -- bindings should be cleaned up explicitly).
**How to avoid:** Always null-check the loaded entity (`actor = await session.get(Actor, binding.actor_id)` -- if None, treat as unresolved). The existing `_resolve_char_tag` already handles this pattern.
**Warning signs:** KeyError or AttributeError in tag resolution when an asset has been deleted.

### Pitfall 5: N+1 Query Problem in format_binding_registry()
**What goes wrong:** Loading all bindings for a bible, then loading each actor/set/prop individually, creates N+1 queries.
**Why it happens:** Sequential `session.get()` calls inside a loop.
**How to avoid:** Batch-load: first query all CastBindings, collect actor_ids, then `select(Actor).where(Actor.id.in_(actor_ids))`. Same for sets and props. This pattern is already used in the binding list endpoints in `bindings.py`.
**Warning signs:** Slow storyboard generation startup, excessive DB query logs.

### Pitfall 6: Storyboard Prompt Duplication
**What goes wrong:** Both `format_binding_registry()` and `format_asset_registry()` inject asset descriptions into the LLM prompt, resulting in duplicate or conflicting information.
**Why it happens:** The fallback logic is not exclusive.
**How to avoid:** The decision is clear: use `format_binding_registry()` when bindings exist, `format_asset_registry()` otherwise. They are mutually exclusive paths.
**Warning signs:** LLM sees two "AVAILABLE ASSETS" blocks in the system prompt.

## Code Examples

### Existing Tag Resolution Call in storyboard.py (Lines 428-438)
```python
# Phase 22: Resolve [CHAR:TAG], [SET:TAG], [PROP:TAG] in scene prompt
resolved_prompt_text = scene.prompt
if scene.production_bible_id:
    from vidpipe.services.tag_resolver import has_tags, resolve_tags
    if has_tags(scene.prompt):
        resolved = await resolve_tags(scene.prompt, scene.production_bible_id, session)
        resolved_prompt_text = resolved.text
```
This will be extended to also call `has_any_tags()` (which includes `@tag` detection) and the existing `resolve_tags()` will internally handle both patterns.

### Existing Binding Batch-Load Pattern (bindings.py Lines 248-260)
```python
# Bulk fetch actor names and primary refs
actor_ids = [b.actor_id for b in bindings]
actor_result = await session.execute(
    select(Actor).where(Actor.id.in_(actor_ids))
)
actors_by_id = {a.id: a for a in actor_result.scalars().all()}

ref_result = await session.execute(
    select(ActorRef).where(
        ActorRef.actor_id.in_(actor_ids),
        ActorRef.is_primary == True,
    )
)
primary_refs = {r.actor_id: r.image_url for r in ref_result.scalars().all()}
```
This pattern should be reused in `format_binding_registry()` and the summary endpoint.

### Existing format_asset_registry() Pattern (manifest_service.py Lines 798-845)
```python
def format_asset_registry(assets: list[Asset]) -> str:
    lines = ["AVAILABLE ASSETS FOR THIS SCENE:", "━" * 40]
    for asset in assets:
        quality_str = f"{asset.quality_score:.1f}/10" if asset.quality_score is not None else "N/A"
        lines.append(f"[{asset.manifest_tag}] \"{asset.name}\" ({asset.asset_type}, quality: {quality_str})")
        ...
    return "\n".join(lines)
```
The new `format_binding_registry()` follows this same structure but reads from binding+entity tables instead of the Asset table. It should also include `@tag` references in the output so the LLM knows to use `@tag` syntax.

### Summary Endpoint Response Shape
```python
# Each item in the flat list:
{
    "tag": "BRANDON",
    "name": "Brandon Mercer",  # character_name / production_name / library entity name
    "type": "CHARACTER",  # CHARACTER / PROP / SET
    "primary_thumbnail_url": "/api/asset-library/actors/{id}/refs/{ref_id}/image",  # or null
    "description": "tall athletic man in his 30s..."  # base_appearance_prompt / appearance_prompt / reverse_prompt
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manifest Asset tags (`[CHAR_01]`, `[ENV_02]`) | Binding tags (`[CHAR:BRANDON]`, `@brandon`) | Phase 22 | Tags are now human-readable names, not sequential IDs |
| `format_asset_registry()` from Asset table | `format_binding_registry()` from binding tables | This phase | LLM sees named assets with richer metadata |
| `resolve_tags()` for `[TYPE:TAG]` only | Extended to also handle `@tag` cross-type lookup | This phase | User-facing prompts use natural `@tag` syntax |

## Open Questions

1. **Tag collision across binding types**
   - What we know: Tags are unique within each binding type per bible (enforced by UniqueConstraints). But a CastBinding tag "HOTEL" and a SetBinding tag "HOTEL" could coexist.
   - What's unclear: Is this a realistic scenario? The cross-type lookup order (CastBinding first) means CastBinding always wins.
   - Recommendation: Log a warning when `@tag` matches multiple binding types. The first match (CastBinding) wins per the CONTEXT.md decision. This is sufficient for now.

2. **Performance of cross-type lookup for many @tags**
   - What we know: A typical scene prompt might have 2-5 `@tag` references. Each triggers 3 sequential DB queries (one per binding type).
   - What's unclear: Whether this adds noticeable latency.
   - Recommendation: For resolve_tags_with_assets(), pre-load ALL bindings for the bible into memory dicts, then resolve all tags from the dicts. This converts N*3 queries to 3 queries total regardless of tag count.

3. **Should `resolve_tags()` itself be updated, or should `resolve_tags_with_assets()` be a separate function?**
   - What we know: CONTEXT.md calls for a new `resolve_tags_with_assets()` function, and the existing `resolve_tags()` callers should continue working unchanged.
   - Recommendation: Update the existing `resolve_tags()` to detect both patterns (so `has_tags()` becomes `has_any_tags()`) and handle `@tag` resolution. Create `resolve_tags_with_assets()` as a superset that also populates `ResolvedAssetRef` objects. Both functions share the same resolution logic but `resolve_tags_with_assets()` does the extra entity loading for structured metadata.

## Sources

### Primary (HIGH confidence)
- `backend/vidpipe/services/tag_resolver.py` -- existing tag resolution implementation
- `backend/vidpipe/pipeline/storyboard.py` -- existing storyboard pipeline with tag resolution
- `backend/vidpipe/services/manifest_service.py` -- existing format_asset_registry() pattern
- `backend/vidpipe/db/models.py` (lines 629-743) -- CastBinding, SetBinding, PropBinding, SoundBinding models
- `backend/vidpipe/api/bindings.py` -- existing binding CRUD endpoints with batch-load patterns
- `frontend/src/api/types.ts` -- existing TypeScript types for all binding entities
- `frontend/src/api/client.ts` -- existing binding API client functions
- `docs/assets_mapping.md` -- PRD with full architecture specification

### Secondary (MEDIUM confidence)
- `.planning/phases/23-tag-syntax-binding-pipeline-wiring/23-CONTEXT.md` -- locked decisions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing libraries
- Architecture: HIGH -- extending well-established patterns in the codebase with clear precedent
- Pitfalls: HIGH -- identified from direct code reading and understanding existing patterns

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable -- internal codebase patterns, no external dependency risk)
