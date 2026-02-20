# Storyboard Manifest Asset Binding: Problem & Solution Spec

## Problem Statement

When a manifest with CHARACTER assets is attached to a project, the storyboard LLM frequently creates **new character tags** (e.g. `CHAR_04`, `CHAR_05`) instead of using the existing manifest assets (`CHAR_01`, `CHAR_02`, `CHAR_03`). This breaks the entire downstream reference image pipeline — keyframes are generated without face reference photos, face verification is skipped, and the final output bears no resemblance to the provided reference images.

### Observed Failure (Project `ec5f4278`)

- **Manifest**: "Hoyack" — 3 CHARACTER assets (CHAR_01, CHAR_02, CHAR_03), all crops of the same person with distinctive silver-framed sunglasses, fair skin, coral polo
- **Image model**: `gemini-2.5-flash-image` (Vertex AI — supports multimodal reference images)
- **Storyboard output**: Created `CHAR_04` ("Bald adult male mid-50s, dark suit") and `CHAR_05` as `new_asset_declarations` instead of referencing existing manifest tags
- **Result**: All 4 keyframes show a generic AI-generated bald executive with zero facial similarity to the reference photos. `selected_reference_tags` = empty for both scenes.

---

## Root Cause Analysis

The failure cascades through four layers, each silently degrading instead of catching or correcting the tag mismatch.

### Layer 1: Storyboard LLM Prompt (Root Cause)

**File**: `backend/vidpipe/pipeline/storyboard.py` — `ENHANCED_STORYBOARD_PROMPT` (line 88)

The prompt gives the LLM permission to ignore existing assets:

```
Line 103: "Reference registered assets by their [TAG] (e.g., [CHAR_01], [ENV_02])"
Line 109: "You MAY declare new_asset_declarations for assets NOT in the registry"
```

The word "MAY" is permissive. The LLM interprets this as license to create fresh character tags for distinctness, even when the prompt describes characters that clearly correspond to existing manifest assets.

**File**: `backend/vidpipe/services/manifest_service.py` — `format_asset_registry()` (line 755)

The asset registry footer reinforces the permissive behavior:

```
Line 796: "Reference assets by [TAG]. You may declare NEW assets not in the registry."
```

**File**: `backend/vidpipe/pipeline/storyboard.py` — Post-validation (line 286)

When the LLM outputs unrecognized tags, the post-processing only logs a warning:

```python
if placement.asset_tag not in asset_tags_set:
    logger.warning("...unrecognized asset tag '%s' (not in registry, may be declared as new asset)...")
```

No correction, no rejection, no remapping. The invalid tags are persisted to `scene_manifests.manifest_json` as-is.

### Layer 2: Prompt Rewriter — "MUST SELECT" Annotation Missed

**File**: `backend/vidpipe/services/prompt_rewriter.py` — `_list_available_references()` (line 436)

The rewriter lists available reference images and marks placed CHARACTER assets with `"★ PLACED IN SCENE (MUST SELECT)"`. But the marking uses `placed_tags` derived from the scene manifest's placements:

```python
placed_tags = {p["asset_tag"] for p in scene_manifest_json.get("placements", []) if "asset_tag" in p}
# Result: {CHAR_04, CHAR_05} — the INVALID tags

placed_assets = [a for a in with_images if a.manifest_tag in placed_tags]
# Result: [] — EMPTY, because CHAR_04/05 don't exist in the asset registry
```

So CHAR_01/02/03 are listed as available references but **not** annotated as mandatory. The LLM in the rewriter sees them as optional and may or may not select them.

**File**: `backend/vidpipe/services/prompt_rewriter.py` — `_format_placed_assets()` (line 291)

For the invalid tags, the context shows:

```
[CHAR_04] — asset not found in registry
[CHAR_05] — asset not found in registry
```

This tells the LLM that the scene references assets that don't exist, but there's no instruction to substitute with existing manifest assets.

### Layer 3: Post-LLM Enforcement — No-Op

**File**: `backend/vidpipe/pipeline/keyframes.py` (line 539)

The enforcement logic tries to ensure placed CHARACTER assets appear in the selected references:

```python
placed_char_tags = {
    p["asset_tag"]
    for p in placements
    if "asset_tag" in p
    and asset_map.get(p["asset_tag"])       # CHAR_04 → None → FILTERED OUT
    and asset_map[p["asset_tag"]].asset_type == "CHARACTER"
    and asset_map[p["asset_tag"]].reference_image_url
}
# Result: {} (empty set) — enforcement does nothing
```

Since the placements use tags not in the manifest, the filter produces an empty set. The `missing_chars` check finds nothing missing. **Enforcement is silently skipped.**

### Layer 4: Face Verification — Disabled

**File**: `backend/vidpipe/pipeline/keyframes.py` (line 559)

Face verification uses `placed_char_assets` derived from `placed_char_tags`:

```python
placed_char_assets = [asset_map[tag] for tag in placed_char_tags if tag in asset_map]
# Result: [] — no face verification happens
```

The identity retry loop (lines 603-636) only fires when `placed_char_assets` is non-empty. With an empty list, face verification is completely bypassed.

### Cascade Summary

```
Storyboard LLM: creates CHAR_04/05 instead of CHAR_01/02/03    ← ROOT CAUSE
    ↓
Prompt Rewriter: CHAR_01/02/03 listed but NOT marked MUST SELECT ← missed safety net
    ↓
Post-LLM Enforcement: placed_char_tags = {} (empty)              ← no-op
    ↓
Face Verification: placed_char_assets = [] → skipped             ← disabled
    ↓
Image Generation: no reference images sent to Gemini             ← no identity grounding
    ↓
Output: generic AI face, zero similarity to manifest references
```

---

## Proposed Solution

Four-layer defense-in-depth fix. Each layer independently prevents the failure case so the system is resilient even if one layer regresses.

### Fix 1: Storyboard Prompt — Mandate Existing Tags

**Files**: `storyboard.py` (ENHANCED_STORYBOARD_PROMPT), `manifest_service.py` (format_asset_registry)

**Change the prompt instructions** from permissive ("MAY declare new") to mandatory ("MUST use existing"):

```
SCENE MANIFEST INSTRUCTIONS:
When creating scenes, generate a scene_manifest for each scene:
- You MUST use registered asset tags from the Available Assets list for ALL characters
  and environments that match existing assets. Do NOT create new CHARACTER tags when
  a matching character already exists in the registry.
- Reference assets by their exact [TAG] (e.g., [CHAR_01], [ENV_02])
- Use the asset's reverse_prompt for visual detail — it's already optimized for generation
- Assign roles: subject, background, prop, interaction_target, environment
- Specify spatial positions and actions for each placed asset
- Include composition metadata: shot_type, camera_movement, focal_point
- Add continuity_notes describing visual continuity with previous scenes
- new_asset_declarations: ONLY for genuinely new assets that have NO match in the
  registry (e.g., background extras, props, environments not yet registered).
  NEVER declare a new CHARACTER when the registry already has CHARACTER assets
  that could represent that person.
```

**Update the asset registry footer** (`manifest_service.py:796`):

```
Reference assets by [TAG]. You MUST use existing CHARACTER tags — do NOT
create new CHARACTER tags for people already represented in the registry.
You may declare new ENVIRONMENT or PROP assets not in the registry.
```

### Fix 2: Post-Storyboard Tag Remapping

**File**: `storyboard.py` — after `generate_with_retry()` (around line 284)

Add a **deterministic remapping pass** that catches LLM mistakes before persisting scene manifests:

```python
def _remap_unrecognized_tags(scene_manifest, asset_tags_set, manifest_characters):
    """Remap unrecognized CHARACTER tags to existing manifest assets.

    Strategy:
    1. Collect all placement tags not in asset_tags_set
    2. For each unrecognized tag that is a CHARACTER type (from new_asset_declarations
       or heuristic: tag starts with CHAR_), attempt to map to an existing character
    3. Mapping order: match by new_asset_declaration description similarity to
       existing asset reverse_prompt/visual_description, or by positional order
       (first unrecognized CHAR → first manifest CHAR, etc.)
    4. Replace tags in-place in the placements list
    """
```

**Mapping heuristic**: When unrecognized CHARACTER tags are found and the manifest has CHARACTER assets:
- If only 1 manifest CHARACTER and 1 unrecognized CHAR: direct 1:1 map
- If counts match: map by declaration order (CHAR_04 → CHAR_01, CHAR_05 → CHAR_02)
- If counts differ: map first N by order, leave extras as new declarations
- Log every remap at INFO level for debugging

This is a deterministic backstop — it doesn't require another LLM call.

### Fix 3: Prompt Rewriter — Fallback MUST SELECT

**File**: `prompt_rewriter.py` — `_list_available_references()` (line 436)

When `placed_tags` from the scene manifest don't match any assets in the registry, **fall back to marking ALL manifest CHARACTER assets as MUST SELECT**:

```python
placed_assets = [a for a in with_images if a.manifest_tag in placed_tags]

# FALLBACK: if no placed assets resolved but manifest has CHARACTER refs, mark all as MUST SELECT
if not placed_assets:
    placed_assets = [
        a for a in with_images
        if a.asset_type == "CHARACTER" and a.reference_image_url
    ]
    unplaced_assets = [a for a in with_images if a not in placed_assets]
```

This ensures the LLM rewriter sees `★ PLACED IN SCENE (MUST SELECT)` even when storyboard tags were wrong.

### Fix 4: Keyframe Enforcement — Fallback to All Manifest Characters

**File**: `keyframes.py` (line 539)

When `placed_char_tags` resolves to empty but the manifest has CHARACTER assets with reference images, **fall back to using all manifest CHARACTER assets**:

```python
placed_char_tags = {
    p["asset_tag"]
    for p in placements
    if "asset_tag" in p
    and asset_map.get(p["asset_tag"])
    and asset_map[p["asset_tag"]].asset_type == "CHARACTER"
    and asset_map[p["asset_tag"]].reference_image_url
}

# FALLBACK: if scene has placements but none resolved to manifest characters,
# use ALL manifest CHARACTER assets with reference images
if not placed_char_tags and project.manifest_id:
    placed_char_tags = {
        a.manifest_tag
        for a in all_assets
        if a.asset_type == "CHARACTER" and a.reference_image_url
    }
    if placed_char_tags:
        logger.warning(
            f"Scene {scene.scene_index}: no placed chars resolved from scene manifest, "
            f"falling back to all manifest CHARACTER assets: {placed_char_tags}"
        )
```

This ensures:
- Reference images are always sent to the image adapter when the manifest has characters
- Face verification is always active when the manifest has face embeddings
- The identity retry loop fires on low-similarity results

---

## Files Modified

| File | Fix | Change |
|------|-----|--------|
| `backend/vidpipe/pipeline/storyboard.py` | 1, 2 | Tighten ENHANCED_STORYBOARD_PROMPT, add `_remap_unrecognized_tags()` post-processing |
| `backend/vidpipe/services/manifest_service.py` | 1 | Update `format_asset_registry()` footer text |
| `backend/vidpipe/services/prompt_rewriter.py` | 3 | Add fallback MUST SELECT logic in `_list_available_references()` |
| `backend/vidpipe/pipeline/keyframes.py` | 4 | Add fallback to all manifest CHARACTERs when placed_char_tags is empty |
| `backend/vidpipe/schemas/storyboard_enhanced.py` | — | No schema changes needed (new_asset_declarations stays optional) |

---

## Verification Plan

### Test 1: Storyboard Tag Compliance
- Generate a storyboard with a manifest containing CHAR_01, CHAR_02
- Assert: all scene_manifest placements use CHAR_01/CHAR_02, not CHAR_03+ invented tags
- Assert: `new_asset_declarations` contains no CHARACTER-type entries (only ENV/PROP if any)

### Test 2: Remapping Catches LLM Mistakes
- Simulate a storyboard output with CHAR_04/CHAR_05 in placements
- Run `_remap_unrecognized_tags()` with manifest containing CHAR_01/CHAR_02
- Assert: placements are remapped to CHAR_01/CHAR_02
- Assert: `new_asset_declarations` is cleared for the remapped entries

### Test 3: Prompt Rewriter Fallback
- Call `_list_available_references()` with `placed_tags={"CHAR_99"}` (non-existent)
- Assert: output contains `★ PLACED IN SCENE (MUST SELECT)` for all CHARACTER assets

### Test 4: Keyframe Enforcement Fallback
- Simulate keyframe generation with scene manifest placements referencing invalid tags
- Assert: `placed_char_tags` falls back to all manifest CHARACTER assets
- Assert: `ref_image_bytes_list` is populated with reference images
- Assert: face verification retry loop is active

### Test 5: End-to-End Regression
- Run full pipeline with manifest containing 1+ CHARACTER assets
- Assert: keyframe images show facial similarity to reference photos
- Assert: `selected_reference_tags` is populated in `scene_manifests` table
- Assert: `placed_char_assets` is non-empty during keyframe generation (check logs)

### Test 6: No-Manifest Backward Compatibility
- Run pipeline without a manifest
- Assert: storyboard uses original `STORYBOARD_SYSTEM_PROMPT` (no manifest instructions)
- Assert: keyframe generation works as before (no fallback logic triggered)

---

## Design Notes

### Why defense-in-depth (4 layers)?
LLM outputs are fundamentally non-deterministic. Even with strict prompt instructions (Fix 1), the LLM may occasionally create new tags. Each downstream layer provides an independent safety net:
- Fix 1 (prompt): prevents ~90% of cases with better instructions
- Fix 2 (remap): catches the remaining ~10% deterministically
- Fix 3 (rewriter): ensures reference images are selected even if remap misses edge cases
- Fix 4 (enforcement): guarantees reference images reach the image adapter as a last resort

### Why not remove `new_asset_declarations` entirely?
Projects legitimately need new ENVIRONMENT and PROP assets that aren't in the manifest. A manifest might contain only CHARACTER assets, while the storyboard needs to declare server rooms, props, vehicles, etc. The fix restricts *character* tag creation, not all new declarations.

### Why deterministic remapping instead of an LLM call?
The remap in Fix 2 is a simple positional/count-based mapping. An LLM call would add latency, cost, and another failure mode. The deterministic approach is sufficient because:
- Manifests typically have 1-5 CHARACTER assets
- The storyboard LLM creates CHARACTER tags in order (CHAR_04 before CHAR_05)
- A count-match + order-match heuristic handles >95% of cases

### ComfyUI image models (qwen-fast)
Fixes 1-4 benefit all image adapters. Even though ComfyUI models can't use reference images for generation, correct tag binding still enables:
- Accurate `reverse_prompt` injection into the text prompt
- Future ComfyUI workflows that may support IP-Adapter conditioning
- Consistent `scene_manifests` data for debugging and QA
