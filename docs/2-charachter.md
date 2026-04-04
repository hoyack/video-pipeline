# Character Identity & Keyframe Verification Overhaul

Changes since commit `c231210` (wardrobe upload).

---

## 1. Identity Type System

Characters are no longer assumed to be human. A new `identity_type` field (`HUMAN | ANIMAL | CREATURE | OBJECT`) was added throughout the stack:

- **DB model:** `CastBinding.identity_type` column (String(20), default `"HUMAN"`, server_default `'HUMAN'`).
- **Migration:** `ALTER TABLE cast_bindings ADD COLUMN identity_type` with a backfill that sets NULL/empty rows to `"HUMAN"`.
- **API:** `CastBindingCreate` / `CastBindingUpdate` Pydantic models accept `identity_type`. The `/bound-assets-summary` endpoint now returns `identity_type` per asset.
- **Frontend:** `CastingSection` gains an Identity Type dropdown (create + edit forms) with color-coded badges. `types.ts` and `client.ts` carry the new field end-to-end.

### How identity_type flows through the pipeline

1. User sets identity_type on a CastBinding via the Casting UI.
2. `tag_resolver.resolve_tags_with_assets()` propagates `identity_type` on `ResolvedAssetRef`.
3. `keyframes.py` reads identity_type per character tag and branches:
   - **HUMAN:** Face prequalification (InsightFace crop + embedding), face verification gate, strict face-matching thresholds.
   - **Non-human (ANIMAL/CREATURE/OBJECT):** Bypasses face screening entirely. Identity verification uses vision-model-only checks for species, markings, silhouette, and texture fidelity.
4. `_build_identity_instruction()` now generates separate prompt phrasing for non-human subjects ("preserve species, markings, silhouette, texture") vs humans ("preserve facial identity and proportions").

---

## 2. Nano Banana Reference Assembly (complete rewrite)

The old reference resolution logic (scattered across the keyframe loop, CastBinding fallback, and prompt rewriter enforcement) was replaced by a single unified function: `_assemble_nano_banana_reference_context()`.

### What changed

- **Old flow:** Prompt rewriter selected exactly 3 reference tags → post-LLM enforcement inserted placed CHARACTER assets → CastBinding fallback tried scene.prompt → prequalification happened on flat URL list.
- **New flow:**
  1. Collect mandatory character tags from shot manifest placements + `shot.characters_present`.
  2. Canonicalize tags via `canonicalize_character_tags()` (handles LLM typos).
  3. For each mandatory character, gather face refs, wardrobe refs, and fallback refs as typed `_ReferenceCandidate` objects.
  4. Round-robin pack mandatory candidates into a 13-image budget (`_NANO_BANANA_MAX_REFERENCE_IMAGES`), then fill remaining slots with optional/supplemental refs from the prompt rewriter.
  5. Prompt rewriter `selected_reference_tags` is now "up to 3 supplemental tags" rather than "exactly 3 mandatory."

### New dataclasses

| Dataclass | Purpose |
|-----------|---------|
| `_ReferenceCandidate` | One reference image with tag, bytes, asset_type, source, identity_type, reference_kind (face/wardrobe/supplemental) |
| `_NanoBananaReferenceContext` | Full assembled context: ref bytes list, final tag order, mandatory/optional splits, identity types, trimmed counts, canonical remaps |
| `_CharacterVerificationTarget` | Per-character verification target with expected position and grouped ref candidates |
| `_CharacterCropSelection` | Per-character crop from YOLO detection with bbox and full-frame fallback flag |
| `_CharacterCropPlan` | Full detection plan with face/person/object counts |

---

## 3. Keyframe Verification System (new)

Replaces the old single-pass `_verify_keyframe_faces()` with a multi-layer verification pipeline.

### Architecture

```
Generated keyframe
    │
    ├── YOLO object detection (person/non-person crops)
    │       └── _select_character_candidate_boxes()
    │
    ├── Vision model verification (per-character)
    │       └── _verify_keyframe_characters_with_vision()
    │           ├── Compose verification board (crop + refs side-by-side)
    │           ├── LLM structured output (CharacterKeyframeVerificationOutput)
    │           └── Partial-visibility human check (back-turned tolerance)
    │
    └── Face embedding verification (HUMAN characters only)
            └── _verify_target_face()
                ├── InsightFace embedding extraction from crop
                └── Cosine similarity against reference embeddings
```

### Verification modes

- **`strict_face_and_vision`**: Single human target, clean detection. Both face embedding AND vision model must pass.
- **`vision_primary_face_advisory`**: Multiple humans, crowded frame, or full-frame fallback. Face check becomes advisory (logged but non-blocking). Vision model is the gate.

### Retry loop changes

- Each attempt is stored as `_GeneratedKeyframeAttempt` (bytes + full verification report).
- On verification failure, `_build_retry_correction_prompt()` generates specific remediation instructions (IDENTITY LOCK, WARDROBE LOCK, visibility fixes) prepended to the next attempt's prompt.
- Reference candidates narrow on retry (`_build_retry_reference_candidates`): level 0 = full pack, level 1 = character-only, level 2 = minimal cast pack.
- **Best-effort fallback:** If all attempts fail, `_select_best_effort_attempt()` picks the best attempt by a composite score (visible count → passed count → identity score → wardrobe score → face similarity → fewest issues). The keyframe is saved with `verification_status = "accepted_with_warnings"`.
- **Transport exhaustion:** `RetryError` from tenacity is caught separately. If image generation itself fails after all transport retries, the system still falls back to best-effort from prior attempts before raising.

### New Pydantic schema

`CharacterKeyframeVerificationOutput` in `schemas/llm_vision.py`:
- `passed`, `character_visible`, `identity_match`, `wardrobe_match` (booleans)
- `identity_score`, `wardrobe_score` (0.0–10.0)
- `issues` (list of concrete mismatch descriptions)

### Keyframe metadata persistence

Three new columns on the `Keyframe` model:
- `verification_status` (VARCHAR(64)): "passed", "failed", "accepted_with_warnings", "inherited", "verification_skipped"
- `verification_attempts` (INTEGER)
- `verification_summary` (TEXT): Full verification detail string

These are exposed through the API on `ShotDetail` as `start_verification_status`, `start_verification_attempts`, `start_verification_summary` (and matching `end_*` fields).

---

## 4. Identity Policy Filtering

`_apply_identity_policy_to_reference_candidates()` applies per-identity-type filtering to the selected reference candidates before image generation:

- **HUMAN face refs** → run through `prequalify_refs()` to extract face crops and embeddings. Only refs with a detectable face survive. The face crop replaces the full image for generation input.
- **HUMAN wardrobe refs** → passed through unchanged (full-body clothing refs).
- **Non-human refs** → passed through unchanged (no face screening).

The `QualifiedRef` dataclass in `ref_prequalification.py` now includes `face_crop_bytes` alongside the full image bytes and embedding.

---

## 5. CV Detection Enhancements

`CVDetectionService.detect_objects_and_faces_from_bytes()` added: runs YOLO detection on raw image bytes (no file path needed). Returns both object detections and synthetic face bounding boxes (top 40% of person bboxes).

New dependencies in `pyproject.toml`:
- `ultralytics>=8.3.0` (YOLOv8)
- `insightface>=0.7.3` (ArcFace)
- `onnxruntime-gpu>=1.18.0`

Docker: `docker-compose.yml` now passes `gpus: all` and `NVIDIA_VISIBLE_DEVICES=all` to the backend container.

---

## 6. Tag Canonicalization

### Problem
The screenwriter agent and prompt rewriter sometimes produce character tags that don't exactly match the Production Bible CastBinding tags (e.g., `FRANK_JR` vs `FRANK_JR_UNDERWOOD`).

### Solution
New `canonicalize_character_tags()` function in `tag_resolver.py`:
- Queries CastBinding and CastLook tags for the production bible.
- Uses `SequenceMatcher` ratio + shared prefix token count for fuzzy matching.
- Requires ≥90% sequence similarity AND shared prefix tokens ≥2 (for multi-token tags) AND clear winner (gap ≥0.03 over second-best).
- Applied in: `_assemble_nano_banana_reference_context`, `_resolve_at_tag_cross_type`, `resolve_tags_with_assets`, and `validate_screenplay`.

`_find_character_tag_alias()` is the core matching function. It's intentionally conservative — only remaps when there's a single clearly-better candidate.

---

## 7. Screenwriter Agent: Dynamic Shot Count

### New capability
Scenes can now use `dynamic_shot_count=true` to let the screenwriter agent choose the optimal number of shots instead of a fixed count.

### Changes
- `Scene.dynamic_shot_count` column (Boolean, default false).
- `CreateSceneRequest` gains `dynamic_shot_count` field.
- `_shot_count_instruction()` builds a range-based instruction ("Use between N and M shots, aiming for about R") based on story beat count.
- `_dynamic_shot_budget()` computes recommended/max shots from beat count.
- `_needs_dynamic_retry()` detects over-compressed breakdowns (e.g., 1 shot for 3+ beats) and triggers an automatic retry with stricter instructions.
- `total_duration` is recalculated when dynamic shot count changes the actual shot count.

---

## 8. Checkpoint Service Improvements

- **More fields captured:** `dynamic_shot_count`, `script_analysis`, `screenplay_context`, `style_guide`, `storyboard_raw`, `error_message` on scenes. `generation_status`, `characters_present`, `beat_index`, `narrative_intent`, `emotional_weight` on shots.
- **JSON compatibility:** New `json_compat.py` module with `normalize_json_value/dict/list` handles TEXT-stored JSON in older SQLite databases (parses stringified JSON, passes native dicts/lists through).
- **PostgreSQL migration:** `_promote_json_text_columns()` converts `scenes.script_analysis` and `shots.characters_present` from TEXT to JSONB.
- **Diff computation:** `compute_diff()` now tracks the additional shot fields.

---

## 9. Tag Resolver Enhancements

- `ResolvedAssetRef` gains `face_reference_image_urls`, `wardrobe_reference_image_urls`, and `identity_type` fields.
- **Look resolution:** `_build_look_resolution()` now interleaves wardrobe preset refs with base actor refs for HUMAN characters (so the model sees both the face and the outfit). Non-human looks only get wardrobe refs.
- **Ordered queries:** `ActorRef`, `LibraryPropRef`, `LibrarySetRef` queries now `ORDER BY is_primary DESC, created_at ASC` for deterministic reference ordering.
- **Cross-type resolution** returns a third value `canonical_tag` for alias tracking.

---

## 10. Frontend Changes

### Scene Detail
- Edit button always visible (not just for terminal scenes).
- Auto-enters edit mode for non-terminal (in-progress) scenes.

### Shot Cards (ShotCard, EditableShotCard, ShotEditorCard)
- Display keyframe source badges ("Generated" / "Inherited").
- Show verification metadata (status, attempts, summary) from new API fields.

### Edit Mode Overlay
- Disables drag-and-drop reordering during active generation (uses canonical shot order).
- Falls back to `ShotEditorCard` (non-sortable) when reordering is disabled.

### Library Set Detail
- Fixes null vs undefined handling for optional fields in `updateLibrarySet()`.

### Prompt Rewriter
- System prompt updated: "exactly 3 reference asset tags" → "up to 3 supplemental reference asset tags." Mandatory character refs are now handled by the Nano Banana path, not the rewriter.

---

## 11. Pipeline Error Handling

- New `_mark_scene_failed()` helper persists `status="failed"` and `error_message` to the scene.
- All chained regeneration phases (`_run_storyboard_regeneration`, `_run_keyframes_regeneration`, `_run_clips_regeneration`, `_run_all_phases_regeneration`) now call `_mark_scene_failed()` on exception when `_emit_complete=False` (chained mode).
- Status restoration (`scene.status = saved_status`) is conditional: only restores for standalone regeneration (`_emit_complete=True`), not chained.

---

## 12. New Tests

| Test file | Coverage |
|-----------|----------|
| `test_cast_binding_identity_type.py` | identity_type field on CastBinding CRUD |
| `test_checkpoint_service.py` | Snapshot build/restore with new fields, JSON compat |
| `test_nano_banana_refs.py` | Reference assembly, packing, deduplication, retry candidates, identity policy |
| `test_scene_detail_sources.py` | Keyframe source/verification fields in scene detail API |
| `test_screenwriter_agent.py` | Dynamic shot count, tag canonicalization, retry logic |
