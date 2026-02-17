# Phase 12: Fork System Integration with Manifests - Research

**Researched:** 2026-02-16
**Domain:** Fork inheritance for Asset Registry, manifest system, and incremental manifesting
**Confidence:** HIGH

---

## Summary

Phase 12 is the final V2 milestone phase. Its purpose is to make forked projects inherit the full manifest ecosystem that phases 4-11 built: the Asset Registry, project manifest, scene manifests, and reference uploads. The fork system already handles scene-level inheritance (keyframes, clips, storyboard data), but it knows nothing about assets, manifest_id, or scene manifest records.

This phase is primarily a database integration and backend extension task, not a new technology problem. Every technology is already in use. The work involves: adding three columns to the `assets` table, extending the `fork_project` endpoint to copy assets, extending `_compute_invalidation` to account for asset modifications, adding new fork request fields for asset changes (`modified_assets`, `removed_asset_ids`, `new_uploads`), running incremental manifesting when new uploads are present, inheriting scene manifest rows for unchanged scenes, and extending EditForkPanel in the frontend to show asset controls with lock/edit/remove states.

The most complex piece is the incremental manifesting path: when a user adds new reference uploads in a fork, only those new uploads go through YOLO → ArcFace → reverse-prompting, face-matched against all inherited assets. This reuses the existing `ManifestingEngine` but calls it scoped to only new uploads and passes inherited embeddings for cross-matching.

**Primary recommendation:** Implement in four sequential tasks: (1) DB schema migration adding inheritance fields to assets, (2) backend fork endpoint extension with asset copy + scene manifest inheritance + incremental manifesting, (3) frontend ForkRequest type extension and API wiring, (4) EditForkPanel asset section UI.

---

## Standard Stack

### Core (all already installed, no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy async | 2.0 | ORM for asset copy operations | Already used throughout |
| FastAPI | current | Fork endpoint extension | Already the API framework |
| React + TypeScript | current | EditForkPanel extension | Already the frontend stack |
| ManifestingEngine | internal | Incremental manifesting for new uploads | Built in Phase 5 |
| manifest_service | internal | Asset CRUD | Built in Phase 4 |

### No New Dependencies

Phase 12 adds NO new libraries. All operations use existing services:
- Asset copying uses SQLAlchemy + existing `manifest_service`
- Incremental manifesting uses existing `ManifestingEngine.process_manifest()`
- Face cross-matching uses existing `face_matching.py` with ArcFace embeddings
- Frontend uses existing component patterns

---

## Architecture Patterns

### Recommended Task Structure

```
12-01: DB schema + Asset model inheritance fields
12-02: Backend fork endpoint extension (asset copy, scene manifest inheritance, incremental manifesting)
12-03: Frontend ForkRequest + API types
12-04: EditForkPanel asset section
```

### Pattern 1: Asset Inheritance on Fork

Assets belong to manifests (`assets.manifest_id`), NOT to projects. The fork inherits the manifest reference (`project.manifest_id`), which gives access to all manifest assets. However, the design calls for per-fork asset modifications (edit/remove), which means the fork needs its own asset copies that can diverge from the shared manifest.

**Key architectural decision:** Fork creates new `Asset` rows with `is_inherited=True` pointing back to parent via `inherited_from_asset`. GCS URLs are shared (no file copying). This allows fork-specific overrides (edited reverse_prompt, removed assets) without mutating the parent manifest.

```python
# Source: docs/v2-pipe-optimization.md Section 13 "Backend Fork Processing"
async def _copy_assets_for_fork(session, source_manifest_id, new_project_id, fork_request):
    parent_assets = await session.execute(
        select(Asset).where(Asset.manifest_id == source_manifest_id)
    )
    for asset in parent_assets.scalars().all():
        if str(asset.id) in fork_request.removed_asset_ids:
            continue  # Skip removed assets

        new_asset = Asset(
            manifest_id=source_manifest_id,  # Shares the manifest
            asset_type=asset.asset_type,
            name=asset.name,
            manifest_tag=asset.manifest_tag,
            reference_image_url=asset.reference_image_url,  # Shared GCS URL, no copy
            reverse_prompt=asset.reverse_prompt,
            visual_description=asset.visual_description,
            face_embedding=asset.face_embedding,
            clip_embedding=asset.clip_embedding,
            quality_score=asset.quality_score,
            is_inherited=True,
            inherited_from_asset=asset.id,
            inherited_from_project=source_project_id,
        )
        # Apply modifications if user edited this asset
        if str(asset.id) in fork_request.modified_assets:
            changes = fork_request.modified_assets[str(asset.id)]
            if "reverse_prompt" in changes:
                new_asset.reverse_prompt = changes["reverse_prompt"]
            if "reference_image" in changes:
                # Upload new image, re-run reverse-prompting
                new_asset.reference_image_url = await upload_new_image(changes["reference_image"])
                new_asset.reverse_prompt = await reverse_prompt_service.run(new_asset.reference_image_url)
                new_asset.is_inherited = False  # Modified = no longer pure inheritance
        session.add(new_asset)
```

### Pattern 2: Scene Manifest Inheritance

Scene manifests use composite PK `(project_id, scene_index)`. For unchanged scenes, copy the row to the new project. For invalidated scenes, do NOT copy — leave blank for storyboarding to regenerate.

```python
# Source: docs/v2-pipe-optimization.md Section 13 "Fork Inheritance Rules"
async def _copy_scene_manifests(session, source_project_id, new_project_id, invalidation_point):
    parent_manifests = await session.execute(
        select(SceneManifest).where(SceneManifest.project_id == source_project_id)
    )
    for sm in parent_manifests.scalars().all():
        if sm.scene_index < invalidation_point:
            # Unchanged scene — inherit manifest
            new_sm = SceneManifest(
                project_id=new_project_id,
                scene_index=sm.scene_index,
                manifest_json=sm.manifest_json,
                composition_shot_type=sm.composition_shot_type,
                composition_camera_movement=sm.composition_camera_movement,
                asset_tags=sm.asset_tags,
                selected_reference_tags=sm.selected_reference_tags,
                rewritten_keyframe_prompt=sm.rewritten_keyframe_prompt,
                rewritten_video_prompt=sm.rewritten_video_prompt,
            )
            session.add(new_sm)
        # Invalidated scenes: no copy, storyboard regenerates manifest
```

### Pattern 3: Asset-Modification Invalidation

When a user modifies an asset (edits `reverse_prompt`, swaps reference image), scenes using that asset must be invalidated starting from the first scene that asset appears in. The existing `_compute_invalidation` function must be extended.

```python
# Extension to existing _compute_invalidation in routes.py
def _compute_asset_invalidation(
    session,
    source_project_id: uuid.UUID,
    modified_asset_ids: list[str],
    scene_manifests: list[SceneManifest],
) -> int:
    """Find earliest scene that uses any modified asset.

    Returns scene_index of earliest invalidation, or scene_count if none.
    """
    earliest = float("inf")
    for sm in scene_manifests:
        if sm.asset_tags:
            # Check if any modified asset's tag appears in this scene's manifest
            # Need to map asset_id -> manifest_tag first
            for tag in sm.asset_tags:
                if tag in modified_asset_tags:  # modified_asset_tags built from modified_asset_ids
                    earliest = min(earliest, sm.scene_index)
    return int(earliest)
```

### Pattern 4: Incremental Manifesting for New Uploads

When new reference images are added in the fork, the incremental manifesting path processes ONLY new uploads. Critically, face cross-matching must include inherited assets (to detect if a new upload matches an existing character).

```python
# Source: docs/v2-pipe-optimization.md Section 13
async def _run_incremental_manifesting(
    session,
    manifest_id: uuid.UUID,
    new_uploads: list[NewUpload],
    inherited_assets: list[Asset],  # For face cross-matching
):
    engine = ManifestingEngine(session)
    # Process only new uploads through full pipeline
    # Pass inherited_assets so ArcFace can cross-match faces
    await engine.process_new_uploads(
        manifest_id=manifest_id,
        new_uploads=new_uploads,
        existing_face_embeddings=[
            (a.id, a.face_embedding)
            for a in inherited_assets
            if a.face_embedding is not None
        ],
    )
```

### Pattern 5: ForkRequest Extension

The existing `ForkRequest` Pydantic schema must be extended with the asset changes fields:

```python
# Backend: routes.py ForkRequest
class ModifiedAsset(BaseModel):
    """Asset modification in a fork."""
    changes: dict  # {"reverse_prompt": "...", "name": "...", "reference_image": base64}

class NewUpload(BaseModel):
    """New reference image to add in fork."""
    image_data: str  # base64
    name: str
    asset_type: str
    description: Optional[str] = None
    tags: Optional[list[str]] = None

class ForkRequest(BaseModel):
    # ... existing fields unchanged ...
    # NEW:
    asset_changes: Optional["AssetChanges"] = None

class AssetChanges(BaseModel):
    modified_assets: dict[str, ModifiedAsset] = Field(default_factory=dict)
    removed_asset_ids: list[str] = Field(default_factory=list)
    new_uploads: list[NewUpload] = Field(default_factory=list)
```

### Anti-Patterns to Avoid

- **Duplicating GCS files for inherited assets:** GCS URLs are shared. No file copy. Only new reference images for swapped assets get uploaded.
- **Running full manifesting for inherited assets:** Only new uploads go through YOLO + reverse-prompting. Inherited assets carry their embeddings forward at $0 cost.
- **Mutating the parent manifest:** Fork creates new Asset rows; it does NOT modify the parent manifest or its assets.
- **Invalidating all scenes when adding new uploads:** Adding new reference images DOES NOT invalidate any scenes (it just expands the asset pool). Only modifying or removing existing assets triggers invalidation.
- **Forgetting to inherit scene manifests:** The existing fork code copies keyframes and clips but does NOT copy scene_manifest rows. Phase 12 adds this.
- **Allowing forks of non-terminal projects:** Existing validation already enforces this; do not relax it.

---

## What's Already Built

Understanding the existing codebase is critical for planning — most of the infrastructure is in place.

### Existing fork infrastructure (routes.py)
- `_compute_invalidation(source, overrides, scene_edits, deleted_scenes, clear_keyframes)` — computes `(resume_stage, scene_copy_boundary)`
- `fork_project()` endpoint — copies Project row, copies Scenes, copies Keyframes, copies VideoClips, updates storyboard_raw
- Existing `ForkRequest` schema with prompt/style/scene edits
- Asset inheritance tracking pattern: `Keyframe.source = "inherited"` and `VideoClip.source = "inherited"` already exists

### Existing manifest infrastructure
- `Manifest`, `Asset`, `SceneManifest`, `ManifestSnapshot` ORM models
- `manifest_service` with CRUD operations
- `ManifestingEngine` with YOLO + ArcFace + reverse-prompting
- `face_matching.py` with ArcFace embedding comparison
- `Project.manifest_id` FK already set at generate-time
- `SceneManifest` has composite PK `(project_id, scene_index)`

### What Phase 12 adds
1. DB migration: `assets` table needs `is_inherited BOOLEAN DEFAULT FALSE`, `inherited_from_asset UUID`, `inherited_from_project UUID`
2. ORM model: `Asset` class needs these three fields
3. Fork endpoint: copy assets, copy scene manifests, handle `asset_changes`
4. `_compute_invalidation`: extend to accept asset modification info
5. `ForkRequest` + `AssetChanges` schemas
6. `ManifestingEngine` method: `process_new_uploads()` with inherited embeddings
7. Frontend `ForkRequest` type extension
8. EditForkPanel: asset registry section

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reverse-prompting for swapped images | Custom Gemini call | `ManifestingEngine` or `reverse_prompt_service.reverse_prompt_asset()` | Already implemented in Phase 5 |
| Face cross-matching new vs inherited | Custom ArcFace logic | `face_matching.py` — already handles multi-embedding comparison | Built and tested in Phase 5 |
| Asset CRUD in fork | Direct SQL | `manifest_service` functions | Handles tag uniqueness, asset_count tracking |
| Scene manifest copying | JSON merging | SQLAlchemy ORM copy with new project_id | Avoids serialization bugs |
| Invalidation point for asset edits | Fresh algorithm | Extend `_compute_invalidation()` with asset path | Keeps all invalidation logic in one place |

**Key insight:** Every custom problem in this phase has already been solved by earlier phases. The pattern is "wire existing services together" not "build new services."

---

## Common Pitfalls

### Pitfall 1: Asset Scope Confusion

**What goes wrong:** Assets in this system belong to MANIFESTS (not projects). A fork project can share the same `manifest_id` as the parent. But if user modifications diverge assets, the fork may need its own asset rows pointing to the same manifest.

**Why it happens:** The v2-manifest.md design doc describes assets as manifest-scoped. But the fork system needs per-fork asset overrides without affecting the parent manifest.

**How to avoid:** Create new Asset rows for each asset the user modifies or adds. Unmodified inherited assets can be left as-is in the shared manifest (the fork simply references the same `manifest_id`). Only create new rows for explicitly changed assets.

**Alternative approach:** If even simpler, the fork inherits the manifest_id and manifest_version — inherited assets are READ-ONLY from the manifest, while modifications create a fork-specific "asset overlay" tracked separately. This avoids new asset rows entirely for pure read scenarios.

**Warning signs:** If you're creating new Asset rows for ALL assets (even unmodified ones), you've over-engineered it.

### Pitfall 2: Scene Manifest Not Copied

**What goes wrong:** The existing fork code copies keyframes and clips but DOES NOT copy `scene_manifest` rows. After forking, the new project has no scene manifests for its inherited scenes, causing prompt-rewriting to fall back to unoptimized prompts.

**Why it happens:** `scene_manifests` was added in Phase 7, after the core fork logic was written. The fork code was never updated.

**How to avoid:** After copying scenes, copy `SceneManifest` rows for all scenes with index < `scene_boundary`.

**Warning signs:** Scene manifests are empty for inherited scenes in a fork.

### Pitfall 3: Incremental Manifesting Tag Collision

**What goes wrong:** New uploads in a fork get auto-assigned manifest tags (CHAR_01, CHAR_02...). If the inherited manifest already has CHAR_01 through CHAR_04, the new upload gets CHAR_01 again — collision.

**Why it happens:** The existing `manifest_service.create_asset()` generates tags by counting existing assets of that type. If inherited assets aren't considered, numbering restarts.

**How to avoid:** When running incremental manifesting, pass the max existing tag numbers so new tags continue the sequence. Tag assignment must query ALL assets in the manifest (inherited + new) before numbering.

**Warning signs:** Two assets with the same `manifest_tag` in the same manifest.

### Pitfall 4: Asset Modification Does Not Invalidate Correctly

**What goes wrong:** User modifies CHAR_01's reverse_prompt. Only scenes that contain CHAR_01 in their `scene_manifests.asset_tags` should be invalidated. If `asset_tags` is null or empty for some scene manifests, no invalidation occurs even though the scene uses that asset.

**Why it happens:** Scene manifests may not have `asset_tags` populated if storyboarding didn't record which assets were used (populated in Phase 7).

**How to avoid:** Fall back to invalidating from scene 0 if asset usage can't be determined from scene manifests. Safe (if costly) over failing silently.

**Warning signs:** After modifying CHAR_01, all scenes are regenerated instead of only the affected ones — this is the safe fallback, not a bug.

### Pitfall 5: New Uploads and Fork Asset_Changes on Non-Manifest Projects

**What goes wrong:** User tries to add new reference uploads via fork on a project that has NO manifest (generated without one). The fork endpoint receives `asset_changes.new_uploads` but `source.manifest_id` is None.

**Why it happens:** Not all projects use manifests (manifest is optional in GenerateRequest).

**How to avoid:** Validate in the fork endpoint: if `asset_changes` is provided and `source.manifest_id` is None, either auto-create a manifest for the fork or return 422 with a clear message.

**Warning signs:** NullPointerException / attribute error on `source.manifest_id`.

### Pitfall 6: EditForkPanel Loads Manifest Assets Lazily

**What goes wrong:** EditForkPanel currently receives `detail: ProjectDetail` which does NOT include the manifest's assets. The UI needs to fetch assets separately to display them with lock/edit/remove controls.

**Why it happens:** ProjectDetail doesn't include nested manifest data (only `project_id`, not the manifest's asset list).

**How to avoid:** EditForkPanel fetches manifest assets on mount using `GET /api/manifests/{manifest_id}/assets`. This requires knowing the `manifest_id` — need to add `manifest_id` to `ProjectDetail` response.

---

## Code Examples

### DB Migration for Asset Inheritance Fields

```sql
-- migrate_phase12.sql
-- Add fork inheritance tracking to assets table
ALTER TABLE assets ADD COLUMN is_inherited BOOLEAN DEFAULT FALSE;
ALTER TABLE assets ADD COLUMN inherited_from_asset TEXT;  -- UUID stored as TEXT in SQLite
ALTER TABLE assets ADD COLUMN inherited_from_project TEXT;  -- UUID stored as TEXT in SQLite
```

### ORM Model Extension

```python
# backend/vidpipe/db/models.py — Asset class additions
class Asset(Base):
    # ... existing fields ...

    # Phase 12: Fork inheritance tracking
    is_inherited: Mapped[bool] = mapped_column(Boolean, default=False)
    inherited_from_asset: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("assets.id"), nullable=True
    )
    inherited_from_project: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("projects.id"), nullable=True
    )
```

### ProjectDetail Needs manifest_id

```python
# routes.py — ProjectDetail schema needs manifest_id for EditForkPanel
class ProjectDetail(BaseModel):
    # ... existing fields ...
    manifest_id: Optional[str] = None  # NEW: needed by EditForkPanel to fetch assets
```

### Frontend ForkRequest Extension

```typescript
// frontend/src/api/types.ts

/** Asset modification in a fork */
export interface ModifiedAsset {
  changes: {
    reverse_prompt?: string;
    name?: string;
    reference_image?: string;  // base64
  };
}

/** New reference upload to add in fork */
export interface NewForkUpload {
  image_data: string;  // base64
  name: string;
  asset_type: string;
  description?: string;
  tags?: string[];
}

/** Asset changes for fork request */
export interface AssetChanges {
  modified_assets?: Record<string, ModifiedAsset>;  // asset_id -> changes
  removed_asset_ids?: string[];
  new_uploads?: NewForkUpload[];
}

/** Updated ForkRequest */
export interface ForkRequest {
  // ... existing fields unchanged ...
  asset_changes?: AssetChanges;  // NEW
}
```

### EditForkPanel Asset Section Pattern

```typescript
// Pattern for asset state in EditForkPanel
const [modifiedAssets, setModifiedAssets] = useState<Record<string, ModifiedAsset>>({});
const [removedAssetIds, setRemovedAssetIds] = useState<Set<string>>(new Set());
const [newUploads, setNewUploads] = useState<NewForkUpload[]>([]);

// Asset status helper
function getAssetStatus(assetId: string): "locked" | "modified" | "removed" {
  if (removedAssetIds.has(assetId)) return "removed";
  if (modifiedAssets[assetId]) return "modified";
  return "locked";
}
```

---

## Open Questions

1. **Should fork create new Asset rows for ALL inherited assets, or only for modified ones?**
   - What we know: Unmodified assets are accessible through the shared `manifest_id`. Modified assets need their own rows to avoid mutating the parent.
   - What's unclear: Does the UI/backend ever need to distinguish "fork's version of CHAR_01" from "parent's CHAR_01" for unmodified assets?
   - Recommendation: Copy ALL assets to fork (with `is_inherited=True`) for simplicity. This avoids the "does this asset belong to the fork or the parent?" ambiguity. The $0 cost tracking is clean. The row count overhead is trivial.

2. **What happens when parent project has no manifest (manifest_id is None)?**
   - What we know: `GenerateRequest.manifest_id` is optional — some projects have no manifest.
   - What's unclear: Should fork with `asset_changes` on a non-manifest project auto-create a manifest?
   - Recommendation: Return 422 if `asset_changes` is provided but `source.manifest_id` is None. Keep the feature scoped to manifest-enabled projects.

3. **Where does the incremental manifesting run — in the fork endpoint or in the pipeline?**
   - What we know: The pseudocode in v2-pipe-optimization.md runs it directly in `_process_fork`. But the pipeline already manages manifesting as a state.
   - What's unclear: Should new uploads trigger a "manifesting" pipeline state in the fork, or be processed synchronously during the fork request?
   - Recommendation: Process new uploads synchronously during fork creation (before returning the fork response). It's a small number of images. This avoids a two-phase fork where the frontend must poll for manifesting completion before proceeding.

4. **Does `ProjectDetail` need `manifest_id` in the response?**
   - What we know: EditForkPanel receives `ProjectDetail` but needs to fetch manifest assets.
   - What's unclear: Currently `manifest_id` is not in the `ProjectDetail` response schema.
   - Recommendation: Add `manifest_id: Optional[str]` to `ProjectDetail`. This is a non-breaking additive change. Required for EditForkPanel to fetch assets.

---

## Sources

### Primary (HIGH confidence)
- `/home/ubuntu/work/video-pipeline/docs/v2-manifest.md` — Section 13: Fork System design, fork request schema, backend pseudocode
- `/home/ubuntu/work/video-pipeline/docs/v2-pipe-optimization.md` — Section 13: Fork Inheritance Rules, Asset Schema with inheritance fields, fork processing pseudocode
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/models.py` — Confirmed existing ORM models, confirmed missing `is_inherited`/`inherited_from_*` fields on Asset
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/routes.py` — Confirmed existing `_compute_invalidation()`, `fork_project()`, `ForkRequest` schema, and full fork copy logic
- `/home/ubuntu/work/video-pipeline/frontend/src/components/EditForkPanel.tsx` — Confirmed current EditForkPanel structure and `buildForkRequest()` pattern
- `/home/ubuntu/work/video-pipeline/frontend/src/api/types.ts` — Confirmed existing TypeScript types, confirmed `ForkRequest` lacks `asset_changes`
- `/home/ubuntu/work/video-pipeline/backend/migrate_phase10.sql` — Confirmed migration pattern for adding columns (ALTER TABLE + ADD COLUMN)

### Secondary (MEDIUM confidence)
- `docs/v2-manifest.md` Section 13 UI wireframe — Shows lock/edit/remove icons per asset in EditForkPanel
- `docs/v2-manifest.md` Section 13 Fork Request schema — JavaScript object structure for `asset_changes`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new tech, all existing services
- Architecture: HIGH — detailed pseudocode in design docs, confirmed against actual codebase
- Pitfalls: HIGH — identified from codebase examination (missing fields, missing scene manifest copy)
- Schema gaps: HIGH — confirmed by grep that `is_inherited`, `inherited_from_asset`, `inherited_from_project` do NOT exist in models.py

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (30 days, stable domain)

---

## Key Findings Summary

1. **Asset inheritance fields are missing from the ORM model.** The `Asset` class in `models.py` has NO `is_inherited`, `inherited_from_asset`, or `inherited_from_project` columns. These are only in the design docs. Phase 12 must add the DB migration and ORM fields.

2. **Scene manifests are NOT copied in the existing fork code.** The `fork_project()` endpoint copies Projects, Scenes, Keyframes, and VideoClips — but NOT `SceneManifest` rows. This is a gap Phase 12 fills.

3. **`ProjectDetail` doesn't expose `manifest_id`.** EditForkPanel needs the manifest_id to fetch assets. The `ProjectDetail` response schema must be extended.

4. **The `_compute_invalidation` function needs an asset path.** Currently handles prompt/model/scene edits. Phase 12 adds a new path: when an asset is modified, invalidate scenes using that asset.

5. **Incremental manifesting already works** — `ManifestingEngine` is fully implemented. Phase 12 just needs to call it from the fork code path with inherited embeddings for face cross-matching.

6. **No new dependencies required.** This phase is purely integration of existing services.
