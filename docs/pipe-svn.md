# PipeSVN: Project Version Control & In-Place Editing

## Overview

Today, the only way to modify a completed project is **Edit & Fork**, which creates an entirely new project row. There is no way to edit a project in-place, no change history, and no way to revert. This spec introduces:

1. **In-place project editing** — modify scenes, keyframes, clips, prompts, and manifest bindings on the existing project without forking.
2. **SHA-1 checkpoint system** — every committed edit creates an immutable, content-addressed snapshot of the full project state, forming a linear version chain.
3. **Changelog & revert** — a detailed, per-field changelog with the ability to revert to any previous checkpoint.
4. **Separated Fork / Edit actions** — "Edit & Fork" becomes two distinct buttons: **Fork** (copy to new project) and **Edit** (modify in-place with version tracking).
5. **Rich manifest-tag text integration** — inline `[TAG]` references in scene text fields that resolve to manifest assets, with autocomplete and visual decoration.
6. **Prompt transparency** — surface the actual prompts sent to image and video models, editable before regeneration.

---

## 1. Checkpoint System (PipeSVN Core)

### 1.1 Data Model

#### `project_checkpoints` table

| Column | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `project_id` | FK → projects.id | The project this checkpoint belongs to |
| `sha` | String(40) | SHA-1 hash of the serialized snapshot content |
| `parent_sha` | String(40), nullable | SHA of the previous checkpoint (`null` for the initial checkpoint) |
| `snapshot_data` | JSON | Full serialized project state (see 1.2) |
| `message` | Text | Human-readable commit message (auto-generated or user-provided) |
| `metadata` | JSON, nullable | Extra context: `{ source: "auto" \| "manual", fields_changed: [...], scenes_affected: [...] }` |
| `created_at` | DateTime | Timestamp of the commit |

**Indexes:** `(project_id, sha)` unique, `(project_id, created_at)` for log queries.

**Chain structure:** `parent_sha` forms a singly-linked list from newest → oldest. The initial checkpoint (created when the project first reaches `complete` status) has `parent_sha = null`.

#### New column on `projects`

| Column | Type | Description |
|---|---|---|
| `head_sha` | String(40), nullable | Points to the current (latest) checkpoint SHA. `null` means no checkpoints yet (pre-edit project). |

### 1.2 Snapshot Content & SHA Computation

A checkpoint snapshot captures the full project state needed to restore it:

```json
{
  "version": 1,
  "project": {
    "title": "...",
    "prompt": "...",
    "style": "...",
    "aspect_ratio": "16:9",
    "target_clip_duration": 6,
    "target_scene_count": 3,
    "total_duration": 18,
    "text_model": "...",
    "image_model": "...",
    "video_model": "...",
    "vision_model": "...",
    "audio_enabled": true,
    "seed": null,
    "quality_mode": false,
    "candidate_count": 1,
    "style_guide": { ... },
    "storyboard_raw": { ... },
    "manifest_id": "uuid-or-null",
    "manifest_version": 1
  },
  "scenes": [
    {
      "scene_index": 0,
      "scene_description": "...",
      "start_frame_prompt": "...",
      "end_frame_prompt": "...",
      "video_motion_prompt": "...",
      "transition_notes": "...",
      "keyframes": [
        {
          "position": "start",
          "prompt_used": "...",
          "file_path": "...",
          "file_hash": "sha1-of-file-content",
          "source": "generated"
        },
        {
          "position": "end",
          "prompt_used": "...",
          "file_path": "...",
          "file_hash": "sha1-of-file-content",
          "source": "generated"
        }
      ],
      "video_clips": [
        {
          "id": "uuid",
          "local_path": "...",
          "file_hash": "sha1-of-file-content",
          "duration_seconds": 6.0,
          "source": "generated",
          "status": "complete"
        }
      ],
      "scene_manifest": { ... },
      "audio_manifest": { ... }
    }
  ],
  "assets": [
    {
      "id": "uuid",
      "manifest_tag": "CHAR_01",
      "name": "Alice",
      "asset_type": "CHARACTER",
      "reference_image_url": "...",
      "reverse_prompt": "...",
      "visual_description": "..."
    }
  ]
}
```

**SHA computation:**
```python
import hashlib, json

def compute_checkpoint_sha(snapshot_data: dict) -> str:
    """Deterministic SHA-1 of the snapshot content."""
    canonical = json.dumps(snapshot_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()
```

The SHA is computed from the serialized snapshot JSON. Since `file_hash` values for binary assets (images, clips) are included in the snapshot, the SHA transitively covers all binary content without embedding the binaries themselves.

### 1.3 Checkpoint Lifecycle

| Event | Action |
|---|---|
| Project reaches `complete` status for the first time | Create **initial checkpoint** (`parent_sha = null`). Set `project.head_sha`. |
| User commits an edit (see Section 2) | Create new checkpoint with `parent_sha = head_sha`. Update `head_sha`. |
| User reverts to checkpoint X | Restore project state from `X.snapshot_data`. Create a **new** checkpoint (revert is itself a commit, like `git revert`). The old checkpoint chain remains intact. |
| User deletes a checkpoint | Remove the checkpoint row **and** any asset files that are not referenced by any other checkpoint in the same project. Re-link `parent_sha` of the child checkpoint to point to the deleted checkpoint's parent (splicing out the node). |

### 1.4 Asset File Retention

Binary files (keyframe PNGs, video clips) are **never deleted** when a new edit replaces them. They remain on disk and referenced by their original checkpoint snapshots. Deletion only occurs when:

- A specific checkpoint is deleted by the user, **and** the files it references are not referenced by any other checkpoint of the same project.
- The entire project is deleted (hard delete, not soft delete).

File paths follow the existing pattern but are **immutable once written**. New generations produce new file paths (UUID-based), never overwrite existing files.

---

## 2. In-Place Project Editing

### 2.1 Core Principle: Edits Are Data-Only, Generation Is Always Explicit

The fundamental rule of in-place editing is **lazy evaluation**:

- **Editing a text field** (prompt, description, model selection, scene count, etc.) is a **data-only operation**. It writes to the database and creates a checkpoint. It does **not** trigger any pipeline stage. Nothing is regenerated, re-run, or re-stitched.
- **Generation only happens when the user explicitly requests it** — by clicking a Regenerate button on a specific asset, a group of assets, or the whole project.
- When generation does run, it picks up whatever the current state of the data is. If the user changed `start_frame_prompt` on scene 3 ten minutes ago and now clicks Regenerate on scene 3's start keyframe, it uses the current (edited) prompt. If they never click Regenerate, the old keyframe stays.

This means a user can:

1. Change the base project prompt, edit three scene descriptions, swap a model — **commit** — and nothing regenerates.
2. Later, come back and regenerate just scene 2's video clip. The pipeline reads the current scene 2 data (which includes their earlier text edits) and generates against it.
3. Or regenerate everything in one shot if they want a full refresh.

**The edit system is a text/metadata editor.** The pipeline is a separate, on-demand tool the user invokes when ready.

### 2.2 Staleness Indicators

Since edits don't trigger regeneration, the UI needs to communicate what's out of sync:

| Indicator | Meaning | Shown On |
|---|---|---|
| **Fresh** (no badge) | The asset was generated from the current text/settings. Nothing has changed since. | Keyframe, clip |
| **Stale** (amber dot) | Upstream data has changed since this asset was generated. E.g., the prompt was edited but the keyframe hasn't been regenerated. | Keyframe, clip |
| **Missing** (empty placeholder) | No asset exists yet — either removed by the user or never generated (new scene). | Keyframe, clip |

Staleness is determined by comparing the asset's `prompt_used` (or generation inputs) against the current scene/project data. The checkpoint diff can also surface this: if a text field changed in a commit but the downstream asset was not regenerated, it's stale.

**Staleness does not block anything.** A stale keyframe is still a valid keyframe. The user decides when (or if) to regenerate.

### 2.3 Scene Expansion (Adding Scenes)

Increasing `target_scene_count` in edit mode is a data-only edit with a **continuation option**:

1. User changes `target_scene_count` from 5 to 8 in the project settings.
2. The edit is saved. The project now has 5 populated scenes and 3 empty scene slots (scene_index 5, 6, 7).
3. The UI shows the 3 new slots as empty scene cards with a **Generate** button.
4. The user clicks **Continue** (or **Generate New Scenes**) — this runs the storyboard stage for only the new scenes (indices 5–7), using the existing project context (prompt, style guide, manifest, existing scene continuity).
5. The pipeline uses the existing `_compute_invalidation()` logic: `resume_stage = "storyboarding"`, `scene_boundary = 5` (the first new scene). Existing scenes 0–4 are untouched.
6. After storyboarding, the new scenes have text but no keyframes or clips. The user can then selectively generate those, or hit **Regenerate All New** to pipeline them through.

**Scene removal** works the same way — data-only. Removing scene 3 marks it as deleted in the project state. The stitcher, next time it runs, produces a video without that scene. The scene's assets remain on disk for checkpoint history.

**Scene reordering** — swap `scene_index` values. Data-only. The stitcher picks up the new order on next run.

### 2.4 Edit Mode UX Flow

1. User opens a completed project and clicks **Edit** (distinct from **Fork**).
2. The project enters **edit mode** — an overlay/panel where all fields become editable (reusing the existing `EditableSceneCard` patterns).
3. The user makes changes. A **dirty state tracker** highlights what has changed relative to the current `head_sha` checkpoint. Staleness indicators appear on assets whose upstream data has been modified.
4. The user clicks **Commit** to save changes, providing an optional commit message. An auto-generated message summarizes the changes (e.g., "Edit scene 2 video prompt, add 3 scenes, change image model").
5. A new checkpoint is created. The project's live data is updated in the database. **No generation runs.**
6. The user can then optionally trigger regeneration on any stale or missing assets.

### 2.5 Edit Capabilities (Scene Card)

Each scene card in edit mode exposes:

| Element | Actions |
|---|---|
| **Scene description** | Edit text inline. Supports `[TAG]` manifest references (see Section 4). Data-only — no regeneration. |
| **Start keyframe image** | View current image + the prompt that generated it (`keyframe.prompt_used`). **Regenerate** (explicit click) with same or edited prompt. **Replace** with an uploaded image. **Remove** (sets to null). Shows staleness badge if prompt has diverged from `prompt_used`. |
| **End keyframe image** | Same as start keyframe. |
| **Video clip** | View current clip + the prompt that generated it. **Regenerate** (explicit click). **Replace** with an uploaded video file. **Remove** from the scene. Shows staleness badge if upstream prompts or keyframes have changed. |
| **Start frame prompt** | Edit the LLM-generated image prompt directly. Data-only — keyframe becomes stale. |
| **End frame prompt** | Edit the LLM-generated image prompt directly. Data-only — keyframe becomes stale. |
| **Video motion prompt** | Edit the LLM-generated video prompt directly. Data-only — clip becomes stale. |
| **Transition notes** | Edit inline. Data-only. |
| **Scene text** (description) | Edit inline with manifest tag autocomplete. **Clear** to remove all text. Data-only. |
| **Scene settings** | Per-scene model overrides (image_model, video_model) — optional, falls back to project-level. Data-only — affected assets become stale. |

**Bulk actions on a scene:**
- **Remove clip** — detaches the video clip (keeps the file for checkpoint history). Data-only.
- **Remove all text** — clears `scene_description`, `start_frame_prompt`, `end_frame_prompt`, `video_motion_prompt`, `transition_notes`. Data-only.
- **Remove keyframes** — clears both start and end keyframes. Data-only.
- **Regenerate all** — explicitly re-runs the full pipeline for this scene (keyframes → video). This is the only action here that triggers generation.

### 2.6 Edit Capabilities (Project Level)

| Element | Actions | Effect |
|---|---|---|
| **Title** | Edit inline. | Data-only. |
| **Original prompt/script** | Edit the source prompt. | Data-only. All scenes become stale (prompt is upstream of everything). |
| **Style** | Change style preset. | Data-only. All keyframes and clips become stale. |
| **Aspect ratio** | Change. | Data-only. All assets become stale (dimensional change). |
| **Clip duration** | Change target duration. | Data-only. All clips become stale. |
| **Scene count** | Increase or decrease. | Data-only. New scenes appear as empty slots. Removed scenes are marked deleted. |
| **Models** | Change text_model, image_model, video_model, vision_model. | Data-only. Assets generated by the changed model type become stale. |
| **Audio** | Toggle audio_enabled. | Data-only. Clips become stale. |
| **Manifest** | Change associated manifest or detach. | Data-only. All scene manifests and rewritten prompts become stale. |

### 2.7 Regeneration Within Edit Mode

Regeneration is always **user-initiated** and **scoped**:

| Scope | Trigger | What Runs |
|---|---|---|
| Single keyframe | Click Regenerate on one keyframe | Image generation for that position only |
| Single clip | Click Regenerate on one clip | Video generation for that scene only (uses current keyframes as input) |
| Single scene (all assets) | Click "Regenerate All" on a scene card | Keyframes → video for that scene |
| New scenes only | Click "Continue" after adding scenes | Storyboard → keyframes → video for new scenes only |
| All stale assets | Click "Regenerate Stale" in project toolbar | Pipeline runs for all assets with staleness badges |
| Full project | Click "Regenerate All" in project toolbar | Full pipeline re-run for all scenes |
| Stitch only | Click "Re-stitch" in project toolbar | FFmpeg stitcher re-runs with current clips in current scene order |

When the user clicks **Regenerate** on a keyframe or clip:

1. The system creates a **generation job** targeting that specific asset.
2. The user can continue editing other parts of the project while generation runs.
3. On completion, the new asset replaces the old one in the live project state. The old asset file remains on disk (covered by the previous checkpoint).
4. The generation is included in the next commit's changelog (or auto-committed if the user has left edit mode).

**Selective regeneration API:**

```
POST /api/projects/{id}/scenes/{scene_index}/regenerate
{
  "targets": ["start_keyframe", "end_keyframe", "video_clip"],
  "prompt_overrides": {
    "start_frame_prompt": "optional new prompt",
    "video_motion_prompt": "optional new prompt"
  }
}
```

```
POST /api/projects/{id}/regenerate
{
  "scope": "stale" | "all" | "new_scenes" | "stitch_only",
  "scene_indices": [5, 6, 7]  // optional, for "new_scenes" scope
}
```

These endpoints run only the necessary pipeline stages for the requested targets, never more.

### 2.8 Asset Upload/Replace

Users can replace any generated asset with their own:

```
PUT /api/projects/{id}/scenes/{scene_index}/keyframes/{position}
Content-Type: multipart/form-data
file: <image file>
```

```
PUT /api/projects/{id}/scenes/{scene_index}/clip
Content-Type: multipart/form-data
file: <video file>
```

Uploaded assets are stored alongside generated ones with `source: "uploaded"`. The original generated asset remains on disk for checkpoint history. Uploaded assets are never marked stale (they are user-provided, not derived from prompts).

---

## 3. Changelog & Revert

### 3.1 Changelog Structure

Each checkpoint's `metadata.changes` records a structured diff:

```json
{
  "changes": [
    {
      "scope": "project",
      "field": "title",
      "old": "My Video",
      "new": "My Updated Video"
    },
    {
      "scope": "scene",
      "scene_index": 1,
      "field": "start_frame_prompt",
      "old": "A wide shot of...",
      "new": "A close-up of..."
    },
    {
      "scope": "scene",
      "scene_index": 1,
      "field": "start_keyframe",
      "action": "regenerated",
      "old_file_hash": "abc123...",
      "new_file_hash": "def456..."
    },
    {
      "scope": "scene",
      "scene_index": 2,
      "field": "video_clip",
      "action": "replaced",
      "old_file_hash": "...",
      "new_file_hash": "...",
      "source": "uploaded"
    }
  ]
}
```

### 3.2 Changelog API

```
GET /api/projects/{id}/checkpoints
→ [{ sha, parent_sha, message, created_at, summary: "3 changes across 2 scenes" }]

GET /api/projects/{id}/checkpoints/{sha}
→ { sha, parent_sha, message, created_at, metadata, snapshot_data }

GET /api/projects/{id}/checkpoints/{sha}/diff
→ { changes: [...structured diff against parent...] }
```

### 3.3 Revert

```
POST /api/projects/{id}/revert
{ "target_sha": "abc123..." }
```

Revert restores the project's live database state from the target checkpoint's `snapshot_data`:

1. Update `Project` fields from snapshot.
2. Upsert/delete `Scene` rows to match snapshot scene list.
3. For each scene, restore `Keyframe` and `VideoClip` references (files already exist on disk from the original checkpoint).
4. Restore `SceneManifest` and `SceneAudioManifest` rows.
5. Create a **new checkpoint** recording this as a revert: `message: "Revert to checkpoint {short_sha}"`, `parent_sha = current head_sha`.
6. Update `project.head_sha` to the new revert checkpoint.

This is a forward-moving operation — no history is lost. The revert itself becomes part of the version chain.

### 3.4 Checkpoint Deletion

```
DELETE /api/projects/{id}/checkpoints/{sha}
```

- Cannot delete the current `head_sha` checkpoint (the project must have at least one checkpoint while in versioned state).
- Splices the deleted node out of the chain: child checkpoint's `parent_sha` is re-pointed to the deleted checkpoint's `parent_sha`.
- Orphaned files (only referenced by the deleted checkpoint, not by any remaining checkpoint in the project) are removed from disk.

---

## 4. Rich Manifest-Tag Text Integration

### 4.1 Tag Syntax

Manifest tags are referenced inline in scene text fields using the bracket syntax already used by the LLM storyboarding system: `[CHAR_01]`, `[ENV_02]`, `[PROP_01]`, etc.

In the frontend text editors, these tags are:

- **Autocompleted** — typing `[` triggers a dropdown populated from the project's manifest assets. Filtered as the user types (`[CH` → shows `CHAR_01`, `CHAR_02`). Each option shows the asset name, type, and thumbnail.
- **Visually decorated** — rendered as inline chips/badges showing the tag, asset name, and a tiny thumbnail. Color-coded by asset type (characters = blue, environments = green, props = orange, etc.).
- **Clickable** — clicking a tag chip opens a popover showing the full asset details (image, description, reverse_prompt, visual_description).
- **Validated** — tags that don't resolve to a valid asset in the project's manifest are highlighted in red with a warning tooltip.

### 4.2 Tag Binding Storage

The existing `SceneManifest.asset_tags` JSON array already tracks which tags appear in each scene. The edit system updates this array whenever scene text is saved:

1. Parse all `[TAG]` references from `scene_description`, `start_frame_prompt`, `end_frame_prompt`, `video_motion_prompt`.
2. Union them into `asset_tags`.
3. Update `SceneManifest.manifest_json.placements` — ensure each referenced tag has a placement entry (add minimal entries for newly-referenced tags).

This keeps the downstream pipeline (prompt rewriting, reference image selection) consistent with the user's manual tag references.

### 4.3 Tag-Aware Prompt Display

When viewing or editing prompts, the system shows:

- **Original prompt** — the scene's `start_frame_prompt` / `video_motion_prompt` with `[TAG]` references.
- **Rewritten prompt** — the expanded prompt from `SceneManifest.rewritten_keyframe_prompt` / `rewritten_video_prompt`, showing how tags were resolved into full descriptions.
- **Sent prompt** — the actual prompt passed to the generation model (`Keyframe.prompt_used`).

All three are viewable. The "original prompt" is the editable one; rewriting happens automatically on regeneration.

---

## 5. Prompt Transparency

### 5.1 Prompt Viewer Per Asset

Each generated asset (keyframe, video clip) exposes the prompt chain that produced it:

**Keyframe prompt chain:**
1. `scene.start_frame_prompt` — LLM-generated base prompt
2. `scene_manifest.rewritten_keyframe_prompt` — manifest-aware rewrite (if manifest project)
3. `keyframe.prompt_used` — final prompt sent to the image model

**Video clip prompt chain:**
1. `scene.video_motion_prompt` — LLM-generated base prompt
2. `scene_manifest.rewritten_video_prompt` — manifest-aware rewrite (if manifest project)
3. *(Currently not stored on VideoClip — see 5.2)*

### 5.2 Schema Addition: Video Clip Prompt Storage

Add to `video_clips`:

| Column | Type | Description |
|---|---|---|
| `prompt_used` | Text, nullable | The actual prompt sent to Veo. Populated at generation time. |

This closes the gap where video prompts are computed at runtime but not persisted. The video generation pipeline should store the final prompt in `VideoClip.prompt_used` alongside submission.

### 5.3 Prompt Editor

In edit mode, the prompt viewer becomes an editor:

- The user can modify any level of the prompt chain.
- Editing the base prompt (`scene.start_frame_prompt`) and regenerating will re-run the rewrite pipeline and produce a new `prompt_used`.
- Editing the `prompt_used` directly and regenerating will use that exact prompt, bypassing the rewrite pipeline (power-user override).

---

## 6. Fork / Edit Button Separation

### 6.1 Current State

The `Edit & Fork` button triggers a combined flow where all edits create a new forked project.

### 6.2 New Button Layout

For a completed/terminal project, the action bar shows:

```
[ Edit ]  [ Fork ]  [ Delete ]  [ Re-run ]
```

| Button | Behavior |
|---|---|
| **Edit** | Enters edit mode on the current project. Changes are committed in-place with version tracking (PipeSVN). No new project is created. |
| **Fork** | Opens the fork panel (existing `EditForkPanel` minus the edit-specific parts). Creates a new project with `forked_from_id` pointing to the source. The fork panel allows pre-fork modifications (same as today). No version history is carried over to the fork — it starts fresh. |

### 6.3 Edit Mode vs Fork Panel

| Feature | Edit Mode | Fork Panel |
|---|---|---|
| Creates new project | No | Yes |
| Version history | Yes (PipeSVN checkpoints) | No (fork starts with initial checkpoint on completion) |
| Execution model | **Lazy** — edits are data-only, generation is explicit and scoped | **Eager** — fork computes invalidation and auto-runs the pipeline from the earliest affected stage |
| Granular regeneration | Yes (per-keyframe, per-clip, per-scene, stale-only, or all) | No (invalidation-based re-run from a single resume point) |
| Scene expansion | Yes — increase scene count, commit, then generate new scenes on demand | Yes — change scene count and fork triggers full storyboard + pipeline |
| Asset upload/replace | Yes | No (only text edits + asset registry changes) |
| Prompt editing | Yes (all three levels) | Yes (base prompts only) |
| Manifest tag autocomplete | Yes | No (plain text editing) |
| Staleness tracking | Yes — visual indicators on out-of-sync assets | No — fork re-runs everything affected |
| Commit workflow | Explicit commit with message | Automatic on fork creation |

The key philosophical difference: **Edit mode trusts the user to decide what to regenerate and when.** Fork mode is a "make changes and go" workflow that figures out invalidation automatically. Both are valuable — edit for surgical control, fork for broad sweeps.

### 6.4 Fork Inherits Head Checkpoint

When forking a versioned project, the fork copies the state at the current `head_sha`, not the initial state. This means if a user edits a project (advancing to checkpoint 5) and then forks, the fork gets the checkpoint-5 state.

---

## 7. API Summary

### Checkpoint Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects/{id}/checkpoints` | List all checkpoints (newest first) |
| `GET` | `/api/projects/{id}/checkpoints/{sha}` | Get checkpoint detail + snapshot |
| `GET` | `/api/projects/{id}/checkpoints/{sha}/diff` | Structured diff against parent |
| `POST` | `/api/projects/{id}/checkpoints` | Manual checkpoint (snapshot current state) |
| `POST` | `/api/projects/{id}/revert` | Revert to a target checkpoint |
| `DELETE` | `/api/projects/{id}/checkpoints/{sha}` | Delete a checkpoint + orphaned files |

### Edit Endpoints

| Method | Path | Description |
|---|---|---|
| `PATCH` | `/api/projects/{id}/edit` | Apply a batch of edits + auto-commit (data-only, no generation) |
| `POST` | `/api/projects/{id}/regenerate` | Project-wide regeneration: `scope` = `stale` / `all` / `new_scenes` / `stitch_only` |
| `POST` | `/api/projects/{id}/scenes/{idx}/regenerate` | Regenerate specific assets for a scene |
| `PUT` | `/api/projects/{id}/scenes/{idx}/keyframes/{pos}` | Upload/replace a keyframe image |
| `PUT` | `/api/projects/{id}/scenes/{idx}/clip` | Upload/replace a video clip |
| `DELETE` | `/api/projects/{id}/scenes/{idx}/clip` | Remove a clip from a scene |
| `DELETE` | `/api/projects/{id}/scenes/{idx}/keyframes/{pos}` | Remove a keyframe |
| `DELETE` | `/api/projects/{id}/scenes/{idx}/text` | Clear all text fields on a scene |

### Updated Existing Endpoints

| Method | Path | Change |
|---|---|---|
| `POST` | `/api/projects/{id}/fork` | Unchanged, but renamed in UI from "Edit & Fork" to just "Fork". |
| `PATCH` | `/api/projects/{id}` | Extended beyond just `title` to support full project-level field edits within edit mode. Each PATCH auto-creates a checkpoint. |

---

## 8. Frontend Components

### 8.1 New Components

| Component | Description |
|---|---|
| `EditModeOverlay` | Wraps the project detail view in edit mode. Manages dirty state tracking, commit dialog, and edit toolbar. |
| `SceneEditor` | Enhanced `SceneCard` for edit mode. Shows all editable fields, regenerate/replace/remove buttons, prompt chain viewer. Built on top of existing `EditableSceneCard` patterns. |
| `PromptChainViewer` | Expandable panel showing the 3-level prompt chain (base → rewritten → sent) for a given asset. Editable in edit mode. |
| `ManifestTagInput` | Text input component with `[TAG]` autocomplete, chip rendering, and validation. Used in all scene text fields during edit mode. |
| `CheckpointLog` | Sidebar or drawer showing the checkpoint history. Each entry shows SHA (short), message, timestamp, and a diff summary. Expand to see full diff. Revert button per entry. |
| `CheckpointDiff` | Visual diff view comparing two checkpoint states. Highlights changed fields, added/removed scenes, and regenerated assets with before/after thumbnails. |

### 8.2 Modified Components

| Component | Changes |
|---|---|
| `ProjectDetail` | Add **Edit** and **Fork** as separate buttons. Show `CheckpointLog` toggle when project has checkpoints. |
| `SceneCard` | Add "View prompts" expand section showing the prompt chain. Add regenerate/replace/remove action buttons in edit mode. |
| `EditForkPanel` | Rename to `ForkPanel`. Remove edit-specific features (those move to `EditModeOverlay`). Simplify to fork-only workflow. |

---

## 9. Implementation Considerations

### 9.1 Snapshot Size

A full project snapshot with 10 scenes, 20 keyframes, 10 clips, and 30 assets is roughly 20-50KB of JSON (text fields + metadata, no binary data). SQLite handles this comfortably. For projects with 50+ checkpoints, consider pagination on the checkpoint list API.

### 9.2 File Hash Computation

Binary file hashes (`file_hash` in the snapshot) are computed lazily and cached. On first checkpoint creation, hash all existing files. On subsequent edits, only hash newly-created files. Store computed hashes in a lightweight cache (in-memory or a `file_hashes` table) to avoid re-reading large video files.

### 9.3 Orphan Cleanup

When deleting a checkpoint, the orphan detection query is:

```sql
-- Find file paths in the deleted checkpoint's snapshot that don't appear
-- in any other checkpoint's snapshot for the same project
SELECT DISTINCT file_path FROM deleted_snapshot_files
WHERE file_path NOT IN (
  SELECT DISTINCT file_path FROM other_checkpoint_snapshot_files
  WHERE project_id = :project_id AND sha != :deleted_sha
)
```

Since snapshots are JSON, this requires application-level extraction of file paths from `snapshot_data`, not pure SQL. A helper function extracts all `file_path` and `local_path` values from a snapshot.

### 9.4 Concurrency

For the single-user case (current scope), optimistic locking is sufficient:

- Each edit request includes the expected `head_sha`.
- If `project.head_sha != expected_sha` at commit time, the edit is rejected with a conflict error.
- The frontend refreshes and lets the user retry.

This lays the groundwork for future multi-user support without implementing full conflict resolution now.

### 9.5 Migration Path

1. Add `head_sha` column to `projects` (nullable, default null).
2. Add `prompt_used` column to `video_clips` (nullable).
3. Create `project_checkpoints` table.
4. Existing projects have `head_sha = null` — no checkpoints until the user first enters edit mode, at which point the initial checkpoint is auto-created from the current state.
5. Backfill: optionally run a migration that creates initial checkpoints for all existing `complete` projects.

---

## 10. Out of Scope (Future)

- **Multi-user / authentication** — no user model yet. Checkpoints don't record `author`. Will be added when user system is implemented.
- **Branching** — checkpoints are linear (single `parent_sha` chain). Branching would require a DAG structure. Forking serves as the current branching mechanism.
- **Merge** — combining changes from two checkpoint chains. Requires branching first.
- **Diff visualization for binary assets** — showing visual diffs of keyframe images (side-by-side comparison). The checkpoint diff shows before/after thumbnails but not pixel-level diffs.
- **Collaborative real-time editing** — multiple users editing simultaneously with live cursors. Requires WebSocket infrastructure and operational transforms or CRDTs.
- **Partial scene revert** — reverting individual fields from a checkpoint rather than the full project state. Could be built on top of the structured diff system.
