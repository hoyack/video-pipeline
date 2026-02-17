# VidPipe V2: Reusable Manifest System — Architecture Addendum

## Addendum to: VidPipe V2 Production-Grade AI Video Generation Pipeline

---

## 1. The Conceptual Shift: Manifests as First-Class Entities

The original architecture treats manifests as **per-project artifacts** — generated during Phase 0 as a side effect of uploading reference images during video generation. This addendum proposes a fundamental evolution:

**Manifests become standalone, reusable, transportable resources that exist independently of any single project.**

This mirrors how professional studios operate:

| Studio Concept | Original VidPipe V2 | Revised Model |
|---|---|---|
| **Character Bible** | Generated per-project, lives inside project | Standalone manifest, selectable across projects |
| **Asset Library** | Scoped to one project's Asset Registry | Shared pool accessible from Manifest Library |
| **Production Kit** | Rebuilt every time | Created once, refined over time, reused endlessly |

### Why This Matters

1. **Reusability** — A user who spends 10 minutes tagging and refining a character manifest shouldn't redo that for every video. A "Detective Marcus" manifest works across a noir short film, a trailer, a sequel, and a marketing clip.

2. **Transportability** — Manifests can be shared between projects, duplicated as starting points, or even shared between users (future: marketplace potential).

3. **Quality Iteration** — The Manifest Creator becomes a dedicated workspace for refining asset descriptions, swapping reference images, and curating the asset pool *before* committing to expensive video generation. This is the studio pre-production phase separated from principal photography.

4. **Separation of Concerns** — Decoupling asset preparation from video generation simplifies both workflows. The generation pipeline receives a pre-built manifest ID rather than running decomposition inline.

---

## 2. Manifest Entity Schema

```
Manifest {
  manifest_id: UUID
  user_id: UUID

  # Identity
  name: string                     # "Noir Detective Pack", "Corporate Brand Kit"
  description: string
  thumbnail_url: string            # Auto-generated from contact sheet or primary asset

  # Classification
  category: enum [
    CHARACTERS,
    ENVIRONMENT,
    FULL_PRODUCTION,
    STYLE_KIT,
    BRAND_KIT,
    CUSTOM
  ]
  tags: string[]                   # ["noir", "detective", "1940s"]

  # Asset Pool
  assets: Asset[]                  # Full Asset Registry (same schema as original doc)
  asset_count: int

  # Contact Sheet
  contact_sheet_url: string

  # Processing State
  status: enum [DRAFT, PROCESSING, READY, ERROR]
  processing_progress: {
    uploads_total: int,
    uploads_processed: int,
    crops_total: int,
    crops_reverse_prompted: int,
    face_merges_completed: int,
    current_step: string           # "yolo_detection", "reverse_prompting", "face_matching"
  }

  # Cost & Usage Tracking
  total_processing_cost: float
  projects_used_in: UUID[]
  times_used: int
  last_used_at: datetime

  # Versioning
  version: int
  parent_manifest_id: UUID | null  # If duplicated from another manifest

  created_at: datetime
  updated_at: datetime
}
```

### Manifest-to-Project Relationship

```
PROJECT references MANIFEST (many-to-one)
  - A project selects ONE manifest at generation time
  - The manifest's Asset Registry becomes the project's asset pool
  - The project can declare ADDITIONAL ad-hoc assets during storyboarding
    (these live in the project scope, not the manifest)
  - Forked projects inherit the parent's manifest reference

MANIFEST is used by PROJECTS (one-to-many)
  - A single manifest can be referenced by unlimited projects
  - Changes to a manifest do NOT retroactively affect completed projects
  - Changes CAN be pulled into in-progress projects (user choice)
```

---

## 3. Manifest Library View

```
+-------------------------------------------------------------------------+
|  MANIFEST LIBRARY                                     [+ New Manifest]  |
|                                                                         |
|  +- Filter Bar -------------------------------------------------------+|
|  | [All] [Characters] [Environments] [Full Production] [Style Kits]   ||
|  | Search: [________________________]   Sort: [Last Used v]           ||
|  +--------------------------------------------------------------------+|
|                                                                         |
|  +------------------+  +------------------+  +------------------+       |
|  | [contact sheet]  |  | [contact sheet]  |  | [contact sheet]  |       |
|  |                  |  |                  |  |                  |       |
|  | Noir Detective   |  | Corporate Brand  |  | Sci-Fi Colony    |       |
|  | Pack             |  | Kit              |  | World            |       |
|  |                  |  |                  |  |                  |       |
|  | 11 assets        |  | 6 assets         |  | 23 assets        |       |
|  | 3 chars, 2 envs  |  | 1 char, 2 props  |  | 8 chars, 6 envs  |       |
|  | 2 props, 1 veh   |  | 2 styles, 1 env  |  | 5 props, 4 vehs  |       |
|  |                  |  |                  |  |                  |       |
|  | Used in 4 proj.  |  | Used in 12 proj. |  | Used in 1 proj.  |       |
|  | Last: 2 days ago |  | Last: today      |  | Last: 1 week ago |       |
|  |                  |  |                  |  |                  |       |
|  | * READY          |  | * READY          |  | ~ PROCESSING     |       |
|  |                  |  |                  |  |                  |       |
|  | [Edit] [Dup] [..]|  | [Edit] [Dup] [..]|  | [View] [..]      |       |
|  +------------------+  +------------------+  +------------------+       |
|                                                                         |
+-------------------------------------------------------------------------+
```

### Card Actions

- **Edit** -> Opens Manifest Creator with this manifest loaded
- **Duplicate** -> Creates a copy (new manifest_id, parent_manifest_id set)
- **Delete** -> Soft delete (warns if used by active projects)
- **View** -> Read-only detail view showing all assets and contact sheet
- **Use in Project** -> Navigates to GenerateForm with this manifest pre-selected

---

## 4. Manifest Creator View

The Manifest Creator is the dedicated workspace for building and refining manifests. Three-panel canvas layout:

```
+---------------+------------------------------------+---------------------------+
| LEFT PANEL    |         CENTER CANVAS              |     RIGHT PANEL           |
| Upload &      |                                    |     Asset Detail          |
| Source Images  |   Contact Sheet / Detection View   |     & Editing             |
+---------------+------------------------------------+---------------------------+
|               |                                    |                           |
| [Drag & Drop] |  PROJECT REFERENCE SHEET           |  Selected: CHAR_01        |
|               |                                    |  "Detective Marcus"       |
| Uploads (6):  |  +----+ +----+ +----+              |                           |
|               |  |[1] | |[2] | |[3] |              |  [reference image]        |
| [thumb] 1     |  |Det.| |Fem.| |Red |              |                           |
| Det.Marcus    |  |Marc| |Fat.| |Must|              |  Name:                    |
| CHAR  check   |  +----+ +----+ +----+              |  [Detective Marcus_____]  |
|               |  +----+ +----+ +----+              |                           |
| [thumb] 2     |  |[4] | |[5] | |[6] |              |  Type:                    |
| Fem.Fatale    |  |Noir| |Rev.| |Rain|              |  (*) CHAR ( ) OBJ ( ) ENV |
| CHAR  check   |  |Off.| |    | |Scn.|              |  ( ) PROP ( ) VEH ( ) STY |
|               |  +----+ +----+ +----+              |                           |
| [thumb] 3     |                                    |  Description:             |
| Red Mustang   |  Toggle: [Sheet] [Detections]      |  [Main character, always  |
| VEH   check   |                                    |   wears trench coat.]     |
|               |  -- Detection View --              |                           |
| [thumb] 4     |  Upload 1: "Det. Marcus"           |  Tags:                    |
| Noir Office   |  +---------------------------+     |  [protagonist] [noir] [+] |
| ENV   check   |  | +--person(0.97)--------+  |     |                           |
|               |  | |  +--face(0.99)--+    |  |     |  -- After Processing --   |
| [thumb] 5     |  | |  |  CHAR_01    |    |  |     |                           |
| Revolver      |  | |  +-------------+    |  |     |  Quality Score: 8.5/10    |
| PROP  check   |  | +---------------------+  |     |                           |
|               |  +---------------------------+     |  Reverse Prompt:          |
| [thumb] 6     |                                    |  [A weathered middle-     |
| Rain Scene    |  Extracted: CHAR_01 (face),        |   aged Caucasian man,     |
| ENV   check   |  CHAR_01 (full body)               |   mid-50s, salt-and-      |
|               |                                    |   pepper close-cropped    |
| [+ Add More]  |  -- Extracted Assets Grid --       |   beard, deep-set...]     |
|               |  +----++----++----++----++----+     |  [Edit Prompt]            |
|               |  |C_01||C_02||V_01||C_03||E_01|     |                           |
|               |  |face||face||car ||drvr||offc|     |  Visual Description:      |
|               |  |8.5 ||9.0 ||7.5 ||3.5 ||8.0 |     |  [Signature trench coat   |
|               |  +----++----++----++----++----+     |   (always present)...]    |
|               |  [Show all 11 extracted assets]     |  [Edit Description]       |
|               |                                    |                           |
|               |                                    |  [Remove Asset]           |
|               |                                    |  [Re-process]             |
|               |                                    |  [Swap Reference Image]   |
|               |                                    |  [Generate Clean Sheet]   |
|               |                                    |  [Pin as Primary Ref]     |
+---------------+------------------------------------+---------------------------+
| Asset Summary: 11 total | 3 CHAR | 1 VEH | 2 ENV | 1 PROP | 1 OBJ            |
| Face Merges: 1 (uploads 1+4 -> CHAR_01)  |  Low Quality: 1 (CHAR_03: 3.5)     |
| Processing Cost: $0.42  |  [Save Draft]  [Reprocess All]  [Finalize]           |
+---------------------------------------------------------------------------------+
```

### Manifest Creator Workflow Stages

```
STAGE 1: UPLOAD & TAG (user-driven, no processing yet)
  User drags in reference images.
  For each: Name (required), Type, Description (optional), Tags (optional).
  Left panel populates. Center shows raw contact sheet. Right shows metadata form.
  Status: DRAFT. No API calls. No cost incurred.
  User clicks [Process] when ready.

STAGE 2: PROCESSING (automated, live progress)
  Step 2a: Contact Sheet Assembly (Pillow, instant)
  Step 2b: YOLO Detection Sweep (local GPU, per-image)
    -> Bounding boxes appear in Detection View
    -> Extracted crops appear in grid
    -> Face cross-matching after all images done
  Step 2c: Vision LLM Reverse-Prompting (Gemini, per-crop)
    -> Reverse_prompt text streams in right panel
    -> Quality scores appear on thumbnails
  Step 2d: Tag Assignment + Registry Population
  Status: PROCESSING -> READY

STAGE 3: REVIEW & REFINE (user-driven, post-processing)
  Per-asset edit operations:
  - Edit Reverse Prompt (manual override, saves immediately, no reprocessing)
  - Edit Visual Description (manual override)
  - Edit Name / Type / Tags
  - Swap Reference Image (re-triggers reverse-prompting, ~$0.02)
  - Re-process (fresh YOLO + reverse-prompt, ~$0.02)
  - Remove Asset (does not delete GCS files)
  - Pin as Primary Reference (designate which ref gets sent to Veo)
  - Generate Clean Reference (see Section 6)
  User clicks [Finalize] -> Status: READY

STAGE 4: USE IN PROJECTS (ongoing)
  Manifest appears in library with READY status.
  User selects in GenerateForm. Can re-open for editing anytime.
  Editing creates new version. In-progress projects unaffected (snapshot).
```

---

## 5. GenerateForm Integration

### Manifest Selector

```
+------------------------------------------------------------------+
|  ASSET MANIFEST                                                   |
|                                                                   |
|  Choose how to provide reference assets:                          |
|                                                                   |
|  (*) Select Existing Manifest    ( ) Quick Upload (inline)        |
|                                                                   |
|  -- Selected Manifest --                                          |
|  +--------------------------------------------------------------+|
|  | [contact sheet]  Noir Detective Pack                         ||
|  |                  11 assets: 3 chars, 2 envs, 2 props,       ||
|  |                  1 vehicle, 2 styles, 1 object               ||
|  |                  Quality: avg 7.6/10  |  Used 4 times        ||
|  |                                                              ||
|  |  Key Assets:                                                 ||
|  |  [C_01] [C_02] [V_01] [E_01] [P_01]  +6 more               ||
|  |                                                              ||
|  |  [Change Manifest]  [Edit in Creator]  [View Details]        ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  -- OR: Quick Upload --                                           |
|  (Creates a one-off manifest automatically. Can be saved to       |
|   library after generation.)                                      |
|  [Drop reference images here]                                     |
+------------------------------------------------------------------+
```

### Pipeline Flow Change

```
WITH PRE-BUILT MANIFEST (from library):
  pending -> storyboarding -> keyframing -> video_gen -> stitching -> complete
  Phase 0 (Manifesting) is SKIPPED.
  Saves ~$0.40 and ~30-60 seconds per project reusing a manifest.

WITH QUICK UPLOAD (inline):
  pending -> manifesting -> storyboarding -> keyframing -> video_gen -> complete
  Same as original. Auto-manifest created. User prompted to save to library.
```

---

## 6. Strategic Addition A: Reference Image Preprocessing ("Clean Sheets")

### Problem

Raw uploads are often suboptimal for Veo's reference system:
- Busy backgrounds competing with the subject
- Inconsistent lighting / color casts
- Partial occlusion
- Mixed aspect ratios and resolutions

Veo docs recommend "well-lit, neutral background" references.

### Solution: Generate Clean Reference Images

```
INPUT: Raw crop of Detective Marcus (busy background, side-lit, 
       partially occluded by doorframe)

PROCESS:
  1. Take existing reverse_prompt for this asset
  2. Append: "Portrait on solid neutral gray background. Subject facing
     slightly left. Clean even studio lighting. No occlusion. Full head
     and shoulders. Photorealistic."
  3. Feed BOTH original crop (as reference) AND enhanced prompt to 
     Gemini Image generation
  4. Generate 2-4 candidates
  5. Quality assessment:
     - Face similarity (ArcFace embedding distance from original)
     - YOLO detection confidence on clean version
     - Gemini Vision quality rating
  6. Present top candidate to user in Manifest Creator
  7. User accepts, regenerates, or keeps original

OUTPUT: Clean character sheet image optimized for Veo reference input.
```

### Quality Tiers

```
TIER 1: No preprocessing (default)
  - Use raw crop as-is. Cost: $0

TIER 2: Background removal only (fast, cheap)
  - rembg or similar, replace with neutral gray. Cost: $0 (local)

TIER 3: Full clean sheet generation (best quality)
  - Gemini Image generates idealized version
  - Validated via face embedding similarity. Cost: ~$0.02-0.05

TIER 4: Multi-angle sheet (premium)
  - Generate 3 angles: frontal, 3/4 left, 3/4 right
  - Gives pipeline multiple reference options per character
  - Cost: ~$0.06-0.15 (3 generations)
```

User-initiated per asset in Manifest Creator. Recommended for key characters.
"CHAR_01 has quality score 5.5. Generate a clean reference for better results? (~$0.03)"

---

## 7. Strategic Addition B: Multi-Candidate Selection with Quality Scoring

### Standard Mode (default): sampleCount=1, no selection needed

### Quality Mode (user-selectable):

```
sampleCount: 2-4 (configurable, default 2)

SCORING PIPELINE (per candidate):

  1. MANIFEST ADHERENCE (weight: 0.35)
     Does the scene contain the expected assets?
     Face matching confirms character identity.
     Spatial analysis checks composition hints.
     Score: 0-10

  2. VISUAL QUALITY (weight: 0.25)
     Gemini Vision quality assessment.
     Artifact detection (glitches, morphing).
     Resolution / sharpness analysis.
     Score: 0-10

  3. CONTINUITY (weight: 0.25)
     CLIP embedding similarity: last frame of N-1 vs first frame of candidate.
     Character appearance consistency (face embedding distance).
     Environment/lighting consistency.
     Score: 0-10

  4. PROMPT ADHERENCE (weight: 0.15)
     Gemini Vision: "Does this match the description?"
     Action/motion assessment.
     Score: 0-10

  COMPOSITE SCORE = weighted sum -> 0-10
  WINNER: Highest composite selected. User can manually override.
```

### Cost Impact (10-scene project)

| Component | Standard Mode | Quality Mode (2x) | Delta |
|---|---|---|---|
| Video generation | $32.00 | $64.00 | +$32.00 |
| CV Analysis (scoring) | $0.50 | $1.00 | +$0.50 |
| **Total** | **~$34.35** | **~$67.35** | **+$33.00** |

Position as premium option for final renders. Standard Mode for drafting.

---

## 8. Strategic Addition C: Audio Manifest

### Audio Manifest Schema (per scene)

```
SceneAudioManifest {
  scene_index: int

  dialogue_lines: [
    {
      speaker_tag: "CHAR_01"       # Maps to Asset Registry character
      speaker_name: "Marcus"
      line: "This doesn't add up..."
      delivery: "muttered, tired, slightly slurred"
      timing: "mid-scene"
      emphasis: ["doesn't", "up"]
    }
  ]

  sfx: [
    {
      effect: "paper rustling"
      trigger: "Marcus examines the letter"
      timing: "0:02-0:04"
      volume: "subtle"             # "subtle", "prominent", "background"
    }
  ]

  ambient: {
    base_layer: "quiet office hum, electrical buzz from desk lamp"
    environmental: "distant traffic through closed window"
    weather: null
    time_cues: "ticking wall clock"
  }

  music: {
    style: "low, tense jazz piano"
    mood: "suspenseful, noir"
    tempo: "slow, deliberate"
    instruments: ["solo piano", "brushed cymbal"]
    transition: "fade in from silence over first 2 seconds"
  }

  audio_continuity: {
    carries_from_previous: ["ambient.base_layer", "ambient.environmental"]
    new_in_this_scene: ["music", "sfx.paper_rustling"]
    cuts_from_previous: ["dialogue"]
  }
}
```

### Usage in Pipeline

- **Storyboarding**: LLM generates audio manifest alongside visual manifest. Dialogue mapped to character tags.
- **Prompt Rewriting**: Audio entries compiled into Veo prompt audio section with speaker attribution matching visual characters.
- **Post-Generation Analysis**: Gemini assesses dialogue audibility, SFX-action alignment, ambient consistency. Audio continuity score added to overall scoring.

---

## 9. Database Schema Additions

```sql
-- Manifests as standalone entities
CREATE TABLE manifests (
    manifest_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    thumbnail_url TEXT,
    category TEXT DEFAULT 'CUSTOM',
    tags TEXT[],
    status TEXT NOT NULL DEFAULT 'DRAFT',
    processing_progress JSONB,
    contact_sheet_url TEXT,
    asset_count INTEGER DEFAULT 0,
    total_processing_cost REAL DEFAULT 0.0,
    times_used INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    parent_manifest_id UUID REFERENCES manifests(manifest_id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_manifests_user ON manifests(user_id);
CREATE INDEX idx_manifests_status ON manifests(user_id, status);

-- Assets now belong to MANIFESTS (not directly to projects)
ALTER TABLE assets ADD COLUMN manifest_id UUID REFERENCES manifests(manifest_id);

-- Project-manifest relationship
ALTER TABLE projects ADD COLUMN manifest_id UUID REFERENCES manifests(manifest_id);
ALTER TABLE projects ADD COLUMN manifest_version INTEGER;

-- Manifest snapshots (frozen state when project starts)
CREATE TABLE manifest_snapshots (
    snapshot_id UUID PRIMARY KEY,
    manifest_id UUID REFERENCES manifests(manifest_id),
    project_id UUID REFERENCES projects(project_id),
    version_at_snapshot INTEGER,
    snapshot_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Clean reference sheets
CREATE TABLE asset_clean_references (
    id UUID PRIMARY KEY,
    asset_id UUID REFERENCES assets(asset_id),
    tier TEXT NOT NULL,
    clean_image_url TEXT NOT NULL,
    generation_prompt TEXT,
    face_similarity_score REAL,
    quality_score REAL,
    is_primary BOOLEAN DEFAULT FALSE,
    generation_cost REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Multi-candidate scoring
CREATE TABLE generation_candidates (
    candidate_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id),
    scene_index INTEGER NOT NULL,
    candidate_number INTEGER NOT NULL,
    clip_url TEXT,
    thumbnail_url TEXT,
    manifest_adherence_score REAL,
    visual_quality_score REAL,
    continuity_score REAL,
    prompt_adherence_score REAL,
    composite_score REAL,
    scoring_details JSONB,
    is_selected BOOLEAN DEFAULT FALSE,
    selected_by TEXT DEFAULT 'auto',
    generation_cost REAL,
    scoring_cost REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_candidates_project_scene
  ON generation_candidates(project_id, scene_index);

-- Audio manifests
CREATE TABLE scene_audio_manifests (
    project_id UUID REFERENCES projects(project_id),
    scene_index INTEGER NOT NULL,
    dialogue_json JSONB,
    sfx_json JSONB,
    ambient_json JSONB,
    music_json JSONB,
    audio_continuity_json JSONB,
    PRIMARY KEY (project_id, scene_index)
);
```

---

## 10. Revised Implementation Phases

### Phase 1: Manifest System Foundation
- Database: `manifests` table, updated `assets` with manifest_id
- Backend: Manifest CRUD API
- Frontend: Manifest Library view (cards with filter/sort)
- Frontend: Manifest Creator Stage 1 (upload + tag, no processing)

### Phase 2: Manifesting Engine
- Backend: YOLO detection service (local GPU)
- Backend: ArcFace face embedding + cross-matching
- Backend: Gemini reverse-prompting (per-crop)
- Backend: Contact sheet assembly (Pillow)
- Frontend: Manifest Creator Stages 2+3 (processing + review/refine)

### Phase 3: GenerateForm Integration
- Frontend: Manifest selector (select existing or quick upload)
- Backend: Manifest snapshotting
- Pipeline: Conditional Phase 0 skip with pre-built manifest
- Backend: Usage tracking

### Phase 4: Manifest-Aware Storyboarding + Audio Manifest
- Enhanced storyboard LLM with manifest context
- Scene manifests + audio manifests
- Prompt rewriting with reverse_prompt injection

### Phase 5: Veo Reference Passthrough + Clean Sheets
- 3-reference selection logic per scene
- Reference preprocessing (bg removal, clean sheet generation)
- Hybrid first-frame + reference images
- Clean sheet UI in Manifest Creator

### Phase 6: CV Analysis Pipeline + Progressive Enrichment
- Post-generation YOLO + face matching + CLIP
- Asset extraction from generated content -> manifest
- Continuity scoring

### Phase 7: Multi-Candidate Quality Mode
- sampleCount configuration
- Candidate scoring pipeline
- Candidate comparison UI
- Manual override selection

### Phase 8: Fork System Integration with Manifests
- Fork inherits manifest reference
- Fork can switch manifests
- Invalidation rules for manifest changes

---

## 11. API Endpoints

```
# Manifest CRUD
GET    /api/manifests                              # List user's manifests
POST   /api/manifests                              # Create new (DRAFT)
GET    /api/manifests/{id}                         # Get details
PUT    /api/manifests/{id}                         # Update metadata
DELETE /api/manifests/{id}                         # Delete

# Processing
POST   /api/manifests/{id}/process                 # Trigger YOLO + reverse-prompting
GET    /api/manifests/{id}/progress                # SSE/poll for progress
POST   /api/manifests/{id}/reprocess               # Re-run all

# Assets within Manifest
GET    /api/manifests/{id}/assets                  # List assets
POST   /api/manifests/{id}/assets                  # Add new upload
PUT    /api/manifests/{id}/assets/{aid}            # Edit asset
DELETE /api/manifests/{id}/assets/{aid}            # Remove asset
POST   /api/manifests/{id}/assets/{aid}/reprocess  # Re-process single
POST   /api/manifests/{id}/assets/{aid}/clean-sheet # Generate clean ref
POST   /api/manifests/{id}/assets/{aid}/swap-image # Swap reference

# Versioning
POST   /api/manifests/{id}/duplicate               # Duplicate manifest
GET    /api/manifests/{id}/contact-sheet            # Get contact sheet

# Quality Mode
GET    /api/projects/{id}/scenes/{idx}/candidates          # List candidates
PUT    /api/projects/{id}/scenes/{idx}/candidates/{cid}/select  # Override
```

---

## 12. Key Architectural Decisions

### Snapshot vs. Live Reference
**Chosen: Snapshot at generation start.** Editing a manifest after generation starts does NOT affect in-progress projects. Completed projects always reference the exact manifest state used.

### Manifest Ownership
**Chosen: User-scoped with future sharing potential.** Schema supports expansion to team/org scope, public manifests, template starter kits.

### Asset Storage
**Chosen: Assets belong to manifests.** Same image in two manifests = two asset records (shared GCS files). Keeps manifests self-contained.

### Quick Upload Conversion
**Chosen: Always create a manifest behind the scenes.** Even inline uploads create an auto-manifest. After generation, user prompted to save to library. Keeps pipeline uniform.