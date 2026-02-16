# Phase 6: GenerateForm Integration - Research

**Researched:** 2026-02-16
**Domain:** Frontend Form Design, Database Snapshotting, Conditional Pipeline Execution, Manifest Selection UX
**Confidence:** HIGH

## Summary

Phase 6 integrates the manifest system (Phases 4-5) into the video generation workflow by adding manifest selection to GenerateForm, implementing snapshot isolation for manifests at project creation, and conditionally skipping Phase 0 (manifesting) when a pre-built manifest is selected. This phase bridges the manifest library with the generation pipeline, allowing users to reuse previously processed assets while maintaining isolation between projects (editing a manifest won't affect in-progress videos). The key technical challenges are: (1) manifest selector UI with preview, (2) manifest snapshot creation on project start, (3) conditional pipeline state machine that skips manifesting when manifest_id is present, and (4) usage tracking updates.

The phase builds on existing patterns: Phase 4 established manifest CRUD and the library view; Phase 5 added processing and the creator stages. Phase 6 completes the loop by making manifests consumable in the generation flow. Users gain two workflows: "Select Existing Manifest" (reuses library manifest, skips Phase 0, saves ~$0.40 and 30-60s) or "Quick Upload" (inline upload, creates auto-manifest, runs Phase 0 as before).

**Primary recommendation:** Add manifest selector component to GenerateForm using existing ManifestCard pattern, create `manifest_snapshots` table for immutable references, modify pipeline orchestrator state machine to conditionally skip manifesting state when `project.manifest_id IS NOT NULL`, and update manifest usage tracking via database trigger or explicit service call.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 19+ | GenerateForm manifest selector component | Already in project, functional components with hooks established pattern |
| FastAPI | 0.115+ | Snapshot creation and usage tracking endpoints | Already in project for all API routes |
| SQLAlchemy 2.0 | 2.0+ | `manifest_snapshots` table and async ORM | Already in project, async session pattern established |
| Pydantic | 2.0+ | Request/response schemas for manifest selection | Already in project, all API uses pydantic models |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| clsx | - | Conditional className in manifest selector | Already in project, used in all form components |
| None needed | - | All requirements met by existing stack | Phase 6 is integration, not new tech |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Snapshot table | Copy manifest JSON into project record | Simpler schema but violates normalization, harder to track manifest history |
| Explicit snapshot API call | Auto-snapshot on project create | More control but adds complexity, explicit call could fail separately |
| Pipeline state machine change | Always run Phase 0, skip internally | Cleaner state transitions but wastes DB writes for noop manifesting |

**Installation:**
```bash
# No new dependencies — Phase 6 uses existing stack
```

## Architecture Patterns

### Recommended Project Structure
```
backend/vidpipe/
├── db/
│   └── models.py                    # Add ManifestSnapshot model
├── api/
│   └── routes.py                    # Enhanced generate endpoint
├── orchestrator/
│   └── pipeline.py                  # Conditional manifesting skip
└── services/
    └── manifest_service.py          # Snapshot creation, usage tracking

frontend/src/
├── components/
│   ├── GenerateForm.tsx             # Add manifest selector section
│   └── ManifestSelector.tsx         # NEW: Radio + manifest card preview
└── api/
    ├── client.ts                    # GenerateRequest includes manifest_id
    └── types.ts                     # Updated interfaces
```

### Pattern 1: Manifest Snapshot Isolation
**What:** When a project is created with a manifest, freeze the manifest's current state (all assets + metadata) in a separate `manifest_snapshots` table. Projects reference the snapshot, not the live manifest. Editing the live manifest never affects existing projects.

**When to use:** Any time versioned data needs to be frozen at a point in time for immutability. Studio-grade pipelines require this — "dailies" from yesterday shouldn't change because the asset library was updated today.

**Example:**
```python
# backend/vidpipe/db/models.py
class ManifestSnapshot(Base):
    """Immutable snapshot of a manifest at project creation time.

    Spec reference: V2 Manifest System, Section 5
    """
    __tablename__ = "manifest_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    manifest_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("manifests.id"), index=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id"), index=True
    )
    version_at_snapshot: Mapped[int] = mapped_column(Integer)
    snapshot_data: Mapped[dict] = mapped_column(JSON)  # Full manifest + assets serialized
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

# backend/vidpipe/services/manifest_service.py
async def create_snapshot(
    manifest_id: uuid.UUID,
    project_id: uuid.UUID,
    session: AsyncSession
) -> ManifestSnapshot:
    """Freeze manifest state at project creation.

    Serializes manifest + all assets into snapshot_data JSON.
    Future edits to manifest/assets do NOT affect this snapshot.
    """
    # Fetch manifest with all assets
    manifest = await session.get(Manifest, manifest_id,
                                 options=[selectinload(Manifest.assets)])

    # Serialize to JSON
    snapshot_data = {
        "manifest": {
            "id": str(manifest.id),
            "name": manifest.name,
            "description": manifest.description,
            "category": manifest.category,
            "tags": manifest.tags,
            "contact_sheet_url": manifest.contact_sheet_url,
            "version": manifest.version,
        },
        "assets": [
            {
                "id": str(asset.id),
                "asset_type": asset.asset_type,
                "name": asset.name,
                "manifest_tag": asset.manifest_tag,
                "reference_image_url": asset.reference_image_url,
                "reverse_prompt": asset.reverse_prompt,
                "visual_description": asset.visual_description,
                "detection_class": asset.detection_class,
                "quality_score": asset.quality_score,
                "face_embedding": asset.face_embedding.hex() if asset.face_embedding else None,
                # ... all asset fields
            }
            for asset in manifest.assets
        ]
    }

    snapshot = ManifestSnapshot(
        manifest_id=manifest_id,
        project_id=project_id,
        version_at_snapshot=manifest.version,
        snapshot_data=snapshot_data
    )
    session.add(snapshot)
    await session.commit()
    return snapshot
```

**Why this pattern:**
- Completed projects always reference the exact assets used during generation
- Manifest edits don't break existing projects (referential integrity)
- Snapshots enable manifest version history tracking
- Aligns with professional VFX "freeze asset versions for production" workflow

### Pattern 2: Conditional State Machine Execution
**What:** Pipeline state machine checks `project.manifest_id` on startup. If present, skip `manifesting` state entirely and jump to `storyboarding`. If null, run manifesting first (quick upload case).

**When to use:** State machines where certain states are optional based on initial conditions. Common in workflow engines.

**Example:**
```python
# backend/vidpipe/orchestrator/pipeline.py
async def run_pipeline(project_id: uuid.UUID):
    """Execute generation pipeline with conditional manifesting."""
    async with async_session() as session:
        project = await session.get(Project, project_id)

        # Determine starting state
        if project.manifest_id:
            # Pre-built manifest selected — skip manifesting
            logger.info(f"Project {project_id} using manifest {project.manifest_id}, skipping Phase 0")
            project.status = "storyboarding"
            starting_phase = "storyboarding"
        else:
            # Quick upload or no manifest — run manifesting first
            logger.info(f"Project {project_id} has no manifest, starting with Phase 0")
            project.status = "manifesting"
            starting_phase = "manifesting"

        await session.commit()

        # Execute state machine from starting phase
        if starting_phase == "manifesting":
            await run_manifesting_phase(project_id)
            project.status = "storyboarding"
            await session.commit()

        await run_storyboarding_phase(project_id)
        # ... rest of pipeline
```

**Why this pattern:**
- Saves API cost (~$0.40) and time (~30-60s) when reusing manifests
- State machine remains linear (no branches mid-execution)
- Database status field always reflects current phase accurately
- Easy to debug — log shows "skipped manifesting" vs "ran manifesting"

### Pattern 3: Manifest Selector with Radio Toggle + Preview
**What:** Radio buttons toggle between "Select Existing Manifest" and "Quick Upload (inline)". When "Select Existing" is active, show manifest card preview with asset summary. When "Quick Upload" is active, show drag-drop uploader (existing AssetUploader component).

**When to use:** Forms with mutually exclusive input methods. Common in e-commerce (ship to address vs pickup), payment (credit card vs PayPal), content creation (template vs from scratch).

**Example:**
```tsx
// frontend/src/components/ManifestSelector.tsx
import { useState, useEffect } from "react";
import { listManifests, getManifestDetail } from "../api/client";
import { ManifestCard } from "./ManifestCard";
import { AssetUploader } from "./AssetUploader";

interface ManifestSelectorProps {
  selectedManifestId: string | null;
  onManifestSelect: (manifestId: string | null) => void;
  onQuickUpload: (files: File[]) => void;
}

export function ManifestSelector({
  selectedManifestId,
  onManifestSelect,
  onQuickUpload,
}: ManifestSelectorProps) {
  const [mode, setMode] = useState<"existing" | "quick">("existing");
  const [manifests, setManifests] = useState([]);
  const [selectedManifest, setSelectedManifest] = useState(null);

  useEffect(() => {
    listManifests({ sort_by: "last_used_at", sort_order: "desc" })
      .then(setManifests);
  }, []);

  useEffect(() => {
    if (selectedManifestId) {
      getManifestDetail(selectedManifestId).then(setSelectedManifest);
    }
  }, [selectedManifestId]);

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <label className="flex items-center gap-2">
          <input
            type="radio"
            checked={mode === "existing"}
            onChange={() => setMode("existing")}
          />
          <span>Select Existing Manifest</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            checked={mode === "quick"}
            onChange={() => setMode("quick")}
          />
          <span>Quick Upload (inline)</span>
        </label>
      </div>

      {mode === "existing" ? (
        <div>
          {selectedManifest ? (
            <div className="border rounded p-4">
              <ManifestCard manifest={selectedManifest} compact />
              <button onClick={() => onManifestSelect(null)}>
                Change Manifest
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-4">
              {manifests.slice(0, 6).map((m) => (
                <button
                  key={m.id}
                  onClick={() => onManifestSelect(m.id)}
                  className="text-left"
                >
                  <ManifestCard manifest={m} compact />
                </button>
              ))}
            </div>
          )}
        </div>
      ) : (
        <AssetUploader onUpload={onQuickUpload} />
      )}
    </div>
  );
}
```

**Why this pattern:**
- Clear mutual exclusion — user can't accidentally do both
- Preview shows manifest summary before committing to generation
- Reuses existing ManifestCard component (DRY principle)
- Quick upload path creates auto-manifest transparently (user sees upload UI, backend handles manifest creation)

### Pattern 4: Usage Tracking with Database Trigger or Service Call
**What:** When a project is created with a manifest, increment `manifests.times_used` and update `manifests.last_used_at`. Can be done via database trigger (automatic) or explicit service call (controlled).

**When to use:** Metrics tracking that should happen atomically with a related operation. Database triggers are reliable but harder to test; service calls are explicit but could be forgotten.

**Example (Service Call Approach — Recommended):**
```python
# backend/vidpipe/services/manifest_service.py
async def increment_usage(
    manifest_id: uuid.UUID,
    session: AsyncSession
) -> None:
    """Update usage tracking when manifest is selected for a project."""
    manifest = await session.get(Manifest, manifest_id)
    manifest.times_used += 1
    manifest.last_used_at = datetime.now(timezone.utc)
    await session.commit()

# backend/vidpipe/api/routes.py
@router.post("/api/generate")
async def generate_video(request: GenerateRequest):
    async with async_session() as session:
        # Create project
        project = Project(
            prompt=request.prompt,
            manifest_id=request.manifest_id,
            # ...
        )
        session.add(project)
        await session.flush()  # Get project.id

        # Create snapshot if manifest provided
        if request.manifest_id:
            await manifest_service.create_snapshot(
                request.manifest_id, project.id, session
            )
            # Update usage tracking
            await manifest_service.increment_usage(
                request.manifest_id, session
            )

        await session.commit()
        # ...
```

**Why this pattern:**
- Explicit call makes dependency clear in code
- Easy to test (mock service, verify call)
- Can add business logic (e.g., don't track if project fails immediately)
- No hidden database behavior (triggers are "magic")

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deep copy manifest JSON | Manual dict iteration with nested loops | `copy.deepcopy(manifest.__dict__)` or serialize via pydantic | Handles nested structures, references, edge cases like circular refs |
| Manifest preview thumbnails | Custom image grid layout | Reuse existing `ManifestCard` component with `compact` prop | DRY principle, consistent styling, already tested |
| Conditional form validation | Manual if/else chains in submit handler | Pydantic schema with `Optional[manifest_id]` XOR `reference_uploads` | Declarative validation, clear error messages, type safety |
| Usage tracking metrics | Custom incrementer logic | Database TIMESTAMP DEFAULT and counter column with atomic increment | Handles concurrency, time zone correctness, atomic operations |

**Key insight:** Phase 6 is 90% integration of existing components. Don't rebuild ManifestCard, AssetUploader, or CRUD services — compose them. The only net-new code is the snapshot table and conditional state machine logic.

## Common Pitfalls

### Pitfall 1: Snapshot Data Staleness
**What goes wrong:** Project references `manifest_id` directly without snapshot. User edits manifest. Generated video now shows different assets than what the project was created with. Continuity breaks. User is confused.

**Why it happens:** Developer assumes foreign key reference is sufficient for data integrity. Doesn't account for mutable referenced data.

**How to avoid:** Always create snapshot on project creation. Projects should reference `manifest_snapshots.id`, not `manifests.id` directly. The `project.manifest_id` field is metadata for "which manifest was selected" but generation uses the snapshot.

**Warning signs:**
- User reports "the video doesn't match the manifest I used"
- Editing manifest changes existing project behavior
- No `manifest_snapshots` table in schema

### Pitfall 2: Quick Upload Auto-Manifest Leaks
**What goes wrong:** User does quick upload. Auto-manifest is created with status=DRAFT. Manifest processing fails. Manifest stays in DRAFT forever. Manifest Library fills with orphaned auto-manifests. User never requested these manifests explicitly.

**Why it happens:** Quick upload creates manifest as implementation detail, not user-visible entity. No cleanup logic for failed auto-manifests.

**How to avoid:**
1. Mark auto-manifests with `category="AUTO_GENERATED"` or similar flag
2. After project completes, prompt user: "Save this manifest to library?" (default: No)
3. If No, soft-delete the auto-manifest after pipeline completes
4. Manifest Library filters out `category=AUTO_GENERATED` by default

**Warning signs:**
- Manifest Library shows many manifests user doesn't recognize
- Manifest count grows faster than projects created
- User never visited Manifest Creator but has manifests

### Pitfall 3: State Machine Deadlock on Conditional Skip
**What goes wrong:** Pipeline checks `manifest_id`, skips manifesting, but assets aren't actually loaded. Storyboarding fails because Asset Registry is empty. Project stuck in `storyboarding` state with cryptic error.

**Why it happens:** Conditional skip logic only checked `manifest_id IS NOT NULL`, didn't validate snapshot exists or has assets.

**How to avoid:**
1. Validate snapshot exists immediately after skipping manifesting
2. Load snapshot data into memory/session before storyboarding
3. Log asset count from snapshot: "Loaded 11 assets from snapshot {snapshot_id}"
4. Add explicit check in storyboarding: if no assets available, fail fast with clear error

**Warning signs:**
- Project status stuck at `storyboarding` with error "No assets found"
- Logs show "Skipping manifesting" but no "Loaded assets from snapshot"
- Storyboarding phase tries to query `assets` table by `project_id` (wrong — should query snapshot)

### Pitfall 4: Frontend Preview Shows Outdated Manifest
**What goes wrong:** User selects manifest A, previews it, then manifest A is edited by another user/tab. Preview still shows old data. User generates video expecting old data, gets new data.

**Why it happens:** Frontend caches manifest detail, doesn't re-fetch when manifest is selected. Race condition between selection and snapshot creation.

**How to avoid:**
1. Always fetch fresh manifest detail when user clicks "Select" or "Generate"
2. Show manifest version number in preview
3. Backend snapshot creation validates manifest.version matches request (optimistic locking)
4. If version mismatch, reject with clear error: "Manifest was updated, please re-select"

**Warning signs:**
- User reports "generated video doesn't match preview"
- Manifest shows different asset counts in different views
- Concurrent edits to same manifest cause unexpected behavior

## Code Examples

Verified patterns from design docs and existing codebase:

### GenerateForm Enhancement with Manifest Selector
```tsx
// frontend/src/components/GenerateForm.tsx
import { ManifestSelector } from "./ManifestSelector";

export function GenerateForm({ onGenerated }: GenerateFormProps) {
  const [selectedManifestId, setSelectedManifestId] = useState<string | null>(null);
  const [quickUploadFiles, setQuickUploadFiles] = useState<File[]>([]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    const payload = {
      prompt,
      style,
      aspect_ratio: aspectRatio,
      // ... existing fields
      manifest_id: selectedManifestId,  // NEW: optional manifest reference
      reference_uploads: quickUploadFiles.length > 0
        ? await Promise.all(quickUploadFiles.map(fileToBase64))
        : undefined,  // NEW: inline uploads
    };

    const res = await generateVideo(payload);
    onGenerated(res.project_id);
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* Existing prompt, style, aspect ratio, etc. */}

      {/* NEW: Manifest Selector Section */}
      <div className="space-y-2">
        <h3 className="text-lg font-medium">Asset Manifest</h3>
        <p className="text-sm text-gray-400">
          Choose reference images from a pre-built manifest or upload inline.
        </p>
        <ManifestSelector
          selectedManifestId={selectedManifestId}
          onManifestSelect={setSelectedManifestId}
          onQuickUpload={setQuickUploadFiles}
        />
      </div>

      <button type="submit">Generate Video</button>
    </form>
  );
}
```

### Backend Generate Endpoint with Snapshot Creation
```python
# backend/vidpipe/api/routes.py
class GenerateRequest(BaseModel):
    prompt: str
    style: str
    aspect_ratio: str
    # ... existing fields
    manifest_id: Optional[str] = None  # NEW: pre-built manifest
    reference_uploads: Optional[list[ReferenceUpload]] = None  # NEW: quick upload

@router.post("/api/generate")
async def generate_video(request: GenerateRequest):
    async with async_session() as session:
        project = Project(
            prompt=request.prompt,
            style=request.style,
            manifest_id=uuid.UUID(request.manifest_id) if request.manifest_id else None,
            status="pending",
            # ...
        )
        session.add(project)
        await session.flush()

        # Snapshot creation for pre-built manifests
        if request.manifest_id:
            snapshot = await manifest_service.create_snapshot(
                uuid.UUID(request.manifest_id),
                project.id,
                session
            )
            await manifest_service.increment_usage(
                uuid.UUID(request.manifest_id),
                session
            )
            logger.info(
                f"Project {project.id} using manifest {request.manifest_id}, "
                f"snapshot {snapshot.id} created"
            )

        # Quick upload creates auto-manifest
        elif request.reference_uploads:
            # Create auto-manifest
            auto_manifest = Manifest(
                name=f"Auto-manifest for {project.prompt[:30]}",
                category="AUTO_GENERATED",
                status="DRAFT",
            )
            session.add(auto_manifest)
            await session.flush()

            # Add uploaded assets to auto-manifest
            for upload in request.reference_uploads:
                asset = Asset(
                    manifest_id=auto_manifest.id,
                    name=upload.name,
                    asset_type=upload.asset_type,
                    # ...
                )
                session.add(asset)

            project.manifest_id = auto_manifest.id
            # Note: auto-manifest goes through normal manifesting phase

        await session.commit()

        # Spawn background pipeline
        asyncio.create_task(run_pipeline(project.id))

        return {"project_id": str(project.id), "status": project.status}
```

### Conditional Pipeline State Machine
```python
# backend/vidpipe/orchestrator/pipeline.py
async def run_pipeline(project_id: uuid.UUID):
    """Main pipeline orchestrator with conditional manifesting."""
    async with async_session() as session:
        project = await session.get(Project, project_id)

        # Phase 0: Manifesting (conditional)
        if project.manifest_id:
            # Validate snapshot exists
            snapshot = await session.execute(
                select(ManifestSnapshot)
                .where(ManifestSnapshot.project_id == project_id)
            )
            snapshot = snapshot.scalar_one_or_none()

            if not snapshot:
                raise ValueError(
                    f"Project {project_id} references manifest {project.manifest_id} "
                    f"but snapshot does not exist"
                )

            asset_count = len(snapshot.snapshot_data["assets"])
            logger.info(
                f"Project {project_id}: Skipping manifesting, "
                f"loaded {asset_count} assets from snapshot {snapshot.id}"
            )
            project.status = "storyboarding"
        else:
            logger.info(f"Project {project_id}: Running manifesting phase")
            project.status = "manifesting"
            await session.commit()

            await run_manifesting_phase(project_id)
            project.status = "storyboarding"

        await session.commit()

        # Phase 1: Storyboarding
        await run_storyboarding_phase(project_id)
        # ... rest of pipeline unchanged
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Inline reference uploads only | Manifest library + quick upload | V2 design (Phase 6) | Users can reuse asset processing, saves cost/time |
| Reference images uploaded per-project | Manifests as standalone entities | V2 design (Phase 4) | Asset pool grows across projects, not siloed |
| Static manifest reference | Snapshot isolation | V2 design (Phase 6) | Editing manifests doesn't break existing projects |
| Always run Phase 0 | Conditional skip when manifest provided | V2 design (Phase 6) | Saves ~$0.40 and 30-60s per reused manifest |

**Deprecated/outdated:**
- **Direct manifest_id reference without snapshot:** Violates immutability. Use `manifest_snapshots` table.
- **Separate "upload references" API before generate:** Merged into single generate endpoint with optional `reference_uploads` array.

## Open Questions

1. **Should auto-manifests created by quick upload be automatically saved to the library after successful generation?**
   - What we know: Quick upload creates auto-manifest, user never explicitly requested it
   - What's unclear: User intent — did they want a throwaway manifest or reusable one?
   - Recommendation: Prompt after generation completes: "Save manifest to library? [Yes] [No]". Default: Yes (optimistic). If No, soft-delete auto-manifest. Avoids orphaned manifests in library while preserving user value.

2. **What happens if a manifest is soft-deleted while a project is mid-generation?**
   - What we know: Project has snapshot, manifest is soft-deleted (deleted_at set)
   - What's unclear: Should project continue? Should UI show "Manifest Deleted" warning?
   - Recommendation: Project continues normally (snapshot is independent). ProjectDetail shows "(manifest deleted)" badge next to manifest name. No functional impact, just informational.

3. **How should manifest version mismatches be handled if snapshot fails due to concurrent edit?**
   - What we know: User selects manifest v1, another user edits to v2, snapshot tries to freeze v1 but it's gone
   - What's unclear: Should generation fail? Should it auto-use v2? Should it retry?
   - Recommendation: Fail fast with clear error: "Manifest was updated during generation request. Please re-select manifest and try again." Prevents silent behavior change. User re-selects, gets v2 explicitly.

4. **Should usage tracking count failed projects (status=failed) or only successful ones?**
   - What we know: `times_used` increments on project creation, before pipeline runs
   - What's unclear: Is a failed generation a "use" of the manifest?
   - Recommendation: Count all projects (including failed) because processing cost was incurred. Add separate `times_succeeded` column if success-only metric is needed. Failed generations still consumed API calls (storyboarding, etc.).

## Sources

### Primary (HIGH confidence)
- `/home/ubuntu/work/video-pipeline/docs/v2-manifest.md` - Section 5: GenerateForm Integration with wireframes, manifest selector design, snapshot schema, pipeline flow changes
- `/home/ubuntu/work/video-pipeline/docs/v2-pipe-optimization.md` - V2 architecture overview, asset registry, manifest lifecycle
- `/home/ubuntu/work/video-pipeline/.planning/ROADMAP.md` - Phase 6 success criteria, dependencies, prior phase context
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/models.py` - Existing Manifest, Asset, Project models from Phase 4
- `/home/ubuntu/work/video-pipeline/frontend/src/components/GenerateForm.tsx` - Current form structure, submission pattern
- `/home/ubuntu/work/video-pipeline/frontend/src/api/client.ts` - API client patterns, type definitions
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/routes.py` - Generate endpoint structure, async session usage

### Secondary (MEDIUM confidence)
- Phase 4 completion artifacts - Manifest CRUD API established, ManifestCard component exists
- Phase 5 research - Processing tasks pattern, progress polling established

### Tertiary (LOW confidence)
- None — all research derived from project docs and existing code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All requirements met by existing project dependencies, no new libraries needed
- Architecture: HIGH - Patterns match existing codebase (async SQLAlchemy, pydantic, React hooks), design docs provide detailed wireframes
- Pitfalls: HIGH - Derived from common database snapshot, state machine, and frontend preview patterns; verified against design doc constraints

**Research date:** 2026-02-16
**Valid until:** 30 days (stable integration work, unlikely to change)

## Additional Context: Design Doc Key Decisions

From `docs/v2-manifest.md` Section 5:

**Snapshot vs. Live Reference:** Chosen: Snapshot at generation start. Editing a manifest after generation starts does NOT affect in-progress projects. Completed projects always reference the exact manifest state used.

**Manifest Ownership:** Manifests are user-scoped entities. Same manifest can be used by unlimited projects. Changes to manifests do NOT retroactively affect completed projects.

**Quick Upload Conversion:** Always create a manifest behind the scenes. Even inline uploads create an auto-manifest. After generation, user prompted to save to library. Keeps pipeline uniform — all projects have a manifest, whether explicitly selected or auto-created.

**Pipeline Flow Change:**
- WITH PRE-BUILT MANIFEST: `pending → storyboarding → keyframing → video_gen → stitching → complete` (Phase 0 SKIPPED)
- WITH QUICK UPLOAD: `pending → manifesting → storyboarding → keyframing → video_gen → stitching → complete` (Phase 0 RUNS)

**Usage Tracking:** `times_used` and `last_used_at` updated when manifest is selected for a project, regardless of project success/failure.
