# Phase 11: Multi-Candidate Quality Mode - Research

**Researched:** 2026-02-16
**Domain:** Veo multi-candidate generation, composite scoring pipeline, candidate comparison UI
**Confidence:** HIGH

---

## Summary

Phase 11 adds Quality Mode to video generation: instead of generating a single clip per scene (sampleCount=1), users can request 2-4 candidates and the system automatically scores and selects the best one. This is a pure addition on top of the existing video_gen.py pipeline — nothing is rewritten, just extended.

The Veo SDK already supports multi-candidate generation via `number_of_videos` in `GenerateVideosConfig` (Python SDK name; REST API calls it `sampleCount`). The response returns a list of `generated_videos`, making it straightforward to collect multiple candidates in a single API call. The scoring pipeline reuses every service already built in Phases 9 and 10: CLIP embeddings for continuity, ArcFace for character identity, and Gemini Vision for semantic quality. The composite score is a weighted sum across four dimensions (manifest adherence 0.35, visual quality 0.25, continuity 0.25, prompt adherence 0.15).

The largest decision point is storage: each candidate needs its own video file and score record. The spec defines a `generation_candidates` table. The selected candidate's local_path replaces what currently goes into the `video_clips` table. The UI requires a new `CandidateComparisonPanel` within `SceneCard` that shows thumbnail grids with scores and allows manual override.

**Primary recommendation:** Add `quality_mode` and `candidate_count` columns to the `projects` table. Extend `_submit_video_job` to pass `number_of_videos`. Store all candidates in `generation_candidates`. Wire scoring to reuse existing `CVAnalysisService`. Expose two new API endpoints: GET candidates and PUT select.

---

## Standard Stack

### Core (already present — no new dependencies)

| Service | Location | Purpose | Phase Introduced |
|---------|----------|---------|-----------------|
| `CVAnalysisService` | `services/cv_analysis_service.py` | YOLO + face + CLIP + Gemini Vision orchestrator | Phase 9 |
| `CLIPEmbeddingService` | `services/clip_embedding_service.py` | 512-dim CLIP embeddings for visual continuity | Phase 9 |
| `FaceMatchingService` | `services/face_matching.py` | ArcFace 512-dim embeddings for character identity | Phase 5 |
| `CVDetectionService` | `services/cv_detection.py` | YOLO object + face bounding boxes | Phase 5 |
| `frame_sampler` | `services/frame_sampler.py` | Extract 5-8 key frames from video clips | Phase 9 |
| `PromptRewriterService` | `services/prompt_rewriter.py` | LLM-assembled prompts with manifest context | Phase 10 |

### New Code Required

| Component | Type | Purpose |
|-----------|------|---------|
| `CandidateScoringService` | New service | Orchestrate composite score across 4 dimensions |
| `generation_candidates` DB table | New ORM model | Store per-candidate clips, individual scores, composite |
| `GenerationCandidate` ORM | New model | One row per candidate per scene |
| Two API endpoints | Routes extension | GET candidates, PUT select |
| `CandidateComparisonPanel` React component | New UI component | Show all candidates with scores; manual override |

### No New Python Dependencies

All scoring uses services already instantiated in video_gen.py. Gemini Vision calls go through the existing `vertex_client`. The `number_of_videos` parameter is already in `google-genai==1.63.0` (confirmed from installed SDK).

---

## Architecture Patterns

### How `number_of_videos` Works in the SDK

Confirmed by inspecting installed `google-genai==1.63.0`:

```python
# Source: local SDK inspection
# types.GenerateVideosConfig has field: number_of_videos: Optional[int]
# Response: GenerateVideosResponse.generated_videos: Optional[list[GeneratedVideo]]

video_config = types.GenerateVideosConfig(
    aspect_ratio=project.aspect_ratio,
    duration_seconds=8,
    number_of_videos=candidate_count,  # 1-4
    last_frame=types.Image(image_bytes=end_frame_bytes, mime_type="image/png"),
    # ...
)
operation = await client.aio.models.generate_videos(
    model=video_model,
    prompt=video_prompt,
    image=types.Image(image_bytes=start_frame_bytes, mime_type="image/png"),
    config=video_config,
)
# After polling operation.done:
# operation.response.generated_videos is a list with candidate_count items
for i, gen_video in enumerate(operation.response.generated_videos):
    video_bytes = gen_video.video.video_bytes  # or gen_video.video.gcs_uri
```

**Critical:** With `number_of_videos > 1`, the Veo API still returns a single operation. All candidates are bundled in `operation.response.generated_videos` when the operation completes. The poll loop does not change — just the response handling.

### Project Schema Extension

Add two new columns to the `projects` table:

```sql
-- migrate_phase11.sql
ALTER TABLE projects ADD COLUMN quality_mode BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE projects ADD COLUMN candidate_count INTEGER NOT NULL DEFAULT 1;
```

`candidate_count` is 1 for Standard Mode, 2-4 for Quality Mode. The UI should validate 1 <= candidate_count <= 4.

### New `generation_candidates` Table

```sql
CREATE TABLE generation_candidates (
    id TEXT PRIMARY KEY,                      -- UUID as TEXT (existing pattern)
    project_id TEXT NOT NULL REFERENCES projects(id),
    scene_index INTEGER NOT NULL,
    candidate_number INTEGER NOT NULL,        -- 0-based index within batch
    local_path TEXT,                          -- Saved video file path
    thumbnail_url TEXT,                       -- First frame as thumbnail (optional, future)
    manifest_adherence_score REAL,            -- 0-10
    visual_quality_score REAL,               -- 0-10
    continuity_score REAL,                   -- 0-10
    prompt_adherence_score REAL,             -- 0-10
    composite_score REAL,                    -- Weighted sum 0-10
    scoring_details TEXT,                    -- JSON blob with per-score details
    is_selected BOOLEAN NOT NULL DEFAULT FALSE,
    selected_by TEXT DEFAULT 'auto',         -- 'auto' or 'user'
    generation_cost REAL DEFAULT 0.0,
    scoring_cost REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_candidates_project_scene ON generation_candidates(project_id, scene_index);
```

Note: existing project uses SQLite with TEXT UUIDs (confirmed from `database_url` pattern in config). Use TEXT for id fields, not UUID type.

### Composite Score Formula

```python
SCORE_WEIGHTS = {
    "manifest_adherence": 0.35,  # Face matching + spatial analysis
    "visual_quality": 0.25,       # Gemini Vision quality assessment
    "continuity": 0.25,           # CLIP similarity N-1 last frame vs candidate first frame
    "prompt_adherence": 0.15,     # Gemini Vision "does this match the description?"
}

def compute_composite_score(scores: dict) -> float:
    return sum(
        scores.get(dim, 0.0) * weight
        for dim, weight in SCORE_WEIGHTS.items()
    )
```

### Scoring Pipeline Per Candidate

Each candidate is scored independently:

1. **Manifest Adherence (0.35):** Run face matching against manifest assets (ArcFace). Score = `(matched_faces / expected_faces) * 10`. If no character assets expected, use CLIP similarity vs reference images instead.

2. **Visual Quality (0.25):** Gemini Vision call with first frame + question: "Rate this video frame 0-10 for visual quality: sharpness, coherence, absence of artifacts." Return just the score.

3. **Continuity (0.25):** CLIP cosine similarity between last frame of scene N-1 (already extracted by Phase 9's CV analysis on the previous scene) and first frame of this candidate. Scale to 0-10. If scene_index=0, skip and score 10.0 (first scene has no continuity requirement).

4. **Prompt Adherence (0.15):** Gemini Vision call with first frame + rewritten_video_prompt: "Does this frame match the following scene description? Score 0-10: {prompt}." Single number response.

**Cost optimization:** Manifest adherence and continuity use local GPU (free). Visual quality and prompt adherence use a single batched Gemini call to reduce API calls. Use Gemini 2.5 Flash (not Pro).

### Storing Last-Frame for Continuity

The continuity score requires the last frame of scene N-1. Phase 9's frame_sampler already extracts frames including the last frame (frame index = total_frames - 1). The `CVAnalysisResult` stores `clip_embeddings` but they're excluded from the JSON stored in `scene_manifests.cv_analysis_json`.

Two options:
- **Option A:** Re-extract last frame from the selected clip's local_path when scoring scene N. Low overhead (one opencv call).
- **Option B:** Store last-frame CLIP embedding in SceneManifest after selection. Requires schema addition.

**Recommendation:** Option A. Re-extract the last frame of the SELECTED candidate from the previous scene during scoring. The selected candidate's local_path is known by the time the next scene is being scored.

### Recommended Project Structure Change

Extend `video_gen.py`'s `_generate_video_for_scene` to handle multi-candidate flow:

```
Standard Mode (candidate_count=1):
  _submit_video_job() → returns 1 video → save to VideoClip → done

Quality Mode (candidate_count=2-4):
  _submit_video_job(number_of_videos=N) → returns N videos
  For each candidate:
    → save video file (project_id/scene_idx/candidate_N.mp4)
    → create GenerationCandidate record
  → run CandidateScoringService on all candidates
  → pick winner (highest composite_score)
  → mark winner.is_selected = True
  → save winner.local_path as VideoClip.local_path (stitcher uses this)
```

The stitcher reads from `VideoClip.local_path` — this doesn't change. Only `_generate_video_for_scene` needs to know about Quality Mode.

### API Endpoints

```
GET /api/projects/{id}/scenes/{idx}/candidates
  → Returns list of GenerationCandidate with all scores

PUT /api/projects/{id}/scenes/{idx}/candidates/{cid}/select
  → Deselects current winner, marks cid as selected (selected_by='user')
  → Updates VideoClip.local_path to point to newly selected candidate
```

### Anti-Patterns to Avoid

- **Scoring during polling:** Do not run scoring WHILE waiting for Veo to complete. Score AFTER all candidates are downloaded. The Veo operation returns all candidates together.
- **Blocking stitcher:** The stitcher reads VideoClip.local_path. Always ensure exactly one candidate is marked selected and VideoClip.local_path points to it before the stitching phase begins.
- **Scoring all scenes in parallel before any scene finishes:** Maintain the existing sequential scene generation. Scene N's scoring uses scene N-1's selected clip for continuity — they must complete in order.
- **Double-billing for scoring:** Gemini Vision calls for scoring should use Flash (not Pro). Two calls per candidate (visual quality + prompt adherence) = ~$0.01-0.03 per candidate per scene.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLIP continuity between scenes | Custom frame similarity | `CLIPEmbeddingService.compute_similarity()` | Already exists, tested, handles normalization |
| Face matching for manifest adherence | Custom face detector | `FaceMatchingService` + `CVDetectionService` | Existing pipeline with YOLO + ArcFace |
| Gemini Vision quality assessment | Own vision model | `CVAnalysisService._run_semantic_analysis()` pattern | Established pattern with retry, structured output |
| Multiple video downloads | Custom async downloader | Existing `_download_from_gcs` or `gen_video.video.video_bytes` | Already handles both bytes and GCS URI |
| File path conventions | Custom path builder | `FileManager.save_clip()` pattern | Ensures consistent directory structure |

---

## Common Pitfalls

### Pitfall 1: Veo Bills for All Candidates Regardless of Selection

**What goes wrong:** User generates 4 candidates, doesn't like any, regenerates. Billed for all 8 Veo submissions.
**Why it happens:** Veo charges per second of video generated, not per candidate selected.
**How to avoid:** Surface cost estimate prominently in Quality Mode UI. Show "Standard: $3.20, Quality Mode 2x: $6.40" before submission. The spec explicitly calls this out.
**Warning signs:** User confusion about why costs doubled.

### Pitfall 2: RAI Filtering with Multiple Candidates

**What goes wrong:** Veo generates 4 candidates but RAI filters 2 of them. `rai_media_filtered_count > 0` but some candidates survive. Current code treats any filtered count as full failure.
**Why it happens:** `_is_content_policy_operation` checks `rai_media_filtered_count > 0` and returns True even if other candidates succeeded.
**How to avoid:** In Quality Mode, check if at least one candidate survived (`len(operation.response.generated_videos) > 0`). Only escalate if ZERO candidates survive.

### Pitfall 3: Continuity Scoring for Scene 0

**What goes wrong:** Continuity scorer tries to load scene N-1 clip for scene 0, crashes.
**Why it happens:** scene_index check missing.
**How to avoid:** In `CandidateScoringService`, when `scene_index == 0`, set continuity_score = 10.0 unconditionally (no previous scene to compare).

### Pitfall 4: VideoClip Table Mismatch After Manual Override

**What goes wrong:** User manually selects candidate B via PUT endpoint. VideoClip.local_path still points to candidate A (auto-selected). Stitcher uses the wrong file.
**Why it happens:** PUT endpoint updates `generation_candidates.is_selected` but forgets to update `video_clips.local_path`.
**How to avoid:** The PUT endpoint MUST update both: set `old_winner.is_selected=False`, set `new_winner.is_selected=True`, AND update `VideoClip.local_path = new_winner.local_path`.

### Pitfall 5: Standard Mode Projects Fail with New Schema

**What goes wrong:** Projects created before Phase 11 (without `quality_mode` and `candidate_count` columns) fail to load.
**Why it happens:** SQLAlchemy reads all mapped columns on model load.
**How to avoid:** Migration SQL adds columns with DEFAULT values. Verify migration file includes `DEFAULT FALSE` and `DEFAULT 1`.

### Pitfall 6: Scoring Cost Not Tracked

**What goes wrong:** `scoring_cost` in `generation_candidates` stays 0.0 even after Gemini Vision calls.
**Why it happens:** Vision API token costs are not automatically tracked.
**How to avoid:** `CandidateScoringService` must estimate and record the scoring cost (approximately $0.01 per candidate for two Flash calls).

---

## Code Examples

### Submit Multi-Candidate Veo Job

```python
# Source: local SDK inspection of google-genai==1.63.0
# In _submit_video_job, add number_of_videos to GenerateVideosConfig:
candidate_count = project.candidate_count if project.quality_mode else 1

video_config = types.GenerateVideosConfig(
    aspect_ratio=project.aspect_ratio,
    duration_seconds=duration_seconds,
    number_of_videos=candidate_count,  # 1-4
    last_frame=types.Image(image_bytes=end_frame_bytes, mime_type="image/png"),
    negative_prompt="...",
)
# Response: operation.response.generated_videos is a list of len candidate_count
```

### Score One Candidate

```python
# Composite score computation
async def score_candidate(
    candidate_video_path: str,
    scene_index: int,
    scene_manifest_json: dict,
    rewritten_video_prompt: str,
    existing_assets: list[Asset],
    previous_scene_clip_path: Optional[str],  # None for scene 0
) -> dict:
    scores = {}

    # 1. Manifest adherence — local GPU (free)
    cv_result = await cv_service.analyze_generated_content(
        scene_index=scene_index,
        keyframe_paths=None,
        clip_path=candidate_video_path,
        scene_manifest_json=scene_manifest_json,
        existing_assets=existing_assets,
    )
    # Use semantic analysis manifest_adherence if available, else face match ratio
    if cv_result.semantic_analysis:
        scores["manifest_adherence"] = cv_result.semantic_analysis.manifest_adherence
    else:
        expected = len(scene_manifest_json.get("asset_tags", []))
        matched = sum(1 for m in cv_result.face_matches if not m.is_new)
        scores["manifest_adherence"] = min(10.0, (matched / max(expected, 1)) * 10)

    # 2. Visual quality — Gemini Vision (call batched with prompt adherence)
    scores["visual_quality"] = await _score_visual_quality_gemini(candidate_video_path)

    # 3. Continuity — CLIP similarity (local GPU, free)
    if scene_index > 0 and previous_scene_clip_path:
        prev_last_frame = extract_last_frame(previous_scene_clip_path)
        candidate_first_frame = extract_first_frame(candidate_video_path)
        prev_emb = clip_service.generate_embedding(prev_last_frame)
        cand_emb = clip_service.generate_embedding(candidate_first_frame)
        sim = CLIPEmbeddingService.compute_similarity(prev_emb, cand_emb)
        # Scale [-1,1] → [0,10]
        scores["continuity"] = max(0.0, (sim + 1.0) / 2.0 * 10.0)
    else:
        scores["continuity"] = 10.0  # First scene: no continuity requirement

    # 4. Prompt adherence — Gemini Vision
    scores["prompt_adherence"] = await _score_prompt_adherence_gemini(
        candidate_video_path, rewritten_video_prompt
    )

    # Composite
    weights = {
        "manifest_adherence": 0.35,
        "visual_quality": 0.25,
        "continuity": 0.25,
        "prompt_adherence": 0.15,
    }
    composite = sum(scores[dim] * w for dim, w in weights.items())
    return {**scores, "composite": composite}
```

### Extract First/Last Frame for Continuity

```python
# Pattern: use cv2 inside function (Phase 9 convention)
def extract_first_frame(clip_path: str, output_path: str) -> str:
    import cv2
    cap = cv2.VideoCapture(clip_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_path, frame)
    return output_path

def extract_last_frame(clip_path: str, output_path: str) -> str:
    import cv2
    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_path, frame)
    return output_path
```

### API Endpoint: List Candidates

```python
@router.get("/api/projects/{project_id}/scenes/{scene_idx}/candidates")
async def list_candidates(project_id: str, scene_idx: int):
    async with async_session() as session:
        result = await session.execute(
            select(GenerationCandidate)
            .where(
                GenerationCandidate.project_id == project_id,
                GenerationCandidate.scene_index == scene_idx,
            )
            .order_by(GenerationCandidate.candidate_number)
        )
        candidates = result.scalars().all()
        return [CandidateResponse.from_orm(c) for c in candidates]
```

### API Endpoint: Select Candidate

```python
@router.put("/api/projects/{project_id}/scenes/{scene_idx}/candidates/{candidate_id}/select")
async def select_candidate(project_id: str, scene_idx: int, candidate_id: str):
    async with async_session() as session:
        # Deselect all candidates for this scene
        all_result = await session.execute(
            select(GenerationCandidate).where(
                GenerationCandidate.project_id == project_id,
                GenerationCandidate.scene_index == scene_idx,
            )
        )
        all_candidates = all_result.scalars().all()
        for c in all_candidates:
            c.is_selected = False

        # Select the chosen candidate
        chosen = next((c for c in all_candidates if str(c.id) == candidate_id), None)
        if not chosen:
            raise HTTPException(404, "Candidate not found")
        chosen.is_selected = True
        chosen.selected_by = "user"

        # CRITICAL: update VideoClip.local_path to point to selected candidate
        scene_result = await session.execute(
            select(Scene).where(
                Scene.project_id == project_id,
                Scene.scene_index == scene_idx,
            )
        )
        scene = scene_result.scalar_one_or_none()
        if scene:
            clip_result = await session.execute(
                select(VideoClip).where(VideoClip.scene_id == scene.id)
            )
            clip = clip_result.scalar_one_or_none()
            if clip:
                clip.local_path = chosen.local_path

        await session.commit()
        return {"selected": candidate_id, "selected_by": "user"}
```

### Frontend: GenerateRequest Extension

```typescript
// Add to GenerateRequest in types.ts
export interface GenerateRequest {
  // ... existing fields ...
  quality_mode?: boolean;    // false = Standard Mode, true = Quality Mode
  candidate_count?: number;  // 2-4, only relevant when quality_mode=true
}

// Add to types.ts
export interface CandidateScore {
  candidate_id: string;
  candidate_number: number;
  local_path: string | null;
  manifest_adherence_score: number | null;
  visual_quality_score: number | null;
  continuity_score: number | null;
  prompt_adherence_score: number | null;
  composite_score: number | null;
  is_selected: boolean;
  selected_by: string;
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `sampleCount` (REST API name) | `number_of_videos` in Python SDK | SDK 1.x | Must use SDK field name, not REST name |
| All frames analyzed for CV | 5-8 key frames sampled | Phase 9 | Scoring fast enough for multi-candidate |
| Single clip per scene | N candidates → score → select best | Phase 11 (this) | Quality Mode feature |
| Static storyboard prompts | Rewritten prompts from Phase 10 | Phase 10 | Rewritten prompt feeds into prompt_adherence scoring |

---

## Open Questions

1. **Thumbnail generation for candidates**
   - What we know: The spec mentions `thumbnail_url` in `generation_candidates`
   - What's unclear: How to generate it efficiently (first frame extraction, serve as static file?)
   - Recommendation: Extract first frame of each candidate video as JPEG, save alongside MP4. Use FileManager pattern. Not required for MVP scoring — add as enhancement if time allows.

2. **Scoring Gemini calls: one call vs two calls per candidate**
   - What we know: Spec lists visual_quality and prompt_adherence as separate scoring dimensions, both using Gemini Vision
   - What's unclear: Whether to batch into a single Gemini call or make two separate calls
   - Recommendation: One batched Gemini call asking for both scores simultaneously. Return JSON with `{"visual_quality": N, "prompt_adherence": N}`. Reduces API latency and cost by 50%.

3. **Parallel vs sequential candidate scoring**
   - What we know: Multiple candidates can potentially be scored in parallel since they're independent
   - What's unclear: Whether scoring 4 candidates in parallel would hit Gemini rate limits
   - Recommendation: Score candidates in parallel using `asyncio.gather()` with the existing `asyncio.Semaphore(3)` pattern from Phase 9. Limits concurrency to 3 Gemini Vision calls at once.

4. **Cost display in Quality Mode**
   - What we know: Spec says "cost impact clearly shown" as a success criterion
   - What's unclear: Where in the UI (GenerateForm vs ProgressView vs ProjectDetail)?
   - Recommendation: Show in GenerateForm when Quality Mode is toggled (live estimate update). Show actual cost in ProjectDetail from `generation_candidates.generation_cost + scoring_cost` sum.

---

## Sources

### Primary (HIGH confidence)

- Local SDK inspection: `google-genai==1.63.0` installed at `/home/ubuntu/anaconda3/envs/video-pipeline/lib/python3.14/site-packages/google/genai/` — `GenerateVideosConfig.number_of_videos` field confirmed, `GenerateVideosResponse.generated_videos: list[GeneratedVideo]` confirmed
- Codebase: `backend/vidpipe/pipeline/video_gen.py` — current video generation implementation with Phase 8/9/10 hooks
- Codebase: `backend/vidpipe/services/cv_analysis_service.py` — scoring services already available
- Codebase: `backend/vidpipe/services/clip_embedding_service.py` — CLIP similarity already implemented
- Codebase: `backend/vidpipe/services/face_matching.py` — ArcFace already implemented
- Spec: `docs/v2-manifest.md` Section 7 — scoring weights, dimensions, DB schema
- Spec: `docs/v2-pipe-optimization.md` — sampleCount 1-4, Veo 3.1 constraints

### Secondary (MEDIUM confidence)

- Veo REST API docs (via WebSearch): `sampleCount` range 1-4 confirmed for Veo 3.1; Python SDK uses `number_of_videos` as field name

---

## Metadata

**Confidence breakdown:**

- SDK parameter: HIGH — confirmed from installed SDK source code
- Standard stack: HIGH — all services exist and are used in production (Phases 9/10)
- Database schema: HIGH — follows exact SQLite TEXT UUID pattern already in codebase
- Architecture patterns: HIGH — extends existing video_gen.py patterns (non-invasive)
- Scoring weights: HIGH — taken verbatim from spec document
- UI patterns: MEDIUM — new CandidateComparisonPanel follows SceneCard component patterns but is new

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (30 days — stable tech domain, SDK version pinned)
