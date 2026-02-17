---
phase: 09-cv-analysis-pipeline-and-progressive-enrichment
plan: "01"
subsystem: cv-analysis
tags:
  - clip-embeddings
  - frame-sampling
  - motion-detection
  - asset-appearance
  - db-models
  - configuration
dependency_graph:
  requires:
    - "05-01 (CVDetectionService lazy-loading pattern)"
    - "08-01 (AssetCleanReference model pattern)"
  provides:
    - "CLIPEmbeddingService for visual similarity"
    - "frame_sampler for video frame extraction"
    - "AssetAppearance model for detection tracking"
    - "CVAnalysisConfig for tunable thresholds"
  affects:
    - "09-02 (CV analysis orchestrator will compose these services)"
tech_stack:
  added:
    - "transformers (CLIPModel, CLIPProcessor) - lazy-loaded"
    - "torch (device detection, no_grad context)"
    - "opencv-python cv2 (video frame extraction, motion detection)"
    - "Pillow PIL (image loading for CLIP)"
  patterns:
    - "Lazy model loading (import + load inside method, not at module level)"
    - "cv2 imported inside functions to gracefully handle missing opencv"
    - "numpy.tobytes() for binary embedding storage"
    - "Idempotent ALTER TABLE migrations with try/except"
key_files:
  created:
    - backend/vidpipe/services/clip_embedding_service.py
    - backend/vidpipe/services/frame_sampler.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/config.py
    - config.yaml
decisions:
  - "Store clip_embedding as bytes (numpy.tobytes()) matching face_embedding pattern for 10x storage reduction vs JSON"
  - "cv2 imported inside frame_sampler functions (not top-level) to avoid ImportError when opencv not installed"
  - "CLIPEmbeddingService._load_model() auto-detects CUDA via torch.cuda.is_available() if device not specified"
  - "CVAnalysisConfig uses Field(default_factory=CVAnalysisConfig) so cv_analysis section is optional in config.yaml"
  - "extract_frames() reads sequentially and saves only target frames for efficiency (avoids random seeks)"
metrics:
  duration: "2 min"
  completed_date: "2026-02-16"
  tasks_completed: 2
  files_modified: 6
---

# Phase 9 Plan 01: CV Analysis Foundation Services Summary

CLIP embedding service with lazy-loaded openai/clip-vit-base-patch32, video frame sampler with motion delta detection, AssetAppearance ORM model, Asset.clip_embedding column, and CVAnalysisConfig with 5 tunable thresholds.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | CLIP embedding service and video frame sampler | 0668422 | clip_embedding_service.py, frame_sampler.py |
| 2 | AssetAppearance model, Asset.clip_embedding, cv_analysis config | dbc4c89 | models.py, __init__.py, config.py, config.yaml |

## What Was Built

### CLIPEmbeddingService (clip_embedding_service.py)
- Lazy-loads `openai/clip-vit-base-patch32` via HuggingFace transformers on first use
- `generate_embedding(image_path)`: PIL image load → CLIPProcessor → model.get_image_features() → L2-normalized 512-dim numpy float32 array
- `compute_similarity(emb1, emb2)`: static cosine similarity (dot product of unit vectors)
- Auto-detects CUDA via `torch.cuda.is_available()` when device=None
- RuntimeError with troubleshooting guidance if loading fails

### Frame Sampler (frame_sampler.py)
- `sample_video_frames()`: 5 base frames (first, 2s, 4s, 6s, last) + motion delta frames, capped at max_frames=8
- `detect_motion_deltas()`: cv2.absdiff between consecutive grayscale frames, threshold at 30 pixel intensity, ratio > 0.15 triggers inclusion
- `extract_frame()`: single frame extraction to `tmp/cv_analysis/frame_{N}.jpg`
- `extract_frames()`: batch extraction reading sequentially (efficient for multiple indices)
- cv2 imported inside each function to avoid import failures

### Database Models (models.py)
- `Asset.clip_embedding`: BLOB column (nullable), stores numpy.tobytes() 512-dim float32, matches face_embedding pattern
- `AssetAppearance`: new model with asset_id (FK), project_id (FK), scene_index, frame_index, timestamp_sec, bbox (JSON), confidence, source ("yolo"/"face_match"/"clip_match"), created_at
- `SceneManifest.cv_analysis_json`: JSON column for storing CV analysis results dict
- `SceneManifest.continuity_score`: REAL column for overall continuity quality score

### Configuration
- `CVAnalysisConfig`: pydantic BaseModel with 5 thresholds (clip_similarity_threshold=0.65, motion_delta_threshold=0.15, max_frames_per_clip=8, quality_gate_threshold=5.0, face_match_threshold=0.6)
- `Settings.cv_analysis`: optional field with `Field(default_factory=CVAnalysisConfig)` — cv_analysis section in config.yaml is optional
- config.yaml: cv_analysis section added with all 5 defaults

### Database Migrations (db/__init__.py)
- 3 new idempotent ALTER TABLE migrations: `assets.clip_embedding BLOB`, `scene_manifests.cv_analysis_json TEXT`, `scene_manifests.continuity_score REAL`
- `AssetAppearance` table created via `Base.metadata.create_all` (new model in Base)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| backend/vidpipe/services/clip_embedding_service.py | FOUND |
| backend/vidpipe/services/frame_sampler.py | FOUND |
| Commit 0668422 (Task 1) | FOUND |
| Commit dbc4c89 (Task 2) | FOUND |
