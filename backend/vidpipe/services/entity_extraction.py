"""Entity extraction and registration from CV analysis results.

Identifies new entities detected in generated content that do not match
existing assets, and registers qualifying ones into the Asset Registry
after quality-gating via reverse-prompting.

Design constraint: Extracted assets do NOT auto-add to scene manifests.
They enrich the Asset Registry only. Manifests remain "intent"
(storyboard-driven); CV analysis is "validation."
"""

import asyncio
import logging
import uuid
from typing import Optional

import numpy as np
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, Manifest
from vidpipe.services.clip_embedding_service import CLIPEmbeddingService
from vidpipe.services.cv_analysis_service import CVAnalysisResult, FaceMatchResult
from vidpipe.services.face_matching import FaceMatchingService
from vidpipe.services.reverse_prompt_service import ReversePromptService

logger = logging.getLogger(__name__)

# YOLO COCO class → Asset type mapping
_VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "boat",
    "train", "airplane",
}
_PROP_CLASSES = {
    "chair", "couch", "bed", "dining table", "toilet", "desk",
    "bench", "sofa",
}
_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
}


def _yolo_class_to_asset_type(detection_class: str) -> str:
    """Map YOLO COCO class names to asset types.

    Args:
        detection_class: YOLO class name (e.g. "person", "car", "chair")

    Returns:
        Asset type string: "CHARACTER", "VEHICLE", "PROP", or "OBJECT"
    """
    cls = detection_class.lower()
    if cls == "person":
        return "CHARACTER"
    if cls in _VEHICLE_CLASSES:
        return "VEHICLE"
    if cls in _PROP_CLASSES:
        return "PROP"
    if cls in _ANIMAL_CLASSES:
        return "OBJECT"
    # Default for all other COCO classes (electronics, food, sports, etc.)
    return "OBJECT"


def _compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute intersection-over-union between two bounding boxes.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU score in range [0.0, 1.0]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class NewEntityDetection(BaseModel):
    """A detected entity not matching any existing asset."""

    crop_path: str
    bbox: list[float]
    detection_class: str  # YOLO class: "person", "car", etc.
    confidence: float
    frame_index: int
    suggested_type: str  # Inferred: "CHARACTER", "VEHICLE", "OBJECT", etc.
    source: str  # "KEYFRAME_EXTRACT" or "CLIP_EXTRACT"


def identify_new_entities(
    analysis_result: CVAnalysisResult,
    existing_assets: list[Asset],
    clip_service: Optional[CLIPEmbeddingService] = None,
) -> list[NewEntityDetection]:
    """Find detections that did not match any existing asset.

    For face detections: any FaceMatchResult where is_new=True and
    confidence > 0.5 from the face detection (faces extracted from persons).

    For object detections: any YOLO detection not matched to an existing
    asset (by class, using CLIP similarity if available).

    Deduplicates by proximity: if two detections overlap >70% IoU, keep
    the higher-confidence one.

    Args:
        analysis_result: CVAnalysisResult from analyze_generated_content()
        existing_assets: Assets from Asset Registry (for filtering already-known)
        clip_service: Optional CLIPEmbeddingService for CLIP-based matching

    Returns:
        List of NewEntityDetection sorted by confidence descending
    """
    new_entities: list[NewEntityDetection] = []

    # Track face bboxes that are "new" (unmatched)
    new_face_bboxes: set[tuple] = set()
    for face_match in analysis_result.face_matches:
        if face_match.is_new and face_match.similarity >= 0.5:
            new_face_bboxes.add(tuple(face_match.bbox))

    # Collect all per-frame object detections as candidates
    for fd in analysis_result.frame_detections:
        for obj in fd.objects:
            bbox = obj["bbox"]
            cls = obj["class"]
            confidence = obj["confidence"]
            suggested_type = _yolo_class_to_asset_type(cls)

            # For "person" class: check if this person bbox contains a new face
            # (person detections that were matched to existing assets are excluded)
            if cls == "person":
                # Check if any known-new face overlaps with this person's face region
                x1, y1, x2, y2 = bbox
                face_region = [x1, y1, x2, y1 + (y2 - y1) * 0.4]
                is_new_person = any(
                    _compute_iou(face_region, list(fb)) > 0.3
                    for fb in new_face_bboxes
                )
                # Also check if this person was matched to an existing asset
                matched_asset_tags = {
                    m.matched_asset_tag
                    for m in analysis_result.face_matches
                    if not m.is_new and m.frame_index == fd.frame_index
                }
                if matched_asset_tags:
                    # Matched — skip this person detection
                    continue
                if not is_new_person and new_face_bboxes:
                    # Face matching ran and this person wasn't flagged as new
                    continue

            # Non-person objects: check against existing asset detection classes
            elif cls != "person":
                existing_classes = {
                    a.detection_class for a in existing_assets
                    if a.detection_class is not None
                }
                # Simple heuristic: if this YOLO class appears in existing assets, skip
                if cls in existing_classes:
                    continue

            new_entities.append(
                NewEntityDetection(
                    crop_path="",  # Will be populated by caller with actual crop
                    bbox=bbox,
                    detection_class=cls,
                    confidence=confidence,
                    frame_index=fd.frame_index,
                    suggested_type=suggested_type,
                    source="CLIP_EXTRACT",
                )
            )

    # Deduplicate by IoU > 0.7 — keep highest confidence per overlapping group
    deduped: list[NewEntityDetection] = []
    for candidate in sorted(new_entities, key=lambda e: e.confidence, reverse=True):
        is_duplicate = any(
            _compute_iou(candidate.bbox, kept.bbox) > 0.7
            for kept in deduped
        )
        if not is_duplicate:
            deduped.append(candidate)

    # Sort by confidence descending
    deduped.sort(key=lambda e: e.confidence, reverse=True)
    logger.info(
        f"identify_new_entities: {len(deduped)} new entities found "
        f"(from {len(new_entities)} raw candidates)"
    )
    return deduped


async def extract_and_register_new_entities(
    session: AsyncSession,
    project_id: uuid.UUID,
    manifest_id: uuid.UUID,
    scene_index: int,
    new_entities: list[NewEntityDetection],
    source: str = "CLIP_EXTRACT",
) -> list[Asset]:
    """Reverse-prompt, quality-gate, and register new entities as assets.

    For each NewEntityDetection:
    1. Quality gate via ReversePromptService.reverse_prompt_asset()
       - If quality_score < settings.cv_analysis.quality_gate_threshold → skip
    2. Generate face_embedding for CHARACTER entities (try/except ValueError)
    3. Generate CLIP embedding
    4. Auto-generate manifest_tag (CHAR_01, OBJ_02, etc.)
    5. Create Asset record and add to session (caller commits)

    Uses asyncio.Semaphore(3) to rate-limit concurrent Gemini calls.

    Args:
        session: Active database session (caller manages commit)
        project_id: Project UUID (for logging context)
        manifest_id: Target manifest for new assets
        scene_index: Scene index (for logging)
        new_entities: New entity detections from identify_new_entities()
        source: Source tag for Asset.source column

    Returns:
        List of newly created Asset objects (not yet committed)
    """
    from vidpipe.config import settings
    from vidpipe.services.manifest_service import TAG_PREFIX_MAP

    semaphore = asyncio.Semaphore(3)
    reverse_service = ReversePromptService()
    face_service = FaceMatchingService()
    clip_service = CLIPEmbeddingService()

    registered_assets: list[Asset] = []

    # Fetch existing asset count per type for tag generation
    async def _get_type_count(asset_type: str) -> int:
        """Count existing assets of given type in this manifest."""
        result = await session.execute(
            select(func.count(Asset.id)).where(
                Asset.manifest_id == manifest_id,
                Asset.asset_type == asset_type,
            )
        )
        return result.scalar() or 0

    for idx, entity in enumerate(new_entities):
        # Validate crop path exists
        from pathlib import Path as _Path
        if not entity.crop_path or not _Path(entity.crop_path).exists():
            logger.warning(
                f"Scene {scene_index}: entity {idx} has no valid crop_path, skipping"
            )
            continue

        async with semaphore:
            # Step 1: Quality gate via reverse-prompting
            try:
                reverse_result = await reverse_service.reverse_prompt_asset(
                    image_path=entity.crop_path,
                    asset_type=entity.suggested_type,
                )
            except Exception as exc:
                logger.warning(
                    f"Scene {scene_index}: reverse-prompt failed for entity {idx}: {exc}"
                )
                continue

            quality_score = float(reverse_result.get("quality_score", 0.0))
            threshold = settings.cv_analysis.quality_gate_threshold

            if quality_score < threshold:
                logger.info(
                    f"Scene {scene_index}: Entity skipped: "
                    f"quality {quality_score:.1f} < threshold {threshold:.1f}"
                )
                continue

            # Step 2: Face embedding for CHARACTER
            face_embedding_bytes: Optional[bytes] = None
            if entity.suggested_type == "CHARACTER":
                try:
                    face_emb = await asyncio.to_thread(
                        face_service.generate_embedding, entity.crop_path
                    )
                    face_embedding_bytes = face_emb.tobytes()
                except ValueError:
                    logger.info(
                        f"Scene {scene_index}: No face in CHARACTER crop, "
                        "skipping face embedding"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Scene {scene_index}: face embedding failed: {exc}"
                    )

            # Step 3: CLIP embedding
            clip_embedding_bytes: Optional[bytes] = None
            try:
                clip_emb = await asyncio.to_thread(
                    clip_service.generate_embedding, entity.crop_path
                )
                clip_embedding_bytes = clip_emb.tobytes()
            except Exception as exc:
                logger.warning(
                    f"Scene {scene_index}: CLIP embedding failed: {exc}"
                )

            # Step 4: Generate manifest_tag
            asset_type = entity.suggested_type
            prefix = TAG_PREFIX_MAP.get(asset_type, "OTHER")
            existing_count = await _get_type_count(asset_type)
            # Account for assets registered in this batch
            same_type_in_batch = sum(
                1 for a in registered_assets if a.asset_type == asset_type
            )
            tag_index = existing_count + same_type_in_batch + 1
            manifest_tag = f"{prefix}_{tag_index:02d}"

            # Determine asset name
            suggested_name = reverse_result.get("suggested_name") or ""
            asset_name = suggested_name if suggested_name else f"Extracted {manifest_tag}"

            # Step 5: Create Asset record
            sort_order = existing_count + same_type_in_batch
            asset = Asset(
                manifest_id=manifest_id,
                asset_type=asset_type,
                name=asset_name,
                manifest_tag=manifest_tag,
                source=source,
                reference_image_url=entity.crop_path,
                reverse_prompt=reverse_result.get("reverse_prompt"),
                visual_description=reverse_result.get("visual_description"),
                quality_score=quality_score,
                detection_class=entity.detection_class,
                detection_confidence=entity.confidence,
                crop_bbox=entity.bbox,
                face_embedding=face_embedding_bytes,
                clip_embedding=clip_embedding_bytes,
                sort_order=sort_order,
            )
            # Step 6: Add to session (caller commits)
            session.add(asset)
            registered_assets.append(asset)

            logger.info(
                f"Scene {scene_index}: Registered new asset {manifest_tag} "
                f"({asset_type}) '{asset_name}' — quality={quality_score:.1f}"
            )

    # Step 7: Update manifest asset_count
    if registered_assets:
        manifest = await session.get(Manifest, manifest_id)
        if manifest is not None:
            manifest.asset_count = (manifest.asset_count or 0) + len(registered_assets)

    logger.info(
        f"Scene {scene_index}: extract_and_register_new_entities complete — "
        f"{len(registered_assets)}/{len(new_entities)} entities registered"
    )
    return registered_assets
