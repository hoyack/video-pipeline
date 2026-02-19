"""CV analysis orchestrator service for post-generation analysis.

Composes Phase 5 services (YOLO, ArcFace) with Phase 9 additions (CLIP,
frame sampling) into a complete post-generation analysis pipeline.

Analyzes generated keyframes/clips, matches detections against the Asset
Registry, identifies new entities, and produces structured analysis results
with continuity scores.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import numpy as np
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, AssetAppearance
from vidpipe.schemas.llm_vision import SemanticAnalysisOutput
from vidpipe.services.clip_embedding_service import CLIPEmbeddingService
from vidpipe.services.cv_detection import CVDetectionService
from vidpipe.services.face_matching import FaceMatchingService
from vidpipe.services.frame_sampler import extract_frames, sample_video_frames
from vidpipe.services.llm import get_adapter, LLMAdapter

logger = logging.getLogger(__name__)


class FrameDetection(BaseModel):
    """Detection results for a single sampled frame."""

    frame_index: int
    timestamp_sec: Optional[float] = None
    objects: list[dict]  # From CVDetectionService
    faces: list[dict]  # From CVDetectionService


class FaceMatchResult(BaseModel):
    """Result of matching a detected face against Asset Registry."""

    bbox: list[float]
    frame_index: int
    matched_asset_id: Optional[uuid.UUID] = None
    matched_asset_tag: Optional[str] = None
    similarity: float = 0.0
    is_new: bool = True


class SemanticAnalysis(BaseModel):
    """Gemini Vision structured analysis of generated scene."""

    manifest_adherence: float = 0.0  # 0-10
    visual_quality: float = 0.0  # 0-10
    continuity_issues: list[str] = []
    new_entities_description: list[dict] = []
    overall_scene_description: str = ""


class CVAnalysisResult(BaseModel):
    """Complete CV analysis output for a generated scene."""

    scene_index: int
    frame_detections: list[FrameDetection] = []
    face_matches: list[FaceMatchResult] = []
    clip_embeddings: list[dict] = []  # [{frame_index, embedding_bytes}]
    semantic_analysis: Optional[SemanticAnalysis] = None
    continuity_score: float = 0.0
    new_entity_count: int = 0
    analysis_cost: float = 0.0


class CVAnalysisService:
    """Orchestrates post-generation CV analysis on keyframes and video clips."""

    def __init__(self, vision_adapter: Optional[LLMAdapter] = None):
        """Initialize service. Child services are lazy-loaded on first use.

        Args:
            vision_adapter: Optional LLMAdapter for semantic analysis.
                If None, falls back to get_adapter("gemini-2.5-flash").
        """
        self._vision_adapter = vision_adapter
        self._cv_service: Optional[CVDetectionService] = None
        self._face_service: Optional[FaceMatchingService] = None
        self._clip_service: Optional[CLIPEmbeddingService] = None

    def _get_cv_service(self) -> CVDetectionService:
        """Lazy getter for CVDetectionService."""
        if self._cv_service is None:
            self._cv_service = CVDetectionService()
        return self._cv_service

    def _get_face_service(self) -> FaceMatchingService:
        """Lazy getter for FaceMatchingService."""
        if self._face_service is None:
            self._face_service = FaceMatchingService()
        return self._face_service

    def _get_clip_service(self) -> CLIPEmbeddingService:
        """Lazy getter for CLIPEmbeddingService."""
        if self._clip_service is None:
            self._clip_service = CLIPEmbeddingService()
        return self._clip_service

    async def analyze_generated_content(
        self,
        scene_index: int,
        keyframe_paths: Optional[list[str]],
        clip_path: Optional[str],
        scene_manifest_json: Optional[dict],
        existing_assets: list[Asset],
    ) -> CVAnalysisResult:
        """Orchestrate full CV analysis on a generated scene.

        Steps:
        1. Frame extraction (from clip or keyframes)
        2. YOLO detection sweep on each frame
        3. Face embedding + matching against Asset Registry
        4. CLIP embeddings for visual similarity
        5. Gemini Vision semantic analysis (optional, requires manifest)
        6. Compute continuity score

        Args:
            scene_index: Zero-based scene index
            keyframe_paths: Paths to keyframe images (start/end frames)
            clip_path: Path to generated video clip (optional)
            scene_manifest_json: Scene manifest for semantic analysis (optional)
            existing_assets: Assets from Asset Registry for face/CLIP matching

        Returns:
            CVAnalysisResult with all collected analysis data
        """
        from vidpipe.config import settings

        result = CVAnalysisResult(scene_index=scene_index)
        cv_service = self._get_cv_service()
        face_service = self._get_face_service()
        clip_service = self._get_clip_service()

        # ── Step 1: Frame extraction ──────────────────────────────────────────
        t0 = time.time()
        frame_paths: list[str] = []

        if clip_path and Path(clip_path).exists():
            # Sample key frames from the video clip
            frame_indices = await asyncio.to_thread(
                sample_video_frames,
                clip_path,
                motion_threshold=settings.cv_analysis.motion_delta_threshold,
                max_frames=settings.cv_analysis.max_frames_per_clip,
            )
            # Extract those frames to a temp directory
            tmp_dir = mkdtemp(prefix=f"cv_scene{scene_index}_")
            frame_paths = await asyncio.to_thread(
                extract_frames, clip_path, frame_indices, tmp_dir
            )
            logger.info(
                f"Scene {scene_index}: extracted {len(frame_paths)} frames from clip "
                f"({time.time() - t0:.0f}ms)"
            )
        elif keyframe_paths:
            frame_paths = [p for p in keyframe_paths if Path(p).exists()]
            logger.info(
                f"Scene {scene_index}: using {len(frame_paths)} keyframe paths"
            )

        if not frame_paths:
            logger.warning(
                f"Scene {scene_index}: no frames available for CV analysis"
            )
            return result

        # ── Step 2: YOLO detection sweep ──────────────────────────────────────
        t1 = time.time()
        frame_detections: list[FrameDetection] = []

        for idx, frame_path in enumerate(frame_paths):
            detections = await asyncio.to_thread(
                cv_service.detect_objects_and_faces, frame_path
            )
            frame_detections.append(
                FrameDetection(
                    frame_index=idx,
                    objects=detections.get("objects", []),
                    faces=detections.get("faces", []),
                )
            )

        result.frame_detections = frame_detections
        logger.info(
            f"Scene {scene_index}: YOLO detection: "
            f"{(time.time() - t1) * 1000:.0f}ms for {len(frame_paths)} frames"
        )

        # ── Step 3: Face embedding + matching ─────────────────────────────────
        t2 = time.time()
        face_matches: list[FaceMatchResult] = []

        # Pre-compute existing asset face embeddings (deserialized)
        asset_embeddings: list[tuple[Asset, np.ndarray]] = []
        for asset in existing_assets:
            if asset.face_embedding is not None:
                emb = np.frombuffer(asset.face_embedding, dtype=np.float32)
                asset_embeddings.append((asset, emb))

        for fd in frame_detections:
            for face_det in fd.faces:
                bbox = face_det["bbox"]
                # Save face crop temporarily
                tmp_crop_path = str(
                    Path(mkdtemp(prefix=f"face_crop_s{scene_index}_f{fd.frame_index}_"))
                    / "crop.jpg"
                )
                try:
                    await asyncio.to_thread(
                        cv_service.save_crop,
                        frame_paths[fd.frame_index],
                        bbox,
                        tmp_crop_path,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Scene {scene_index}: failed to save face crop: {exc}"
                    )
                    continue

                # Generate face embedding
                try:
                    face_emb = await asyncio.to_thread(
                        face_service.generate_embedding, tmp_crop_path
                    )
                except ValueError:
                    # No face detected in crop — skip
                    continue
                except Exception as exc:
                    logger.warning(
                        f"Scene {scene_index}: face embedding failed: {exc}"
                    )
                    continue

                # Match against existing assets
                best_asset: Optional[Asset] = None
                best_sim = 0.0
                for asset, asset_emb in asset_embeddings:
                    sim = FaceMatchingService.cosine_similarity(face_emb, asset_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_asset = asset

                threshold = settings.cv_analysis.face_match_threshold
                if best_asset is not None and best_sim >= threshold:
                    face_matches.append(
                        FaceMatchResult(
                            bbox=bbox,
                            frame_index=fd.frame_index,
                            matched_asset_id=best_asset.id,
                            matched_asset_tag=best_asset.manifest_tag,
                            similarity=best_sim,
                            is_new=False,
                        )
                    )
                else:
                    face_matches.append(
                        FaceMatchResult(
                            bbox=bbox,
                            frame_index=fd.frame_index,
                            similarity=best_sim,
                            is_new=True,
                        )
                    )

        result.face_matches = face_matches
        logger.info(
            f"Scene {scene_index}: face matching: "
            f"{(time.time() - t2) * 1000:.0f}ms, "
            f"{sum(1 for m in face_matches if not m.is_new)} matched / "
            f"{len(face_matches)} total"
        )

        # ── Step 4: CLIP embeddings ───────────────────────────────────────────
        t3 = time.time()
        clip_embeddings: list[dict] = []

        for idx, frame_path in enumerate(frame_paths):
            try:
                emb = await asyncio.to_thread(
                    clip_service.generate_embedding, frame_path
                )
                clip_embeddings.append(
                    {
                        "frame_index": idx,
                        "embedding_bytes": emb.tobytes(),
                    }
                )
            except Exception as exc:
                logger.warning(
                    f"Scene {scene_index}: CLIP embedding failed for frame {idx}: {exc}"
                )

        result.clip_embeddings = clip_embeddings
        logger.info(
            f"Scene {scene_index}: CLIP embeddings: "
            f"{(time.time() - t3) * 1000:.0f}ms for {len(clip_embeddings)} frames"
        )

        # ── Step 5: Vision semantic analysis ─────────────────────────────────
        if scene_manifest_json is not None:
            semantic = await self._run_semantic_analysis(
                frame_paths=frame_paths,
                detections=frame_detections,
                scene_manifest_json=scene_manifest_json,
                face_matches=face_matches,
                vision_adapter=self._vision_adapter,
            )
            result.semantic_analysis = semantic

        # ── Step 6: Compute continuity score ─────────────────────────────────
        if result.semantic_analysis is not None:
            result.continuity_score = (
                result.semantic_analysis.manifest_adherence
                + result.semantic_analysis.visual_quality
            ) / 2.0
        else:
            result.continuity_score = 0.0

        # Count new entities from face matches
        result.new_entity_count = sum(1 for m in face_matches if m.is_new)

        logger.info(
            f"Scene {scene_index}: CV analysis complete — "
            f"continuity_score={result.continuity_score:.2f}, "
            f"new_entities={result.new_entity_count}"
        )
        return result

    async def _run_semantic_analysis(
        self,
        frame_paths: list[str],
        detections: list[FrameDetection],
        scene_manifest_json: dict,
        face_matches: list[FaceMatchResult],
        vision_adapter: Optional[LLMAdapter] = None,
    ) -> Optional[SemanticAnalysis]:
        """Run LLM Vision semantic analysis on sampled frames.

        Uses the first sampled frame with full detection context in the text
        prompt (all frames' context is embedded in the text, not as image parts).
        Returns structured SemanticAnalysis or None on failure.

        Args:
            frame_paths: Paths to extracted frames
            detections: YOLO detection results per frame
            scene_manifest_json: Scene manifest dict with expected assets
            face_matches: Face matching results
            vision_adapter: Optional LLMAdapter for vision. Falls back to
                get_adapter("gemini-2.5-flash") if None.

        Returns:
            SemanticAnalysis or None if API call fails
        """
        try:
            adapter = vision_adapter or get_adapter("gemini-2.5-flash")

            # Read first frame bytes
            if not frame_paths:
                return None
            first_frame_path = frame_paths[0]
            try:
                first_frame_bytes = Path(first_frame_path).read_bytes()
            except Exception as exc:
                logger.warning(f"Could not load first frame for semantic analysis: {exc}")
                return None

            suffix = Path(first_frame_path).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
            mime_type = mime_map.get(suffix, "image/jpeg")

            # Build detection summary for context (all frames, not just first)
            total_objects = sum(len(fd.objects) for fd in detections)
            total_faces = sum(len(fd.faces) for fd in detections)
            matched_faces = sum(1 for m in face_matches if not m.is_new)
            matched_tags = [
                m.matched_asset_tag
                for m in face_matches
                if m.matched_asset_tag is not None
            ]

            detection_summary = (
                f"Detected {total_objects} objects and {total_faces} faces across "
                f"{len(detections)} frames. "
                f"Matched {matched_faces} faces to known assets: "
                f"{', '.join(matched_tags) if matched_tags else 'none'}."
            )

            # Build manifest expectations summary
            expected_assets = scene_manifest_json.get("asset_tags", [])
            shot_type = scene_manifest_json.get("shot_type", "unknown")

            manifest_summary = (
                f"Scene manifest expects: {', '.join(expected_assets) if expected_assets else 'no specific assets'}. "
                f"Shot type: {shot_type}."
            )

            full_prompt = (
                f"Analyze this frame from a generated video scene.\n\n"
                f"{manifest_summary}\n\n"
                f"Detection results: {detection_summary}\n\n"
                f"[Analysis of {len(frame_paths)} sampled frames — showing representative first frame]\n\n"
                f"Evaluate and return a JSON object with:\n"
                f"1. manifest_adherence (0-10): How well does the scene match the manifest expectations?\n"
                f"2. visual_quality (0-10): How visually coherent and high-quality is the scene?\n"
                f"3. continuity_issues: List of specific continuity problems noticed (empty list if none).\n"
                f"4. new_entities_description: List of new/unexpected entities seen (objects, characters not in manifest).\n"
                f"5. overall_scene_description: One-sentence description of what is happening in the scene.\n"
            )

            result = await adapter.analyze_image(
                image_bytes=first_frame_bytes,
                prompt=full_prompt,
                schema=SemanticAnalysisOutput,
                mime_type=mime_type,
                temperature=0.2,
                max_retries=2,
            )

            return SemanticAnalysis(**result.model_dump())

        except Exception as exc:
            logger.warning(
                f"Semantic analysis failed (non-fatal): {exc}. "
                "Continuing without semantic results."
            )
            return None

    async def track_appearances(
        self,
        session: AsyncSession,
        project_id: uuid.UUID,
        scene_index: int,
        result: CVAnalysisResult,
    ) -> None:
        """Persist matched asset detections as AssetAppearance records.

        Creates AssetAppearance records for:
        - Face matches: assets matched via ArcFace similarity
        - Object detections with source="clip_match" for future CLIP-matched assets

        Args:
            session: Active database session (caller manages commit)
            project_id: Project UUID for the appearance record
            scene_index: Scene index for the appearance record
            result: CVAnalysisResult containing face_matches to persist
        """
        appearances: list[AssetAppearance] = []

        # Face-match appearances
        for face_match in result.face_matches:
            if face_match.matched_asset_id is None:
                continue

            appearance = AssetAppearance(
                asset_id=face_match.matched_asset_id,
                project_id=project_id,
                scene_index=scene_index,
                frame_index=face_match.frame_index,
                bbox=face_match.bbox,
                confidence=face_match.similarity,
                source="face_match",
            )
            appearances.append(appearance)

        # Object appearances matched via CLIP similarity (placeholder for Plan 03+)
        # When CLIP matching against assets is implemented, add records here with
        # source="clip_match"

        if appearances:
            session.add_all(appearances)
            logger.info(
                f"Scene {scene_index}: persisted {len(appearances)} AssetAppearance records"
            )
        else:
            logger.info(
                f"Scene {scene_index}: no matched asset appearances to persist"
            )
