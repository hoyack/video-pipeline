"""Candidate scoring service for Multi-Candidate Quality Mode.

Evaluates generated video candidates across four weighted dimensions:
  - manifest_adherence (0.35): via CVAnalysisService face matching
  - visual_quality     (0.25): via Gemini Flash visual assessment
  - continuity         (0.25): via CLIP embeddings between scenes
  - prompt_adherence   (0.15): via Gemini Flash prompt match assessment

Spec reference: Phase 11 - Multi-Candidate Quality Mode
"""

import asyncio
import logging
import tempfile
import os
from typing import Optional

from vidpipe.schemas.llm_vision import VisualPromptScoreOutput
from vidpipe.services.llm import get_adapter, LLMAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights — must sum to 1.0
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "manifest_adherence": 0.35,
    "visual_quality": 0.25,
    "continuity": 0.25,
    "prompt_adherence": 0.15,
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 0.001, "SCORE_WEIGHTS must sum to 1.0"


# ---------------------------------------------------------------------------
# Module-level helper functions (cv2 imported inside, per Phase 9 convention)
# ---------------------------------------------------------------------------

def _extract_first_frame(clip_path: str, output_path: str) -> str:
    """Extract the first frame from a video clip and save as JPEG.

    Args:
        clip_path: Path to the input video file.
        output_path: Path to write the extracted JPEG frame.

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If frame extraction fails.
    """
    import cv2  # noqa: PLC0415 — lazy import per Phase 9 convention

    cap = cv2.VideoCapture(clip_path)
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read first frame from {clip_path}")
        cv2.imwrite(output_path, frame)
        return output_path
    finally:
        cap.release()


def _extract_last_frame(clip_path: str, output_path: str) -> str:
    """Extract the last frame from a video clip and save as JPEG.

    Args:
        clip_path: Path to the input video file.
        output_path: Path to write the extracted JPEG frame.

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If frame extraction fails.
    """
    import cv2  # noqa: PLC0415 — lazy import per Phase 9 convention

    cap = cv2.VideoCapture(clip_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read last frame from {clip_path}")
        cv2.imwrite(output_path, frame)
        return output_path
    finally:
        cap.release()


def _frame_to_jpeg_bytes(frame_path: str) -> bytes:
    """Read a JPEG frame from disk and return its raw bytes.

    Args:
        frame_path: Path to a JPEG image file.

    Returns:
        Raw bytes of the image file.
    """
    with open(frame_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# CandidateScoringService
# ---------------------------------------------------------------------------

class CandidateScoringService:
    """Scores generation candidates across four quality dimensions.

    Child services are lazy-loaded on first use to avoid import-time overhead
    and allow graceful failure when optional dependencies (cv2, CLIP) are
    not installed.
    """

    def __init__(self, vision_adapter: Optional[LLMAdapter] = None):
        """Initialize service. Child services are lazy-loaded on first use.

        Args:
            vision_adapter: Optional LLMAdapter for visual/prompt scoring.
                If None, falls back to get_adapter("gemini-2.5-flash").
        """
        self._vision_adapter = vision_adapter
        self._cv_service = None
        self._clip_service = None

    def _get_cv_service(self):
        """Lazy getter for CVAnalysisService."""
        if self._cv_service is None:
            from vidpipe.services.cv_analysis_service import CVAnalysisService
            self._cv_service = CVAnalysisService()
        return self._cv_service

    def _get_clip_service(self):
        """Lazy getter for CLIPEmbeddingService."""
        if self._clip_service is None:
            from vidpipe.services.clip_embedding_service import CLIPEmbeddingService
            self._clip_service = CLIPEmbeddingService()
        return self._clip_service

    async def score_candidate(
        self,
        candidate_video_path: str,
        scene_index: int,
        scene_manifest_json: Optional[dict],
        rewritten_video_prompt: str,
        existing_assets: list,
        previous_scene_clip_path: Optional[str] = None,
    ) -> dict:
        """Score a single candidate video across four quality dimensions.

        Args:
            candidate_video_path: Path to the candidate .mp4 file.
            scene_index: Zero-based scene index (0 means first scene).
            scene_manifest_json: Scene manifest dict for adherence scoring.
            rewritten_video_prompt: The rewritten video prompt for this scene.
            existing_assets: List of Asset ORM objects for face matching.
            previous_scene_clip_path: Path to the previous scene's clip
                (used for continuity scoring; None if scene_index == 0).

        Returns:
            Dict with keys: manifest_adherence_score, visual_quality_score,
            continuity_score, prompt_adherence_score, composite_score,
            scoring_details, scoring_cost.
        """
        tmp_dir = tempfile.mkdtemp(prefix=f"scoring_scene{scene_index}_")
        scoring_cost = 0.0
        scoring_details: dict = {}

        try:
            # ── a. Manifest Adherence (weight 0.35) ───────────────────────────
            manifest_adherence = await self._score_manifest_adherence(
                candidate_video_path,
                scene_index,
                scene_manifest_json,
                existing_assets,
                scoring_details,
            )

            # ── b. Continuity (weight 0.25) ───────────────────────────────────
            continuity = await self._score_continuity(
                candidate_video_path,
                scene_index,
                previous_scene_clip_path,
                tmp_dir,
                scoring_details,
            )

            # ── c + d. Visual Quality + Prompt Adherence (batched Gemini call) ─
            visual_quality, prompt_adherence, gemini_cost = await self._score_visual_and_prompt(
                candidate_video_path,
                rewritten_video_prompt,
                tmp_dir,
                scoring_details,
            )
            scoring_cost += gemini_cost

            # ── Composite score ────────────────────────────────────────────────
            composite = (
                manifest_adherence * SCORE_WEIGHTS["manifest_adherence"]
                + visual_quality * SCORE_WEIGHTS["visual_quality"]
                + continuity * SCORE_WEIGHTS["continuity"]
                + prompt_adherence * SCORE_WEIGHTS["prompt_adherence"]
            )

            logger.info(
                f"Scene {scene_index} candidate scored: composite={composite:.2f} "
                f"(manifest={manifest_adherence:.1f}, visual={visual_quality:.1f}, "
                f"continuity={continuity:.1f}, prompt={prompt_adherence:.1f})"
            )

            return {
                "manifest_adherence_score": manifest_adherence,
                "visual_quality_score": visual_quality,
                "continuity_score": continuity,
                "prompt_adherence_score": prompt_adherence,
                "composite_score": composite,
                "scoring_details": scoring_details,
                "scoring_cost": scoring_cost,
            }

        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _score_manifest_adherence(
        self,
        clip_path: str,
        scene_index: int,
        scene_manifest_json: Optional[dict],
        existing_assets: list,
        scoring_details: dict,
    ) -> float:
        """Score manifest adherence via CVAnalysisService face matching.

        Returns a score in [0, 10].
        """
        try:
            cv_service = self._get_cv_service()
            result = await cv_service.analyze_generated_content(
                scene_index=scene_index,
                keyframe_paths=None,
                clip_path=clip_path,
                scene_manifest_json=scene_manifest_json,
                existing_assets=existing_assets,
            )

            # Prefer semantic_analysis.manifest_adherence if available
            if result.semantic_analysis is not None:
                score = float(result.semantic_analysis.manifest_adherence)
                scoring_details["manifest_adherence_source"] = "semantic_analysis"
                scoring_details["manifest_adherence_raw"] = score
                return min(10.0, score)

            # Fallback: count face matches vs expected characters
            expected_faces = 0
            if scene_manifest_json:
                asset_tags = scene_manifest_json.get("asset_tags", [])
                # Count CHAR_ tags as expected characters
                expected_faces = sum(1 for t in asset_tags if str(t).startswith("CHAR_"))

            matched_faces = len([m for m in result.face_matches if not m.is_new])
            score = min(10.0, (matched_faces / max(expected_faces, 1)) * 10.0)
            scoring_details["manifest_adherence_source"] = "face_count"
            scoring_details["manifest_adherence_matched"] = matched_faces
            scoring_details["manifest_adherence_expected"] = expected_faces
            return score

        except Exception as e:
            logger.warning(f"Manifest adherence scoring failed (scene {scene_index}): {e}")
            scoring_details["manifest_adherence_error"] = str(e)
            return 5.0  # neutral fallback

    async def _score_continuity(
        self,
        candidate_clip_path: str,
        scene_index: int,
        previous_clip_path: Optional[str],
        tmp_dir: str,
        scoring_details: dict,
    ) -> float:
        """Score visual continuity with the previous scene using CLIP embeddings.

        Returns a score in [0, 10]. Scene 0 always scores 10 (no prior scene).
        """
        if scene_index == 0:
            scoring_details["continuity_source"] = "first_scene"
            return 10.0

        if not previous_clip_path:
            scoring_details["continuity_source"] = "no_previous_clip"
            return 5.0  # neutral when previous clip unavailable

        try:
            clip_service = self._get_clip_service()

            # Extract frames in a thread (cv2 is CPU-bound)
            first_frame_path = os.path.join(tmp_dir, "candidate_first.jpg")
            last_frame_path = os.path.join(tmp_dir, "prev_last.jpg")

            await asyncio.to_thread(_extract_first_frame, candidate_clip_path, first_frame_path)
            await asyncio.to_thread(_extract_last_frame, previous_clip_path, last_frame_path)

            # Generate CLIP embeddings (CPU-bound, wrap in to_thread)
            emb_current = await asyncio.to_thread(clip_service.generate_embedding, first_frame_path)
            emb_previous = await asyncio.to_thread(clip_service.generate_embedding, last_frame_path)

            # Compute cosine similarity [-1, 1] and scale to [0, 10]
            similarity = clip_service.compute_similarity(emb_current, emb_previous)
            score = max(0.0, (similarity + 1.0) / 2.0 * 10.0)

            scoring_details["continuity_source"] = "clip_similarity"
            scoring_details["continuity_raw_similarity"] = similarity
            logger.info(f"Scene {scene_index} continuity: similarity={similarity:.3f} → score={score:.1f}")
            return score

        except Exception as e:
            logger.warning(f"Continuity scoring failed (scene {scene_index}): {e}")
            scoring_details["continuity_error"] = str(e)
            return 5.0  # neutral fallback

    async def _score_visual_and_prompt(
        self,
        clip_path: str,
        rewritten_video_prompt: str,
        tmp_dir: str,
        scoring_details: dict,
    ) -> tuple[float, float, float]:
        """Score visual quality and prompt adherence via a single LLM adapter call.

        Returns:
            Tuple of (visual_quality_score, prompt_adherence_score, cost_estimate).
            Scores are in [0, 10]. Cost estimate is in USD.
        """
        default_visual = 5.0
        default_prompt_adh = 5.0
        cost_estimate = 0.01  # ~$0.01 per LLM call

        try:
            # Extract first frame as JPEG bytes
            first_frame_path = os.path.join(tmp_dir, "scoring_frame.jpg")
            await asyncio.to_thread(_extract_first_frame, clip_path, first_frame_path)
            frame_bytes = await asyncio.to_thread(_frame_to_jpeg_bytes, first_frame_path)

            adapter = self._vision_adapter or get_adapter("gemini-2.5-flash")

            system_prompt = (
                "You are a professional video quality assessor. "
                "Analyze the provided video frame and return a JSON object with two scores (0-10):\n"
                "- visual_quality: Rate the sharpness, coherence, absence of artifacts, and compositional quality.\n"
                "- prompt_adherence: Rate how well this frame matches the scene description provided.\n"
            )

            user_prompt = (
                f"Scene description to evaluate against:\n{rewritten_video_prompt}\n\n"
                "Rate this frame's visual_quality and prompt_adherence (0-10 each)."
            )

            result = await adapter.analyze_image(
                image_bytes=frame_bytes,
                prompt=f"{system_prompt}\n\n{user_prompt}",
                schema=VisualPromptScoreOutput,
                mime_type="image/jpeg",
                temperature=0.1,
                max_retries=2,
            )

            visual_quality = max(0.0, min(10.0, result.visual_quality))
            prompt_adherence = max(0.0, min(10.0, result.prompt_adherence))

            scoring_details["visual_quality_source"] = "llm_adapter"
            scoring_details["prompt_adherence_source"] = "llm_adapter"

            return visual_quality, prompt_adherence, cost_estimate

        except Exception as e:
            logger.warning(f"LLM visual/prompt scoring failed: {e}. Using defaults.")
            scoring_details["llm_scoring_error"] = str(e)
            return default_visual, default_prompt_adh, 0.0

    async def score_all_candidates(
        self,
        candidates_info: list[dict],
        scene_index: int,
        scene_manifest_json: Optional[dict],
        rewritten_video_prompt: str,
        existing_assets: list,
        previous_scene_clip_path: Optional[str] = None,
    ) -> list[dict]:
        """Score all candidates in parallel with Semaphore(3) for rate limiting.

        Args:
            candidates_info: List of dicts, each with at minimum
                ``{"local_path": str}`` for the candidate video path.
            scene_index: Zero-based scene index.
            scene_manifest_json: Scene manifest dict for adherence scoring.
            rewritten_video_prompt: Rewritten video prompt for prompt adherence.
            existing_assets: List of Asset ORM objects for face matching.
            previous_scene_clip_path: Path to previous scene clip for continuity.

        Returns:
            List of score dicts in the same order as ``candidates_info``.
        """
        semaphore = asyncio.Semaphore(3)  # limit concurrent Gemini calls

        async def _score_with_semaphore(info: dict) -> dict:
            async with semaphore:
                return await self.score_candidate(
                    candidate_video_path=info["local_path"],
                    scene_index=scene_index,
                    scene_manifest_json=scene_manifest_json,
                    rewritten_video_prompt=rewritten_video_prompt,
                    existing_assets=existing_assets,
                    previous_scene_clip_path=previous_scene_clip_path,
                )

        results = await asyncio.gather(
            *[_score_with_semaphore(info) for info in candidates_info],
            return_exceptions=False,
        )
        return list(results)
