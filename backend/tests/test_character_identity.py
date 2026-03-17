"""End-to-end character identity pipeline test.

Resolves a cast actor from a real Production Bible, prequalifies reference
images, generates keyframes with iterative identity refinement (Levels 0-2),
then verifies the generated face matches using YOLO + ArcFace.

Requires: running backend with DB access + Vertex AI credentials.
"""

import io
import logging
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vidpipe.db.engine import async_session
from vidpipe.pipeline.keyframes import (
    _build_identity_instruction,
    _generate_image_conditioned,
    _generate_image_from_text,
)
from vidpipe.services.cv_detection import CVDetectionService
from vidpipe.services.face_matching import FaceMatchingService
from vidpipe.services.file_manager import FileManager
from vidpipe.services.ref_prequalification import prequalify_refs
from vidpipe.services.tag_resolver import resolve_tags_with_assets
from vidpipe.services.vertex_client import get_vertex_client

logger = logging.getLogger(__name__)

# ── Test constants ──────────────────────────────────────────────────────
PRODUCTION_BIBLE_ID = uuid.UUID("6815b853-59f5-4529-a188-0052224a8f08")
CAST_TAG = "BRANDON_ONE"
IMAGE_MODEL = "gemini-2.5-flash-image"
PROMPT = (
    "A futuristic tech explorer in a sleek cyberpunk outfit stands in an "
    "abandoned datacenter. Neon reflections on wet concrete floor, "
    "server racks with blinking LEDs in the background. "
    "Cinematic wide shot, 16:9 aspect ratio."
)
SIMILARITY_THRESHOLD = 0.35  # Hard assert — above noise floor (0.2), allows for generation variance
STRETCH_TARGET = 0.45  # Logged but not asserted — achievable with good generation rolls

OUTPUT_DIR = Path("/tmp/test_outputs/character_identity")


def _score_image(
    image_bytes: bytes,
    ref_embeddings: list[np.ndarray],
    face_svc: FaceMatchingService,
    cv_svc: CVDetectionService,
) -> float:
    """Score a generated image against reference embeddings. Returns best cosine similarity."""
    detected = cv_svc.detect_faces_from_bytes(image_bytes, confidence_threshold=0.3)
    if not detected:
        # Fallback: try ArcFace on full image
        try:
            emb = face_svc.generate_embedding_from_bytes(image_bytes)
            return max(
                float(FaceMatchingService.cosine_similarity(emb, ref))
                for ref in ref_embeddings
            )
        except ValueError:
            return -1.0

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    best = -1.0

    for face in detected:
        x1, y1, x2, y2 = [int(c) for c in face["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_np.shape[1], x2)
        y2 = min(img_np.shape[0], y2)
        crop = Image.fromarray(img_np[y1:y2, x1:x2])
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        try:
            emb = face_svc.generate_embedding_from_bytes(buf.getvalue())
        except ValueError:
            continue
        for ref in ref_embeddings:
            sim = float(FaceMatchingService.cosine_similarity(emb, ref))
            if sim > best:
                best = sim
    return best


@pytest.mark.asyncio
async def test_character_identity_pipeline():
    """End-to-end: resolve actor -> prequalify refs -> multi-level generation -> verify."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str):
        logger.info(msg)
        print(msg)
        report_lines.append(msg)

    # ── Step 1: Resolve @BRANDON_ONE via tag resolver ───────────────────
    log("=" * 60)
    log("Step 1: Resolve cast tag")
    log("=" * 60)

    async with async_session() as session:
        resolved = await resolve_tags_with_assets(
            prompt=f"@{CAST_TAG} {PROMPT}",
            production_bible_id=PRODUCTION_BIBLE_ID,
            session=session,
        )

    char_refs = [r for r in resolved.asset_refs if r.asset_type == "CHARACTER"]
    assert len(char_refs) >= 1, f"Expected CHARACTER ref for @{CAST_TAG}"

    char_ref = char_refs[0]
    log(f"  Tag: {char_ref.tag}")
    log(f"  Display name: {char_ref.display_name}")
    log(f"  Description: {char_ref.text_description[:80]}...")
    log(f"  Reference image URLs: {len(char_ref.reference_image_urls)}")
    assert len(char_ref.reference_image_urls) > 0, "Actor must have reference images"

    # ── Step 2: Prequalify reference images ─────────────────────────────
    log("")
    log("=" * 60)
    log("Step 2: Prequalify reference images (ArcFace face detection)")
    log("=" * 60)

    file_mgr = FileManager()
    face_svc = FaceMatchingService()
    qualified = await prequalify_refs(
        char_ref.reference_image_urls, file_mgr, face_svc,
    )

    log(f"  Qualified: {len(qualified)} / {len(char_ref.reference_image_urls)} refs")
    for i, q in enumerate(qualified):
        log(f"  Ref {i}: det_score={q.detection_score:.3f} — {q.image_url}")

    assert len(qualified) >= 1, "Must have at least 1 qualified reference"

    qualified_bytes = [q.image_bytes for q in qualified]
    ref_embeddings = [q.face_embedding for q in qualified]

    # ── Step 3: Multi-level identity generation ─────────────────────────
    log("")
    log("=" * 60)
    log("Step 3: Multi-level identity generation")
    log("=" * 60)
    log(f"  Model: {IMAGE_MODEL}")
    log(f"  Qualified refs: {len(qualified_bytes)}")
    log(f"  Character description: {char_ref.text_description[:60]}...")

    client = get_vertex_client()
    cv_svc = CVDetectionService(device="cpu")

    best_bytes: bytes | None = None
    best_sim: float = -1.0
    level_results: list[dict] = []

    for level in range(3):
        log(f"\n  --- Level {level} ---")

        identity_instr = _build_identity_instruction(
            char_ref.text_description, emphasis_level=level,
        )
        log(f"  Identity instruction: {identity_instr[:80]}...")

        if level >= 2 and best_bytes is not None:
            # Level 2: iterative refinement — use best attempt as conditioning
            log("  Using best-so-far as conditioning frame (image-to-image)")
            gen_bytes = await _generate_image_conditioned(
                client=client,
                reference_image_bytes=best_bytes,
                prompt=PROMPT,
                aspect_ratio="16:9",
                conditioned_model=IMAGE_MODEL,
                reference_images=qualified_bytes,
                identity_instruction=identity_instr,
            )
        else:
            gen_bytes = await _generate_image_from_text(
                client=client,
                prompt=PROMPT,
                aspect_ratio="16:9",
                image_model=IMAGE_MODEL,
                reference_images=qualified_bytes,
                identity_instruction=identity_instr,
            )

        assert gen_bytes and len(gen_bytes) > 1000, f"Level {level}: image too small"

        gen_img = Image.open(io.BytesIO(gen_bytes))
        log(f"  Generated: {gen_img.size[0]}x{gen_img.size[1]}, {len(gen_bytes):,} bytes")

        # Save for inspection
        level_path = OUTPUT_DIR / f"level{level}.png"
        level_path.write_bytes(gen_bytes)
        log(f"  Saved: {level_path}")

        # Score against qualified ref embeddings
        sim = _score_image(gen_bytes, ref_embeddings, face_svc, cv_svc)
        log(f"  Similarity: {sim:.4f}")

        level_results.append({"level": level, "similarity": sim})

        if sim > best_sim:
            best_sim = sim
            best_bytes = gen_bytes

        if sim >= STRETCH_TARGET:
            log(f"  Hit stretch target ({STRETCH_TARGET}) — stopping early")
            break

    # ── Step 4: Results ─────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("Step 4: Results")
    log("=" * 60)
    for r in level_results:
        log(f"  Level {r['level']}: similarity = {r['similarity']:.4f}")
    log(f"  Best similarity: {best_sim:.4f}")
    log(f"  Hard threshold: {SIMILARITY_THRESHOLD}")
    log(f"  Stretch target: {STRETCH_TARGET}")
    hit_stretch = best_sim >= STRETCH_TARGET
    log(f"  Verdict: {'PASS' if best_sim >= SIMILARITY_THRESHOLD else 'FAIL'}"
        f"{' (stretch target hit!)' if hit_stretch else ''}")

    # Save best image and report
    if best_bytes:
        (OUTPUT_DIR / "best.png").write_bytes(best_bytes)
    report_path = OUTPUT_DIR / "character_identity_report.txt"
    report_path.write_text("\n".join(report_lines))
    log(f"  Report: {report_path}")

    assert best_sim >= SIMILARITY_THRESHOLD, (
        f"Best face similarity {best_sim:.4f} < threshold {SIMILARITY_THRESHOLD}. "
        f"Check {OUTPUT_DIR} for per-level images."
    )
