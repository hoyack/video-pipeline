"""Sequential keyframe generation with visual continuity.

This module implements KEYF-01 through KEYF-06 requirements:
- Shot 0 start frame generated from text prompt (KEYF-01)
- End frames use image-conditioned generation (KEYF-02)
- Visual continuity via end-to-start frame inheritance (KEYF-03)
- Sequential processing, no parallelization (KEYF-04)
- Rate limiting with configurable delays (KEYF-05)
- Keyframe images saved as PNG files (KEYF-06)
"""

import asyncio
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from google.genai import types
from google.genai.errors import ClientError, ServerError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log,
)

from vidpipe.config import settings
from vidpipe.db.models import DEFAULT_USER_ID, Asset, Scene, Shot, Keyframe, UserSettings
from vidpipe.schemas.llm_vision import CharacterKeyframeVerificationOutput
from vidpipe.services.event_bus import emit_task_log
from vidpipe.services.file_manager import FileManager
from vidpipe.services.json_compat import normalize_json_list
from vidpipe.services.llm import LLMAdapter, get_adapter
from vidpipe.services.vertex_client import get_vertex_client, location_for_model
from vidpipe.services.model_catalog import canonical_model_id

logger = logging.getLogger(__name__)


def _detect_image_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    return "image/png"  # safe fallback for Gemini


# ---------------------------------------------------------------------------
# ComfyUI image models (routed to ComfyUI instead of Vertex AI)
# ---------------------------------------------------------------------------
COMFYUI_IMAGE_MODELS = {
    "qwen-fast",
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "flux-dev",
    "flux-dev-lora",
    "flux-dev-redux",
    "flux-dev-full",
    "flux-2-klein",
}
# ComfyUI models that consume production-bible identity references via the
# same reference-selection machinery as Nano Banana (face/wardrobe candidate
# selection, identity policy, emphasis escalation, post-gen verification).
COMFYUI_MULTIREF_IMAGE_MODELS = {"qwen-image-edit-2509", "flux-2-klein"}
_NANO_BANANA_MAX_REFERENCE_IMAGES = 13


def _uses_comfyui_vision_primary(scene: object) -> bool:
    return getattr(scene, "image_model", None) in COMFYUI_MULTIREF_IMAGE_MODELS


# ---------------------------------------------------------------------------
# Identity emphasis escalation prefixes for face verification retry
# ---------------------------------------------------------------------------
_IDENTITY_EMPHASIS_PREFIXES = [
    # Level 0: normal generation (no prefix)
    "",
    # Level 1: strong identity-matching instruction
    (
        "CRITICAL: The character's FACE must EXACTLY match the reference photo(s). "
        "Pay close attention to facial bone structure, eye shape, nose bridge, "
        "jawline, and skin tone. The generated face must be recognizable as the "
        "SAME PERSON shown in the reference images. "
    ),
    # Level 2: maximum emphasis — sacrifice composition for face identity
    (
        "ABSOLUTE PRIORITY — FACE IDENTITY: Every facial feature in the generated "
        "image must match the reference photos with photographic accuracy. "
        "The face is the single most important element of this image. Sacrifice "
        "background detail or composition fidelity if needed to preserve exact "
        "face identity. "
    ),
]


def _build_identity_instruction(
    text_description: str | list[str] | None = None,
    identity_types: list[str] | None = None,
    emphasis_level: int = 0,
) -> str:
    """Build feature-anchored identity instruction for Gemini.

    Injects specific facial features from Actor.base_appearance_prompt
    into the identity grounding instruction, improving retention ~41%
    over generic "same person" instructions.

    Args:
        text_description: Actor/base look facial features text. Can be a single
            description or multiple per-character descriptions.
        emphasis_level: 0=standard, 1=escalated, 2=maximum.

    Returns:
        Identity instruction string to prepend before reference images.
    """
    identity_types = [
        (identity_type or "HUMAN").upper()
        for identity_type in (identity_types or [])
        if identity_type
    ]
    has_non_human = any(identity_type != "HUMAN" for identity_type in identity_types)

    descriptions: list[str] = []
    if isinstance(text_description, list):
        descriptions = [d.strip() for d in text_description if d and d.strip()]
    elif text_description:
        descriptions = [text_description.strip()]

    descriptions = descriptions[:4]
    features = ""
    subject_label = "person"
    same_subject = "SAME PERSON"
    if descriptions:
        if len(descriptions) == 1:
            desc = descriptions[0][:300]
            features = f" This person's key features: {desc}"
        else:
            joined = " ".join(
                f"Character {idx + 1}: {desc[:180]}."
                for idx, desc in enumerate(descriptions)
            )
            features = f" Key identity features by character: {joined}"
            subject_label = "characters"
            same_subject = "SAME CHARACTERS"

    if has_non_human:
        if emphasis_level <= 0:
            return (
                "The following reference photo(s) show the exact recurring character(s) or subject(s) "
                "that must appear in the generated image."
                + features
                + " Match each subject precisely. For humans, preserve facial identity and proportions. "
                "For non-human characters, preserve species, markings, silhouette, texture, and other "
                "distinctive visual traits.\n\n"
            )
        if emphasis_level == 1:
            return (
                "CRITICAL IDENTITY REQUIREMENT: Match the exact recurring subjects in the reference "
                "photo(s)."
                + features
                + " For humans, preserve the same face and proportions. For non-human characters, "
                "preserve the same species, markings, fur/skin texture, silhouette, and signature traits.\n\n"
            )
        return (
            "ABSOLUTE PRIORITY — SUBJECT IDENTITY: Match the reference photo(s) with photographic accuracy."
            + features
            + " Keep the same recurring human faces, and keep the same recurring non-human species, "
            "markings, silhouette, and distinguishing visual traits even if other composition details must give way.\n\n"
        )

    if emphasis_level <= 0:
        return (
            f"The following reference photo(s) show the EXACT {subject_label} who must appear "
            "in the generated image." + features + " "
            "Match their face structure, skin tone, and distinguishing features precisely. "
            f"The generated character(s) MUST be recognizable as the {same_subject}.\n\n"
        )
    elif emphasis_level == 1:
        return (
            "CRITICAL IDENTITY REQUIREMENT: The character face(s) must EXACTLY match "
            "the reference photo(s)." + features + " "
            "Pay extreme attention to facial bone structure, eye shape and spacing, "
            "nose bridge, jawline contour, and skin tone. "
            f"The face(s) must be the {same_subject}, not merely similar.\n\n"
        )
    else:  # Level 2+
        return (
            "ABSOLUTE PRIORITY — FACE IDENTITY: Match the reference photos with "
            "photographic accuracy." + features + " "
            "The face is the most important element. Sacrifice background detail or "
            "composition fidelity if needed to preserve exact face identity.\n\n"
        )


def _is_retriable(exc: BaseException) -> bool:
    """Return True only for transient errors worth retrying."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", 0) == 429
    # Retry on connection/timeout errors
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _is_nano_banana_model(image_model: str) -> bool:
    return image_model.startswith("gemini-") and "image" in image_model


def _normalize_reference_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    clean = tag.strip().lstrip("@")
    if not clean:
        return None
    return clean.upper()


def _ordered_unique_tags(tags: list[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags or []:
        normalized = _normalize_reference_tag(tag)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _is_human_identity_type(identity_type: str | None) -> bool:
    return (identity_type or "HUMAN").upper() == "HUMAN"


@dataclass
class _ReferenceCandidate:
    tag: str
    image_url: str
    image_bytes: bytes
    asset_type: str
    source: str
    asset: Asset | None = None
    text_description: str | None = None
    identity_type: str | None = None
    reference_kind: str = "supplemental"


@dataclass
class _CharacterVerificationTarget:
    tag: str
    identity_type: str
    expected_position: str | None = None
    # Authoritative wardrobe for this shot (from the manifest placement).
    # When set, wardrobe is judged against this text — NOT against the
    # clothing worn in the identity reference photos.
    expected_wardrobe: str | None = None
    face_candidates: list[_ReferenceCandidate] = field(default_factory=list)
    wardrobe_candidates: list[_ReferenceCandidate] = field(default_factory=list)
    fallback_candidates: list[_ReferenceCandidate] = field(default_factory=list)


@dataclass
class _CharacterCropSelection:
    tag: str
    image_bytes: bytes
    bbox: list[float] | None
    used_full_frame: bool = False


@dataclass
class _CharacterCropPlan:
    selections: dict[str, _CharacterCropSelection] = field(default_factory=dict)
    detected_face_count: int = 0
    detected_person_count: int = 0
    detected_object_count: int = 0


@dataclass
class _FrameVerificationResult:
    passed: bool
    attempts: int
    summary: str
    detail: str
    status: str = "passed"


@dataclass
class _CharacterVisionVerificationResult:
    tag: str
    passed: bool
    character_visible: bool
    identity_match: bool
    wardrobe_match: bool
    identity_score: float
    wardrobe_score: float
    issues: list[str] = field(default_factory=list)
    used_full_frame: bool = False


@dataclass
class _VisionVerificationReport:
    passed: bool
    detail: str
    results: list[_CharacterVisionVerificationResult] = field(default_factory=list)


@dataclass
class _HumanFaceVerificationResult:
    tag: str
    passed: bool
    advisory: bool
    similarity: float
    detail: str


@dataclass
class _KeyframeVerificationReport:
    passed: bool
    detail: str
    verification_mode: str
    face_results: list[_HumanFaceVerificationResult] = field(default_factory=list)
    vision_report: _VisionVerificationReport | None = None


@dataclass
class _GeneratedKeyframeAttempt:
    attempt_number: int
    retry_level: int
    keyframe_bytes: bytes
    verification_report: _KeyframeVerificationReport


@dataclass
class _NanoBananaReferenceContext:
    ref_image_bytes_list: list[bytes] = field(default_factory=list)
    final_reference_tags: list[str] = field(default_factory=list)
    selected_ref_urls: list[str] = field(default_factory=list)
    character_ref_urls: list[str] = field(default_factory=list)
    placed_char_assets: list[Asset] = field(default_factory=list)
    character_text_descriptions: list[str] = field(default_factory=list)
    mandatory_character_tags: list[str] = field(default_factory=list)
    optional_reference_tags: list[str] = field(default_factory=list)
    trimmed_reference_counts: dict[str, int] = field(default_factory=dict)
    canonical_tag_remaps: dict[str, str] = field(default_factory=dict)
    identity_types_by_tag: dict[str, str] = field(default_factory=dict)
    selected_candidates: list[_ReferenceCandidate] = field(default_factory=list)


def _dedupe_reference_candidates(
    candidates: list[_ReferenceCandidate],
) -> list[_ReferenceCandidate]:
    seen_urls: set[str] = set()
    deduped: list[_ReferenceCandidate] = []
    for candidate in candidates:
        if candidate.image_url in seen_urls:
            continue
        seen_urls.add(candidate.image_url)
        deduped.append(candidate)
    return deduped


def _crop_image_bytes(
    image_bytes: bytes,
    bbox: list[float] | None,
    *,
    padding: float = 0.12,
) -> bytes:
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not bbox:
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    crop = image.crop((
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(image.width, int(x2 + pad_x)),
        min(image.height, int(y2 + pad_y)),
    ))
    out = io.BytesIO()
    crop.save(out, format="PNG")
    return out.getvalue()


def _build_character_verification_targets(
    *,
    shot_manifest_json: dict | None,
    selected_candidates: list[_ReferenceCandidate],
    identity_types_by_tag: dict[str, str],
) -> list[_CharacterVerificationTarget]:
    positions_by_tag: dict[str, str | None] = {}
    wardrobe_by_tag: dict[str, str | None] = {}
    for placement in (shot_manifest_json or {}).get("placements", []):
        tag = _normalize_reference_tag(placement.get("asset_tag"))
        if tag and tag not in positions_by_tag:
            positions_by_tag[tag] = placement.get("position")
        if tag and placement.get("wardrobe_note") and tag not in wardrobe_by_tag:
            wardrobe_by_tag[tag] = placement.get("wardrobe_note")

    targets: dict[str, _CharacterVerificationTarget] = {}
    for candidate in selected_candidates:
        if candidate.asset_type != "CHARACTER":
            continue
        target = targets.setdefault(
            candidate.tag,
            _CharacterVerificationTarget(
                tag=candidate.tag,
                identity_type=identity_types_by_tag.get(candidate.tag, candidate.identity_type or "HUMAN"),
                expected_position=positions_by_tag.get(candidate.tag),
                expected_wardrobe=wardrobe_by_tag.get(candidate.tag),
            ),
        )
        if candidate.reference_kind == "face":
            target.face_candidates.append(candidate)
        elif candidate.reference_kind == "wardrobe":
            target.wardrobe_candidates.append(candidate)
        else:
            target.fallback_candidates.append(candidate)

    return list(targets.values())


def _build_retry_reference_candidates(
    candidates: list[_ReferenceCandidate],
    *,
    mandatory_tags: list[str],
    retry_level: int,
) -> list[_ReferenceCandidate]:
    if retry_level <= 0:
        return list(candidates)

    mandatory_tag_set = set(mandatory_tags)
    mandatory_candidates = [
        candidate
        for candidate in candidates
        if candidate.asset_type == "CHARACTER" and candidate.tag in mandatory_tag_set
    ]
    if retry_level == 1:
        return mandatory_candidates or list(candidates)

    grouped: dict[str, list[_ReferenceCandidate]] = {}
    for candidate in mandatory_candidates:
        grouped.setdefault(candidate.tag, []).append(candidate)

    packed: list[_ReferenceCandidate] = []
    for tag in mandatory_tags:
        group = grouped.get(tag, [])
        if not group:
            continue
        face_candidate = next((c for c in group if c.reference_kind == "face"), None)
        wardrobe_candidate = next((c for c in group if c.reference_kind == "wardrobe"), None)
        if face_candidate:
            packed.append(face_candidate)
        if wardrobe_candidate and wardrobe_candidate is not face_candidate:
            packed.append(wardrobe_candidate)
        if not packed or packed[-1].tag != tag:
            for candidate in group:
                if candidate not in packed:
                    packed.append(candidate)
                if len([c for c in packed if c.tag == tag]) >= 2:
                    break
    return packed or mandatory_candidates or list(candidates)


def _retry_mode_label(retry_level: int) -> str:
    if retry_level <= 0:
        return "full_reference_pack"
    if retry_level == 1:
        return "character_only_pack"
    return "minimal_cast_pack"


def _best_effort_metrics(
    report: _KeyframeVerificationReport,
) -> tuple[int, int, float, float, float, int]:
    vision_results = report.vision_report.results if report.vision_report else []
    visible_count = sum(1 for result in vision_results if result.character_visible)
    passed_count = sum(1 for result in vision_results if result.passed)
    identity_total = sum(result.identity_score for result in vision_results)
    wardrobe_total = sum(result.wardrobe_score for result in vision_results)
    face_similarity_total = sum(result.similarity for result in report.face_results)
    issue_count = sum(len(result.issues) for result in vision_results)
    issue_count += sum(
        1 for result in report.face_results
        if not result.passed and not result.advisory
    )
    return (
        visible_count,
        passed_count,
        identity_total,
        wardrobe_total,
        face_similarity_total,
        issue_count,
    )


def _best_effort_sort_key(attempt: _GeneratedKeyframeAttempt) -> tuple[float, ...]:
    report = attempt.verification_report
    (
        visible_count,
        passed_count,
        identity_total,
        wardrobe_total,
        face_similarity_total,
        issue_count,
    ) = _best_effort_metrics(report)
    return (
        1.0 if report.passed else 0.0,
        float(visible_count),
        float(passed_count),
        identity_total,
        wardrobe_total,
        face_similarity_total,
        float(-issue_count),
        float(-attempt.attempt_number),
    )


def _select_best_effort_attempt(
    attempts: list[_GeneratedKeyframeAttempt],
) -> _GeneratedKeyframeAttempt | None:
    if not attempts:
        return None
    return max(attempts, key=_best_effort_sort_key)


def _build_best_effort_detail(
    *,
    position: str,
    attempts: list[_GeneratedKeyframeAttempt],
    selected_attempt: _GeneratedKeyframeAttempt,
    transport_detail: str | None = None,
) -> str:
    report = selected_attempt.verification_report
    target_count = len(report.vision_report.results if report.vision_report else [])
    (
        visible_count,
        passed_count,
        identity_total,
        wardrobe_total,
        face_similarity_total,
        issue_count,
    ) = _best_effort_metrics(report)
    detail = (
        "best_effort_fallback accepted after full verification miss. "
        f"selected_attempt={selected_attempt.attempt_number}/{len(attempts)} "
        f"position={position} retry_mode={_retry_mode_label(selected_attempt.retry_level)} "
        f"verification_mode={report.verification_mode} "
        f"crowded_best_effort={report.verification_mode == 'vision_primary_face_advisory'} "
        f"visible_targets={visible_count}/{target_count} "
        f"passed_targets={passed_count}/{target_count} "
        f"identity_total={identity_total:.1f} "
        f"wardrobe_total={wardrobe_total:.1f} "
        f"face_similarity_total={face_similarity_total:.3f} "
        f"issue_count={issue_count} || {report.detail}"
    )
    if transport_detail:
        detail += f" || transport_exhausted={transport_detail}"
    return detail


def _build_best_effort_fallback_result(
    *,
    position: str,
    attempts: list[_GeneratedKeyframeAttempt],
    transport_detail: str | None = None,
) -> tuple[_GeneratedKeyframeAttempt, _FrameVerificationResult] | None:
    selected_attempt = _select_best_effort_attempt(attempts)
    if selected_attempt is None:
        return None
    return (
        selected_attempt,
        _FrameVerificationResult(
            passed=True,
            attempts=len(attempts),
            summary="accepted_with_warnings",
            detail=_build_best_effort_detail(
                position=position,
                attempts=attempts,
                selected_attempt=selected_attempt,
                transport_detail=transport_detail,
            ),
            status="accepted_with_warnings",
        ),
    )


def _describe_retry_error(exc: RetryError) -> str:
    last_exc = exc.last_attempt.exception() if exc.last_attempt else None
    if last_exc is None:
        return "transport_retries_exhausted"
    code = getattr(last_exc, "code", None)
    if code is not None:
        return f"{type(last_exc).__name__}(code={code}): {last_exc}"
    return f"{type(last_exc).__name__}: {last_exc}"


_VISIBILITY_LIMITED_ISSUE_SNIPPETS = (
    "seen from behind",
    "face and head are not visible",
    "face is not visible",
    "no visible facial features",
    "identity cannot be verified",
    "obscured by",
    "silhouette",
    "low light",
    "backlit",
)


def _issues_indicate_visibility_limited_identity(
    issues: list[str] | None,
) -> bool:
    text = " ".join(issue.lower() for issue in (issues or []))
    return any(snippet in text for snippet in _VISIBILITY_LIMITED_ISSUE_SNIPPETS)


def _passes_partial_visibility_human_check(
    *,
    target: _CharacterVerificationTarget,
    crop: _CharacterCropSelection,
    result: CharacterKeyframeVerificationOutput,
) -> bool:
    if not _is_human_identity_type(target.identity_type):
        return False
    if not crop.used_full_frame:
        return False
    if not result.character_visible:
        return False
    if result.wardrobe_score < 6.0:
        return False
    return _issues_indicate_visibility_limited_identity(result.issues)


def _vision_can_corroborate_near_threshold_face(
    *,
    face_result: _HumanFaceVerificationResult,
    vision_result: _CharacterVisionVerificationResult | None,
) -> bool:
    if face_result.passed or face_result.advisory:
        return False
    if vision_result is None:
        return False
    if not (
        vision_result.passed
        and vision_result.character_visible
        and vision_result.identity_match
        and vision_result.wardrobe_match
    ):
        return False
    if vision_result.identity_score < 8.0 or vision_result.wardrobe_score < 7.0:
        return False

    threshold = settings.cv_analysis.keyframe_face_match_threshold
    # Historical references and generated documentary frames often miss the
    # embedding threshold on expression/pose while the multimodal verifier still
    # sees a clean identity match. Keep the band narrow enough to reject weak
    # matches, but wide enough to avoid burning repeated provider calls on
    # clean 8+/10 vision passes.
    near_threshold_margin = max(0.07, threshold * 0.15)
    return face_result.similarity >= threshold - near_threshold_margin


def _compose_numbered_reference_board(image_bytes_list: list[bytes]) -> bytes:
    """Compose candidate reference images into one numbered horizontal board."""
    from PIL import Image, ImageDraw

    panel = 320
    tiles = []
    for raw in image_bytes_list:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((panel, panel))
        tiles.append(img)
    board = Image.new("RGB", (panel * len(tiles), panel + 28), (16, 16, 16))
    draw = ImageDraw.Draw(board)
    for i, tile in enumerate(tiles):
        board.paste(tile, (i * panel, 28))
        draw.text((i * panel + 6, 4), f"IMAGE {i}", fill=(255, 255, 0))
    buf = io.BytesIO()
    board.save(buf, format="PNG")
    return buf.getvalue()


async def _qualify_multiref_identity_candidates(
    *,
    session: AsyncSession,
    scene: Scene,
    candidates: list,
    max_character_refs: int = 2,
) -> tuple[list, str]:
    """Pick the best CHARACTER reference images for multi-ref ComfyUI models.

    Occluded references (sunglasses, faces turned away) dilute identity
    conditioning in reference-latent models — observed: a lead character
    rendered as a different person when sunglasses selfies were mixed into
    the reference set. One vision call rates each candidate; the top
    ``max_character_refs`` clear-face references are kept. Non-character
    candidates pass through untouched. Any failure falls back to capping
    the character refs in their original order.
    """
    from vidpipe.schemas.llm_vision import IdentityReferenceQualificationOutput

    char_candidates = [c for c in candidates if c.asset_type == "CHARACTER"]
    other_candidates = [c for c in candidates if c.asset_type != "CHARACTER"]
    if len(char_candidates) <= max_character_refs:
        return candidates, "qualification_skipped: at or under character ref cap"

    fallback = char_candidates[:max_character_refs] + other_candidates

    try:
        vision_model_id = scene.vision_model or scene.text_model or settings.models.storyboard_llm
        user_settings = None
        if vision_model_id.startswith("ollama/"):
            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
            )
            user_settings = result.scalar_one_or_none()
        vision_adapter = get_adapter(vision_model_id, user_settings=user_settings)

        board = _compose_numbered_reference_board(
            [c.image_bytes for c in char_candidates]
        )
        result = await vision_adapter.analyze_image(
            board,
            (
                f"The board contains {len(char_candidates)} numbered photos of the same "
                "person, candidates for use as identity reference images in image "
                "generation. Rate each numbered panel for identity-reference "
                "suitability. Penalize sunglasses, occluded or averted faces, and "
                "blur. Reward sharp, frontal, unobstructed faces."
            ),
            IdentityReferenceQualificationOutput,
            mime_type="image/png",
            temperature=0.1,
            max_retries=2,
        )
        rated = {
            r.index: r for r in result.ratings if 0 <= r.index < len(char_candidates)
        }
        if not rated:
            return fallback, "qualification_fallback: vision returned no usable ratings"

        def _rank_key(i: int) -> tuple:
            r = rated.get(i)
            if r is None:
                return (0, 0, 0.0)
            return (int(r.face_clearly_visible), int(r.eyes_unobstructed), r.suitability)

        order = sorted(range(len(char_candidates)), key=_rank_key, reverse=True)
        keep = sorted(order[:max_character_refs])
        detail = "; ".join(
            f"img{i}: visible={rated[i].face_clearly_visible} "
            f"eyes={rated[i].eyes_unobstructed} score={rated[i].suitability:.1f}"
            f"{' KEPT' if i in keep else ''}"
            for i in sorted(rated)
        )
        selected = [char_candidates[i] for i in keep] + other_candidates
        return selected, f"qualification_ok: {detail}"
    except Exception as exc:
        logger.warning("Identity ref qualification failed (non-fatal): %s", exc)
        return fallback, f"qualification_fallback: {exc}"


def _pack_mandatory_reference_candidates(
    grouped_candidates: list[tuple[str, list[_ReferenceCandidate]]],
    *,
    max_reference_images: int,
) -> tuple[list[_ReferenceCandidate], dict[str, int]]:
    if max_reference_images <= 0:
        trimmed = {
            tag: len(candidates)
            for tag, candidates in grouped_candidates
            if candidates
        }
        return [], trimmed

    queues = [
        [tag, list(candidates)]
        for tag, candidates in grouped_candidates
        if candidates
    ]
    selected: list[_ReferenceCandidate] = []

    for _, queue in queues:
        if len(selected) >= max_reference_images:
            break
        selected.append(queue.pop(0))

    while len(selected) < max_reference_images:
        progressed = False
        for _, queue in queues:
            if not queue:
                continue
            selected.append(queue.pop(0))
            progressed = True
            if len(selected) >= max_reference_images:
                break
        if not progressed:
            break

    trimmed = {
        tag: len(queue)
        for tag, queue in queues
        if queue
    }
    return selected, trimmed


async def _assemble_nano_banana_reference_context(
    session: AsyncSession,
    *,
    production_bible_id,
    scene_prompt: str | None,
    shot: Shot,
    shot_manifest_json: dict | None,
    selected_reference_tags: list[str] | None,
    all_assets: list[Asset],
    file_mgr: FileManager,
    max_reference_images: int = _NANO_BANANA_MAX_REFERENCE_IMAGES,
) -> _NanoBananaReferenceContext:
    from vidpipe.services.reference_selection import resolve_asset_image_bytes
    from vidpipe.services.tag_resolver import (
        canonicalize_character_tags,
        resolve_tags_with_assets,
    )

    context = _NanoBananaReferenceContext()

    asset_map = {
        normalized: asset
        for asset in all_assets
        if (normalized := _normalize_reference_tag(asset.manifest_tag))
    }

    placement_tags = _ordered_unique_tags([
        placement.get("asset_tag")
        for placement in (shot_manifest_json or {}).get("placements", [])
        if placement.get("asset_tag")
    ])
    shot_character_tags = _ordered_unique_tags(
        normalize_json_list(shot.characters_present)
    )
    optional_tags = _ordered_unique_tags(selected_reference_tags)

    character_aliases = await canonicalize_character_tags(
        _ordered_unique_tags(placement_tags + shot_character_tags + optional_tags),
        production_bible_id,
        session,
    )

    def _canonicalize_tag_list(tags: list[str]) -> list[str]:
        canonicalized: list[str] = []
        for tag in tags:
            canonical = character_aliases.get(tag, tag)
            if canonical != tag:
                context.canonical_tag_remaps[tag] = canonical
            canonicalized.append(canonical)
        return _ordered_unique_tags(canonicalized)

    placement_tags = _canonicalize_tag_list(placement_tags)
    shot_character_tags = _canonicalize_tag_list(shot_character_tags)
    optional_tags = _canonicalize_tag_list(optional_tags)
    shot_specific_tags = _ordered_unique_tags(shot_character_tags + placement_tags)

    resolved_binding_refs: dict[str, object] = {}

    async def _merge_resolved_tags(tags: list[str]) -> None:
        if not tags:
            return
        for requested_tag in tags:
            normalized_requested = _normalize_reference_tag(requested_tag)
            resolved = await resolve_tags_with_assets(
                f"@{requested_tag}",
                production_bible_id,
                session,
            )
            for asset_ref in resolved.asset_refs:
                normalized = _normalize_reference_tag(asset_ref.tag)
                if normalized and normalized not in resolved_binding_refs:
                    resolved_binding_refs[normalized] = asset_ref
                if normalized_requested and normalized_requested not in resolved_binding_refs:
                    resolved_binding_refs[normalized_requested] = asset_ref

    await _merge_resolved_tags(_ordered_unique_tags(shot_specific_tags + optional_tags))

    fallback_resolved = None
    if not shot_specific_tags and scene_prompt:
        fallback_resolved = await resolve_tags_with_assets(
            scene_prompt,
            production_bible_id,
            session,
        )
        for asset_ref in fallback_resolved.asset_refs:
            normalized = _normalize_reference_tag(asset_ref.tag)
            if normalized and normalized not in resolved_binding_refs:
                resolved_binding_refs[normalized] = asset_ref

    for tag in shot_specific_tags:
        asset = asset_map.get(tag)
        if asset and asset.asset_type == "CHARACTER":
            context.mandatory_character_tags.append(tag)
            resolved = resolved_binding_refs.get(tag)
            if resolved and getattr(resolved, "identity_type", None):
                context.identity_types_by_tag[tag] = resolved.identity_type
            continue
        resolved = resolved_binding_refs.get(tag)
        if resolved and resolved.asset_type == "CHARACTER":
            context.mandatory_character_tags.append(tag)
            context.identity_types_by_tag[tag] = (
                getattr(resolved, "identity_type", None) or "HUMAN"
            )

    if not context.mandatory_character_tags and fallback_resolved is not None:
        for asset_ref in fallback_resolved.asset_refs:
            normalized = _normalize_reference_tag(asset_ref.tag)
            if normalized and asset_ref.asset_type == "CHARACTER":
                context.mandatory_character_tags.append(normalized)
                resolved_binding_refs.setdefault(normalized, asset_ref)
                context.identity_types_by_tag[normalized] = (
                    getattr(asset_ref, "identity_type", None) or "HUMAN"
                )

    context.mandatory_character_tags = _ordered_unique_tags(context.mandatory_character_tags)
    context.optional_reference_tags = [
        tag for tag in optional_tags if tag not in set(context.mandatory_character_tags)
    ]
    context.placed_char_assets = [
        asset_map[tag]
        for tag in _ordered_unique_tags(shot_character_tags + placement_tags)
        if (
            tag in asset_map
            and asset_map[tag].asset_type == "CHARACTER"
            and _is_human_identity_type(context.identity_types_by_tag.get(tag))
        )
    ]

    seen_descriptions: set[str] = set()
    grouped_candidates: list[tuple[str, list[_ReferenceCandidate]]] = []
    for tag in context.mandatory_character_tags:
        candidates: list[_ReferenceCandidate] = []
        asset = asset_map.get(tag)
        if asset and asset.asset_type == "CHARACTER":
            ref_bytes = await resolve_asset_image_bytes(session, asset)
            resolved = resolved_binding_refs.get(tag)
            identity_type = (
                getattr(resolved, "identity_type", None)
                or context.identity_types_by_tag.get(tag)
                or "HUMAN"
            )
            context.identity_types_by_tag.setdefault(tag, identity_type)
            if ref_bytes and asset.reference_image_url:
                candidates.append(_ReferenceCandidate(
                    tag=tag,
                    image_url=asset.reference_image_url,
                    image_bytes=ref_bytes,
                    asset_type="CHARACTER",
                    source="manifest_asset",
                    asset=asset,
                    text_description=asset.reverse_prompt or asset.visual_description,
                    identity_type=identity_type,
                    reference_kind="wardrobe",
                ))

        if not candidates:
            resolved = resolved_binding_refs.get(tag)
            if resolved and resolved.asset_type == "CHARACTER":
                identity_type = getattr(resolved, "identity_type", None) or "HUMAN"
                context.identity_types_by_tag.setdefault(tag, identity_type)
                face_urls = list(getattr(resolved, "face_reference_image_urls", []) or [])
                wardrobe_urls = list(getattr(resolved, "wardrobe_reference_image_urls", []) or [])
                fallback_urls = list(resolved.reference_image_urls or [])

                ordered_refs: list[tuple[str, str]] = []
                if _is_human_identity_type(identity_type):
                    ordered_refs.extend(("face", url) for url in face_urls)
                ordered_refs.extend(("wardrobe", url) for url in wardrobe_urls)
                if not ordered_refs:
                    kind = "face" if _is_human_identity_type(identity_type) else "character"
                    ordered_refs.extend((kind, url) for url in fallback_urls)

                for reference_kind, url in ordered_refs:
                    try:
                        ref_bytes = await file_mgr.read_bytes(url)
                    except Exception as exc:
                        logger.warning("Failed to read CHARACTER reference %s for %s: %s", url, tag, exc)
                        continue
                    candidates.append(_ReferenceCandidate(
                        tag=tag,
                        image_url=url,
                        image_bytes=ref_bytes,
                        asset_type="CHARACTER",
                        source="binding",
                        text_description=resolved.text_description,
                        identity_type=identity_type,
                        reference_kind=reference_kind,
                    ))

        candidates = _dedupe_reference_candidates(candidates)
        if candidates:
            grouped_candidates.append((tag, candidates))
            description = candidates[0].text_description
            if description and description not in seen_descriptions:
                seen_descriptions.add(description)
                context.character_text_descriptions.append(description)

    selected_candidates, context.trimmed_reference_counts = _pack_mandatory_reference_candidates(
        grouped_candidates,
        max_reference_images=max_reference_images,
    )

    remaining_slots = max_reference_images - len(selected_candidates)
    if remaining_slots > 0:
        optional_candidates: list[_ReferenceCandidate] = []
        for tag in context.optional_reference_tags:
            asset = asset_map.get(tag)
            if asset and asset.reference_image_url:
                ref_bytes = await resolve_asset_image_bytes(session, asset)
                if ref_bytes:
                    optional_candidates.append(_ReferenceCandidate(
                        tag=tag,
                        image_url=asset.reference_image_url,
                        image_bytes=ref_bytes,
                        asset_type=asset.asset_type,
                        source="manifest_asset",
                        asset=asset,
                        text_description=asset.reverse_prompt or asset.visual_description,
                        identity_type=context.identity_types_by_tag.get(tag),
                        reference_kind="supplemental",
                    ))
                continue

            resolved = resolved_binding_refs.get(tag)
            if resolved and resolved.reference_image_urls:
                url = resolved.reference_image_urls[0]
                try:
                    ref_bytes = await file_mgr.read_bytes(url)
                except Exception as exc:
                    logger.warning("Failed to read optional reference %s for %s: %s", url, tag, exc)
                    continue
                optional_candidates.append(_ReferenceCandidate(
                    tag=tag,
                    image_url=url,
                    image_bytes=ref_bytes,
                    asset_type=resolved.asset_type,
                    source="binding",
                    text_description=resolved.text_description,
                    identity_type=getattr(resolved, "identity_type", None),
                    reference_kind="supplemental",
                ))

        selected_candidates.extend(optional_candidates[:remaining_slots])

    context.ref_image_bytes_list = [candidate.image_bytes for candidate in selected_candidates]
    context.final_reference_tags = [candidate.tag for candidate in selected_candidates]
    context.selected_ref_urls = [candidate.image_url for candidate in selected_candidates]
    context.character_ref_urls = [
        candidate.image_url
        for candidate in selected_candidates
        if candidate.asset_type == "CHARACTER"
    ]
    context.selected_candidates = selected_candidates
    return context


async def _apply_identity_policy_to_reference_candidates(
    *,
    selected_candidates: list[_ReferenceCandidate],
    file_mgr: FileManager,
) -> tuple[list[_ReferenceCandidate], dict[str, list], dict[str, object]]:
    """Apply identity-specific filtering to selected reference candidates.

    HUMAN refs go through face prequalification when no stored embeddings exist.
    Non-human refs bypass face screening and are passed through unchanged.
    """
    human_face_candidates = [
        candidate
        for candidate in selected_candidates
        if (
            candidate.asset_type == "CHARACTER"
            and _is_human_identity_type(candidate.identity_type)
            and candidate.reference_kind == "face"
        )
    ]
    passthrough_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate not in human_face_candidates
    ]

    policy_report = {
        "human_face_tags": _ordered_unique_tags([
            candidate.tag for candidate in human_face_candidates
        ]),
        "human_wardrobe_tags": _ordered_unique_tags([
            candidate.tag
            for candidate in passthrough_candidates
            if (
                candidate.asset_type == "CHARACTER"
                and _is_human_identity_type(candidate.identity_type)
                and candidate.reference_kind == "wardrobe"
            )
        ]),
        "passthrough_tags": _ordered_unique_tags([candidate.tag for candidate in passthrough_candidates]),
        "qualified_face_urls": 0,
        "dropped_face_urls": [],
    }

    if not human_face_candidates:
        return list(selected_candidates), {}, policy_report

    from vidpipe.services.ref_prequalification import prequalify_refs

    human_face_urls: list[str] = []
    seen_urls: set[str] = set()
    for candidate in human_face_candidates:
        if candidate.image_url in seen_urls:
            continue
        seen_urls.add(candidate.image_url)
        human_face_urls.append(candidate.image_url)

    qualified = await prequalify_refs(human_face_urls, file_mgr)
    qualified_by_url = {entry.image_url: entry for entry in qualified}
    policy_report["qualified_face_urls"] = len(qualified)
    policy_report["dropped_face_urls"] = [
        url for url in human_face_urls if url not in qualified_by_url
    ]

    filtered_candidates: list[_ReferenceCandidate] = []
    prequalified_embeddings_by_tag: dict[str, list] = {}
    for candidate in selected_candidates:
        if candidate not in human_face_candidates:
            filtered_candidates.append(candidate)
            continue

        qualified_ref = qualified_by_url.get(candidate.image_url)
        if not qualified_ref:
            continue

        filtered_candidates.append(
            _ReferenceCandidate(
                tag=candidate.tag,
                image_url=candidate.image_url,
                image_bytes=qualified_ref.face_crop_bytes,
                asset_type=candidate.asset_type,
                source=candidate.source,
                asset=candidate.asset,
                text_description=candidate.text_description,
                identity_type=candidate.identity_type,
                reference_kind=candidate.reference_kind,
            )
        )
        prequalified_embeddings_by_tag.setdefault(candidate.tag, []).append(
            qualified_ref.face_embedding
        )

    return filtered_candidates, prequalified_embeddings_by_tag, policy_report


def _candidate_sort_key(candidate: _ReferenceCandidate) -> tuple[int, str]:
    priority = {
        "face": 0,
        "wardrobe": 1,
        "character": 2,
        "supplemental": 3,
    }.get(candidate.reference_kind, 4)
    return (priority, candidate.image_url)


def _select_character_candidate_boxes(
    keyframe_bytes: bytes,
    targets: list[_CharacterVerificationTarget],
) -> _CharacterCropPlan:
    from PIL import Image

    image = Image.open(io.BytesIO(keyframe_bytes)).convert("RGB")
    image_width = max(1, image.width)
    try:
        from vidpipe.services.cv_detection import CVDetectionService

        detector = CVDetectionService()
        detections = detector.detect_objects_and_faces_from_bytes(keyframe_bytes, confidence_threshold=0.35)
    except Exception as exc:
        logger.warning("Character crop detection failed, falling back to full frame: %s", exc)
        detections = {"objects": [], "faces": []}

    objects = list(detections.get("objects", []))
    persons = [obj for obj in objects if obj.get("class") == "person"]
    non_persons = [obj for obj in objects if obj.get("class") != "person"]
    plan = _CharacterCropPlan(
        detected_face_count=len(detections.get("faces", [])),
        detected_person_count=len(persons),
        detected_object_count=len(objects),
    )
    used_ids: set[tuple[str, int]] = set()

    def _rank_candidates(position: str | None, pool: list[dict], pool_name: str) -> list[tuple[tuple[str, int], dict]]:
        if not pool:
            return []
        ranked: list[tuple[float, tuple[str, int], dict]] = []
        for index, obj in enumerate(pool):
            candidate_id = (pool_name, index)
            if candidate_id in used_ids:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if position == "left":
                score = cx
            elif position == "right":
                score = -cx
            elif position == "center":
                score = abs((cx / image_width) - 0.5)
            elif position == "background":
                score = area
            else:
                score = -area
            ranked.append((score, candidate_id, obj))
        ranked.sort(key=lambda item: item[0])
        return [(candidate_id, obj) for _, candidate_id, obj in ranked]

    for target in targets:
        if _is_human_identity_type(target.identity_type):
            pool = persons
            pool_name = "person"
        else:
            pool = non_persons or objects
            pool_name = "object"
        ranked = _rank_candidates((target.expected_position or "").lower(), pool, pool_name)
        if ranked:
            candidate_id, obj = ranked[0]
            used_ids.add(candidate_id)
            crop_bytes = _crop_image_bytes(keyframe_bytes, obj["bbox"])
            plan.selections[target.tag] = _CharacterCropSelection(
                tag=target.tag,
                image_bytes=crop_bytes,
                bbox=obj["bbox"],
                used_full_frame=False,
            )
            continue

        plan.selections[target.tag] = _CharacterCropSelection(
            tag=target.tag,
            image_bytes=keyframe_bytes,
            bbox=None,
            used_full_frame=True,
        )

    return plan


def _full_frame_character_crop_plan(
    keyframe_bytes: bytes,
    targets: list[_CharacterVerificationTarget],
) -> _CharacterCropPlan:
    return _CharacterCropPlan(
        selections={
            target.tag: _CharacterCropSelection(
                tag=target.tag,
                image_bytes=keyframe_bytes,
                bbox=None,
                used_full_frame=True,
            )
            for target in targets
        },
        detected_face_count=0,
        detected_person_count=0,
        detected_object_count=0,
    )


def _compose_verification_board(
    subject_bytes: bytes,
    references: list[tuple[str, bytes]],
) -> bytes:
    from PIL import Image, ImageDraw

    def _fit_image(image_bytes: bytes, size: tuple[int, int]) -> Image.Image:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail(size)
        canvas = Image.new("RGB", size, (24, 24, 24))
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas

    subject_size = (640, 640)
    ref_size = (320, 320)
    refs = references[:4]
    board_width = 32 + subject_size[0] + 24 + (ref_size[0] * 2) + 24
    board_height = 32 + max(subject_size[1], ref_size[1] * 2 + 40) + 32
    board = Image.new("RGB", (board_width, board_height), (10, 10, 14))
    draw = ImageDraw.Draw(board)

    subject = _fit_image(subject_bytes, subject_size)
    board.paste(subject, (16, 40))
    draw.text((16, 12), "GENERATED CROP", fill=(240, 240, 240))

    for idx, (label, ref_bytes) in enumerate(refs):
        col = idx % 2
        row = idx // 2
        x = 16 + subject_size[0] + 24 + col * ref_size[0]
        y = 40 + row * (ref_size[1] + 28)
        board.paste(_fit_image(ref_bytes, ref_size), (x, y))
        draw.text((x, y - 24), label[:28], fill=(220, 220, 220))

    out = io.BytesIO()
    board.save(out, format="PNG")
    return out.getvalue()


async def _verify_keyframe_characters_with_vision(
    *,
    session: AsyncSession,
    scene: Scene,
    shot: Shot,
    shot_manifest_json: dict | None,
    keyframe_bytes: bytes,
    selected_candidates: list[_ReferenceCandidate],
    identity_types_by_tag: dict[str, str],
) -> _VisionVerificationReport:
    targets = _build_character_verification_targets(
        shot_manifest_json=shot_manifest_json,
        selected_candidates=selected_candidates,
        identity_types_by_tag=identity_types_by_tag,
    )
    if not targets:
        return _VisionVerificationReport(passed=True, detail="no_character_targets")

    try:
        vision_model_id = scene.vision_model or scene.text_model or settings.models.storyboard_llm
        user_settings = None
        if vision_model_id.startswith("ollama/"):
            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == DEFAULT_USER_ID)
            )
            user_settings = result.scalar_one_or_none()
        vision_adapter = get_adapter(vision_model_id, user_settings=user_settings)
    except Exception as exc:
        logger.warning("Vision verifier unavailable (non-fatal): %s", exc)
        return _VisionVerificationReport(
            passed=True,
            detail=f"vision_adapter_unavailable: {exc}",
        )

    if _uses_comfyui_vision_primary(scene):
        crop_plan = _full_frame_character_crop_plan(keyframe_bytes, targets)
    else:
        crop_plan = _select_character_candidate_boxes(keyframe_bytes, targets)
    detail_lines: list[str] = []
    overall_pass = True
    results: list[_CharacterVisionVerificationResult] = []

    for target in targets:
        crop = crop_plan.selections.get(
            target.tag,
            _CharacterCropSelection(target.tag, keyframe_bytes, None, True),
        )
        references: list[tuple[str, bytes]] = []
        for candidate in sorted(target.face_candidates, key=_candidate_sort_key)[:2]:
            references.append((f"FACE REF @{target.tag}", candidate.image_bytes))
        for candidate in sorted(target.wardrobe_candidates, key=_candidate_sort_key)[:2]:
            references.append((f"WARDROBE REF @{target.tag}", candidate.image_bytes))
        if not references:
            for candidate in sorted(target.fallback_candidates, key=_candidate_sort_key)[:2]:
                references.append((f"REFERENCE @{target.tag}", candidate.image_bytes))
        if not references:
            detail_lines.append(f"@{target.tag}: skipped (no refs)")
            continue

        board_bytes = _compose_verification_board(crop.image_bytes, references)
        expected_position = target.expected_position or "unspecified"
        if target.expected_wardrobe and not target.wardrobe_candidates:
            # Scene-specific costume: the identity photos establish WHO the
            # person is, not what they wear in this shot. Without this, the
            # verifier fails correct costumes for not matching the street
            # clothes in the reference photos.
            wardrobe_clause = (
                f"Expected wardrobe for this shot (authoritative): "
                f"'{target.expected_wardrobe}'. Judge wardrobe against that "
                "description ONLY — the reference photos establish identity, "
                "and the clothing worn in them is NOT the expected wardrobe. "
            )
        else:
            wardrobe_clause = "Judge wardrobe/look fidelity against the reference images. "
        prompt = (
            f"Verify a generated keyframe crop for @{target.tag}. "
            f"Expected position: {expected_position}. "
            f"Identity type: {target.identity_type}. "
            f"Judge this character's identity/species/markings against the reference images. "
            f"{wardrobe_clause}"
            f"Ignore camera angle, pose, lighting, and composition changes unless they hide the character completely. "
            f"The left panel is the generated crop. The right panels are reference images."
        )

        try:
            result = await vision_adapter.analyze_image(
                board_bytes,
                prompt,
                CharacterKeyframeVerificationOutput,
                mime_type="image/png",
                temperature=0.1,
                max_retries=2,
            )
        except Exception as exc:
            logger.warning("Vision verification skipped for %s (non-fatal): %s", target.tag, exc)
            detail_lines.append(f"@{target.tag}: verification_error={exc}")
            continue

        partial_visibility_pass = _passes_partial_visibility_human_check(
            target=target,
            crop=crop,
            result=result,
        )
        char_pass = bool(
            result.passed
            and result.character_visible
            and result.identity_match
            and result.wardrobe_match
        )
        if not char_pass and partial_visibility_pass:
            char_pass = True
        overall_pass = overall_pass and char_pass
        issues = ", ".join(result.issues or []) or "none"
        results.append(_CharacterVisionVerificationResult(
            tag=target.tag,
            passed=char_pass,
            character_visible=result.character_visible,
            identity_match=result.identity_match,
            wardrobe_match=result.wardrobe_match,
            identity_score=result.identity_score,
            wardrobe_score=result.wardrobe_score,
            issues=list(result.issues or []),
            used_full_frame=crop.used_full_frame,
        ))
        detail_lines.append(
            f"@{target.tag}: passed={char_pass} visible={result.character_visible} "
            f"identity={result.identity_score:.1f}/10 wardrobe={result.wardrobe_score:.1f}/10 "
            f"full_frame_fallback={crop.used_full_frame} "
            f"partial_visibility_pass={partial_visibility_pass} issues={issues}"
        )

    return _VisionVerificationReport(
        passed=overall_pass,
        detail=" | ".join(detail_lines) if detail_lines else "no_verification_details",
        results=results,
    )


def _collect_reference_face_embeddings_by_tag(
    *,
    placed_char_assets: list,
    prequalified_ref_embeddings_by_tag: dict[str, list] | None,
) -> dict[str, list]:
    embeddings_by_tag: dict[str, list] = {
        tag: list(embeddings)
        for tag, embeddings in (prequalified_ref_embeddings_by_tag or {}).items()
    }
    for asset in placed_char_assets:
        if getattr(asset, "face_embedding", None) is None:
            continue
        tag = _normalize_reference_tag(getattr(asset, "manifest_tag", None))
        if not tag:
            continue
        embeddings_by_tag.setdefault(tag, []).append(
            np.frombuffer(asset.face_embedding, dtype=np.float32).copy()
        )
    return embeddings_by_tag


async def _verify_target_face(
    crop_bytes: bytes,
    ref_embeddings: list,
    threshold: float | None = None,
) -> tuple[bool, float, str]:
    if threshold is None:
        threshold = settings.cv_analysis.keyframe_face_match_threshold

    if not ref_embeddings:
        return True, 0.0, "no_target_embeddings"

    try:
        from vidpipe.services.face_matching import FaceMatchingService

        face_matcher = FaceMatchingService()
        gen_embedding = await asyncio.to_thread(
            face_matcher.generate_embedding_from_bytes, crop_bytes
        )
        best_similarity = max(
            FaceMatchingService.cosine_similarity(gen_embedding, ref_emb)
            for ref_emb in ref_embeddings
        )
        passed = best_similarity >= threshold
        detail = (
            f"best_sim={best_similarity:.3f} threshold={threshold:.3f} "
            f"refs_checked={len(ref_embeddings)}"
        )
        return passed, best_similarity, detail
    except ValueError:
        return True, 0.0, "no_target_face_detected"
    except Exception as e:
        logger.warning("Face verification error (non-fatal): %s", e)
        return True, 0.0, f"verification_error: {e}"


def _build_retry_correction_prompt(report: _KeyframeVerificationReport) -> str:
    lines: list[str] = []

    if report.vision_report:
        for result in report.vision_report.results:
            if not result.character_visible:
                lines.append(
                    f"Keep @{result.tag} clearly visible in frame; do not crop or obscure this subject."
                )
            if not result.identity_match:
                lines.append(
                    f"IDENTITY LOCK for @{result.tag}: preserve the exact same recurring subject identity."
                )
            if not result.wardrobe_match:
                lines.append(
                    f"WARDROBE LOCK for @{result.tag}: match the wardrobe reference exactly. "
                    "Do not invent or borrow jackets, coats, backpacks, accessories, colors, textures, or garments "
                    "from any appearance reference. The wardrobe reference is authoritative."
                )
            for issue in result.issues[:2]:
                lines.append(f"For @{result.tag}, correct this mismatch: {issue}")

    for face_result in report.face_results:
        if face_result.advisory or face_result.passed:
            continue
        lines.append(
            f"FACE LOCK for @{face_result.tag}: preserve the exact same human face and facial structure as the face reference."
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    if not deduped:
        return ""
    return "RETRY CORRECTION REQUIREMENTS:\n- " + "\n- ".join(deduped)


async def _verify_generated_keyframe(
    *,
    session: AsyncSession,
    scene: Scene,
    shot: Shot,
    position: str,
    keyframe_bytes: bytes,
    shot_manifest_json: dict | None,
    selected_candidates: list[_ReferenceCandidate],
    identity_types_by_tag: dict[str, str],
    placed_char_assets: list,
    prequalified_ref_embeddings_by_tag: dict[str, list] | None,
) -> _KeyframeVerificationReport:
    checks: list[str] = []
    targets = _build_character_verification_targets(
        shot_manifest_json=shot_manifest_json,
        selected_candidates=selected_candidates,
        identity_types_by_tag=identity_types_by_tag,
    )
    comfyui_multiref_vision_primary = _uses_comfyui_vision_primary(scene)
    if comfyui_multiref_vision_primary:
        crop_plan = _full_frame_character_crop_plan(keyframe_bytes, targets)
    else:
        crop_plan = _select_character_candidate_boxes(keyframe_bytes, targets)
    vision_report = await _verify_keyframe_characters_with_vision(
        session=session,
        scene=scene,
        shot=shot,
        shot_manifest_json=shot_manifest_json,
        keyframe_bytes=keyframe_bytes,
        selected_candidates=selected_candidates,
        identity_types_by_tag=identity_types_by_tag,
    )

    human_targets = [
        target for target in targets
        if _is_human_identity_type(target.identity_type)
    ]
    crowded = bool(
        len(human_targets) > 1
        or crop_plan.detected_face_count > 1
        or crop_plan.detected_person_count > 1
        or any(
            crop_plan.selections.get(
                target.tag,
                _CharacterCropSelection(target.tag, keyframe_bytes, None, True),
            ).used_full_frame
            for target in human_targets
        )
        or comfyui_multiref_vision_primary
    )
    verification_mode = (
        "vision_primary_face_advisory" if crowded else "strict_face_and_vision"
    )

    reference_face_embeddings_by_tag = _collect_reference_face_embeddings_by_tag(
        placed_char_assets=placed_char_assets,
        prequalified_ref_embeddings_by_tag=prequalified_ref_embeddings_by_tag,
    )
    face_results: list[_HumanFaceVerificationResult] = []
    face_gate_pass = True

    for target in human_targets:
        crop = crop_plan.selections.get(
            target.tag,
            _CharacterCropSelection(target.tag, keyframe_bytes, None, True),
        )
        if comfyui_multiref_vision_primary:
            face_pass = False
            sim = 0.0
            face_detail = "skipped_for_comfyui_vision_primary"
        else:
            face_pass, sim, face_detail = await _verify_target_face(
                crop.image_bytes,
                reference_face_embeddings_by_tag.get(target.tag, []),
            )
        advisory = crowded or crop.used_full_frame
        if not advisory:
            face_gate_pass = face_gate_pass and face_pass
        face_results.append(_HumanFaceVerificationResult(
            tag=target.tag,
            passed=face_pass,
            advisory=advisory,
            similarity=sim,
            detail=face_detail,
        ))

    if face_results:
        checks.append(
            f"face_check mode={verification_mode} detected_faces={crop_plan.detected_face_count} "
            f"detected_persons={crop_plan.detected_person_count} "
            + " | ".join(
                (
                    f"@{result.tag}: passed={result.passed} advisory={result.advisory} "
                    f"sim={result.similarity:.3f} {result.detail}"
                )
                for result in face_results
            )
        )

    checks.append(f"vision_check={vision_report.passed} {vision_report.detail}")
    vision_results_by_tag = {
        result.tag: result
        for result in (vision_report.results if vision_report else [])
    }
    corroborated_face_tags = [
        face_result.tag
        for face_result in face_results
        if _vision_can_corroborate_near_threshold_face(
            face_result=face_result,
            vision_result=vision_results_by_tag.get(face_result.tag),
        )
    ]
    if not face_gate_pass and corroborated_face_tags:
        blocking_face_tags = [
            face_result.tag
            for face_result in face_results
            if (
                not face_result.passed
                and not face_result.advisory
                and face_result.tag not in corroborated_face_tags
            )
        ]
        if not blocking_face_tags:
            face_gate_pass = True
            checks.append(
                "vision_corroborated_near_threshold_face="
                + ",".join(f"@{tag}" for tag in corroborated_face_tags)
            )

    overall_pass = bool(vision_report.passed and face_gate_pass)

    detail = " || ".join(checks) if checks else "no_checks"
    logger.info("Shot %s %s verification: %s", shot.shot_index, position, detail)
    return _KeyframeVerificationReport(
        passed=overall_pass,
        detail=detail,
        verification_mode=verification_mode,
        face_results=face_results,
        vision_report=vision_report,
    )


_ASPECT_RATIOS: dict[str, float] = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "21:9": 21 / 9,
}


def _coerce_image_aspect(image_bytes: bytes, aspect_ratio: str) -> bytes:
    """Crop a generated image to the requested aspect ratio if it deviates.

    Safety net for image models that occasionally ignore the aspect config
    (observed: portrait keyframes for a 16:9 scene, later squashed into the
    video resolution — the "fisheye" look). Crops are biased toward the top
    of the frame when trimming height, since faces sit in the upper half.
    """
    target = _ASPECT_RATIOS.get(aspect_ratio)
    if target is None:
        return image_bytes

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    actual = img.width / img.height
    if abs(actual - target) / target < 0.05:
        return image_bytes

    logger.warning(
        "Generated image aspect %.3f deviates from requested %s (%.3f) — cropping",
        actual, aspect_ratio, target,
    )
    if actual > target:
        # Too wide — trim width, centered
        new_w = int(img.height * target)
        x0 = (img.width - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, img.height))
    else:
        # Too tall — trim height, biased toward the top (faces live there)
        new_h = int(img.width / target)
        y0 = min(int((img.height - new_h) * 0.25), img.height - new_h)
        img = img.crop((0, y0, img.width, y0 + new_h))

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _generate_image_from_text(
    client, prompt: str, aspect_ratio: str, image_model: str,
    seed: int | None = None,
    reference_images: list[bytes] | None = None,
    identity_instruction: str | None = None,
) -> bytes:
    """Generate image from text prompt using Gemini generate_content().

    When reference_images are provided, they are prepended as image parts
    so Gemini can use them for visual identity grounding (face, clothing, etc.).

    Args:
        client: Vertex AI client instance
        prompt: Text description for image generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")
        image_model: Model ID to use for generation
        seed: Optional seed for reproducibility
        reference_images: Optional list of PNG bytes for identity grounding
        identity_instruction: Optional feature-anchored identity prompt
            (from _build_identity_instruction). Falls back to generic prefix.

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    # Build contents: [identity_instruction, ref_image_1, ..., text_prompt]
    # When reference images are present, prepend an identity-matching instruction
    # so Gemini knows these images define the characters' visual appearance.
    contents: list = []
    if reference_images:
        ref_prefix = identity_instruction or (
            "The following reference photo(s) show the EXACT person(s) who must appear "
            "in the generated image. Match their face, skin tone, head shape, and "
            "distinguishing features as closely as possible. "
            "These are real reference photos — the generated character MUST look like "
            "the same person, not just a similar description.\n\n"
        )
        contents.append(ref_prefix)
        for ref_bytes in reference_images:
            contents.append(
                types.Part.from_bytes(data=ref_bytes, mime_type=_detect_image_mime(ref_bytes))
            )
    contents.append(prompt)

    response = await client.aio.models.generate_content(
        model=image_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            # Without an explicit aspect ratio, Gemini image models follow the
            # aspect of the reference images (a portrait selfie produced
            # portrait keyframes that were then squashed into 16:9 video).
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return _coerce_image_aspect(part.inline_data.data, aspect_ratio)

    raise ValueError("No image generated in response")


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    retry=retry_if_exception(_is_retriable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _generate_image_conditioned(
    client,
    reference_image_bytes: bytes,
    prompt: str,
    aspect_ratio: str,
    conditioned_model: str,
    reference_images: list[bytes] | None = None,
    identity_instruction: str | None = None,
) -> bytes:
    """Generate image using conditioning frame + optional asset reference images.

    Contents order: [conditioning_frame, ref_image_1, ..., text_prompt]
    The conditioning frame comes first (strongest weight for visual continuity),
    followed by asset reference images for identity grounding.

    Args:
        client: Vertex AI client instance
        reference_image_bytes: PNG image data from previous frame (conditioning)
        prompt: Text description for conditioned generation
        aspect_ratio: Image aspect ratio (e.g., "16:9", "9:16", "1:1")
        conditioned_model: Model ID to use for conditioned generation
        reference_images: Optional list of PNG bytes for identity grounding
        identity_instruction: Optional feature-anchored identity prompt
            (from _build_identity_instruction). Falls back to generic prefix.

    Returns:
        PNG image data as bytes

    Raises:
        ValueError: If no image found in response
    """
    # Build contents: [conditioning_frame, identity_instruction, ref_images..., text_prompt]
    contents: list = [
        types.Part.from_bytes(data=reference_image_bytes, mime_type=_detect_image_mime(reference_image_bytes)),
    ]
    if reference_images:
        ref_prefix = identity_instruction or (
            "The following reference photo(s) show the EXACT person(s) who must appear "
            "in the generated image. Match their face, skin tone, head shape, and "
            "distinguishing features as closely as possible."
        )
        contents.append(types.Part.from_text(text=ref_prefix))
        for ref_bytes in reference_images:
            contents.append(
                types.Part.from_bytes(data=ref_bytes, mime_type=_detect_image_mime(ref_bytes))
            )
    contents.append(types.Part.from_text(text=prompt))

    response = await client.aio.models.generate_content(
        model=conditioned_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            # See _generate_image_from_text — explicit aspect ratio prevents
            # the model from following the reference images' aspect.
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    # Extract image bytes from response
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return _coerce_image_aspect(part.inline_data.data, aspect_ratio)

    raise ValueError("No image generated in response")


async def _verify_keyframe_faces(
    keyframe_bytes: bytes,
    placed_char_assets: list,
    threshold: float | None = None,
    ref_embeddings: list | None = None,
) -> tuple[bool, float, str]:
    """Verify generated keyframe contains faces matching placed CHARACTER assets.

    Uses YOLO face detection + ArcFace embedding comparison.

    Supports two embedding sources:
    - Legacy: Asset.face_embedding bytes from manifest pathway
    - CastBinding: ref_embeddings (numpy arrays) from prequalification service

    Soft degradation — returns (True, ...) when:
    - No embeddings available from either source → "no_embeddings_available"
    - No faces detected in keyframe → "no_faces_detected"
    - CV services fail → "verification_error"

    Args:
        keyframe_bytes: Generated keyframe image bytes
        placed_char_assets: Asset objects for placed CHARACTERs (must have face_embedding)
        threshold: Cosine similarity threshold (default from config)
        ref_embeddings: Pre-computed numpy embeddings from prequalification
            (bridges CastBinding flow where no Asset.face_embedding exists)

    Returns:
        (passed, best_similarity, detail_string)
    """
    if threshold is None:
        threshold = settings.cv_analysis.keyframe_face_match_threshold

    # Collect reference embeddings from both sources
    all_ref_embeddings: list = []

    # Source 1: Legacy Asset.face_embedding bytes
    for a in placed_char_assets:
        if a.face_embedding is not None:
            emb = np.frombuffer(a.face_embedding, dtype=np.float32).copy()
            all_ref_embeddings.append(emb)

    # Source 2: Prequalified ActorRef embeddings (numpy arrays)
    if ref_embeddings:
        all_ref_embeddings.extend(ref_embeddings)

    if not all_ref_embeddings:
        return True, 0.0, "no_embeddings_available"

    try:
        from vidpipe.services.cv_detection import CVDetectionService
        from vidpipe.services.face_matching import FaceMatchingService

        cv_detector = CVDetectionService()
        face_matcher = FaceMatchingService()

        # Detect faces in generated keyframe
        faces = await asyncio.to_thread(
            cv_detector.detect_faces_from_bytes, keyframe_bytes
        )
        if not faces:
            return True, 0.0, "no_faces_detected"

        # Crop the best face from the keyframe and get its embedding
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(keyframe_bytes)).convert("RGB")

        best_similarity = 0.0

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            # Add 10% padding
            fw, fh = x2 - x1, y2 - y1
            px, py = fw * 0.1, fh * 0.1
            cx1 = max(0, x1 - px)
            cy1 = max(0, y1 - py)
            cx2 = min(img.width, x2 + px)
            cy2 = min(img.height, y2 + py)

            face_crop = img.crop((cx1, cy1, cx2, cy2))

            # Convert crop to bytes for embedding
            buf = io.BytesIO()
            face_crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            try:
                gen_embedding = await asyncio.to_thread(
                    face_matcher.generate_embedding_from_bytes, crop_bytes
                )
            except ValueError:
                continue

            # Compare against all reference embeddings
            for ref_emb in all_ref_embeddings:
                sim = FaceMatchingService.cosine_similarity(gen_embedding, ref_emb)
                best_similarity = max(best_similarity, sim)

        passed = best_similarity >= threshold
        detail = (
            f"best_sim={best_similarity:.3f} threshold={threshold:.3f} "
            f"faces_detected={len(faces)} refs_checked={len(all_ref_embeddings)}"
        )
        return passed, best_similarity, detail

    except Exception as e:
        logger.warning(f"Face verification error (non-fatal): {e}")
        return True, 0.0, f"verification_error: {e}"


async def _run_comfyui_image_job(comfy_client, workflow: dict, label: str) -> bytes:
    """Queue a ComfyUI image workflow, poll until success, download the output.

    Args:
        comfy_client: ComfyUIClient instance
        workflow: API-format workflow dict ready to submit
        label: Human-readable job label for logging

    Returns:
        PNG image data as bytes

    Raises:
        RuntimeError: On ComfyUI job failure or timeout
        ValueError: If no image output found in history
    """
    import time as _time

    from vidpipe.services.comfyui_adapter import QUEUED_STATUSES
    from vidpipe.services.comfyui_client import find_comfyui_image_output

    prompt_id = await comfy_client.queue_prompt(workflow)
    logger.info(f"ComfyUI {label} queued: prompt_id={prompt_id}")

    # Poll until completion (check for "success", not "completed").
    # Time spent in Comfy Cloud queue states ("queued_limited" under account
    # concurrency limits) does not count against the execution timeout.
    max_polls = 120
    poll_interval = 3
    polls_used = 0
    queue_deadline = _time.monotonic() + settings.pipeline.comfy_queue_timeout
    while True:
        await asyncio.sleep(poll_interval)
        status, error_msg = await comfy_client.poll_status(prompt_id)
        if status == "success":
            break
        if status in ("error", "failed", "cancelled"):
            raise RuntimeError(
                f"ComfyUI {label} job {prompt_id} failed: status={status}, error={error_msg}"
            )
        if status in QUEUED_STATUSES:
            if _time.monotonic() > queue_deadline:
                raise RuntimeError(
                    f"ComfyUI {label} job {prompt_id} stayed queued for more than "
                    f"{settings.pipeline.comfy_queue_timeout}s"
                )
            continue
        # Executing — consume execution poll budget
        polls_used += 1
        if polls_used >= max_polls:
            raise RuntimeError(
                f"ComfyUI {label} job {prompt_id} timed out after "
                f"{max_polls * poll_interval}s of execution"
            )

    # Fetch history and extract output image
    history = await comfy_client.get_history(prompt_id)
    filename, subfolder = find_comfyui_image_output(history, prompt_id)
    image_bytes = await comfy_client.download_output(filename, subfolder)
    logger.info(
        f"ComfyUI {label} complete: {filename} ({len(image_bytes)} bytes)"
    )
    return image_bytes


async def _generate_image_comfyui(
    comfy_client,
    prompt: str,
    seed: int,
    width: int = 1328,
    height: int = 1328,
) -> bytes:
    """Generate an image via ComfyUI Cloud using the Qwen txt2img workflow."""
    from vidpipe.services.comfyui_client import build_qwen_txt2img_workflow

    workflow = build_qwen_txt2img_workflow(
        prompt=prompt, width=width, height=height, seed=seed,
    )
    return await _run_comfyui_image_job(comfy_client, workflow, "Qwen txt2img")


async def _generate_image_comfyui_edit(
    comfy_client,
    prompt: str,
    input_image_bytes: bytes,
    seed: int,
) -> bytes:
    """Generate an edited image via ComfyUI Cloud using the Qwen Image Edit workflow.

    Uploads the input image, builds the edit workflow, and runs the job.
    """
    from vidpipe.services.comfyui_client import build_qwen_image_edit_workflow

    input_filename = await comfy_client.upload_image(
        input_image_bytes, "image_qwen_image_edit_input_image.png"
    )
    workflow = build_qwen_image_edit_workflow(
        prompt=prompt, input_image_filename=input_filename, seed=seed,
    )
    return await _run_comfyui_image_job(comfy_client, workflow, "Qwen Image Edit")


async def _generate_image_comfyui_flux(
    comfy_client,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    lora_filename: Optional[str] = None,
    lora_strength: float = 0.8,
    reference_image_filenames: Optional[list[str]] = None,
    reference_strengths: Optional[list[float]] = None,
) -> bytes:
    """Generate an image via ComfyUI Cloud using the Flux.1 Dev txt2img workflow.

    Builds the workflow with optional LoRA and reference images (max 3,
    pre-uploaded filenames) and runs the job.
    """
    from vidpipe.services.comfyui_client import build_flux_txt2img_workflow

    workflow = build_flux_txt2img_workflow(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        lora_filename=lora_filename,
        lora_strength=lora_strength,
        reference_image_filenames=reference_image_filenames,
        reference_strengths=reference_strengths,
    )
    return await _run_comfyui_image_job(comfy_client, workflow, "Flux txt2img")


async def _generate_image_comfyui_qwen_edit_2509(
    comfy_client,
    prompt: str,
    seed: int,
    image_bytes_list: list[bytes],
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
) -> bytes:
    """Generate an image via Qwen Image Edit 2509 with 1-3 input images.

    Uploads each input image, builds the multi-ref edit workflow, and runs
    the job. The first image is image1: in edit mode (no output dims) the
    output size follows it; in generation mode (output dims given) the output
    uses the requested dimensions and the images act purely as references.

    Args:
        comfy_client: ComfyUIClient instance
        prompt: Edit/composition instruction
        seed: Random seed for reproducibility
        image_bytes_list: 1-3 input images as PNG/JPEG bytes
        output_width: Explicit output width (generation mode)
        output_height: Explicit output height (generation mode)
    """
    from vidpipe.services.comfyui_client import build_qwen_edit_2509_workflow

    filenames = []
    for i, img_bytes in enumerate(image_bytes_list[:3]):
        filenames.append(
            await comfy_client.upload_image(img_bytes, f"qwen2509_image{i + 1}.png")
        )
    workflow = build_qwen_edit_2509_workflow(
        prompt=prompt,
        image_filenames=filenames,
        seed=seed,
        output_width=output_width,
        output_height=output_height,
    )
    return await _run_comfyui_image_job(comfy_client, workflow, "Qwen Edit 2509")


async def _generate_image_comfyui_flux2_klein(
    comfy_client,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    reference_image_bytes_list: Optional[list[bytes]] = None,
) -> bytes:
    """Generate an image via FLUX.2 Klein 4B with 0-4 reference images.

    Uploads each reference image, builds the klein workflow (plain txt2img
    when no references), and runs the job.

    Args:
        comfy_client: ComfyUIClient instance
        prompt: Positive prompt text
        seed: Random seed for reproducibility
        width: Output width in pixels
        height: Output height in pixels
        reference_image_bytes_list: Up to 4 reference images as PNG/JPEG bytes
    """
    from vidpipe.services.comfyui_client import build_flux2_klein_workflow

    filenames = []
    for i, img_bytes in enumerate((reference_image_bytes_list or [])[:4]):
        filenames.append(
            await comfy_client.upload_image(img_bytes, f"flux2_ref{i + 1}.png")
        )
    workflow = build_flux2_klein_workflow(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        reference_image_filenames=filenames or None,
    )
    return await _run_comfyui_image_job(comfy_client, workflow, "FLUX.2 Klein")


async def generate_keyframes(
    session: AsyncSession,
    scene: Scene,
    text_adapter: Optional[LLMAdapter] = None,
) -> None:
    """Generate keyframes sequentially with visual continuity across shots.

    Implements sequential keyframe generation where:
    - Shot 0 start frame is generated from text prompt alone
    - Shot N start frame inherits shot N-1 end frame
    - All end frames use image-conditioned generation for continuity

    Args:
        session: Database session for persisting keyframes
        scene: Scene containing shots to generate keyframes for
        text_adapter: Optional LLMAdapter for prompt rewriting. If None,
            PromptRewriterService falls back to get_adapter("gemini-3.5-flash").

    Process:
        1. Query shots ordered by shot_index
        2. For each shot sequentially:
           a. Generate or inherit start frame
           b. Save start keyframe to filesystem and database
           c. Generate end frame using image-conditioned generation
           d. Save end keyframe to filesystem and database
           e. Update shot status and commit
           f. Rate limit delay before next shot
        3. Update scene status to "generating_video"

    Note:
        - Commits after each shot for crash recovery
        - Uses rate limiting to prevent 429 errors
        - Sequential processing ensures visual continuity (KEYF-04)
    """
    # Resolve image model from scene (with fallback to settings)
    image_model = canonical_model_id(scene.image_model or settings.models.image_gen) or settings.models.image_gen
    base_seed = int(scene.seed or 0)

    # Guard: Imagen models no longer supported — fall back to config default
    if image_model.startswith("imagen-"):
        logger.warning(
            f"Scene uses unsupported Imagen model '{image_model}', "
            f"falling back to '{settings.models.image_gen}'"
        )
        image_model = settings.models.image_gen

    # Build character bible prefix from storyboard data
    character_prefix = ""
    if scene.storyboard_raw and "characters" in scene.storyboard_raw:
        char_lines = []
        for ch in scene.storyboard_raw["characters"]:
            char_lines.append(
                f"{ch.get('name', 'Character')}: {ch.get('physical_description', '')}. "
                f"Wearing {ch.get('clothing_description', '')}."
            )
        if char_lines:
            character_prefix = "Characters: " + " ".join(char_lines) + " "

    # Build style prefix from style guide
    style_guide = scene.style_guide or {}
    style_prefix = ""
    if style_guide:
        parts = []
        if style_guide.get("visual_style"):
            parts.append(f"Style: {style_guide['visual_style']}")
        if style_guide.get("color_palette"):
            parts.append(f"Palette: {style_guide['color_palette']}")
        if parts:
            style_prefix = ". ".join(parts) + ". "

    # Route to ComfyUI or Vertex AI based on model
    is_comfyui = image_model in COMFYUI_IMAGE_MODELS
    comfy_client = None
    image_client = None
    if is_comfyui:
        from vidpipe.services.comfyui_client import get_comfyui_client
        comfy_client = await get_comfyui_client()
    else:
        image_client = get_vertex_client(location=location_for_model(image_model))
    file_mgr = FileManager()

    # Query shots ordered by shot_index for sequential processing
    result = await session.execute(
        select(Shot)
        .where(Shot.scene_id == scene.id)
        .order_by(Shot.shot_index)
    )
    shots = result.scalars().all()

    # Track previous shot's end frame for inheritance
    previous_end_frame_bytes = None

    from vidpipe.services.event_bus import event_bus
    event_bus.emit(
        scene.id,
        "phase_started",
        phase="keyframes",
        total_shots=len(shots),
        message=f"Starting keyframe generation for {len(shots)} shot(s)",
    )

    # Process each shot sequentially (no parallelization)
    for shot in shots:
        # Per-shot stop flag check (VGED-11)
        await session.refresh(scene)
        if scene.status == "stopped":
            from vidpipe.orchestrator.pipeline import PipelineStopped
            logger.info(f"Pipeline stopped by user at shot {shot.shot_index}")
            raise PipelineStopped("Stopped by user")

        # Gap-filling: check for existing keyframes per position (VGED-06)
        existing_kfs_result = await session.execute(
            select(Keyframe).where(Keyframe.shot_id == shot.id)
        )
        existing_kfs = existing_kfs_result.scalars().all()
        existing_start_kf = next((k for k in existing_kfs if k.position == "start"), None)
        existing_end_kf = next((k for k in existing_kfs if k.position == "end"), None)

        # If both keyframes exist, skip entire shot (fork or user upload)
        if existing_start_kf and existing_end_kf:
            previous_end_frame_bytes = await file_mgr.read_bytes(existing_end_kf.file_path)
            # Don't downgrade shots that already have completed clips
            if shot.status != "video_done":
                shot.status = "keyframes_done"
            shot.generation_status = None
            await session.commit()
            logger.info(
                f"Shot {shot.shot_index}: both keyframes exist, skipping"
            )
            continue

        # KEYF-03 continuity: if shot has an uploaded start keyframe,
        # use it as previous_end for daisy-chain conditioning
        if existing_start_kf:
            # The uploaded start keyframe serves as "previous end" for conditioning
            _existing_start_bytes = await file_mgr.read_bytes(existing_start_kf.file_path)

        try:
            # Set generation_status for the shot (VGED-05)
            if not existing_start_kf:
                shot.generation_status = "generating_start_kf"
                await session.commit()
                event_bus.emit(
                    scene.id,
                    "shot_status",
                    shot_index=shot.shot_index,
                    status="generating_start_kf",
                    phase="keyframes",
                    message=f"Shot {shot.shot_index + 1}: generating start keyframe",
                )

            # Phase 10: Adaptive Prompt Rewriting for manifest scenes
            # Also resolves asset reference images for multimodal keyframe generation
            rewritten_start_prompt = None
            ref_image_bytes_list: list[bytes] = []
            character_text_descriptions: list[str] = []
            placed_char_assets: list = []  # Legacy CHARACTER assets used for face verification
            ref_context = _NanoBananaReferenceContext()
            shot_manifest_row = None
            all_assets: list[Asset] = []
            if scene.production_bible_id:
                try:
                    from vidpipe.services.prompt_rewriter import PromptRewriterService
                    from vidpipe.db.models import ShotManifest as ShotManifestModel

                    # Load shot manifest
                    sm_result = await session.execute(
                        select(ShotManifestModel).where(
                            ShotManifestModel.scene_id == scene.id,
                            ShotManifestModel.shot_index == shot.shot_index
                        )
                    )
                    shot_manifest_row = sm_result.scalar_one_or_none()

                    if shot_manifest_row and shot_manifest_row.manifest_json:
                        # Load assets
                        from vidpipe.services import manifest_service
                        all_assets = await manifest_service.load_manifest_assets(session, scene.production_bible_id)

                        # Load previous shot CV analysis for continuity
                        previous_cv = None
                        if shot.shot_index > 0:
                            prev_sm_result = await session.execute(
                                select(ShotManifestModel).where(
                                    ShotManifestModel.scene_id == scene.id,
                                    ShotManifestModel.shot_index == shot.shot_index - 1
                                )
                            )
                            prev_sm = prev_sm_result.scalar_one_or_none()
                            if prev_sm:
                                previous_cv = prev_sm.cv_analysis_json

                        rewriter = PromptRewriterService(text_adapter=text_adapter)
                        result = await rewriter.rewrite_keyframe_prompt(
                            shot=shot,
                            shot_manifest_json=shot_manifest_row.manifest_json,
                            placed_assets=all_assets,  # rewriter filters to placed internally
                            previous_cv_analysis=previous_cv,
                            all_assets=all_assets,
                        )

                        rewritten_start_prompt = result.rewritten_prompt
                        selected_reference_tags = list(result.selected_reference_tags or [])
                        if scene.production_bible_id and selected_reference_tags:
                            from vidpipe.services.tag_resolver import canonicalize_character_tags

                            selected_tag_aliases = await canonicalize_character_tags(
                                selected_reference_tags,
                                scene.production_bible_id,
                                session,
                            )
                            normalized_selected_tags: list[str] = []
                            for tag in selected_reference_tags:
                                normalized = _normalize_reference_tag(tag) or tag
                                normalized_selected_tags.append(
                                    selected_tag_aliases.get(normalized, normalized)
                                )
                            selected_reference_tags = _ordered_unique_tags(normalized_selected_tags)

                        # Persist rewritten prompt and selected reference tags
                        shot_manifest_row.rewritten_keyframe_prompt = result.rewritten_prompt
                        shot_manifest_row.selected_reference_tags = selected_reference_tags
                        await session.commit()

                        logger.info(
                            f"Shot {shot.shot_index}: keyframe prompt rewritten "
                            f"(refs: {selected_reference_tags})"
                        )

                    if (
                        not is_comfyui and _is_nano_banana_model(image_model)
                    ) or _uses_comfyui_vision_primary(scene):
                        selected_tags = (
                            list(shot_manifest_row.selected_reference_tags or [])
                            if shot_manifest_row and shot_manifest_row.selected_reference_tags
                            else []
                        )
                        ref_context = await _assemble_nano_banana_reference_context(
                            session,
                            production_bible_id=scene.production_bible_id,
                            scene_prompt=scene.prompt,
                            shot=shot,
                            shot_manifest_json=shot_manifest_row.manifest_json if shot_manifest_row else None,
                            selected_reference_tags=selected_tags,
                            all_assets=all_assets,
                            file_mgr=file_mgr,
                        )

                        ref_image_bytes_list = ref_context.ref_image_bytes_list
                        character_text_descriptions = ref_context.character_text_descriptions
                        placed_char_assets = ref_context.placed_char_assets

                        if all_assets and placed_char_assets:
                            asset_map_by_id = {str(asset.id): asset for asset in all_assets}
                            for asset in placed_char_assets:
                                if asset.face_embedding is None and asset.source_asset_id:
                                    parent = asset_map_by_id.get(str(asset.source_asset_id))
                                    if parent and parent.face_embedding:
                                        asset.face_embedding = parent.face_embedding
                                        logger.info(
                                            "Shot %s: borrowed face embedding from parent %s for %s",
                                            shot.shot_index,
                                            parent.manifest_tag,
                                            asset.manifest_tag,
                                        )

                        if ref_image_bytes_list:
                            logger.info(
                                "Shot %s: resolved %s Nano Banana reference image(s) across tags %s",
                                shot.shot_index,
                                len(ref_image_bytes_list),
                                ref_context.final_reference_tags,
                            )
                            identity_details = {
                                tag: ref_context.identity_types_by_tag.get(tag, "HUMAN")
                                for tag in _ordered_unique_tags(ref_context.final_reference_tags)
                                if tag in ref_context.identity_types_by_tag
                            }
                            detail_lines = [
                                f"Mandatory character tags: {ref_context.mandatory_character_tags or ['none']}",
                                f"Optional extra tags: {ref_context.optional_reference_tags or ['none']}",
                                f"Final reference order: {ref_context.final_reference_tags}",
                            ]
                            if identity_details:
                                detail_lines.append(f"Identity policies: {identity_details}")
                            if ref_context.trimmed_reference_counts:
                                detail_lines.append(
                                    f"Trimmed refs by tag: {ref_context.trimmed_reference_counts}"
                                )
                            if ref_context.canonical_tag_remaps:
                                detail_lines.append(
                                    f"Canonical tag remaps: {ref_context.canonical_tag_remaps}"
                                )
                            emit_task_log(
                                scene.id,
                                summary=(
                                    f"Shot {shot.shot_index + 1}: resolved {len(ref_image_bytes_list)} "
                                    "Nano Banana reference image(s)"
                                ),
                                detail="\n".join(detail_lines),
                                phase="keyframes",
                                shot_index=shot.shot_index,
                                kind="keyframe.references",
                                source="nano_banana_resolver",
                            )
                except Exception as e:
                    logger.warning(
                        f"Shot {shot.shot_index}: keyframe rewriter failed (non-fatal): {e}"
                    )
                    rewritten_start_prompt = None
                    ref_image_bytes_list = []
                    character_text_descriptions = []
                    placed_char_assets = []

            # Face verification retry config (max 3 retries = 4 total attempts)
            # Level 0: standard, Level 1: escalated, Level 2+: iterative refinement
            _max_identity_retries = 3

            # Collect prequalified human face embeddings by character tag.
            # Appearance refs become face crops; wardrobe refs remain full-body refs.
            _prequalified_ref_embeddings_by_tag: dict[str, list] = {}

            # Extract character appearance descriptions for identity prompt anchoring
            _char_text_description: str | list[str] | None = None
            if character_text_descriptions:
                _char_text_description = character_text_descriptions
            else:
                for ca in placed_char_assets:
                    if getattr(ca, "reverse_prompt", None):
                        _char_text_description = ca.reverse_prompt
                        break

            _reference_identity_types = [
                candidate.identity_type or "HUMAN"
                for candidate in getattr(ref_context, "selected_candidates", [])
                if candidate.asset_type == "CHARACTER"
            ] if ref_image_bytes_list else []

            if ref_image_bytes_list and _uses_comfyui_vision_primary(scene):
                logger.info(
                    "Shot %s: skipping local face prequalification for ComfyUI multi-ref model %s",
                    shot.shot_index,
                    scene.image_model,
                )
                # Occluded refs (sunglasses, averted faces) dilute identity
                # conditioning in reference-latent models — qualify the
                # character refs and keep only the clearest faces.
                qualified_candidates, qualification_detail = (
                    await _qualify_multiref_identity_candidates(
                        session=session,
                        scene=scene,
                        candidates=list(getattr(ref_context, "selected_candidates", [])),
                    )
                )
                ref_context.selected_candidates = list(qualified_candidates)
                ref_image_bytes_list = [c.image_bytes for c in qualified_candidates]
                emit_task_log(
                    scene.id,
                    summary=(
                        f"Shot {shot.shot_index + 1}: qualified "
                        f"{len(ref_image_bytes_list)} reference image(s) for "
                        "ComfyUI multi-ref generation"
                    ),
                    detail=(
                        "Local face prequalification was skipped for this ComfyUI "
                        "multi-reference image model; vision verification runs "
                        "against the generated frame after image generation.\n"
                        f"Identity ref qualification: {qualification_detail}"
                    ),
                    phase="keyframes",
                    shot_index=shot.shot_index,
                    kind="keyframe.identity_policy",
                    source="identity_policy",
                )
            elif ref_image_bytes_list:
                # HUMAN face refs are converted to face crops; non-human and wardrobe refs pass through.
                try:
                    filtered_candidates, _prequalified_ref_embeddings_by_tag, policy_report = (
                        await _apply_identity_policy_to_reference_candidates(
                            selected_candidates=getattr(ref_context, "selected_candidates", []),
                            file_mgr=file_mgr,
                        )
                    )
                    ref_image_bytes_list = [candidate.image_bytes for candidate in filtered_candidates]
                    ref_context.selected_candidates = list(filtered_candidates)
                    _reference_identity_types = [
                        candidate.identity_type or "HUMAN"
                        for candidate in filtered_candidates
                        if candidate.asset_type == "CHARACTER"
                    ]
                    if (
                        policy_report["human_face_tags"]
                        or policy_report["human_wardrobe_tags"]
                        or policy_report["passthrough_tags"]
                    ):
                        detail_lines = [
                            f"HUMAN face tags prequalified: {policy_report['human_face_tags'] or ['none']}",
                            f"HUMAN wardrobe tags preserved: {policy_report['human_wardrobe_tags'] or ['none']}",
                            f"Non-human passthrough tags: {policy_report['passthrough_tags'] or ['none']}",
                            f"Qualified HUMAN face refs: {policy_report['qualified_face_urls']}",
                        ]
                        if policy_report["dropped_face_urls"]:
                            detail_lines.append(
                                f"Dropped HUMAN face refs without detectable face: {policy_report['dropped_face_urls']}"
                            )
                        emit_task_log(
                            scene.id,
                            summary=(
                                f"Shot {shot.shot_index + 1}: applied identity policy to "
                                f"{len(getattr(ref_context, 'selected_candidates', []))} reference image(s)"
                            ),
                            detail="\n".join(detail_lines),
                            phase="keyframes",
                            shot_index=shot.shot_index,
                            kind="keyframe.identity_policy",
                            source="identity_policy",
                        )
                    logger.info(
                        "Shot %s: identity policy kept %s ref(s); human_faces=%s human_wardrobe=%s passthrough=%s",
                        shot.shot_index,
                        len(ref_image_bytes_list),
                        policy_report["human_face_tags"],
                        policy_report["human_wardrobe_tags"],
                        policy_report["passthrough_tags"],
                    )
                except Exception as e:
                    logger.warning(f"Ref prequalification failed (non-fatal): {e}")

            selected_reference_candidates = list(getattr(ref_context, "selected_candidates", []))
            shot_manifest_json = shot_manifest_row.manifest_json if shot_manifest_row else None

            # ---- START FRAME: Generate or inherit (skip if existing) ----
            if existing_start_kf:
                # Gap-filling: start keyframe already exists (user upload or fork)
                start_frame_bytes = await file_mgr.read_bytes(existing_start_kf.file_path)
                start_source = existing_start_kf.source
                logger.info(
                    f"Shot {shot.shot_index}: start keyframe exists, skipping generation"
                )
            elif shot.shot_index == 0:
                # Shot 0: Generate from text prompt (KEYF-01)
                # Prepend style guide + character bible for maximum fidelity
                # Phase 10: Use rewritten prompt when available (already includes asset details)
                if rewritten_start_prompt:
                    enriched_prompt = f"{style_prefix}{rewritten_start_prompt}"
                else:
                    enriched_prompt = f"{style_prefix}{character_prefix}{shot.start_frame_prompt}"

                start_frame_bytes = None
                previous_attempt_bytes: bytes | None = None
                start_retry_guidance = ""
                start_attempts: list[_GeneratedKeyframeAttempt] = []
                start_verification = _FrameVerificationResult(
                    passed=True,
                    attempts=0,
                    summary="no_verification_needed",
                    detail="no_verification_needed",
                    status="passed",
                )
                for identity_level in range(_max_identity_retries + 1):
                    attempt_candidates = _build_retry_reference_candidates(
                        selected_reference_candidates,
                        mandatory_tags=ref_context.mandatory_character_tags,
                        retry_level=identity_level,
                    )
                    attempt_ref_image_bytes = [candidate.image_bytes for candidate in attempt_candidates]
                    attempt_identity_types = [
                        candidate.identity_type or "HUMAN"
                        for candidate in attempt_candidates
                        if candidate.asset_type == "CHARACTER"
                    ]
                    # Build identity instruction with feature anchoring
                    identity_instr = _build_identity_instruction(
                        _char_text_description,
                        identity_types=attempt_identity_types or _reference_identity_types,
                        emphasis_level=identity_level,
                    ) if attempt_ref_image_bytes else None

                    prompt_with_emphasis = (
                        _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                        + enriched_prompt
                    )
                    if start_retry_guidance:
                        prompt_with_emphasis = f"{start_retry_guidance}\n\n{prompt_with_emphasis}"
                    # ComfyUI reference-latent models treat refs as a weak hint;
                    # the feature-anchored identity text measurably improves
                    # likeness, so include it in the prompt for those branches.
                    comfy_identity_prompt = (
                        f"{prompt_with_emphasis}\n{identity_instr}"
                        if identity_instr and _uses_comfyui_vision_primary(scene)
                        else prompt_with_emphasis
                    )
                    try:
                        if image_model == "qwen-image-edit-2509":
                            # Multi-ref edit model: compose start frame from
                            # identity references at the scene aspect ratio.
                            from vidpipe.services.comfyui_client import _QWEN_RESOLUTIONS
                            qw, qh = _QWEN_RESOLUTIONS.get(scene.aspect_ratio, (1328, 1328))
                            if attempt_ref_image_bytes:
                                compose_prompt = (
                                    "Using the people, characters, and objects shown in the "
                                    "input images as exact identity references, create a new "
                                    f"scene: {comfy_identity_prompt}"
                                )
                                start_frame_bytes = await _generate_image_comfyui_qwen_edit_2509(
                                    comfy_client, compose_prompt, seed=base_seed,
                                    image_bytes_list=attempt_ref_image_bytes[:3],
                                    output_width=qw, output_height=qh,
                                )
                            else:
                                # Edit model needs input images; without refs
                                # fall back to qwen txt2img at scene aspect.
                                logger.info(
                                    f"Shot {shot.shot_index}: qwen-image-edit-2509 has no "
                                    "reference images, falling back to qwen txt2img"
                                )
                                start_frame_bytes = await _generate_image_comfyui(
                                    comfy_client, prompt_with_emphasis, seed=base_seed,
                                    width=qw, height=qh,
                                )
                        elif image_model == "flux-2-klein":
                            from vidpipe.services.comfyui_client import _FLUX2_RESOLUTIONS
                            f2w, f2h = _FLUX2_RESOLUTIONS.get(scene.aspect_ratio, (1024, 1024))
                            start_frame_bytes = await _generate_image_comfyui_flux2_klein(
                                comfy_client, comfy_identity_prompt, seed=base_seed,
                                width=f2w, height=f2h,
                                reference_image_bytes_list=attempt_ref_image_bytes[:4] or None,
                            )
                        elif is_comfyui and image_model.startswith("flux-"):
                            # Flux model: use binding-based reference resolution
                            flux_lora = None
                            flux_ref_filenames: list[str] = []
                            flux_ref_strengths: list[float] = []
                            if scene.production_bible_id:
                                try:
                                    from vidpipe.services.tag_resolver import resolve_tags_with_assets
                                    resolved = await resolve_tags_with_assets(
                                        prompt_with_emphasis, scene.production_bible_id, session,
                                    )
                                    # Categorize by asset_type
                                    for aref in resolved.asset_refs:
                                        if aref.asset_type == "CHARACTER" and aref.lora_url and not flux_lora:
                                            flux_lora = aref.lora_url
                                        elif aref.reference_image_urls:
                                            # CHARACTER without LoRA, PROP, SET -> reference image
                                            ref_url = aref.reference_image_urls[0]
                                            ref_data = await file_mgr.read_bytes(ref_url)
                                            uploaded_name = await comfy_client.upload_image(
                                                ref_data, f"flux_ref_{aref.tag}.png"
                                            )
                                            flux_ref_filenames.append(uploaded_name)
                                            flux_ref_strengths.append(0.65)
                                    if flux_lora or flux_ref_filenames:
                                        logger.info(
                                            f"Shot {shot.shot_index} start: Flux bindings resolved "
                                            f"lora={flux_lora is not None}, refs={len(flux_ref_filenames)}"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"Shot {shot.shot_index}: Flux binding resolution failed "
                                        f"(non-fatal): {e}"
                                    )
                                    flux_lora = None
                                    flux_ref_filenames = []
                                    flux_ref_strengths = []
                            from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS
                            fw, fh = _FLUX_RESOLUTIONS.get(scene.aspect_ratio, (1024, 1024))
                            start_frame_bytes = await _generate_image_comfyui_flux(
                                comfy_client, prompt_with_emphasis, seed=base_seed,
                                width=fw, height=fh,
                                lora_filename=flux_lora,
                                reference_image_filenames=flux_ref_filenames or None,
                                reference_strengths=flux_ref_strengths or None,
                            )
                        elif is_comfyui:
                            # qwen-image-edit needs an input image; for shot 0 start
                            # there's none, so fall back to qwen-fast txt2img
                            start_frame_bytes = await _generate_image_comfyui(
                                comfy_client, prompt_with_emphasis, seed=base_seed,
                            )
                        elif identity_level >= 2 and previous_attempt_bytes is not None:
                            start_frame_bytes = await _generate_image_conditioned(
                                image_client, previous_attempt_bytes, prompt_with_emphasis,
                                scene.aspect_ratio, image_model,
                                reference_images=attempt_ref_image_bytes or None,
                                identity_instruction=identity_instr,
                            )
                        else:
                            start_frame_bytes = await _generate_image_from_text(
                                image_client, prompt_with_emphasis, scene.aspect_ratio, image_model,
                                seed=base_seed,
                                reference_images=attempt_ref_image_bytes or None,
                                identity_instruction=identity_instr,
                            )
                    except RetryError as exc:
                        transport_detail = _describe_retry_error(exc)
                        emit_task_log(
                            scene.id,
                            summary=f"Shot {shot.shot_index + 1}: start keyframe transport retries exhausted",
                            detail=transport_detail,
                            phase="keyframes",
                            shot_index=shot.shot_index,
                            level="warning",
                            kind="keyframe.transport_retry_exhausted",
                            source="image_generator",
                        )
                        fallback = _build_best_effort_fallback_result(
                            position="start",
                            attempts=start_attempts,
                            transport_detail=transport_detail,
                        )
                        if fallback is not None:
                            selected_attempt, start_verification = fallback
                            start_frame_bytes = selected_attempt.keyframe_bytes
                            emit_task_log(
                                scene.id,
                                summary=(
                                    f"Shot {shot.shot_index + 1}: start keyframe accepted with warnings "
                                    "after transport exhaustion"
                                ),
                                detail=start_verification.detail,
                                phase="keyframes",
                                shot_index=shot.shot_index,
                                level="warning",
                                kind="keyframe.best_effort_fallback",
                                source="vision_verifier",
                            )
                            break
                        raise RuntimeError(
                            f"Shot {shot.shot_index} start keyframe generation exhausted transport retries: "
                            f"{transport_detail}"
                        ) from exc
                    should_verify = bool(
                        attempt_candidates
                        and (
                            _prequalified_ref_embeddings_by_tag
                            or attempt_identity_types
                            or ref_context.identity_types_by_tag
                        )
                    )
                    if should_verify:
                        verification_report = await _verify_generated_keyframe(
                            session=session,
                            scene=scene,
                            shot=shot,
                            position="start",
                            keyframe_bytes=start_frame_bytes,
                            shot_manifest_json=shot_manifest_json,
                            selected_candidates=attempt_candidates,
                            identity_types_by_tag=ref_context.identity_types_by_tag,
                            placed_char_assets=placed_char_assets,
                            prequalified_ref_embeddings_by_tag=_prequalified_ref_embeddings_by_tag or None,
                        )
                        start_attempts.append(_GeneratedKeyframeAttempt(
                            attempt_number=identity_level + 1,
                            retry_level=identity_level,
                            keyframe_bytes=start_frame_bytes,
                            verification_report=verification_report,
                        ))
                        start_verification = _FrameVerificationResult(
                            passed=verification_report.passed,
                            attempts=identity_level + 1,
                            summary=(
                                "verification_passed"
                                if verification_report.passed
                                else "verification_failed"
                            ),
                            detail=verification_report.detail,
                            status="passed" if verification_report.passed else "failed",
                        )
                        emit_task_log(
                            scene.id,
                            summary=(
                                f"Shot {shot.shot_index + 1}: start keyframe verification "
                                f"{'passed' if verification_report.passed else 'failed'} on attempt {identity_level + 1}"
                            ),
                            detail=(
                                f"Retry mode: {_retry_mode_label(identity_level)}\n"
                                f"Verification mode: {verification_report.verification_mode}\n"
                                f"{verification_report.detail}"
                            ),
                            phase="keyframes",
                            shot_index=shot.shot_index,
                            level="success" if verification_report.passed else "info",
                            kind="keyframe.verification",
                            source="vision_verifier",
                        )
                        if verification_report.passed:
                            break
                        start_retry_guidance = _build_retry_correction_prompt(verification_report)
                        previous_attempt_bytes = start_frame_bytes
                        if identity_level >= _max_identity_retries:
                            fallback = _build_best_effort_fallback_result(
                                position="start",
                                attempts=start_attempts,
                            )
                            if fallback is None:
                                raise RuntimeError(
                                    f"Shot {shot.shot_index} start keyframe failed verification after "
                                    f"{identity_level + 1} attempts: {verification_report.detail}"
                                )
                            selected_attempt, start_verification = fallback
                            start_frame_bytes = selected_attempt.keyframe_bytes
                            emit_task_log(
                                scene.id,
                                summary=(
                                    f"Shot {shot.shot_index + 1}: start keyframe accepted with warnings "
                                    "using best-effort fallback"
                                ),
                                detail=start_verification.detail,
                                phase="keyframes",
                                shot_index=shot.shot_index,
                                level="warning",
                                kind="keyframe.best_effort_fallback",
                                source="vision_verifier",
                            )
                            break
                        continue
                    start_verification = _FrameVerificationResult(
                        passed=True,
                        attempts=identity_level + 1,
                        summary="verification_skipped",
                        detail="verification_skipped",
                        status="passed",
                    )
                    break
                start_source = "generated"

                # Save start keyframe
                start_stored_path = await file_mgr.save_keyframe_async(
                    scene.id, shot.shot_index, "start", start_frame_bytes
                )

                # Create start keyframe database record
                start_keyframe = Keyframe(
                    shot_id=shot.id,
                    position="start",
                    file_path=start_stored_path,
                    mime_type="image/png",
                    source=start_source,
                    prompt_used=shot.start_frame_prompt,
                    verification_status=start_verification.status,
                    verification_attempts=start_verification.attempts,
                    verification_summary=start_verification.detail,
                )
                session.add(start_keyframe)
            else:
                # Shot N: Inherit from previous shot's end frame (KEYF-03)
                start_frame_bytes = previous_end_frame_bytes
                start_source = "inherited"
                emit_task_log(
                    scene.id,
                    summary=(
                        f"Shot {shot.shot_index + 1}: reusing shot {shot.shot_index} end "
                        "keyframe as the inherited start frame"
                    ),
                    detail=(
                        "No new start keyframe generation request was sent. "
                        "This start frame is byte-identical to the previous shot's end frame."
                    ),
                    phase="keyframes",
                    shot_index=shot.shot_index,
                    kind="keyframe.inherited_start",
                    source="continuity_chain",
                )

                # Save start keyframe
                start_stored_path = await file_mgr.save_keyframe_async(
                    scene.id, shot.shot_index, "start", start_frame_bytes
                )

                # Create start keyframe database record
                start_keyframe = Keyframe(
                    shot_id=shot.id,
                    position="start",
                    file_path=start_stored_path,
                    mime_type="image/png",
                    source=start_source,
                    prompt_used=shot.start_frame_prompt,
                    verification_status="inherited",
                    verification_attempts=0,
                    verification_summary="Inherited from previous shot end keyframe.",
                )
                session.add(start_keyframe)

            # ---- END FRAME: Generate with conditioning (skip if existing) ----
            if existing_end_kf:
                # Gap-filling: end keyframe already exists (user upload or fork)
                end_frame_bytes = await file_mgr.read_bytes(existing_end_kf.file_path)
                logger.info(
                    f"Shot {shot.shot_index}: end keyframe exists, skipping generation"
                )
            else:
                # Update generation_status for end frame (VGED-05)
                shot.generation_status = "generating_end_kf"
                await session.commit()
                event_bus.emit(
                    scene.id,
                    "shot_status",
                    shot_index=shot.shot_index,
                    status="generating_end_kf",
                    phase="keyframes",
                    message=f"Shot {shot.shot_index + 1}: generating end keyframe",
                )

                style_label = scene.style.replace("_", " ")
                conditioning_prompt = (
                    f"Generate the NEXT keyframe for this {style_label} shot, "
                    f"showing clear visual progression {scene.target_clip_duration} seconds later.\n\n"
                    f"TARGET END STATE (this is what the new image must depict):\n"
                    f"{shot.end_frame_prompt}\n\n"
                    f"The new image MUST show VISIBLE CHANGES from the reference image — "
                    f"different pose, expression, body position, or camera framing. "
                    f"If the reference is a close-up, the new image should show "
                    f"a noticeably different expression, head angle, or gesture.\n\n"
                    f"CONSISTENCY CONSTRAINTS:\n"
                    f"- Same character appearance (face, hair, clothing, proportions)\n"
                    f"- Same {style_label} rendering style\n"
                    f"{character_prefix}"
                )

                end_frame_bytes = None
                previous_end_attempt_bytes: bytes | None = None
                end_retry_guidance = ""
                end_attempts: list[_GeneratedKeyframeAttempt] = []
                end_verification = _FrameVerificationResult(
                    passed=True,
                    attempts=0,
                    summary="no_verification_needed",
                    detail="no_verification_needed",
                    status="passed",
                )
                for identity_level in range(_max_identity_retries + 1):
                    attempt_candidates = _build_retry_reference_candidates(
                        selected_reference_candidates,
                        mandatory_tags=ref_context.mandatory_character_tags,
                        retry_level=identity_level,
                    )
                    attempt_ref_image_bytes = [candidate.image_bytes for candidate in attempt_candidates]
                    attempt_identity_types = [
                        candidate.identity_type or "HUMAN"
                        for candidate in attempt_candidates
                        if candidate.asset_type == "CHARACTER"
                    ]
                    # Build identity instruction with feature anchoring
                    end_identity_instr = _build_identity_instruction(
                        _char_text_description,
                        identity_types=attempt_identity_types or _reference_identity_types,
                        emphasis_level=identity_level,
                    ) if attempt_ref_image_bytes else None

                    prompt_with_emphasis = (
                        _IDENTITY_EMPHASIS_PREFIXES[min(identity_level, len(_IDENTITY_EMPHASIS_PREFIXES) - 1)]
                        + conditioning_prompt
                    )
                    if end_retry_guidance:
                        prompt_with_emphasis = f"{end_retry_guidance}\n\n{prompt_with_emphasis}"
                    # Feature-anchored identity text for ComfyUI reference models
                    # (see start-frame block).
                    comfy_identity_prompt = (
                        f"{prompt_with_emphasis}\n{end_identity_instr}"
                        if end_identity_instr and _uses_comfyui_vision_primary(scene)
                        else prompt_with_emphasis
                    )
                    try:
                        if image_model == "qwen-image-edit-2509":
                            # Multi-ref edit: image1 = start frame (drives output
                            # dimensions + visual conditioning), image2/3 = identity refs.
                            end_frame_bytes = await _generate_image_comfyui_qwen_edit_2509(
                                comfy_client, comfy_identity_prompt,
                                seed=base_seed + shot.shot_index + 1000,
                                image_bytes_list=[start_frame_bytes] + list(attempt_ref_image_bytes[:2]),
                            )
                        elif image_model == "flux-2-klein":
                            # Ref slot 1 = start frame (visual conditioning),
                            # slots 2-4 = identity refs.
                            from vidpipe.services.comfyui_client import _FLUX2_RESOLUTIONS as _F2R
                            ef2w, ef2h = _F2R.get(scene.aspect_ratio, (1024, 1024))
                            end_frame_bytes = await _generate_image_comfyui_flux2_klein(
                                comfy_client, comfy_identity_prompt,
                                seed=base_seed + shot.shot_index + 1000,
                                width=ef2w, height=ef2h,
                                reference_image_bytes_list=(
                                    [start_frame_bytes] + list(attempt_ref_image_bytes[:3])
                                ),
                            )
                        elif is_comfyui and image_model.startswith("flux-"):
                            # Flux model: text-only (no image conditioning), offset seed
                            # Reuse binding-resolved LoRA/refs if available from start frame
                            from vidpipe.services.comfyui_client import _FLUX_RESOLUTIONS as _FR
                            efw, efh = _FR.get(scene.aspect_ratio, (1024, 1024))
                            end_frame_bytes = await _generate_image_comfyui_flux(
                                comfy_client, prompt_with_emphasis,
                                seed=base_seed + shot.shot_index + 1000,
                                width=efw, height=efh,
                            )
                        elif is_comfyui:
                            if image_model == "qwen-image-edit":
                                # Use image-edit workflow with start frame as input
                                end_frame_bytes = await _generate_image_comfyui_edit(
                                    comfy_client, prompt_with_emphasis,
                                    input_image_bytes=start_frame_bytes,
                                    seed=base_seed + shot.shot_index + 1000,
                                )
                            else:
                                # ComfyUI text-only: no image conditioning, use offset seed
                                end_frame_bytes = await _generate_image_comfyui(
                                    comfy_client, prompt_with_emphasis,
                                    seed=base_seed + shot.shot_index + 1000,
                                )
                        elif identity_level >= 2 and previous_end_attempt_bytes is not None:
                            end_frame_bytes = await _generate_image_conditioned(
                                image_client, previous_end_attempt_bytes, prompt_with_emphasis,
                                scene.aspect_ratio, image_model,
                                reference_images=attempt_ref_image_bytes or None,
                                identity_instruction=end_identity_instr,
                            )
                        else:
                            end_frame_bytes = await _generate_image_conditioned(
                                image_client, start_frame_bytes, prompt_with_emphasis,
                                scene.aspect_ratio, image_model,
                                reference_images=attempt_ref_image_bytes or None,
                                identity_instruction=end_identity_instr,
                            )
                    except RetryError as exc:
                        transport_detail = _describe_retry_error(exc)
                        emit_task_log(
                            scene.id,
                            summary=f"Shot {shot.shot_index + 1}: end keyframe transport retries exhausted",
                            detail=transport_detail,
                            phase="keyframes",
                            shot_index=shot.shot_index,
                            level="warning",
                            kind="keyframe.transport_retry_exhausted",
                            source="image_generator",
                        )
                        fallback = _build_best_effort_fallback_result(
                            position="end",
                            attempts=end_attempts,
                            transport_detail=transport_detail,
                        )
                        if fallback is not None:
                            selected_attempt, end_verification = fallback
                            end_frame_bytes = selected_attempt.keyframe_bytes
                            emit_task_log(
                                scene.id,
                                summary=(
                                    f"Shot {shot.shot_index + 1}: end keyframe accepted with warnings "
                                    "after transport exhaustion"
                                ),
                                detail=end_verification.detail,
                                phase="keyframes",
                                shot_index=shot.shot_index,
                                level="warning",
                                kind="keyframe.best_effort_fallback",
                                source="vision_verifier",
                            )
                            break
                        raise RuntimeError(
                            f"Shot {shot.shot_index} end keyframe generation exhausted transport retries: "
                            f"{transport_detail}"
                        ) from exc
                    should_verify = bool(
                        attempt_candidates
                        and (
                            _prequalified_ref_embeddings_by_tag
                            or attempt_identity_types
                            or ref_context.identity_types_by_tag
                        )
                    )
                    if should_verify:
                        verification_report = await _verify_generated_keyframe(
                            session=session,
                            scene=scene,
                            shot=shot,
                            position="end",
                            keyframe_bytes=end_frame_bytes,
                            shot_manifest_json=shot_manifest_json,
                            selected_candidates=attempt_candidates,
                            identity_types_by_tag=ref_context.identity_types_by_tag,
                            placed_char_assets=placed_char_assets,
                            prequalified_ref_embeddings_by_tag=_prequalified_ref_embeddings_by_tag or None,
                        )
                        end_attempts.append(_GeneratedKeyframeAttempt(
                            attempt_number=identity_level + 1,
                            retry_level=identity_level,
                            keyframe_bytes=end_frame_bytes,
                            verification_report=verification_report,
                        ))
                        end_verification = _FrameVerificationResult(
                            passed=verification_report.passed,
                            attempts=identity_level + 1,
                            summary=(
                                "verification_passed"
                                if verification_report.passed
                                else "verification_failed"
                            ),
                            detail=verification_report.detail,
                            status="passed" if verification_report.passed else "failed",
                        )
                        emit_task_log(
                            scene.id,
                            summary=(
                                f"Shot {shot.shot_index + 1}: end keyframe verification "
                                f"{'passed' if verification_report.passed else 'failed'} on attempt {identity_level + 1}"
                            ),
                            detail=(
                                f"Retry mode: {_retry_mode_label(identity_level)}\n"
                                f"Verification mode: {verification_report.verification_mode}\n"
                                f"{verification_report.detail}"
                            ),
                            phase="keyframes",
                            shot_index=shot.shot_index,
                            level="success" if verification_report.passed else "info",
                            kind="keyframe.verification",
                            source="vision_verifier",
                        )
                        if verification_report.passed:
                            break
                        end_retry_guidance = _build_retry_correction_prompt(verification_report)
                        previous_end_attempt_bytes = end_frame_bytes
                        if identity_level >= _max_identity_retries:
                            fallback = _build_best_effort_fallback_result(
                                position="end",
                                attempts=end_attempts,
                            )
                            if fallback is None:
                                raise RuntimeError(
                                    f"Shot {shot.shot_index} end keyframe failed verification after "
                                    f"{identity_level + 1} attempts: {verification_report.detail}"
                                )
                            selected_attempt, end_verification = fallback
                            end_frame_bytes = selected_attempt.keyframe_bytes
                            emit_task_log(
                                scene.id,
                                summary=(
                                    f"Shot {shot.shot_index + 1}: end keyframe accepted with warnings "
                                    "using best-effort fallback"
                                ),
                                detail=end_verification.detail,
                                phase="keyframes",
                                shot_index=shot.shot_index,
                                level="warning",
                                kind="keyframe.best_effort_fallback",
                                source="vision_verifier",
                            )
                            break
                        continue
                    end_verification = _FrameVerificationResult(
                        passed=True,
                        attempts=identity_level + 1,
                        summary="verification_skipped",
                        detail="verification_skipped",
                        status="passed",
                    )
                    break

                # Save end keyframe
                end_stored_path = await file_mgr.save_keyframe_async(
                    scene.id, shot.shot_index, "end", end_frame_bytes
                )

                # Create end keyframe database record
                end_keyframe = Keyframe(
                    shot_id=shot.id,
                    position="end",
                    file_path=end_stored_path,
                    mime_type="image/png",
                    source="generated",
                    prompt_used=shot.end_frame_prompt,
                    verification_status=end_verification.status,
                    verification_attempts=end_verification.attempts,
                    verification_summary=end_verification.detail,
                )
                session.add(end_keyframe)

            # Update shot status and prepare for next iteration
            shot.status = "keyframes_done"
            shot.generation_status = None  # Clear generation_status (VGED-05)
            previous_end_frame_bytes = end_frame_bytes

            # Commit after each shot for crash recovery
            await session.commit()
            event_bus.emit(
                scene.id,
                "shot_keyframe_ready",
                shot_index=shot.shot_index,
                position="end",
                message=f"Shot {shot.shot_index + 1} keyframes ready",
            )
            event_bus.emit(scene.id, "refresh")

            # Rate limiting delay (KEYF-05)
            await asyncio.sleep(settings.pipeline.image_gen_delay)

        except Exception:
            # On exception: set generation_status to "failed" (VGED-05)
            shot.generation_status = "failed"
            await session.commit()
            raise

    # Restore exact character identity via post-generation face swap (opt-in).
    # Runs after all keyframes are generated/verified and before video generation
    # so the swapped frames are what FLF2V/clip generation consumes.
    await apply_scene_face_swaps(session, scene)

    # Update scene status after all keyframes generated
    scene.status = "generating_video"
    await session.commit()
    event_bus.emit(
        scene.id,
        "phase_completed",
        phase="keyframes",
        message="Keyframe generation complete",
    )


async def _resolve_primary_source_face(
    session: AsyncSession,
    scene: Scene,
    shot: Shot,
    file_mgr: FileManager,
    svc,
) -> tuple[Optional[bytes], Optional[str]]:
    """Resolve the clearest real human face image to use as the swap source.

    Looks at the shot's on-screen characters, resolves each to its bound actor
    references, and returns the clearest real face for the first human character
    that has a usable reference. Returns ``(None, None)`` when the shot has no
    on-screen human character with a real reference (e.g. non-human characters,
    synthetic-only refs, or no production bible) — the caller then skips the shot.

    Resolution is derived purely from the DB (not the live generation loop's
    in-memory state), so it works on resume.
    """
    if not scene.production_bible_id:
        return None, None
    tags = _ordered_unique_tags(normalize_json_list(shot.characters_present))
    if not tags:
        return None, None

    from vidpipe.services.tag_resolver import (
        canonicalize_character_tags,
        resolve_tags_with_assets,
    )

    aliases = await canonicalize_character_tags(tags, scene.production_bible_id, session)
    canonical_tags = _ordered_unique_tags([aliases.get(t, t) for t in tags])

    for tag in canonical_tags:
        resolved = await resolve_tags_with_assets(
            f"@{tag}", scene.production_bible_id, session
        )
        for ref in resolved.asset_refs:
            if ref.asset_type != "CHARACTER":
                continue
            if not _is_human_identity_type(getattr(ref, "identity_type", None)):
                continue
            # Prefer explicit face/identity refs; fall back to generic refs.
            urls = list(getattr(ref, "face_reference_image_urls", []) or [])
            if not urls:
                urls = list(getattr(ref, "reference_image_urls", []) or [])
            candidates: list[bytes] = []
            for url in urls:
                try:
                    candidates.append(await file_mgr.read_bytes(url))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Face-swap: failed to read source ref %s — %s", url, e
                    )
            if not candidates:
                continue
            best = await asyncio.to_thread(svc.pick_clearest, candidates)
            if best is not None:
                return best, (_normalize_reference_tag(ref.tag) or tag)
    return None, None


async def apply_scene_face_swaps(session: AsyncSession, scene: Scene) -> None:
    """Swap each on-screen character's real face onto the scene's keyframes.

    Opt-in (``settings.face_swap.enabled``), idempotent, and resume-safe: each
    keyframe is marked ``face_swapped`` and committed individually, so a crash
    mid-pass resumes at the first un-swapped keyframe and re-runs never double
    swap. The swapped image overwrites the keyframe in place (so video
    generation transparently consumes it) and the pre-swap original is archived
    as a ``_preswap`` sibling for threshold/restoration tuning.
    """
    if not settings.face_swap.enabled:
        return

    from vidpipe.services.face_restore_service import get_face_restore_service
    from vidpipe.services.face_swap_service import get_face_swap_service

    svc = get_face_swap_service()
    if not await asyncio.to_thread(svc.available):
        logger.warning(
            "Face-swap enabled but service unavailable (inswapper model missing?); "
            "leaving keyframes un-swapped for scene %s",
            scene.id,
        )
        return

    file_mgr = FileManager()
    result = await session.execute(
        select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
    )
    shots = result.scalars().all()

    # Cache resolved source face per identical character-tag set (avoids
    # re-detecting the source ref for every shot sharing the same cast).
    source_cache: dict[tuple, tuple[Optional[bytes], Optional[str]]] = {}
    swapped_count = 0

    for shot in shots:
        cache_key = tuple(
            _ordered_unique_tags(normalize_json_list(shot.characters_present))
        )
        if cache_key in source_cache:
            source_png, source_tag = source_cache[cache_key]
        else:
            source_png, source_tag = await _resolve_primary_source_face(
                session, scene, shot, file_mgr, svc
            )
            source_cache[cache_key] = (source_png, source_tag)
        if source_png is None:
            continue

        kfs_result = await session.execute(
            select(Keyframe).where(Keyframe.shot_id == shot.id)
        )
        for kf in kfs_result.scalars().all():
            if kf.face_swapped:
                continue
            # Inherited start frames are byte-identical aliases of the previous
            # shot's end frame — swap the shared pixels once on the owning row.
            if kf.source == "inherited":
                continue
            try:
                target_png = await file_mgr.read_bytes(kf.file_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Face-swap: cannot read keyframe %s — %s", kf.file_path, e
                )
                continue

            swapped, sim = await asyncio.to_thread(
                svc.swap_face_with_score, target_png, source_png
            )
            if swapped is None:
                # No detectable target face (e.g. stylised render) — keep the
                # generated frame but mark done so we don't retry every resume.
                kf.face_swapped = True
                kf.verification_summary = (
                    (kf.verification_summary or "") + " || face_swap=skipped_no_target_face"
                )
                await session.commit()
                continue

            # Post-swap quality: restore the soft 128px face (CodeFormer, GPU) so
            # it matches the keyframe's native sharpness. This restored frame is
            # what feeds video generation.
            final_png = swapped
            note = ""
            if settings.face_swap.restore:
                rsvc = get_face_restore_service()
                if await asyncio.to_thread(rsvc.has_restore):
                    restored = await asyncio.to_thread(
                        rsvc.restore_primary_face, final_png
                    )
                    if restored is not None:
                        final_png = restored
                        note += f" restore=w{settings.face_swap.restore_weight}"

            # Archive the pre-swap original (tuning/QA) then save the final
            # (swapped + restored) keyframe in place.
            await file_mgr.save_keyframe_async(
                scene.id, shot.shot_index, f"{kf.position}_preswap", target_png
            )
            await file_mgr.save_keyframe_async(
                scene.id, shot.shot_index, kf.position, final_png
            )

            # Optional: a separate 4K still artifact (does NOT feed video gen).
            if settings.face_swap.upscale_keyframes:
                rsvc = get_face_restore_service()
                if await asyncio.to_thread(rsvc.has_upscale):
                    up = await asyncio.to_thread(
                        rsvc.upscale, final_png, settings.face_swap.upscale_scale
                    )
                    if up is not None:
                        await file_mgr.save_keyframe_async(
                            scene.id, shot.shot_index, f"{kf.position}_4k", up
                        )
                        note += f" upscale={settings.face_swap.upscale_scale}x"

            kf.face_swapped = True
            kf.verification_summary = (
                (kf.verification_summary or "")
                + f" || face_swap=ok src=@{source_tag} sim={sim:.3f}{note}"
            )
            await session.commit()
            swapped_count += 1

    if swapped_count:
        logger.info(
            "Face-swap: swapped %d keyframe(s) for scene %s", swapped_count, scene.id
        )
    # Release the torch GPU cache used by per-keyframe restoration/upscale so the
    # reserved pool doesn't linger between scenes (model weights stay resident).
    if settings.face_swap.restore or settings.face_swap.upscale_keyframes:
        get_face_restore_service().release()
