"""Storyboard generation using Gemini structured output.

Transforms user text prompts into shot-by-shot breakdowns with:
- Keyframe image prompts (start/end)
- Motion descriptions for video interpolation
- Cross-shot style guide for visual consistency

Spec reference: STOR-01 through STOR-05
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from vidpipe.config import settings
from vidpipe.db.models import Scene, Shot
from vidpipe.db.models import ShotManifest as ShotManifestModel
from vidpipe.db.models import ShotAudioManifest as ShotAudioManifestModel
from vidpipe.schemas.storyboard import CharacterDescription, StoryboardOutput, StyleGuide
from vidpipe.schemas.storyboard_enhanced import (
    EnhancedStoryboardOutput,
    ShotManifestPackageOutput,
    ShotPromptPackageOutput,
)
from vidpipe.services.event_bus import emit_task_log
from vidpipe.services.llm import get_adapter, LLMAdapter
from vidpipe.services.manifest_service import load_manifest_assets, format_asset_registry, format_binding_registry

logger = logging.getLogger(__name__)


_DUPLICATE_AT_TAG_PATTERN = re.compile(r"@{2,}([A-Za-z0-9_]+)")


def _format_llm_detail(*, model_label: str | None, schema_name: str, content: Any) -> str:
    if isinstance(content, str):
        body = content
    else:
        body = json.dumps(content, indent=2)
    return (
        f"MODEL: {model_label or 'unknown'}\n"
        f"SCHEMA: {schema_name}\n\n"
        f"{body}"
    )


def _display_at_tag(tag: str) -> str:
    """Render a tag with exactly one leading @ for human-readable prompts."""
    normalized = (tag or "").lstrip("@")
    return f"@{normalized}" if normalized else "@"


def _collapse_duplicate_at_tags(text: str | None) -> str | None:
    """Collapse repeated @ prefixes in freeform prompt text."""
    if text is None:
        return None
    return _DUPLICATE_AT_TAG_PATTERN.sub(r"@\1", text)


def _sanitized_prompt_fields(
    *,
    start_frame_prompt: str | None,
    end_frame_prompt: str | None,
    video_motion_prompt: str | None,
    transition_notes: str | None,
) -> dict[str, str | None]:
    """Normalize duplicated @tag mentions in persisted storyboard prompts."""
    return {
        "start_frame_prompt": _collapse_duplicate_at_tags(start_frame_prompt),
        "end_frame_prompt": _collapse_duplicate_at_tags(end_frame_prompt),
        "video_motion_prompt": _collapse_duplicate_at_tags(video_motion_prompt),
        "transition_notes": _collapse_duplicate_at_tags(transition_notes),
    }


def _remap_unrecognized_tags(
    shot_manifest_dict: dict,
    asset_tags_set: set[str],
    manifest_characters: list[str],
) -> dict:
    """Remap unrecognized CHARACTER tags to existing manifest assets.

    Deterministic backstop that catches LLM mistakes before persisting
    shot manifests. Does not require an LLM call.

    Strategy:
    1. Collect all placement tags not in asset_tags_set
    2. For each unrecognized tag that looks like a CHARACTER (starts with
       CHAR_ or is declared as CHARACTER in new_asset_declarations),
       attempt to map to an existing manifest character
    3. Mapping order: first unrecognized CHAR -> first manifest CHAR, etc.
    4. Replace tags in-place in the placements list
    5. Remove remapped entries from new_asset_declarations

    Args:
        shot_manifest_dict: Mutable dict of shot_manifest (placements, new_asset_declarations, etc.)
        asset_tags_set: Set of valid manifest tags (e.g., {"CHAR_01", "CHAR_02", "ENV_01"})
        manifest_characters: Ordered list of CHARACTER tags from the manifest
            (e.g., ["CHAR_01", "CHAR_02", "CHAR_03"])

    Returns:
        The mutated shot_manifest_dict (also mutated in-place)
    """
    if not manifest_characters:
        return shot_manifest_dict

    placements = shot_manifest_dict.get("placements", [])
    new_declarations = shot_manifest_dict.get("new_asset_declarations") or []

    # Build set of CHARACTER tags declared in new_asset_declarations
    declared_char_tags = set()
    for decl in new_declarations:
        decl_type = (decl.get("type") or decl.get("asset_type") or "").upper()
        decl_tag = decl.get("tag") or decl.get("asset_tag") or ""
        if decl_type == "CHARACTER" or decl_tag.startswith("CHAR_"):
            declared_char_tags.add(decl_tag)

    # Identify unrecognized CHARACTER tags in placements
    unrecognized_char_tags = []
    for placement in placements:
        tag = placement.get("asset_tag", "")
        if tag not in asset_tags_set:
            # Consider it a CHARACTER tag if it starts with CHAR_ or is in declared chars
            if tag.startswith("CHAR_") or tag in declared_char_tags:
                if tag not in unrecognized_char_tags:
                    unrecognized_char_tags.append(tag)

    if not unrecognized_char_tags:
        return shot_manifest_dict

    # Build the remap: match by order (CHAR_04 -> CHAR_01, CHAR_05 -> CHAR_02, etc.)
    remap: dict[str, str] = {}
    for i, bad_tag in enumerate(unrecognized_char_tags):
        if i < len(manifest_characters):
            remap[bad_tag] = manifest_characters[i]

    if not remap:
        return shot_manifest_dict

    # Apply remap to placements
    remapped_count = 0
    for placement in placements:
        tag = placement.get("asset_tag", "")
        if tag in remap:
            old_tag = tag
            placement["asset_tag"] = remap[tag]
            remapped_count += 1
            logger.info(
                "Tag remap: %s -> %s in placement (role=%s)",
                old_tag, remap[old_tag], placement.get("role", "unknown"),
            )

    # Remove remapped CHARACTER entries from new_asset_declarations
    if new_declarations:
        surviving = []
        for decl in new_declarations:
            decl_tag = decl.get("tag") or decl.get("asset_tag") or ""
            if decl_tag not in remap:
                surviving.append(decl)
            else:
                logger.info(
                    "Tag remap: removed new_asset_declaration for %s (remapped to %s)",
                    decl_tag, remap[decl_tag],
                )
        shot_manifest_dict["new_asset_declarations"] = surviving

    # Also remap any audio dialogue speaker_tags (if audio_manifest references CHAR tags)
    # This is done separately in the caller if needed

    logger.info(
        "Tag remap summary: %d placement(s) remapped, mapping=%s",
        remapped_count, remap,
    )

    return shot_manifest_dict


# System prompt template for Gemini storyboard generation.
# {style} and {aspect_ratio} are filled at runtime.
STORYBOARD_SYSTEM_PROMPT = """You are a storyboard director specializing in short-form video content.

Your task is to transform the user's script into a visual storyboard optimized for AI-powered video generation.

VISUAL STYLE: {style}
This is the MANDATORY visual style. Do NOT fall back to photorealistic or cinematic unless that is the explicitly requested style.

ASPECT RATIO: {aspect_ratio}
Compose all keyframe image prompts for this aspect ratio. For 16:9, favor wide establishing shots and horizontal compositions. For 9:16, favor close-ups, vertical framing, and portrait-oriented compositions.

REQUIREMENTS:
- Break the script into 3-5 distinct visual shots
- Each shot should have clear narrative progression
- Describe all characters that appear in the video with consistent physical details (see characters field)
- Provide detailed image prompts for start and end keyframes
- Include motion descriptions for video interpolation between keyframes
- Add transition notes to ensure visual continuity between shots
- Create a comprehensive style guide aligned with {style}

DETAIL PRESERVATION:
Before composing shots, identify ALL specific details from the script:
- Proper names (people, organizations, products, frameworks, standards)
- Technical terms, acronyms, and domain jargon
- Numbers, dates, statistics, and quantitative claims
- Specific processes, methodologies, or workflows described

Every identified detail MUST appear in at least one shot via visual vehicles:
- On-screen text overlays, titles, or captions
- Documents, reports, or slides visible in the shot
- Whiteboards, screens, monitors showing text
- Signage, nameplates, logos, or banners

The key_details field for each shot must list 3-6 specific terms the shot conveys.

KEYFRAME PROMPT FORMAT (start_frame_prompt and end_frame_prompt):
Each prompt MUST follow this exact structure:
1. MEDIUM DECLARATION (first words): "A {style} rendering of..."
2. SUBJECT: Detailed character description matching the character bible — same face, hair, clothing, proportions every time
3. ACTION/POSE: What the character is doing, body position
4. SETTING: Environment, background elements, including any visible text/signage/screens from key_details
5. LIGHTING: Light source direction, quality, mood
6. CAMERA: Shot type (wide/medium/close-up), angle, lens.
   CRITICAL: The subject's framing must be compatible with the video_motion_prompt's camera movement.
   If the motion calls for upward camera movement, frame the subject LOW in the composition.
7. STYLE CUES: Rendering technique details specific to {style} (e.g., line weight, color fills, shading approach, texture)
8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic

VIDEO MOTION PROMPT FORMAT (video_motion_prompt):
Describe ONLY motion and camera movement. Do NOT re-describe characters, setting, or style.
The keyframe images already provide the visual context — the motion prompt controls what moves.
Focus on: camera movement (pan, dolly, track, crane), subject animation, environmental animation.
Good example: "Slow dolly forward as the subject turns to face the camera, hair gently blowing in the breeze"
Bad example: "A blonde woman in anime style turns around in a congressional hearing room" (re-describes visuals)

FRAMING-MOTION SAFETY:
Camera movement must be compatible with subject positioning in the start keyframe.
- If camera PANS UP or TILTS UP: position the subject in the LOWER half of the start keyframe
- If camera PANS DOWN or TILTS DOWN: position the subject in the UPPER half of the start keyframe
- If camera DOLLIES IN: ensure subject face is clearly visible and centered in start keyframe
- NEVER generate motion that would push the primary subject's head out of frame
- PREFER subtle camera movements (slow dolly, gentle tracking) over dramatic pans/tilts
  when the subject must remain recognizable throughout
- When in doubt, use a STATIC or SLOW DOLLY camera — these preserve character identity best

GOAL: Ensure all shots maintain visual coherence in {style} style while telling a compelling story. Preserve the original script's specific terminology, names, and details — do not reduce domain-specific content to generic visual metaphors."""


def _split_appearance_and_clothing(text: str | None) -> tuple[str, str]:
    """Split a freeform appearance prompt into physical/clothing halves when possible."""
    if not text:
        return "", ""
    cleaned = " ".join(str(text).replace("\n", " ").split())
    match = re.search(r"\bwearing\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return cleaned, ""
    physical = cleaned[:match.start()].strip(" ,.;:-")
    clothing = cleaned[match.end():].strip(" ,.;:-")
    return (physical or cleaned), clothing


def _deterministic_style_guide(style: str, aspect_ratio: str) -> StyleGuide:
    """Build a lightweight style guide without another LLM call."""
    style_label = style.replace("_", " ").strip() or "cinematic"
    if aspect_ratio == "9:16":
        camera_style = (
            "portrait-first framing with close and medium compositions designed for vertical viewing"
        )
    else:
        camera_style = (
            "wide-to-medium cinematic framing with strong horizontal staging for 16:9 playback"
        )
    return StyleGuide(
        visual_style=style_label,
        color_palette=f"{style_label} palette with lighting choices that reinforce the requested mood",
        camera_style=camera_style,
    )


async def _load_storyboard_character_entries(
    session: AsyncSession,
    production_bible_id,
    *,
    using_bindings: bool,
    legacy_assets: list[Any],
) -> list[dict[str, str]]:
    """Load deterministic character descriptions from bindings or legacy assets."""
    entries: list[dict[str, str]] = []

    if not production_bible_id:
        return entries

    if using_bindings:
        from sqlalchemy import select as sa_select
        from vidpipe.db.models import Actor, CastBinding

        result = await session.execute(
            sa_select(CastBinding, Actor)
            .outerjoin(Actor, Actor.id == CastBinding.actor_id)
            .where(CastBinding.production_bible_id == production_bible_id)
            .order_by(CastBinding.created_at)
        )
        seen: set[str] = set()
        for binding, actor in result.all():
            if not binding.tag or binding.tag in seen:
                continue
            seen.add(binding.tag)
            source_text = (
                (actor.base_appearance_prompt if actor else None)
                or binding.character_description
                or binding.character_name
                or binding.tag
            )
            physical, clothing = _split_appearance_and_clothing(source_text)
            if not clothing and binding.wardrobe_override:
                clothing = ", ".join(
                    f"{k}: {v}" for k, v in binding.wardrobe_override.items()
                )
            entries.append({
                "tag": binding.tag,
                "name": binding.character_name or (actor.name if actor else binding.tag),
                "physical_description": physical or binding.character_name or binding.tag,
                "clothing_description": clothing,
            })
        return entries

    seen = set()
    for asset in sorted(legacy_assets, key=lambda a: (a.sort_order, a.manifest_tag, a.name)):
        if asset.manifest_tag in seen:
            continue
        if asset.asset_type != "CHARACTER" and not asset.manifest_tag.startswith("CHAR_"):
            continue
        seen.add(asset.manifest_tag)
        source_text = asset.reverse_prompt or asset.description or asset.name
        physical, clothing = _split_appearance_and_clothing(source_text)
        entries.append({
            "tag": asset.manifest_tag,
            "name": asset.name,
            "physical_description": physical or asset.name,
            "clothing_description": clothing,
        })
    return entries


def _character_models_from_entries(
    entries: list[dict[str, str]],
) -> list[CharacterDescription]:
    """Convert internal character registry entries into public storyboard characters."""
    return [
        CharacterDescription(
            name=entry["name"],
            physical_description=entry["physical_description"],
            clothing_description=entry["clothing_description"],
        )
        for entry in entries
    ]


def _character_bible_block(
    entries: list[dict[str, str]],
    visible_tags: list[str],
) -> str:
    """Render a compact character bible block for the current shot."""
    if not visible_tags:
        return "CHARACTER BIBLE:\n- No visible characters in this shot."

    by_tag = {entry["tag"]: entry for entry in entries}
    lines = ["CHARACTER BIBLE:"]
    for tag in visible_tags:
        entry = by_tag.get(tag)
        if entry:
            clothing = entry["clothing_description"] or "same established wardrobe"
            lines.append(
                f"- {_display_at_tag(tag)}: {entry['name']} — {entry['physical_description']}. Wearing {clothing}."
            )
        else:
            lines.append(f"- {_display_at_tag(tag)}: visible in this shot; keep the established appearance consistent.")
    return "\n".join(lines)


def _existing_storyboard_shots_map(
    scene: Scene,
    existing_shots: list[Shot],
) -> dict[int, dict[str, Any]]:
    """Build a mutable shot-index map from existing storyboard_raw and DB rows."""
    by_index: dict[int, dict[str, Any]] = {}

    raw = scene.storyboard_raw or {}
    if isinstance(raw, dict):
        for shot_entry in raw.get("shots", []):
            if isinstance(shot_entry, dict) and "shot_index" in shot_entry:
                by_index[int(shot_entry["shot_index"])] = dict(shot_entry)

    for shot in existing_shots:
        entry = by_index.get(shot.shot_index, {})
        entry.setdefault("shot_index", shot.shot_index)
        entry.setdefault("key_details", [])
        entry["shot_description"] = shot.shot_description or entry.get("shot_description", "")
        entry["start_frame_prompt"] = shot.start_frame_prompt or entry.get("start_frame_prompt", "")
        entry["end_frame_prompt"] = shot.end_frame_prompt or entry.get("end_frame_prompt", "")
        entry["video_motion_prompt"] = shot.video_motion_prompt or entry.get("video_motion_prompt", "")
        entry["transition_notes"] = shot.transition_notes or entry.get("transition_notes", "")
        by_index[shot.shot_index] = entry

    return by_index


def _storyboard_raw_payload(
    style_guide: StyleGuide,
    characters: list[CharacterDescription],
    shots_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild the public storyboard_raw payload in the legacy shape."""
    return {
        "style_guide": style_guide.model_dump(),
        "characters": [character.model_dump() for character in characters],
        "shots": [shots_by_index[idx] for idx in sorted(shots_by_index)],
    }


def _build_storyboard_shot_entry(
    shot_index: int,
    shot_description: str,
    key_details: list[str],
    prompt_package: ShotPromptPackageOutput,
    manifest_dict: dict[str, Any],
    audio_dict: dict[str, Any],
) -> dict[str, Any]:
    """Build one shot entry for scene.storyboard_raw."""
    sanitized_fields = _sanitized_prompt_fields(
        start_frame_prompt=prompt_package.start_frame_prompt,
        end_frame_prompt=prompt_package.end_frame_prompt,
        video_motion_prompt=prompt_package.video_motion_prompt,
        transition_notes=prompt_package.transition_notes,
    )
    return {
        "shot_index": shot_index,
        "shot_description": shot_description,
        "key_details": key_details,
        "start_frame_prompt": sanitized_fields["start_frame_prompt"],
        "end_frame_prompt": sanitized_fields["end_frame_prompt"],
        "video_motion_prompt": sanitized_fields["video_motion_prompt"],
        "transition_notes": sanitized_fields["transition_notes"],
        "shot_manifest": manifest_dict,
        "audio_manifest": audio_dict,
    }


def _build_manifest_generation_prompt(
    *,
    style_label: str,
    aspect_ratio: str,
    asset_registry_block: str,
    resolved_prompt_text: str,
    screenplay_enrichment: str,
    shot_assignment: Any,
    previous_assignment: Any,
    next_assignment: Any,
    fixed_description: str | None,
) -> str:
    """Prompt for per-shot manifest and audio package generation."""
    sections = [
        "You are generating ONE storyboard shot package for an AI video pipeline.",
        f"VISUAL STYLE: {style_label}",
        f"ASPECT RATIO: {aspect_ratio}",
        "",
        asset_registry_block,
        "",
    ]
    if screenplay_enrichment.strip():
        sections.extend([screenplay_enrichment.strip(), ""])
    sections.extend([
        "ORIGINAL SCRIPT:",
        resolved_prompt_text,
        "",
        "SHOT ASSIGNMENT (authoritative):",
        json.dumps(shot_assignment.model_dump(), indent=2),
    ])
    if previous_assignment is not None:
        sections.extend([
            "",
            "PREVIOUS SHOT SUMMARY:",
            json.dumps(previous_assignment.model_dump(), indent=2),
        ])
    if next_assignment is not None:
        sections.extend([
            "",
            "NEXT SHOT SUMMARY:",
            json.dumps(next_assignment.model_dump(), indent=2),
        ])
    if fixed_description:
        sections.extend([
            "",
            "FIXED SHOT DESCRIPTION:",
            fixed_description,
            "Keep the shot_description semantically identical to the fixed description above.",
        ])
    sections.extend([
        "",
        "Return ONLY this shot's package.",
        "Requirements:",
        "- Use the shot_assignment exactly; do not add characters not listed in characters_present.",
        "- If characters_present is empty, placements must contain NO character assets.",
        "- Produce shot_description, key_details, shot_manifest, and audio_manifest only.",
        "- shot_manifest.asset_tag and audio_manifest.speaker_tag values must use existing raw tags with NO leading @.",
        "- shot_manifest.shot_index and audio_manifest.shot_index must equal the requested shot_index.",
        "- shot_manifest must include at least one placement.",
        "- audio_manifest must always include ambient or another minimal sound layer.",
        "- Do not generate start/end prompts, motion prompts, style guides, or top-level character lists.",
    ])
    return "\n".join(sections)


def _build_prompt_generation_prompt(
    *,
    style_label: str,
    aspect_ratio: str,
    style_guide: StyleGuide,
    shot_assignment: Any,
    manifest_package: ShotManifestPackageOutput,
    character_entries: list[dict[str, str]],
    previous_assignment: Any,
    next_assignment: Any,
    fixed_description: str | None,
) -> str:
    """Prompt for per-shot keyframe/motion prompt writing."""
    sections = [
        "You are writing detailed storyboard prompts for ONE shot.",
        f"VISUAL STYLE: {style_label}",
        f"ASPECT RATIO: {aspect_ratio}",
        "",
        "STYLE GUIDE:",
        json.dumps(style_guide.model_dump(), indent=2),
        "",
        _character_bible_block(character_entries, shot_assignment.characters_present or []),
        "",
        "SHOT ASSIGNMENT:",
        json.dumps(shot_assignment.model_dump(), indent=2),
        "",
        "SHOT PACKAGE:",
        json.dumps(manifest_package.model_dump(), indent=2),
    ]
    if previous_assignment is not None:
        sections.extend([
            "",
            "PREVIOUS SHOT SUMMARY:",
            json.dumps(previous_assignment.model_dump(), indent=2),
        ])
    if next_assignment is not None:
        sections.extend([
            "",
            "NEXT SHOT SUMMARY:",
            json.dumps(next_assignment.model_dump(), indent=2),
        ])
    if fixed_description:
        sections.extend([
            "",
            "FIXED SHOT DESCRIPTION:",
            fixed_description,
        ])
    sections.extend([
        "",
        "Write ONLY: start_frame_prompt, end_frame_prompt, video_motion_prompt, transition_notes.",
        "Prompt rules:",
        f"- Both frame prompts MUST begin with: \"A {style_label} rendering of...\"",
        "- Frame prompts must describe subject, action, setting, lighting, camera, style cues, and color palette.",
        "- video_motion_prompt must describe motion and camera movement only; do not restate appearance details.",
        "- transition_notes must connect this shot to the next shot, or resolve the scene if it is the last shot.",
        "- Keep character appearance consistent with the character bible and shot placements.",
        "- Do not generate style guides, top-level characters, manifests, or audio.",
    ])
    return "\n".join(sections)


async def _run_manifest_shot_pipeline(
    *,
    semaphore: asyncio.Semaphore,
    adapter: LLMAdapter,
    scene_id,
    event_bus,
    manifest_prompt: str,
    style_label: str,
    aspect_ratio: str,
    style_guide: StyleGuide,
    shot_assignment: Any,
    character_entries: list[dict[str, str]],
    previous_assignment: Any,
    next_assignment: Any,
    fixed_description: str | None,
    shot_index: int,
    model_label: str | None,
) -> dict[str, Any]:
    """Run the per-shot manifest + prompt pipeline and return a structured result."""
    try:
        event_bus.emit(
            scene_id,
            "shot_status",
            shot_index=shot_index,
            status="generating_manifest",
            phase="storyboard",
            message=f"Shot {shot_index + 1}: generating manifest and audio plan",
        )
        emit_task_log(
            scene_id,
            phase="storyboard",
            shot_index=shot_index,
            kind="prompt",
            source="storyboard.manifest.prompt",
            summary=f"Shot {shot_index + 1} manifest prompt sent",
            detail=_format_llm_detail(
                model_label=model_label,
                schema_name=ShotManifestPackageOutput.__name__,
                content=manifest_prompt,
            ),
        )
        async with semaphore:
            manifest_package = await adapter.generate_text(
                prompt=manifest_prompt,
                schema=ShotManifestPackageOutput,
                temperature=0.5,
                max_retries=2,
            )
        emit_task_log(
            scene_id,
            phase="storyboard",
            shot_index=shot_index,
            level="success",
            kind="response",
            source="storyboard.manifest.response",
            summary=f"Shot {shot_index + 1} manifest response received",
            detail=_format_llm_detail(
                model_label=model_label,
                schema_name=ShotManifestPackageOutput.__name__,
                content=manifest_package.model_dump(),
            ),
        )

        event_bus.emit(
            scene_id,
            "shot_status",
            shot_index=shot_index,
            status="generating_prompts",
            phase="storyboard",
            message=f"Shot {shot_index + 1}: writing storyboard prompts",
        )
        prompt_prompt = _build_prompt_generation_prompt(
            style_label=style_label,
            aspect_ratio=aspect_ratio,
            style_guide=style_guide,
            shot_assignment=shot_assignment,
            manifest_package=manifest_package,
            character_entries=character_entries,
            previous_assignment=previous_assignment,
            next_assignment=next_assignment,
            fixed_description=fixed_description,
        )
        emit_task_log(
            scene_id,
            phase="storyboard",
            shot_index=shot_index,
            kind="prompt",
            source="storyboard.prompts.prompt",
            summary=f"Shot {shot_index + 1} prompt-writer request sent",
            detail=_format_llm_detail(
                model_label=model_label,
                schema_name=ShotPromptPackageOutput.__name__,
                content=prompt_prompt,
            ),
        )
        async with semaphore:
            prompt_package = await adapter.generate_text(
                prompt=prompt_prompt,
                schema=ShotPromptPackageOutput,
                temperature=0.5,
                max_retries=2,
            )
        emit_task_log(
            scene_id,
            phase="storyboard",
            shot_index=shot_index,
            level="success",
            kind="response",
            source="storyboard.prompts.response",
            summary=f"Shot {shot_index + 1} prompt-writer response received",
            detail=_format_llm_detail(
                model_label=model_label,
                schema_name=ShotPromptPackageOutput.__name__,
                content=prompt_package.model_dump(),
            ),
        )
        return {
            "shot_index": shot_index,
            "manifest_package": manifest_package,
            "prompt_package": prompt_package,
        }
    except Exception as exc:
        emit_task_log(
            scene_id,
            phase="storyboard",
            shot_index=shot_index,
            level="error",
            kind="error",
            source="storyboard.pipeline.error",
            summary=f"Shot {shot_index + 1} storyboard worker failed",
            detail=str(exc),
        )
        return {
            "shot_index": shot_index,
            "error": exc,
        }


async def _generate_manifest_storyboard_parallel(
    session: AsyncSession,
    scene: Scene,
    adapter: LLMAdapter,
    *,
    existing_shots: list[Shot],
    filled_shots: list[Shot],
    empty_shots: list[Shot],
    asset_registry_block: str,
    asset_tags_set: set[str],
    style_label: str,
    screenplay_enrichment: str,
    resolved_prompt_text: str,
    screenwriter_breakdown: Any,
    is_dynamic: bool,
    using_bindings: bool,
    legacy_assets: list[Any],
) -> None:
    """Generate manifest-aware storyboard text via per-shot parallel calls."""
    from sqlalchemy import delete as sa_delete
    from vidpipe.services.event_bus import event_bus

    breakdown_by_index = {
        assignment.shot_index: assignment
        for assignment in screenwriter_breakdown.shots
    }
    existing_by_index = {shot.shot_index: shot for shot in existing_shots}
    if existing_shots:
        target_indices = sorted({
            *[shot.shot_index for shot in empty_shots],
            *[
                assignment.shot_index
                for assignment in screenwriter_breakdown.shots
                if assignment.shot_index not in existing_by_index
            ],
        })
    else:
        target_indices = [
            assignment.shot_index for assignment in screenwriter_breakdown.shots
        ]
    total_targets = len(target_indices)

    if is_dynamic and screenwriter_breakdown.shots:
        scene.target_shot_count = len(screenwriter_breakdown.shots)
        scene.total_duration = scene.target_shot_count * (scene.target_clip_duration or 6)
        logger.info(
            "Scene %s: dynamic shot count → %d shots",
            scene.id, scene.target_shot_count,
        )

    style_guide = _deterministic_style_guide(scene.style, scene.aspect_ratio)
    character_entries = await _load_storyboard_character_entries(
        session,
        scene.production_bible_id,
        using_bindings=using_bindings,
        legacy_assets=legacy_assets,
    )
    character_models = _character_models_from_entries(character_entries)

    shots_by_index = _existing_storyboard_shots_map(scene, existing_shots)

    for shot_row in existing_shots:
        assignment = breakdown_by_index.get(shot_row.shot_index)
        if not assignment:
            continue
        shot_row.characters_present = assignment.characters_present
        shot_row.beat_index = assignment.beat_index
        shot_row.narrative_intent = assignment.narrative_intent
        shot_row.emotional_weight = assignment.emotional_weight

    scene.style_guide = style_guide.model_dump()
    scene.storyboard_raw = _storyboard_raw_payload(
        style_guide,
        character_models,
        shots_by_index,
    )
    await session.commit()

    progress = {
        "completed": 0,
        "active": total_targets,
    }
    heartbeat_stop = asyncio.Event()

    async def _heartbeat() -> None:
        while not heartbeat_stop.is_set():
            await asyncio.sleep(10)
            if heartbeat_stop.is_set():
                break
            event_bus.emit(
                scene.id,
                "phase_progress",
                phase="storyboard",
                completed_shots=progress["completed"],
                total_shots=total_targets,
                active_shots=progress["active"],
                message=(
                    f"Storyboard still generating: {progress['completed']}/{total_targets} shots complete, "
                    f"{progress['active']} active"
                ),
            )

    heartbeat_task = asyncio.create_task(_heartbeat())

    failures: list[tuple[int, Exception | str]] = []
    semaphore = asyncio.Semaphore(max(1, min(4, total_targets or 1)))
    tasks = []

    for shot_index in target_indices:
        assignment = breakdown_by_index.get(shot_index)
        if assignment is None:
            failures.append((shot_index, "missing shot breakdown"))
            progress["active"] -= 1
            continue

        existing_shot = existing_by_index.get(shot_index)
        fixed_description = None
        if existing_shot and existing_shot.shot_description and existing_shot.shot_description.strip():
            fixed_description = existing_shot.shot_description.strip()

        manifest_prompt = _build_manifest_generation_prompt(
            style_label=style_label,
            aspect_ratio=scene.aspect_ratio,
            asset_registry_block=asset_registry_block,
            resolved_prompt_text=resolved_prompt_text,
            screenplay_enrichment=screenplay_enrichment,
            shot_assignment=assignment,
            previous_assignment=breakdown_by_index.get(shot_index - 1),
            next_assignment=breakdown_by_index.get(shot_index + 1),
            fixed_description=fixed_description,
        )

        tasks.append(asyncio.create_task(
            _run_manifest_shot_pipeline(
                semaphore=semaphore,
                adapter=adapter,
                scene_id=scene.id,
                event_bus=event_bus,
                manifest_prompt=manifest_prompt,
                style_label=style_label,
                aspect_ratio=scene.aspect_ratio,
                style_guide=style_guide,
                shot_assignment=assignment,
                character_entries=character_entries,
                previous_assignment=breakdown_by_index.get(shot_index - 1),
                next_assignment=breakdown_by_index.get(shot_index + 1),
                fixed_description=fixed_description,
                shot_index=shot_index,
                model_label=scene.text_model,
            )
        ))

    try:
        event_bus.emit(
            scene.id,
            "phase_progress",
            phase="storyboard",
            completed_shots=0,
            total_shots=total_targets,
            active_shots=progress["active"],
            message=f"Generating storyboard for {total_targets} shot(s)",
        )

        manifest_characters = sorted(
            [tag for tag in asset_tags_set if tag.startswith("CHAR_")]
        )

        for task in asyncio.as_completed(tasks):
            result = await task
            shot_index = result["shot_index"]

            if "error" in result:
                failures.append((shot_index, result["error"]))
                logger.error(
                    "Scene %s shot %d: storyboard generation failed: %s",
                    scene.id, shot_index, result["error"],
                    exc_info=result["error"],
                )
                progress["active"] -= 1
                event_bus.emit(
                    scene.id,
                    "phase_progress",
                    phase="storyboard",
                    completed_shots=progress["completed"],
                    total_shots=total_targets,
                    active_shots=progress["active"],
                    message=(
                        f"Shot {shot_index + 1} failed; {progress['completed']}/{total_targets} shots complete"
                    ),
                )
                continue

            manifest_package: ShotManifestPackageOutput = result["manifest_package"]
            prompt_package: ShotPromptPackageOutput = result["prompt_package"]
            sanitized_fields = _sanitized_prompt_fields(
                start_frame_prompt=prompt_package.start_frame_prompt,
                end_frame_prompt=prompt_package.end_frame_prompt,
                video_motion_prompt=prompt_package.video_motion_prompt,
                transition_notes=prompt_package.transition_notes,
            )
            existing = existing_by_index.get(shot_index)

            shot_description = manifest_package.shot_description
            if existing and existing.shot_description and existing.shot_description.strip():
                shot_description = existing.shot_description

            if existing:
                if not existing.shot_description or not existing.shot_description.strip():
                    existing.shot_description = shot_description
                for attr in (
                    "start_frame_prompt",
                    "end_frame_prompt",
                    "video_motion_prompt",
                    "transition_notes",
                ):
                    current = getattr(existing, attr)
                    if not current or not current.strip():
                        setattr(existing, attr, sanitized_fields[attr])
                shot_row = existing
            else:
                shot_row = Shot(
                    scene_id=scene.id,
                    shot_index=shot_index,
                    shot_description=shot_description,
                    start_frame_prompt=sanitized_fields["start_frame_prompt"],
                    end_frame_prompt=sanitized_fields["end_frame_prompt"],
                    video_motion_prompt=sanitized_fields["video_motion_prompt"],
                    transition_notes=sanitized_fields["transition_notes"],
                    status="pending",
                )
                session.add(shot_row)
                existing_by_index[shot_index] = shot_row

            assignment = breakdown_by_index.get(shot_index)
            if assignment:
                shot_row.characters_present = assignment.characters_present
                shot_row.beat_index = assignment.beat_index
                shot_row.narrative_intent = assignment.narrative_intent
                shot_row.emotional_weight = assignment.emotional_weight
            shot_row.status = "pending"
            shot_row.generation_status = None

            manifest_dict = manifest_package.shot_manifest.model_dump()
            for placement in manifest_dict.get("placements", []):
                tag = placement.get("asset_tag")
                if isinstance(tag, str) and tag.startswith("@"):
                    placement["asset_tag"] = tag[1:]
            _remap_unrecognized_tags(manifest_dict, asset_tags_set, manifest_characters)

            audio_dict = manifest_package.audio_manifest.model_dump()
            audio_dialogue = audio_dict.get("dialogue_lines", []) or []
            for dialogue_entry in audio_dialogue:
                speaker = dialogue_entry.get("speaker_tag", "")
                if isinstance(speaker, str) and speaker.startswith("@"):
                    speaker = speaker[1:]
                    dialogue_entry["speaker_tag"] = speaker
                if speaker.startswith("CHAR_") and speaker not in asset_tags_set:
                    for placement in manifest_dict.get("placements", []):
                        candidate = placement.get("asset_tag", "")
                        if isinstance(candidate, str) and candidate in asset_tags_set and candidate.startswith("CHAR_"):
                            logger.info(
                                "Audio tag remap: speaker_tag %s -> %s",
                                speaker, candidate,
                            )
                            dialogue_entry["speaker_tag"] = candidate
                            break

            await session.execute(
                sa_delete(ShotManifestModel).where(
                    ShotManifestModel.scene_id == scene.id,
                    ShotManifestModel.shot_index == shot_index,
                )
            )
            await session.execute(
                sa_delete(ShotAudioManifestModel).where(
                    ShotAudioManifestModel.scene_id == scene.id,
                    ShotAudioManifestModel.shot_index == shot_index,
                )
            )

            shot_manifest = ShotManifestModel(
                scene_id=scene.id,
                shot_index=shot_index,
                manifest_json=manifest_dict,
                composition_shot_type=manifest_package.shot_manifest.composition.shot_type,
                composition_camera_movement=manifest_package.shot_manifest.composition.camera_movement,
                asset_tags=[p.get("asset_tag", "") for p in manifest_dict.get("placements", [])],
                new_asset_count=len(manifest_dict.get("new_asset_declarations") or []),
            )
            session.add(shot_manifest)

            audio_manifest = ShotAudioManifestModel(
                scene_id=scene.id,
                shot_index=shot_index,
                dialogue_json=audio_dialogue,
                sfx_json=audio_dict.get("sfx", []),
                ambient_json=audio_dict.get("ambient"),
                music_json=audio_dict.get("music"),
                audio_continuity_json=audio_dict.get("audio_continuity"),
                speaker_tags=[d.get("speaker_tag", "") for d in audio_dialogue],
                has_dialogue=len(audio_dialogue) > 0,
                has_music=audio_dict.get("music") is not None,
            )
            session.add(audio_manifest)

            shots_by_index[shot_index] = _build_storyboard_shot_entry(
                shot_index,
                shot_description,
                manifest_package.key_details,
                prompt_package,
                manifest_dict,
                audio_dict,
            )
            scene.storyboard_raw = _storyboard_raw_payload(
                style_guide,
                character_models,
                shots_by_index,
            )
            scene.style_guide = style_guide.model_dump()
            scene.error_message = None

            await session.commit()

            progress["completed"] += 1
            progress["active"] -= 1
            event_bus.emit(
                scene.id,
                "shot_text_ready",
                shot_index=shot_index,
                message=f"Shot {shot_index + 1} storyboard text ready",
            )
            event_bus.emit(
                scene.id,
                "phase_progress",
                phase="storyboard",
                completed_shots=progress["completed"],
                total_shots=total_targets,
                active_shots=progress["active"],
                message=(
                    f"Storyboard complete for shot {shot_index + 1}; "
                    f"{progress['completed']}/{total_targets} shots ready"
                ),
            )

    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    for shot in filled_shots:
        shot.generation_status = None
    for shot in empty_shots:
        if shot.generation_status == "generating_text":
            shot.generation_status = None

    if failures:
        failed_labels = ", ".join(str(index + 1) for index, _ in failures)
        scene.status = "failed"
        scene.error_message = f"Storyboard generation failed for shot(s): {failed_labels}"
        await session.commit()
        raise RuntimeError(scene.error_message)

    scene.status = "keyframing"
    scene.error_message = None
    await session.commit()

    event_bus.emit(
        scene.id,
        "phase_completed",
        phase="storyboard",
        message="Storyboard generation complete",
    )
    event_bus.emit(scene.id, "refresh")


# Enhanced system prompt for manifest-aware storyboarding
ENHANCED_STORYBOARD_PROMPT = """You are a storyboard director specializing in short-form video content.

Your task is to transform the user's script into a visual storyboard optimized for AI-powered video generation.

VISUAL STYLE: {style}
This is the MANDATORY visual style. Do NOT fall back to photorealistic or cinematic unless that is the explicitly requested style.

ASPECT RATIO: {aspect_ratio}
Compose all keyframe image prompts for this aspect ratio. For 16:9, favor wide establishing shots and horizontal compositions. For 9:16, favor close-ups, vertical framing, and portrait-oriented compositions.

AVAILABLE ASSETS (from Asset Registry):
{asset_registry_block}

SHOT MANIFEST INSTRUCTIONS:
When creating shots, generate a shot_manifest for each shot:
- You MUST use registered asset tags from the Available Assets list for ALL characters
  and environments that match existing assets. Do NOT create new CHARACTER tags when
  a matching character already exists in the registry.
- Reference assets by their exact [TAG] (e.g., [CHAR_01], [ENV_02])
- Use the asset's reverse_prompt for visual detail — it's already optimized for generation
- Assign roles: subject, background, prop, interaction_target, environment
- Specify spatial positions and actions for each placed asset
- Include composition metadata: shot_type, camera_movement, focal_point
- Add continuity_notes describing visual continuity with previous shots
- new_asset_declarations: ONLY for genuinely new assets that have NO match in the
  registry (e.g., background extras, props, environments not yet registered).
  NEVER declare a new CHARACTER when the registry already has CHARACTER assets
  that could represent that person.

AUDIO MANIFEST INSTRUCTIONS:
For each shot, generate an audio_manifest with:
- dialogue_lines: Map speech to character tags (speaker_tag must be a registered [TAG] like CHAR_01). Include delivery notes (muttered, shouted, whispered) and timing (start, mid-shot, end)
- sfx: Sound effects with trigger descriptions and relative timing. Use "SFX:" mental model
- ambient: Base layer soundscape + environmental context. Describe what you HEAR, not see
- music: Style, mood, tempo, instruments, and transition cues (fade in, cut, swell)
- audio_continuity: What carries from previous shot, what's new, what cuts

REQUIREMENTS:
- Break the script into 3-5 distinct visual shots
- Each shot should have clear narrative progression
- Describe all characters that appear in the video with consistent physical details (see characters field)
- Provide detailed image prompts for start and end keyframes
- Include motion descriptions for video interpolation between keyframes
- Add transition notes to ensure visual continuity between shots
- Create a comprehensive style guide aligned with {style}
- Each shot MUST include shot_manifest with at least one asset placement
- Each shot MUST include audio_manifest (even if minimal — at least ambient)
- Asset tag references MUST match tags from the Available Assets list above
- Character wardrobe notes in placements MUST be consistent with character descriptions

DETAIL PRESERVATION:
Before composing shots, identify ALL specific details from the script:
- Proper names (people, organizations, products, frameworks, standards)
- Technical terms, acronyms, and domain jargon
- Numbers, dates, statistics, and quantitative claims
- Specific processes, methodologies, or workflows described

Every identified detail MUST appear in at least one shot via visual vehicles:
- On-screen text overlays, titles, or captions
- Documents, reports, or slides visible in the shot
- Whiteboards, screens, monitors showing text
- Signage, nameplates, logos, or banners

The key_details field for each shot must list 3-6 specific terms the shot conveys.

KEYFRAME PROMPT FORMAT (start_frame_prompt and end_frame_prompt):
Each prompt MUST follow this exact structure:
1. MEDIUM DECLARATION (first words): "A {style} rendering of..."
2. SUBJECT: Detailed character description matching the character bible — same face, hair, clothing, proportions every time
3. ACTION/POSE: What the character is doing, body position
4. SETTING: Environment, background elements, including any visible text/signage/screens from key_details
5. LIGHTING: Light source direction, quality, mood
6. CAMERA: Shot type (wide/medium/close-up), angle, lens.
   CRITICAL: The subject's framing must be compatible with the video_motion_prompt's camera movement.
   If the motion calls for upward camera movement, frame the subject LOW in the composition.
7. STYLE CUES: Rendering technique details specific to {style} (e.g., line weight, color fills, shading approach, texture)
8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic

VIDEO MOTION PROMPT FORMAT (video_motion_prompt):
Describe ONLY motion and camera movement. Do NOT re-describe characters, setting, or style.
The keyframe images already provide the visual context — the motion prompt controls what moves.
Focus on: camera movement (pan, dolly, track, crane), subject animation, environmental animation.
Good example: "Slow dolly forward as the subject turns to face the camera, hair gently blowing in the breeze"
Bad example: "A blonde woman in anime style turns around in a congressional hearing room" (re-describes visuals)

FRAMING-MOTION SAFETY:
Camera movement must be compatible with subject positioning in the start keyframe.
- If camera PANS UP or TILTS UP: position the subject in the LOWER half of the start keyframe
- If camera PANS DOWN or TILTS DOWN: position the subject in the UPPER half of the start keyframe
- If camera DOLLIES IN: ensure subject face is clearly visible and centered in start keyframe
- NEVER generate motion that would push the primary subject's head out of frame
- PREFER subtle camera movements (slow dolly, gentle tracking) over dramatic pans/tilts
  when the subject must remain recognizable throughout
- When in doubt, use a STATIC or SLOW DOLLY camera — these preserve character identity best

GOAL: Ensure all shots maintain visual coherence in {style} style while telling a compelling story. Preserve the original script's specific terminology, names, and details — do not reduce domain-specific content to generic visual metaphors."""


async def generate_storyboard(
    session: AsyncSession,
    scene: Scene,
    text_adapter: Optional[LLMAdapter] = None,
) -> None:
    """Generate storyboard from scene prompt using LLM structured output.

    Transforms scene.prompt into structured storyboard with:
    - StyleGuide stored in scene.style_guide
    - Shot records created in database
    - Scene status updated to "keyframing"

    Supports gap-filling mode (VGED-06): if shots already exist (draft scenes),
    only generates text for shots with empty shot_description. User-provided
    shot text is preserved and passed as context to the LLM for narrative continuity.

    Implements retry logic per STOR-05: up to 3 attempts with temperature
    reduction on JSON parse failures.

    Args:
        session: AsyncSession for database operations
        scene: Scene instance with prompt to transform
        text_adapter: Optional LLMAdapter. If None, one is created from scene.text_model.

    Raises:
        json.JSONDecodeError: If JSON parsing fails after retries
        ValidationError: If Pydantic validation fails after retries
    """
    from sqlalchemy import select as sa_select

    model_id = scene.text_model or settings.models.storyboard_llm
    adapter = text_adapter or get_adapter(model_id)

    style_label = scene.style.replace("_", " ")

    # ----------------------------------------------------------------
    # Gap-filling: Check for existing shots (draft scenes, VGED-06)
    # ----------------------------------------------------------------
    existing_shots_result = await session.execute(
        sa_select(Shot)
        .where(Shot.scene_id == scene.id)
        .order_by(Shot.shot_index)
    )
    existing_shots = list(existing_shots_result.scalars().all())

    # Partition into complete (all key fields filled) and incomplete (need generation)
    def _shot_is_complete(s):
        return (bool(s.shot_description and s.shot_description.strip())
                and bool(s.start_frame_prompt and s.start_frame_prompt.strip()))

    filled_shots = [s for s in existing_shots if _shot_is_complete(s)]
    empty_shots = [s for s in existing_shots if not _shot_is_complete(s)]

    if existing_shots and not empty_shots:
        # All shots already have text — skip storyboard generation entirely
        logger.info(
            "Scene %s: all %d shots have text, skipping storyboard generation",
            scene.id, len(existing_shots),
        )
        scene.status = "keyframing"
        await session.commit()
        return

    # Determine if production-bible-aware mode
    use_manifests = scene.production_bible_id is not None

    binding_registry_used = False
    legacy_assets: list[Any] = []
    if use_manifests:
        # Try binding registry path first (Phase 23 -- @tag syntax)
        binding_block = await format_binding_registry(session, scene.production_bible_id)
        if binding_block:
            asset_registry_block = binding_block
            binding_registry_used = True
            from vidpipe.db.models import CastBinding as _CB
            from vidpipe.db.models import PropBinding as _PB
            from vidpipe.db.models import SetBinding as _SB
            _cast_res = await session.execute(
                sa_select(_CB.tag).where(_CB.production_bible_id == scene.production_bible_id)
            )
            _set_res = await session.execute(
                sa_select(_SB.tag).where(_SB.production_bible_id == scene.production_bible_id)
            )
            _prop_res = await session.execute(
                sa_select(_PB.tag).where(_PB.production_bible_id == scene.production_bible_id)
            )
            asset_tags_set = {
                *(t for (t,) in _cast_res.all()),
                *(t for (t,) in _set_res.all()),
                *(t for (t,) in _prop_res.all()),
            }
            logger.info(
                "Scene %s: binding-aware storyboard with binding registry",
                scene.id,
            )
        else:
            # Fallback to legacy manifest asset path
            legacy_assets = await load_manifest_assets(session, scene.production_bible_id)
            asset_registry_block = format_asset_registry(legacy_assets)
            asset_tags_set = {a.manifest_tag for a in legacy_assets}
            logger.info(
                "Scene %s: manifest-aware storyboard with %d assets (legacy path)",
                scene.id, len(legacy_assets),
            )
    else:
        asset_registry_block = ""
        asset_tags_set = set()

    # Build context block for filled shots (gap-filling narrative continuity)
    filled_context = ""
    if filled_shots:
        context_lines = []
        for s in filled_shots:
            context_lines.append(
                f"Shot {s.shot_index} (provided by user): {s.shot_description}"
            )
        # Include partial shots (have description but missing prompts) as context
        partial_with_desc = [s for s in empty_shots
                             if s.shot_description and s.shot_description.strip()]
        for s in partial_with_desc:
            context_lines.append(
                f"Shot {s.shot_index} (user-provided description — keep this description, "
                f"generate start_frame_prompt/end_frame_prompt/video_motion_prompt/transition_notes): "
                f"{s.shot_description}"
            )
        empty_indices = [s.shot_index for s in empty_shots]
        filled_context = (
            "\n\nEXISTING SHOTS (provided by user — do NOT regenerate these):\n"
            + "\n".join(context_lines)
            + f"\n\nGenerate content ONLY for the following shot indices: {empty_indices}. "
            "Maintain narrative continuity with the provided shots."
        )

    # Build system prompt with style, aspect ratio, and shot count
    is_dynamic = getattr(scene, "dynamic_shot_count", False)
    if use_manifests:
        system_prompt = ENHANCED_STORYBOARD_PROMPT.format(
            style=style_label,
            aspect_ratio=scene.aspect_ratio,
            asset_registry_block=asset_registry_block,
        )
    else:
        system_prompt = STORYBOARD_SYSTEM_PROMPT.format(
            style=style_label,
            aspect_ratio=scene.aspect_ratio,
        )

    if not is_dynamic:
        # Fixed shot count: replace the default range with an exact count
        system_prompt = system_prompt.replace(
            "- Break the script into 3-5 distinct visual shots",
            f"- Break the script into exactly {scene.target_shot_count} distinct visual shots",
        )
    else:
        system_prompt = system_prompt.replace(
            "- Break the script into 3-5 distinct visual shots",
            "- Break the script into the number of distinct visual shots needed to clearly cover every beat; do not collapse a multi-beat scene into a single continuous shot.",
        )

    # Phase 18: Screenplay context enrichment (additive-only, SCRN-13/SCRN-14)
    screenplay_enrichment = ""
    if hasattr(scene, 'screenplay_context') and scene.screenplay_context:
        breakdown = scene.screenplay_context
        parts = []
        parts.append("SCREENPLAY DIRECTION:")
        if breakdown.get("slugline"):
            parts.append(f"  Scene: {breakdown['slugline']}")
        if breakdown.get("intent"):
            parts.append(f"  Intent: {breakdown['intent']}")
        if breakdown.get("emotional_beat"):
            parts.append(f"  Emotional beat: {breakdown['emotional_beat']}")
        if breakdown.get("story_state_in"):
            parts.append(f"  Story state (entering): {breakdown['story_state_in']}")
        if breakdown.get("story_state_out"):
            parts.append(f"  Story state (leaving): {breakdown['story_state_out']}")
        if breakdown.get("characters_present"):
            chars = ", ".join(breakdown["characters_present"])
            parts.append(f"  Characters in scene: {chars}")
        if breakdown.get("set_ref"):
            parts.append(f"  Location/Set: {breakdown['set_ref']}")
        if breakdown.get("props_required"):
            props = ", ".join(breakdown["props_required"])
            parts.append(f"  Props: {props}")
        screenplay_enrichment = "\n".join(parts) + "\n\n"

    # Phase 22/23: Resolve [CHAR:TAG], [SET:TAG], [PROP:TAG] and @tag in scene prompt
    # before sending to LLM. Original prompt preserved in DB, resolved text
    # used only for generation. Skips resolution if no bible_id or no tags.
    resolved_prompt_text = scene.prompt
    if scene.production_bible_id:
        from vidpipe.services.tag_resolver import has_any_tags, resolve_tags
        if has_any_tags(scene.prompt):
            resolved = await resolve_tags(scene.prompt, scene.production_bible_id, session)
            resolved_prompt_text = resolved.text
            if resolved.unresolved_tags:
                logger.warning(
                    "Scene %s: unresolved tags in prompt: %s",
                    scene.id, resolved.unresolved_tags,
                )
            else:
                logger.info(
                    "Scene %s: resolved tags in prompt for generation",
                    scene.id,
                )

    # Set generation_status on empty shots FIRST so the UI shows progress immediately
    for s in empty_shots:
        s.generation_status = "generating_text"
    if empty_shots:
        await session.commit()

    from vidpipe.services.event_bus import event_bus
    start_message = (
        "Starting storyboard generation with automatic shot count selection"
        if is_dynamic
        else f"Starting storyboard generation for {scene.target_shot_count} shot(s)"
    )
    event_bus.emit(
        scene.id,
        "phase_started",
        phase="storyboard",
        total_shots=scene.target_shot_count,
        message=start_message,
    )

    # ── Screenwriter Agent: Script Analysis + Shot Breakdown ──────────
    # Runs 2 LLM calls BEFORE the storyboard call to produce explicit
    # characters_present[] per shot as hard constraints.
    screenwriter_breakdown = None  # ShotBreakdown | None
    if use_manifests or not existing_shots:
        try:
            from vidpipe.services.screenwriter_agent import ScreenwriterAgentService
            agent = ScreenwriterAgentService(
                adapter,
                scene_id=scene.id,
                model_label=scene.text_model,
            )

            # Step 1: Analyze script
            analysis = await agent.analyze_script(
                scene_prompt=scene.prompt,
                binding_registry_block=asset_registry_block,
            )
            scene.script_analysis = analysis.model_dump()
            await session.commit()

            # Step 2: Break into shots
            filled_context_for_agent = ""
            if filled_shots:
                lines = []
                for s in filled_shots:
                    lines.append(f"Shot {s.shot_index}: {s.shot_description}")
                filled_context_for_agent = (
                    "EXISTING SHOTS (preserve these, assign characters to them):\n"
                    + "\n".join(lines)
                )

            screenwriter_breakdown = await agent.break_into_shots(
                scene_prompt=scene.prompt,
                analysis=analysis,
                target_shot_count=scene.target_shot_count,
                dynamic_shot_count=is_dynamic,
                binding_registry_block=asset_registry_block,
                existing_shot_context=filled_context_for_agent,
            )
            logger.info(
                "Scene %s: screenwriter agent produced %d shot assignments",
                scene.id, len(screenwriter_breakdown.shots),
            )
        except Exception as e:
            logger.warning("Screenwriter agent failed (non-fatal, continuing): %s", e)

    if use_manifests and screenwriter_breakdown:
        await _generate_manifest_storyboard_parallel(
            session,
            scene,
            adapter,
            existing_shots=existing_shots,
            filled_shots=filled_shots,
            empty_shots=empty_shots,
            asset_registry_block=asset_registry_block,
            asset_tags_set=asset_tags_set,
            style_label=style_label,
            screenplay_enrichment=screenplay_enrichment,
            resolved_prompt_text=resolved_prompt_text,
            screenwriter_breakdown=screenwriter_breakdown,
            is_dynamic=is_dynamic,
            using_bindings=binding_registry_used,
            legacy_assets=legacy_assets,
        )
        return

    # Inject shot breakdown as constraint into storyboard prompt
    shot_constraints = ""
    if screenwriter_breakdown:
        constraint_lines = ["\nSHOT BREAKDOWN (follow these character assignments exactly):"]
        for sa in screenwriter_breakdown.shots:
            chars = (
                ", ".join(_display_at_tag(c) for c in sa.characters_present)
                if sa.characters_present
                else "(none — scenery only)"
            )
            constraint_lines.append(
                f"  Shot {sa.shot_index}: {sa.narrative_intent}. Characters visible: {chars}"
            )
        constraint_lines.append(
            "\nIMPORTANT: Each shot's placements MUST include ALL characters listed above "
            "and MUST NOT include characters NOT listed. Scenery shots must have NO character placements."
        )
        shot_constraints = "\n".join(constraint_lines)

    full_prompt = f"{system_prompt}{filled_context}\n\n{screenplay_enrichment}{shot_constraints}\n\nScript: {resolved_prompt_text}"

    # Retry strategy: up to 3 attempts, reduce temperature on each retry
    attempt = 0
    max_attempts = 3
    base_temperature = 0.7

    @retry(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type((json.JSONDecodeError, ValidationError))
    )
    async def generate_with_retry():
        nonlocal attempt
        # Reduce temperature by 0.15 on each retry
        temperature = base_temperature - (attempt * 0.15)
        attempt += 1

        # Determine response schema based on manifest mode
        response_schema = EnhancedStoryboardOutput if use_manifests else StoryboardOutput

        # Call LLM adapter with structured output constraint
        # max_retries=1 so temperature reduction (outer retry) works correctly
        emit_task_log(
            scene.id,
            phase="storyboard",
            kind="prompt",
            source="storyboard.full.prompt",
            summary=f"Storyboard prompt attempt {attempt} sent",
            detail=_format_llm_detail(
                model_label=scene.text_model,
                schema_name=response_schema.__name__,
                content=full_prompt,
            ),
        )
        storyboard = await adapter.generate_text(
            prompt=full_prompt,
            schema=response_schema,
            temperature=max(0.0, temperature),
            max_retries=1,
        )
        emit_task_log(
            scene.id,
            phase="storyboard",
            level="success",
            kind="response",
            source="storyboard.full.response",
            summary=f"Storyboard response received on attempt {attempt}",
            detail=_format_llm_detail(
                model_label=scene.text_model,
                schema_name=response_schema.__name__,
                content=storyboard.model_dump(),
            ),
        )
        return storyboard

    # Execute with retry logic
    storyboard = await generate_with_retry()

    # Update scene with storyboard data
    scene.style_guide = storyboard.style_guide.model_dump()
    scene.storyboard_raw = storyboard.model_dump()

    # Dynamic shot count: sync target_shot_count to what the LLM actually produced
    if is_dynamic and storyboard.shots:
        scene.target_shot_count = len(storyboard.shots)
        scene.total_duration = scene.target_shot_count * (scene.target_clip_duration or 6)
        logger.info(
            f"Scene {scene.id}: dynamic shot count → {scene.target_shot_count} shots"
        )

    # ----------------------------------------------------------------
    # Persist shots: update existing rows (draft) or create new ones
    # ----------------------------------------------------------------
    if existing_shots:
        # Draft scene: Shot rows already exist — update only empty shots
        existing_by_index = {s.shot_index: s for s in existing_shots}
        for shot_data in storyboard.shots:
            sanitized_fields = _sanitized_prompt_fields(
                start_frame_prompt=shot_data.start_frame_prompt,
                end_frame_prompt=shot_data.end_frame_prompt,
                video_motion_prompt=shot_data.video_motion_prompt,
                transition_notes=shot_data.transition_notes,
            )
            existing = existing_by_index.get(shot_data.shot_index)
            if existing:
                # Per-field update: only fill in empty fields, preserve user-provided text
                for attr in ("shot_description", "start_frame_prompt", "end_frame_prompt",
                             "video_motion_prompt", "transition_notes"):
                    current = getattr(existing, attr)
                    if not current or not current.strip():
                        if attr == "shot_description":
                            setattr(existing, attr, getattr(shot_data, attr))
                        else:
                            setattr(existing, attr, sanitized_fields[attr])
                existing.status = "pending"
                existing.generation_status = None  # Clear generation_status
            else:
                # Shot index from LLM not in existing rows — create new
                shot = Shot(
                    scene_id=scene.id,
                    shot_index=shot_data.shot_index,
                    shot_description=shot_data.shot_description,
                    start_frame_prompt=sanitized_fields["start_frame_prompt"],
                    end_frame_prompt=sanitized_fields["end_frame_prompt"],
                    video_motion_prompt=sanitized_fields["video_motion_prompt"],
                    transition_notes=sanitized_fields["transition_notes"],
                    status="pending",
                )
                session.add(shot)
        # Clear generation_status on filled shots too
        for s in filled_shots:
            s.generation_status = None
        # Safety net: clear generation_status on any empty shot the LLM
        # didn't include in its output (prevents stuck "Generating..." UI)
        processed_indices = {sd.shot_index for sd in storyboard.shots}
        for s in empty_shots:
            if s.shot_index not in processed_indices and s.generation_status == "generating_text":
                s.generation_status = None
    else:
        # Non-draft scene: no existing shots — create all from scratch
        for shot_data in storyboard.shots:
            sanitized_fields = _sanitized_prompt_fields(
                start_frame_prompt=shot_data.start_frame_prompt,
                end_frame_prompt=shot_data.end_frame_prompt,
                video_motion_prompt=shot_data.video_motion_prompt,
                transition_notes=shot_data.transition_notes,
            )
            shot = Shot(
                scene_id=scene.id,
                shot_index=shot_data.shot_index,
                shot_description=shot_data.shot_description,
                start_frame_prompt=sanitized_fields["start_frame_prompt"],
                end_frame_prompt=sanitized_fields["end_frame_prompt"],
                video_motion_prompt=sanitized_fields["video_motion_prompt"],
                transition_notes=sanitized_fields["transition_notes"],
                status="pending",
            )
            session.add(shot)

    # Persist shot and audio manifests if manifest-aware mode
    if use_manifests:
        # Clean up existing manifests from prior runs to avoid unique constraint violations
        from sqlalchemy import delete as sa_delete
        await session.execute(
            sa_delete(ShotManifestModel).where(ShotManifestModel.scene_id == scene.id)
        )
        await session.execute(
            sa_delete(ShotAudioManifestModel).where(ShotAudioManifestModel.scene_id == scene.id)
        )

        # Pre-compute ordered list of manifest CHARACTER tags for remapping
        manifest_characters = sorted(
            [tag for tag in asset_tags_set if tag.startswith("CHAR_")]
        )

        for shot_data in storyboard.shots:
            # Convert shot manifest to mutable dict for potential remapping
            manifest_dict = shot_data.shot_manifest.model_dump()

            # Fix 2: Deterministic remap of unrecognized CHARACTER tags
            # before persisting — catches LLM mistakes without another LLM call
            _remap_unrecognized_tags(manifest_dict, asset_tags_set, manifest_characters)

            # Also remap audio dialogue speaker_tags that reference bad CHAR tags
            audio = shot_data.audio_manifest
            audio_dialogue = [d.model_dump() for d in audio.dialogue_lines]
            for dialogue_entry in audio_dialogue:
                speaker = dialogue_entry.get("speaker_tag", "")
                if speaker.startswith("CHAR_") and speaker not in asset_tags_set:
                    # Find what this tag was remapped to in the shot manifest
                    for placement in manifest_dict.get("placements", []):
                        if placement.get("asset_tag", "").startswith("CHAR_"):
                            # Use the first valid CHARACTER tag as fallback
                            if placement["asset_tag"] in asset_tags_set:
                                logger.info(
                                    "Audio tag remap: speaker_tag %s -> %s",
                                    speaker, placement["asset_tag"],
                                )
                                dialogue_entry["speaker_tag"] = placement["asset_tag"]
                                break

            # Post-validate: log any remaining unrecognized tags (non-CHAR, or unmapped)
            for placement_d in manifest_dict.get("placements", []):
                tag = placement_d.get("asset_tag", "")
                if tag not in asset_tags_set:
                    logger.warning(
                        "Scene %s shot %d: unrecognized asset tag '%s' "
                        "(not in registry, may be declared as new asset)",
                        scene.id, shot_data.shot_index, tag
                    )

            # Persist shot manifest (using remapped dict)
            shot_manifest = ShotManifestModel(
                scene_id=scene.id,
                shot_index=shot_data.shot_index,
                manifest_json=manifest_dict,
                composition_shot_type=shot_data.shot_manifest.composition.shot_type,
                composition_camera_movement=shot_data.shot_manifest.composition.camera_movement,
                asset_tags=[p.get("asset_tag", "") for p in manifest_dict.get("placements", [])],
                new_asset_count=len(manifest_dict.get("new_asset_declarations") or []),
            )
            session.add(shot_manifest)

            # Persist audio manifest (with remapped speaker_tags)
            audio_manifest = ShotAudioManifestModel(
                scene_id=scene.id,
                shot_index=shot_data.shot_index,
                dialogue_json=audio_dialogue,
                sfx_json=[s.model_dump() for s in audio.sfx],
                ambient_json=audio.ambient.model_dump() if audio.ambient else None,
                music_json=audio.music.model_dump() if audio.music else None,
                audio_continuity_json=audio.audio_continuity.model_dump() if audio.audio_continuity else None,
                speaker_tags=[d.get("speaker_tag", "") for d in audio_dialogue],
                has_dialogue=len(audio_dialogue) > 0,
                has_music=audio.music is not None,
            )
            session.add(audio_manifest)

        logger.info(
            "Scene %s: persisted %d shot manifests and audio manifests",
            scene.id, len(storyboard.shots)
        )

    # Populate screenwriter agent metadata on shots (characters_present, etc.)
    if screenwriter_breakdown:
        breakdown_by_index = {sa.shot_index: sa for sa in screenwriter_breakdown.shots}
        all_shots_result = await session.execute(
            sa_select(Shot).where(Shot.scene_id == scene.id).order_by(Shot.shot_index)
        )
        for shot_row in all_shots_result.scalars():
            sa = breakdown_by_index.get(shot_row.shot_index)
            if sa:
                shot_row.characters_present = sa.characters_present
                shot_row.beat_index = sa.beat_index
                shot_row.narrative_intent = sa.narrative_intent
                shot_row.emotional_weight = sa.emotional_weight

    # Update scene status to indicate storyboard completion
    scene.status = "keyframing"

    # Commit all changes BEFORE emitting events so that frontend refreshes
    # see the committed data (generation_status cleared, text fields populated).
    await session.commit()

    # Now emit events — any frontend refresh will see the committed state
    for shot_data in storyboard.shots:
        event_bus.emit(
            scene.id,
            "shot_text_ready",
            shot_index=shot_data.shot_index,
            message=f"Shot {shot_data.shot_index + 1} storyboard text ready",
        )
    event_bus.emit(
        scene.id,
        "phase_completed",
        phase="storyboard",
        message="Storyboard generation complete",
    )
    event_bus.emit(scene.id, "refresh")
