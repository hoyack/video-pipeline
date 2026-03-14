"""Storyboard generation using Gemini structured output.

Transforms user text prompts into shot-by-shot breakdowns with:
- Keyframe image prompts (start/end)
- Motion descriptions for video interpolation
- Cross-shot style guide for visual consistency

Spec reference: STOR-01 through STOR-05
"""

import json
import logging
from typing import Optional
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from vidpipe.config import settings
from vidpipe.db.models import Scene, Shot
from vidpipe.db.models import ShotManifest as ShotManifestModel
from vidpipe.db.models import ShotAudioManifest as ShotAudioManifestModel
from vidpipe.schemas.storyboard import StoryboardOutput
from vidpipe.schemas.storyboard_enhanced import EnhancedStoryboardOutput
from vidpipe.services.llm import get_adapter, LLMAdapter
from vidpipe.services.manifest_service import load_manifest_assets, format_asset_registry, format_binding_registry

logger = logging.getLogger(__name__)


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
6. CAMERA: Shot type (wide/medium/close-up), angle, lens
7. STYLE CUES: Rendering technique details specific to {style} (e.g., line weight, color fills, shading approach, texture)
8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic

VIDEO MOTION PROMPT FORMAT (video_motion_prompt):
Describe ONLY motion and camera movement. Do NOT re-describe characters, setting, or style.
The keyframe images already provide the visual context — the motion prompt controls what moves.
Focus on: camera movement (pan, dolly, track, crane), subject animation, environmental animation.
Good example: "Slow dolly forward as the subject turns to face the camera, hair gently blowing in the breeze"
Bad example: "A blonde woman in anime style turns around in a congressional hearing room" (re-describes visuals)

GOAL: Ensure all shots maintain visual coherence in {style} style while telling a compelling story. Preserve the original script's specific terminology, names, and details — do not reduce domain-specific content to generic visual metaphors."""


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
6. CAMERA: Shot type (wide/medium/close-up), angle, lens
7. STYLE CUES: Rendering technique details specific to {style} (e.g., line weight, color fills, shading approach, texture)
8. COLOR PALETTE: Dominant colors that reinforce the {style} aesthetic

VIDEO MOTION PROMPT FORMAT (video_motion_prompt):
Describe ONLY motion and camera movement. Do NOT re-describe characters, setting, or style.
The keyframe images already provide the visual context — the motion prompt controls what moves.
Focus on: camera movement (pan, dolly, track, crane), subject animation, environmental animation.
Good example: "Slow dolly forward as the subject turns to face the camera, hair gently blowing in the breeze"
Bad example: "A blonde woman in anime style turns around in a congressional hearing room" (re-describes visuals)

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

    if use_manifests:
        # Try binding registry path first (Phase 23 -- @tag syntax)
        binding_block = await format_binding_registry(session, scene.production_bible_id)
        if binding_block:
            asset_registry_block = binding_block
            # For asset_tags_set, extract tags from bindings for post-LLM remapping
            from vidpipe.db.models import CastBinding as _CB
            _cast_res = await session.execute(
                sa_select(_CB.tag).where(_CB.production_bible_id == scene.production_bible_id)
            )
            asset_tags_set = {t.upper() for (t,) in _cast_res.all()}
            logger.info(
                "Scene %s: binding-aware storyboard with binding registry",
                scene.id,
            )
        else:
            # Fallback to legacy manifest asset path
            assets = await load_manifest_assets(session, scene.production_bible_id)
            asset_registry_block = format_asset_registry(assets)
            asset_tags_set = {a.manifest_tag for a in assets}
            logger.info(
                "Scene %s: manifest-aware storyboard with %d assets (legacy path)",
                scene.id, len(assets),
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
    if use_manifests:
        system_prompt = ENHANCED_STORYBOARD_PROMPT.format(
            style=style_label,
            aspect_ratio=scene.aspect_ratio,
            asset_registry_block=asset_registry_block,
        ).replace(
            "- Break the script into 3-5 distinct visual shots",
            f"- Break the script into exactly {scene.target_shot_count} distinct visual shots",
        )
    else:
        system_prompt = STORYBOARD_SYSTEM_PROMPT.format(
            style=style_label,
            aspect_ratio=scene.aspect_ratio,
        ).replace(
            "- Break the script into 3-5 distinct visual shots",
            f"- Break the script into exactly {scene.target_shot_count} distinct visual shots",
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

    full_prompt = f"{system_prompt}{filled_context}\n\n{screenplay_enrichment}Script: {resolved_prompt_text}"

    # Set generation_status on empty shots before generating
    for s in empty_shots:
        s.generation_status = "generating_text"
    if empty_shots:
        await session.commit()

    from vidpipe.services.event_bus import event_bus
    event_bus.emit(scene.id, "phase_started", phase="storyboard", total_shots=scene.target_shot_count)

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
        storyboard = await adapter.generate_text(
            prompt=full_prompt,
            schema=response_schema,
            temperature=max(0.0, temperature),
            max_retries=1,
        )
        return storyboard

    # Execute with retry logic
    storyboard = await generate_with_retry()

    # Update scene with storyboard data
    scene.style_guide = storyboard.style_guide.model_dump()
    scene.storyboard_raw = storyboard.model_dump()

    # ----------------------------------------------------------------
    # Persist shots: update existing rows (draft) or create new ones
    # ----------------------------------------------------------------
    if existing_shots:
        # Draft scene: Shot rows already exist — update only empty shots
        existing_by_index = {s.shot_index: s for s in existing_shots}
        for shot_data in storyboard.shots:
            existing = existing_by_index.get(shot_data.shot_index)
            if existing:
                # Per-field update: only fill in empty fields, preserve user-provided text
                for attr in ("shot_description", "start_frame_prompt", "end_frame_prompt",
                             "video_motion_prompt", "transition_notes"):
                    current = getattr(existing, attr)
                    if not current or not current.strip():
                        setattr(existing, attr, getattr(shot_data, attr))
                existing.status = "pending"
                existing.generation_status = None  # Clear generation_status
            else:
                # Shot index from LLM not in existing rows — create new
                shot = Shot(
                    scene_id=scene.id,
                    shot_index=shot_data.shot_index,
                    shot_description=shot_data.shot_description,
                    start_frame_prompt=shot_data.start_frame_prompt,
                    end_frame_prompt=shot_data.end_frame_prompt,
                    video_motion_prompt=shot_data.video_motion_prompt,
                    transition_notes=shot_data.transition_notes,
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
            shot = Shot(
                scene_id=scene.id,
                shot_index=shot_data.shot_index,
                shot_description=shot_data.shot_description,
                start_frame_prompt=shot_data.start_frame_prompt,
                end_frame_prompt=shot_data.end_frame_prompt,
                video_motion_prompt=shot_data.video_motion_prompt,
                transition_notes=shot_data.transition_notes,
                status="pending",
            )
            session.add(shot)

    # Persist shot and audio manifests if manifest-aware mode
    if use_manifests:
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

    # Update scene status to indicate storyboard completion
    scene.status = "keyframing"

    # Commit all changes BEFORE emitting events so that frontend refreshes
    # see the committed data (generation_status cleared, text fields populated).
    await session.commit()

    # Now emit events — any frontend refresh will see the committed state
    for shot_data in storyboard.shots:
        event_bus.emit(scene.id, "shot_text_ready", shot_index=shot_data.shot_index)
    event_bus.emit(scene.id, "phase_completed", phase="storyboard")
    event_bus.emit(scene.id, "refresh")
