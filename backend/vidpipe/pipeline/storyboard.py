"""Storyboard generation using Gemini structured output.

Transforms user text prompts into scene-by-scene breakdowns with:
- Keyframe image prompts (start/end)
- Motion descriptions for video interpolation
- Cross-scene style guide for visual consistency

Spec reference: STOR-01 through STOR-05
"""

import json
import logging
from typing import Optional
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from vidpipe.config import settings
from vidpipe.db.models import Project, Scene
from vidpipe.db.models import SceneManifest as SceneManifestModel
from vidpipe.db.models import SceneAudioManifest as SceneAudioManifestModel
from vidpipe.schemas.storyboard import StoryboardOutput
from vidpipe.schemas.storyboard_enhanced import EnhancedStoryboardOutput
from vidpipe.services.llm import get_adapter, LLMAdapter
from vidpipe.services.manifest_service import load_manifest_assets, format_asset_registry

logger = logging.getLogger(__name__)


# System prompt template for Gemini storyboard generation.
# {style} and {aspect_ratio} are filled at runtime.
STORYBOARD_SYSTEM_PROMPT = """You are a storyboard director specializing in short-form video content.

Your task is to transform the user's script into a visual storyboard optimized for AI-powered video generation.

VISUAL STYLE: {style}
This is the MANDATORY visual style. Do NOT fall back to photorealistic or cinematic unless that is the explicitly requested style.

ASPECT RATIO: {aspect_ratio}
Compose all keyframe image prompts for this aspect ratio. For 16:9, favor wide establishing shots and horizontal compositions. For 9:16, favor close-ups, vertical framing, and portrait-oriented compositions.

REQUIREMENTS:
- Break the script into 3-5 distinct visual scenes
- Each scene should have clear narrative progression
- Describe all characters that appear in the video with consistent physical details (see characters field)
- Provide detailed image prompts for start and end keyframes
- Include motion descriptions for video interpolation between keyframes
- Add transition notes to ensure visual continuity between scenes
- Create a comprehensive style guide aligned with {style}

DETAIL PRESERVATION:
Before composing scenes, identify ALL specific details from the script:
- Proper names (people, organizations, products, frameworks, standards)
- Technical terms, acronyms, and domain jargon
- Numbers, dates, statistics, and quantitative claims
- Specific processes, methodologies, or workflows described

Every identified detail MUST appear in at least one scene via visual vehicles:
- On-screen text overlays, titles, or captions
- Documents, reports, or slides visible in the scene
- Whiteboards, screens, monitors showing text
- Signage, nameplates, logos, or banners

The key_details field for each scene must list 3-6 specific terms the scene conveys.

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

GOAL: Ensure all scenes maintain visual coherence in {style} style while telling a compelling story. Preserve the original script's specific terminology, names, and details — do not reduce domain-specific content to generic visual metaphors."""


# Enhanced system prompt for manifest-aware storyboarding
ENHANCED_STORYBOARD_PROMPT = """You are a storyboard director specializing in short-form video content.

Your task is to transform the user's script into a visual storyboard optimized for AI-powered video generation.

VISUAL STYLE: {style}
This is the MANDATORY visual style. Do NOT fall back to photorealistic or cinematic unless that is the explicitly requested style.

ASPECT RATIO: {aspect_ratio}
Compose all keyframe image prompts for this aspect ratio. For 16:9, favor wide establishing shots and horizontal compositions. For 9:16, favor close-ups, vertical framing, and portrait-oriented compositions.

AVAILABLE ASSETS (from Asset Registry):
{asset_registry_block}

SCENE MANIFEST INSTRUCTIONS:
When creating scenes, generate a scene_manifest for each scene:
- Reference registered assets by their [TAG] (e.g., [CHAR_01], [ENV_02])
- Use the asset's reverse_prompt for visual detail — it's already optimized for generation
- Assign roles: subject, background, prop, interaction_target, environment
- Specify spatial positions and actions for each placed asset
- Include composition metadata: shot_type, camera_movement, focal_point
- Add continuity_notes describing visual continuity with previous scenes
- You MAY declare new_asset_declarations for assets NOT in the registry — describe them textually with name, type, and description

AUDIO MANIFEST INSTRUCTIONS:
For each scene, generate an audio_manifest with:
- dialogue_lines: Map speech to character tags (speaker_tag must be a registered [TAG] like CHAR_01). Include delivery notes (muttered, shouted, whispered) and timing (start, mid-scene, end)
- sfx: Sound effects with trigger descriptions and relative timing. Use "SFX:" mental model
- ambient: Base layer soundscape + environmental context. Describe what you HEAR, not see
- music: Style, mood, tempo, instruments, and transition cues (fade in, cut, swell)
- audio_continuity: What carries from previous scene, what's new, what cuts

REQUIREMENTS:
- Break the script into 3-5 distinct visual scenes
- Each scene should have clear narrative progression
- Describe all characters that appear in the video with consistent physical details (see characters field)
- Provide detailed image prompts for start and end keyframes
- Include motion descriptions for video interpolation between keyframes
- Add transition notes to ensure visual continuity between scenes
- Create a comprehensive style guide aligned with {style}
- Each scene MUST include scene_manifest with at least one asset placement
- Each scene MUST include audio_manifest (even if minimal — at least ambient)
- Asset tag references MUST match tags from the Available Assets list above
- Character wardrobe notes in placements MUST be consistent with character descriptions

DETAIL PRESERVATION:
Before composing scenes, identify ALL specific details from the script:
- Proper names (people, organizations, products, frameworks, standards)
- Technical terms, acronyms, and domain jargon
- Numbers, dates, statistics, and quantitative claims
- Specific processes, methodologies, or workflows described

Every identified detail MUST appear in at least one scene via visual vehicles:
- On-screen text overlays, titles, or captions
- Documents, reports, or slides visible in the scene
- Whiteboards, screens, monitors showing text
- Signage, nameplates, logos, or banners

The key_details field for each scene must list 3-6 specific terms the scene conveys.

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

GOAL: Ensure all scenes maintain visual coherence in {style} style while telling a compelling story. Preserve the original script's specific terminology, names, and details — do not reduce domain-specific content to generic visual metaphors."""


async def generate_storyboard(
    session: AsyncSession,
    project: Project,
    text_adapter: Optional[LLMAdapter] = None,
) -> None:
    """Generate storyboard from project prompt using LLM structured output.

    Transforms project.prompt into structured storyboard with:
    - StyleGuide stored in project.style_guide
    - Scene records created in database
    - Project status updated to "keyframing"

    Implements retry logic per STOR-05: up to 3 attempts with temperature
    reduction on JSON parse failures.

    Args:
        session: AsyncSession for database operations
        project: Project instance with prompt to transform
        text_adapter: Optional LLMAdapter. If None, one is created from project.text_model.

    Raises:
        json.JSONDecodeError: If JSON parsing fails after retries
        ValidationError: If Pydantic validation fails after retries
    """
    model_id = project.text_model or settings.models.storyboard_llm
    adapter = text_adapter or get_adapter(model_id)

    style_label = project.style.replace("_", " ")

    # Determine if manifest-aware mode
    use_manifests = project.manifest_id is not None

    if use_manifests:
        # Load asset registry for LLM context
        assets = await load_manifest_assets(session, project.manifest_id)
        asset_registry_block = format_asset_registry(assets)
        asset_tags_set = {a.manifest_tag for a in assets}
        logger.info(
            "Project %s: manifest-aware storyboard with %d assets",
            project.id, len(assets)
        )
    else:
        asset_registry_block = ""
        asset_tags_set = set()

    # Build system prompt with style, aspect ratio, and scene count
    if use_manifests:
        system_prompt = ENHANCED_STORYBOARD_PROMPT.format(
            style=style_label,
            aspect_ratio=project.aspect_ratio,
            asset_registry_block=asset_registry_block,
        ).replace(
            "- Break the script into 3-5 distinct visual scenes",
            f"- Break the script into exactly {project.target_scene_count} distinct visual scenes",
        )
    else:
        system_prompt = STORYBOARD_SYSTEM_PROMPT.format(
            style=style_label,
            aspect_ratio=project.aspect_ratio,
        ).replace(
            "- Break the script into 3-5 distinct visual scenes",
            f"- Break the script into exactly {project.target_scene_count} distinct visual scenes",
        )

    full_prompt = f"{system_prompt}\n\nScript: {project.prompt}"

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

    # Update project with storyboard data
    project.style_guide = storyboard.style_guide.model_dump()
    project.storyboard_raw = storyboard.model_dump()

    # Create Scene records from storyboard
    for scene_data in storyboard.scenes:
        scene = Scene(
            project_id=project.id,
            scene_index=scene_data.scene_index,
            scene_description=scene_data.scene_description,
            start_frame_prompt=scene_data.start_frame_prompt,
            end_frame_prompt=scene_data.end_frame_prompt,
            video_motion_prompt=scene_data.video_motion_prompt,
            transition_notes=scene_data.transition_notes,
            status="pending"
        )
        session.add(scene)

    # Persist scene and audio manifests if manifest-aware mode
    if use_manifests:
        for scene_data in storyboard.scenes:
            # Post-validate asset tags
            for placement in scene_data.scene_manifest.placements:
                if placement.asset_tag not in asset_tags_set:
                    logger.warning(
                        "Project %s scene %d: unrecognized asset tag '%s' "
                        "(not in registry, may be declared as new asset)",
                        project.id, scene_data.scene_index, placement.asset_tag
                    )

            # Persist scene manifest
            scene_manifest = SceneManifestModel(
                project_id=project.id,
                scene_index=scene_data.scene_index,
                manifest_json=scene_data.scene_manifest.model_dump(),
                composition_shot_type=scene_data.scene_manifest.composition.shot_type,
                composition_camera_movement=scene_data.scene_manifest.composition.camera_movement,
                asset_tags=[p.asset_tag for p in scene_data.scene_manifest.placements],
                new_asset_count=len(scene_data.scene_manifest.new_asset_declarations or []),
            )
            session.add(scene_manifest)

            # Persist audio manifest
            audio = scene_data.audio_manifest
            audio_manifest = SceneAudioManifestModel(
                project_id=project.id,
                scene_index=scene_data.scene_index,
                dialogue_json=[d.model_dump() for d in audio.dialogue_lines],
                sfx_json=[s.model_dump() for s in audio.sfx],
                ambient_json=audio.ambient.model_dump() if audio.ambient else None,
                music_json=audio.music.model_dump() if audio.music else None,
                audio_continuity_json=audio.audio_continuity.model_dump() if audio.audio_continuity else None,
                speaker_tags=[d.speaker_tag for d in audio.dialogue_lines],
                has_dialogue=len(audio.dialogue_lines) > 0,
                has_music=audio.music is not None,
            )
            session.add(audio_manifest)

        logger.info(
            "Project %s: persisted %d scene manifests and audio manifests",
            project.id, len(storyboard.scenes)
        )

    # Update project status to indicate storyboard completion
    project.status = "keyframing"

    # Commit all changes
    await session.commit()
