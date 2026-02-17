"""PromptRewriterService — adaptive prompt rewriting via Gemini 2.5 Flash.

Assembles five structured inputs per scene and calls Gemini to produce
cinematography-formula prompts with LLM-reasoned reference selection.

Inputs assembled:
  1. Original storyboard prompt (scene.start_frame_prompt or scene.video_motion_prompt)
  2. Manifest metadata (shot_type, camera_movement, placements from scene_manifest_json)
  3. Asset reverse_prompts (from placed asset registry entries)
  4. Continuity patch (from previous scene's cv_analysis_json)
  5. Audio direction (from audio_manifest_json — video only)

Called from:
  - keyframes.py: rewrite_keyframe_prompt() before Imagen generation
  - video_gen.py: rewrite_video_prompt() before Veo submission

Spec reference: Phase 10 - Adaptive Prompt Rewriting
"""

import asyncio
import logging
from typing import Optional

from google.genai import types
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from vidpipe.db.models import Asset, Scene
from vidpipe.schemas.prompt_rewrite import RewrittenKeyframePromptOutput, RewrittenVideoPromptOutput
from vidpipe.services.vertex_client import get_vertex_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt constants
# ---------------------------------------------------------------------------

KEYFRAME_REWRITER_SYSTEM_PROMPT = """You are a professional cinematographer and VFX director.
You are assembling a static image generation prompt for a keyframe.

Your output must follow this EXACT formula:
[Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]

RULES:
1. Start with shot type and camera details from the scene manifest
2. Describe subjects using their EXACT reverse_prompt details (not what you imagine)
3. Include spatial positions from manifest placements (left/right/center/foreground/background)
4. Include wardrobe_note details for continuity
5. Apply any continuity corrections from the previous scene's CV analysis
6. Preserve the original prompt's narrative intent, but upgrade its visual specificity
7. Keep under 400 words (Imagen sweet spot for keyframes)
8. Select exactly 3 reference asset tags — explain why

WHAT NOT TO DO:
- Do not re-invent character descriptions (use reverse_prompt verbatim)
- Do not ignore continuity corrections
- Do not exceed 400 words
- Do not reference audio (keyframes are static images)
"""

VIDEO_REWRITER_SYSTEM_PROMPT = """You are a professional cinematographer and VFX director.
You are assembling a video generation prompt for Veo 3.1.

Your output must follow this EXACT formula:
[Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance] + [Audio Direction]

RULES:
1. Describe MOTION only — the reference images provide visual context
2. Camera movement from manifest (dolly, pan, static, etc.)
3. Character actions and expressions from manifest placements
4. Embed audio direction at the end:
   - Dialogue: Character name says "exact words" (delivery note)
   - SFX: brief description at timing
   - Ambient: base soundscape
   - Music: style, mood, transition
5. Apply continuity corrections from CV analysis of previous scene
6. Keep under 500 words (Veo 3.1 prompt sweet spot)
7. Select exactly 3 reference asset tags — explain why

CRITICAL:
- Include ALL audio direction from the manifest (Veo generates audio from this)
- Do not re-describe visual appearance (reference images handle that)
- Motion prompt should tell Veo what MOVES, not how things look
"""


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class PromptRewriterService:
    """Assembles final generation prompts by injecting manifest metadata,
    asset reverse_prompts, continuity corrections, and audio direction.

    Called once per scene for keyframe generation, and once per scene
    for video generation (separate calls with different formula).
    """

    # Rate limiting: 5 concurrent Gemini rewriter requests (matches Phase 5 pattern)
    _semaphore = asyncio.Semaphore(5)

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy-init Vertex AI client (singleton via get_vertex_client)."""
        if self._client is None:
            self._client = get_vertex_client()
        return self._client

    async def rewrite_keyframe_prompt(
        self,
        scene: Scene,
        scene_manifest_json: dict,
        placed_assets: list[Asset],
        previous_cv_analysis: Optional[dict],
        all_assets: list[Asset],
    ) -> RewrittenKeyframePromptOutput:
        """Rewrite keyframe prompt with manifest enrichment.

        Args:
            scene: Scene ORM row (provides start_frame_prompt, scene_index)
            scene_manifest_json: scene_manifest_row.manifest_json dict
            placed_assets: Asset instances for tags present in placements
            previous_cv_analysis: cv_analysis_json from scene N-1 (None if scene 0)
            all_assets: All assets in the manifest (for reference listing)

        Returns:
            RewrittenKeyframePromptOutput with rewritten_prompt, selected_reference_tags, etc.

        Raises:
            Exception: After 3 retries; caller should catch and fall back to original prompt
        """
        async with self._semaphore:
            system_prompt = KEYFRAME_REWRITER_SYSTEM_PROMPT
            user_context = self._assemble_keyframe_context(
                scene, scene_manifest_json, placed_assets, previous_cv_analysis, all_assets
            )
            return await self._call_rewriter(
                system_prompt, user_context, RewrittenKeyframePromptOutput
            )

    async def rewrite_video_prompt(
        self,
        scene: Scene,
        scene_manifest_json: dict,
        audio_manifest_json: Optional[dict],
        placed_assets: list[Asset],
        previous_cv_analysis: Optional[dict],
        all_assets: list[Asset],
    ) -> RewrittenVideoPromptOutput:
        """Rewrite video prompt with manifest enrichment and audio direction.

        Args:
            scene: Scene ORM row (provides video_motion_prompt, scene_index)
            scene_manifest_json: scene_manifest_row.manifest_json dict
            audio_manifest_json: dict with dialogue_lines, sfx, ambient, music (or None)
            placed_assets: Asset instances for tags present in placements
            previous_cv_analysis: cv_analysis_json from scene N-1 (None if scene 0)
            all_assets: All assets in the manifest (for reference listing)

        Returns:
            RewrittenVideoPromptOutput with rewritten_prompt, selected_reference_tags, etc.

        Raises:
            Exception: After 3 retries; caller should catch and fall back to original prompt
        """
        async with self._semaphore:
            system_prompt = VIDEO_REWRITER_SYSTEM_PROMPT
            user_context = self._assemble_video_context(
                scene, scene_manifest_json, audio_manifest_json,
                placed_assets, previous_cv_analysis, all_assets
            )
            return await self._call_rewriter(
                system_prompt, user_context, RewrittenVideoPromptOutput
            )

    async def _call_rewriter(self, system_prompt: str, user_context: str, schema):
        """Call Gemini 2.5 Flash with structured output schema.

        Retries up to 3 times on any exception. Temperature 0.4 (lower than
        storyboard's 0.7 — less creative, more precise reference injection).

        Args:
            system_prompt: Role/instructions string
            user_context: Assembled scene data block
            schema: Pydantic model class for response_schema

        Returns:
            Validated pydantic model instance

        Raises:
            Exception: After 3 retry attempts exhausted
        """
        @retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
        async def _attempt():
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[f"{system_prompt}\n\n{user_context}"],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.4,
                )
            )
            return schema.model_validate_json(response.text)

        return await _attempt()

    # -----------------------------------------------------------------------
    # Context assembly
    # -----------------------------------------------------------------------

    def _assemble_keyframe_context(
        self,
        scene: Scene,
        scene_manifest_json: dict,
        placed_assets: list[Asset],
        previous_cv_analysis: Optional[dict],
        all_assets: list[Asset],
    ) -> str:
        """Assemble the user context block for keyframe rewriting."""
        sections = [
            "=== ORIGINAL PROMPT ===",
            scene.start_frame_prompt,
            "",
            "=== SCENE COMPOSITION ===",
            _format_composition(scene_manifest_json),
            "",
            _format_placed_assets(scene_manifest_json, all_assets),
            "",
            _build_continuity_patch(previous_cv_analysis, scene.scene_index),
            "",
            _list_available_references(all_assets),
        ]
        return "\n".join(sections)

    def _assemble_video_context(
        self,
        scene: Scene,
        scene_manifest_json: dict,
        audio_manifest_json: Optional[dict],
        placed_assets: list[Asset],
        previous_cv_analysis: Optional[dict],
        all_assets: list[Asset],
    ) -> str:
        """Assemble the user context block for video rewriting."""
        sections = [
            "=== ORIGINAL PROMPT ===",
            scene.video_motion_prompt,
            "",
            "=== SCENE COMPOSITION ===",
            _format_composition(scene_manifest_json),
            "",
            _format_placed_assets(scene_manifest_json, all_assets),
            "",
            _build_continuity_patch(previous_cv_analysis, scene.scene_index),
            "",
            _format_audio_direction(audio_manifest_json),
            "",
            _list_available_references(all_assets),
        ]
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def _format_composition(scene_manifest_json: dict) -> str:
    """Format composition metadata from scene manifest."""
    composition = scene_manifest_json.get("composition", {})
    if not composition:
        return "No composition metadata available."

    shot_type = composition.get("shot_type", "unspecified")
    camera_movement = composition.get("camera_movement", "unspecified")
    focal_point = composition.get("focal_point", "")

    lines = [
        f"Shot type: {shot_type}",
        f"Camera movement: {camera_movement}",
    ]
    if focal_point:
        lines.append(f"Focal point: {focal_point}")

    continuity_notes = scene_manifest_json.get("continuity_notes", "")
    if continuity_notes:
        lines.append(f"Storyboard continuity notes: {continuity_notes}")

    return "\n".join(lines)


def _format_placed_assets(scene_manifest_json: dict, all_assets: list[Asset]) -> str:
    """Format placed asset descriptions for LLM rewriter context.

    Follows truncation rules from research Pattern 4:
    - reverse_prompt: truncated to 200 chars
    - visual_description: included only if quality >= 7.0, truncated to 150 chars
    """
    asset_map = {a.manifest_tag: a for a in all_assets}
    placements = scene_manifest_json.get("placements", [])

    lines = ["PLACED ASSETS IN THIS SCENE:", "=" * 40]

    if not placements:
        lines.append("No asset placements specified.")
        return "\n".join(lines)

    for placement in placements:
        tag = placement.get("asset_tag")
        asset = asset_map.get(tag)
        if not asset:
            lines.append(f"[{tag}] — asset not found in registry")
            lines.append("")
            continue

        quality_str = f"{asset.quality_score:.1f}/10" if asset.quality_score is not None else "N/A"
        role = placement.get("role", "unknown")
        position = placement.get("position", "")
        action = placement.get("action", "")
        wardrobe = placement.get("wardrobe_note", "")
        expression = placement.get("expression", "")

        lines.append(
            f'[{tag}] "{asset.name}" ({asset.asset_type}) — {role}'
            + (f" at {position}" if position else "")
            + f" (quality: {quality_str})"
        )
        if action:
            lines.append(f"  Action: {action}")
        if expression:
            lines.append(f"  Expression: {expression}")
        if wardrobe:
            lines.append(f"  Wardrobe: {wardrobe}")
        if asset.reverse_prompt:
            rp = asset.reverse_prompt[:200] + ("..." if len(asset.reverse_prompt) > 200 else "")
            lines.append(f"  Visual (reverse_prompt): {rp}")
        if asset.visual_description and asset.quality_score is not None and asset.quality_score >= 7.0:
            vd = asset.visual_description[:150] + ("..." if len(asset.visual_description) > 150 else "")
            lines.append(f"  Visual description: {vd}")
        lines.append("")

    return "\n".join(lines)


def _build_continuity_patch(previous_cv_analysis: Optional[dict], scene_index: int) -> str:
    """Build continuity correction block from previous scene's CV analysis.

    Guards scene_index == 0 (first scene has no previous scene CV data).

    Args:
        previous_cv_analysis: cv_analysis_json from scene N-1, or None
        scene_index: Current scene index (0-based)

    Returns:
        Formatted continuity block string for LLM context
    """
    if scene_index == 0 or previous_cv_analysis is None:
        return "CONTINUITY: This is the first scene — no previous scene continuity needed."

    semantic = previous_cv_analysis.get("semantic_analysis") or {}
    issues = semantic.get("continuity_issues") or []
    scene_desc = semantic.get("overall_scene_description", "")
    score = previous_cv_analysis.get("continuity_score", 0.0)

    lines = [
        f"CONTINUITY PATCH (from Scene {scene_index - 1} CV Analysis):",
        f"Previous scene continuity score: {score:.1f}/10",
    ]

    if scene_desc:
        lines.append(f"What scene {scene_index - 1} actually showed: {scene_desc}")

    if issues:
        lines.append("Issues flagged (MUST address in this scene):")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("No continuity issues flagged — maintain current appearance.")

    return "\n".join(lines)


def _format_audio_direction(audio_manifest_json: Optional[dict]) -> str:
    """Format audio manifest into prompt-injectable audio direction block.

    Formats all four audio tracks: dialogue, SFX, ambient, music.

    Args:
        audio_manifest_json: dict with dialogue_lines, sfx, ambient, music keys

    Returns:
        Formatted audio direction block string for LLM context
    """
    if not audio_manifest_json:
        return "AUDIO: No audio direction specified."

    lines = ["AUDIO DIRECTION:"]

    dialogue = audio_manifest_json.get("dialogue_lines") or []
    for d in dialogue:
        speaker = d.get("speaker_name") or d.get("speaker_tag", "Character")
        line = d.get("line", "")
        delivery = d.get("delivery", "")
        timing = d.get("timing", "")
        delivery_note = f" ({delivery})" if delivery else ""
        lines.append(f'  Dialogue ({timing}): {speaker} says{delivery_note}: "{line}"')

    sfx = audio_manifest_json.get("sfx") or []
    for s in sfx:
        effect = s.get("effect", "")
        timing = s.get("timing", "")
        volume = s.get("volume", "subtle")
        lines.append(f"  SFX ({timing}, {volume}): {effect}")

    ambient = audio_manifest_json.get("ambient") or {}
    if ambient:
        base = ambient.get("base_layer", "")
        env = ambient.get("environmental", "")
        if base:
            lines.append(f"  Ambient: {base}" + (f", {env}" if env else ""))

    music = audio_manifest_json.get("music") or {}
    if music:
        style = music.get("style", "")
        mood = music.get("mood", "")
        tempo = music.get("tempo", "")
        transition = music.get("transition", "")
        parts = [p for p in [style, mood, f"{tempo} tempo" if tempo else "", transition] if p]
        lines.append(f"  Music: {', '.join(parts)}" if parts else "  Music: (unspecified)")

    if len(lines) == 1:
        return "AUDIO: No audio direction specified."

    return "\n".join(lines)


def _list_available_references(all_assets: list[Asset]) -> str:
    """List all assets that have reference_image_url (available as Veo references).

    Only assets WITH reference images can be passed as Veo reference inputs.
    The LLM selects from this list for selected_reference_tags.

    Args:
        all_assets: All Asset instances in the manifest

    Returns:
        Formatted available references block for LLM context
    """
    lines = ["AVAILABLE REFERENCE IMAGES (select exactly 3 tags):"]

    with_images = [a for a in all_assets if a.reference_image_url]
    if not with_images:
        lines.append("No assets have reference images. Select from all assets instead.")
        for a in all_assets:
            face_note = " [face crop]" if a.is_face_crop else ""
            quality_str = f"{a.quality_score:.1f}" if a.quality_score is not None else "N/A"
            lines.append(f"  [{a.manifest_tag}] {a.name} ({a.asset_type}){face_note} — quality: {quality_str}")
        return "\n".join(lines)

    for asset in with_images:
        face_note = " [face crop]" if asset.is_face_crop else ""
        quality_str = f"{asset.quality_score:.1f}" if asset.quality_score is not None else "N/A"
        lines.append(
            f"  [{asset.manifest_tag}] {asset.name} ({asset.asset_type}){face_note}"
            f" — quality: {quality_str} — HAS REFERENCE IMAGE"
        )

    # Also list assets without images so LLM knows what's NOT available
    without_images = [a for a in all_assets if not a.reference_image_url]
    if without_images:
        lines.append("")
        lines.append("Assets WITHOUT reference images (cannot be selected):")
        for asset in without_images:
            lines.append(f"  [{asset.manifest_tag}] {asset.name} ({asset.asset_type}) — no image")

    return "\n".join(lines)
