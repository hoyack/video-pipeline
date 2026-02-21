"""PipeSVN checkpoint service — snapshot, SHA computation, and checkpoint management.

Provides:
- build_snapshot(): Serialize full project state to a JSON-serializable dict
- compute_checkpoint_sha(): Deterministic SHA-1 from snapshot data
- create_checkpoint(): Build snapshot + SHA, create ProjectCheckpoint row, update head_sha
- extract_file_paths_from_snapshot(): Walk snapshot extracting all file paths
"""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import (
    Asset,
    Keyframe,
    Project,
    ProjectCheckpoint,
    Scene,
    SceneAudioManifest,
    SceneManifest,
    VideoClip,
)

logger = logging.getLogger(__name__)

# Module-level cache: file_path -> SHA-1 hex digest
_file_hash_cache: dict[str, str] = {}


def _hash_file(path: str) -> Optional[str]:
    """Compute SHA-1 of file bytes, with module-level caching."""
    if path in _file_hash_cache:
        return _file_hash_cache[path]
    try:
        data = Path(path).read_bytes()
        h = hashlib.sha1(data).hexdigest()
        _file_hash_cache[path] = h
        return h
    except (OSError, IOError):
        return None


async def build_snapshot(session: AsyncSession, project: Project) -> dict:
    """Query all project artifacts and serialize to a JSON-serializable dict.

    Structure:
    {
        "project": { ... project fields ... },
        "scenes": [
            {
                "scene_index": 0,
                "scene_description": "...",
                ...
                "keyframes": [ { "position": "start", "file_path": "...", "file_hash": "...", ... } ],
                "clip": { "local_path": "...", "file_hash": "...", ... } | null,
                "scene_manifest": { ... } | null,
                "audio_manifest": { ... } | null,
            }
        ],
        "assets": [ ... ] | null,
    }
    """
    # Project fields
    project_data = {
        "id": str(project.id),
        "prompt": project.prompt,
        "title": project.title,
        "style": project.style,
        "aspect_ratio": project.aspect_ratio,
        "target_clip_duration": project.target_clip_duration,
        "target_scene_count": project.target_scene_count,
        "total_duration": project.total_duration,
        "text_model": project.text_model,
        "image_model": project.image_model,
        "video_model": project.video_model,
        "audio_enabled": project.audio_enabled,
        "seed": project.seed,
        "manifest_id": str(project.manifest_id) if project.manifest_id else None,
        "quality_mode": project.quality_mode,
        "candidate_count": project.candidate_count,
        "vision_model": project.vision_model,
        "output_path": project.output_path,
        "status": project.status,
    }

    # Load scenes ordered by index
    scenes_result = await session.execute(
        select(Scene)
        .where(Scene.project_id == project.id)
        .order_by(Scene.scene_index)
    )
    scenes = scenes_result.scalars().all()

    # Load scene manifests
    sm_result = await session.execute(
        select(SceneManifest).where(SceneManifest.project_id == project.id)
    )
    manifests_by_idx = {sm.scene_index: sm for sm in sm_result.scalars().all()}

    # Load audio manifests
    am_result = await session.execute(
        select(SceneAudioManifest).where(SceneAudioManifest.project_id == project.id)
    )
    audio_by_idx = {am.scene_index: am for am in am_result.scalars().all()}

    scene_list = []
    for scene in scenes:
        # Keyframes
        kf_result = await session.execute(
            select(Keyframe).where(Keyframe.scene_id == scene.id)
        )
        keyframes = kf_result.scalars().all()
        kf_data = []
        for kf in keyframes:
            kf_data.append({
                "id": str(kf.id),
                "position": kf.position,
                "prompt_used": kf.prompt_used,
                "file_path": kf.file_path,
                "file_hash": _hash_file(kf.file_path) if kf.file_path else None,
                "mime_type": kf.mime_type,
                "source": kf.source,
            })

        # Clip
        clip_result = await session.execute(
            select(VideoClip).where(VideoClip.scene_id == scene.id)
        )
        clip = clip_result.scalar_one_or_none()
        clip_data = None
        if clip:
            clip_data = {
                "id": str(clip.id),
                "status": clip.status,
                "local_path": clip.local_path,
                "file_hash": _hash_file(clip.local_path) if clip.local_path else None,
                "duration_seconds": clip.duration_seconds,
                "source": clip.source,
                "prompt_used": clip.prompt_used,
                "operation_name": clip.operation_name,
            }

        # Scene manifest
        sm = manifests_by_idx.get(scene.scene_index)
        sm_data = None
        if sm:
            sm_data = {
                "manifest_json": sm.manifest_json,
                "composition_shot_type": sm.composition_shot_type,
                "composition_camera_movement": sm.composition_camera_movement,
                "asset_tags": sm.asset_tags,
                "selected_reference_tags": sm.selected_reference_tags,
                "rewritten_keyframe_prompt": sm.rewritten_keyframe_prompt,
                "rewritten_video_prompt": sm.rewritten_video_prompt,
            }

        # Audio manifest
        am = audio_by_idx.get(scene.scene_index)
        am_data = None
        if am:
            am_data = {
                "dialogue_json": am.dialogue_json,
                "sfx_json": am.sfx_json,
                "ambient_json": am.ambient_json,
                "music_json": am.music_json,
                "audio_continuity_json": am.audio_continuity_json,
                "speaker_tags": am.speaker_tags,
            }

        scene_list.append({
            "scene_index": scene.scene_index,
            "scene_id": str(scene.id),
            "scene_description": scene.scene_description,
            "start_frame_prompt": scene.start_frame_prompt,
            "end_frame_prompt": scene.end_frame_prompt,
            "video_motion_prompt": scene.video_motion_prompt,
            "transition_notes": scene.transition_notes,
            "status": scene.status,
            "keyframes": kf_data,
            "clip": clip_data,
            "scene_manifest": sm_data,
            "audio_manifest": am_data,
        })

    # Assets (if manifest project)
    assets_data = None
    if project.manifest_id:
        assets_result = await session.execute(
            select(Asset).where(Asset.manifest_id == project.manifest_id)
        )
        assets = assets_result.scalars().all()
        assets_data = []
        for a in assets:
            assets_data.append({
                "id": str(a.id),
                "manifest_tag": a.manifest_tag,
                "name": a.name,
                "asset_type": a.asset_type,
                "reference_image_url": a.reference_image_url,
                "description": a.description,
            })

    return {
        "project": project_data,
        "scenes": scene_list,
        "assets": assets_data,
    }


def compute_checkpoint_sha(snapshot_data: dict) -> str:
    """Deterministic SHA-1 from snapshot data via canonical JSON serialization."""
    canonical = json.dumps(snapshot_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


async def create_checkpoint(
    session: AsyncSession,
    project: Project,
    message: str,
    metadata: Optional[dict] = None,
) -> "ProjectCheckpoint":
    """Build snapshot, compute SHA, create checkpoint row, update project.head_sha.

    Does NOT commit — caller is responsible for committing the session.
    """
    snapshot = await build_snapshot(session, project)
    sha = compute_checkpoint_sha(snapshot)

    parent_sha = project.head_sha  # None for first checkpoint

    checkpoint = ProjectCheckpoint(
        project_id=project.id,
        sha=sha,
        parent_sha=parent_sha,
        snapshot_data=snapshot,
        message=message,
        metadata_json=metadata,
    )
    session.add(checkpoint)

    project.head_sha = sha
    await session.flush()

    logger.info(
        "Checkpoint %s created for project %s (parent=%s): %s",
        sha[:8], project.id, parent_sha[:8] if parent_sha else "None", message,
    )
    return checkpoint


def compute_keyframe_staleness(
    scene, keyframe, scene_manifest=None
) -> str:
    """Compare keyframe.prompt_used against the current expected prompt.

    Returns "fresh", "stale", or "missing".
    """
    if keyframe is None:
        return "missing"

    # Determine the current expected prompt
    if keyframe.position == "start":
        current_prompt = scene.start_frame_prompt
    else:
        current_prompt = scene.end_frame_prompt

    # If scene_manifest has a rewritten prompt, that's the real expected prompt
    if scene_manifest and scene_manifest.rewritten_keyframe_prompt:
        current_prompt = scene_manifest.rewritten_keyframe_prompt

    if not keyframe.prompt_used:
        return "stale"  # No record of what prompt was used

    if keyframe.prompt_used == current_prompt:
        return "fresh"

    return "stale"


def compute_clip_staleness(
    scene, clip, scene_manifest=None
) -> str:
    """Compare clip.prompt_used against the current expected video prompt.

    Returns "fresh", "stale", or "missing".
    """
    if clip is None:
        return "missing"

    # Determine the current expected prompt
    current_prompt = scene.video_motion_prompt
    if scene_manifest and scene_manifest.rewritten_video_prompt:
        current_prompt = scene_manifest.rewritten_video_prompt

    if not clip.prompt_used:
        return "stale"  # No record of what prompt was used

    # For clips, prompt_used may include safety prefix + style suffix
    # so we check if the current prompt is contained within prompt_used
    if current_prompt in (clip.prompt_used or ""):
        return "fresh"

    return "stale"


async def restore_from_snapshot(
    session: AsyncSession, project: Project, snapshot: dict
) -> None:
    """Restore project state from a snapshot, creating a forward-commit checkpoint.

    Restores Project fields, upserts/deletes Scene/Keyframe/VideoClip rows.
    Does NOT commit — caller is responsible.
    """
    proj_data = snapshot.get("project", {})

    # Restore project-level fields
    restorable_fields = [
        "prompt", "title", "style", "aspect_ratio", "target_clip_duration",
        "target_scene_count", "total_duration", "text_model", "image_model",
        "video_model", "audio_enabled", "seed", "quality_mode", "candidate_count",
        "vision_model", "output_path",
    ]
    for field in restorable_fields:
        if field in proj_data:
            setattr(project, field, proj_data[field])

    # Delete existing scenes (cascades handled below manually)
    existing_scenes = await session.execute(
        select(Scene).where(Scene.project_id == project.id)
    )
    for scene in existing_scenes.scalars().all():
        # Delete keyframes
        kfs = await session.execute(select(Keyframe).where(Keyframe.scene_id == scene.id))
        for kf in kfs.scalars().all():
            await session.delete(kf)
        # Delete clips
        clips = await session.execute(select(VideoClip).where(VideoClip.scene_id == scene.id))
        for clip in clips.scalars().all():
            await session.delete(clip)
        await session.delete(scene)

    # Delete scene manifests and audio manifests
    sms = await session.execute(select(SceneManifest).where(SceneManifest.project_id == project.id))
    for sm in sms.scalars().all():
        await session.delete(sm)
    ams = await session.execute(select(SceneAudioManifest).where(SceneAudioManifest.project_id == project.id))
    for am in ams.scalars().all():
        await session.delete(am)

    await session.flush()

    # Recreate scenes from snapshot
    for scene_data in snapshot.get("scenes", []):
        scene = Scene(
            id=uuid.UUID(scene_data["scene_id"]) if scene_data.get("scene_id") else uuid.uuid4(),
            project_id=project.id,
            scene_index=scene_data["scene_index"],
            scene_description=scene_data.get("scene_description", ""),
            start_frame_prompt=scene_data.get("start_frame_prompt", ""),
            end_frame_prompt=scene_data.get("end_frame_prompt", ""),
            video_motion_prompt=scene_data.get("video_motion_prompt", ""),
            transition_notes=scene_data.get("transition_notes"),
            status=scene_data.get("status", "complete"),
        )
        session.add(scene)
        await session.flush()

        # Recreate keyframes
        for kf_data in scene_data.get("keyframes", []):
            kf = Keyframe(
                id=uuid.UUID(kf_data["id"]) if kf_data.get("id") else uuid.uuid4(),
                scene_id=scene.id,
                position=kf_data["position"],
                prompt_used=kf_data.get("prompt_used", ""),
                file_path=kf_data.get("file_path", ""),
                mime_type=kf_data.get("mime_type", "image/png"),
                source=kf_data.get("source", "generated"),
            )
            session.add(kf)

        # Recreate clip
        clip_data = scene_data.get("clip")
        if clip_data:
            clip = VideoClip(
                id=uuid.UUID(clip_data["id"]) if clip_data.get("id") else uuid.uuid4(),
                scene_id=scene.id,
                operation_name=clip_data.get("operation_name"),
                source=clip_data.get("source", "generated"),
                status=clip_data.get("status", "complete"),
                local_path=clip_data.get("local_path"),
                duration_seconds=clip_data.get("duration_seconds"),
                prompt_used=clip_data.get("prompt_used"),
            )
            session.add(clip)

        # Recreate scene manifest
        sm_data = scene_data.get("scene_manifest")
        if sm_data:
            sm = SceneManifest(
                project_id=project.id,
                scene_index=scene_data["scene_index"],
                manifest_json=sm_data.get("manifest_json", {}),
                composition_shot_type=sm_data.get("composition_shot_type"),
                composition_camera_movement=sm_data.get("composition_camera_movement"),
                asset_tags=sm_data.get("asset_tags"),
                selected_reference_tags=sm_data.get("selected_reference_tags"),
                rewritten_keyframe_prompt=sm_data.get("rewritten_keyframe_prompt"),
                rewritten_video_prompt=sm_data.get("rewritten_video_prompt"),
            )
            session.add(sm)

        # Recreate audio manifest
        am_data = scene_data.get("audio_manifest")
        if am_data:
            am = SceneAudioManifest(
                project_id=project.id,
                scene_index=scene_data["scene_index"],
                dialogue_json=am_data.get("dialogue_json"),
                sfx_json=am_data.get("sfx_json"),
                ambient_json=am_data.get("ambient_json"),
                music_json=am_data.get("music_json"),
                audio_continuity_json=am_data.get("audio_continuity_json"),
                speaker_tags=am_data.get("speaker_tags"),
            )
            session.add(am)

    await session.flush()
    logger.info("Restored project %s from snapshot", project.id)


def compute_diff(old_snapshot: dict, new_snapshot: dict) -> list[dict]:
    """Compute structured diff between two snapshots."""
    changes: list[dict] = []

    # Compare project fields
    old_proj = old_snapshot.get("project", {})
    new_proj = new_snapshot.get("project", {})
    for key in set(old_proj.keys()) | set(new_proj.keys()):
        if key == "id":
            continue
        old_val = old_proj.get(key)
        new_val = new_proj.get(key)
        if old_val != new_val:
            changes.append({
                "type": "project_field",
                "field": key,
                "old": str(old_val) if old_val is not None else None,
                "new": str(new_val) if new_val is not None else None,
            })

    # Compare scenes
    old_scenes = {s["scene_index"]: s for s in old_snapshot.get("scenes", [])}
    new_scenes = {s["scene_index"]: s for s in new_snapshot.get("scenes", [])}

    for idx in sorted(set(old_scenes.keys()) | set(new_scenes.keys())):
        if idx not in old_scenes:
            changes.append({"type": "scene_added", "scene_index": idx})
        elif idx not in new_scenes:
            changes.append({"type": "scene_removed", "scene_index": idx})
        else:
            old_s = old_scenes[idx]
            new_s = new_scenes[idx]
            for field in ["scene_description", "start_frame_prompt", "end_frame_prompt",
                          "video_motion_prompt", "transition_notes", "status"]:
                if old_s.get(field) != new_s.get(field):
                    changes.append({
                        "type": "scene_field",
                        "scene_index": idx,
                        "field": field,
                    })

    return changes


def extract_file_paths_from_snapshot(snapshot: dict) -> set[str]:
    """Walk snapshot extracting all file_path / local_path values."""
    paths: set[str] = set()

    # Project output
    proj = snapshot.get("project", {})
    if proj.get("output_path"):
        paths.add(proj["output_path"])

    for scene in snapshot.get("scenes", []):
        for kf in scene.get("keyframes", []):
            if kf.get("file_path"):
                paths.add(kf["file_path"])
        clip = scene.get("clip")
        if clip and clip.get("local_path"):
            paths.add(clip["local_path"])

    return paths
