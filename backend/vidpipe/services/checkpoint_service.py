"""PipeSVN checkpoint service — snapshot, SHA computation, and checkpoint management.

Provides:
- build_snapshot(): Serialize full scene state to a JSON-serializable dict
- compute_checkpoint_sha(): Deterministic SHA-1 from snapshot data
- create_checkpoint(): Build snapshot + SHA, create SceneCheckpoint row, update head_sha
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
    Scene,
    SceneCheckpoint,
    Shot,
    ShotAudioManifest,
    ShotManifest,
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


async def build_snapshot(session: AsyncSession, scene: Scene) -> dict:
    """Query all scene artifacts and serialize to a JSON-serializable dict.

    Structure:
    {
        "scene": { ... scene fields ... },
        "shots": [
            {
                "shot_index": 0,
                "shot_description": "...",
                ...
                "keyframes": [ { "position": "start", "file_path": "...", "file_hash": "...", ... } ],
                "clip": { "local_path": "...", "file_hash": "...", ... } | null,
                "shot_manifest": { ... } | null,
                "audio_manifest": { ... } | null,
            }
        ],
        "assets": [ ... ] | null,
    }
    """
    # Scene fields
    scene_data = {
        "id": str(scene.id),
        "prompt": scene.prompt,
        "title": scene.title,
        "style": scene.style,
        "aspect_ratio": scene.aspect_ratio,
        "target_clip_duration": scene.target_clip_duration,
        "target_shot_count": scene.target_shot_count,
        "total_duration": scene.total_duration,
        "text_model": scene.text_model,
        "image_model": scene.image_model,
        "video_model": scene.video_model,
        "audio_enabled": scene.audio_enabled,
        "seed": scene.seed,
        "manifest_id": str(scene.manifest_id) if scene.manifest_id else None,
        "quality_mode": scene.quality_mode,
        "candidate_count": scene.candidate_count,
        "vision_model": scene.vision_model,
        "output_path": scene.output_path,
        "status": scene.status,
    }

    # Load shots ordered by index
    shots_result = await session.execute(
        select(Shot)
        .where(Shot.scene_id == scene.id)
        .order_by(Shot.shot_index)
    )
    shots = shots_result.scalars().all()

    # Load shot manifests
    sm_result = await session.execute(
        select(ShotManifest).where(ShotManifest.scene_id == scene.id)
    )
    manifests_by_idx = {sm.shot_index: sm for sm in sm_result.scalars().all()}

    # Load audio manifests
    am_result = await session.execute(
        select(ShotAudioManifest).where(ShotAudioManifest.scene_id == scene.id)
    )
    audio_by_idx = {am.shot_index: am for am in am_result.scalars().all()}

    shot_list = []
    for shot in shots:
        # Keyframes
        kf_result = await session.execute(
            select(Keyframe).where(Keyframe.shot_id == shot.id)
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
            select(VideoClip).where(VideoClip.shot_id == shot.id)
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

        # Shot manifest
        sm = manifests_by_idx.get(shot.shot_index)
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
        am = audio_by_idx.get(shot.shot_index)
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

        shot_list.append({
            "shot_index": shot.shot_index,
            "shot_id": str(shot.id),
            "shot_description": shot.shot_description,
            "start_frame_prompt": shot.start_frame_prompt,
            "end_frame_prompt": shot.end_frame_prompt,
            "video_motion_prompt": shot.video_motion_prompt,
            "transition_notes": shot.transition_notes,
            "status": shot.status,
            "keyframes": kf_data,
            "clip": clip_data,
            "shot_manifest": sm_data,
            "audio_manifest": am_data,
        })

    # Assets (if manifest scene)
    assets_data = None
    if scene.manifest_id:
        assets_result = await session.execute(
            select(Asset).where(Asset.manifest_id == scene.manifest_id)
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
        "scene": scene_data,
        "shots": shot_list,
        "assets": assets_data,
    }


def compute_checkpoint_sha(snapshot_data: dict) -> str:
    """Deterministic SHA-1 from snapshot data via canonical JSON serialization."""
    canonical = json.dumps(snapshot_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


async def create_checkpoint(
    session: AsyncSession,
    scene: Scene,
    message: str,
    metadata: Optional[dict] = None,
) -> "SceneCheckpoint":
    """Build snapshot, compute SHA, create checkpoint row, update scene.head_sha.

    Does NOT commit — caller is responsible for committing the session.
    """
    snapshot = await build_snapshot(session, scene)
    sha = compute_checkpoint_sha(snapshot)

    parent_sha = scene.head_sha  # None for first checkpoint

    # Skip if snapshot is unchanged (same SHA as current head)
    if sha == parent_sha:
        logger.info(
            "Checkpoint skipped for scene %s — snapshot unchanged (sha=%s): %s",
            scene.id, sha[:8], message,
        )
        # Return existing checkpoint so callers don't break
        existing = await session.execute(
            select(SceneCheckpoint).where(
                SceneCheckpoint.scene_id == scene.id,
                SceneCheckpoint.sha == sha,
            )
        )
        return existing.scalar_one()

    checkpoint = SceneCheckpoint(
        scene_id=scene.id,
        sha=sha,
        parent_sha=parent_sha,
        snapshot_data=snapshot,
        message=message,
        metadata_json=metadata,
    )
    session.add(checkpoint)

    scene.head_sha = sha
    await session.flush()

    logger.info(
        "Checkpoint %s created for scene %s (parent=%s): %s",
        sha[:8], scene.id, parent_sha[:8] if parent_sha else "None", message,
    )
    return checkpoint


def compute_keyframe_staleness(
    shot, keyframe, shot_manifest=None
) -> str:
    """Compare keyframe.prompt_used against the current expected prompt.

    Returns "fresh", "stale", or "missing".
    """
    if keyframe is None:
        return "missing"

    # Determine the current expected prompt
    if keyframe.position == "start":
        current_prompt = shot.start_frame_prompt
    else:
        current_prompt = shot.end_frame_prompt

    # If shot_manifest has a rewritten prompt, that's the real expected prompt
    if shot_manifest and shot_manifest.rewritten_keyframe_prompt:
        current_prompt = shot_manifest.rewritten_keyframe_prompt

    if not keyframe.prompt_used:
        return "stale"  # No record of what prompt was used

    if keyframe.prompt_used == current_prompt:
        return "fresh"

    return "stale"


def compute_clip_staleness(
    shot, clip, shot_manifest=None
) -> str:
    """Compare clip.prompt_used against the current expected video prompt.

    Returns "fresh", "stale", or "missing".
    """
    if clip is None:
        return "missing"

    # Determine the current expected prompt
    current_prompt = shot.video_motion_prompt
    if shot_manifest and shot_manifest.rewritten_video_prompt:
        current_prompt = shot_manifest.rewritten_video_prompt

    if not clip.prompt_used:
        return "stale"  # No record of what prompt was used

    # For clips, prompt_used may include safety prefix + style suffix
    # so we check if the current prompt is contained within prompt_used
    if current_prompt in (clip.prompt_used or ""):
        return "fresh"

    return "stale"


async def restore_from_snapshot(
    session: AsyncSession, scene: Scene, snapshot: dict
) -> None:
    """Restore scene state from a snapshot, creating a forward-commit checkpoint.

    Restores Scene fields, upserts/deletes Shot/Keyframe/VideoClip rows.
    Does NOT commit — caller is responsible.
    """
    # Backward compat: accept both "scene" and legacy "project" keys
    proj_data = snapshot.get("scene") or snapshot.get("project") or {}

    # Restore scene-level fields
    restorable_fields = [
        "prompt", "title", "style", "aspect_ratio", "target_clip_duration",
        "target_shot_count", "total_duration", "text_model", "image_model",
        "video_model", "audio_enabled", "seed", "quality_mode", "candidate_count",
        "vision_model", "output_path",
    ]
    for field in restorable_fields:
        if field in proj_data:
            setattr(scene, field, proj_data[field])

    # Delete existing shots (cascades handled below manually)
    existing_shots = await session.execute(
        select(Shot).where(Shot.scene_id == scene.id)
    )
    for shot in existing_shots.scalars().all():
        # Delete keyframes
        kfs = await session.execute(select(Keyframe).where(Keyframe.shot_id == shot.id))
        for kf in kfs.scalars().all():
            await session.delete(kf)
        # Delete clips
        clips = await session.execute(select(VideoClip).where(VideoClip.shot_id == shot.id))
        for clip in clips.scalars().all():
            await session.delete(clip)
        await session.delete(shot)

    # Delete shot manifests and audio manifests
    sms = await session.execute(select(ShotManifest).where(ShotManifest.scene_id == scene.id))
    for sm in sms.scalars().all():
        await session.delete(sm)
    ams = await session.execute(select(ShotAudioManifest).where(ShotAudioManifest.scene_id == scene.id))
    for am in ams.scalars().all():
        await session.delete(am)

    await session.flush()

    # Recreate shots from snapshot
    for shot_data in snapshot.get("shots", []):
        shot = Shot(
            id=uuid.UUID(shot_data["shot_id"]) if shot_data.get("shot_id") else uuid.uuid4(),
            scene_id=scene.id,
            shot_index=shot_data["shot_index"],
            shot_description=shot_data.get("shot_description", ""),
            start_frame_prompt=shot_data.get("start_frame_prompt", ""),
            end_frame_prompt=shot_data.get("end_frame_prompt", ""),
            video_motion_prompt=shot_data.get("video_motion_prompt", ""),
            transition_notes=shot_data.get("transition_notes"),
            status=shot_data.get("status", "complete"),
        )
        session.add(shot)
        await session.flush()

        # Recreate keyframes
        for kf_data in shot_data.get("keyframes", []):
            kf = Keyframe(
                id=uuid.UUID(kf_data["id"]) if kf_data.get("id") else uuid.uuid4(),
                shot_id=shot.id,
                position=kf_data["position"],
                prompt_used=kf_data.get("prompt_used", ""),
                file_path=kf_data.get("file_path", ""),
                mime_type=kf_data.get("mime_type", "image/png"),
                source=kf_data.get("source", "generated"),
            )
            session.add(kf)

        # Recreate clip
        clip_data = shot_data.get("clip")
        if clip_data:
            clip = VideoClip(
                id=uuid.UUID(clip_data["id"]) if clip_data.get("id") else uuid.uuid4(),
                shot_id=shot.id,
                operation_name=clip_data.get("operation_name"),
                source=clip_data.get("source", "generated"),
                status=clip_data.get("status", "complete"),
                local_path=clip_data.get("local_path"),
                duration_seconds=clip_data.get("duration_seconds"),
                prompt_used=clip_data.get("prompt_used"),
            )
            session.add(clip)

        # Recreate shot manifest
        sm_data = shot_data.get("shot_manifest")
        if sm_data:
            sm = ShotManifest(
                scene_id=scene.id,
                shot_index=shot_data["shot_index"],
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
        am_data = shot_data.get("audio_manifest")
        if am_data:
            am = ShotAudioManifest(
                scene_id=scene.id,
                shot_index=shot_data["shot_index"],
                dialogue_json=am_data.get("dialogue_json"),
                sfx_json=am_data.get("sfx_json"),
                ambient_json=am_data.get("ambient_json"),
                music_json=am_data.get("music_json"),
                audio_continuity_json=am_data.get("audio_continuity_json"),
                speaker_tags=am_data.get("speaker_tags"),
            )
            session.add(am)

    await session.flush()
    logger.info("Restored scene %s from snapshot", scene.id)


def compute_diff(old_snapshot: dict, new_snapshot: dict) -> list[dict]:
    """Compute structured diff between two snapshots."""
    changes: list[dict] = []

    # Compare scene fields (backward compat: accept both "scene" and "project" keys)
    old_proj = old_snapshot.get("scene") or old_snapshot.get("project") or {}
    new_proj = new_snapshot.get("scene") or new_snapshot.get("project") or {}
    for key in set(old_proj.keys()) | set(new_proj.keys()):
        if key == "id":
            continue
        old_val = old_proj.get(key)
        new_val = new_proj.get(key)
        if old_val != new_val:
            changes.append({
                "type": "scene_field",
                "field": key,
                "old": str(old_val) if old_val is not None else None,
                "new": str(new_val) if new_val is not None else None,
            })

    # Compare shots
    old_shots = {s["shot_index"]: s for s in old_snapshot.get("shots", [])}
    new_shots = {s["shot_index"]: s for s in new_snapshot.get("shots", [])}

    for idx in sorted(set(old_shots.keys()) | set(new_shots.keys())):
        if idx not in old_shots:
            changes.append({"type": "shot_added", "shot_index": idx})
        elif idx not in new_shots:
            changes.append({"type": "shot_removed", "shot_index": idx})
        else:
            old_s = old_shots[idx]
            new_s = new_shots[idx]
            for field in ["shot_description", "start_frame_prompt", "end_frame_prompt",
                          "video_motion_prompt", "transition_notes", "status"]:
                if old_s.get(field) != new_s.get(field):
                    changes.append({
                        "type": "shot_field",
                        "shot_index": idx,
                        "field": field,
                    })

    return changes


def extract_file_paths_from_snapshot(snapshot: dict) -> set[str]:
    """Walk snapshot extracting all file_path / local_path values."""
    paths: set[str] = set()

    # Scene output (backward compat: accept both "scene" and "project" keys)
    proj = snapshot.get("scene") or snapshot.get("project") or {}
    if proj.get("output_path"):
        paths.add(proj["output_path"])

    for shot in snapshot.get("shots", []):
        for kf in shot.get("keyframes", []):
            if kf.get("file_path"):
                paths.add(kf["file_path"])
        clip = shot.get("clip")
        if clip and clip.get("local_path"):
            paths.add(clip["local_path"])

    return paths
