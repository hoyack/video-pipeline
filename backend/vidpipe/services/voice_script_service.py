"""Voice script generation, voice binding, TTS, mixing, and lip-sync service."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.config import settings
from vidpipe.db.models import (
    ActorVoiceProfile,
    CastBinding,
    Character,
    LipSyncJob,
    Production,
    Scene,
    Screenplay,
    Shot,
    VideoClip,
    VoiceLine,
    VoiceMixArtifact,
    VoiceProfile,
    VoiceScript,
)
from vidpipe.schemas.voice import GeneratedVoiceScript
from vidpipe.services.audio.base import AudioAdapter, AudioAdapterError
from vidpipe.services.audio.registry import get_audio_adapter
from vidpipe.services.lip_sync import LipSyncRequest, get_lip_sync_adapter
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend
from vidpipe.services.voice_mixer import VoiceMixer, VoiceMixerError

logger = logging.getLogger(__name__)


VOICE_SCRIPT_SYSTEM_PROMPT = """You convert production screenplays into a timeline-aligned voice script.
Return only JSON matching the schema. For documentary productions, write enough narration
to cover the requested spoken-duration target while leaving short visual breathing room.
Use DIALOGUE for character speech and NARRATION for narrator lines. Keep speaker_tag aligned
with Production Bible tags when available. Do not write DIALOGUE for cast members that do not
have a voice profile; turn those beats into NARRATION instead. Keep line text ready for
text-to-speech."""


class VoiceScriptService:
    """Coordinates generated voice scripts and downstream voice artifacts."""

    async def get_or_create_script(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
    ) -> VoiceScript:
        production = await session.get(Production, production_id)
        if production is None:
            raise ValueError("Production not found")

        screenplay_result = await session.execute(
            select(Screenplay).where(Screenplay.production_id == production_id)
        )
        screenplay = screenplay_result.scalar_one_or_none()
        if screenplay is None:
            screenplay = Screenplay(
                production_id=production_id,
                title=production.name,
                status="DRAFT",
            )
            session.add(screenplay)
            await session.flush()

        script_result = await session.execute(
            select(VoiceScript).where(VoiceScript.screenplay_id == screenplay.id)
        )
        voice_script = script_result.scalar_one_or_none()
        if voice_script is None:
            voice_script = VoiceScript(
                screenplay_id=screenplay.id,
                production_id=production_id,
                source_screenplay_updated_at=screenplay.updated_at,
            )
            session.add(voice_script)
            await session.flush()
        return voice_script

    async def get_script_by_id(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
    ) -> VoiceScript:
        voice_script = await session.get(VoiceScript, voice_script_id)
        if voice_script is None:
            raise ValueError("Voice script not found")
        return voice_script

    async def generate_from_screenplay(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        llm_adapter: Any,
        *,
        text_model: str | None = None,
    ) -> VoiceScript:
        voice_script = await self.get_or_create_script(session, production_id)
        screenplay = await session.get(Screenplay, voice_script.screenplay_id)
        if screenplay is None:
            raise ValueError("Screenplay not found")

        requirements = await self._build_voice_requirements(session, production_id)
        prompt = await self._build_generation_prompt(
            session,
            production_id,
            screenplay,
            requirements,
        )
        generated = await llm_adapter.generate_text(
            prompt,
            GeneratedVoiceScript,
            temperature=0.4,
            system_prompt=VOICE_SCRIPT_SYSTEM_PROMPT,
        )
        if not isinstance(generated, GeneratedVoiceScript):
            generated = GeneratedVoiceScript.model_validate(generated)
        if not self._voice_script_meets_coverage(generated, requirements):
            retry_prompt = (
                f"{prompt}\n"
                "Coverage failure: the previous voice script was too sparse for the timeline. "
                f"Regenerate it with at least {requirements['min_line_count']} lines and "
                f"{requirements['min_word_count']} spoken words, distributed across the listed "
                "scenes and shots. Return the complete replacement JSON only."
            )
            regenerated = await llm_adapter.generate_text(
                retry_prompt,
                GeneratedVoiceScript,
                temperature=0.45,
                system_prompt=VOICE_SCRIPT_SYSTEM_PROMPT,
            )
            if not isinstance(regenerated, GeneratedVoiceScript):
                regenerated = GeneratedVoiceScript.model_validate(regenerated)
            if self._voice_script_meets_coverage(regenerated, requirements):
                generated = regenerated
            else:
                logger.warning(
                    "Generated voice script remains sparse: lines=%s words=%s minimum_lines=%s minimum_words=%s",
                    len(regenerated.lines),
                    self._count_script_words(regenerated),
                    requirements["min_line_count"],
                    requirements["min_word_count"],
                )
                generated = regenerated

        await session.execute(
            delete(VoiceLine).where(VoiceLine.voice_script_id == voice_script.id)
        )
        await session.flush()

        for index, generated_line in enumerate(generated.lines):
            scene_id, shot_id = await self._resolve_scene_shot(
                session,
                production_id,
                generated_line.scene_number,
                generated_line.shot_number,
            )
            line = VoiceLine(
                voice_script_id=voice_script.id,
                production_id=production_id,
                scene_number=generated_line.scene_number,
                scene_id=scene_id,
                shot_number=generated_line.shot_number,
                shot_id=shot_id,
                line_index=index,
                line_type=generated_line.line_type,
                speaker_tag=generated_line.speaker_tag,
                speaker_name=generated_line.speaker_name,
                text=generated_line.text,
                delivery_notes=generated_line.delivery_notes,
                timing_hint=generated_line.timing_hint,
                lip_sync_mode=generated_line.lip_sync_mode,
                generation_status="PENDING",
            )
            session.add(line)

        voice_script.version += 1
        voice_script.status = "DRAFT"
        voice_script.script_model = text_model
        voice_script.source_screenplay_updated_at = screenplay.updated_at
        voice_script.voice_generation_status = "PENDING"
        voice_script.mix_status = None
        voice_script.lip_sync_status = None
        voice_script.error_message = None
        await session.flush()

        await self.resolve_all_bindings(session, voice_script.id)
        await session.commit()
        await session.refresh(voice_script)
        return voice_script

    async def list_lines(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
    ) -> list[VoiceLine]:
        result = await session.execute(
            select(VoiceLine)
            .where(VoiceLine.voice_script_id == voice_script_id)
            .order_by(VoiceLine.line_index)
        )
        return list(result.scalars().all())

    async def list_mix_artifacts(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
    ) -> list[VoiceMixArtifact]:
        result = await session.execute(
            select(VoiceMixArtifact)
            .where(VoiceMixArtifact.voice_script_id == voice_script_id)
            .order_by(VoiceMixArtifact.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_lip_sync_jobs(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
    ) -> list[LipSyncJob]:
        result = await session.execute(
            select(LipSyncJob)
            .join(VoiceLine, LipSyncJob.voice_line_id == VoiceLine.id)
            .where(VoiceLine.voice_script_id == voice_script_id)
            .order_by(LipSyncJob.created_at.desc())
        )
        return list(result.scalars().all())

    async def resolve_all_bindings(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
    ) -> list[VoiceLine]:
        voice_script = await self.get_script_by_id(session, voice_script_id)
        production = await session.get(Production, voice_script.production_id)
        if production is None:
            raise ValueError("Production not found")
        lines = await self.list_lines(session, voice_script_id)
        for line in lines:
            line.scene_id, line.shot_id = await self._resolve_scene_shot(
                session,
                production.id,
                line.scene_number,
                line.shot_number,
            )
            await self.resolve_line_binding(session, line, production)
        await session.flush()
        return lines

    async def resolve_line_binding(
        self,
        session: AsyncSession,
        line: VoiceLine,
        production: Production,
    ) -> VoiceLine:
        if production.production_bible_id is None:
            line.error_message = "Production has no Production Bible for voice binding"
            line.generation_status = "SKIPPED"
            return line

        binding = await self._match_cast_binding(session, line, production.production_bible_id)
        if binding is not None:
            line.cast_binding_id = binding.id
            line.speaker_tag = line.speaker_tag or binding.tag
            line.speaker_name = line.speaker_name or binding.character_name
            profile = await self._actor_voice_profile(session, binding)
            if profile is not None and profile.voice_id:
                line.actor_voice_profile_id = profile.id
                line.character_voice_profile_id = None
                line.voice_id = profile.voice_id
                line.adapter_type = profile.adapter_type or "ELEVENLABS"
                line.error_message = None
                if line.generation_status == "SKIPPED":
                    line.generation_status = "PENDING"
                return line

        profile = await self._legacy_character_voice_profile(session, line, production.production_bible_id)
        if profile is not None and profile.voice_id:
            line.character_voice_profile_id = profile.id
            line.actor_voice_profile_id = None
            line.voice_id = profile.voice_id
            line.adapter_type = profile.adapter_type or "ELEVENLABS"
            line.error_message = None
            if line.generation_status == "SKIPPED":
                line.generation_status = "PENDING"
            return line

        if line.line_type == "NARRATION" and line.voice_id:
            line.error_message = None
            return line

        line.generation_status = "SKIPPED"
        line.error_message = "No voice profile matched this line"
        return line

    async def update_line(
        self,
        session: AsyncSession,
        line_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> VoiceLine:
        line = await session.get(VoiceLine, line_id)
        if line is None:
            raise ValueError("Voice line not found")

        uuid_fields = {
            "cast_binding_id",
            "actor_voice_profile_id",
            "character_voice_profile_id",
        }
        dirty_generation_fields = {
            "text",
            "delivery_notes",
            "speaker_tag",
            "speaker_name",
            "voice_id",
            "adapter_type",
            "cast_binding_id",
            "actor_voice_profile_id",
            "character_voice_profile_id",
        }
        should_reset_audio = bool(dirty_generation_fields.intersection(updates))
        for field_name, value in updates.items():
            if field_name in uuid_fields and value is not None:
                value = uuid.UUID(str(value))
            setattr(line, field_name, value)

        if should_reset_audio:
            line.generation_status = "PENDING"
            line.audio_path = None
            line.duration_seconds = None
            line.error_message = None
        await session.commit()
        await session.refresh(line)
        return line

    async def delete_line(self, session: AsyncSession, line_id: uuid.UUID) -> None:
        line = await session.get(VoiceLine, line_id)
        if line is None:
            raise ValueError("Voice line not found")
        await session.delete(line)
        await session.commit()

    async def generate_line_audio(
        self,
        session: AsyncSession,
        line_id: uuid.UUID,
        *,
        audio_adapter: AudioAdapter | None = None,
        api_key: str | None = None,
    ) -> VoiceLine:
        line = await session.get(VoiceLine, line_id)
        if line is None:
            raise ValueError("Voice line not found")
        if not line.voice_id:
            line.generation_status = "SKIPPED"
            line.error_message = "No voice_id assigned"
            await self._update_voice_generation_status(session, line.voice_script_id)
            await session.commit()
            return line

        existing_audio_path = line.audio_path
        existing_duration_seconds = line.duration_seconds
        line.generation_status = "GENERATING"
        line.error_message = None
        await session.flush()

        try:
            adapter = audio_adapter or get_audio_adapter(line.adapter_type, api_key=api_key)
            content = await adapter.generate_voice(
                line.voice_id,
                line.text,
                style_notes=line.delivery_notes,
            )
            audio_key = f"productions/{line.production_id}/voice/{line.voice_script_id}/lines/{line.id}.mp3"
            storage = get_storage_backend()
            await storage.put(audio_key, content, "audio/mpeg")

            line.audio_path = audio_key
            line.audio_mime_type = "audio/mpeg"
            line.duration_seconds = await self._probe_stored_audio_duration(audio_key)
            line.generation_status = "READY"
            line.error_message = None
        except (AudioAdapterError, Exception) as exc:
            if existing_audio_path:
                line.audio_path = existing_audio_path
                line.duration_seconds = existing_duration_seconds
                line.generation_status = "READY"
                line.error_message = f"Regeneration failed; existing audio retained: {exc}"[:500]
            else:
                line.generation_status = "FAILED"
                line.error_message = str(exc)[:500]
            logger.exception("Voice line generation failed for %s", line.id)
        await self._update_voice_generation_status(session, line.voice_script_id)
        await session.commit()
        await session.refresh(line)
        return line

    async def generate_pending_audio(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
        *,
        audio_adapter: AudioAdapter | None = None,
        api_key: str | None = None,
    ) -> list[VoiceLine]:
        voice_script = await self.get_script_by_id(session, voice_script_id)
        lines = await self.list_lines(session, voice_script_id)
        generated: list[VoiceLine] = []
        for line in lines:
            if line.generation_status not in {"PENDING", "FAILED"}:
                continue
            if not line.voice_id:
                line.generation_status = "SKIPPED"
                line.error_message = "No voice_id assigned"
                generated.append(line)
                continue
            generated.append(
                await self.generate_line_audio(
                    session,
                    line.id,
                    audio_adapter=audio_adapter,
                    api_key=api_key,
                )
            )
        await self._update_voice_generation_status(session, voice_script.id)
        await session.commit()
        return generated

    async def _update_voice_generation_status(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
    ) -> None:
        voice_script = await self.get_script_by_id(session, voice_script_id)
        lines = await self.list_lines(session, voice_script_id)
        statuses = {line.generation_status for line in lines}
        if not statuses:
            voice_script.voice_generation_status = "PENDING"
        elif statuses <= {"READY", "SKIPPED"}:
            voice_script.voice_generation_status = "READY" if "READY" in statuses else "SKIPPED"
        elif "READY" in statuses or "SKIPPED" in statuses:
            voice_script.voice_generation_status = "PARTIAL"
        elif "GENERATING" in statuses:
            voice_script.voice_generation_status = "GENERATING"
        elif "FAILED" in statuses:
            voice_script.voice_generation_status = "FAILED"
        else:
            voice_script.voice_generation_status = "PENDING"

    async def build_mix_artifacts(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
        *,
        mixer: VoiceMixer | None = None,
    ) -> list[VoiceMixArtifact]:
        voice_script = await self.get_script_by_id(session, voice_script_id)
        lines = [line for line in await self.list_lines(session, voice_script_id) if line.audio_path]
        if not lines:
            voice_script.mix_status = "SKIPPED"
            voice_script.error_message = "No generated voice audio to mix"
            await session.commit()
            return []

        await session.execute(
            delete(VoiceMixArtifact).where(VoiceMixArtifact.voice_script_id == voice_script_id)
        )
        groups: dict[uuid.UUID | None, list[VoiceLine]] = defaultdict(list)
        for line in lines:
            groups[line.scene_id].append(line)

        mixer = mixer or VoiceMixer()
        artifacts: list[VoiceMixArtifact] = []
        for scene_id, group_lines in groups.items():
            payload_lines = [
                {"audio_path": line.audio_path, "line_index": line.line_index}
                for line in sorted(group_lines, key=lambda item: item.line_index)
            ]
            artifact = VoiceMixArtifact(
                voice_script_id=voice_script.id,
                scene_id=scene_id,
                artifact_type="SCENE_VOICE_STEM",
                status="GENERATING",
            )
            session.add(artifact)
            await session.flush()
            try:
                key, duration = await mixer.build_stem(
                    voice_script.production_id,
                    voice_script.id,
                    payload_lines,
                    scene_id=scene_id,
                )
                artifact.audio_path = key
                artifact.duration_seconds = duration
                artifact.status = "READY"
            except VoiceMixerError as exc:
                artifact.status = "FAILED"
                artifact.error_message = str(exc)[:500]
            artifacts.append(artifact)

        voice_script.mix_status = "READY" if all(a.status == "READY" for a in artifacts) else "PARTIAL"
        await session.commit()
        return artifacts

    async def queue_lip_sync_jobs(
        self,
        session: AsyncSession,
        voice_script_id: uuid.UUID,
        *,
        adapter_type: str = "FAKE",
        run_now: bool = True,
    ) -> list[LipSyncJob]:
        voice_script = await self.get_script_by_id(session, voice_script_id)
        lines = [line for line in await self.list_lines(session, voice_script_id) if line.audio_path]

        jobs: list[LipSyncJob] = []
        for line in lines:
            if line.lip_sync_mode == "NEVER" or line.line_type != "DIALOGUE" or line.shot_id is None:
                continue
            clip = await self._completed_clip_for_shot(session, line.shot_id)
            if clip is None or not clip.local_path:
                line.lip_sync_status = "SKIPPED"
                continue
            job = LipSyncJob(
                voice_line_id=line.id,
                shot_id=line.shot_id,
                input_clip_id=clip.id,
                input_audio_path=line.audio_path,
                adapter_type=adapter_type,
                status="QUEUED",
            )
            session.add(job)
            await session.flush()
            if run_now:
                await self._run_lip_sync_job(job, line, clip)
            line.lip_sync_status = job.status
            jobs.append(job)

        voice_script.lip_sync_status = "READY" if jobs and all(j.status == "READY" for j in jobs) else "PARTIAL"
        await session.commit()
        return jobs

    async def _run_lip_sync_job(
        self,
        job: LipSyncJob,
        line: VoiceLine,
        clip: VideoClip,
    ) -> None:
        storage = get_storage_backend()
        base_dir = settings.storage.tmp_dir
        input_video = await self._ensure_local_path(clip.local_path, base_dir)
        input_audio = await self._ensure_local_path(line.audio_path, base_dir)
        output_key = f"productions/{line.production_id}/voice/{line.voice_script_id}/lip_sync/{line.id}.mp4"
        output_path = base_dir / output_key

        try:
            adapter = get_lip_sync_adapter(job.adapter_type)
            result = await adapter.sync(
                LipSyncRequest(
                    input_video_path=input_video,
                    input_audio_path=input_audio,
                    output_video_path=output_path,
                    speaker_tag=line.speaker_tag,
                )
            )
            data = result.output_video_path.read_bytes()
            await storage.put(output_key, data, "video/mp4")
            job.output_clip_path = output_key
            job.metrics_json = result.metrics
            job.status = "READY"
            job.completed_at = datetime.now(UTC).replace(tzinfo=None)
        except Exception as exc:
            job.status = "FAILED"
            job.error_message = str(exc)[:500]

    async def _build_generation_prompt(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        screenplay: Screenplay,
        requirements: dict[str, Any],
    ) -> str:
        production = await session.get(Production, production_id)
        cast = []
        if production and production.production_bible_id:
            result = await session.execute(
                select(CastBinding).where(CastBinding.production_bible_id == production.production_bible_id)
            )
            cast = []
            for binding in result.scalars().all():
                profile = await self._actor_voice_profile(session, binding)
                cast.append(
                    {
                        "tag": binding.tag,
                        "name": binding.character_name,
                        "role": binding.role,
                        "notes": binding.behavioral_notes,
                        "has_voice_profile": bool(profile and profile.voice_id),
                    }
                )

        return (
            f"Title: {screenplay.title or (production.name if production else 'Untitled')}\n"
            f"Logline: {screenplay.logline or ''}\n"
            f"Treatment: {screenplay.treatment or ''}\n"
            f"Character breakdowns: {screenplay.character_breakdowns or []}\n"
            f"Scene breakdown: {screenplay.scene_breakdown or []}\n"
            f"Shot list: {screenplay.shot_list or []}\n"
            f"Script:\n{screenplay.script or ''}\n"
            f"Production Bible cast bindings: {cast}\n"
            "Voice timing requirements:\n"
            f"- Production runtime: {requirements['production_duration_seconds']:.1f} seconds.\n"
            f"- Target spoken narration/dialogue duration: {requirements['target_spoken_seconds']:.1f} seconds.\n"
            f"- Target total spoken words: {requirements['target_word_count']} "
            f"(acceptable minimum {requirements['min_word_count']}).\n"
            f"- Target voice lines: {requirements['target_line_count']} "
            f"(minimum {requirements['min_line_count']}).\n"
            f"- Scene timing plan: {requirements['scene_plan']}.\n"
            "Write narration that spans the whole timeline, not only the opening. "
            "For documentaries, prefer one concise NARRATION line per shot or major visual beat. "
            "Only use DIALOGUE when the named cast binding has has_voice_profile=true. "
            "If no character voice is available, rewrite quotes and character speech as narrator context. "
            "Bind each line to the relevant scene_number and shot_number. "
            "Keep individual lines natural for TTS, usually 8-16 words, and set timing_hint "
            "to describe where the line should fall inside the shot or scene.\n"
        )

    async def _build_voice_requirements(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
    ) -> dict[str, Any]:
        result = await session.execute(
            select(Scene)
            .where(Scene.production_id == production_id)
            .where(Scene.deleted_at.is_(None))
            .order_by(Scene.scene_order, Scene.screenplay_breakdown_index, Scene.created_at)
        )
        scenes = list(result.scalars().all())

        scene_plan: list[dict[str, Any]] = []
        production_duration = 0.0
        total_shots = 0
        for scene_number, scene in enumerate(scenes, start=1):
            shot_count_result = await session.execute(
                select(func.count(Shot.id)).where(Shot.scene_id == scene.id)
            )
            shot_count = int(shot_count_result.scalar_one() or 0)
            planned_shots = shot_count or scene.target_shot_count or 1
            scene_duration = float(scene.total_duration or (scene.target_clip_duration * planned_shots))
            production_duration += scene_duration
            total_shots += planned_shots
            scene_plan.append(
                {
                    "scene_number": scene_number,
                    "scene_id": str(scene.id),
                    "title": scene.title,
                    "duration_seconds": round(scene_duration, 2),
                    "shot_count": planned_shots,
                    "target_voice_lines": planned_shots,
                    "target_words": max(1, round(scene_duration * 2.15)),
                }
            )

        if production_duration <= 0:
            production_duration = float(
                screenplay_duration
                if (screenplay_duration := len(scene_plan) * 30) > 0
                else 0
            )

        target_spoken_seconds = production_duration * 0.84 if production_duration else 0.0
        target_word_count = round(target_spoken_seconds * 2.55) if target_spoken_seconds else 0
        min_word_count = round(target_spoken_seconds * 2.3) if target_spoken_seconds else 0
        target_line_count = total_shots or (math.ceil(production_duration / 7) if production_duration else 0)
        min_line_count = max(0, min(target_line_count, math.ceil(production_duration / 8))) if production_duration else 0

        return {
            "production_duration_seconds": production_duration,
            "target_spoken_seconds": target_spoken_seconds,
            "target_word_count": target_word_count,
            "min_word_count": min_word_count,
            "target_line_count": target_line_count,
            "min_line_count": min_line_count,
            "scene_plan": scene_plan,
        }

    def _voice_script_meets_coverage(
        self,
        generated: GeneratedVoiceScript,
        requirements: dict[str, Any],
    ) -> bool:
        if requirements["production_duration_seconds"] < 20:
            return True
        return (
            len(generated.lines) >= requirements["min_line_count"]
            and self._count_script_words(generated) >= requirements["min_word_count"]
        )

    @staticmethod
    def _count_script_words(generated: GeneratedVoiceScript) -> int:
        return sum(len(line.text.split()) for line in generated.lines)

    async def _resolve_scene_shot(
        self,
        session: AsyncSession,
        production_id: uuid.UUID,
        scene_number: int | None,
        shot_number: int | None,
    ) -> tuple[uuid.UUID | None, uuid.UUID | None]:
        scene_id = None
        shot_id = None
        if scene_number is not None:
            scene_result = await session.execute(
                select(Scene)
                .where(Scene.production_id == production_id)
                .where(Scene.deleted_at.is_(None))
                .order_by(
                    Scene.scene_order,
                    Scene.screenplay_breakdown_index,
                    Scene.created_at,
                )
            )
            scenes = list(scene_result.scalars().all())
            scene = self._match_scene_number(scenes, scene_number)
            scene_id = scene.id if scene is not None else None
            if scene is not None and shot_number is not None:
                shot_result = await session.execute(
                    select(Shot)
                    .where(Shot.scene_id == scene.id)
                    .order_by(Shot.shot_index)
                )
                shots = list(shot_result.scalars().all())
                shot = self._match_shot_number(shots, shot_number)
                shot_id = shot.id if shot is not None else None
        return scene_id, shot_id

    @staticmethod
    def _match_scene_number(scenes: list[Scene], scene_number: int) -> Scene | None:
        for scene in scenes:
            if scene.screenplay_breakdown_index == scene_number:
                return scene
        for scene in scenes:
            if scene.screenplay_breakdown_index == scene_number - 1:
                return scene
        for scene in scenes:
            if scene.scene_order == scene_number:
                return scene
        index = scene_number - 1
        if 0 <= index < len(scenes):
            return scenes[index]
        return None

    @staticmethod
    def _match_shot_number(shots: list[Shot], shot_number: int) -> Shot | None:
        for shot in shots:
            if shot.shot_index == shot_number - 1:
                return shot
        for shot in shots:
            if shot.shot_index == shot_number:
                return shot
        index = shot_number - 1
        if 0 <= index < len(shots):
            return shots[index]
        return None

    async def _match_cast_binding(
        self,
        session: AsyncSession,
        line: VoiceLine,
        production_bible_id: uuid.UUID,
    ) -> CastBinding | None:
        if line.line_type == "NARRATION":
            narrator_result = await session.execute(
                select(CastBinding)
                .where(CastBinding.production_bible_id == production_bible_id)
                .where(func.upper(CastBinding.role) == "NARRATOR")
            )
            narrator = narrator_result.scalar_one_or_none()
            if narrator is not None:
                return narrator

        if line.speaker_tag:
            result = await session.execute(
                select(CastBinding)
                .where(CastBinding.production_bible_id == production_bible_id)
                .where(func.lower(CastBinding.tag) == line.speaker_tag.lower())
            )
            binding = result.scalar_one_or_none()
            if binding is not None:
                return binding
        if line.speaker_name:
            result = await session.execute(
                select(CastBinding)
                .where(CastBinding.production_bible_id == production_bible_id)
                .where(func.lower(CastBinding.character_name) == line.speaker_name.lower())
            )
            return result.scalar_one_or_none()
        return None

    async def _actor_voice_profile(
        self,
        session: AsyncSession,
        binding: CastBinding,
    ) -> ActorVoiceProfile | None:
        if binding.voice_profile_id:
            profile = await session.get(ActorVoiceProfile, binding.voice_profile_id)
            if profile is not None:
                return profile
        result = await session.execute(
            select(ActorVoiceProfile)
            .where(ActorVoiceProfile.actor_id == binding.actor_id)
            .order_by(ActorVoiceProfile.created_at.asc())
        )
        return result.scalars().first()

    async def _legacy_character_voice_profile(
        self,
        session: AsyncSession,
        line: VoiceLine,
        production_bible_id: uuid.UUID,
    ) -> VoiceProfile | None:
        if not line.speaker_name:
            return None
        character_result = await session.execute(
            select(Character)
            .where(Character.production_bible_id == production_bible_id)
            .where(func.lower(Character.name) == line.speaker_name.lower())
        )
        character = character_result.scalar_one_or_none()
        if character is None:
            return None
        profile_result = await session.execute(
            select(VoiceProfile).where(VoiceProfile.character_id == character.id)
        )
        return profile_result.scalar_one_or_none()

    async def _completed_clip_for_shot(
        self,
        session: AsyncSession,
        shot_id: uuid.UUID,
    ) -> VideoClip | None:
        result = await session.execute(
            select(VideoClip)
            .where(VideoClip.shot_id == shot_id)
            .where(VideoClip.status == "complete")
            .order_by(VideoClip.completed_at.desc().nullslast(), VideoClip.created_at.desc())
        )
        return result.scalars().first()

    async def _probe_stored_audio_duration(self, audio_key: str) -> float | None:
        local_path = await self._ensure_local_path(audio_key, settings.storage.tmp_dir)
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(local_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return float(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None

    async def _ensure_local_path(self, key_or_path: str | None, base_dir: Path) -> Path:
        if not key_or_path:
            raise FileNotFoundError("Missing media path")
        path = Path(key_or_path)
        if path.is_absolute() and path.exists():
            return path

        storage = get_storage_backend()
        if isinstance(storage, LocalStorageBackend):
            local_path = storage.resolve_local_path(key_or_path)
            if local_path.exists():
                return local_path
        local_path = base_dir / key_or_path
        if local_path.exists():
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(storage, LocalStorageBackend):
            source = storage.resolve_local_path(key_or_path)
            if source.exists():
                shutil.copyfile(source, local_path)
                return local_path
        local_path.write_bytes(await storage.get(key_or_path))
        return local_path
