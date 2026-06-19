"""Workflow helpers used by the Vidpipe MCP server.

The MCP server intentionally drives the public HTTP API instead of reaching
around it into internals. That keeps agent-driven production runs aligned with
the browser UI and with the existing opt-in E2E tests.
"""

from __future__ import annotations

import asyncio
import base64
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator


ProgressCallback = Callable[[str], Awaitable[None]]
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class VidpipeMCPError(RuntimeError):
    """Raised when a Vidpipe API workflow cannot continue."""


class SceneSpec(BaseModel):
    """Screenplay scene metadata accepted by ``produce_project``."""

    scene_number: int | None = None
    slugline: str
    intent: str
    characters_present: list[str] = Field(default_factory=list)
    set_ref: str | None = None
    props_required: list[str] = Field(default_factory=list)
    emotional_beat: str | None = None

    def to_screenplay_entry(self, fallback_number: int) -> dict[str, Any]:
        return {
            "scene_number": self.scene_number or fallback_number,
            "slugline": self.slugline,
            "intent": self.intent,
            "characters_present": self.characters_present,
            "set_ref": self.set_ref,
            "props_required": self.props_required,
            "emotional_beat": self.emotional_beat,
        }


class ShotSpec(BaseModel):
    """Shot-list entry accepted by ``produce_project``."""

    scene_number: int
    shot_number: int
    description: str

    def to_screenplay_entry(self) -> dict[str, Any]:
        return {
            "scene_number": self.scene_number,
            "shot_number": self.shot_number,
            "description": self.description,
        }


class ProductionSpec(BaseModel):
    """Full project specification for an MCP-driven production run."""

    title: str
    story_brief: str
    genre: str = "Cinematic short"
    logline: str | None = None
    treatment: str | None = None
    character_breakdowns: list[dict[str, Any]] | None = None
    scene_breakdown: list[SceneSpec] | None = None
    shot_list: list[ShotSpec] | None = None
    script: str | None = None
    source_production_id: str | None = None
    production_bible_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    scene_count: int = Field(default=3, ge=1)
    shots_per_scene: int = Field(default=2, ge=1)
    clip_duration: int = Field(default=8, ge=1)
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    text_model: str = "gemini-3.5-flash"
    vision_model: str | None = "gemini-3.5-flash"
    image_model: str = "flux-2-klein"
    video_model: str = "ltx-2.3-flf2v"
    audio_enabled: bool = True
    generate_voice: bool = True
    generate_sound: bool = True
    render_master: bool = True
    create_default_narrator_bible: bool = True
    force_regenerate_screenplay_scenes: bool = True
    poll_interval_seconds: float = Field(default=20.0, ge=1.0)
    scene_timeout_seconds: int = Field(default=3600, ge=60)

    @field_validator("aspect_ratio")
    @classmethod
    def _validate_aspect_ratio(cls, value: str) -> str:
        if value not in {"16:9", "9:16"}:
            raise ValueError("aspect_ratio must be '16:9' or '9:16'")
        return value


class APIRequestSpec(BaseModel):
    """Generic Vidpipe API request accepted by the MCP bridge."""

    method: str
    path: str
    query: dict[str, Any] | None = None
    json_body: dict[str, Any] | list[Any] | None = None
    form_fields: dict[str, Any] | None = None
    file_path: str | None = None
    file_field: str = "file"
    file_content_type: str | None = None
    allow_mutation: bool = False
    include_binary: bool = False
    max_binary_bytes: int = Field(default=2_000_000, ge=1, le=20_000_000)

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        method = value.upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("method must be GET, POST, PUT, PATCH, or DELETE")
        return method

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return normalize_api_path(value)

    @model_validator(mode="after")
    def _validate_mutation(self) -> "APIRequestSpec":
        if self.method in MUTATING_METHODS and not self.allow_mutation:
            raise ValueError(
                f"{self.method} requests require allow_mutation=true because they may "
                "change state, delete data, or spend provider credits"
            )
        if self.json_body is not None and (self.form_fields is not None or self.file_path):
            raise ValueError("json_body cannot be combined with form_fields or file_path")
        if self.file_path and not self.file_field:
            raise ValueError("file_field is required when file_path is provided")
        return self


class VidpipeAPI:
    """Thin async client for the Vidpipe API."""

    def __init__(self, api_base: str) -> None:
        self.api_base = api_base.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=1200.0),
            follow_redirects=True,
        )

    async def __aenter__(self) -> "VidpipeAPI":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def api_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.api_base}/api{path}"

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._client.request(method, self.api_url(path), **kwargs)
        if response.status_code >= 400:
            raise VidpipeMCPError(
                f"{method} {path} failed with HTTP {response.status_code}: {response.text[:2000]}"
            )
        if response.status_code == 204 or not response.content:
            return None
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.content

    async def raw_request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._client.request(method, self.api_url(path), **kwargs)
        if response.status_code >= 400:
            raise VidpipeMCPError(
                f"{method} {path} failed with HTTP {response.status_code}: {response.text[:2000]}"
            )
        return response

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs: Any) -> Any:
        return await self.request("PUT", path, **kwargs)

    async def patch(self, path: str, **kwargs: Any) -> Any:
        return await self.request("PATCH", path, **kwargs)


@dataclass(frozen=True)
class ProductionArtifacts:
    """IDs and URLs produced by a completed workflow."""

    production_id: str
    production_url: str
    scene_ids: list[str]
    screenplay_id: str | None = None
    voice_script_id: str | None = None
    master_url: str | None = None


def normalize_api_base(api_base: str | None) -> str:
    return (api_base or "http://localhost:8100").rstrip("/")


def normalize_api_path(path: str) -> str:
    if "://" in path:
        raise ValueError("path must be relative to the Vidpipe API, not an absolute URL")
    normalized = path.strip()
    if not normalized:
        raise ValueError("path is required")
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    if normalized == "/api":
        normalized = "/"
    elif normalized.startswith("/api/"):
        normalized = normalized[4:]
    if ".." in normalized.split("/"):
        raise ValueError("path must not contain '..' segments")
    return normalized


def _binary_omitted_reason(request: APIRequestSpec, byte_count: int) -> str:
    if not request.include_binary:
        return "include_binary=false"
    if byte_count > request.max_binary_bytes:
        return f"binary response exceeds max_binary_bytes={request.max_binary_bytes}"
    return "binary response omitted"


async def get_api_catalog(
    api_base: str | None = None,
    *,
    method: str | None = None,
    path_contains: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Return a filtered catalog of the live Vidpipe OpenAPI surface."""

    base = normalize_api_base(api_base)
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(f"{base}/openapi.json")
        if response.status_code >= 400:
            raise VidpipeMCPError(
                f"GET /openapi.json failed with HTTP {response.status_code}: {response.text[:2000]}"
            )
        openapi = response.json()

    method_filter = method.upper() if method else None
    path_filter = path_contains.lower() if path_contains else None
    tag_filter = tag.lower() if tag else None
    routes = []
    valid_methods = {"get", "post", "put", "patch", "delete"}

    for route_path, operations in sorted(openapi.get("paths", {}).items()):
        for route_method, spec in sorted(operations.items()):
            if route_method not in valid_methods or not isinstance(spec, dict):
                continue
            method_upper = route_method.upper()
            tags = spec.get("tags") or []
            if method_filter and method_upper != method_filter:
                continue
            if path_filter and path_filter not in route_path.lower():
                continue
            if tag_filter and tag_filter not in {str(item).lower() for item in tags}:
                continue
            request_body = spec.get("requestBody") or {}
            routes.append(
                {
                    "method": method_upper,
                    "path": route_path,
                    "operation_id": spec.get("operationId"),
                    "summary": spec.get("summary"),
                    "tags": tags,
                    "has_request_body": bool(request_body),
                    "parameters": [
                        {
                            "name": parameter.get("name"),
                            "in": parameter.get("in"),
                            "required": parameter.get("required", False),
                        }
                        for parameter in spec.get("parameters", [])
                    ],
                }
            )

    return {
        "api_base": base,
        "total_paths": len(openapi.get("paths", {})),
        "total_operations": sum(
            1
            for operations in openapi.get("paths", {}).values()
            for route_method in operations
            if route_method in valid_methods
        ),
        "filtered_operations": len(routes),
        "routes": routes,
    }


async def call_api_endpoint(
    request: APIRequestSpec,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Call any Vidpipe ``/api`` endpoint through MCP with mutation safeguards."""

    base = normalize_api_base(api_base)
    kwargs: dict[str, Any] = {}
    if request.query:
        kwargs["params"] = request.query
    if request.json_body is not None:
        kwargs["json"] = request.json_body
    if request.form_fields:
        kwargs["data"] = request.form_fields

    async with VidpipeAPI(base) as api:
        if request.file_path:
            upload_path = Path(request.file_path).expanduser()
            if not upload_path.is_file():
                raise VidpipeMCPError(f"Upload file not found: {upload_path}")
            with upload_path.open("rb") as handle:
                kwargs["files"] = {
                    request.file_field: (
                        upload_path.name,
                        handle,
                        request.file_content_type or "application/octet-stream",
                    )
                }
                response = await api.raw_request(request.method, request.path, **kwargs)
        else:
            response = await api.raw_request(request.method, request.path, **kwargs)

    content_type = response.headers.get("content-type", "")
    shaped: dict[str, Any] = {
        "api_base": base,
        "method": request.method,
        "path": request.path,
        "status_code": response.status_code,
        "content_type": content_type,
    }
    if response.status_code == 204 or not response.content:
        shaped["body"] = None
    elif "application/json" in content_type:
        shaped["body"] = response.json()
    elif content_type.startswith("text/") or "charset=" in content_type:
        shaped["body"] = response.text
    else:
        shaped["binary"] = {
            "byte_count": len(response.content),
            "base64": base64.b64encode(response.content).decode("ascii")
            if request.include_binary and len(response.content) <= request.max_binary_bytes
            else None,
            "base64_omitted_reason": None
            if request.include_binary and len(response.content) <= request.max_binary_bytes
            else _binary_omitted_reason(request, len(response.content)),
        }
    return shaped


async def preflight(api_base: str | None = None) -> dict[str, Any]:
    """Check whether the Vidpipe API and provider prerequisites are reachable."""

    base = normalize_api_base(api_base)
    async with VidpipeAPI(base) as api:
        settings = await api.get("/settings")
        models = await api.get("/settings/models")

    missing: list[str] = []
    if not settings.get("has_comfyui_key") or not settings.get("comfyui_host"):
        missing.append("ComfyUI host/API key")
    if not settings.get("has_gcp_service_account") and not settings.get("has_api_key"):
        missing.append("Google API key or Vertex service account")
    if not settings.get("has_elevenlabs_key") or not settings.get("default_voice_id"):
        missing.append("ElevenLabs key/default voice")

    return {
        "api_base": base,
        "ok": not missing,
        "missing": missing,
        "settings": {
            "comfyui_host": settings.get("comfyui_host"),
            "has_comfyui_key": settings.get("has_comfyui_key"),
            "has_elevenlabs_key": settings.get("has_elevenlabs_key"),
            "default_voice_id": settings.get("default_voice_id"),
            "ollama_use_cloud": settings.get("ollama_use_cloud"),
            "has_ollama_key": settings.get("has_ollama_key"),
        },
        "models": models,
    }


async def get_project_status(
    production_id: str,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Return production, screenplay, scene, sound, and master status."""

    base = normalize_api_base(api_base)
    async with VidpipeAPI(base) as api:
        production = await api.get(f"/productions/{production_id}")
        screenplay = await api.get(f"/productions/{production_id}/screenplay")
        scenes = await _list_scenes_for_production(api, production_id)
        sound_deck = await _optional(api.get(f"/productions/{production_id}/sound-deck"))
        master = await _optional(api.get(f"/productions/{production_id}/master"))

    return {
        "production": production,
        "screenplay": screenplay,
        "scenes": scenes,
        "sound_deck": sound_deck,
        "master": master,
        "production_url": f"{base}/productions/{production_id}",
        "master_url": f"{base}{master['video_url']}" if master else None,
    }


async def produce_project(
    spec: ProductionSpec,
    api_base: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create and fully produce a Vidpipe project through the public API."""

    base = normalize_api_base(api_base)
    async with VidpipeAPI(base) as api:
        await _progress(progress, f"Creating production: {spec.title}")
        settings = await api.get("/settings")
        production_bible_id = await _resolve_or_create_bible(api, spec, settings, progress)

        production = await api.post(
            "/productions",
            json={
                "name": spec.title,
                "description": _production_description(spec),
                "tags": sorted(set([*spec.tags, "mcp"])),
            },
        )
        production_id = production["id"]
        if production_bible_id:
            await api.put(
                f"/productions/{production_id}",
                json={"production_bible_id": production_bible_id},
            )

        await _progress(progress, f"Seeding screenplay for production {production_id}")
        screenplay = await _seed_screenplay(api, production_id, spec)
        await api.patch(
            f"/productions/{production_id}/screenplay/status",
            json={"status": "LOCKED"},
        )

        await _progress(progress, "Creating scenes from locked screenplay")
        force = str(spec.force_regenerate_screenplay_scenes).lower()
        created_scenes = await api.post(
            f"/productions/{production_id}/screenplay/generate-scenes?force={force}"
        )
        if len(created_scenes) < spec.scene_count:
            raise VidpipeMCPError(
                f"Screenplay generated {len(created_scenes)} scenes, "
                f"but scene_count requested {spec.scene_count}. "
                "Pass a matching scene_breakdown or lower scene_count."
            )
        target_scenes = created_scenes[: spec.scene_count]
        scene_ids = [scene["scene_id"] for scene in target_scenes]

        completed = []
        for index, scene in enumerate(target_scenes, start=1):
            scene_id = scene["scene_id"]
            title = scene.get("title") or f"Scene {index}"
            await _progress(progress, f"Configuring scene {index}/{len(target_scenes)}: {title}")
            await api.patch(
                f"/scenes/{scene_id}/edit",
                json={
                    "clip_duration": spec.clip_duration,
                    "target_shot_count": spec.shots_per_scene,
                    "text_model": spec.text_model,
                    "vision_model": spec.vision_model,
                    "image_model": spec.image_model,
                    "video_model": spec.video_model,
                    "audio_enabled": spec.audio_enabled,
                    "production_bible_id": production_bible_id,
                    "dynamic_shot_count": False,
                    "style": spec.style,
                    "aspect_ratio": spec.aspect_ratio,
                    "commit_message": "Configure MCP production scene generation",
                },
            )
            await api.post(
                f"/scenes/{scene_id}/regenerate",
                json={
                    "scope": "all_phases",
                    "text_model": spec.text_model,
                    "image_model": spec.image_model,
                    "video_model": spec.video_model,
                },
            )
            completed_scene = await _wait_for_scene(
                api,
                scene_id,
                title,
                timeout_seconds=spec.scene_timeout_seconds,
                poll_interval=spec.poll_interval_seconds,
                progress=progress,
            )
            completed.append(completed_scene)

        voice_script_id = None
        if spec.generate_voice:
            await _progress(progress, "Generating and mixing voice script")
            voice_script = await _generate_voice(api, production_id, spec.text_model)
            voice_script_id = voice_script["id"]

        if spec.generate_sound:
            await _progress(progress, "Generating and mixing sound deck")
            await _generate_sound(api, production_id, spec.text_model)

        master = None
        if spec.render_master:
            await _progress(progress, "Rendering production master")
            master = await api.post(f"/productions/{production_id}/render-master")

        artifacts = ProductionArtifacts(
            production_id=production_id,
            production_url=f"{base}/productions/{production_id}",
            scene_ids=scene_ids,
            screenplay_id=screenplay.get("id"),
            voice_script_id=voice_script_id,
            master_url=f"{base}{master['video_url']}" if master else None,
        )
        return {
            "status": "complete",
            "artifacts": artifacts.__dict__,
            "production": await api.get(f"/productions/{production_id}"),
            "screenplay": await api.get(f"/productions/{production_id}/screenplay"),
            "scenes": [
                {
                    "scene_id": scene["scene_id"],
                    "title": scene.get("title"),
                    "status": scene.get("status"),
                    "shot_count": scene.get("shot_count"),
                }
                for scene in completed
            ],
            "master": master,
        }


async def _resolve_or_create_bible(
    api: VidpipeAPI,
    spec: ProductionSpec,
    settings: dict[str, Any],
    progress: ProgressCallback | None,
) -> str | None:
    if spec.production_bible_id:
        return spec.production_bible_id

    if spec.source_production_id:
        source = await api.get(f"/productions/{spec.source_production_id}")
        if source.get("production_bible_id"):
            return source["production_bible_id"]

    if not spec.create_default_narrator_bible:
        return None

    default_voice_id = settings.get("default_voice_id")
    if spec.generate_voice and default_voice_id:
        await _progress(progress, "Creating default voice-only narrator Production Bible")
        return await _create_narrator_bible(api, spec, default_voice_id)

    return None


async def _create_narrator_bible(
    api: VidpipeAPI,
    spec: ProductionSpec,
    default_voice_id: str,
) -> str:
    bible = await api.post(
        "/production-bibles",
        json={
            "name": f"{spec.title} Bible",
            "description": f"MCP-generated production bible for {spec.title}",
            "category": "FULL_PRODUCTION",
            "tags": sorted(set([*spec.tags, "mcp"])),
        },
    )
    bible_id = bible["production_bible_id"]
    actor = await api.post(
        "/asset-library/actors",
        json={
            "name": f"{spec.title} Narrator",
            "description": "Voice-only narrator created by the Vidpipe MCP server.",
            "base_appearance_prompt": "Voice-only narrator; no on-screen appearance.",
            "prompt_tags": ["NARRATOR", "voice-only"],
        },
    )
    voice = await api.post(
        f"/asset-library/actors/{actor['id']}/voice-profiles",
        json={
            "voice_id": default_voice_id,
            "adapter_type": "ELEVENLABS",
            "style_notes": "Clear cinematic narration matched to the production tone.",
        },
    )
    wardrobe = await api.post(
        f"/asset-library/actors/{actor['id']}/wardrobe-presets",
        json={
            "label": "Voice only",
            "description": "No wardrobe; narrator does not appear on screen.",
        },
    )
    binding = await api.post(
        f"/production-bibles/{bible_id}/cast",
        json={
            "actor_id": actor["id"],
            "tag": "NARRATOR",
            "character_name": "Narrator",
            "character_description": "Voice-only narrator. Never appears on screen.",
            "role": "NARRATOR",
            "voice_profile_id": voice["id"],
            "prompt_tags": ["NARRATOR", "voice-only"],
        },
    )
    await api.post(
        f"/cast-bindings/{binding['id']}/looks",
        json={
            "wardrobe_preset_id": wardrobe["id"],
            "tag": "NARRATOR_LOOK",
            "is_default": True,
        },
    )
    await api.post(f"/production-bibles/{bible_id}/finalize")
    return bible_id


async def _seed_screenplay(
    api: VidpipeAPI,
    production_id: str,
    spec: ProductionSpec,
) -> dict[str, Any]:
    await api.get(f"/productions/{production_id}/screenplay")

    source_context = None
    if spec.source_production_id:
        source_context = await _optional(
            api.get(f"/productions/{spec.source_production_id}/screenplay")
        )

    scene_breakdown = [
        scene.to_screenplay_entry(index)
        for index, scene in enumerate(
            spec.scene_breakdown or _default_scene_breakdown(spec, source_context),
            start=1,
        )
    ]
    shot_list = [
        shot.to_screenplay_entry()
        for shot in (spec.shot_list or _default_shot_list(spec, scene_breakdown))
    ]
    character_breakdowns = spec.character_breakdowns or _default_character_breakdowns(
        source_context=source_context,
    )
    script = spec.script or _default_script(spec, scene_breakdown)

    return await api.put(
        f"/productions/{production_id}/screenplay",
        json={
            "title": spec.title,
            "genre": spec.genre,
            "logline": spec.logline or spec.story_brief,
            "treatment": spec.treatment or _default_treatment(spec, source_context),
            "character_breakdowns": character_breakdowns,
            "scene_breakdown": scene_breakdown,
            "shot_list": shot_list,
            "script": script,
            "text_model": spec.text_model,
        },
    )


def _default_character_breakdowns(
    source_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if source_context and source_context.get("character_breakdowns"):
        return source_context["character_breakdowns"]
    return [
        {
            "tag": "NARRATOR",
            "name": "Narrator",
            "role": "voiceover narrator",
        }
    ]


def _default_treatment(
    spec: ProductionSpec,
    source_context: dict[str, Any] | None,
) -> str:
    source_summary = ""
    if source_context:
        source_summary = (
            f"\n\nContinue after this prior story state: {source_context.get('logline') or ''} "
            f"{source_context.get('treatment') or ''}"
        )
    return (
        f"{spec.story_brief}\n\n"
        f"Target structure: {spec.scene_count} scenes, about {spec.shots_per_scene} "
        f"shots per scene, {spec.clip_duration}s per shot. Each scene should have a "
        "clear visual objective, a forward story beat, and concrete cinematic action."
        f"{source_summary}"
    )


def _default_scene_breakdown(
    spec: ProductionSpec,
    source_context: dict[str, Any] | None,
) -> list[SceneSpec]:
    beats = _story_beats(spec.scene_count)
    visible_characters = _default_visible_characters(source_context)
    continuation = ""
    if source_context and source_context.get("title"):
        continuation = f" Continue from {source_context['title']}."
    scenes = []
    for index, beat in enumerate(beats, start=1):
        scenes.append(
            SceneSpec(
                scene_number=index,
                slugline=f"SCENE {index} - {beat['slugline']}",
                intent=(
                    f"{beat['intent']} Story brief: {spec.story_brief}.{continuation} "
                    "Use specific visual action, clear subject movement, and cinematic continuity."
                ),
                characters_present=visible_characters,
                emotional_beat=beat["emotional_beat"],
            )
        )
    return scenes


def _default_shot_list(
    spec: ProductionSpec,
    scene_breakdown: list[dict[str, Any]],
) -> list[ShotSpec]:
    shots = []
    for scene in scene_breakdown:
        scene_number = int(scene.get("scene_number") or len(shots) + 1)
        scene_intent = scene.get("intent", spec.story_brief)
        for shot_number in range(1, spec.shots_per_scene + 1):
            if shot_number == 1:
                description = f"Establish the scene visually: {scene_intent}"
            elif shot_number == spec.shots_per_scene:
                description = f"Resolve this scene beat and set up the next story turn: {scene_intent}"
            else:
                description = f"Advance the action with a clear character or camera movement: {scene_intent}"
            shots.append(
                ShotSpec(
                    scene_number=scene_number,
                    shot_number=shot_number,
                    description=description,
                )
            )
    return shots


def _default_script(
    spec: ProductionSpec,
    scene_breakdown: list[dict[str, Any]],
) -> str:
    lines = []
    for scene in scene_breakdown:
        beat = scene.get("emotional_beat") or scene.get("intent") or spec.story_brief
        lines.append(f"NARRATOR: {beat}")
    return "\n".join(lines)


def _story_beats(scene_count: int) -> list[dict[str, str]]:
    base = [
        ("OPENING IMAGE", "Open on the world and the central pressure of the story.", "Arrival and orientation."),
        ("INCITING TURN", "Introduce the disruption that changes what the protagonist must do.", "Curiosity turning to urgency."),
        ("COMPLICATION", "Escalate the obstacle and make the cost visible.", "Pressure and adaptation."),
        ("MIDPOINT", "Reveal a new truth that reframes the story goal.", "Recognition and commitment."),
        ("CRISIS", "Force the hardest choice and narrow the path forward.", "Loss, danger, and resolve."),
        ("CLIMAX", "Pay off the central conflict through decisive action.", "Release and transformation."),
        ("RESOLUTION", "Show the changed world after the decision lands.", "Aftermath and meaning."),
    ]
    if scene_count <= len(base):
        selected = _spread(base, scene_count)
    else:
        selected = list(base[:-1])
        for index in range(scene_count - len(base)):
            selected.append(
                (
                    f"ESCALATION {index + 1}",
                    "Deepen the pursuit, obstacle, or discovery while preserving continuity.",
                    "Rising tension.",
                )
            )
        selected.append(base[-1])
    return [
        {"slugline": slugline, "intent": intent, "emotional_beat": emotional_beat}
        for slugline, intent, emotional_beat in selected
    ]


def _spread(items: list[tuple[str, str, str]], count: int) -> list[tuple[str, str, str]]:
    if count == 1:
        return [items[-1]]
    indexes = [
        round(i * (len(items) - 1) / (count - 1))
        for i in range(count)
    ]
    return [items[index] for index in indexes]


async def _wait_for_scene(
    api: VidpipeAPI,
    scene_id: str,
    title: str,
    *,
    timeout_seconds: int,
    poll_interval: float,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    last_counts = None
    while time.monotonic() < deadline:
        detail = await api.get(f"/scenes/{scene_id}")
        shots = detail.get("shots") or []
        counts = {
            "shots": detail.get("shot_count"),
            "clips": sum(1 for shot in shots if shot.get("has_clip")),
            "start_keyframes": sum(1 for shot in shots if shot.get("has_start_keyframe")),
            "end_keyframes": sum(1 for shot in shots if shot.get("has_end_keyframe")),
        }
        status = detail.get("status")
        if status != last_status or counts != last_counts:
            await _progress(progress, f"{title}: status={status}, counts={counts}")
            last_status = status
            last_counts = counts
        if status == "complete":
            return detail
        if status in {"failed", "stopped"}:
            raise VidpipeMCPError(
                f"Scene {scene_id} ended as {status}: {detail.get('error_message')}"
            )
        await asyncio.sleep(poll_interval)
    raise VidpipeMCPError(f"Timed out waiting for scene {scene_id}")


async def _generate_voice(
    api: VidpipeAPI,
    production_id: str,
    text_model: str,
) -> dict[str, Any]:
    generated = await api.post(
        f"/productions/{production_id}/voice-script/generate",
        json={"text_model": text_model},
    )
    voice_script = generated["voice_script"]
    await api.post(f"/voice-scripts/{voice_script['id']}/resolve-bindings")
    audio = await api.post(f"/voice-scripts/{voice_script['id']}/generate-audio")
    ready_lines = [
        line
        for line in audio["voice_script"].get("lines", [])
        if line.get("generation_status") == "READY"
    ]
    if not ready_lines:
        raise VidpipeMCPError("Voice generation completed with no READY lines")
    mixed = await api.post(f"/voice-scripts/{voice_script['id']}/mix")
    ready_stems = [
        artifact
        for artifact in mixed["voice_script"].get("mix_artifacts", [])
        if artifact.get("status") == "READY"
    ]
    if not ready_stems:
        raise VidpipeMCPError("Voice mix completed with no READY artifacts")
    return mixed["voice_script"]


async def _generate_sound(
    api: VidpipeAPI,
    production_id: str,
    text_model: str,
) -> dict[str, Any]:
    await api.post(
        f"/productions/{production_id}/sound-deck/generate",
        json={"text_model": text_model},
    )
    audio = await api.post(f"/productions/{production_id}/sound-deck/generate-audio")
    ready_cues = [
        cue
        for cue in audio["sound_deck"].get("cues", [])
        if cue.get("generation_status") == "READY"
    ]
    if not ready_cues:
        raise VidpipeMCPError("Sound generation completed with no READY cues")
    mixed = await api.post(f"/productions/{production_id}/sound-deck/mix")
    ready_stems = [
        artifact
        for artifact in mixed["sound_deck"].get("mix_artifacts", [])
        if artifact.get("status") == "READY"
    ]
    if not ready_stems:
        raise VidpipeMCPError("Sound mix completed with no READY artifacts")
    return mixed["sound_deck"]


async def _optional(awaitable: Awaitable[Any]) -> Any | None:
    try:
        return await awaitable
    except VidpipeMCPError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise


async def _list_scenes_for_production(
    api: VidpipeAPI,
    production_id: str,
) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    page = 1
    per_page = 96
    while True:
        scenes_data = await api.get(
            "/scenes",
            params={"page": page, "per_page": per_page, "view": "cards"},
        )
        items = scenes_data.get("items", [])
        scenes.extend(
            scene
            for scene in items
            if scene.get("production_id") == production_id
        )
        total = int(scenes_data.get("total") or 0)
        if not items or page * per_page >= total:
            break
        page += 1

    scenes.sort(key=_scene_sort_key)
    return scenes


async def _progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        await progress(message)


def _production_description(spec: ProductionSpec) -> str:
    total_seconds = spec.scene_count * spec.shots_per_scene * spec.clip_duration
    return (
        f"MCP-generated production. {spec.story_brief} "
        f"Target: {spec.scene_count} scenes, {spec.shots_per_scene} shots/scene, "
        f"~{math.ceil(total_seconds / 60)} min maximum raw clip timeline. "
        f"Models: {spec.text_model}/{spec.image_model}/{spec.video_model}."
    )


def _default_visible_characters(source_context: dict[str, Any] | None) -> list[str]:
    if not source_context:
        return []
    for character in source_context.get("character_breakdowns") or []:
        tag = character.get("tag") or character.get("name")
        if tag and str(tag).upper() != "NARRATOR":
            return [str(tag)]
    return []


def _scene_sort_key(scene: dict[str, Any]) -> tuple[int, int, str, str]:
    max_order = 2**31 - 1
    return (
        scene.get("scene_order") if scene.get("scene_order") is not None else max_order,
        scene.get("screenplay_breakdown_index")
        if scene.get("screenplay_breakdown_index") is not None
        else max_order,
        scene.get("created_at") or "",
        scene.get("scene_id") or "",
    )
