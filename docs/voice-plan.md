# Voice Script, TTS, and Lip Sync Plan

## Goal

Add a first-class voice layer to productions:

- Generate a structured voice script from the production story, screenplay, and Production Bible bindings.
- Bind every spoken line to either narration or a specific cast character.
- Generate narration and character dialogue through the existing ElevenLabs audio adapter.
- Let users edit, preview, regenerate, and lock voice lines in the Productions Screenplay UI.
- Mix generated voice into scene or shot videos.
- Add an optional post-video lip-sync pass for visible character dialogue.

This should extend the current screenplay and binding system rather than creating a parallel story format.

## Current Codebase Anchors

Existing pieces to reuse:

- `Screenplay` stores logline, treatment, character breakdowns, scene breakdown, script, and shot list in `backend/vidpipe/db/models.py`.
- `ScreenwriterService` generates the six current screenplay steps in `backend/vidpipe/services/screenwriter.py`.
- `ScreenplayEditor` renders the current screenplay tabs in `frontend/src/components/ScreenplayEditor.tsx`.
- `CastBinding.voice_profile_id` already links a Production Bible cast role to an `ActorVoiceProfile`.
- `ActorVoiceProfile` stores reusable actor voice metadata for library actors.
- `VoiceProfile` stores bible-scoped character voice metadata.
- `audio_gen.py` and `asset_library.py` already call `get_audio_adapter(...).generate_voice(...)`.
- `ElevenLabsAdapter.generate_voice()` exists in `backend/vidpipe/services/audio/elevenlabs_adapter.py`.
- `ShotAudioManifest` already stores storyboard-level dialogue, SFX, ambient, and music hints per shot.
- `video_gen.py` already passes audio manifest context into prompt rewriting for Veo native audio.
- `stitcher.py` already uses ffmpeg and preserves audio when concatenating clips.

Important current gap: audio manifests are prompt instructions for video generation. They are not persistent generated voice assets, they do not bind each spoken line to a TTS voice, and they do not create audio stems or lip-sync jobs.

## Product Shape

### User Workflow

1. User creates or generates a Production Screenplay.
2. User binds a Production Bible to the production.
3. User casts actors through `CastBinding`, including optional `voice_profile_id`.
4. User opens a new Screenplay "Voice" tab.
5. User clicks "Generate Voice Script".
6. System creates structured narration and dialogue lines from the story and screenplay.
7. User reviews and edits line text, speaker binding, delivery notes, timing hints, and lip-sync eligibility.
8. User clicks "Generate Voices".
9. System generates ElevenLabs audio per line, stores reusable line audio artifacts, and builds shot/scene voice stems.
10. User previews generated voice layers in the UI.
11. During generation or regeneration, the pipeline overlays voice stems after video clip generation.
12. Optional lip-sync pass updates generated clips for eligible character dialogue.

### Voice Types

Support two top-level line types:

- `NARRATION`: voiceover not tied to a visible speaker. Can use a production narrator cast binding, a dedicated narrator voice setting, or a default ElevenLabs voice.
- `DIALOGUE`: spoken by a cast character. Must resolve to a `CastBinding` when possible.

Future line types can include `ADR`, `WALLA`, `ANNOUNCER`, and `INNER_MONOLOGUE`, but v1 should stay to narration and dialogue.

## Data Model

Add durable voice-script tables instead of storing everything as nested JSON on `Screenplay`.

### `voice_scripts`

One active voice script per screenplay, with versioning.

Fields:

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | primary key |
| `screenplay_id` | UUID | FK to `screenplays.id`, indexed |
| `production_id` | UUID | denormalized FK for query convenience |
| `status` | string | `DRAFT`, `IN_REVIEW`, `LOCKED`, `GENERATING`, `FAILED` |
| `version` | int | increments when regenerated from screenplay |
| `script_model` | string nullable | LLM model used for voice-script generation |
| `source_screenplay_updated_at` | datetime nullable | stale detection |
| `voice_generation_status` | string nullable | aggregate TTS status |
| `mix_status` | string nullable | aggregate mix status |
| `lip_sync_status` | string nullable | aggregate lip-sync status |
| `error_message` | text nullable | last failed operation |
| `created_at` | datetime | |
| `updated_at` | datetime | |

Constraints:

- Unique active script by `screenplay_id` if we do not support multiple alternatives in v1.
- Keep version number even if only one active row exists.

### `voice_lines`

One row per spoken line or narration segment.

Fields:

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | primary key |
| `voice_script_id` | UUID | FK, indexed |
| `production_id` | UUID | denormalized for list endpoints |
| `scene_number` | int nullable | screenplay scene number |
| `scene_id` | UUID nullable | FK to generated `scenes.id` once scenes exist |
| `shot_number` | int nullable | screenplay shot number |
| `shot_id` | UUID nullable | FK to generated `shots.id` once shots exist |
| `line_index` | int | order within voice script |
| `line_type` | string | `NARRATION` or `DIALOGUE` |
| `speaker_tag` | string nullable | raw cast tag, no leading `@` |
| `cast_binding_id` | UUID nullable | resolved binding for character dialogue |
| `actor_voice_profile_id` | UUID nullable | selected actor-library voice |
| `character_voice_profile_id` | UUID nullable | fallback bible-scoped voice |
| `voice_id` | string nullable | provider voice id snapshot used for generation |
| `adapter_type` | string | `ELEVENLABS` initially |
| `text` | text | spoken line |
| `delivery_notes` | text nullable | pacing, emotion, emphasis |
| `start_time_seconds` | float nullable | target offset within shot or scene |
| `end_time_seconds` | float nullable | optional target end |
| `duration_seconds` | float nullable | measured generated audio duration |
| `audio_path` | string nullable | generated line audio artifact |
| `audio_mime_type` | string | `audio/mpeg` initially |
| `generation_status` | string | `PENDING`, `GENERATING`, `READY`, `FAILED`, `SKIPPED` |
| `lip_sync_mode` | string | `NONE`, `AUTO`, `FORCE`, `SKIP` |
| `lip_sync_status` | string nullable | per-line sync status |
| `provider_metadata` | JSON nullable | ElevenLabs model, settings, request id if available |
| `error_message` | text nullable | |
| `created_at` | datetime | |
| `updated_at` | datetime | |

Indexes:

- `(voice_script_id, line_index)`
- `(scene_id, shot_id)`
- `(cast_binding_id)`
- `(generation_status)`

### `voice_mix_artifacts`

Represents composed audio stems for playback and final mix.

Fields:

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | primary key |
| `voice_script_id` | UUID | FK |
| `scene_id` | UUID nullable | scene-level mix |
| `shot_id` | UUID nullable | shot-level mix |
| `artifact_type` | string | `SHOT_VOICE_STEM`, `SCENE_VOICE_STEM`, `FINAL_MIX` |
| `audio_path` | string | stored mixed audio |
| `duration_seconds` | float nullable | |
| `status` | string | `READY`, `FAILED` |
| `created_at` | datetime | |

### `lip_sync_jobs`

Tracks post-video lip-sync processing.

Fields:

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUID | primary key |
| `voice_line_id` | UUID nullable | line-level job |
| `shot_id` | UUID | target shot |
| `input_clip_id` | UUID | original `video_clips.id` |
| `input_audio_path` | string | generated voice or shot stem |
| `output_clip_path` | string nullable | lip-synced clip |
| `adapter_type` | string | `WAV2LIP`, `MUSETALK`, `LATENTSYNC`, `EXTERNAL` |
| `status` | string | `QUEUED`, `RUNNING`, `READY`, `FAILED`, `SKIPPED` |
| `eligibility_reason` | text nullable | why job was allowed or skipped |
| `metrics_json` | JSON nullable | face count, sync score, runtime |
| `error_message` | text nullable | |
| `created_at` | datetime | |
| `completed_at` | datetime nullable | |

## Binding Resolution

Voice line generation should use the same Production Bible binding vocabulary as visual generation.

Resolution order for a dialogue line:

1. `voice_line.cast_binding_id` if manually assigned.
2. `voice_line.speaker_tag` matched case-insensitively to `CastBinding.tag` in the production bible.
3. Character name matched to `CastBinding.character_name`.
4. If unresolved, leave line as `generation_status=SKIPPED` with a UI warning.

Voice profile resolution order:

1. `CastBinding.voice_profile_id` -> `ActorVoiceProfile`.
2. Actor's first/default `ActorVoiceProfile` if no binding-specific voice profile exists.
3. Legacy `VoiceProfile` for a matching bible-scoped `Character`, if present.
4. Production narrator voice for `NARRATION`.
5. User settings default ElevenLabs voice, if configured.
6. Skip with actionable error if no voice is resolvable.

For narrators, support two patterns:

- A cast binding with `role="NARRATOR"` and `voice_profile_id`.
- A production-level narrator setting on `voice_scripts` or future `production_voice_settings`.

## Voice Script Generation

Add a seventh Screenwriter step: `voice_script`.

### Prompt Inputs

The voice script generator should receive:

- `Screenplay.logline`
- `Screenplay.treatment`
- `Screenplay.character_breakdowns`
- `Screenplay.scene_breakdown`
- `Screenplay.shot_list`
- `Screenplay.script`
- Production Bible binding registry, including cast tags and voice availability
- Current scene and shot duration hints

Do not ask the LLM to invent provider voice ids. It should output `speaker_tag`, character name, line text, timing intent, and delivery notes only. Backend resolution chooses actual voice profiles.

### Structured Output

Add Pydantic schemas in `backend/vidpipe/schemas/voice.py`:

```python
class VoiceLineOutput(BaseModel):
    scene_number: int | None
    shot_number: int | None
    line_type: Literal["NARRATION", "DIALOGUE"]
    speaker_tag: str | None
    speaker_name: str | None
    text: str
    delivery_notes: str | None
    timing_hint: str | None
    lip_sync_mode: Literal["NONE", "AUTO", "SKIP"]

class VoiceScriptOutput(BaseModel):
    lines: list[VoiceLineOutput]
```

Validation rules:

- `DIALOGUE` lines should have `speaker_tag` when the character is in the Production Bible.
- `NARRATION` lines should not require a visible speaker.
- Keep generated lines short enough for the target shot or scene.
- Prefer one dialogue speaker per shot for lip-syncable v1 output.
- Preserve screenplay story intent. Do not rewrite scene outcomes.

## API Plan

Add `backend/vidpipe/api/voice_script.py` with these endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/productions/{production_id}/voice-script` | Get or create the active voice script |
| `PUT` | `/api/voice-scripts/{voice_script_id}` | Update script metadata/status |
| `POST` | `/api/productions/{production_id}/voice-script/generate` | Generate structured lines from screenplay |
| `POST` | `/api/voice-scripts/{voice_script_id}/resolve-bindings` | Re-resolve cast and voice profiles |
| `POST` | `/api/voice-scripts/{voice_script_id}/generate-audio` | Generate all pending/dirty TTS line audio |
| `POST` | `/api/voice-lines/{voice_line_id}/generate-audio` | Generate or regenerate one line |
| `PATCH` | `/api/voice-lines/{voice_line_id}` | Edit text, speaker, delivery, timing, lip-sync mode |
| `DELETE` | `/api/voice-lines/{voice_line_id}` | Delete a line |
| `POST` | `/api/voice-scripts/{voice_script_id}/mix` | Build shot/scene voice stems |
| `POST` | `/api/voice-scripts/{voice_script_id}/lip-sync` | Queue lip-sync jobs for eligible lines/shots |
| `GET` | `/api/voice-scripts/{voice_script_id}/jobs` | Poll TTS, mix, and lip-sync progress |

Return line-level warnings in all detail responses:

- Missing cast binding
- Missing voice profile
- Generated audio longer than shot duration
- Multiple visible speakers, lip sync skipped
- Native video audio enabled, voice mix may conflict

## ElevenLabs Integration

Keep the provider boundary in `AudioAdapter`, but expand generation options.

### Adapter Updates

Extend `generate_voice()` to accept:

- `output_format`, default `mp3_44100_128`
- `voice_settings`, optional dict
- `previous_text` and `next_text`, optional context for smoother multi-line prosody if supported by the provider
- `seed`, optional if supported

The current ElevenLabs API supports `POST /v1/text-to-speech/:voice_id`, `text`, optional `model_id`, and output format query parameters. Keep `eleven_multilingual_v2` as the compatibility default, but allow per-profile override so we can use faster or more expressive models later.

Storage:

- Store each line at `tmp/productions/{production_id}/voice/lines/{voice_line_id}.mp3` locally.
- For object storage, use `productions/{production_id}/voice/lines/{voice_line_id}.mp3`.
- Store mixed stems under `productions/{production_id}/voice/mixes/...`.

Generation behavior:

- Process lines concurrently with a small semaphore, initially 2 or 3.
- Commit each line independently so UI progress updates survive interruption.
- Hash `(voice_id, model_id, text, delivery_notes, voice_settings)` to avoid regenerating unchanged audio.
- Mark stale lines when text, speaker, delivery notes, or selected voice profile changes.

## Audio Timing and Mixing

### Duration Measurement

Use `ffprobe` or a small Python audio metadata library to measure generated audio duration after writing it. Persist `duration_seconds` on `voice_lines`.

### Timing v1

Start with simple deterministic placement:

- If `start_time_seconds` is set, place line at that offset.
- Else place narration at scene start or between dialogue groups.
- Else place dialogue at the beginning of the target shot with a small configurable lead-in.
- If audio is longer than target shot duration, warn and allow user to regenerate shorter, split, or extend the shot later.

### Mixing v1

Use ffmpeg to build audio stems:

- Normalize line audio to a consistent sample rate.
- Delay each line with `adelay`.
- Mix lines with `amix`.
- Optional limiter/normalizer.
- Overlay stem onto video with `-map` and `-shortest` or explicit duration handling.

Pipeline placement:

1. Storyboard and keyframes remain unchanged.
2. Video clips are generated.
3. Voice lines are generated if missing or stale.
4. Shot voice stems are mixed.
5. Optional lip sync runs on eligible clips.
6. Final stitch uses lip-synced clips when available, otherwise original clips.
7. Final audio mix is preserved or overlaid during scene/final stitching.

Recommendation: when deterministic voice layering is enabled, default `scene.audio_enabled=false` for Veo native audio to avoid generated speech conflicts. SFX and ambience can come from native audio later, but v1 should prefer clean voice control.

## Lip Sync Strategy

Lip sync should be an optional post-video step with a provider abstraction.

### Adapter Interface

Add `backend/vidpipe/services/lip_sync/base.py`:

```python
class LipSyncAdapter(ABC):
    async def sync_clip(
        self,
        input_video_path: Path,
        input_audio_path: Path,
        output_video_path: Path,
        *,
        face_hint: dict | None = None,
        options: dict | None = None,
    ) -> LipSyncResult:
        ...
```

Add registry:

- `WAV2LIP_LOCAL`
- `MUSETALK_LOCAL`
- `LATENTSYNC_LOCAL`
- `EXTERNAL_HTTP`

### Recommended v1 Implementation

Use `WAV2LIP_LOCAL` or an external HTTP wrapper first because it is mature, well understood, and works as a post-processor with an input face video plus speech audio. It is not the highest-fidelity option, but it is the lowest-risk adapter to prove the pipeline.

Eligibility rules for v1:

- Only process `DIALOGUE` lines.
- Only process shots with exactly one active speaker.
- Prefer close-up or medium shots.
- Skip if no face is detected in sampled frames.
- Skip if multiple large faces are detected unless a future `face_hint` can identify the speaker.
- Skip non-human identity types by default.
- Skip lines marked `lip_sync_mode=SKIP`.

Use existing CV infrastructure where possible:

- Sample frames from the shot clip.
- Detect faces.
- Compare actor face embeddings if available.
- Record eligibility and metrics in `lip_sync_jobs.metrics_json`.

### Future Quality Options

Keep the abstraction open for:

- MuseTalk: designed for real-time/high-quality lip synchronization with latent-space inpainting.
- LatentSync: diffusion-based lip-sync method using audio-conditioned latent diffusion.
- Hosted providers such as Sync Labs or similar APIs if local GPU runtime is too expensive.

These should be later adapters, not baked into the data model.

## UI Plan

### Screenplay Voice Tab

Add a new tab in `ScreenplayEditor`:

- `Voice`

Controls:

- Generate Voice Script
- Resolve Bindings
- Generate Voices
- Mix Voice Layer
- Lip Sync Eligible Shots
- Lock Voice Script

Views:

- Scene-grouped line table.
- Columns: order, scene, shot, type, speaker, text, delivery, voice profile, duration, audio status, lip-sync status.
- Inline speaker selector sourced from cast bindings.
- Inline voice profile selector when a cast binding has multiple actor voice profiles.
- Audio preview per generated line.
- Scene or shot voice-stem preview.
- Warnings panel for unresolved speakers, missing voices, overlong lines, and lip-sync skips.

Editing rules:

- Locked screenplay can still allow voice-script edits if voice script status is `DRAFT`.
- Locked voice script blocks line edits and regeneration.
- If screenplay changes after voice generation, show stale warning and offer regenerate or keep current voice script.

### Production Bible and Actor UI

Improve existing voice profile UX:

- On cast binding, show whether a voice profile is attached.
- Add quick action: "Use actor default voice".
- Add quick action: "Test as character", using the binding's `character_name` and `behavioral_notes`.
- Add warnings for cast characters without voice profiles before voice-script generation.

## Pipeline Integration

### Orchestrator Changes

Add optional stages:

- `voice_script`
- `voice_audio`
- `voice_mix`
- `lip_sync`

The existing `run_through` should eventually include:

- `storyboard`
- `keyframes`
- `clips`
- `voice`
- `lip_sync`
- `stitch`

For v1, keep voice operations callable from Screenplay UI and scene regeneration APIs first. Then wire them into full generation once stable.

### Regeneration and Staleness

Mark voice artifacts stale when:

- `Screenplay.script`, `scene_breakdown`, or `shot_list` changes.
- Cast binding speaker assignment changes.
- Actor voice profile changes.
- Line text or delivery notes change.
- Target shot duration changes.

Regeneration scopes:

- One line
- One scene
- Entire voice script
- Mix only
- Lip sync only

### Storage and Serving

Add audio-serving helpers equivalent to existing media serving patterns.

Required URL behavior:

- Local paths should be playable by frontend `<audio>`.
- S3 or MinIO keys should resolve through existing storage-backed media routes or presigned URLs.
- Do not expose arbitrary filesystem paths directly.

## Implementation Phases

### Phase 1: Voice Script Foundation

- Add `VoiceScript`, `VoiceLine`, and `VoiceMixArtifact` ORM models.
- Add startup migration/column-add path consistent with existing app style.
- Add Pydantic schemas and API response types.
- Add CRUD endpoints for voice script and voice lines.
- Add tests for create/read/update/delete and locked edit behavior.

Acceptance:

- A production can have a persistent editable voice script.
- Lines can be assigned to narration or cast characters.
- Unresolved speakers are represented as warnings, not server crashes.

### Phase 2: Auto-Generate Voice Script

- Add `VoiceScriptOutput` schema.
- Extend `ScreenwriterService` or add `VoiceScriptService`.
- Generate lines from current screenplay, shot list, and Production Bible bindings.
- Resolve `speaker_tag` to `CastBinding`.
- Add "Voice" tab and "Generate Voice Script" UI.

Acceptance:

- A locked or draft screenplay can generate editable voice lines.
- Dialogue lines bind to cast tags when available.
- Narration lines are supported.
- Generated lines are grouped by scene and shot in UI.

### Phase 3: ElevenLabs Voice Generation

- Extend `AudioAdapter.generate_voice()` options.
- Add line-level and script-level TTS endpoints.
- Generate, store, and serve line audio.
- Add fake audio adapter for tests.
- Add UI line preview and status.

Acceptance:

- One selected dialogue line can generate an MP3 using its bound ElevenLabs voice.
- A whole voice script can generate pending audio lines.
- Audio status updates per line.
- Missing API key, missing voice id, and quota errors surface clearly.

### Phase 4: Voice Mixing

- Add ffmpeg mix helpers for delayed line placement and shot/scene stems.
- Persist `VoiceMixArtifact` rows.
- Add "Mix Voice Layer" endpoint and UI preview.
- Add optional scene/video overlay path.

Acceptance:

- Generated voice lines can be mixed into a shot or scene stem.
- Stem duration is measured and stored.
- A generated scene can be exported with voice audio overlaid.

### Phase 5: Lip Sync Adapter and Local Baseline

- Add `LipSyncAdapter` interface and registry.
- Add `lip_sync_jobs` model.
- Implement local or HTTP Wav2Lip adapter behind the interface.
- Add eligibility detection and skip reasons.
- Add endpoint to queue lip-sync for eligible shots.
- Use lip-synced clip paths in stitcher when available.

Acceptance:

- Single-speaker close-up dialogue shots can produce a lip-synced replacement clip.
- Multi-speaker or no-face shots are skipped with clear reasons.
- Original clip remains available and is never overwritten.

### Phase 6: End-to-End UX and Pipeline Wiring

- Add voice stages to regeneration controls.
- Add Playwright flow for screenplay -> voice script -> fake TTS -> UI preview.
- Add backend integration test using fake audio and fake lip-sync adapters.
- Add docs for prerequisites and provider setup.

Acceptance:

- A test production can generate a screenplay, generate voice script, generate fake audio, and show playable voice lines in the UI.
- Optional lip-sync job can be queued and reflected in status.

## Testing Plan

Backend unit tests:

- Binding resolution from `speaker_tag` to `CastBinding`.
- Voice profile resolution order.
- Voice script structured output validation.
- Stale hash detection for line audio.
- Audio duration parsing wrapper with fixture MP3/WAV.
- Lip-sync eligibility for no face, one face, multiple faces, narrator, and skipped mode.

Backend API tests:

- Create/get voice script.
- Update voice line.
- Generate voice script with fake LLM adapter.
- Generate one line with fake audio adapter.
- Generate all pending lines.
- Mix stem with small fixture audio.
- Queue lip-sync with fake adapter.

Frontend checks:

- Screenplay Voice tab renders empty state.
- Generate Voice Script action populates line table.
- Missing voice warnings display.
- Generated line audio preview appears.
- Lip-sync status column displays ready/skipped/failed.

End-to-end smoke:

- Use fake adapters by default in CI.
- Real ElevenLabs smoke should be opt-in via environment variables and only generate one short line.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| TTS cost grows with long scripts | Generate line-by-line, show estimated character count, require explicit generate-all action |
| Voice lines exceed shot duration | Measure generated duration, warn, allow split/shorten/regenerate |
| Veo native audio conflicts with deterministic TTS | Default native audio off when voice layer is enabled |
| Speaker binding is ambiguous | Require cast tag for dialogue before TTS generation |
| Lip sync fails on multi-character shots | Skip by default unless one visible speaker can be identified |
| Lip-sync model quality varies | Adapter boundary, original clip preservation, line/job-level status |
| Storage URLs differ between local and MinIO/S3 | Use storage-backed serving helpers, do not expose raw local paths |
| Screenplay edits invalidate voice script | Store source screenplay timestamp/hash and surface stale warnings |

## Open Decisions

1. Should voice scripts be one active version per screenplay, or allow named alternatives?
2. Should narration be represented as a special `CastBinding` with role `NARRATOR`, or as production-level voice settings?
3. Should voice generation be available before screenplay lock, or only after lock?
4. Should voice stems replace native video audio or mix on top with ducking?
5. Which lip-sync backend should be installed in the default Docker environment versus configured as an external service?

## Recommended First Slice

Build the smallest useful vertical slice:

1. Add `VoiceScript` and `VoiceLine`.
2. Add CRUD APIs and frontend Voice tab.
3. Add manual line creation with cast binding selection.
4. Generate one ElevenLabs audio file for one line.
5. Show playable audio in the UI.

Then add auto-generation and lip-sync. This avoids coupling the hard lip-sync work to basic voice-script persistence.

## References

- ElevenLabs text-to-speech API: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
- Wav2Lip repository: https://github.com/Rudrabha/Wav2Lip
- MuseTalk repository: https://github.com/TMElyralab/MuseTalk
- LatentSync repository: https://github.com/bytedance/LatentSync
