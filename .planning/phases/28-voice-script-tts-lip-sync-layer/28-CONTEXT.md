# Phase 28: Voice Script, TTS, and Lip Sync Layer - Context

**Gathered:** 2026-06-07
**Status:** Ready for planning
**Source:** PRD Express Path (`docs/voice-plan.md`)

<domain>
## Phase Boundary

This phase adds a first-class voice layer to Productions and Screenplays. It delivers durable voice-script persistence, Production Bible cast voice binding, ElevenLabs line generation, voice audio preview in the Screenplay UI, ffmpeg-based voice mixing, and an optional lip-sync job abstraction for post-video character dialogue sync.

This phase builds on existing Screenplay, Asset Library, binding, ElevenLabs, and ffmpeg infrastructure. It does not replace the current storyboard `ShotAudioManifest` or Veo native audio prompt flow; it adds deterministic generated voice artifacts beside it.

In scope:

- `VoiceScript`, `VoiceLine`, `VoiceMixArtifact`, and `LipSyncJob` persistence.
- API routes for voice-script CRUD, line editing, binding resolution, ElevenLabs generation, voice mixing, and lip-sync job dispatch/status.
- Structured voice-script generation from existing Screenplay data and Production Bible bindings.
- Screenplay "Voice" tab for line review, speaker binding, audio preview, warnings, and job status.
- Audio adapter option expansion needed for ElevenLabs line generation.
- ffmpeg helpers for duration measurement, line stem composition, and non-destructive video audio overlay.
- Lip-sync adapter interface and baseline no-op/fake/local-adapter boundary with eligibility checks.

Out of scope:

- Installing heavyweight lip-sync model weights in the default app image.
- Solving multi-speaker face assignment in crowded shots.
- Full ADR editing timeline with waveform editing.
- Provider-specific ElevenLabs voice design or cloning flows.
- Replacing score themes, SFX, or sonic identity systems.

</domain>

<decisions>
## Locked Decisions

### Source PRD

- `docs/voice-plan.md` is the source PRD for this phase.
- The implementation should extend the existing Screenplay and Production Bible binding system rather than introducing a parallel story format.
- Narration and character dialogue are the v1 line types.
- Dialogue lines should bind to `CastBinding` rows where possible.
- `CastBinding.voice_profile_id` is the primary production-specific character voice anchor.
- `ActorVoiceProfile` is the reusable actor-library voice source.
- `VoiceProfile` remains a fallback for legacy/bible-scoped character voices.
- ElevenLabs is the initial TTS provider through the existing `AudioAdapter` boundary.
- Generated line audio and mix artifacts must be persisted and playable in the UI.
- Voice mixing and lip sync must preserve original clips and write replacement artifacts separately.
- Lip sync must sit behind an adapter boundary, not hard-code Wav2Lip, MuseTalk, or LatentSync into domain models.

### Implementation Direction

- Prefer one active `VoiceScript` per `Screenplay` in v1, with `version` and stale-source tracking.
- Store spoken content in normalized `VoiceLine` rows rather than nested JSON on `Screenplay`.
- Generate line audio line-by-line with independent commits so long runs survive interruption.
- Use line-level status fields and warnings instead of failing the entire voice script on one missing voice or provider error.
- Default to deterministic voice layering when voice script is enabled; warn if Veo native audio is also enabled.
- Implement lip-sync eligibility before model invocation and record skip reasons.

### Claude's Discretion

- Exact table column names where they can remain compatible with the PRD semantics.
- Whether `VoiceScriptService` lives in a new service file or alongside screenwriter helpers.
- Exact UI layout for the Voice tab, as long as the line workflow is complete and consistent with existing ScreenplayEditor patterns.
- Whether the first lip-sync adapter is a fake/no-op adapter for tests plus configurable external command/HTTP adapter for runtime.
- Exact ffmpeg filter graph, provided output is deterministic and original files are preserved.

</decisions>

<specifics>
## Existing Code Anchors

- `backend/vidpipe/db/models.py`
  - `Screenplay`
  - `Scene`
  - `Shot`
  - `VideoClip`
  - `ShotAudioManifest`
  - `CastBinding`
  - `ActorVoiceProfile`
  - `VoiceProfile`
- `backend/vidpipe/api/screenplay.py`
  - Screenplay CRUD and generate endpoints.
- `backend/vidpipe/services/screenwriter.py`
  - Existing six-step ScreenwriterService generation chain.
- `backend/vidpipe/api/audio_gen.py`
  - Existing ElevenLabs key lookup, voice search, character voice sample, SFX generation, and `_store_audio`.
- `backend/vidpipe/services/audio/base.py`
  - `AudioAdapter.generate_voice(...)`.
- `backend/vidpipe/services/audio/elevenlabs_adapter.py`
  - Existing ElevenLabs TTS implementation.
- `backend/vidpipe/pipeline/stitcher.py`
  - Existing ffmpeg concat/crossfade pattern.
- `frontend/src/components/ScreenplayEditor.tsx`
  - Existing Screenplay tabs and regeneration controls.
- `frontend/src/components/AudioPlayer.tsx`
  - Existing audio preview component.
- `frontend/src/api/client.ts` and `frontend/src/api/types.ts`
  - API client/type extension points.

## Requirement Mapping

| Requirement | Delivery |
|-------------|----------|
| VOICE-01 | `VoiceScript`, `VoiceLine`, CRUD, stale tracking |
| VOICE-02 | speaker and voice profile resolution |
| VOICE-03 | structured voice-script generation |
| VOICE-04 | Screenplay Voice tab |
| VOICE-05 | ElevenLabs line/script TTS |
| VOICE-06 | mix artifacts and ffmpeg overlay |
| VOICE-07 | lip-sync adapter and jobs |
| VOICE-08 | UI/API status and warnings |

</specifics>

<deferred>
## Deferred Ideas

- Named alternate voice-script versions.
- Waveform timeline editing and drag-to-time placement.
- Multi-speaker lip sync with face identity hints.
- Default Docker packaging for Wav2Lip/MuseTalk/LatentSync model weights.
- Advanced voice cloning and voice-design workflows.
- Automated shot extension when generated voice exceeds shot duration.

</deferred>

---

*Phase: 28-voice-script-tts-lip-sync-layer*
*Context gathered: 2026-06-07 via PRD Express Path*
