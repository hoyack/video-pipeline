# API Guide: Creating a Full Production

This guide walks through the complete HTTP API workflow for producing a finished, multi-scene video with narration and sound effects — from an empty database to a downloadable master MP4.

The sequence below mirrors the automated E2E test (`backend/tests/test_nasa_documentary_e2e.py`), which produces output like:

```
3 scenes, 12 complete clips
62.75s final master
12/12 narration lines generated and mixed
7/7 SFX cues generated, bound to bible sound assets, and mixed
```

All endpoints are served under `/api` on the backend port (check `VIDPIPE_SERVER__PORT` in `.env` — examples below use `http://localhost:8100`).

## Concepts

| Entity | What it is |
|--------|------------|
| **Production** | The top-level container — an ordered collection of scenes that render into one master MP4 |
| **Production Bible** | A reusable asset library bound to a production: cast (actors + voices + wardrobe), sets, props, and sound assets |
| **Screenplay** | 1:1 with a production. Logline, treatment, character breakdowns, scene breakdown, script, shot list. Must be `LOCKED` before scenes are generated from it |
| **Scene** | One segment of the timeline. Owns its own storyboard → keyframes → clips → stitched output pipeline, with per-scene model selection |
| **Voice Script** | Narration/dialogue lines derived from the screenplay. Lines are TTS-generated (ElevenLabs) and mixed into per-scene voice stems |
| **Sound Deck** | SFX/ambience cues derived from the production timeline. Cues are generated (ElevenLabs SFX), optionally bound to bible sound assets, and mixed into per-scene SFX stems |
| **Production Master** | The final render: scene videos concatenated, voice + SFX stems mixed in, served as one MP4 |

## Workflow at a glance

```
1. Create Production Bible + Production, link them
2. Build the bible: actors (refs, voice profiles, wardrobe), sets, props → bind to bible → finalize
3. Screenplay: generate (or seed), LOCK, generate scenes from breakdown
4. Per scene: configure models/duration → regenerate all phases → poll to complete
5. Voice script: generate lines → resolve speaker bindings → TTS audio → mix scene stems
6. Sound deck: generate cues → TTS SFX audio → mix scene stems
7. Render master → download MP4
```

## Prerequisites

- Backend running (see README Quick Start / Docker)
- Vertex AI credentials configured (Gemini/Imagen/Veo), or ComfyUI for WAN models
- ElevenLabs API key saved in settings (`PUT /api/settings`) — required for narration and SFX audio
- Check current settings: `GET /api/settings`

---

## Step 1 — Create the Production Bible and Production

```bash
# Create a bible (reusable asset library)
curl -X POST http://localhost:8100/api/production-bibles \
  -H 'Content-Type: application/json' \
  -d '{"name": "NASA Documentary Bible", "category": "CUSTOM"}'
# → { "id": "<bible_id>", "status": "DRAFT", ... }

# Create the production
curl -X POST http://localhost:8100/api/productions \
  -H 'Content-Type: application/json' \
  -d '{"name": "Early NASA Documentary", "description": "Short historical documentary"}'
# → { "id": "<production_id>", ... }

# Link the bible to the production
curl -X PUT http://localhost:8100/api/productions/<production_id> \
  -H 'Content-Type: application/json' \
  -d '{"production_bible_id": "<bible_id>"}'
```

Scenes generated from the screenplay later inherit `production_bible_id` automatically.

## Step 2 — Build the bible: assets and bindings

Assets live in the global **asset library**; **bindings** attach them to a specific bible with a production-unique `tag` (e.g. `NARRATOR`, `KATHERINE_JOHNSON`) that prompts and the voice script reference.

### Actors (cast)

```bash
# Create an actor
curl -X POST http://localhost:8100/api/asset-library/actors \
  -H 'Content-Type: application/json' \
  -d '{"name": "Katherine Johnson", "description": "NASA mathematician, 1960s"}'
# → { "id": "<actor_id>", ... }

# Upload a reference image (drives visual identity in keyframes)
curl -X POST http://localhost:8100/api/asset-library/actors/<actor_id>/refs \
  -F 'file=@katherine_johnson.jpg'

# Add a voice profile (ElevenLabs voice ID — used for DIALOGUE lines)
curl -X POST http://localhost:8100/api/asset-library/actors/<actor_id>/voice-profiles \
  -H 'Content-Type: application/json' \
  -d '{"name": "Default", "provider": "elevenlabs", "voice_id": "<elevenlabs_voice_id>"}'

# Add a wardrobe preset
curl -X POST http://localhost:8100/api/asset-library/actors/<actor_id>/wardrobe-presets \
  -H 'Content-Type: application/json' \
  -d '{"name": "1960s office", "description": "Cardigan, skirt, glasses"}'

# Bind the actor into the bible as a character
curl -X POST http://localhost:8100/api/production-bibles/<bible_id>/cast \
  -H 'Content-Type: application/json' \
  -d '{
    "actor_id": "<actor_id>",
    "tag": "KATHERINE_JOHNSON",
    "character_name": "Katherine Johnson",
    "role": "SUPPORTING",
    "voice_profile_id": "<voice_profile_id>"
  }'
# → { "id": "<cast_binding_id>", ... }

# Add a "look" (wardrobe variant for this production)
curl -X POST http://localhost:8100/api/cast-bindings/<cast_binding_id>/looks \
  -H 'Content-Type: application/json' \
  -d '{"wardrobe_preset_id": "<wardrobe_preset_id>"}'
```

For a documentary, also bind a `NARRATOR` cast entry whose voice profile is your narrator voice — `NARRATION` voice lines resolve to it.

### Sets and props

```bash
curl -X POST http://localhost:8100/api/asset-library/sets \
  -H 'Content-Type: application/json' \
  -d '{"name": "Mercury Control", "description": "1960s mission control room"}'
curl -X POST http://localhost:8100/api/asset-library/sets/<set_id>/refs -F 'file=@mercury_control.jpg'
curl -X POST http://localhost:8100/api/production-bibles/<bible_id>/set-bindings \
  -H 'Content-Type: application/json' \
  -d '{"set_id": "<set_id>", "tag": "MERCURY_CONTROL"}'

# Props are analogous: /api/asset-library/props + /api/production-bibles/{bible_id}/prop-bindings
```

### Sound assets (for SFX cue matching)

Bible sound bindings let the sound deck bind generated cues to curated sound assets by tag:

```bash
curl -X POST http://localhost:8100/api/production-bibles/<bible_id>/sound-bindings \
  -H 'Content-Type: application/json' \
  -d '{"sound_asset_id": "<sound_asset_id>", "tag": "ROCKET_RUMBLE"}'
```

### Finalize the bible

```bash
curl -X POST http://localhost:8100/api/production-bibles/<bible_id>/finalize
```

## Step 3 — Screenplay: generate, lock, create scenes

```bash
# Get (auto-creates in DRAFT if missing)
curl http://localhost:8100/api/productions/<production_id>/screenplay

# Option A: full LLM generation (logline → treatment → breakdowns → script → shot list)
curl -X POST http://localhost:8100/api/productions/<production_id>/screenplay/generate \
  -H 'Content-Type: application/json' \
  -d '{"text_model": "gemini-2.5-flash"}'
# → 202 { "status": "generating" } — poll GET .../screenplay until generating_step is null

# (Individual steps also exist: /generate-logline, /generate-treatment,
#  /generate-character-breakdowns, /generate-scene-breakdown, /generate-script, /generate-shot-list)

# Option B: seed the screenplay directly (deterministic scene count/cost)
curl -X PUT http://localhost:8100/api/productions/<production_id>/screenplay \
  -H 'Content-Type: application/json' \
  -d '{"title": "...", "script": "...", "scene_breakdown": [ ... ]}'

# Lock it (required before generating scenes)
curl -X PATCH http://localhost:8100/api/productions/<production_id>/screenplay/status \
  -H 'Content-Type: application/json' \
  -d '{"status": "LOCKED"}'

# Create draft scenes from the scene_breakdown (force=true replaces existing screenplay scenes)
curl -X POST 'http://localhost:8100/api/productions/<production_id>/screenplay/generate-scenes?force=true'
# → [ { "scene_id": "...", "title": "...", "scene_number": 0 }, ... ]
```

## Step 4 — Configure and generate each scene

For each scene returned by `generate-scenes`:

```bash
# Configure models, duration, shot count
curl -X PATCH http://localhost:8100/api/scenes/<scene_id>/edit \
  -H 'Content-Type: application/json' \
  -d '{
    "clip_duration": 5,
    "target_shot_count": 4,
    "text_model": "gemini-2.5-flash",
    "image_model": "gemini-2.5-flash-image",
    "video_model": "wan-2.2-i2v",
    "audio_enabled": false,
    "production_bible_id": "<bible_id>",
    "commit_message": "Configure models and shot count"
  }'

# Run the full pipeline: storyboard → keyframes → clips → stitch
curl -X POST http://localhost:8100/api/scenes/<scene_id>/regenerate \
  -H 'Content-Type: application/json' \
  -d '{
    "scope": "all_phases",
    "text_model": "gemini-2.5-flash",
    "image_model": "gemini-2.5-flash-image",
    "video_model": "wan-2.2-i2v"
  }'

# Poll until status == "complete"
curl http://localhost:8100/api/scenes/<scene_id>/status
# → { "status": "draft|generating|complete|failed", "progress_percent": ..., "progress_label": ... }
```

Scene generation is a long-running background task — minutes per scene depending on shot count and video model. `GET /api/scenes/{scene_id}` returns full detail including shots, keyframes, and clips. Failed scenes can be resumed with `POST /api/scenes/{scene_id}/resume`.

**Model constraints:**
- Veo durations are discrete: Veo 2 `[5,6,7,8]`s, Veo 3/3.1 `[4,6,8]`s; WAN 2.2 requires 5s clips
- Only Veo 3+ produces native clip audio (`audio_enabled`). WAN clips are silent — narration/SFX are mixed in at the production level (Steps 5–7), so `audio_enabled: false` is typical here
- Aspect ratios for Veo: `16:9` or `9:16` only

## Step 5 — Voice script: narration and dialogue

```bash
# Generate voice lines from the screenplay (NARRATION + DIALOGUE)
curl -X POST http://localhost:8100/api/productions/<production_id>/voice-script/generate \
  -H 'Content-Type: application/json' \
  -d '{"text_model": "gemini-2.5-flash"}'
# → { "voice_script": { "id": "<voice_script_id>", "lines": [...] } }

# Resolve speaker tags to cast bindings / voice profiles
curl -X POST http://localhost:8100/api/voice-scripts/<voice_script_id>/resolve-bindings

# Generate TTS audio for all pending lines (ElevenLabs)
curl -X POST http://localhost:8100/api/voice-scripts/<voice_script_id>/generate-audio
# → lines[].generation_status: PENDING → READY (audio_url playable per line)

# Mix per-scene voice stems (positions lines on each scene's timeline)
curl -X POST http://localhost:8100/api/voice-scripts/<voice_script_id>/mix
# → mix_artifacts[]: SCENE_VOICE_STEM per scene, status READY
```

Useful extras:
- `GET /api/productions/{production_id}/voice-script` — current state (lines, statuses, stems)
- `PATCH /api/voice-lines/{voice_line_id}` — edit a line's text/speaker; `POST /api/voice-lines/{voice_line_id}/generate-audio` — regenerate one line
- `POST /api/voice-scripts/{voice_script_id}/lip-sync` — queue lip-sync jobs for dialogue lines with visible speakers; poll `GET /api/voice-scripts/{voice_script_id}/jobs`

## Step 6 — Sound deck: SFX and ambience

```bash
# Generate cues from the production timeline (AMBIENCE / FOLEY / SFX / MUSIC)
curl -X POST http://localhost:8100/api/productions/<production_id>/sound-deck/generate \
  -H 'Content-Type: application/json' \
  -d '{"text_model": "gemini-2.5-flash"}'
# → { "sound_deck": { "cues": [...] } } — cues matching bible sound-binding tags
#   get sound_asset_id populated automatically

# Generate audio for all pending cues (ElevenLabs SFX)
curl -X POST http://localhost:8100/api/productions/<production_id>/sound-deck/generate-audio

# Mix per-scene SFX stems (cues placed at start_time_seconds with volume_db)
curl -X POST http://localhost:8100/api/productions/<production_id>/sound-deck/mix
# → mix_artifacts[]: SCENE_SFX_STEM per scene, status READY
```

Cue editing: `PATCH /api/sound-cues/{cue_id}` (prompt, timing, volume, bound asset), `DELETE /api/sound-cues/{cue_id}`, `POST /api/sound-cues/{cue_id}/generate-audio` for a single cue.

## Step 7 — Render and download the master

```bash
curl -X POST http://localhost:8100/api/productions/<production_id>/render-master
# → {
#     "production_id": "...",
#     "video_url": "/api/productions/<production_id>/master-video",
#     "scene_count": 3,
#     "duration_seconds": 62.75,
#     "audio_stem_count": 6        // voice + SFX stems mixed in
#   }

# Download the MP4
curl -o master.mp4 http://localhost:8100/api/productions/<production_id>/master-video
```

The renderer concatenates completed scene outputs in production order, positions each scene's voice and SFX stems at its timeline offset, mixes them into a single audio track, and muxes the result. Render is synchronous (typically 30s–2min). Re-running `render-master` overwrites the stored master; `GET /api/productions/{production_id}/master` returns metadata for an existing render without re-rendering.

### Validate the output

```bash
ffprobe -v error -show_entries stream=codec_type -of csv=p=0 master.mp4
# expect: video + audio streams

ffmpeg -i master.mp4 -af volumedetect -f null - 2>&1 | grep mean_volume
# expect: non-silent mean_volume
```

---

## Polling reference

| What | Endpoint | Field(s) | Done when |
|------|----------|----------|-----------|
| Screenplay generation | `GET /api/productions/{id}/screenplay` | `generating_step` | null/empty |
| Scene pipeline | `GET /api/scenes/{id}/status` | `status`, `progress_percent` | `complete` (or `failed`) |
| Voice TTS | `GET /api/productions/{id}/voice-script` | `lines[].generation_status` | all `READY` |
| Voice stems | same | `mix_artifacts[].status` | `READY` per scene |
| SFX TTS | `GET /api/productions/{id}/sound-deck` | `cues[].generation_status` | all `READY` |
| SFX stems | same | `mix_artifacts[].status` | `READY` per scene |
| Lip-sync jobs | `GET /api/voice-scripts/{id}/jobs` | job `status` | all terminal |

## Endpoint reference

### Productions
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/productions` | Create production |
| `GET/PUT/DELETE` | `/api/productions/{id}` | Read / update (incl. `production_bible_id`) / delete |
| `POST` | `/api/productions/{id}/scenes` | Attach existing scenes |
| `POST` | `/api/productions/{id}/render-master` | Render master MP4 |
| `GET` | `/api/productions/{id}/master` | Existing master metadata |
| `GET` | `/api/productions/{id}/master-video` | Download master MP4 |

### Production bible & asset library
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/production-bibles` | Create bible |
| `POST` | `/api/production-bibles/{id}/finalize` | Finalize bible |
| `POST` | `/api/asset-library/actors` / `sets` / `props` | Create library assets |
| `POST` | `/api/asset-library/actors/{id}/refs` | Upload reference image |
| `POST` | `/api/asset-library/actors/{id}/voice-profiles` | Add ElevenLabs voice |
| `POST` | `/api/asset-library/actors/{id}/wardrobe-presets` | Add wardrobe preset |
| `POST` | `/api/production-bibles/{id}/cast` | Bind actor as character (tag) |
| `POST` | `/api/cast-bindings/{id}/looks` | Add wardrobe look |
| `POST` | `/api/production-bibles/{id}/set-bindings` / `prop-bindings` / `sound-bindings` | Bind sets/props/sounds |

### Screenplay
| Method | Path | Purpose |
|--------|------|---------|
| `GET/PUT` | `/api/productions/{id}/screenplay` | Get-or-create / seed manually |
| `PATCH` | `/api/productions/{id}/screenplay/status` | `DRAFT` → `IN_REVIEW` → `LOCKED` |
| `POST` | `/api/productions/{id}/screenplay/generate` | Full LLM chain (async) |
| `POST` | `/api/productions/{id}/screenplay/generate-scenes` | Create scenes from breakdown |

### Scenes
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/scenes` | Create standalone draft scene |
| `GET` | `/api/scenes/{id}` / `/status` | Detail / progress |
| `PATCH` | `/api/scenes/{id}/edit` | Configure models, duration, shots, bible |
| `POST` | `/api/scenes/{id}/generate` / `/regenerate` | Run pipeline (scope: `all_phases`, …) |
| `POST` | `/api/scenes/{id}/resume` / `/stop` | Resume / stop |
| `GET` | `/api/scenes/{id}/download` | Download stitched scene MP4 |
| `POST` | `/api/scenes/{id}/shots/{idx}/regenerate` | Regenerate one shot's clip |

### Voice script
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/productions/{id}/voice-script` | Get-or-create + current state |
| `POST` | `/api/productions/{id}/voice-script/generate` | Generate lines from screenplay |
| `POST` | `/api/voice-scripts/{id}/resolve-bindings` | Match speakers → cast voices |
| `POST` | `/api/voice-scripts/{id}/generate-audio` | TTS all pending lines |
| `POST` | `/api/voice-scripts/{id}/mix` | Build per-scene voice stems |
| `POST` | `/api/voice-scripts/{id}/lip-sync` | Queue lip-sync jobs |
| `PATCH/DELETE` | `/api/voice-lines/{id}` | Edit / delete a line |
| `POST` | `/api/voice-lines/{id}/generate-audio` | TTS one line |
| `GET` | `/api/voice-lines/{id}/audio`, `/api/voice-mix-artifacts/{id}/audio` | Stream audio |

### Sound deck
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/productions/{id}/sound-deck` | Current cues + stems |
| `POST` | `/api/productions/{id}/sound-deck/generate` | Generate cues (LLM) |
| `POST` | `/api/productions/{id}/sound-deck/generate-audio` | Generate all cue audio |
| `POST` | `/api/productions/{id}/sound-deck/mix` | Build per-scene SFX stems |
| `PATCH/DELETE` | `/api/sound-cues/{id}` | Edit / delete a cue |
| `POST` | `/api/sound-cues/{id}/generate-audio` | Generate one cue's audio |
| `GET` | `/api/sound-cues/{id}/audio`, `/api/sound-mix-artifacts/{id}/audio` | Stream audio |

### Settings
| Method | Path | Purpose |
|--------|------|---------|
| `GET/PUT` | `/api/settings` | Global settings incl. ElevenLabs API key, default models |
| `GET` | `/api/settings/models` | Allowed model lists |

## Notes

- **Async vs sync:** Scene generation and screenplay generation run as background tasks (poll). Voice/sound generation, mixing, and master render are synchronous calls (use generous client timeouts — TTS for a full script can take minutes; master render 30s–2min).
- **Idempotency:** `GET .../screenplay` and `GET .../voice-script` are get-or-create. Mix and render-master can be re-run safely; they rebuild artifacts from the latest ready inputs.
- **Error recovery:** Scene failures persist to `error_message` and are resumable. Individual voice lines / sound cues that fail can be retried with their single-item `generate-audio` endpoints without redoing the rest.
