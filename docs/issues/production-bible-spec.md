# vidpipe — Production Bible & Asset Pipeline Specification

**Version:** 0.1 (Draft)
**Date:** March 5, 2026
**Status:** Alpha Scoping

---

## 1. Problem Statement

The current Production Bible workflow requires uploading a video clip to bootstrap asset extraction (via ResNet + YOLO). While this automated path is useful, it creates several problems:

- Characters are duplicated per-pose rather than consolidated into a single identity.
- There is no way to manually define characters, sets, props, or sound assets without first uploading video.
- Assets are tightly coupled to the Production Bible they were extracted into — they cannot be reused across productions.
- There is no distinction between an **actor** (a reusable identity with appearance, voice, etc.) and a **character** (a role in a specific production that an actor is cast into).
- Scene prompts have no structured way to reference Production Bible assets by name or tag.

This spec defines a revised asset model that supports manual creation, standalone reuse, binding into Production Bibles, and integration into Scenes via the Screenplay pipeline.

---

## 2. Core Concepts

### 2.1 Asset Types

There are four top-level asset categories, each of which can exist independently and be bound into one or more Production Bibles.

| Asset Type | Description | Standalone Entity |
|---|---|---|
| **Actor** | A reusable identity — a real person or a fabricated persona with appearance refs, voice profiles, and base description. | Yes |
| **Set** | A reusable environment/location with visual references, reverse prompts, lighting notes. | Yes |
| **Prop** | A reusable object or item with visual references and description. | Yes |
| **Sound Asset** | A reusable audio element — score theme, SFX clip, ambience loop. | Yes |

### 2.2 Actor vs. Character

This is the critical distinction the current system lacks.

**Actor** — A persistent, reusable identity that lives in a global asset library. An actor has:

- Name and description
- One or more **appearance references** (uploaded images)
- A **base appearance prompt** (text description for image generation)
- One or more **voice profiles** (each with a Voice ID, adapter type, style notes)
- Optional wardrobe presets (default looks)
- Prompt tags (e.g., `ACTOR_BRANDON`, `ACTOR_JANE`)

**Character** — A role within a specific Production Bible. A character is created by **casting** an actor into a role. A character adds production-specific context on top of the actor:

- Character name (e.g., "Detective Morrow")
- Character arc / description
- Wardrobe overrides (what this character wears in this production)
- Voice profile selection (which of the actor's voice profiles to use)
- Behavioral notes (how this character acts, moves, speaks)
- Prompt tags for this production (e.g., `CHAR_DETECTIVE`)

A single actor can be cast as different characters across different Production Bibles. The same actor could play "Detective Morrow" in one production and "Uncle Ray" in another.

### 2.3 Binding

**Binding** is the act of associating a standalone asset with a Production Bible. Once bound:

- The Production Bible holds a reference to the standalone asset (not a copy).
- Production-specific overrides can be layered on top (e.g., character wardrobe, set lighting adjustments).
- Changes to the underlying actor/set/prop propagate to all Production Bibles that reference them, unless overridden.

### 2.4 Production Bible

A Production Bible is a **composed collection** of bound assets, organized into departments:

- **Casting** — Characters (actors cast into roles)
- **Art Department** — Sets and Props bound to this production
- **Sound** — Sound assets (score themes, SFX, ambience) bound to this production

A Production Bible can be associated with one or more Productions, and its assets flow into the Scenes of those Productions.

---

## 3. Data Model

### 3.1 Actor

```
Actor {
  id: UUID
  name: string
  description: text
  base_appearance_prompt: text
  prompt_tags: string[]           // e.g., ["ACTOR_BRANDON"]
  appearance_refs: ActorRef[]     // uploaded images
  voice_profiles: VoiceProfile[]
  wardrobe_presets: WardrobeItem[]
  created_at: datetime
  updated_at: datetime
}

ActorRef {
  id: UUID
  actor_id: UUID
  image_url: string
  label: string                   // e.g., "front", "profile", "3/4"
  is_primary: boolean
}

VoiceProfile {
  id: UUID
  actor_id: UUID
  label: string                   // e.g., "Default", "Whisper", "Narrator"
  voice_id: string                // ElevenLabs voice ID or similar
  adapter_type: enum              // ELEVENLABS | BARK | XTTS | CUSTOM
  style_notes: text
  sample_url: string?             // optional generated sample
}

WardrobeItem {
  id: UUID
  label: string                   // e.g., "Casual", "Formal", "Combat Gear"
  description: text
  reference_images: string[]
}
```

### 3.2 Set

```
Set {
  id: UUID
  name: string
  description: text
  reverse_prompt: text            // auto-generated or manual visual description
  style_tags: string[]            // e.g., ["ENV_WAREHOUSE"]
  prompt_tags: string[]
  reference_images: SetRef[]
  lighting_notes: text
  sonic_identity: text?           // ambient sound description
  created_at: datetime
  updated_at: datetime
}

SetRef {
  id: UUID
  set_id: UUID
  image_url: string
  label: string
}
```

### 3.3 Prop

```
Prop {
  id: UUID
  name: string
  description: text
  appearance_prompt: text
  prompt_tags: string[]
  reference_images: PropRef[]
  created_at: datetime
  updated_at: datetime
}

PropRef {
  id: UUID
  prop_id: UUID
  image_url: string
  label: string
}
```

### 3.4 Sound Asset

```
SoundAsset {
  id: UUID
  name: string
  category: enum                  // SCORE_THEME | SFX | AMBIENCE | FOLEY | UI
  subcategory: string?            // e.g., "IMPACT", "MECHANICAL", "NATURAL"
  description: text
  audio_url: string?              // uploaded or generated clip
  generation_prompt: text?        // for AI-generated audio
  tags: string[]
  created_at: datetime
  updated_at: datetime
}
```

### 3.5 Production Bible

```
ProductionBible {
  id: UUID
  name: string
  description: text
  category: enum                  // FULL_PRODUCTION | CHARACTERS | CUSTOM | etc.
  tags: string[]
  status: enum                    // DRAFT | READY
  cast: CastBinding[]
  sets: SetBinding[]
  props: PropBinding[]
  sound: SoundBinding[]
  created_at: datetime
  updated_at: datetime
}
```

### 3.6 Bindings (the glue)

Each binding links a standalone asset into a Production Bible, with optional production-specific overrides.

```
CastBinding {
  id: UUID
  bible_id: UUID
  actor_id: UUID                  // reference to the standalone Actor
  character_name: string          // role name in this production
  character_description: text
  character_arc: text
  role: enum                      // LEAD | SUPPORTING | EXTRA | NARRATOR
  wardrobe_override: WardrobeItem[]   // production-specific costume
  voice_profile_id: UUID?         // which of the actor's voice profiles to use
  behavioral_notes: text
  prompt_tags: string[]           // e.g., ["CHAR_DETECTIVE"]
}

SetBinding {
  id: UUID
  bible_id: UUID
  set_id: UUID                    // reference to the standalone Set
  production_name: string?        // optional rename for this production
  lighting_override: text?
  sonic_override: text?
  prompt_tags: string[]
}

PropBinding {
  id: UUID
  bible_id: UUID
  prop_id: UUID
  production_name: string?
  notes: text?
  prompt_tags: string[]
}

SoundBinding {
  id: UUID
  bible_id: UUID
  sound_asset_id: UUID
  usage_notes: text?              // e.g., "Use for chase sequences"
  prompt_tags: string[]
}
```

---

## 4. Workflows

### 4.1 Manual Asset Creation (New Primary Path)

**Creating an Actor:**

1. Navigate to a new **Asset Library** (global, outside Production Bibles).
2. Select "New Actor."
3. Fill in name, description, base appearance prompt.
4. Upload appearance reference images (multi-angle).
5. Add one or more voice profiles (Voice ID, adapter, style notes). Can generate sample.
6. Optionally add wardrobe presets.
7. Assign prompt tags.
8. Save. The actor now exists in the library, unattached to any production.

**Creating a Set, Prop, or Sound Asset** follows the same pattern — navigate to the library, create, fill in fields, save.

### 4.2 Automated Asset Extraction (Existing Path, Revised)

1. Upload a video clip to either the Asset Library or directly to a Production Bible.
2. The system runs ResNet + YOLO extraction pipeline as it does today.
3. **New step:** Extracted faces are grouped by visual similarity and presented as **candidate actors** (not auto-committed).
4. The user reviews candidates, merges duplicates, names them, and promotes them to the Actor library.
5. Extracted backgrounds become candidate Sets; identified objects become candidate Props.
6. The user reviews, edits, and promotes each to the library.
7. If done within a Production Bible, promoted assets are auto-bound to that Bible.

### 4.3 Binding Assets to a Production Bible

1. Open a Production Bible (new or existing).
2. In the Casting tab, click "Add Character."
3. A picker opens showing all Actors in the library. Select one.
4. Fill in the character-specific fields: character name, arc, wardrobe override, voice profile selection, behavioral notes.
5. Save the binding. The character now appears in the Casting list.
6. Repeat for Sets (Art Department → Sets), Props (Art Department → Props), and Sound Assets (Sound).

### 4.4 New Production Bible Creation (Revised View)

The current "New Production Bible" view only has Name, Description, Category, Tags, and a video upload zone. It needs to be expanded:

**Proposed layout:**

```
┌─────────────────────────────────────────────────────┐
│ New Production Bible                                │
│                                                     │
│ Name*: [________________________]                   │
│ Description: [__________________]                   │
│ Category: [CUSTOM ▼]    Tags: [_________________]   │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Import from Source (optional)                        │
│                                                     │
│ ┌─ Upload Video ──────────────────────────────────┐ │
│ │  Drag and drop images or video here             │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─ Import from Scene ─────────────────────────────┐ │
│ │  Scene ID (UUID): [____________] [Import]       │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Casting                                    [+ Add]  │
│ ┌───────────────────────────────────────────────┐   │
│ │ No characters yet. Add from Actor Library or  │   │
│ │ import from a video/scene above.              │   │
│ └───────────────────────────────────────────────┘   │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Art Department                                      │
│                                                     │
│ Sets                                       [+ Add]  │
│ ┌───────────────────────────────────────────────┐   │
│ │ No sets yet.                                  │   │
│ └───────────────────────────────────────────────┘   │
│                                                     │
│ Props                                      [+ Add]  │
│ ┌───────────────────────────────────────────────┐   │
│ │ No props yet.                                 │   │
│ └───────────────────────────────────────────────┘   │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Sound                                               │
│                                                     │
│ Score Themes                               [+ Add]  │
│ ┌───────────────────────────────────────────────┐   │
│ │ No score themes yet.                          │   │
│ └───────────────────────────────────────────────┘   │
│                                                     │
│ SFX Library                                [+ Add]  │
│ ┌───────────────────────────────────────────────┐   │
│ │ No SFX items yet.                             │   │
│ └───────────────────────────────────────────────┘   │
│                                                     │
├─────────────────────────────────────────────────────┤
│                           [Cancel]  [Save Draft]    │
└─────────────────────────────────────────────────────┘
```

Each "+ Add" button opens a picker that browses the global Asset Library and allows selecting existing assets or creating new ones inline.

### 4.5 Scene Integration via Screenplay

This is how Production Bible assets flow into scene generation.

**Association:** A Production (which contains Scenes) is linked to one or more Production Bibles. When a Scene belongs to that Production, it inherits access to all bound assets.

**Prompt tagging in Scenes:** When writing or editing a scene prompt (or screenplay shot descriptions), the user can reference characters, sets, and props by their prompt tags or names.

Example scene prompt:
```
In a candlelit throne room [SET:THRONE_ROOM], an aging king [CHAR:KING_ALDRIC]
grips his son's arm [CHAR:PRINCE_TOREN], eyes fierce with urgency, as golden
light flickers across ancient stone walls.
```

**Tag resolution at generation time:** When the pipeline processes a shot, it resolves tags against the Production Bible:

1. `[CHAR:KING_ALDRIC]` → looks up CastBinding with tag `KING_ALDRIC` → finds the bound Actor → injects the actor's base appearance prompt + the character's wardrobe override into the image generation prompt.
2. `[SET:THRONE_ROOM]` → looks up SetBinding with tag `THRONE_ROOM` → injects the set's reverse prompt and lighting notes into the image generation prompt.
3. For audio pipeline: `[CHAR:KING_ALDRIC]` → resolves to the selected voice profile for dialogue generation.

**Scene Edit View changes:**

The existing Scene edit view already has a "Production Bible" selector (to attach reference images). This should be extended to:

- Show all assets from the attached Production Bible(s) as browsable/searchable.
- Provide autocomplete for `[CHAR:...]`, `[SET:...]`, `[PROP:...]` tags while editing prompts.
- Display a resolved preview showing what prompts will actually be injected at generation time.

---

## 5. Asset Library (New Top-Level Section)

A new navigation item is needed: **Asset Library** (or it could live under an "Assets" section). This is the global pool of standalone assets, independent of any Production Bible.

### 5.1 Library Views

```
Asset Library
├── Actors          (list/grid of all actors, with search + filter)
├── Sets            (list/grid of all sets)
├── Props           (list/grid of all props)
└── Sound Assets    (list/grid, filterable by category)
```

Each asset type has its own creation form, detail/edit view, and a listing page showing where it's currently bound (which Production Bibles reference it).

### 5.2 Asset Detail View (Actor Example)

```
┌─────────────────────────────────────────────────────┐
│ Actor: Brandon                                      │
│ Tags: ACTOR_BRANDON                                 │
│                                                     │
│ [Overview] [Appearance Refs] [Voice Profiles]       │
│ [Wardrobe Presets] [Usage]                          │
│                                                     │
│ ── Overview ──────────────────────────────────────── │
│ Description: [editable text field]                  │
│ Base Appearance: [editable text field]              │
│                                                     │
│ ── Bound In ──────────────────────────────────────── │
│ • "Cyberpunk One" Bible → as "Detective Morrow"     │
│ • "Brandon Manifest" Bible → as "Brandon (Self)"    │
│                                                     │
│                          [Edit]  [Duplicate]        │
└─────────────────────────────────────────────────────┘
```

---

## 6. Audio Pipeline Integration (Future Phase)

Once the Production Bible is structured with proper character voice profiles and sound assets, the audio pipeline can be integrated into Scenes:

1. **Dialogue generation:** Screenplay shot descriptions that include dialogue → routed to TTS using the character's selected voice profile.
2. **SFX placement:** Screenplay transition notes or shot descriptions referencing sound tags → matched against bound Sound Assets.
3. **Score/ambience:** Set sonic identity + bound score themes → used to generate or select background audio for each shot.
4. **Mix:** Final audio mix stitched alongside the video stitch step.

The audio pipeline phases would extend the existing Scene pipeline:

```
Storyboard → Keyframes → Video Gen → Stitch → Audio Gen → Audio Mix → Complete
```

This is out of scope for the current spec but is the motivating use case.

---

## 7. Migration Path

### 7.1 Existing Production Bibles

Production Bibles created via the old video-upload flow should continue to work. The migration path:

- Existing extracted characters (e.g., "Frame 1 - person") remain as-is in their Bible.
- A "Promote to Actor Library" action is added to each character, allowing users to extract them into standalone Actors and convert the Bible entry into a proper CastBinding.
- Sets and props get the same treatment.

### 7.2 Existing Scenes

Scenes that already reference Production Bibles via the attachment system continue to work without changes. The new tag-based resolution is additive — old-style scenes that don't use tags still function normally.

---

## 8. UI Changes Summary

| Area | Current State | Proposed Change |
|---|---|---|
| **Navigation** | Dashboard, Productions, Scenes, Production Bibles, Settings | Add **Asset Library** as new top-level nav item |
| **New Production Bible** | Name, description, category, tags, video upload, scene import | Add Casting, Art Department, Sound sections with library pickers |
| **Production Bible Edit** | Casting/Art/Sound tabs with extracted assets | Add "Bind from Library" option alongside extracted assets; add "Promote to Library" for extracted assets |
| **Scene Edit — Prompt** | Plain text prompt field | Add tag autocomplete for `[CHAR:...]`, `[SET:...]`, `[PROP:...]` referencing the attached Bible |
| **Scene Edit — Production Bible** | Simple Bible selector | Show browsable asset summary from attached Bible(s) |
| **Scene View — Pipeline** | Storyboard → Keyframes → Video Gen → Stitch → Complete | (Future) Extend with Audio Gen → Audio Mix |
| **Asset Library** | Does not exist | New section: Actors, Sets, Props, Sound Assets with CRUD, search, filter |

---

## 9. Open Questions

1. **Should assets be version-controlled?** If an actor's appearance prompt changes after being bound into three Production Bibles, should those Bibles see the change or pin to a version?
2. **Tag syntax:** Is `[CHAR:TAG]` the right syntax, or should it use `@TAG` or `{{TAG}}` or something else that plays well with prompt engineering?
3. **Multiple Production Bibles per Production:** Should a Production support multiple Bibles (e.g., one for characters, one for environments), or should everything be consolidated into one?
4. **Inline asset creation during Bible setup:** When adding a character to a new Bible and no suitable actor exists yet, should the user be able to create the actor inline, or should they always go to the Asset Library first?
5. **Audio pipeline scope:** How much of the audio pipeline needs to be functional before shipping the revised Production Bible? Can we ship the Bible restructure first and add audio later?
6. **Prop extraction accuracy:** The current YOLO pipeline for props — is it worth keeping, or should prop creation be manual-only for now?