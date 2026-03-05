# Phase 22: Asset Library & Actor-Character Model - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning
**Source:** PRD Express Path (docs/issues/production-bible-spec.md)

<domain>
## Phase Boundary

This phase introduces a global Asset Library with standalone entity types (Actor, Set, Prop, Sound Asset) that exist independently of any Production Bible. It replaces the current model where assets are tightly coupled to Production Bibles with a composable architecture:

- **Actors** become persistent, reusable identities with appearance refs, voice profiles, and wardrobe presets
- **Characters** become bindings of Actors into Production Bible roles (casting)
- **Sets, Props, Sound Assets** become standalone library entities that can be bound into multiple Production Bibles
- A **binding system** connects library assets to Production Bibles with production-specific overrides
- **Scene tag resolution** maps `[CHAR:TAG]`, `[SET:TAG]`, `[PROP:TAG]` to bound assets at generation time

This phase does NOT include the audio pipeline (dialogue generation, SFX placement, score/ambience generation, audio mix) — that is explicitly deferred as a future phase per the PRD.

</domain>

<decisions>
## Implementation Decisions

### Data Model — Actor
- Actor is a standalone entity with: id, name, description, base_appearance_prompt, prompt_tags[], appearance_refs (ActorRef[]), voice_profiles (VoiceProfile[]), wardrobe_presets (WardrobeItem[])
- ActorRef stores image_url, label (front/profile/3-4), is_primary flag
- VoiceProfile stores voice_id, adapter_type (ELEVENLABS|BARK|XTTS|CUSTOM), style_notes, sample_url
- WardrobeItem stores label, description, reference_images[]

### Data Model — Set, Prop, Sound Asset
- Set: name, description, reverse_prompt, style_tags[], prompt_tags[], reference_images (SetRef[]), lighting_notes, sonic_identity
- Prop: name, description, appearance_prompt, prompt_tags[], reference_images (PropRef[])
- Sound Asset: name, category (SCORE_THEME|SFX|AMBIENCE|FOLEY|UI), subcategory, description, audio_url, generation_prompt, tags[]

### Data Model — Bindings
- CastBinding: bible_id, actor_id, character_name, character_description, character_arc, role (LEAD|SUPPORTING|EXTRA|NARRATOR), wardrobe_override[], voice_profile_id, behavioral_notes, prompt_tags[]
- SetBinding: bible_id, set_id, production_name, lighting_override, sonic_override, prompt_tags[]
- PropBinding: bible_id, prop_id, production_name, notes, prompt_tags[]
- SoundBinding: bible_id, sound_asset_id, usage_notes, prompt_tags[]

### Data Model — Production Bible
- ProductionBible gains: cast (CastBinding[]), sets (SetBinding[]), props (PropBinding[]), sound (SoundBinding[])
- Bindings hold references (not copies) — changes to underlying assets propagate unless overridden

### Asset Library UI
- New top-level navigation item: Asset Library
- Sub-sections: Actors, Sets, Props, Sound Assets — each with list/grid view, search, filter
- Each asset type has creation form, detail/edit view, and usage tracking (which Production Bibles reference it)

### Production Bible Creation — Revised
- New Production Bible view adds Casting, Art Department (Sets + Props), and Sound sections
- Each section has "+ Add" button that opens a picker browsing the global Asset Library
- Picker supports selecting existing assets or creating new ones inline

### Scene Tag Integration
- Tag syntax: `[CHAR:TAG]`, `[SET:TAG]`, `[PROP:TAG]` in scene prompts
- Tag resolution at generation time: CHAR → CastBinding → Actor appearance + wardrobe override; SET → SetBinding → reverse prompt + lighting; CHAR → VoiceProfile for audio
- Scene edit view: autocomplete for tags, browsable asset summary from attached Bible(s), resolved preview

### Automated Extraction (Revised Path)
- Extracted faces grouped by visual similarity → presented as candidate actors (not auto-committed)
- User reviews, merges duplicates, names, promotes to Actor Library
- Extracted backgrounds → candidate Sets; objects → candidate Props
- Assets promoted from within a Production Bible are auto-bound to that Bible

### Migration
- Existing Production Bible characters remain as-is
- "Promote to Actor Library" action added to each existing character/set/prop
- Promotion converts Bible entry into proper binding referencing new standalone asset
- Existing scenes without tags continue to work (tag resolution is additive)

### Claude's Discretion
- Database migration strategy (ALTER TABLE vs new tables with migration service)
- Whether to reuse existing Character/Set/Prop ORM models from Phase 17 or create new Actor-level models alongside them
- API route organization for Asset Library endpoints
- Frontend routing structure for Asset Library views
- How to handle the relationship between Phase 17's Character model and the new Actor + CastBinding model
- Tag autocomplete implementation details (debounce, matching strategy)
- Asset Library search/filter implementation
- Inline asset creation UX in Production Bible picker

</decisions>

<specifics>
## Specific Ideas

### Actor Detail View (from PRD)
- Tabs: Overview, Appearance Refs, Voice Profiles, Wardrobe Presets, Usage
- "Bound In" section shows all Production Bibles referencing this actor with their character names
- Edit and Duplicate actions

### Production Bible Layout (from PRD)
- Sections: Import from Source (video upload + scene import), Casting (+Add), Art Department — Sets (+Add) / Props (+Add), Sound — Score Themes (+Add) / SFX Library (+Add)
- Cancel and Save Draft buttons

### Tag Resolution Example
```
In a candlelit throne room [SET:THRONE_ROOM], an aging king [CHAR:KING_ALDRIC]
grips his son's arm [CHAR:PRINCE_TOREN]...
```
- `[CHAR:KING_ALDRIC]` → CastBinding → Actor base_appearance_prompt + wardrobe_override → injected into image gen prompt
- `[SET:THRONE_ROOM]` → SetBinding → reverse_prompt + lighting_notes → injected into image gen prompt
- For audio: `[CHAR:KING_ALDRIC]` → voice_profile_id → VoiceProfile → voice_id for TTS

</specifics>

<deferred>
## Deferred Ideas

- **Audio pipeline integration** — dialogue generation, SFX placement, score/ambience, audio mix (PRD Section 6, explicitly out of scope)
- **Asset versioning** — whether bound Production Bibles see changes or pin to a version (PRD Open Question 1)
- **Tag syntax finalization** — `[CHAR:TAG]` vs `@TAG` vs `{{TAG}}` (PRD Open Question 2, using `[CHAR:TAG]` as default)
- **Multiple Production Bibles per Production** — consolidation vs separation (PRD Open Question 3)
- **Prop extraction accuracy** — whether to keep YOLO prop pipeline or go manual-only (PRD Open Question 6)

</deferred>

---

*Phase: 22-asset-library-actor-character-model*
*Context gathered: 2026-03-05 via PRD Express Path*
