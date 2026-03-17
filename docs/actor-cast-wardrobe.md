# Actor Wardrobe Image Generation & Multi-Tag Cast Looks

## Overview

Two interconnected features that give screenwriters control over what actors **wear** in each scene:

1. **Wardrobe Image Generation** — Generate reference images for a wardrobe preset using the actor's face ref + the wardrobe description as prompt, producing images of "the same person wearing X outfit."
2. **Multi-Tag Cast Looks** — Let a CastBinding have multiple "look" tags (each tied to a wardrobe preset), so screenplays can call `@BRANDON_PIRATE` to pull pirate-outfit ref images and `@BRANDON_FORMAL` for formal-outfit refs.

### User Workflow

1. **Actor setup:** Create actor, upload face reference photos, write `base_appearance_prompt`
2. **Wardrobe creation:** Add wardrobe preset "Pirate Captain" with description "weathered leather tricorn hat, long naval coat, brass buttons"
3. **Image generation:** Click "Generate Image" on the preset → select face ref → AI generates the same person in pirate costume
4. **Cast binding:** Bind actor to production bible as `@BRANDON`
5. **Add look:** Create CastLook `@BRANDON_PIRATE` tied to the "Pirate Captain" wardrobe preset
6. **Screenplay:** Screenwriter uses `@BRANDON_PIRATE` in scene prompts → tag resolver returns wardrobe-specific ref images + description
7. **Keyframe generation:** Pipeline uses wardrobe ref images for visual conditioning and wardrobe description for prompt text

---

## Feature A: Wardrobe Image Generation

### Endpoints

**`POST /api/asset-library/actor-wardrobe-presets/{preset_id}/generate-image`**

| Field | Type | Notes |
|-------|------|-------|
| `reference_image_id` | `str` (required) | ActorRef ID for face/identity source |
| `image_model` | `str?` | Model override |
| `additional_prompt` | `str?` | Extra prompt details |

Logic: Load preset description + actor's `base_appearance_prompt` + face ref bytes → compose identity-preserving prompt → generate image → save to `actors/{actor_id}/wardrobe/{preset_id}/` → append URL to preset's `reference_images` JSON.

**`GET /api/asset-library/actor-wardrobe-presets/{preset_id}/images/{index}`**

Serves wardrobe ref images by index from `reference_images` list.

**`DELETE /api/asset-library/actor-wardrobe-presets/{preset_id}/images/{index}`**

Removes image from storage + `reference_images` list.

---

## Feature B: Multi-Tag Cast Looks (CastLook Model)

### Data Model

```
CastLook
  id: UUID (PK)
  cast_binding_id: UUID (FK → CastBinding, NOT NULL)
  wardrobe_preset_id: UUID (FK → ActorWardrobePreset, NOT NULL)
  tag: str (NOT NULL)  -- e.g. "BRANDON_PIRATE"
  is_default: bool (default False)
  created_at: datetime
```

Constraints:
- `UNIQUE(cast_binding_id, tag)` — no duplicate tags within a binding
- Bible-wide tag uniqueness enforced at application level

Semantics:
- `CastBinding.tag` = default/base look (uses ActorRef images + `base_appearance_prompt`)
- Each `CastLook` = wardrobe-specific look (uses wardrobe preset's `reference_images` + `description`)
- `is_default=True` → CastBinding.tag resolves to wardrobe preset instead of base ActorRefs

### Endpoints

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/api/cast-bindings/{id}/looks` | GET | List looks |
| `/api/cast-bindings/{id}/looks` | POST | Add look (tag + wardrobe_preset_id) |
| `/api/cast-looks/{look_id}` | PUT | Update |
| `/api/cast-looks/{look_id}` | DELETE | Remove |

---

## Tag Resolver Changes

In `resolve_tags_with_assets()`:

1. Load CastLooks for all cast bindings (batch query)
2. Load referenced wardrobe presets (batch query)
3. Check `looks_by_tag` FIRST, then `cast_by_tag` (existing priority)
4. CastLook resolution uses wardrobe preset's `reference_images` for visual conditioning and `description` for prompt text (combined with `base_appearance_prompt` for facial features)
5. When `is_default=True` CastLook exists, the CastBinding's own tag resolves to wardrobe preset refs

---

## Screenwriter Integration

`format_binding_registry()` includes CastLook tags:

```
[@BRANDON] "Brandon Cross" (CHARACTER — default look)
  Tall man, dark hair, defined jawline...

[@BRANDON_PIRATE] "Brandon Cross — Pirate Captain" (CHARACTER — wardrobe look)
  Tall man, dark hair, defined jawline...
  Wearing: weathered leather tricorn hat, long naval coat...
```

---

## Migration

- **SQLite:** `CREATE TABLE IF NOT EXISTS cast_looks` in `_run_migrations()`
- **PostgreSQL:** Auto-created via `create_all()` on startup
- No changes to existing tables
