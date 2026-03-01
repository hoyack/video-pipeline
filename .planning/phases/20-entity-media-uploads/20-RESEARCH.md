# Phase 20: Entity Media Uploads - Research

**Researched:** 2026-03-01
**Domain:** FastAPI file upload endpoints, React file upload UI, audio playback, dual storage backend
**Confidence:** HIGH

## Summary

Phase 20 is a gap-closure phase that adds missing upload endpoints and frontend UI for media files (images and audio) across all Production Bible entity types. The entities and CRUD APIs already exist from Phase 17, but several media upload capabilities were deferred. Specifically: actor reference image uploads for Characters, wardrobe reference image uploads, a "Generate Base Appearance" endpoint, a standalone reverse-prompt endpoint, SonicIdentity reference audio upload, prop reference image upload wiring in the frontend, ScoreTheme/SFXItem audio upload endpoints, and inline audio playback components.

The existing codebase has a well-established pattern for file uploads (see `upload_set_reference` and `upload_prop_reference` in `sets_props.py`) using FastAPI's `UploadFile` with the dual storage backend (`LocalStorageBackend` / `S3StorageBackend`). The frontend uses raw `FormData` + `fetch` for uploads (see `uploadSetReference` in `client.ts`). This phase replicates these patterns across the remaining entities. No new dependencies are needed -- all tools (Pillow, storage backend, LLM adapter, FastAPI UploadFile) are already in the stack.

**Primary recommendation:** Follow the existing upload pattern from `sets_props.py` exactly (content-type validation, 10MB size limit, dual storage path, graceful LLM degradation) for all new upload endpoints. Add new endpoints to existing route files by domain (characters.py, sets_props.py, sound.py). Create a reusable `AudioPlayer` React component for inline playback across Sound Department and Sonic Identity entities.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PBEX-01 | Character entity with actor_refs images, base_appearance | Actor ref upload endpoint + Generate Base Appearance endpoint patterns |
| PBEX-02 | Wardrobe sub-entity with reference_images | Wardrobe image upload endpoint pattern |
| PBEX-07 | Set entity with reference_image, reverse_prompt | Standalone generate-reverse-prompt endpoint |
| PBEX-08 | SonicIdentity with reference_audio | Audio upload endpoint + UI component |
| PBEX-13 | Prop entity with reference_image | Frontend prop upload button wiring |
| PBEX-16 | ScoreTheme with reference_audio | Audio upload endpoint for score themes |
| PBEX-17 | SFXItem with source_audio | Audio upload endpoint for SFX items |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | existing | File upload via `UploadFile` + `File(...)` | Already used for set/prop upload |
| asyncio | stdlib | `asyncio.to_thread()` for sync file I/O | Existing pattern in sets_props.py |
| storage_backend | existing | Dual local/S3 storage abstraction | `get_storage_backend()` singleton |
| ReversePromptService | existing | LLM Vision reverse-prompting for images | Used in set upload already |
| React | 19 | Frontend UI components | Existing stack |
| Tailwind CSS | 4 | Styling | Existing stack |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow | existing | Image validation/processing if needed | Only for generate-appearance |
| HTMLAudioElement | browser API | Inline audio playback | For audio player component |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom audio player | react-audio-player | Adds dependency for simple `<audio>` tag; not worth it |
| Multipart form for multiple images | Sequential single uploads | Simpler; actor refs rarely have >5 images |

## Architecture Patterns

### Recommended Project Structure
No new files needed except potentially an audio player component:
```
backend/vidpipe/api/
  characters.py          # ADD: actor-refs upload, generate-appearance, wardrobe ref upload
  sets_props.py          # ADD: standalone generate-reverse-prompt, prop upload wiring fix
  sound.py               # ADD: score-theme audio upload, sfx audio upload, sonic-identity audio upload
frontend/src/
  components/
    AudioPlayer.tsx       # NEW: reusable inline audio playback component
    CharacterDetail.tsx   # MODIFY: actor refs upload UI, wardrobe ref upload UI
    SetDetail.tsx         # MODIFY: sonic identity audio upload UI
    SoundDepartment.tsx   # MODIFY: score theme + SFX audio upload UI, playback
  api/
    client.ts            # ADD: upload functions for new endpoints
```

### Pattern 1: File Upload Endpoint (Backend)
**What:** Consistent file upload handler with validation, dual storage, and DB update
**When to use:** Every new upload endpoint
**Example:**
```python
# Source: backend/vidpipe/api/sets_props.py (existing pattern)
@router.post("/entity/{entity_id}/upload-media")
async def upload_media(entity_id: str, file: UploadFile = File(...)):
    # 1. Validate content type
    allowed = ("image/png", "image/jpeg", "image/webp")  # or audio types
    if file.content_type not in allowed:
        raise HTTPException(status_code=422, detail=f"Invalid content type")

    # 2. Read and validate size
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Max 10MB")

    # 3. Store via dual backend
    storage = get_storage_backend()
    filename = file.filename or "upload.png"

    if isinstance(storage, LocalStorageBackend):
        from vidpipe.config import settings as _settings
        local_dir = _settings.storage.tmp_dir / "manifests" / str(bible_id) / "entity" / str(entity_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename
        await asyncio.to_thread(local_path.write_bytes, content)
        entity.field = str(local_path)
    else:
        key = f"manifests/{bible_id}/entity/{entity_id}/{filename}"
        await storage.put(key, content, file.content_type or "image/png")
        from vidpipe.config import settings as _settings
        local_path = _settings.storage.tmp_dir / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        entity.field = key

    # 4. Commit and return
    await session.commit()
    await session.refresh(entity)
    return entity_to_dict(entity)
```

### Pattern 2: File Upload Client (Frontend)
**What:** FormData-based upload via raw fetch
**When to use:** Every frontend upload call
**Example:**
```typescript
// Source: frontend/src/api/client.ts (existing pattern)
export async function uploadActorRef(characterId: string, file: File): Promise<CharacterResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/characters/${characterId}/actor-refs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<CharacterResponse>;
}
```

### Pattern 3: Audio Player Component
**What:** Reusable inline audio playback using native HTML5 `<audio>` element
**When to use:** Any entity with an audio file field (SonicIdentity, ScoreTheme, SFXItem, VoiceProfile)
**Example:**
```typescript
function AudioPlayer({ src, label }: { src: string; label?: string }) {
  if (!src) return null;
  return (
    <div className="flex items-center gap-2 bg-gray-800 rounded px-3 py-2">
      {label && <span className="text-xs text-gray-400">{label}</span>}
      <audio controls src={src} className="h-8 w-full" preload="none">
        Your browser does not support audio playback.
      </audio>
    </div>
  );
}
```

### Pattern 4: Character Actor Refs (JSON Array Append)
**What:** Actor refs stored as JSON array in `Character.actor_refs`. Upload appends to array rather than replacing.
**When to use:** Actor reference upload
**Example:**
```python
# Character.actor_refs is JSON list of image URLs/keys
# Upload appends new ref to the list
existing_refs = char.actor_refs or []
existing_refs.append(stored_path_or_key)
char.actor_refs = existing_refs
```

### Anti-Patterns to Avoid
- **Hardcoding storage paths:** Always use `get_storage_backend()` + conditional local/S3 logic
- **Skipping local copy in S3 mode:** Pipeline (LLM Vision, etc.) reads from local disk, so S3 uploads must also write a local copy
- **Using `request()` helper for uploads:** FormData uploads must NOT set `Content-Type: application/json` header; use raw `fetch`
- **Blocking I/O in async handler:** Always wrap `path.write_bytes()` in `asyncio.to_thread()`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio playback | Custom JS audio player | Native `<audio controls>` element | Browser-native, accessible, no dependencies |
| File type validation | Custom magic-byte detection | `file.content_type` from FastAPI UploadFile | Sufficient for user uploads, matches existing code |
| Storage abstraction | New upload helper | Existing `get_storage_backend()` pattern | Already handles local + S3 dual-write |
| Reverse prompting | New LLM call | `ReversePromptService.reverse_prompt_asset()` | Already encapsulates the vision adapter call |

## Common Pitfalls

### Pitfall 1: Missing Audio MIME Types
**What goes wrong:** Audio uploads rejected because only image types are validated
**Why it happens:** Copy-paste from image upload endpoint without updating allowed types
**How to avoid:** Audio endpoints must allow `audio/mpeg`, `audio/wav`, `audio/ogg`, `audio/webm`, `audio/mp4`, `audio/x-m4a`
**Warning signs:** 422 errors when uploading valid audio files

### Pitfall 2: Actor Refs Array Mutation Without Copy
**What goes wrong:** SQLAlchemy doesn't detect in-place list mutation, so changes aren't persisted
**Why it happens:** `char.actor_refs.append(url)` modifies the list in place without triggering change detection
**How to avoid:** Always create a new list: `char.actor_refs = (char.actor_refs or []) + [new_url]`
**Warning signs:** Upload succeeds but refs don't persist after page reload

### Pitfall 3: Prop Upload Frontend Not Wired
**What goes wrong:** Backend `POST /api/props/{id}/upload-reference` exists but frontend PropEditor has no upload button
**Why it happens:** Phase 17 built the backend endpoint but the frontend PropEditor component was shipped without an upload UI
**How to avoid:** Add file input + handler in PropEditor component, matching SetVisualTab pattern
**Warning signs:** "No image" placeholder with no upload option visible

### Pitfall 4: Generate Base Appearance Without Actor Refs
**What goes wrong:** Generate endpoint called when character has no actor_refs, causing empty generation
**Why it happens:** No guard on the endpoint to require at least one actor ref
**How to avoid:** Return 422 if `char.actor_refs` is empty/None with clear message
**Warning signs:** Base appearance generation produces generic faces unrelated to the character

### Pitfall 5: S3 Key vs Local Path in Audio src
**What goes wrong:** Audio player `src` attribute gets an S3 key (relative path) instead of a servable URL
**Why it happens:** DB stores keys for S3 mode; frontend needs to resolve to a URL
**How to avoid:** Ensure API response returns a servable URL or the frontend constructs one. The existing pattern for images routes through `/api/assets/` or direct Supabase URL. Audio files need the same treatment.
**Warning signs:** Audio player shows but won't play; browser console shows 404 for relative path

### Pitfall 6: Content-Type Header in FormData Fetch
**What goes wrong:** File upload sends `Content-Type: application/json` alongside FormData
**Why it happens:** Using the `request()` helper which auto-sets JSON headers
**How to avoid:** Use raw `fetch` for FormData uploads (no Content-Type header -- browser sets multipart boundary automatically)
**Warning signs:** 422 "Invalid content type" errors from backend

## Code Examples

### Backend: Actor Ref Upload Endpoint
```python
# In characters.py
@character_router.post("/characters/{character_id}/actor-refs")
async def upload_actor_ref(character_id: str, file: UploadFile = File(...)):
    """Upload an actor reference image. Appends to character.actor_refs list."""
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(status_code=422, detail="Must be image/png, image/jpeg, or image/webp")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Max 10MB")

    async with async_session() as session:
        char = await session.get(Character, uuid.UUID(character_id))
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found")

        storage = get_storage_backend()
        filename = file.filename or "actor_ref.png"

        if isinstance(storage, LocalStorageBackend):
            from vidpipe.config import settings as _settings
            local_dir = _settings.storage.tmp_dir / "manifests" / str(char.production_bible_id) / "characters" / str(char.id) / "actor_refs"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            await asyncio.to_thread(local_path.write_bytes, content)
            stored_path = str(local_path)
        else:
            key = f"manifests/{char.production_bible_id}/characters/{char.id}/actor_refs/{filename}"
            await storage.put(key, content, file.content_type or "image/png")
            from vidpipe.config import settings as _settings
            local_path = _settings.storage.tmp_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            stored_path = key

        # Append to actor_refs JSON array (new list to trigger SQLAlchemy change detection)
        char.actor_refs = (char.actor_refs or []) + [stored_path]

        await session.commit()
        await session.refresh(char)

        # Return full character response
        ward_result = await session.execute(select(Wardrobe).where(Wardrobe.character_id == char.id))
        vp_result = await session.execute(select(VoiceProfile).where(VoiceProfile.character_id == char.id))
        return _character_to_dict(char, list(ward_result.scalars().all()), vp_result.scalars().first())
```

### Backend: Audio Upload for ScoreTheme
```python
# In sound.py
@sound_router.post("/score-themes/{score_theme_id}/upload-audio")
async def upload_score_theme_audio(score_theme_id: str, file: UploadFile = File(...)):
    """Upload reference audio for a score theme."""
    ALLOWED_AUDIO = ("audio/mpeg", "audio/wav", "audio/ogg", "audio/webm", "audio/mp4", "audio/x-m4a")
    if file.content_type not in ALLOWED_AUDIO:
        raise HTTPException(status_code=422, detail=f"Must be one of {ALLOWED_AUDIO}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB for audio
        raise HTTPException(status_code=422, detail="Max 20MB")

    # ... storage pattern same as image uploads, but with audio content types
```

### Frontend: Audio Player Component
```typescript
// AudioPlayer.tsx
export function AudioPlayer({ src, label }: { src: string; label?: string }) {
  if (!src) return null;
  return (
    <div className="flex items-center gap-2 rounded border border-gray-700 bg-gray-800 px-3 py-2 mt-1">
      {label && <span className="text-xs text-gray-400 flex-shrink-0">{label}</span>}
      <audio controls src={src} className="h-8 w-full" preload="none" />
    </div>
  );
}
```

### Frontend: Actor Ref Upload in CharacterDetail
```typescript
// In ActorRefsTab, replace the placeholder message with upload UI
function ActorRefsTab({ character, onUploadActorRef }: {
  character: CharacterResponse;
  onUploadActorRef: (charId: string, file: File) => Promise<void>;
}) {
  const refs = character.actor_refs ?? [];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUploadActorRef(character.character_id, file);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-gray-300">Actor References</h4>
        <label className="text-xs px-3 py-1.5 rounded bg-blue-600 text-white hover:bg-blue-500 cursor-pointer">
          + Upload
          <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
        </label>
      </div>
      {refs.length === 0 ? (
        <div className="flex items-center justify-center py-12 rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-500">No actor references yet. Upload reference photos above.</p>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {refs.map((url, idx) => (
            <img key={idx} src={url} alt={`Ref ${idx + 1}`}
              className="w-full aspect-square object-cover rounded border border-gray-700" />
          ))}
        </div>
      )}
    </div>
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Placeholder "coming in future phase" text in Actor Refs tab | Upload UI with file picker | Phase 20 | Characters can now have actor reference photos |
| Prop upload backend-only (no frontend button) | Full upload button in PropEditor | Phase 20 | Props visually representable |
| Audio fields store string URLs manually | File upload endpoints with storage backend | Phase 20 | Audio files consistently managed |
| No audio playback in UI | Inline `<audio>` player component | Phase 20 | Users can preview reference audio |

## Open Questions

1. **Generate Base Appearance implementation**
   - What we know: The endpoint should use actor_refs as input to generate a canonical front-facing appearance image via the image generation adapter (Imagen/Gemini)
   - What's unclear: Exact prompting strategy for the image generation. Should it use an IP adapter or reference image approach?
   - Recommendation: Use the existing Imagen/Gemini image generation with a carefully crafted prompt describing "front-facing headshot, neutral expression, studio lighting" plus the character's description. Pass the first actor_ref as a reference image if the model supports it. Mark as a placeholder if Imagen IP adapter is not available -- can generate from text description of the character's `base_appearance` field only.

2. **Audio file serving in S3 mode**
   - What we know: Images served via redirect to Supabase public URL or FileResponse
   - What's unclear: Whether audio content-type is preserved through Supabase redirect
   - Recommendation: Follow the same pattern as image serving. If issues arise, add an explicit `/api/audio/{entity_type}/{id}` proxy endpoint.

3. **Actor ref deletion**
   - What we know: Issue #8 specifies a DELETE endpoint for individual actor refs
   - What's unclear: How to identify individual refs (by index? by URL?)
   - Recommendation: Use index-based deletion: `DELETE /api/characters/:id/actor-refs/:index`. Simple, matches array storage.

## Sources

### Primary (HIGH confidence)
- `backend/vidpipe/api/sets_props.py` - Existing upload pattern for Sets and Props
- `backend/vidpipe/api/characters.py` - Existing Character CRUD (no upload yet)
- `backend/vidpipe/api/sound.py` - Existing Sound CRUD (no audio upload yet)
- `backend/vidpipe/db/models.py` - All entity model definitions
- `frontend/src/api/client.ts` - Existing upload client pattern (`uploadSetReference`)
- `frontend/src/components/CharacterDetail.tsx` - ActorRefsTab with "coming soon" placeholder
- `frontend/src/components/SetDetail.tsx` - SetSonicTab without audio upload
- `frontend/src/components/SoundDepartment.tsx` - ScoreTheme/SFX without audio upload
- `.planning/v1.0-MILESTONE-AUDIT.md` - 8 tech debt items to close
- GitHub Issues #8, #9, #10, #11 - Detailed acceptance criteria

### Secondary (MEDIUM confidence)
- HTML5 `<audio>` element - Browser-native audio playback (MDN docs)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all tools already exist in the codebase; this is pure pattern replication
- Architecture: HIGH - following established upload patterns exactly
- Pitfalls: HIGH - derived from direct code inspection of existing patterns and their edge cases

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (stable -- no external dependencies, internal patterns only)
