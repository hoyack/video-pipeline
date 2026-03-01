---
status: resolved
trigger: "Production Bible View page crashes with TypeError when clicking a Bible card. API calls to entity endpoints return errors."
created: 2026-03-01T17:00:00Z
updated: 2026-03-01T17:25:00Z
---

## Current Focus

hypothesis: CONFIRMED AND FIXED - Backend API response field names did not match frontend TypeScript type definitions
test: Compared backend response dicts with frontend TypeScript interfaces
expecting: Mismatch found and all mismatches corrected
next_action: None - fix applied and verified

## Symptoms

expected: Clicking a Production Bible card opens detail page with entity tabs (Casting, Art, Sound). API calls to /characters, /migrate-entities work.
actual: View page crashes with TypeError: Cannot read properties of undefined. Browser console error crashes session. API calls to entity endpoints return errors.
errors: TypeError: Cannot read properties of undefined (reading unknown property)
reproduction: Click on a Production Bible Card from the list
started: After Phase 17 code was added and Docker rebuilt

## Eliminated

## Evidence

- timestamp: 2026-03-01T17:05:00Z
  checked: Backend characters.py _character_to_dict() response fields
  found: Returns "id" (line 114), "wardrobes" (line 123), sub-entity "id" fields
  implication: Frontend expects "character_id", "wardrobe" (singular), "wardrobe_id", "voice_profile_id"

- timestamp: 2026-03-01T17:06:00Z
  checked: Backend sets_props.py _set_to_dict(), _sonic_identity_to_dict(), _prop_to_dict() response fields
  found: All return "id" as primary key field name
  implication: Frontend expects "set_id", "sonic_identity_id", "prop_id"

- timestamp: 2026-03-01T17:07:00Z
  checked: Backend sound.py score theme and SFX response dicts
  found: Return "id" as primary key field name (4 separate return sites per entity)
  implication: Frontend expects "score_theme_id", "sfx_item_id"

- timestamp: 2026-03-01T17:08:00Z
  checked: CharacterDetail.tsx line 326 - selectedCharacter.wardrobe.length
  found: Backend returns "wardrobes" (plural), frontend reads "wardrobe" (singular) => undefined => .length crashes
  implication: This is the specific TypeError crash point when a character is selected

- timestamp: 2026-03-01T17:09:00Z
  checked: CharacterDetail.tsx line 46 - characters.find(c => c.character_id === selectedCharacterId)
  found: c.character_id is undefined because API returns "id" not "character_id"
  implication: selectedCharacter will always be null; character selection broken

- timestamp: 2026-03-01T17:10:00Z
  checked: All 8 entity response serializers across 3 backend files
  found: Systematic pattern - ALL use "id" as primary key instead of entity-specific ID field names
  implication: Every entity component (CharacterDetail, SetDetail, SoundDepartment) will fail

- timestamp: 2026-03-01T17:20:00Z
  checked: Post-fix validation - Python syntax, ruff lint, TypeScript compilation
  found: All pass cleanly
  implication: Fix is safe and correct

## Resolution

root_cause: Backend API response serializers used generic "id" field name for all 8 entity primary keys, but frontend TypeScript types expected entity-specific names (character_id, wardrobe_id, voice_profile_id, set_id, sonic_identity_id, prop_id, score_theme_id, sfx_item_id). Additionally, the Character response used "wardrobes" (plural) but frontend expected "wardrobe" (singular). This caused property access on entity objects to return undefined, leading to TypeError crashes when accessing .length, comparing IDs, etc.

fix: Updated all entity response serializer functions and inline response dicts in 3 backend files to use the correct entity-specific field names matching the frontend TypeScript type definitions. 14 response dict sites were corrected across characters.py (6), sets_props.py (4), and sound.py (4+4=8 inline dicts across 4 endpoints x 2 entities).

verification: Python syntax check passes, ruff lint passes, TypeScript compilation passes. No backend tests exist for these endpoints (per CLAUDE.md, tests are not comprehensive for entity UI).

files_changed:
- backend/vidpipe/api/characters.py
- backend/vidpipe/api/sets_props.py
- backend/vidpipe/api/sound.py
