---
status: diagnosed
phase: 17-production-bible-entity-expansion
source: [17-01-SUMMARY.md, 17-02-SUMMARY.md, 17-03-SUMMARY.md, 17-04-SUMMARY.md]
started: 2026-03-01T13:20:00Z
updated: 2026-03-01T16:45:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Create a Character in Casting Tab
expected: Open a Production Bible, go to Casting tab. Click "+ Add Character", enter name and role, submit. Character appears in the list with role badge.
result: issue
reported: "View page crashes with TypeError: Cannot read properties of undefined. Casting tab renders CharacterDetail component (Characters heading, + Add button visible) but API calls to /characters, /migrate-entities etc return errors. After Docker rebuild, View button triggers JS crash."
severity: blocker

### 2. Character Detail 4-Tab Editor
expected: Click a character in the list. Right panel shows detail editor with 4 sub-tabs: Overview, Actor References, Wardrobe, Voice Profile. Overview tab has editable fields for name, role, description, arc, base_appearance, prompt_tags. Edit a field and save — changes persist.
result: issue
reported: "When I click on a Production Bible Card, I get a browser console error that crashes my session"
severity: blocker

### 3. Add Wardrobe Item to Character
expected: In character detail, switch to Wardrobe tab. Click "+ Add Wardrobe", fill in label and prompt descriptor, save. Wardrobe item appears in the list with label and is_default badge.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 4. Voice Profile Save (Disabled Generate)
expected: In character detail, switch to Voice Profile tab. Fill in voice_id and style_notes, click save. Profile is created/updated. "Generate Sample" button is disabled and shows "coming soon" tooltip on hover.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 5. Create a Set in Art Department
expected: Switch to Art Department tab. You see a Sets/Props pill toggle at top, defaulting to Sets view. Click "+ Add Set", enter a name, save. Set appears in the list. Click it to see detail with Visual and Sonic Identity sub-tabs.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 6. Upload Set Reference Image
expected: Select a set, go to Visual tab. Click upload button and select an image file. Image uploads and displays. The reverse_prompt field may auto-populate via LLM Vision (or remain empty with no error if LLM Vision is unavailable).
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 7. Props Thumbnail Grid
expected: In Art Department tab, switch to Props view via the pill toggle. Click "+ Add Prop", enter name and description, save. Prop appears in a thumbnail grid. Click it to open inline editor with name, description, associated_characters, prompt_tags fields.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 8. Sound Department — Score Themes
expected: Switch to Sound tab. You see two sections: Score Themes and SFX Library. In Score Themes, click "+ Add Theme", enter name and mood descriptors, save. Theme appears in the list. "Generate Music" button is disabled with "coming soon" tooltip.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 9. SFX Library with Category Filter
expected: In Sound tab's SFX Library section, add a few SFX items with different categories (e.g. IMPACT, FOLEY). Category filter pills appear (All, Impact, Mechanical, Natural, UI, Foley, Ambience). Clicking a category pill filters the list to show only items of that category.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 10. Delete Character with Cascade
expected: In Casting tab, click the delete/trash icon on a character. A confirmation dialog appears. Confirm deletion. Character is removed from the list. Its wardrobes and voice profile are also deleted (verify by checking the character no longer appears).
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 11. Entity Count Badges on Tabs
expected: Department tab headers show entity counts — e.g. "Casting (2)", "Art (3)", "Sound (1)". Counts update when you add or delete entities.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

### 12. Collapsible Raw Assets Section
expected: Below the entity components in any department tab, there is a collapsible "Raw Assets" section that preserves access to the existing asset upload/editor UI. Clicking it expands to show the old asset list.
result: skipped
reason: Blocked by blocker — Production Bible View page crashes on load

## Summary

total: 12
passed: 0
issues: 2
pending: 0
skipped: 10

## Gaps

- truth: "Production Bible View page loads with Casting tab showing CharacterDetail component and functional API calls"
  status: fixed
  reason: "User reported: View page crashes with TypeError after Docker rebuild. API calls to new entity endpoints return errors. CharacterDetail renders but cannot communicate with backend."
  severity: blocker
  test: 1
  root_cause: "Backend API response serializers used generic 'id' field names but frontend expected entity-specific names (character_id, wardrobe_id, etc). Also 'wardrobes' vs 'wardrobe' mismatch."
  artifacts:
    - path: "backend/vidpipe/api/characters.py"
      issue: "id → character_id, wardrobe_id, voice_profile_id; wardrobes → wardrobe"
    - path: "backend/vidpipe/api/sets_props.py"
      issue: "id → set_id, sonic_identity_id, prop_id"
    - path: "backend/vidpipe/api/sound.py"
      issue: "id → score_theme_id (4x), sfx_item_id (4x)"
  missing: []
  debug_session: ".planning/debug/resolved/bible-view-crash.md"

- truth: "Clicking a Production Bible card opens the detail page with 4-tab editor for character editing"
  status: fixed
  reason: "User reported: When I click on a Production Bible Card, I get a browser console error that crashes my session"
  severity: blocker
  test: 2
  root_cause: "Same root cause as test 1 — response field name mismatches between backend serializers and frontend TypeScript types caused undefined property access crashes"
  artifacts:
    - path: "backend/vidpipe/api/characters.py"
      issue: "Response dict field names mismatched frontend types"
    - path: "backend/vidpipe/api/sets_props.py"
      issue: "Response dict field names mismatched frontend types"
    - path: "backend/vidpipe/api/sound.py"
      issue: "Response dict field names mismatched frontend types"
  missing: []
  debug_session: ".planning/debug/resolved/bible-view-crash.md"
