---
status: testing
phase: 26-asset-tag-frontend-enhancements
source: 26-01-SUMMARY.md, 26-02-SUMMARY.md, 26-03-SUMMARY.md
started: 2026-03-14T23:30:00Z
updated: 2026-03-14T23:30:00Z
---

## Current Test

number: 1
name: @Tag Autocomplete Dropdown
expected: |
  Open a scene that has a Production Bible attached. Edit a shot prompt. Type `@` in the editor. A dropdown should appear showing all bound assets from the Production Bible. Each entry shows @tag_name as the label, TYPE: Name as detail, and a description.
awaiting: user response

## Tests

### 1. @Tag Autocomplete Dropdown
expected: Open a scene that has a Production Bible attached. Edit a shot prompt. Type `@` in the editor. A dropdown should appear showing all bound assets from the Production Bible. Each entry shows @tag_name as the label, TYPE: Name as detail, and a description.
result: [pending]

### 2. Hover Tooltip on @Tag
expected: In a shot prompt editor, hover the mouse over an existing @tag (e.g., @brandon). A small tooltip should appear showing the asset type (e.g., "Character") and the asset name (e.g., "Brandon").
result: [pending]

### 3. Click @Tag Opens Preview Panel
expected: In a shot prompt editor, click on an @tag. A side panel should appear on the right side showing the asset's reference image thumbnail, name, @tag, type badge, and text description. Clicking the X or clicking another area should close the panel.
result: [pending]

### 4. Tag Reference Sheet Tab
expected: Open a Production Bible detail view. Alongside the existing Casting, Art Department, and Sound tabs, a 4th "Tag Reference" tab should be visible.
result: [pending]

### 5. Tag Reference Sheet Content
expected: Click the "Tag Reference" tab. A table/grid should display all bound assets with columns for @tag syntax, type badge (Character/Set/Prop), thumbnail image, asset name, and description. A search/filter input should be present at the top that filters the list by tag, name, or type.
result: [pending]

### 6. LoRA Training Status on Actor Detail
expected: Navigate to Asset Library → Actors → open an Actor detail view. There should be a "LoRA Identity Model" section showing a status badge (e.g., "No Model") and a "Train Identity Model" button. The button should be disabled if the actor has fewer than 5 reference images.
result: [pending]

### 7. No Autocomplete Without Production Bible
expected: Open a scene that does NOT have a Production Bible attached. Edit a shot prompt and type `@`. No autocomplete dropdown should appear — the @ character is simply typed normally.
result: [pending]

## Summary

total: 7
passed: 0
issues: 0
pending: 7
skipped: 0

## Gaps

[none yet]
