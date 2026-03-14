# Phase 26: Asset Tag Frontend Enhancements - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning
**Source:** PRD Express Path (docs/assets_mapping.md, Section D)

<domain>
## Phase Boundary

This phase delivers user-facing frontend improvements that complete the asset-to-generation loop: @tag autocomplete in the scene editor, a tag preview panel, LoRA training status on Actor detail (already partially done in Phase 25), and a Tag Reference Sheet tab on the Production Bible view. All backend APIs needed (bound-assets summary from Phase 23, LoRA status from Phase 25) already exist.

Specifically, this phase delivers:
- CodeMirror @tag autocomplete extension for the scene editor
- Tag preview panel showing asset details on hover/click of @tag
- LoRA training status display enhancements on Actor detail (if not fully covered by Phase 25)
- Production Bible "Tag Reference Sheet" tab

</domain>

<decisions>
## Implementation Decisions

### @tag Autocomplete (ATED-01)
- New CodeMirror extension file: `frontend/src/components/codemirror/assetTagCompletion.ts`
- Uses `@codemirror/autocomplete` package's `autocompletion()` function
- Triggers on `@` character — shows dropdown of all bound assets from Production Bible
- Data sourced from `getBoundAssetsSummary()` (Phase 23 API)
- Each option shows: `@tag_name` as label, `TYPE: Name` as detail, description as info
- Wire into existing `editorExtensions.ts` setup
- Pass `bibleId` and `boundAssets` from scene editor component

### Tag Preview Panel (ATED-02)
- New component: `frontend/src/components/TagPreviewPanel.tsx`
- Collapsible side panel in the scene editor
- On hover/click of an `@tag` in the editor, shows: primary reference image, asset name, text description, asset type
- Data from the same `BoundAssetSummary` list used by autocomplete
- Uses CodeMirror decoration or tooltip API for tag detection in editor

### LoRA Training Status (ATED-03)
- Phase 25 already added the Train Identity Model button and status badge to `ActorLibraryDetail.tsx`
- This requirement may already be fully satisfied — verify before implementing
- If gaps remain: add Train/Regenerate controls, training date display

### Tag Reference Sheet (ATED-04)
- New tab on Production Bible detail view: "Tag Reference Sheet"
- Lists all bound assets (CastBindings, SetBindings, PropBindings) with:
  - `@tag` syntax
  - Asset type (Character/Set/Prop)
  - Thumbnail image
  - Short description
- Data from `getBoundAssetsSummary()` API
- Quick-reference for writers composing shot descriptions

### Claude's Discretion
- CodeMirror tooltip vs decoration API choice for tag preview
- Tag preview panel positioning and styling
- Whether autocomplete fetches on mount or lazily
- Keyboard navigation in autocomplete dropdown
- Mobile/responsive considerations for preview panel

</decisions>

<specifics>
## Specific Ideas

- PRD Section 7.1 provides a code example for `createAssetTagCompletion()` extension
- The `BoundAssetSummary` type and `getBoundAssetsSummary()` function already exist from Phase 23
- Existing `editorExtensions.ts` CodeMirror setup in `frontend/src/components/codemirror/`
- Scene editor component needs to know which Production Bible is attached to pass to autocomplete
- Tag Reference Sheet is essentially a table/grid rendering of the same data autocomplete uses

</specifics>

<deferred>
## Deferred Ideas

- Inline tag syntax highlighting (colored @tags in editor) → Future
- Tag validation (red highlight for unresolved tags) → Future
- Drag-and-drop assets from reference sheet into editor → Future
- Tag usage analytics (which tags used most across scenes) → Future

</deferred>

---

*Phase: 26-asset-tag-frontend-enhancements*
*Context gathered: 2026-03-14 via PRD Express Path*
