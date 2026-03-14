# Phase 26: Asset Tag Frontend Enhancements - Research

**Researched:** 2026-03-14
**Domain:** CodeMirror extensions, React component patterns, frontend-only enhancements
**Confidence:** HIGH

## Summary

This phase delivers four frontend-only enhancements that complete the asset-to-generation loop: @tag autocomplete in the scene editor, a tag preview panel, LoRA training status display on Actor detail, and a Tag Reference Sheet tab on the Production Bible view. All backend APIs already exist -- `getBoundAssetsSummary()` (Phase 23), `getActorLoraStatus()` / `trainActorLora()` (Phase 25) -- so this phase requires zero backend changes.

The primary technical challenge is integrating CodeMirror's `@codemirror/autocomplete` extension into the existing `MarkdownEditorModal` component. The package is already installed (v6.20.0, pulled in transitively by `@codemirror/lang-markdown`) but needs to be listed as a direct dependency in `package.json`. The `MarkdownEditorModal` currently creates extensions via `createEditorExtensions()` with no mechanism for additional extensions -- it needs to accept an optional `extraExtensions` prop so the autocomplete and hover tooltip extensions can be injected from the parent `ShotEditorCard`.

The LoRA training status UI (ATED-03) is already fully implemented in `ActorLibraryDetail.tsx` from Phase 25, including the Train/Retrain button, status badge, training date, polling, and minimum-refs validation. This requirement is satisfied and needs only verification.

**Primary recommendation:** Extend `MarkdownEditorModal` to accept optional extra CodeMirror extensions, create `assetTagCompletion.ts` and `tagHoverPreview.ts` as standalone extension factories, and add a `TagPreviewPanel` side component plus a new "Tag Reference Sheet" tab in `ProductionBibleCreator`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New CodeMirror extension file: `frontend/src/components/codemirror/assetTagCompletion.ts`
- Uses `@codemirror/autocomplete` package's `autocompletion()` function
- Triggers on `@` character -- shows dropdown of all bound assets from Production Bible
- Data sourced from `getBoundAssetsSummary()` (Phase 23 API)
- Each option shows: `@tag_name` as label, `TYPE: Name` as detail, description as info
- Wire into existing `editorExtensions.ts` setup
- Pass `bibleId` and `boundAssets` from scene editor component
- New component: `frontend/src/components/TagPreviewPanel.tsx`
- Collapsible side panel in the scene editor
- On hover/click of an `@tag` in the editor, shows: primary reference image, asset name, text description, asset type
- Data from the same `BoundAssetSummary` list used by autocomplete
- Uses CodeMirror decoration or tooltip API for tag detection in editor
- Phase 25 already added the Train Identity Model button and status badge to `ActorLibraryDetail.tsx` -- verify before implementing
- New tab on Production Bible detail view: "Tag Reference Sheet"
- Lists all bound assets (CastBindings, SetBindings, PropBindings) with @tag syntax, type, thumbnail, description
- Data from `getBoundAssetsSummary()` API

### Claude's Discretion
- CodeMirror tooltip vs decoration API choice for tag preview
- Tag preview panel positioning and styling
- Whether autocomplete fetches on mount or lazily
- Keyboard navigation in autocomplete dropdown
- Mobile/responsive considerations for preview panel

### Deferred Ideas (OUT OF SCOPE)
- Inline tag syntax highlighting (colored @tags in editor) -- Future
- Tag validation (red highlight for unresolved tags) -- Future
- Drag-and-drop assets from reference sheet into editor -- Future
- Tag usage analytics (which tags used most across scenes) -- Future
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ATED-01 | CodeMirror @tag autocomplete extension shows dropdown of bound assets when user types @ in scene editor | `@codemirror/autocomplete` v6.20.0 already installed; `autocompletion()` with `override` source handles custom triggers; `matchBefore(/@\w*/)` detects @ prefix; `BoundAssetSummary` type and `getBoundAssetsSummary()` exist from Phase 23 |
| ATED-02 | Tag preview panel in scene editor shows asset reference image, name, and description on hover/click of @tag | `hoverTooltip()` from `@codemirror/view` supports hover-triggered DOM tooltips; alternatively a standalone `TagPreviewPanel` component receives selected tag via callback |
| ATED-03 | Actor detail view shows LoRA training status with Train/Regenerate buttons and training date | **Already fully implemented** in Phase 25 `ActorLibraryDetail.tsx` lines 361-408 -- Train button, LoraStatusBadge, polling, training date display all present |
| ATED-04 | Production Bible "Tag Reference Sheet" tab lists all bound assets with @tag syntax, type, and thumbnail | `DEPARTMENT_TABS` array in `ProductionBibleCreator.tsx` can be extended with a 4th tab; `getBoundAssetsSummary()` returns the exact data needed |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `@codemirror/autocomplete` | 6.20.0 | Autocomplete dropdown for @tag | Already installed (transitive dep); official CodeMirror autocomplete |
| `@codemirror/view` | (installed) | `hoverTooltip()` for tag preview on hover | Already installed; official CodeMirror view layer |
| `@codemirror/state` | (installed) | `Extension` type, `Facet`, `StateField` | Already installed; needed for extension typing |
| `@uiw/react-codemirror` | 4.25.5 | React wrapper for CodeMirror | Already used in `MarkdownEditorModal` |
| React 19 | 19.x | Component framework | Already the project standard |
| Tailwind CSS 4 | 4.x | Styling | Already the project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `getBoundAssetsSummary()` | Phase 23 | Fetch bound assets for a Production Bible | Data source for autocomplete, preview panel, and reference sheet |
| `getActorLoraStatus()` | Phase 25 | Poll LoRA training status | Already wired in ActorLibraryDetail |
| `trainActorLora()` | Phase 25 | Dispatch LoRA training | Already wired in ActorLibraryDetail |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `hoverTooltip()` for preview | Standalone side panel with event listener | Side panel is more space-flexible, can show images full-size; tooltip is more contextual but constrained in size |
| `autocompletion({ override })` | `CompletionSource` via `languageData` | `override` is simpler for our case (single source, not language-aware); `languageData` is for language-server patterns |

**Installation:**
```bash
# @codemirror/autocomplete is already installed transitively
# Add as explicit dependency for clarity:
cd frontend && npm install @codemirror/autocomplete
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/src/components/
├── codemirror/
│   ├── editorExtensions.ts        # Existing: base editor extensions
│   ├── VidpipeEditorTheme.ts      # Existing: dark theme
│   └── assetTagCompletion.ts      # NEW: @tag autocomplete extension factory
├── TagPreviewPanel.tsx             # NEW: collapsible side panel for tag preview
├── TagReferenceSheet.tsx           # NEW: bound assets table for Production Bible tab
├── MarkdownEditorModal.tsx         # MODIFIED: accept extraExtensions prop
├── ShotEditorCard.tsx              # MODIFIED: pass boundAssets + bibleId
├── EditModeOverlay.tsx             # MODIFIED: fetch boundAssets, pass to ShotEditorCard
├── ProductionBibleCreator.tsx      # MODIFIED: add "Tag Reference Sheet" tab
└── ActorLibraryDetail.tsx          # NO CHANGES (Phase 25 complete)
```

### Pattern 1: Extension Factory with Data Injection
**What:** Create CodeMirror extensions as factory functions that accept runtime data (bound assets list) and return `Extension` objects.
**When to use:** When CodeMirror extensions need access to application state.
**Example:**
```typescript
// Source: @codemirror/autocomplete v6.20.0 API
import { autocompletion, CompletionContext } from "@codemirror/autocomplete";
import type { Extension } from "@codemirror/state";
import type { BoundAssetSummary } from "../../api/types.ts";

export function createAssetTagCompletion(
  boundAssets: BoundAssetSummary[]
): Extension {
  return autocompletion({
    override: [
      (context: CompletionContext) => {
        const before = context.matchBefore(/@\w*/);
        if (!before) return null;
        return {
          from: before.from,
          options: boundAssets.map((asset) => ({
            label: `@${asset.tag.toLowerCase()}`,
            detail: `${asset.type}: ${asset.name}`,
            info: asset.description ?? undefined,
            type: asset.type === "CHARACTER" ? "variable" : asset.type === "SET" ? "class" : "property",
          })),
        };
      },
    ],
  });
}
```

### Pattern 2: Extra Extensions Prop on MarkdownEditorModal
**What:** Extend `MarkdownEditorModal` to accept an optional `extraExtensions` prop that gets merged with the base extensions.
**When to use:** When different uses of the modal need different CodeMirror behavior.
**Example:**
```typescript
interface MarkdownEditorModalProps {
  // ... existing props
  extraExtensions?: Extension[];
}

// Inside component:
const extensions = useMemo(
  () => [...createEditorExtensions(), ...(extraExtensions ?? [])],
  [extraExtensions]
);
```

### Pattern 3: Hover Tooltip for Tag Preview
**What:** Use CodeMirror's `hoverTooltip()` to detect @tags under the cursor and show a tooltip, OR use a callback to a side panel.
**When to use:** ATED-02 tag preview.
**Recommendation:** Use a **hybrid approach** -- `hoverTooltip()` for inline contextual preview (small tooltip with name + type), plus a `TagPreviewPanel` side panel that shows the full detail (image, description) when a tag is clicked. This avoids the constraint of tooltip size for images while still giving hover feedback.
**Example:**
```typescript
import { hoverTooltip } from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import type { BoundAssetSummary } from "../../api/types.ts";

export function createTagHoverPreview(
  boundAssets: BoundAssetSummary[],
  onTagSelect?: (tag: string | null) => void
): Extension {
  const assetMap = new Map(boundAssets.map(a => [a.tag.toLowerCase(), a]));

  return hoverTooltip((view, pos) => {
    const line = view.state.doc.lineAt(pos);
    const text = line.text;
    const col = pos - line.from;

    // Find @tag at position
    const atTagRe = /@([a-zA-Z0-9_]+)/g;
    let match;
    while ((match = atTagRe.exec(text)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      if (col >= start && col <= end) {
        const tag = match[1].toLowerCase();
        const asset = assetMap.get(tag);
        if (!asset) return null;
        if (onTagSelect) onTagSelect(tag);
        return {
          pos: line.from + start,
          end: line.from + end,
          above: true,
          create() {
            const dom = document.createElement("div");
            dom.className = "cm-tag-tooltip";
            dom.innerHTML = `<strong>${asset.type}: ${asset.name}</strong>`;
            return { dom };
          },
        };
      }
    }
    return null;
  }, { hoverTime: 200 });
}
```

### Pattern 4: Tag Reference Sheet as Table Component
**What:** A standalone component that renders bound assets in a reference table.
**When to use:** ATED-04 Production Bible tab.
**Example:**
```typescript
// TagReferenceSheet.tsx
interface TagReferenceSheetProps {
  bibleId: string;
}

export function TagReferenceSheet({ bibleId }: TagReferenceSheetProps) {
  const [assets, setAssets] = useState<BoundAssetSummary[]>([]);
  // Fetch via getBoundAssetsSummary(bibleId) on mount
  // Render as table: @tag | Type | Thumbnail | Name | Description
}
```

### Anti-Patterns to Avoid
- **Recreating extensions on every render:** CodeMirror extensions must be memoized (via `useMemo` with stable deps). Recreating them causes the editor to reset. The existing code already does this correctly -- follow the same pattern.
- **Fetching bound assets inside the CodeMirror extension:** The extension factory should receive pre-fetched data. Fetching inside the completion source would cause network calls on every keystroke.
- **Embedding images in CodeMirror tooltips:** Tooltip DOM is constrained. Use the side panel for image-heavy previews; keep tooltips text-only or minimal.
- **Modifying `editorExtensions.ts` directly to add autocomplete:** The autocomplete depends on runtime data (boundAssets). It cannot be in the static `createEditorExtensions()` function. Instead, compose at the component level via `extraExtensions`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Autocomplete dropdown | Custom dropdown overlay with position tracking | `@codemirror/autocomplete` `autocompletion()` | Handles keyboard nav, scroll, positioning, filtering automatically |
| Hover detection in editor | Manual mouse event listeners + position math | `hoverTooltip()` from `@codemirror/view` | Handles debouncing, position calculation, tooltip lifecycle |
| Text matching for @tags | Custom parser for CodeMirror content | `matchBefore(/@\w*/)` on `CompletionContext` | Built-in regex matching that respects editor state |
| Autocomplete filtering | Manual filter as user types | CodeMirror's built-in fuzzy match | `autocompletion` automatically filters options against typed text |

**Key insight:** CodeMirror 6 has excellent extension APIs for exactly these use cases. The autocomplete and tooltip systems are mature, well-tested, and handle all the edge cases (scrolling, positioning, keyboard interaction, accessibility) that custom solutions would need to replicate.

## Common Pitfalls

### Pitfall 1: Extensions Not Updating When Bound Assets Change
**What goes wrong:** The autocomplete list shows stale data because the extension was memoized with an empty dependency array.
**Why it happens:** `useMemo(() => createAssetTagCompletion(boundAssets), [])` caches the initial empty array.
**How to avoid:** Include `boundAssets` in the `useMemo` dependency array. CodeMirror handles extension reconfiguration gracefully when `@uiw/react-codemirror` receives new extension arrays.
**Warning signs:** Autocomplete shows no results after bound assets load, or shows outdated list after assets change.

### Pitfall 2: Duplicate Autocomplete Extensions
**What goes wrong:** Two autocomplete dropdowns appear, or closeBrackets stops working.
**Why it happens:** `closeBrackets()` from `@codemirror/autocomplete` is already in `createEditorExtensions()`. Adding a second `autocompletion()` call creates a conflict because CodeMirror only supports one autocomplete instance.
**How to avoid:** The `autocompletion()` function should be called only once. Since `closeBrackets()` is a separate feature (not `autocompletion()`), adding `autocompletion({ override: [...] })` alongside `closeBrackets()` is safe -- they coexist. However, do NOT add a second `autocompletion()` call.
**Warning signs:** Console warnings about duplicate facets, or autocomplete appearing twice.

### Pitfall 3: Missing Production Bible Context in ShotEditorCard
**What goes wrong:** Autocomplete never activates because no `boundAssets` are passed.
**Why it happens:** `ShotEditorCard` currently has no `production_bible_id` prop. The `EditModeOverlay` knows the manifestId but doesn't pass it through.
**How to avoid:** Thread `productionBibleId` from `EditModeOverlay` (which has `manifestId` state) through to `ShotEditorCard`, then fetch `getBoundAssetsSummary()` at the overlay level and pass the result down.
**Warning signs:** @-typing in the editor produces no dropdown.

### Pitfall 4: Tooltip DOM Not Styled for Dark Theme
**What goes wrong:** Hover tooltip appears with light background/text against the dark editor.
**Why it happens:** CodeMirror tooltip DOM is outside React's styling context.
**How to avoid:** The existing `VidpipeEditorTheme.ts` already has `.cm-tooltip` styles. Additional styles for `.cm-tag-tooltip` should be added either in the theme or in `index.css`.
**Warning signs:** White/unstyled tooltip flash on hover.

### Pitfall 5: Tag Reference Sheet Not Showing in Production Bible View
**What goes wrong:** The new tab doesn't appear because `DEPARTMENT_TABS` is used for Stage 3 rendering logic that filters assets by type.
**Why it happens:** The Tag Reference Sheet is not a department -- it's a cross-cutting view. Adding it to `DEPARTMENT_TABS` would break the asset filtering logic.
**How to avoid:** Add the Tag Reference Sheet as a separate tab outside the `DEPARTMENT_TABS` array, or add it as a 4th entry with special handling (no `assetTypes` filter, renders `TagReferenceSheet` component instead of asset lists).
**Warning signs:** Tab appears but shows empty content, or breaks other tabs.

## Code Examples

### Existing API Client (Phase 23)
```typescript
// Source: frontend/src/api/client.ts lines 1913-1914
export function getBoundAssetsSummary(bibleId: string): Promise<BoundAssetSummary[]> {
  return request<BoundAssetSummary[]>(`/api/production-bibles/${bibleId}/bound-assets/summary`);
}
```

### Existing Type (Phase 23)
```typescript
// Source: frontend/src/api/types.ts lines 1124-1130
export interface BoundAssetSummary {
  tag: string;
  name: string;
  type: "CHARACTER" | "SET" | "PROP";
  primary_thumbnail_url: string | null;
  description: string | null;
}
```

### Existing LoRA Status UI (Phase 25 -- ATED-03 Already Complete)
```typescript
// Source: frontend/src/components/ActorLibraryDetail.tsx lines 361-408
// LoRA Identity Model section with:
// - LoraStatusBadge component (No Model / Queued / Training / Model Ready / Failed)
// - Training date display
// - Train/Retrain button with refs >= 5 validation
// - 10-second polling interval when QUEUED or TRAINING
// - Error display
```

### Existing Editor Extension Pattern
```typescript
// Source: frontend/src/components/MarkdownEditorModal.tsx line 35
const extensions = useMemo(() => createEditorExtensions(), []);
// ... then passed to:
<CodeMirror extensions={extensions} ... />
```

### Existing Production Bible Tab Pattern
```typescript
// Source: frontend/src/components/ProductionBibleCreator.tsx lines 70-85
const DEPARTMENT_TABS = [
  { id: "casting", label: "Casting", assetTypes: ["CHARACTER"] },
  { id: "art", label: "Art Department", assetTypes: ["ENVIRONMENT", "PROP", "OBJECT", "VEHICLE", "STYLE"] },
  { id: "sound", label: "Sound", assetTypes: [] },
];
```

### CompletionContext.matchBefore API
```typescript
// Source: @codemirror/autocomplete v6.20.0 types
matchBefore(expr: RegExp): { from: number; to: number; text: string } | null;
// Returns the match of the given expression directly before the cursor.
// Used to detect "@..." prefix for triggering asset tag autocomplete.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `[CHAR:TAG]` syntax | `@tag` syntax (primary) | Phase 23 | Simpler user experience; cross-type resolution |
| Bible-scoped entities | Asset Library + Bindings | Phase 22 | Standalone actors/sets/props reusable across productions |
| No LoRA training | LoRA training infrastructure | Phase 25 | Per-actor identity models for consistent generation |

**Already complete (no work needed):**
- ATED-03 (LoRA training status UI) is fully implemented in Phase 25

## Open Questions

1. **Autocomplete data freshness**
   - What we know: `getBoundAssetsSummary()` fetches on mount. New bindings added after the editor opens won't appear.
   - What's unclear: How often do users add bindings while actively editing shots?
   - Recommendation: Fetch on mount. Add a "Refresh" button or refetch when editor re-opens. This matches the lazy-fetch pattern used throughout the project.

2. **Tag Preview Panel vs Tooltip sizing**
   - What we know: `hoverTooltip()` creates DOM elements that are constrained by viewport. Reference images need reasonable display size.
   - What's unclear: Whether a tooltip with a 64px thumbnail is sufficient or if users need larger images.
   - Recommendation (Claude's discretion): Use both -- `hoverTooltip()` for a minimal text tooltip on hover, and a collapsible `TagPreviewPanel` side panel that shows the full detail (image, name, description, type) for the currently selected/clicked tag.

3. **Where to fetch bound assets in the component tree**
   - What we know: `EditModeOverlay` has `manifestId` (which is `production_bible_id`). `ShotEditorCard` opens `MarkdownEditorModal`.
   - What's unclear: Should assets be fetched at the overlay level or per-modal?
   - Recommendation: Fetch once at `EditModeOverlay` level when `manifestId` is set. Pass `boundAssets` as prop through `ShotEditorCard` to `MarkdownEditorModal`. Avoids N fetches for N shots.

## Sources

### Primary (HIGH confidence)
- `@codemirror/autocomplete` v6.20.0 type definitions (installed at `frontend/node_modules/@codemirror/autocomplete/dist/index.d.ts`) -- `autocompletion()`, `CompletionContext`, `Completion` interfaces
- `@codemirror/view` type definitions (installed) -- `hoverTooltip()`, `Tooltip` interface, `EditorView`
- Existing codebase: `frontend/src/components/codemirror/editorExtensions.ts` -- current extension setup
- Existing codebase: `frontend/src/components/MarkdownEditorModal.tsx` -- current editor modal pattern
- Existing codebase: `frontend/src/components/ActorLibraryDetail.tsx` -- complete LoRA status UI (lines 361-460)
- Existing codebase: `frontend/src/api/types.ts` -- `BoundAssetSummary` type (line 1124)
- Existing codebase: `frontend/src/api/client.ts` -- `getBoundAssetsSummary()` function (line 1913)
- Existing codebase: `frontend/src/components/ProductionBibleCreator.tsx` -- `DEPARTMENT_TABS` pattern (line 70)
- PRD: `docs/assets_mapping.md` Section 7.1 -- `createAssetTagCompletion()` code example

### Secondary (MEDIUM confidence)
- `@uiw/react-codemirror` v4.25.5 -- React wrapper handles extension reconfiguration when extensions prop changes

### Tertiary (LOW confidence)
- None -- all findings verified against installed packages and existing code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages already installed, APIs verified against type definitions
- Architecture: HIGH - patterns derived from existing codebase, CodeMirror APIs confirmed in installed types
- Pitfalls: HIGH - identified from actual code inspection (duplicate autocomplete, missing props, theme styling)
- ATED-03 status: HIGH - verified complete by reading ActorLibraryDetail.tsx lines 361-460

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable -- all dependencies already locked in package.json)
