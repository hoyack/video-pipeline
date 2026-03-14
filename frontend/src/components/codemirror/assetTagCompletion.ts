import { autocompletion } from "@codemirror/autocomplete";
import type { CompletionContext, CompletionResult } from "@codemirror/autocomplete";
import type { Extension } from "@codemirror/state";
import { EditorView, hoverTooltip } from "@codemirror/view";
import type { Tooltip } from "@codemirror/view";
import type { BoundAssetSummary } from "../../api/types.ts";

/**
 * CodeMirror autocomplete extension for @tag completion.
 * Triggers when the user types @ and shows matching bound assets.
 */
export function createAssetTagCompletion(
  boundAssets: BoundAssetSummary[],
): Extension {
  function completionSource(
    context: CompletionContext,
  ): CompletionResult | null {
    const match = context.matchBefore(/@\w*/);
    if (!match) return null;

    return {
      from: match.from,
      options: boundAssets.map((asset) => ({
        label: "@" + asset.tag.toLowerCase(),
        detail: asset.type + ": " + asset.name,
        info: asset.description ?? undefined,
        type:
          asset.type === "CHARACTER"
            ? "variable"
            : asset.type === "SET"
              ? "class"
              : "property",
      })),
    };
  }

  return autocompletion({ override: [completionSource], activateOnTyping: true });
}

/**
 * CodeMirror hover tooltip extension for @tag previews.
 * Shows asset type and name when hovering over @tags in the editor.
 * Optionally calls onTagSelect to update a side panel.
 */
export function createTagHoverPreview(
  boundAssets: BoundAssetSummary[],
  onTagSelect?: (tag: string | null) => void,
): Extension {
  const tagMap = new Map<string, BoundAssetSummary>();
  for (const asset of boundAssets) {
    tagMap.set(asset.tag.toLowerCase(), asset);
  }

  const tagRegex = /@([a-zA-Z0-9_]+)/g;

  return hoverTooltip(
    (view: EditorView, pos: number): Tooltip | null => {
      const line = view.state.doc.lineAt(pos);
      const lineText = line.text;
      const col = pos - line.from;

      tagRegex.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = tagRegex.exec(lineText)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        if (col >= start && col <= end) {
          const tagName = m[1].toLowerCase();
          const asset = tagMap.get(tagName);
          if (asset) {
            onTagSelect?.(tagName);
            return {
              pos: line.from + start,
              end: line.from + end,
              above: true,
              create() {
                const dom = document.createElement("div");
                dom.className = "cm-tag-tooltip";

                const strong = document.createElement("strong");
                strong.textContent = `${asset.type}: ${asset.name}`;
                dom.appendChild(strong);

                if (asset.description) {
                  const desc = document.createElement("div");
                  const text = asset.description;
                  desc.textContent =
                    text.length > 80 ? text.slice(0, 80) + "..." : text;
                  desc.style.marginTop = "2px";
                  desc.style.color = "#9ca3af";
                  dom.appendChild(desc);
                }

                return { dom };
              },
            };
          }
        }
      }

      onTagSelect?.(null);
      return null;
    },
    { hoverTime: 200 },
  );
}

/**
 * CodeMirror click handler extension for @tag selection.
 * Clicking an @tag in the editor triggers onTagSelect to open the preview panel.
 */
export function createTagClickHandler(
  boundAssets: BoundAssetSummary[],
  onTagSelect?: (tag: string | null) => void,
): Extension {
  const tagMap = new Map<string, BoundAssetSummary>();
  for (const asset of boundAssets) {
    tagMap.set(asset.tag.toLowerCase(), asset);
  }
  const tagRegex = /@([a-zA-Z0-9_]+)/g;

  return EditorView.domEventHandlers({
    click(event: MouseEvent, view: EditorView) {
      const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos === null) return false;
      const line = view.state.doc.lineAt(pos);
      const col = pos - line.from;
      tagRegex.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = tagRegex.exec(line.text)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        if (col >= start && col <= end) {
          const tagName = m[1].toLowerCase();
          if (tagMap.has(tagName)) {
            onTagSelect?.(tagName);
            return false; // don't prevent default (cursor should still move)
          }
        }
      }
      return false;
    },
  });
}
