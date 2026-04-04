import { Extension } from "@tiptap/core";
import { Suggestion } from "@tiptap/suggestion";
import { PluginKey } from "@tiptap/pm/state";
import type { Editor } from "@tiptap/core";

const assetMentionPluginKey = new PluginKey("assetMention");

export interface AssetMentionItem {
  tag: string;
  name: string;
  type: "CHARACTER" | "SET" | "PROP";
  thumbnailUrl: string | null;
}

const TYPE_ORDER: Record<string, number> = { CHARACTER: 0, SET: 1, PROP: 2 };

function filterAndSort(items: AssetMentionItem[], query: string): AssetMentionItem[] {
  if (!query) {
    return [...items].sort((a, b) => (TYPE_ORDER[a.type] ?? 9) - (TYPE_ORDER[b.type] ?? 9)).slice(0, 15);
  }
  const q = query.toLowerCase();
  return items
    .filter((item) => item.name.toLowerCase().includes(q) || item.tag.toLowerCase().includes(q))
    .sort((a, b) => {
      // Prefix matches rank higher
      const aPrefix = a.name.toLowerCase().startsWith(q) || a.tag.toLowerCase().startsWith(q) ? 0 : 1;
      const bPrefix = b.name.toLowerCase().startsWith(q) || b.tag.toLowerCase().startsWith(q) ? 0 : 1;
      if (aPrefix !== bPrefix) return aPrefix - bPrefix;
      return (TYPE_ORDER[a.type] ?? 9) - (TYPE_ORDER[b.type] ?? 9);
    })
    .slice(0, 10);
}

const TYPE_COLORS: Record<string, { bg: string; text: string }> = {
  CHARACTER: { bg: "rgba(59,130,246,0.15)", text: "#60a5fa" },
  SET: { bg: "rgba(34,197,94,0.15)", text: "#4ade80" },
  PROP: { bg: "rgba(245,158,11,0.15)", text: "#fbbf24" },
};

export const AssetMention = Extension.create({
  name: "assetMention",

  addStorage() {
    return {
      assets: [] as AssetMentionItem[],
    };
  },

  addProseMirrorPlugins() {
    const extensionStorage = this.storage as { assets: AssetMentionItem[] };

    return [
      Suggestion({
        pluginKey: assetMentionPluginKey,
        editor: this.editor,
        char: "@",
        items: ({ query }: { query: string }) => {
          return filterAndSort(extensionStorage.assets, query);
        },
        command: ({
          editor,
          range,
          props,
        }: {
          editor: Editor;
          range: { from: number; to: number };
          props: AssetMentionItem;
        }) => {
          editor
            .chain()
            .focus()
            .deleteRange(range)
            .insertContent(`@${props.tag} `)
            .run();
        },
        render: () => {
          let container: HTMLDivElement | null = null;
          let selectedIndex = 0;
          let currentItems: AssetMentionItem[] = [];
          let currentCommand: ((props: AssetMentionItem) => void) | null = null;

          function render() {
            if (!container) return;
            container.innerHTML = "";
            if (currentItems.length === 0) {
              const empty = document.createElement("div");
              empty.className = "px-3 py-2 text-xs text-gray-500 italic";

              const assets = extensionStorage.assets;
              empty.textContent =
                assets.length === 0
                  ? "No Production Bible attached"
                  : "No matching assets";
              container.appendChild(empty);
              return;
            }

            currentItems.forEach((item, idx) => {
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = `flex w-full items-center gap-2.5 px-3 py-1.5 text-left transition-colors ${
                idx === selectedIndex
                  ? "bg-gray-700 text-white"
                  : "text-gray-300 hover:bg-gray-700/50"
              }`;

              const colors = TYPE_COLORS[item.type] ?? TYPE_COLORS.PROP;

              // Thumbnail (safe DOM construction — no innerHTML)
              if (item.thumbnailUrl) {
                const img = document.createElement("img");
                img.src = item.thumbnailUrl;
                img.alt = "";
                img.className = "h-8 w-8 rounded object-cover bg-gray-900 shrink-0";
                btn.appendChild(img);
              } else {
                const placeholder = document.createElement("div");
                placeholder.className = "flex h-8 w-8 items-center justify-center rounded bg-gray-900 text-[10px] text-gray-500 shrink-0";
                placeholder.textContent = item.type[0];
                btn.appendChild(placeholder);
              }

              const textCol = document.createElement("div");
              textCol.className = "flex flex-col min-w-0 flex-1";
              const nameSpan = document.createElement("span");
              nameSpan.className = "text-xs font-medium truncate";
              nameSpan.textContent = item.name;
              const tagSpan = document.createElement("span");
              tagSpan.className = "text-[10px] text-gray-500 font-mono";
              tagSpan.textContent = `@${item.tag}`;
              textCol.appendChild(nameSpan);
              textCol.appendChild(tagSpan);
              btn.appendChild(textCol);

              const typeBadge = document.createElement("span");
              typeBadge.className = "shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium";
              typeBadge.style.background = colors.bg;
              typeBadge.style.color = colors.text;
              typeBadge.textContent = item.type;
              btn.appendChild(typeBadge);

              btn.addEventListener("mousedown", (e) => {
                e.preventDefault();
                currentCommand?.(item);
              });

              container.appendChild(btn);
            });
          }

          return {
            onStart(props: Record<string, unknown>) {
              container = document.createElement("div");
              container.className =
                "w-64 max-h-72 overflow-y-auto rounded-lg border border-gray-700 bg-gray-800 py-1 shadow-xl";
              container.style.position = "fixed";
              container.style.zIndex = "9999";
              document.body.appendChild(container);

              selectedIndex = 0;
              currentItems = (props.items ?? []) as AssetMentionItem[];
              currentCommand = props.command as (props: AssetMentionItem) => void;

              const clientRect = props.clientRect as (() => DOMRect | null) | undefined;
              const rect = clientRect?.();
              if (rect && container) {
                container.style.left = `${rect.left}px`;
                container.style.top = `${rect.bottom + 4}px`;
              }

              render();
            },

            onUpdate(props: Record<string, unknown>) {
              currentItems = (props.items ?? []) as AssetMentionItem[];
              currentCommand = props.command as (props: AssetMentionItem) => void;
              if (selectedIndex >= currentItems.length) selectedIndex = 0;

              const clientRect = props.clientRect as (() => DOMRect | null) | undefined;
              const rect = clientRect?.();
              if (rect && container) {
                container.style.left = `${rect.left}px`;
                container.style.top = `${rect.bottom + 4}px`;
              }

              render();
            },

            onKeyDown(props: { event: KeyboardEvent }) {
              if (props.event.key === "ArrowDown") {
                selectedIndex = (selectedIndex + 1) % Math.max(1, currentItems.length);
                render();
                return true;
              }
              if (props.event.key === "ArrowUp") {
                selectedIndex =
                  (selectedIndex - 1 + Math.max(1, currentItems.length)) %
                  Math.max(1, currentItems.length);
                render();
                return true;
              }
              if (props.event.key === "Enter" || props.event.key === "Tab") {
                if (currentItems[selectedIndex]) {
                  currentCommand?.(currentItems[selectedIndex]);
                  return true;
                }
              }
              if (props.event.key === "Escape") {
                return true;
              }
              return false;
            },

            onExit() {
              container?.remove();
              container = null;
            },
          };
        },
      }),
    ];
  },
});
