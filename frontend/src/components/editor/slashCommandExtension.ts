import { Extension } from "@tiptap/core";
import { Suggestion } from "@tiptap/suggestion";
import { PluginKey } from "@tiptap/pm/state";
import type { Editor } from "@tiptap/core";
import type { SuggestionKeyDownProps, SuggestionProps } from "@tiptap/suggestion";

const slashCommandPluginKey = new PluginKey("slashCommand");

export interface SlashCommandItem {
  title: string;
  description: string;
  icon: string;
  command: (editor: Editor) => void;
}

const SLASH_COMMANDS: SlashCommandItem[] = [
  {
    title: "Heading 1",
    description: "Large heading",
    icon: "H1",
    command: (editor) => editor.chain().focus().toggleHeading({ level: 1 }).run(),
  },
  {
    title: "Heading 2",
    description: "Medium heading",
    icon: "H2",
    command: (editor) => editor.chain().focus().toggleHeading({ level: 2 }).run(),
  },
  {
    title: "Heading 3",
    description: "Small heading",
    icon: "H3",
    command: (editor) => editor.chain().focus().toggleHeading({ level: 3 }).run(),
  },
  {
    title: "Bullet List",
    description: "Unordered list",
    icon: "\u2022",
    command: (editor) => editor.chain().focus().toggleBulletList().run(),
  },
  {
    title: "Ordered List",
    description: "Numbered list",
    icon: "1.",
    command: (editor) => editor.chain().focus().toggleOrderedList().run(),
  },
  {
    title: "Blockquote",
    description: "Quote block",
    icon: ">",
    command: (editor) => editor.chain().focus().toggleBlockquote().run(),
  },
  {
    title: "Code Block",
    description: "Syntax-highlighted code",
    icon: "{}",
    command: (editor) => editor.chain().focus().toggleCodeBlock().run(),
  },
  {
    title: "Horizontal Rule",
    description: "Divider line",
    icon: "\u2014",
    command: (editor) => editor.chain().focus().setHorizontalRule().run(),
  },
  {
    title: "Image",
    description: "Insert image from URL",
    icon: "\uD83D\uDDBC",
    command: (editor) => {
      const url = window.prompt("Image URL:");
      if (url?.trim()) {
        editor.chain().focus().setImage({ src: url.trim() }).run();
      }
    },
  },
  {
    title: "Bold",
    description: "Bold text",
    icon: "B",
    command: (editor) => editor.chain().focus().toggleBold().run(),
  },
  {
    title: "Italic",
    description: "Italic text",
    icon: "I",
    command: (editor) => editor.chain().focus().toggleItalic().run(),
  },
];

function buildMenuRow(
  item: SlashCommandItem,
  isSelected: boolean,
  onSelect: () => void,
): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors ${
    isSelected ? "bg-gray-700 text-white" : "text-gray-300 hover:bg-gray-700/50"
  }`;

  const iconSpan = document.createElement("span");
  iconSpan.className =
    "flex h-6 w-6 shrink-0 items-center justify-center rounded bg-gray-900 text-[11px] font-mono text-gray-400";
  iconSpan.textContent = item.icon;

  const textCol = document.createElement("span");
  textCol.className = "flex flex-col min-w-0";

  const titleSpan = document.createElement("span");
  titleSpan.className = "text-xs font-medium";
  titleSpan.textContent = item.title;

  const descSpan = document.createElement("span");
  descSpan.className = "text-[10px] text-gray-500";
  descSpan.textContent = item.description;

  textCol.appendChild(titleSpan);
  textCol.appendChild(descSpan);
  btn.appendChild(iconSpan);
  btn.appendChild(textCol);

  btn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    onSelect();
  });

  return btn;
}

export const SlashCommand = Extension.create({
  name: "slashCommand",

  addProseMirrorPlugins() {
    return [
      Suggestion<SlashCommandItem, SlashCommandItem>({
        pluginKey: slashCommandPluginKey,
        editor: this.editor,
        char: "/",
        startOfLine: false,
        items: ({ query }: { query: string }) => {
          const q = query.toLowerCase();
          return SLASH_COMMANDS.filter(
            (item) =>
              item.title.toLowerCase().includes(q) ||
              item.description.toLowerCase().includes(q),
          ).slice(0, 8);
        },
        command: ({
          editor,
          range,
          props,
        }: {
          editor: Editor;
          range: { from: number; to: number };
          props: SlashCommandItem;
        }) => {
          editor.chain().focus().deleteRange(range).run();
          props.command(editor);
        },
        render: () => {
          let container: HTMLDivElement | null = null;
          let selectedIndex = 0;
          let currentItems: SlashCommandItem[] = [];
          let currentCommand: ((props: SlashCommandItem) => void) | null = null;

          function render() {
            if (!container) return;
            container.innerHTML = "";
            if (currentItems.length === 0) return;

            const menu = document.createElement("div");
            menu.className =
              "w-56 max-h-64 overflow-y-auto rounded-lg border border-gray-700 bg-gray-800 py-1 shadow-xl";

            currentItems.forEach((item, idx) => {
              const btn = buildMenuRow(item, idx === selectedIndex, () => {
                currentCommand?.(item);
              });
              menu.appendChild(btn);
            });

            container.appendChild(menu);
          }

          return {
            onStart(props: SuggestionProps<SlashCommandItem, SlashCommandItem>) {
              container = document.createElement("div");
              container.style.position = "fixed";
              container.style.zIndex = "9999";
              document.body.appendChild(container);

              selectedIndex = 0;
              currentItems = props.items ?? [];
              currentCommand = props.command;

              const rect = props.clientRect?.();
              if (rect && container) {
                container.style.left = `${rect.left}px`;
                container.style.top = `${rect.bottom + 4}px`;
              }
              render();
            },

            onUpdate(props: SuggestionProps<SlashCommandItem, SlashCommandItem>) {
              currentItems = props.items ?? [];
              currentCommand = props.command;
              if (selectedIndex >= currentItems.length) selectedIndex = 0;

              const rect = props.clientRect?.();
              if (rect && container) {
                container.style.left = `${rect.left}px`;
                container.style.top = `${rect.bottom + 4}px`;
              }
              render();
            },

            onKeyDown(props: SuggestionKeyDownProps) {
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
                // Let Suggestion plugin handle cleanup via onExit
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
