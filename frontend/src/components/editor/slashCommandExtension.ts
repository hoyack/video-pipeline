import { Extension } from "@tiptap/core";
import { Suggestion } from "@tiptap/suggestion";
import { PluginKey } from "@tiptap/pm/state";
import type { Editor } from "@tiptap/core";

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
    icon: "•",
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
    icon: "—",
    command: (editor) => editor.chain().focus().setHorizontalRule().run(),
  },
  {
    title: "Image",
    description: "Insert image from URL",
    icon: "🖼",
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

export const SlashCommand = Extension.create({
  name: "slashCommand",

  addProseMirrorPlugins() {
    return [
      Suggestion({
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
        command: ({ editor, range, props }: { editor: Editor; range: { from: number; to: number }; props: SlashCommandItem }) => {
          // Delete the slash + query text, then execute the command
          editor.chain().focus().deleteRange(range).run();
          props.command(editor);
        },
        render: () => {
          let component: {
            updateProps: (props: Record<string, unknown>) => void;
            destroy: () => void;
            element: HTMLElement;
          } | null = null;

          return {
            onStart: (props: Record<string, unknown>) => {
              const el = document.createElement("div");
              el.className = "slash-command-portal";
              document.body.appendChild(el);

              component = {
                element: el,
                updateProps: (newProps: Record<string, unknown>) => {
                  renderMenu(el, newProps);
                },
                destroy: () => {
                  el.remove();
                },
              };
              renderMenu(el, props);
            },
            onUpdate: (props: Record<string, unknown>) => {
              component?.updateProps(props);
            },
            onKeyDown: (props: { event: KeyboardEvent }) => {
              if (props.event.key === "Escape") {
                component?.destroy();
                component = null;
                return true;
              }
              // Let the menu handle arrow keys and enter
              const menu = component?.element?.querySelector("[data-slash-menu]") as HTMLElement | null;
              if (menu) {
                const event = new CustomEvent("slash-keydown", { detail: props.event });
                menu.dispatchEvent(event);
                if (props.event.key === "ArrowUp" || props.event.key === "ArrowDown" || props.event.key === "Enter") {
                  return true;
                }
              }
              return false;
            },
            onExit: () => {
              component?.destroy();
              component = null;
            },
          };
        },
      }),
    ];
  },
});

function renderMenu(container: HTMLElement, props: Record<string, unknown>) {
  const items = (props.items ?? []) as SlashCommandItem[];
  const clientRect = props.clientRect as (() => DOMRect | null) | undefined;
  const command = props.command as ((props: SlashCommandItem) => void) | undefined;

  const rect = clientRect?.();
  if (rect) {
    container.style.position = "fixed";
    container.style.left = `${rect.left}px`;
    container.style.top = `${rect.bottom + 4}px`;
    container.style.zIndex = "9999";
  }

  // Simple DOM render (avoids React portal complexity)
  let selectedIndex = parseInt(container.dataset.selectedIndex ?? "0", 10);
  if (selectedIndex >= items.length) selectedIndex = 0;

  container.innerHTML = "";
  if (items.length === 0) return;

  const menu = document.createElement("div");
  menu.setAttribute("data-slash-menu", "true");
  menu.className =
    "w-56 max-h-64 overflow-y-auto rounded-lg border border-gray-700 bg-gray-800 py-1 shadow-xl";

  items.forEach((item, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors ${
      idx === selectedIndex
        ? "bg-gray-700 text-white"
        : "text-gray-300 hover:bg-gray-700/50"
    }`;
    btn.innerHTML = `
      <span class="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-gray-900 text-[11px] font-mono text-gray-400">${item.icon}</span>
      <span class="flex flex-col min-w-0">
        <span class="text-xs font-medium">${item.title}</span>
        <span class="text-[10px] text-gray-500">${item.description}</span>
      </span>
    `;
    btn.addEventListener("mousedown", (e) => {
      e.preventDefault();
      command?.(item);
    });
    menu.appendChild(btn);
  });

  // Keyboard navigation
  menu.addEventListener("slash-keydown" as string, ((e: CustomEvent) => {
    const key = e.detail?.key;
    if (key === "ArrowDown") {
      selectedIndex = (selectedIndex + 1) % items.length;
      container.dataset.selectedIndex = String(selectedIndex);
      renderMenu(container, props);
    } else if (key === "ArrowUp") {
      selectedIndex = (selectedIndex - 1 + items.length) % items.length;
      container.dataset.selectedIndex = String(selectedIndex);
      renderMenu(container, props);
    } else if (key === "Enter") {
      command?.(items[selectedIndex]);
    }
  }) as EventListener);

  container.appendChild(menu);
}
