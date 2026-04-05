import { useEffect, useRef, useCallback } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import clsx from "clsx";
import { AssetMention } from "./assetMentionExtension.ts";
import type { AssetMentionItem } from "./assetMentionExtension.ts";
import type { BoundAssetSummary } from "../../api/types.ts";

interface MiniPromptEditorProps {
  value: string;
  onChange: (text: string) => void;
  /** Border style: amber when dirty */
  isDirty?: boolean;
  /** Border style: indigo when regenerating */
  regenerating?: boolean;
  /** Pre-fetched bound assets for @-mention suggestions */
  boundAssets?: BoundAssetSummary[];
  /** Stop click propagation (for accordion cards) */
  stopPropagation?: boolean;
  placeholder?: string;
}

export function MiniPromptEditor({
  value,
  onChange,
  isDirty,
  regenerating,
  boundAssets,
  stopPropagation,
  placeholder,
}: MiniPromptEditorProps) {
  const lastEmittedRef = useRef(value);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        bold: false,
        italic: false,
        strike: false,
        code: false,
        codeBlock: false,
        blockquote: false,
        bulletList: false,
        orderedList: false,
        listItem: false,
        heading: false,
        horizontalRule: false,
      }),
      AssetMention,
    ],
    content: value ? `<p>${escapeHtml(value).replace(/\n/g, "</p><p>")}</p>` : "",
    editorProps: {
      attributes: {
        class: "mini-prompt-prosemirror",
      },
    },
    onUpdate: ({ editor: ed }) => {
      const text = ed.getText({ blockSeparator: "\n" });
      lastEmittedRef.current = text;
      onChangeRef.current(text);
    },
  });

  // Populate @-mention assets
  useEffect(() => {
    if (!editor || editor.isDestroyed) return;
    const items: AssetMentionItem[] = (boundAssets || []).map((a) => ({
      tag: a.tag,
      name: a.name,
      type: a.type,
      thumbnailUrl: a.primary_thumbnail_url,
    }));
    (editor.storage.assetMention as { assets: AssetMentionItem[] }).assets = items;
  }, [editor, boundAssets]);

  // Sync external value changes
  useEffect(() => {
    if (!editor || editor.isDestroyed) return;
    if (value === lastEmittedRef.current) return;
    lastEmittedRef.current = value;
    const html = value ? `<p>${escapeHtml(value).replace(/\n/g, "</p><p>")}</p>` : "";
    editor.commands.setContent(html);
  }, [value, editor]);

  // Set placeholder via data attribute
  const setPlaceholder = useCallback(
    (el: HTMLDivElement | null) => {
      if (el && placeholder) {
        const pm = el.querySelector(".ProseMirror");
        if (pm) pm.setAttribute("data-placeholder", placeholder);
      }
    },
    [placeholder],
  );

  return (
    <div
      ref={setPlaceholder}
      onClick={stopPropagation ? (e) => e.stopPropagation() : undefined}
      className={clsx(
        "mini-prompt-editor rounded border bg-gray-950 transition-colors focus-within:ring-1",
        regenerating
          ? "border-indigo-600/50"
          : isDirty
            ? "border-amber-600 focus-within:ring-amber-500"
            : "border-gray-700 focus-within:ring-blue-500",
      )}
    >
      <EditorContent editor={editor} />
    </div>
  );
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
