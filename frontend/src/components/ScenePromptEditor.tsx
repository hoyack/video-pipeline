import { useState, useEffect, useRef, useCallback } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import { BubbleMenu } from "@tiptap/react/menus";
import StarterKit from "@tiptap/starter-kit";
import CodeBlockLowlight from "@tiptap/extension-code-block-lowlight";
import Image from "@tiptap/extension-image";
import Link from "@tiptap/extension-link";
import { common, createLowlight } from "lowlight";
import { Markdown } from "tiptap-markdown";
import clsx from "clsx";
import { uploadEditorImage } from "../api/client.ts";
import { ImagePopover } from "./editor/ImagePopover.tsx";
import { LinkPopover } from "./editor/LinkPopover.tsx";
import { SlashCommand } from "./editor/slashCommandExtension.ts";
import { AssetMention } from "./editor/assetMentionExtension.ts";
import { getBoundAssetsSummary } from "../api/client.ts";
import type { AssetMentionItem } from "./editor/assetMentionExtension.ts";

const lowlight = createLowlight(common);

interface ScenePromptEditorProps {
  /** Current content as a markdown string */
  value: string;
  /** Called on every content change with the markdown string */
  onChange: (markdown: string) => void;
  /** Placeholder text when the editor is empty */
  placeholder?: string;
  /** Controls border color (amber when dirty, gray when clean) */
  isDirty?: boolean;
  /** Optional id for label association */
  id?: string;
  /** Optional className for the outer wrapper */
  className?: string;
  /** Callback to expand editor to fullscreen (renders expand button in toolbar) */
  onExpand?: () => void;
  /** Production Bible ID — enables @-mention suggestions for bound assets */
  productionBibleId?: string | null;
}

function setAssetMentionAssets(editor: { storage: unknown }, assets: AssetMentionItem[]) {
  const storage = editor.storage as { assetMention: { assets: AssetMentionItem[] } };
  storage.assetMention.assets = assets;
}

// ── Toolbar button ──────────────────────────────────────────────────────────

function TipBtn({
  active,
  disabled,
  onClick,
  title,
  children,
}: {
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onMouseDown={(e) => e.preventDefault()}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      aria-pressed={active}
      className={clsx(
        "rounded px-1.5 py-1 text-xs font-medium transition-colors",
        active
          ? "bg-gray-700 text-white"
          : "text-gray-400 hover:bg-gray-800 hover:text-gray-200",
        disabled && "opacity-30 pointer-events-none",
      )}
    >
      {children}
    </button>
  );
}

function Sep() {
  return <div className="mx-0.5 h-4 w-px bg-gray-700" />;
}

// ── Toolbar ─────────────────────────────────────────────────────────────────

function EditorToolbar({
  editor,
  onImageUpload,
  onExpand,
}: {
  editor: ReturnType<typeof useEditor>;
  onImageUpload: (file: File) => void;
  onExpand?: () => void;
}) {
  const [showImagePopover, setShowImagePopover] = useState(false);
  const [showLinkPopover, setShowLinkPopover] = useState(false);

  // Ctrl+K opens link popover
  useEffect(() => {
    if (!editor) return;
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setShowLinkPopover((v) => !v);
      }
    };
    const el = editor.view.dom;
    el.addEventListener("keydown", handler);
    return () => el.removeEventListener("keydown", handler);
  }, [editor]);

  if (!editor) return null;

  return (
    <div className="flex flex-wrap items-center gap-0.5 border-b border-gray-700 bg-gray-900 px-2 py-1 rounded-t-lg">
      <TipBtn active={editor.isActive("bold")} onClick={() => editor.chain().focus().toggleBold().run()} title="Bold (Ctrl+B)">
        <strong>B</strong>
      </TipBtn>
      <TipBtn active={editor.isActive("italic")} onClick={() => editor.chain().focus().toggleItalic().run()} title="Italic (Ctrl+I)">
        <em>I</em>
      </TipBtn>
      <TipBtn active={editor.isActive("strike")} onClick={() => editor.chain().focus().toggleStrike().run()} title="Strikethrough">
        <s>S</s>
      </TipBtn>
      <TipBtn active={editor.isActive("code")} onClick={() => editor.chain().focus().toggleCode().run()} title="Inline Code">
        <span className="font-mono text-[11px]">&lt;/&gt;</span>
      </TipBtn>

      <Sep />

      <TipBtn active={editor.isActive("heading", { level: 1 })} onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()} title="Heading 1">
        H1
      </TipBtn>
      <TipBtn active={editor.isActive("heading", { level: 2 })} onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()} title="Heading 2">
        H2
      </TipBtn>
      <TipBtn active={editor.isActive("heading", { level: 3 })} onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()} title="Heading 3">
        H3
      </TipBtn>

      <Sep />

      <TipBtn active={editor.isActive("bulletList")} onClick={() => editor.chain().focus().toggleBulletList().run()} title="Bullet List">
        <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="2" cy="4" r="1.5" />
          <rect x="5" y="3" width="10" height="2" rx="0.5" />
          <circle cx="2" cy="8" r="1.5" />
          <rect x="5" y="7" width="10" height="2" rx="0.5" />
          <circle cx="2" cy="12" r="1.5" />
          <rect x="5" y="11" width="10" height="2" rx="0.5" />
        </svg>
      </TipBtn>
      <TipBtn active={editor.isActive("orderedList")} onClick={() => editor.chain().focus().toggleOrderedList().run()} title="Ordered List">
        <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="currentColor">
          <text x="0" y="5.5" fontSize="5" fontWeight="bold">1</text>
          <rect x="5" y="3" width="10" height="2" rx="0.5" />
          <text x="0" y="9.5" fontSize="5" fontWeight="bold">2</text>
          <rect x="5" y="7" width="10" height="2" rx="0.5" />
          <text x="0" y="13.5" fontSize="5" fontWeight="bold">3</text>
          <rect x="5" y="11" width="10" height="2" rx="0.5" />
        </svg>
      </TipBtn>
      <TipBtn active={editor.isActive("blockquote")} onClick={() => editor.chain().focus().toggleBlockquote().run()} title="Blockquote">
        <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="currentColor">
          <rect x="0" y="1" width="2" height="14" rx="1" />
          <rect x="4" y="4" width="11" height="2" rx="0.5" />
          <rect x="4" y="7" width="9" height="2" rx="0.5" />
          <rect x="4" y="10" width="7" height="2" rx="0.5" />
        </svg>
      </TipBtn>

      <Sep />

      <TipBtn active={editor.isActive("codeBlock")} onClick={() => editor.chain().focus().toggleCodeBlock().run()} title="Code Block">
        <span className="font-mono text-[11px]">{"{ }"}</span>
      </TipBtn>
      <TipBtn onClick={() => editor.chain().focus().setHorizontalRule().run()} title="Horizontal Rule">
        ―
      </TipBtn>

      <Sep />

      {/* Image */}
      <div className="relative">
        <TipBtn onClick={() => { setShowImagePopover((v) => !v); setShowLinkPopover(false); }} title="Insert Image">
          <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="1" y="2" width="14" height="12" rx="1.5" />
            <circle cx="5" cy="6" r="1.5" fill="currentColor" stroke="none" />
            <path d="M1 12l4-4 3 3 2-2 5 5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </TipBtn>
        {showImagePopover && (
          <ImagePopover
            onInsertUrl={(url) => {
              editor.chain().focus().setImage({ src: url }).run();
              setShowImagePopover(false);
            }}
            onUploadFile={(file) => {
              onImageUpload(file);
              setShowImagePopover(false);
            }}
            onClose={() => setShowImagePopover(false)}
          />
        )}
      </div>

      {/* Link */}
      <div className="relative">
        <TipBtn
          active={editor.isActive("link")}
          onClick={() => { setShowLinkPopover((v) => !v); setShowImagePopover(false); }}
          title="Link (Ctrl+K)"
        >
          <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M6.5 9.5a3 3 0 004 .5l2-2a3 3 0 00-4.2-4.3L7 5" strokeLinecap="round" />
            <path d="M9.5 6.5a3 3 0 00-4-.5l-2 2a3 3 0 004.2 4.3L9 11" strokeLinecap="round" />
          </svg>
        </TipBtn>
        {showLinkPopover && (
          <LinkPopover
            initialUrl={(editor.getAttributes("link").href as string) ?? ""}
            onSubmit={(url) => {
              editor.chain().focus().extendMarkRange("link").setLink({ href: url }).run();
              setShowLinkPopover(false);
            }}
            onRemove={() => {
              editor.chain().focus().extendMarkRange("link").unsetLink().run();
              setShowLinkPopover(false);
            }}
            onClose={() => setShowLinkPopover(false)}
          />
        )}
      </div>

      <Sep />

      <TipBtn disabled={!editor.can().undo()} onClick={() => editor.chain().focus().undo().run()} title="Undo (Ctrl+Z)">
        <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M3 7h7a4 4 0 110 8H8" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M6 4L3 7l3 3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </TipBtn>
      <TipBtn disabled={!editor.can().redo()} onClick={() => editor.chain().focus().redo().run()} title="Redo (Ctrl+Shift+Z)">
        <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M13 7H6a4 4 0 100 8h2" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M10 4l3 3-3 3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </TipBtn>

      {onExpand && (
        <>
          <Sep />
          <TipBtn onClick={onExpand} title="Expand to fullscreen editor">
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 6V2h4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M14 10v4h-4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M2 2l5 5" strokeLinecap="round" />
              <path d="M14 14l-5-5" strokeLinecap="round" />
            </svg>
          </TipBtn>
        </>
      )}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export function ScenePromptEditor({
  value,
  onChange,
  placeholder = "Describe your scene...",
  isDirty,
  id,
  className,
  onExpand,
  productionBibleId,
}: ScenePromptEditorProps) {
  const lastEmittedRef = useRef(value);
  const onChangeRef = useRef(onChange);

  // Ref for image upload — avoids stale closure in handleDrop/handlePaste
  const doImageUploadRef = useRef<((file: File) => Promise<void>) | null>(null);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: { levels: [1, 2, 3] },
        codeBlock: false, // Replaced by CodeBlockLowlight
      }),
      CodeBlockLowlight.configure({ lowlight }),
      Image.configure({
        inline: false,
        allowBase64: false,
      }),
      Link.configure({
        openOnClick: false,
        autolink: true,
        HTMLAttributes: {
          target: "_blank",
          rel: "noopener noreferrer",
        },
      }),
      Markdown.configure({
        html: false,
        transformPastedText: true,
        transformCopiedText: true,
      }),
      SlashCommand,
      AssetMention,
    ],
    content: value,
    editorProps: {
      attributes: {
        class: "scene-prompt-prosemirror",
        ...(id ? { id } : {}),
      },
      handleDrop(_view, event, _slice, moved) {
        if (moved || !event.dataTransfer?.files.length) return false;
        const file = Array.from(event.dataTransfer.files).find((f) =>
          ["image/png", "image/jpeg", "image/webp"].includes(f.type),
        );
        if (!file) return false;
        event.preventDefault();
        doImageUploadRef.current?.(file);
        return true;
      },
      handlePaste(_view, event) {
        const items = event.clipboardData?.items;
        if (!items) return false;
        for (const item of Array.from(items)) {
          if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) {
              event.preventDefault();
              doImageUploadRef.current?.(file);
              return true;
            }
          }
        }
        return false;
      },
    },
    onUpdate: ({ editor: ed }) => {
      const md = ((ed.storage as unknown) as { markdown: { getMarkdown: () => string } }).markdown.getMarkdown();
      lastEmittedRef.current = md;
      onChangeRef.current(md);
    },
  });

  // Image upload handler — shared by toolbar, drag-drop, and paste
  const doImageUpload = useCallback(
    async (file: File) => {
      if (!editor) return;
      if (file.size > 10 * 1024 * 1024) {
        console.warn("Image too large (max 10MB)");
        return;
      }
      try {
        const { url } = await uploadEditorImage(file);
        editor.chain().focus().setImage({ src: url }).run();
      } catch (err) {
        console.error("Image upload failed:", err);
      }
    },
    [editor],
  );
  useEffect(() => {
    doImageUploadRef.current = doImageUpload;
  }, [doImageUpload]);

  // Load Production Bible assets for @-mention suggestions
  useEffect(() => {
    if (!editor || editor.isDestroyed) return;
    if (!productionBibleId) {
      setAssetMentionAssets(editor, []);
      return;
    }
    getBoundAssetsSummary(productionBibleId)
      .then((assets) => {
        setAssetMentionAssets(editor, assets.map((a) => ({
          tag: a.tag,
          name: a.name,
          type: a.type,
          thumbnailUrl: a.primary_thumbnail_url,
        })));
      })
      .catch((err) => {
        console.error("Failed to load bible assets for @-mentions:", err);
      });
  }, [editor, productionBibleId]);

  // Sync external value changes (e.g. from MarkdownEditorModal)
  useEffect(() => {
    if (!editor || editor.isDestroyed) return;
    if (value === lastEmittedRef.current) return;
    lastEmittedRef.current = value;
    editor.commands.setContent(value);
  }, [value, editor]);

  // Set placeholder via data attribute
  const setPlaceholder = useCallback(
    (el: HTMLDivElement | null) => {
      if (el) {
        const pm = el.querySelector(".ProseMirror");
        if (pm) pm.setAttribute("data-placeholder", placeholder);
      }
    },
    [placeholder],
  );

  return (
    <div
      ref={setPlaceholder}
      className={clsx(
        "scene-prompt-editor rounded-lg border bg-gray-900 transition-colors focus-within:ring-1",
        isDirty === true
          ? "border-amber-600 focus-within:ring-amber-500"
          : "border-gray-700 focus-within:ring-blue-500",
        className,
      )}
    >
      <EditorToolbar editor={editor} onImageUpload={doImageUpload} onExpand={onExpand} />
      <EditorContent editor={editor} />

      {/* Bubble menu — floating toolbar on text selection */}
      {editor && (
        <BubbleMenu
          editor={editor}
          className="flex items-center gap-0.5 rounded-lg border border-gray-700 bg-gray-800 px-1 py-0.5 shadow-xl"
        >
          <TipBtn active={editor.isActive("bold")} onClick={() => editor.chain().focus().toggleBold().run()} title="Bold">
            <strong>B</strong>
          </TipBtn>
          <TipBtn active={editor.isActive("italic")} onClick={() => editor.chain().focus().toggleItalic().run()} title="Italic">
            <em>I</em>
          </TipBtn>
          <TipBtn active={editor.isActive("strike")} onClick={() => editor.chain().focus().toggleStrike().run()} title="Strike">
            <s>S</s>
          </TipBtn>
          <TipBtn active={editor.isActive("code")} onClick={() => editor.chain().focus().toggleCode().run()} title="Code">
            <span className="font-mono text-[11px]">&lt;/&gt;</span>
          </TipBtn>
          <Sep />
          <TipBtn
            active={editor.isActive("link")}
            onClick={() => {
              if (editor.isActive("link")) {
                editor.chain().focus().extendMarkRange("link").unsetLink().run();
              } else {
                const url = window.prompt("Link URL:");
                if (url?.trim()) {
                  editor.chain().focus().extendMarkRange("link").setLink({ href: url.trim() }).run();
                }
              }
            }}
            title="Link"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6.5 9.5a3 3 0 004 .5l2-2a3 3 0 00-4.2-4.3L7 5" strokeLinecap="round" />
              <path d="M9.5 6.5a3 3 0 00-4-.5l-2 2a3 3 0 004.2 4.3L9 11" strokeLinecap="round" />
            </svg>
          </TipBtn>
        </BubbleMenu>
      )}
    </div>
  );
}
