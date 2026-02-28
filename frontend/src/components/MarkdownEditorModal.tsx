import { useEffect, useState, useRef, useCallback } from "react";
import Markdown from "react-markdown";
import { CopyButton } from "./CopyButton.tsx";
import { EditorView, keymap, placeholder as cmPlaceholder } from "@codemirror/view";
import { EditorState, type Extension } from "@codemirror/state";
import { markdown } from "@codemirror/lang-markdown";
import { languages } from "@codemirror/language-data";
import {
  defaultKeymap,
  history,
  historyKeymap,
  undo,
  redo,
} from "@codemirror/commands";
import {
  syntaxHighlighting,
  defaultHighlightStyle,
  bracketMatching,
} from "@codemirror/language";
import {
  highlightSelectionMatches,
  openSearchPanel,
} from "@codemirror/search";
import { lineNumbers } from "@codemirror/view";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface MarkdownEditorModalProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onClose: () => void;
  onRegen?: () => void;
  regenerating?: boolean;
  extraContext?: string;
  onExtraContextChange?: (v: string) => void;
}

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------

function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className ?? "h-3 w-3"}`}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Custom dark theme matching the app palette
// ---------------------------------------------------------------------------

const editorTheme = EditorView.theme(
  {
    "&": {
      fontSize: "14px",
      height: "100%",
      backgroundColor: "rgb(3 7 18)", // gray-950
    },
    ".cm-content": {
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
      caretColor: "rgb(129 140 248)", // indigo-400
      lineHeight: "1.625",
      padding: "8px 0",
    },
    ".cm-line": {
      padding: "0 16px",
    },
    ".cm-gutters": {
      backgroundColor: "rgba(17 24 39 / 0.8)", // gray-900/80
      color: "rgb(75 85 99)", // gray-600
      borderRight: "1px solid rgb(31 41 55)", // gray-800
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
      fontSize: "12px",
    },
    ".cm-activeLineGutter": {
      backgroundColor: "rgba(55 65 81 / 0.3)",
      color: "rgb(156 163 175)", // gray-400
    },
    ".cm-activeLine": {
      backgroundColor: "rgba(55 65 81 / 0.15)",
    },
    ".cm-cursor": {
      borderLeftColor: "rgb(129 140 248)", // indigo-400
      borderLeftWidth: "2px",
    },
    ".cm-selectionBackground": {
      backgroundColor: "rgba(99 102 241 / 0.25) !important", // indigo-500/25
    },
    "&.cm-focused .cm-selectionBackground": {
      backgroundColor: "rgba(99 102 241 / 0.3) !important",
    },
    ".cm-matchingBracket": {
      backgroundColor: "rgba(99 102 241 / 0.3)",
      outline: "1px solid rgba(99 102 241 / 0.5)",
    },
    ".cm-searchMatch": {
      backgroundColor: "rgba(250 204 21 / 0.25)", // yellow
      outline: "1px solid rgba(250 204 21 / 0.4)",
    },
    ".cm-searchMatch.cm-searchMatch-selected": {
      backgroundColor: "rgba(250 204 21 / 0.4)",
    },
    ".cm-panels": {
      backgroundColor: "rgb(17 24 39)", // gray-900
      color: "rgb(209 213 219)", // gray-300
      borderBottom: "1px solid rgb(31 41 55)",
    },
    ".cm-panels input, .cm-panels button": {
      color: "rgb(209 213 219)",
    },
    ".cm-button": {
      backgroundImage: "none",
      backgroundColor: "rgb(55 65 81)", // gray-700
      border: "1px solid rgb(75 85 99)",
      borderRadius: "4px",
    },
    ".cm-textfield": {
      backgroundColor: "rgb(3 7 18)", // gray-950
      border: "1px solid rgb(55 65 81)",
      borderRadius: "4px",
    },
    ".cm-tooltip": {
      backgroundColor: "rgb(17 24 39)",
      border: "1px solid rgb(55 65 81)",
      color: "rgb(209 213 219)",
    },
    // Scrollbar styling
    ".cm-scroller": {
      overflow: "auto",
    },
  },
  { dark: true },
);

// ---------------------------------------------------------------------------
// Toolbar helpers: wrap / toggle prefix
// ---------------------------------------------------------------------------

function wrapSelection(view: EditorView, before: string, after: string) {
  const { from, to } = view.state.selection.main;
  const selected = view.state.sliceDoc(from, to);
  // If already wrapped, unwrap
  if (
    selected.startsWith(before) &&
    selected.endsWith(after) &&
    selected.length >= before.length + after.length
  ) {
    const inner = selected.slice(before.length, selected.length - after.length);
    view.dispatch({
      changes: { from, to, insert: inner },
      selection: { anchor: from, head: from + inner.length },
    });
  } else {
    const replacement = before + selected + after;
    view.dispatch({
      changes: { from, to, insert: replacement },
      selection: { anchor: from + before.length, head: from + before.length + selected.length },
    });
  }
  view.focus();
}

function toggleLinePrefix(view: EditorView, prefix: string) {
  const { from, to } = view.state.selection.main;
  const startLine = view.state.doc.lineAt(from);
  const endLine = view.state.doc.lineAt(to);

  const changes: { from: number; to: number; insert: string }[] = [];
  let allHavePrefix = true;

  for (let i = startLine.number; i <= endLine.number; i++) {
    const line = view.state.doc.line(i);
    if (!line.text.startsWith(prefix)) {
      allHavePrefix = false;
      break;
    }
  }

  for (let i = startLine.number; i <= endLine.number; i++) {
    const line = view.state.doc.line(i);
    if (allHavePrefix) {
      // Remove prefix
      changes.push({
        from: line.from,
        to: line.from + prefix.length,
        insert: "",
      });
    } else {
      // Add prefix
      if (!line.text.startsWith(prefix)) {
        changes.push({ from: line.from, to: line.from, insert: prefix });
      }
    }
  }

  view.dispatch({ changes });
  view.focus();
}

// ---------------------------------------------------------------------------
// Toolbar button component
// ---------------------------------------------------------------------------

function ToolbarButton({
  onClick,
  title,
  children,
  separator,
}: {
  onClick?: () => void;
  title?: string;
  children?: React.ReactNode;
  separator?: boolean;
}) {
  if (separator) {
    return <div className="mx-1 h-4 w-px bg-gray-700" />;
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="inline-flex h-7 w-7 items-center justify-center rounded text-gray-400 hover:bg-gray-700/60 hover:text-gray-200 transition-colors"
    >
      {children}
    </button>
  );
}

// ===========================================================================
// Main component
// ===========================================================================

export function MarkdownEditorModal({
  label,
  value,
  onChange,
  onClose,
  onRegen,
  regenerating,
  extraContext,
  onExtraContextChange,
}: MarkdownEditorModalProps) {
  const [showPreview, setShowPreview] = useState(true);
  const [showContext, setShowContext] = useState(false);
  const editorContainerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);

  // Keep onChange ref stable so the extension doesn't rebuild on every keystroke
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const lineCount = value.split("\n").length;
  const wordCount = value.trim() ? value.trim().split(/\s+/).length : 0;

  // ---- Build CodeMirror extensions ----
  const getExtensions = useCallback((): Extension[] => {
    return [
      lineNumbers(),
      history(),
      bracketMatching(),
      highlightSelectionMatches(),
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      markdown({ codeLanguages: languages }),
      editorTheme,
      cmPlaceholder("Start typing..."),
      keymap.of([
        ...defaultKeymap,
        ...historyKeymap,
        // Ctrl+B for bold
        {
          key: "Mod-b",
          run: (view) => {
            wrapSelection(view, "**", "**");
            return true;
          },
        },
        // Ctrl+I for italic
        {
          key: "Mod-i",
          run: (view) => {
            wrapSelection(view, "*", "*");
            return true;
          },
        },
        // Ctrl+Shift+K for inline code
        {
          key: "Mod-Shift-k",
          run: (view) => {
            wrapSelection(view, "`", "`");
            return true;
          },
        },
        // Ctrl+H for find & replace
        {
          key: "Mod-h",
          run: (view) => {
            openSearchPanel(view);
            return true;
          },
        },
      ]),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          onChangeRef.current(update.state.doc.toString());
        }
      }),
      EditorView.lineWrapping,
    ];
  }, []);

  // ---- Create / recreate editor ----
  useEffect(() => {
    if (!editorContainerRef.current) return;

    const state = EditorState.create({
      doc: value,
      extensions: getExtensions(),
    });

    const view = new EditorView({
      state,
      parent: editorContainerRef.current,
    });

    viewRef.current = view;
    view.focus();

    return () => {
      view.destroy();
      viewRef.current = null;
    };
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Sync external value changes (e.g. from regen) into the editor ----
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (current !== value) {
      view.dispatch({
        changes: { from: 0, to: current.length, insert: value },
      });
    }
  }, [value]);

  // ---- ESC to close ----
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // ---- Toolbar actions ----
  const cmAction = (fn: (view: EditorView) => void) => () => {
    if (viewRef.current) fn(viewRef.current);
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-gray-950">
      {/* ---- Title bar ---- */}
      <div className="flex items-center justify-between border-b border-gray-800 bg-gray-900 px-4 py-2">
        <div className="flex items-center gap-3">
          <svg
            className="h-4 w-4 text-gray-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931z"
            />
          </svg>
          <span className="text-sm font-medium text-gray-300">{label}</span>
          <span className="text-xs text-gray-600">
            {lineCount} line{lineCount !== 1 ? "s" : ""} &middot;{" "}
            {wordCount} word{wordCount !== 1 ? "s" : ""} &middot;{" "}
            {value.length} chars
          </span>
        </div>
        <div className="flex items-center gap-1">
          {onRegen && (
            <>
              <button
                type="button"
                onClick={onRegen}
                disabled={regenerating}
                className="flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
                title="Regenerate text"
              >
                {regenerating ? (
                  <Spinner className="h-3 w-3 text-indigo-400" />
                ) : (
                  <svg
                    className="h-3.5 w-3.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182"
                    />
                  </svg>
                )}
                {regenerating ? "Generating..." : "Regen"}
              </button>
              {onExtraContextChange && (
                <button
                  type="button"
                  onClick={() => setShowContext(!showContext)}
                  className={`rounded px-1.5 py-1 text-xs font-medium transition-colors ${
                    showContext
                      ? "text-indigo-300 bg-indigo-900/30"
                      : "text-gray-500 hover:text-gray-400"
                  }`}
                  title="Add extra direction for regen"
                >
                  +
                </button>
              )}
            </>
          )}
          <CopyButton text={value} />
          <div className="mx-1 h-4 w-px bg-gray-800" />
          <button
            type="button"
            onClick={() => setShowPreview(!showPreview)}
            className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
              showPreview
                ? "bg-gray-800 text-gray-300"
                : "text-gray-500 hover:text-gray-400"
            }`}
          >
            Preview
          </button>
          <button
            onClick={onClose}
            className="rounded px-3 py-1 text-sm font-medium text-indigo-400 hover:bg-indigo-900/30 transition-colors"
          >
            Done
          </button>
        </div>
      </div>

      {/* ---- Context row ---- */}
      {showContext && onExtraContextChange && (
        <div className="flex items-center gap-2 border-b border-gray-800 bg-gray-900/50 px-4 py-1.5">
          <input
            type="text"
            placeholder="Extra direction for regen (optional)..."
            value={extraContext ?? ""}
            onChange={(e) => onExtraContextChange(e.target.value)}
            className="flex-1 rounded border border-gray-700 bg-gray-950 px-2.5 py-1 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>
      )}

      {/* ---- Formatting toolbar ---- */}
      <div className="flex items-center gap-0.5 border-b border-gray-800 bg-gray-900/50 px-3 py-1">
        {/* Bold */}
        <ToolbarButton
          onClick={cmAction((v) => wrapSelection(v, "**", "**"))}
          title="Bold (Ctrl+B)"
        >
          <span className="text-xs font-bold">B</span>
        </ToolbarButton>
        {/* Italic */}
        <ToolbarButton
          onClick={cmAction((v) => wrapSelection(v, "*", "*"))}
          title="Italic (Ctrl+I)"
        >
          <span className="text-xs italic">I</span>
        </ToolbarButton>
        {/* Strikethrough */}
        <ToolbarButton
          onClick={cmAction((v) => wrapSelection(v, "~~", "~~"))}
          title="Strikethrough"
        >
          <span className="text-xs line-through">S</span>
        </ToolbarButton>
        {/* Inline code */}
        <ToolbarButton
          onClick={cmAction((v) => wrapSelection(v, "`", "`"))}
          title="Inline code (Ctrl+Shift+K)"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
          </svg>
        </ToolbarButton>

        <ToolbarButton separator />

        {/* Heading */}
        <ToolbarButton
          onClick={cmAction((v) => toggleLinePrefix(v, "# "))}
          title="Heading"
        >
          <span className="text-xs font-bold">H</span>
        </ToolbarButton>
        {/* Bullet list */}
        <ToolbarButton
          onClick={cmAction((v) => toggleLinePrefix(v, "- "))}
          title="Bullet list"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 6.75h12M8.25 12h12M8.25 17.25h12M3.75 6.75h.007v.008H3.75V6.75zm0 5.25h.007v.008H3.75V12zm0 5.25h.007v.008H3.75v-.008z" />
          </svg>
        </ToolbarButton>
        {/* Numbered list */}
        <ToolbarButton
          onClick={cmAction((v) => toggleLinePrefix(v, "1. "))}
          title="Numbered list"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.242 5.992h12m-12 6.003h12m-12 5.999h12M4.117 7.495v-3.75H2.99m1.125 3.75H2.99m1.125 0H4.502m-1.384 5.999l.992-1.394c.374-.523-.07-.99-.64-.643l-.655.39m.002 1.647h1.297M2.984 19.993l1.234-.001" />
          </svg>
        </ToolbarButton>
        {/* Quote */}
        <ToolbarButton
          onClick={cmAction((v) => toggleLinePrefix(v, "> "))}
          title="Quote"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
          </svg>
        </ToolbarButton>

        <ToolbarButton separator />

        {/* Link */}
        <ToolbarButton
          onClick={cmAction((v) => {
            const { from, to } = v.state.selection.main;
            const selected = v.state.sliceDoc(from, to);
            const replacement = selected ? `[${selected}](url)` : "[text](url)";
            v.dispatch({
              changes: { from, to, insert: replacement },
              selection: {
                anchor: selected ? from + selected.length + 3 : from + 1,
                head: selected ? from + selected.length + 6 : from + 5,
              },
            });
            v.focus();
          })}
          title="Link"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
          </svg>
        </ToolbarButton>

        <ToolbarButton separator />

        {/* Undo */}
        <ToolbarButton
          onClick={cmAction((v) => undo(v))}
          title="Undo (Ctrl+Z)"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3" />
          </svg>
        </ToolbarButton>
        {/* Redo */}
        <ToolbarButton
          onClick={cmAction((v) => redo(v))}
          title="Redo (Ctrl+Shift+Z)"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l6-6m0 0l-6-6m6 6H9a6 6 0 000 12h3" />
          </svg>
        </ToolbarButton>

        <ToolbarButton separator />

        {/* Find & Replace */}
        <ToolbarButton
          onClick={cmAction((v) => openSearchPanel(v))}
          title="Find & Replace (Ctrl+H)"
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
        </ToolbarButton>
      </div>

      {/* ---- Editor body ---- */}
      <div className="flex flex-1 overflow-hidden">
        {/* Editor pane */}
        <div
          className={`flex flex-col overflow-hidden ${
            showPreview ? "w-1/2 border-r border-gray-800" : "w-full"
          }`}
        >
          {/* Tab bar */}
          <div className="flex items-center border-b border-gray-800 bg-gray-900/50">
            <div className="flex items-center gap-1.5 border-b-2 border-indigo-500 bg-gray-950 px-4 py-1.5">
              <svg
                className="h-3.5 w-3.5 text-gray-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
                />
              </svg>
              <span className="text-xs text-gray-300">{label}</span>
            </div>
          </div>

          {/* CodeMirror editor */}
          <div className="relative flex-1 overflow-hidden">
            <div ref={editorContainerRef} className="h-full" />

            {/* Regen overlay */}
            {regenerating && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-950/60">
                <div className="flex items-center gap-2 rounded-lg border border-indigo-800 bg-gray-900 px-4 py-2 text-sm text-indigo-400">
                  <Spinner className="h-4 w-4 text-indigo-400" />
                  <span>Regenerating...</span>
                </div>
              </div>
            )}
          </div>

          {/* Status bar */}
          <div className="flex items-center justify-between border-t border-gray-800 bg-gray-900/50 px-4 py-1">
            <span className="text-[10px] text-gray-600">Markdown</span>
            <span className="text-[10px] text-gray-600">
              Ln {lineCount} &middot; {wordCount} words &middot;{" "}
              {value.length} chars
            </span>
          </div>
        </div>

        {/* Preview pane */}
        {showPreview && (
          <div className="flex w-1/2 flex-col overflow-hidden">
            <div className="flex items-center border-b border-gray-800 bg-gray-900/50">
              <div className="flex items-center gap-1.5 border-b-2 border-gray-600 bg-gray-950 px-4 py-1.5">
                <svg
                  className="h-3.5 w-3.5 text-gray-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
                <span className="text-xs text-gray-400">Preview</span>
              </div>
            </div>
            <div className="flex-1 overflow-auto bg-gray-950 px-6 py-4 text-sm leading-relaxed text-gray-300 prose prose-invert prose-sm max-w-none prose-headings:text-gray-200 prose-p:text-gray-300 prose-a:text-indigo-400 prose-strong:text-gray-200 prose-code:text-indigo-300 prose-pre:bg-gray-900 prose-pre:border prose-pre:border-gray-800">
              {value ? (
                <Markdown>{value}</Markdown>
              ) : (
                <span className="text-gray-600 italic">
                  Nothing to preview
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
