import { useEffect, useState, useRef, useCallback } from "react";
import Markdown from "react-markdown";
import { CopyButton } from "./CopyButton.tsx";

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

function Spinner({ className }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className ?? "h-3 w-3"}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

export function MarkdownEditorModal({ label, value, onChange, onClose, onRegen, regenerating, extraContext, onExtraContextChange }: MarkdownEditorModalProps) {
  const [showPreview, setShowPreview] = useState(true);
  const [showContext, setShowContext] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Sync line numbers scroll with textarea
  const handleScroll = useCallback(() => {
    if (textareaRef.current && lineNumbersRef.current) {
      lineNumbersRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  }, []);

  const lineCount = value.split("\n").length;

  // Handle Tab key for indentation
  function handleKeyDownTextarea(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = e.currentTarget;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const newValue = value.substring(0, start) + "  " + value.substring(end);
      onChange(newValue);
      // Restore cursor position after React re-render
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 2;
      });
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-gray-950">
      {/* Title bar */}
      <div className="flex items-center justify-between border-b border-gray-800 bg-gray-900 px-4 py-2">
        <div className="flex items-center gap-3">
          <svg className="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931z" />
          </svg>
          <span className="text-sm font-medium text-gray-300">{label}</span>
          <span className="text-xs text-gray-600">
            {lineCount} line{lineCount !== 1 ? "s" : ""} &middot; {value.length} chars
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
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
                  </svg>
                )}
                {regenerating ? "Generating..." : "Regen"}
              </button>
              {onExtraContextChange && (
                <button
                  type="button"
                  onClick={() => setShowContext(!showContext)}
                  className={`rounded px-1.5 py-1 text-xs font-medium transition-colors ${
                    showContext ? "text-indigo-300 bg-indigo-900/30" : "text-gray-500 hover:text-gray-400"
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

      {/* Collapsible context row */}
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

      {/* Editor body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Editor pane */}
        <div className={`flex flex-col overflow-hidden ${showPreview ? "w-1/2 border-r border-gray-800" : "w-full"}`}>
          {/* Tab bar */}
          <div className="flex items-center border-b border-gray-800 bg-gray-900/50">
            <div className="flex items-center gap-1.5 border-b-2 border-indigo-500 bg-gray-950 px-4 py-1.5">
              <svg className="h-3.5 w-3.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
              <span className="text-xs text-gray-300">{label}</span>
            </div>
          </div>

          {/* Code area with line numbers */}
          <div className="relative flex flex-1 overflow-hidden">
            {/* Line numbers gutter */}
            <div
              ref={lineNumbersRef}
              className="select-none overflow-hidden bg-gray-900/80 py-2 text-right font-mono text-xs leading-relaxed text-gray-600 border-r border-gray-800"
              style={{ minWidth: `${Math.max(3, String(lineCount).length + 1)}ch`, paddingLeft: "0.5rem", paddingRight: "0.75rem" }}
            >
              {Array.from({ length: lineCount }, (_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>

            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onScroll={handleScroll}
              onKeyDown={handleKeyDownTextarea}
              spellCheck={false}
              className="flex-1 resize-none overflow-auto bg-gray-950 py-2 pl-4 pr-4 font-mono text-sm leading-relaxed text-gray-300 caret-indigo-400 focus:outline-none"
              autoFocus
            />

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
              Ln {lineCount}, Col {value.length - value.lastIndexOf("\n")}
            </span>
          </div>
        </div>

        {/* Preview pane */}
        {showPreview && (
          <div className="flex w-1/2 flex-col overflow-hidden">
            {/* Preview tab bar */}
            <div className="flex items-center border-b border-gray-800 bg-gray-900/50">
              <div className="flex items-center gap-1.5 border-b-2 border-gray-600 bg-gray-950 px-4 py-1.5">
                <svg className="h-3.5 w-3.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="text-xs text-gray-400">Preview</span>
              </div>
            </div>

            {/* Preview content */}
            <div className="flex-1 overflow-auto bg-gray-950 px-6 py-4 text-sm leading-relaxed text-gray-300 prose prose-invert prose-sm max-w-none prose-headings:text-gray-200 prose-p:text-gray-300 prose-a:text-indigo-400 prose-strong:text-gray-200 prose-code:text-indigo-300 prose-pre:bg-gray-900 prose-pre:border prose-pre:border-gray-800">
              {value ? (
                <Markdown>{value}</Markdown>
              ) : (
                <span className="text-gray-600 italic">Nothing to preview</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
