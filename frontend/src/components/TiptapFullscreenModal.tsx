import { useEffect } from "react";
import { ScenePromptEditor } from "./ScenePromptEditor.tsx";
import { CopyButton } from "./CopyButton.tsx";

interface TiptapFullscreenModalProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onClose: () => void;
}

export function TiptapFullscreenModal({
  label,
  value,
  onChange,
  onClose,
}: TiptapFullscreenModalProps) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown, true);
    return () => document.removeEventListener("keydown", handleKeyDown, true);
  }, [onClose]);

  const lineCount = value.split("\n").length;

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
          <CopyButton text={value} />
          <button
            onClick={onClose}
            className="rounded px-3 py-1 text-sm font-medium text-indigo-400 hover:bg-indigo-900/30 transition-colors"
          >
            Done
          </button>
        </div>
      </div>

      {/* Fullscreen Tiptap editor */}
      <div className="flex-1 overflow-hidden p-4">
        <ScenePromptEditor
          value={value}
          onChange={onChange}
          className="h-full [&_.ProseMirror]:min-h-full [&_.ProseMirror]:max-h-none [&_.ProseMirror]:resize-none"
        />
      </div>
    </div>
  );
}
