import { useRef, useState } from "react";
import type { SequenceResponse, SequenceUpdate } from "../api/types.ts";
import { SequenceContextMenu } from "./SequenceContextMenu.tsx";

interface SequenceHeaderProps {
  sequence: SequenceResponse;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onUpdate: (updates: SequenceUpdate) => void;
  onDelete: (id: string) => void;
}

function formatDuration(seconds: number): string {
  if (seconds >= 60) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  }
  return `${seconds}s`;
}

export function SequenceHeader({
  sequence,
  isCollapsed,
  onToggleCollapse,
  onUpdate,
  onDelete,
}: SequenceHeaderProps) {
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(sequence.title);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDoubleClick() {
    setEditTitle(sequence.title);
    setEditing(true);
    // Focus after next render
    setTimeout(() => inputRef.current?.focus(), 0);
  }

  function handleSaveTitle() {
    const trimmed = editTitle.trim();
    if (trimmed && trimmed !== sequence.title) {
      onUpdate({ title: trimmed });
    }
    setEditing(false);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleSaveTitle();
    if (e.key === "Escape") {
      setEditTitle(sequence.title);
      setEditing(false);
    }
  }

  const dotColor = sequence.color ?? "#6B7280";

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/80 border border-gray-700 select-none">
      {/* Color dot */}
      <div
        className="w-3 h-3 rounded-full flex-shrink-0"
        style={{ backgroundColor: dotColor }}
      />

      {/* Title */}
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            ref={inputRef}
            type="text"
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onBlur={handleSaveTitle}
            onKeyDown={handleKeyDown}
            className="bg-gray-900 text-white text-sm font-semibold px-2 py-0.5 rounded border border-blue-500 w-full outline-none"
          />
        ) : (
          <span
            className="text-sm font-semibold text-white cursor-text truncate block"
            onDoubleClick={handleDoubleClick}
            title="Double-click to edit"
          >
            {sequence.title}
          </span>
        )}
      </div>

      {/* Scene count badge */}
      <span className="text-xs text-gray-400 bg-gray-700 rounded-full px-2 py-0.5 flex-shrink-0">
        {sequence.scene_count} {sequence.scene_count === 1 ? "scene" : "scenes"}
      </span>

      {/* Duration badge */}
      {sequence.total_duration != null && sequence.total_duration > 0 && (
        <span className="text-xs text-gray-400 flex-shrink-0">
          {formatDuration(sequence.total_duration)}
        </span>
      )}

      {/* Act label */}
      {sequence.act && (
        <span className="text-xs text-blue-400 bg-blue-900/40 rounded px-1.5 py-0.5 flex-shrink-0">
          {sequence.act.replace("_", " ")}
        </span>
      )}

      {/* Context menu */}
      <SequenceContextMenu
        sequenceId={sequence.id}
        color={sequence.color}
        act={sequence.act}
        onEdit={() => {
          setEditTitle(sequence.title);
          setEditing(true);
          setTimeout(() => inputRef.current?.focus(), 0);
        }}
        onDelete={onDelete}
        onColorChange={(color) => onUpdate({ color })}
        onActChange={(act) => onUpdate({ act })}
      />

      {/* Collapse toggle */}
      <button
        onClick={onToggleCollapse}
        className="rounded p-1 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
        title={isCollapsed ? "Expand" : "Collapse"}
      >
        <svg
          className={`w-4 h-4 transition-transform ${isCollapsed ? "-rotate-90" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
    </div>
  );
}
