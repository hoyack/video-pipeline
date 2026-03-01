import { useEffect, useRef, useState } from "react";
import { ColorPicker } from "./ColorPicker.tsx";

interface SequenceContextMenuProps {
  sequenceId: string;
  color: string | null;
  act: string | null;
  onEdit: () => void;
  onDelete: (id: string) => void;
  onColorChange: (color: string) => void;
  onActChange: (act: string | null) => void;
}

const ACT_OPTIONS: { label: string; value: string | null }[] = [
  { label: "Act 1", value: "ACT_1" },
  { label: "Act 2", value: "ACT_2" },
  { label: "Act 3", value: "ACT_3" },
  { label: "None", value: null },
];

export function SequenceContextMenu({
  sequenceId,
  color,
  act,
  onEdit,
  onDelete,
  onColorChange,
  onActChange,
}: SequenceContextMenuProps) {
  const [open, setOpen] = useState(false);
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [showActPicker, setShowActPicker] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
        setShowColorPicker(false);
        setShowActPicker(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  function handleEdit() {
    setOpen(false);
    onEdit();
  }

  function handleDelete() {
    if (window.confirm("Delete this sequence? Scenes will be moved to Unsequenced.")) {
      setOpen(false);
      onDelete(sequenceId);
    }
  }

  function handleColorSelect(c: string) {
    onColorChange(c);
    setShowColorPicker(false);
    setOpen(false);
  }

  function handleActSelect(value: string | null) {
    onActChange(value);
    setShowActPicker(false);
    setOpen(false);
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
          setShowColorPicker(false);
          setShowActPicker(false);
        }}
        className="rounded p-1 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
        title="Sequence options"
      >
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 top-7 z-50 min-w-[160px] rounded-lg border border-gray-700 bg-gray-800 shadow-xl py-1">
          <button
            onClick={handleEdit}
            className="w-full text-left px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 transition-colors"
          >
            Edit title
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowColorPicker((v) => !v);
              setShowActPicker(false);
            }}
            className="w-full text-left px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 transition-colors"
          >
            Change color
          </button>
          {showColorPicker && (
            <div className="border-t border-gray-700 px-1 py-1">
              <ColorPicker selectedColor={color} onColorSelect={handleColorSelect} />
            </div>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowActPicker((v) => !v);
              setShowColorPicker(false);
            }}
            className="w-full text-left px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 transition-colors"
          >
            Set act
          </button>
          {showActPicker && (
            <div className="border-t border-gray-700 px-2 py-1 flex flex-wrap gap-1">
              {ACT_OPTIONS.map((opt) => (
                <button
                  key={opt.label}
                  onClick={() => handleActSelect(opt.value)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    act === opt.value
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}
          <div className="border-t border-gray-700 my-1" />
          <button
            onClick={handleDelete}
            className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-gray-700 transition-colors"
          >
            Delete sequence
          </button>
        </div>
      )}
    </div>
  );
}
