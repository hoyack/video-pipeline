import { useState, useRef, useEffect } from "react";

interface ImagePopoverProps {
  onInsertUrl: (url: string) => void;
  onUploadFile: (file: File) => void;
  onClose: () => void;
}

export function ImagePopover({ onInsertUrl, onUploadFile, onClose }: ImagePopoverProps) {
  const [url, setUrl] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Close on Escape or click outside
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    const handleClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKey);
    document.addEventListener("mousedown", handleClick, true);
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.removeEventListener("mousedown", handleClick, true);
    };
  }, [onClose]);

  const handleInsert = () => {
    const trimmed = url.trim();
    if (!trimmed) return;
    if (!trimmed.startsWith("https://") && !trimmed.startsWith("http://") && !trimmed.startsWith("/")) {
      return;
    }
    onInsertUrl(trimmed);
    onClose();
  };

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      return;
    }
    onUploadFile(file);
    onClose();
  };

  return (
    <div
      ref={containerRef}
      className="absolute top-full left-0 z-50 mt-1 w-72 rounded-lg border border-gray-700 bg-gray-800 p-3 shadow-xl"
    >
      <label className="mb-1 block text-[11px] font-medium text-gray-400">Image URL</label>
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleInsert(); }}
          placeholder="https://..."
          className="flex-1 rounded-lg border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-gray-100 outline-none focus:border-blue-500"
        />
        <button
          onClick={handleInsert}
          disabled={!url.trim()}
          className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
        >
          Insert
        </button>
      </div>

      <div className="my-2 flex items-center gap-2">
        <div className="h-px flex-1 bg-gray-700" />
        <span className="text-[10px] text-gray-500">or</span>
        <div className="h-px flex-1 bg-gray-700" />
      </div>

      <button
        onClick={() => fileRef.current?.click()}
        className="w-full rounded-lg border border-dashed border-gray-600 py-2 text-xs text-gray-400 hover:border-gray-500 hover:text-gray-300"
      >
        Upload from file...
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={handleFile}
      />
    </div>
  );
}
