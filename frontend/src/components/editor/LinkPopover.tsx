import { useState, useRef, useEffect } from "react";

interface LinkPopoverProps {
  initialUrl: string;
  onSubmit: (url: string) => void;
  onRemove: () => void;
  onClose: () => void;
}

export function LinkPopover({ initialUrl, onSubmit, onRemove, onClose }: LinkPopoverProps) {
  const [url, setUrl] = useState(initialUrl);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

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

  const handleApply = () => {
    const trimmed = url.trim();
    if (trimmed) {
      onSubmit(trimmed);
      onClose();
    }
  };

  const handleRemove = () => {
    onRemove();
    onClose();
  };

  return (
    <div
      ref={containerRef}
      className="absolute top-full left-0 z-50 mt-1 w-72 rounded-lg border border-gray-700 bg-gray-800 p-3 shadow-xl"
    >
      <label className="mb-1 block text-[11px] font-medium text-gray-400">Link URL</label>
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleApply(); }}
          placeholder="https://..."
          className="flex-1 rounded-lg border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-gray-100 outline-none focus:border-blue-500"
        />
        <button
          onClick={handleApply}
          disabled={!url.trim()}
          className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
        >
          Apply
        </button>
      </div>

      {initialUrl && (
        <button
          onClick={handleRemove}
          className="mt-2 w-full rounded-lg border border-red-900/50 py-1.5 text-xs text-red-400 hover:bg-red-900/20"
        >
          Remove link
        </button>
      )}
    </div>
  );
}
