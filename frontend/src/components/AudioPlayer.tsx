export function AudioPlayer({ src, label }: { src: string; label?: string }) {
  if (!src) return null;
  return (
    <div className="flex items-center gap-2 rounded border border-gray-700 bg-gray-800 px-3 py-2 mt-1">
      {label && <span className="text-xs text-gray-400 flex-shrink-0">{label}</span>}
      <audio controls src={src} className="h-8 w-full" preload="none" />
    </div>
  );
}
