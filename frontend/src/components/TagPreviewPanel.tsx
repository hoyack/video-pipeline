import type { BoundAssetSummary } from "../api/types.ts";

interface TagPreviewPanelProps {
  asset: BoundAssetSummary | null;
  onClose: () => void;
}

const TYPE_COLORS: Record<string, string> = {
  CHARACTER: "bg-blue-900/50 text-blue-300 border-blue-700",
  SET: "bg-green-900/50 text-green-300 border-green-700",
  PROP: "bg-amber-900/50 text-amber-300 border-amber-700",
};

export function TagPreviewPanel({ asset, onClose }: TagPreviewPanelProps) {
  if (!asset) return null;

  const badgeClass =
    TYPE_COLORS[asset.type] ??
    "bg-gray-800 text-gray-300 border-gray-700";

  return (
    <div className="fixed right-0 top-0 z-40 flex h-full w-[280px] flex-col border-l border-gray-800 bg-gray-900 p-4 shadow-xl">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
          Tag Preview
        </span>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition-colors"
          aria-label="Close preview"
        >
          <svg
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Thumbnail */}
      {asset.primary_thumbnail_url && (
        <img
          src={asset.primary_thumbnail_url}
          alt={asset.name}
          className="mb-4 w-full rounded-lg object-cover"
        />
      )}

      {/* Name */}
      <h3 className="text-sm font-bold text-gray-200">{asset.name}</h3>

      {/* Tag */}
      <p className="mt-1 font-mono text-xs text-gray-500">
        @{asset.tag.toLowerCase()}
      </p>

      {/* Type badge */}
      <span
        className={`mt-3 inline-flex w-fit items-center rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${badgeClass}`}
      >
        {asset.type}
      </span>

      {/* Description */}
      {asset.description && (
        <p className="mt-4 text-xs leading-relaxed text-gray-400">
          {asset.description}
        </p>
      )}
    </div>
  );
}
