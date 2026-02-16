import type { ManifestListItem } from "../api/types.ts";
import { StatusBadge } from "./StatusBadge.tsx";

interface ManifestCardProps {
  manifest: ManifestListItem;
  compact?: boolean;
  onEdit?: (id: string) => void;
  onView?: (id: string) => void;
  onDuplicate?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export function ManifestCard({
  manifest,
  compact = false,
  onEdit,
  onView,
  onDuplicate,
  onDelete,
}: ManifestCardProps) {
  const visibleTags = manifest.tags?.slice(0, 3) ?? [];
  const remainingTagCount = (manifest.tags?.length ?? 0) - visibleTags.length;

  return (
    <div
      onClick={onView ? () => onView(manifest.manifest_id) : undefined}
      className={
        compact
          ? "cursor-pointer rounded-lg border border-gray-800 bg-gray-900/50 p-3 transition-colors hover:border-gray-700"
          : "cursor-pointer rounded-lg border border-gray-800 bg-gray-900/50 p-4 transition-colors hover:border-gray-700"
      }
    >
      {/* Header row: Name + Status */}
      <div className="mb-2 flex items-start justify-between gap-2">
        <h3 className={compact ? "truncate text-base font-semibold text-gray-100" : "truncate text-lg font-semibold text-gray-100"}>
          {manifest.name}
        </h3>
        <StatusBadge status={manifest.status} />
      </div>

      {/* Description */}
      <p className={compact ? "mb-2 line-clamp-1 text-xs text-gray-400" : "mb-3 line-clamp-2 text-sm text-gray-400"}>
        {manifest.description || (
          <span className="italic text-gray-600">No description</span>
        )}
      </p>

      {/* Metadata row: Category + Asset count + Version */}
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded bg-gray-800 px-2 py-0.5 text-gray-300">
          {manifest.category}
        </span>
        <span className="text-gray-500">
          {manifest.asset_count} asset{manifest.asset_count !== 1 ? "s" : ""}
        </span>
        {manifest.version > 1 && (
          <span className="text-gray-500">v{manifest.version}</span>
        )}
      </div>

      {/* Tags - hidden in compact mode */}
      {!compact && manifest.tags && manifest.tags.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1.5">
          {visibleTags.map((tag) => (
            <span
              key={tag}
              className="rounded bg-gray-800/50 px-1.5 py-0.5 text-xs text-gray-400"
            >
              {tag}
            </span>
          ))}
          {remainingTagCount > 0 && (
            <span className="rounded bg-gray-800/50 px-1.5 py-0.5 text-xs text-gray-400">
              +{remainingTagCount} more
            </span>
          )}
        </div>
      )}

      {/* Footer: Action buttons - hidden in compact mode */}
      {!compact && (
        <div className="mt-3 border-t border-gray-800 pt-3">
          <div className="flex gap-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onView?.(manifest.manifest_id);
              }}
              className="text-sm text-blue-400 transition-colors hover:text-blue-300"
            >
              View
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onEdit?.(manifest.manifest_id);
              }}
              className="text-sm text-blue-400 transition-colors hover:text-blue-300"
            >
              Edit
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDuplicate?.(manifest.manifest_id);
              }}
              className="text-sm text-gray-400 transition-colors hover:text-gray-300"
            >
              Duplicate
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete?.(manifest.manifest_id);
              }}
              className="text-sm text-red-400 transition-colors hover:text-red-300"
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
