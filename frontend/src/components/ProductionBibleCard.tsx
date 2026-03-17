import { useState } from "react";
import type { ProductionBibleListItem } from "../api/types.ts";
import { StatusBadge } from "./StatusBadge.tsx";

interface ProductionBibleCardProps {
  manifest: ProductionBibleListItem;
  compact?: boolean;
  onEdit?: (id: string) => void;
  onView?: (id: string) => void;
  onDuplicate?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export function ProductionBibleCard({
  manifest,
  compact = false,
  onEdit,
  onView,
  onDuplicate,
  onDelete,
}: ProductionBibleCardProps) {
  const visibleTags = manifest.tags?.slice(0, 3) ?? [];
  const remainingTagCount = (manifest.tags?.length ?? 0) - visibleTags.length;
  const [copied, setCopied] = useState(false);

  const handleCopyId = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(manifest.production_bible_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div
      onClick={onView ? () => onView(manifest.production_bible_id) : undefined}
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

      {/* Metadata row: Asset count + Version */}
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-gray-500">
          {manifest.asset_count} asset{manifest.asset_count !== 1 ? "s" : ""}
        </span>
        {manifest.version > 1 && (
          <span className="text-gray-500">v{manifest.version}</span>
        )}
      </div>

      {/* Production Bible ID with copy - hidden in compact mode */}
      {!compact && (
        <div className="mb-2 flex items-center gap-1.5">
          <span className="font-mono text-xs text-gray-600">
            {manifest.production_bible_id.slice(0, 8)}...
          </span>
          <button
            onClick={handleCopyId}
            className="text-xs text-gray-600 transition-colors hover:text-gray-400"
            title="Copy production bible ID"
          >
            {copied ? "Copied!" : "Copy ID"}
          </button>
        </div>
      )}

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
                onView?.(manifest.production_bible_id);
              }}
              className="text-sm text-blue-400 transition-colors hover:text-blue-300"
            >
              View
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onEdit?.(manifest.production_bible_id);
              }}
              className="text-sm text-blue-400 transition-colors hover:text-blue-300"
            >
              Edit
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDuplicate?.(manifest.production_bible_id);
              }}
              className="text-sm text-gray-400 transition-colors hover:text-gray-300"
            >
              Duplicate
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete?.(manifest.production_bible_id);
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
