import { useState, useEffect, useMemo } from "react";
import type { BoundAssetSummary } from "../api/types.ts";
import { getBoundAssetsSummary } from "../api/client.ts";

interface TagReferenceSheetProps {
  bibleId: string;
}

const TYPE_BADGE_COLORS: Record<string, string> = {
  CHARACTER: "bg-blue-900/50 text-blue-300",
  SET: "bg-green-900/50 text-green-300",
  PROP: "bg-amber-900/50 text-amber-300",
};

export function TagReferenceSheet({ bibleId }: TagReferenceSheetProps) {
  const [assets, setAssets] = useState<BoundAssetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    getBoundAssetsSummary(bibleId)
      .then((data) => {
        if (!cancelled) setAssets(data);
      })
      .catch(() => {
        if (!cancelled) setAssets([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [bibleId]);

  const filtered = useMemo(() => {
    if (!filter.trim()) return assets;
    const q = filter.toLowerCase();
    return assets.filter(
      (a) =>
        a.tag.toLowerCase().includes(q) ||
        a.name.toLowerCase().includes(q) ||
        a.type.toLowerCase().includes(q)
    );
  }, [assets, filter]);

  if (loading) {
    return (
      <div className="flex min-h-[120px] items-center justify-center">
        <p className="text-sm text-gray-500">Loading tag reference...</p>
      </div>
    );
  }

  if (assets.length === 0) {
    return (
      <div className="flex min-h-[120px] items-center justify-center rounded-lg border border-dashed border-gray-700">
        <p className="text-sm text-gray-500 text-center px-4">
          No assets bound to this Production Bible. Add cast, set, and prop
          bindings in the Casting and Art Department tabs.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Search / filter input */}
      <div className="mb-4">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by tag, name, or type..."
          className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-500 outline-none focus:border-blue-500"
        />
      </div>

      {/* Results table */}
      {filtered.length === 0 ? (
        <div className="flex min-h-[80px] items-center justify-center rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-500">
            No assets match "{filter}".
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-gray-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-900/80 text-left text-xs uppercase tracking-wide text-gray-500">
                <th className="px-4 py-2.5">Tag</th>
                <th className="px-4 py-2.5 w-12"></th>
                <th className="px-4 py-2.5">Type</th>
                <th className="px-4 py-2.5">Name</th>
                <th className="px-4 py-2.5">Description</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((asset) => (
                <tr
                  key={asset.tag}
                  className="border-b border-gray-800/50 transition-colors hover:bg-gray-800/50"
                >
                  <td className="px-4 py-2.5">
                    <span className="font-mono text-indigo-400">
                      @{asset.tag.toLowerCase()}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    {asset.primary_thumbnail_url ? (
                      <img
                        src={asset.primary_thumbnail_url}
                        alt={asset.name}
                        className="h-10 w-10 rounded object-cover"
                      />
                    ) : (
                      <div className="flex h-10 w-10 items-center justify-center rounded bg-gray-700 text-gray-500">
                        <TypeIcon type={asset.type} />
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        TYPE_BADGE_COLORS[asset.type] ||
                        "bg-gray-700 text-gray-300"
                      }`}
                    >
                      {asset.type}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 font-medium text-white">
                    {asset.name}
                  </td>
                  <td className="px-4 py-2.5 text-gray-400">
                    {asset.description
                      ? asset.description.length > 100
                        ? asset.description.slice(0, 100) + "..."
                        : asset.description
                      : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Simple SVG icons per asset type for the thumbnail placeholder */
function TypeIcon({ type }: { type: string }) {
  switch (type) {
    case "CHARACTER":
      return (
        <svg
          className="h-5 w-5"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" />
        </svg>
      );
    case "SET":
      return (
        <svg
          className="h-5 w-5"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
        </svg>
      );
    case "PROP":
      return (
        <svg
          className="h-5 w-5"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M6 6V5a3 3 0 013-3h2a3 3 0 013 3v1h2a2 2 0 012 2v3.57A22.952 22.952 0 0110 13a22.95 22.95 0 01-8-1.43V8a2 2 0 012-2h2zm2-1a1 1 0 011-1h2a1 1 0 011 1v1H8V5zm1 5a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z"
            clipRule="evenodd"
          />
          <path d="M2 13.692V16a2 2 0 002 2h12a2 2 0 002-2v-2.308A24.974 24.974 0 0110 15c-2.796 0-5.487-.46-8-1.308z" />
        </svg>
      );
    default:
      return (
        <svg
          className="h-5 w-5"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
            clipRule="evenodd"
          />
        </svg>
      );
  }
}
