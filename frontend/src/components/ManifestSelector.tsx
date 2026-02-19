import { useState, useEffect } from "react";
import { StatusBadge } from "./StatusBadge.tsx";
import { getManifestDetail, listManifests } from "../api/client.ts";
import type { ManifestDetail, ManifestListItem } from "../api/types.ts";

interface ManifestSelectorProps {
  selectedManifestId: string | null;
  onManifestSelect: (manifestId: string | null) => void;
}

export function ManifestSelector({
  selectedManifestId,
  onManifestSelect,
}: ManifestSelectorProps) {
  const [selectedDetail, setSelectedDetail] = useState<ManifestDetail | null>(null);
  const [showPicker, setShowPicker] = useState(false);
  const [manifests, setManifests] = useState<ManifestListItem[]>([]);
  const [fetchingList, setFetchingList] = useState(false);
  const [search, setSearch] = useState("");

  const openPicker = async () => {
    setShowPicker(true);
    setSearch("");
    setFetchingList(true);
    try {
      const items = await listManifests({ sort_by: "updated_at", sort_order: "desc" });
      setManifests(items.filter((m) => m.status === "READY"));
    } catch (err) {
      console.error("Failed to fetch manifests:", err);
      setManifests([]);
    } finally {
      setFetchingList(false);
    }
  };

  const filteredManifests = manifests.filter((m) =>
    m.name.toLowerCase().includes(search.toLowerCase()),
  );

  // Fetch detail when external selection changes (e.g. initial value)
  useEffect(() => {
    if (!selectedManifestId) {
      setSelectedDetail(null);
      return;
    }
    async function fetchDetail() {
      try {
        const detail = await getManifestDetail(selectedManifestId!);
        setSelectedDetail(detail);
      } catch (err) {
        console.error("Failed to load manifest:", err);
        setSelectedDetail(null);
        onManifestSelect(null);
      }
    }
    fetchDetail();
  }, [selectedManifestId]);

  const handleRemove = () => {
    setSelectedDetail(null);
    onManifestSelect(null);
  };

  // Show selected manifest card
  if (selectedDetail) {
    return (
      <div className="rounded-lg border border-blue-500/30 bg-blue-500/5 p-4">
        <div className="flex items-start justify-between">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h4 className="truncate font-semibold text-gray-100">{selectedDetail.name}</h4>
              <StatusBadge status={selectedDetail.status} />
            </div>
            <p className="text-sm text-gray-400">
              {selectedDetail.asset_count} assets | {selectedDetail.category} | v{selectedDetail.version}
            </p>
            {selectedDetail.description && (
              <p className="mt-1 line-clamp-2 text-sm text-gray-500">{selectedDetail.description}</p>
            )}
          </div>
          <button
            type="button"
            onClick={handleRemove}
            className="ml-3 flex-shrink-0 rounded-full p-1 text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
            title="Remove manifest"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
        </div>
        {/* Asset thumbnails */}
        {selectedDetail.assets.length > 0 && (
          <div className="mt-3 flex gap-2 overflow-x-auto">
            {selectedDetail.assets
              .filter((a) => a.reference_image_url)
              .slice(0, 5)
              .map((asset) => (
                <div key={asset.asset_id} className="flex-shrink-0">
                  <img
                    src={asset.reference_image_url!}
                    alt={asset.name}
                    className="h-12 w-12 rounded object-cover"
                  />
                  <p className="mt-0.5 text-center text-[10px] text-gray-500">{asset.manifest_tag}</p>
                </div>
              ))}
            {selectedDetail.assets.filter((a) => a.reference_image_url).length > 5 && (
              <span className="flex h-12 w-12 items-center justify-center rounded bg-gray-800 text-xs text-gray-400">
                +{selectedDetail.assets.filter((a) => a.reference_image_url).length - 5}
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  // Show picker UI
  if (!showPicker) {
    return (
      <button
        type="button"
        onClick={openPicker}
        className="flex items-center gap-2 rounded-lg border border-dashed border-gray-700 px-4 py-3 text-sm text-gray-400 transition-colors hover:border-gray-500 hover:text-gray-300"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
          <path d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z" />
        </svg>
        Add
      </button>
    );
  }

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-900">
      {/* Search filter */}
      <div className="flex items-center gap-2 border-b border-gray-700 p-2">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter by name..."
          autoFocus
          className="flex-1 rounded border border-gray-700 bg-gray-800 px-2.5 py-1.5 text-sm text-gray-200 placeholder-gray-500 outline-none focus:border-blue-500"
        />
        <button
          type="button"
          onClick={() => setShowPicker(false)}
          className="rounded p-1 text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
          title="Cancel"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
            <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
          </svg>
        </button>
      </div>

      {/* List */}
      <div className="max-h-64 overflow-y-auto">
        {fetchingList ? (
          <div className="flex items-center justify-center py-6 text-sm text-gray-500">
            <svg className="mr-2 h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading...
          </div>
        ) : filteredManifests.length === 0 ? (
          <div className="py-6 text-center text-sm text-gray-500">
            {search ? "No manifests match your filter" : "No ready manifests found"}
          </div>
        ) : (
          filteredManifests.map((m) => (
            <button
              key={m.manifest_id}
              type="button"
              onClick={() => {
                onManifestSelect(m.manifest_id);
                setShowPicker(false);
              }}
              className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-gray-800"
            >
              {m.thumbnail_url ? (
                <img src={m.thumbnail_url} alt="" className="h-8 w-8 flex-shrink-0 rounded object-cover" />
              ) : (
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded bg-gray-800 text-xs text-gray-500">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                    <path fillRule="evenodd" d="M1 5.25A2.25 2.25 0 013.25 3h13.5A2.25 2.25 0 0119 5.25v9.5A2.25 2.25 0 0116.75 17H3.25A2.25 2.25 0 011 14.75v-9.5zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 00.75-.75v-2.69l-2.22-2.219a.75.75 0 00-1.06 0l-1.91 1.909-4.97-4.969a.75.75 0 00-1.06 0L2.5 11.06z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium text-gray-200">{m.name}</span>
                  <StatusBadge status={m.status} />
                </div>
                <p className="text-xs text-gray-500">
                  {m.asset_count} asset{m.asset_count !== 1 ? "s" : ""} · {m.category}
                </p>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
