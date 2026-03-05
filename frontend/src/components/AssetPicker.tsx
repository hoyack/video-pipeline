import { useState, useEffect, useRef } from "react";
import type {
  ActorListItem,
  LibrarySetListItem,
  LibraryPropListItem,
  SoundAssetListItem,
} from "../api/types.ts";
import {
  listActors,
  listLibrarySets,
  listLibraryProps,
  listSoundAssets,
} from "../api/client.ts";

export interface AssetPickerProps {
  isOpen: boolean;
  onClose: () => void;
  assetType: "actor" | "set" | "prop" | "sound";
  onSelect: (asset: { id: string; name: string; primary_ref_url?: string | null }) => void;
  excludeIds?: string[];
}

type AnyListItem = ActorListItem | LibrarySetListItem | LibraryPropListItem | SoundAssetListItem;

const ASSET_TYPE_LABELS: Record<string, string> = {
  actor: "Actor",
  set: "Set",
  prop: "Prop",
  sound: "Sound Asset",
};

export function AssetPicker({ isOpen, onClose, assetType, onSelect, excludeIds = [] }: AssetPickerProps) {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [items, setItems] = useState<AnyListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounce search input
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setDebouncedSearch(search), 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [search]);

  // Fetch items when modal opens or search changes
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    setLoading(true);

    const fetchItems = async () => {
      try {
        let result: AnyListItem[];
        const q = debouncedSearch || undefined;
        switch (assetType) {
          case "actor":
            result = await listActors(q);
            break;
          case "set":
            result = await listLibrarySets(q);
            break;
          case "prop":
            result = await listLibraryProps(q);
            break;
          case "sound":
            result = await listSoundAssets(q);
            break;
        }
        if (!cancelled) setItems(result);
      } catch {
        if (!cancelled) setItems([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchItems();
    return () => { cancelled = true; };
  }, [isOpen, assetType, debouncedSearch]);

  // Reset search on open
  useEffect(() => {
    if (isOpen) {
      setSearch("");
      setDebouncedSearch("");
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const excludeSet = new Set(excludeIds);
  const filtered = items.filter((item) => !excludeSet.has(item.id));

  const handleSelect = (item: AnyListItem) => {
    const refUrl = "primary_ref_url" in item ? item.primary_ref_url : null;
    onSelect({ id: item.id, name: item.name, primary_ref_url: refUrl });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-full max-w-2xl max-h-[80vh] flex flex-col rounded-xl border border-gray-700 bg-gray-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-700 px-6 py-4">
          <h2 className="text-lg font-semibold text-white">
            Select {ASSET_TYPE_LABELS[assetType]}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl leading-none"
          >
            &times;
          </button>
        </div>

        {/* Search */}
        <div className="px-6 py-3 border-b border-gray-800">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={`Search ${ASSET_TYPE_LABELS[assetType].toLowerCase()}s...`}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder-gray-500 outline-none focus:border-blue-500"
            autoFocus
          />
        </div>

        {/* Grid/list */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <span className="text-sm text-gray-500">
                {items.length > 0
                  ? "All items already added"
                  : `No ${ASSET_TYPE_LABELS[assetType].toLowerCase()}s found`}
              </span>
            </div>
          ) : assetType === "sound" ? (
            /* Sound assets: table layout */
            <div className="space-y-2">
              {(filtered as SoundAssetListItem[]).map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleSelect(item)}
                  className="w-full flex items-center gap-3 rounded-lg border border-gray-800 bg-gray-800/50 px-4 py-3 text-left transition-colors hover:border-blue-500 hover:bg-gray-800"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">{item.name}</div>
                    {item.description && (
                      <div className="text-xs text-gray-400 truncate mt-0.5">{item.description}</div>
                    )}
                  </div>
                  <span className="shrink-0 rounded-full bg-purple-900/50 px-2 py-0.5 text-xs text-purple-300">
                    {item.category}
                  </span>
                  {item.has_audio && (
                    <span className="shrink-0 text-xs text-green-400">has audio</span>
                  )}
                </button>
              ))}
            </div>
          ) : (
            /* Actor/Set/Prop: card grid */
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {filtered.map((item) => {
                const refUrl = "primary_ref_url" in item ? (item as ActorListItem).primary_ref_url : null;
                const refCount = "ref_count" in item ? (item as ActorListItem).ref_count : undefined;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleSelect(item)}
                    className="flex flex-col items-center rounded-lg border border-gray-800 bg-gray-800/50 p-3 transition-colors hover:border-blue-500 hover:bg-gray-800"
                  >
                    {refUrl ? (
                      <img
                        src={refUrl}
                        alt={item.name}
                        className="w-20 h-20 rounded-lg object-cover border border-gray-700 mb-2"
                      />
                    ) : (
                      <div className="w-20 h-20 rounded-lg bg-gray-700 border border-gray-600 mb-2 flex items-center justify-center text-gray-500 text-xs">
                        No ref
                      </div>
                    )}
                    <span className="text-sm font-medium text-white text-center truncate w-full">
                      {item.name}
                    </span>
                    {refCount !== undefined && refCount > 0 && (
                      <span className="text-xs text-gray-400 mt-0.5">{refCount} ref{refCount !== 1 ? "s" : ""}</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-800 px-6 py-3 flex justify-between items-center">
          <a
            href="/asset-library"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            Open Asset Library
          </a>
          <button
            onClick={onClose}
            className="rounded-lg bg-gray-800 px-4 py-1.5 text-sm text-gray-300 hover:text-white transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
