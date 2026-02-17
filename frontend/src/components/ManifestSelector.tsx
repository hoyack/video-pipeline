import { useState, useEffect } from "react";
import clsx from "clsx";
import { ManifestCard } from "./ManifestCard.tsx";
import { StatusBadge } from "./StatusBadge.tsx";
import { listManifests, getManifestDetail } from "../api/client.ts";
import type { ManifestListItem, ManifestDetail } from "../api/types.ts";

interface ManifestSelectorProps {
  selectedManifestId: string | null;
  onManifestSelect: (manifestId: string | null) => void;
}

export function ManifestSelector({
  selectedManifestId,
  onManifestSelect,
}: ManifestSelectorProps) {
  const [mode, setMode] = useState<"existing" | "quick">("existing");
  const [manifests, setManifests] = useState<ManifestListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDetail, setSelectedDetail] = useState<ManifestDetail | null>(null);

  // Fetch READY manifests on mount
  useEffect(() => {
    async function fetchManifests() {
      setLoading(true);
      try {
        const data = await listManifests({
          sort_by: "last_used_at",
          sort_order: "desc",
        });
        // Filter to READY only (backend doesn't filter by status yet)
        const readyManifests = data.filter((m) => m.status === "READY");
        setManifests(readyManifests);
      } catch (err) {
        console.error("Failed to load manifests:", err);
        setManifests([]);
      } finally {
        setLoading(false);
      }
    }
    fetchManifests();
  }, []);

  // Fetch detail when selection changes
  useEffect(() => {
    if (!selectedManifestId) {
      setSelectedDetail(null);
      return;
    }
    async function fetchDetail() {
      try {
        const detail = await getManifestDetail(selectedManifestId);
        setSelectedDetail(detail);
      } catch (err) {
        console.error("Failed to load manifest detail:", err);
        setSelectedDetail(null);
      }
    }
    fetchDetail();
  }, [selectedManifestId]);

  // Clear selection when switching to quick mode
  function handleModeChange(newMode: "existing" | "quick") {
    setMode(newMode);
    if (newMode === "quick") {
      onManifestSelect(null);
    }
  }

  return (
    <div className="space-y-4">
      {/* Radio toggle */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => handleModeChange("existing")}
          className={clsx(
            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
            mode === "existing"
              ? "border-blue-500 bg-blue-500/20 text-blue-300"
              : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
          )}
        >
          Select Existing
        </button>
        <button
          type="button"
          onClick={() => handleModeChange("quick")}
          className={clsx(
            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
            mode === "quick"
              ? "border-blue-500 bg-blue-500/20 text-blue-300"
              : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
          )}
        >
          Quick Upload (inline)
        </button>
      </div>

      {/* Content area */}
      {mode === "existing" ? (
        selectedDetail ? (
          // Selected manifest preview
          <div className="rounded-lg border border-blue-500/30 bg-blue-500/5 p-4">
            <div className="flex items-start justify-between">
              <div>
                <h4 className="font-semibold text-gray-100">{selectedDetail.name}</h4>
                <p className="text-sm text-gray-400">
                  {selectedDetail.asset_count} assets | {selectedDetail.category} | v{selectedDetail.version}
                </p>
                {selectedDetail.description && (
                  <p className="mt-1 line-clamp-2 text-sm text-gray-500">{selectedDetail.description}</p>
                )}
              </div>
              <StatusBadge status={selectedDetail.status} />
            </div>
            {/* Key asset thumbnails - show first 5 assets with images */}
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
            <button
              onClick={() => onManifestSelect(null)}
              className="mt-3 text-sm text-gray-400 hover:text-gray-300"
            >
              Change Manifest
            </button>
          </div>
        ) : (
          // Manifest selection grid
          <div>
            {loading ? (
              <p className="text-sm text-gray-500">Loading manifests...</p>
            ) : manifests.length === 0 ? (
              <p className="text-sm text-gray-500">
                No ready manifests found. Create one in the Manifest Library first.
              </p>
            ) : (
              <div className="grid max-h-64 grid-cols-2 gap-3 overflow-y-auto">
                {manifests.slice(0, 6).map((m) => (
                  <button
                    key={m.manifest_id}
                    type="button"
                    onClick={() => onManifestSelect(m.manifest_id)}
                    className="text-left"
                  >
                    <ManifestCard manifest={m} compact />
                  </button>
                ))}
              </div>
            )}
          </div>
        )
      ) : (
        // Quick Upload placeholder
        <div className="rounded-lg border border-dashed border-gray-700 bg-gray-900/50 p-6 text-center">
          <p className="text-sm text-gray-400">
            Upload reference images inline. An auto-manifest will be created behind the scenes.
          </p>
          <p className="mt-1 text-xs text-gray-600">
            (Upload UI coming in a future phase — for now, generate without a manifest)
          </p>
        </div>
      )}
    </div>
  );
}
