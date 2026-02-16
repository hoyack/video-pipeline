import { useState, useCallback, useEffect } from "react";
import type { ManifestListItem } from "../api/types.ts";
import { listManifests, deleteManifest, duplicateManifest } from "../api/client.ts";
import { ManifestCard } from "./ManifestCard.tsx";

interface ManifestLibraryProps {
  onCreateNew: () => void;
  onEditManifest: (manifestId: string) => void;
  onViewManifest: (manifestId: string) => void;
}

const CATEGORIES = [
  { value: "ALL", label: "All" },
  { value: "CHARACTERS", label: "Characters" },
  { value: "ENVIRONMENT", label: "Environment" },
  { value: "FULL_PRODUCTION", label: "Full Production" },
  { value: "STYLE_KIT", label: "Style Kit" },
  { value: "BRAND_KIT", label: "Brand Kit" },
  { value: "CUSTOM", label: "Custom" },
];

const SORT_OPTIONS = [
  { value: "updated_at", label: "Updated" },
  { value: "created_at", label: "Created" },
  { value: "name", label: "Name" },
  { value: "times_used", label: "Most Used" },
  { value: "asset_count", label: "Most Assets" },
];

export function ManifestLibrary({
  onCreateNew,
  onEditManifest,
  onViewManifest,
}: ManifestLibraryProps) {
  const [manifests, setManifests] = useState<ManifestListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState("ALL");
  const [sortBy, setSortBy] = useState("updated_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const fetchManifests = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listManifests({
        category: categoryFilter !== "ALL" ? categoryFilter : undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      setManifests(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load manifests");
    } finally {
      setLoading(false);
    }
  }, [categoryFilter, sortBy, sortOrder]);

  useEffect(() => {
    fetchManifests();
  }, [fetchManifests]);

  const handleDelete = async (manifestId: string) => {
    setDeleteError(null);
    try {
      await deleteManifest(manifestId);
      setManifests((prev) => prev.filter((m) => m.manifest_id !== manifestId));
      setDeleteConfirm(null);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Failed to delete manifest");
    }
  };

  const handleDuplicate = async (manifestId: string) => {
    try {
      const newManifest = await duplicateManifest(manifestId);
      setManifests((prev) => [newManifest, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to duplicate manifest");
    }
  };

  const toggleSortOrder = () => {
    setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
  };

  if (loading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <p className="text-gray-400">Loading manifests...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3">
        <p className="text-red-400">{error}</p>
        <button
          onClick={fetchManifests}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Header section */}
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Manifest Library</h1>
        <button
          onClick={onCreateNew}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500"
        >
          + New Manifest
        </button>
      </div>

      {/* Filter/Sort bar */}
      <div className="mb-6 flex flex-wrap items-center gap-4">
        {/* Category filter pills */}
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setCategoryFilter(value)}
              className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
                categoryFilter === value
                  ? "bg-gray-700 text-white"
                  : "border border-gray-800 bg-gray-900 text-gray-400 hover:text-gray-300"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Sort dropdown */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="rounded border border-gray-800 bg-gray-900 px-2 py-1.5 text-sm text-gray-300"
        >
          {SORT_OPTIONS.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>

        {/* Sort order toggle */}
        <button
          onClick={toggleSortOrder}
          className="rounded border border-gray-800 bg-gray-900 px-2 py-1.5 text-sm text-gray-400 transition-colors hover:text-gray-300"
        >
          {sortOrder === "asc" ? "Asc ↑" : "Desc ↓"}
        </button>
      </div>

      {/* Results summary */}
      <p className="mb-4 text-sm text-gray-500">
        {manifests.length} manifest{manifests.length !== 1 ? "s" : ""}
      </p>

      {/* Card grid */}
      {manifests.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {manifests.map((manifest) => (
            <ManifestCard
              key={manifest.manifest_id}
              manifest={manifest}
              onView={onViewManifest}
              onEdit={onEditManifest}
              onDuplicate={handleDuplicate}
              onDelete={(id) => setDeleteConfirm(id)}
            />
          ))}
        </div>
      ) : (
        /* Empty state */
        <div className="flex min-h-[40vh] flex-col items-center justify-center gap-3">
          <p className="text-lg text-gray-400">No manifests yet.</p>
          <p className="text-sm text-gray-500">
            Create your first manifest to start building reusable asset collections.
          </p>
          <button
            onClick={onCreateNew}
            className="mt-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500"
          >
            + New Manifest
          </button>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="max-w-sm rounded-lg border border-gray-800 bg-gray-900 p-6">
            <h2 className="mb-2 text-lg font-semibold text-white">Delete Manifest?</h2>
            <p className="mb-4 text-sm text-gray-400">
              This action can be undone.
            </p>
            {deleteError && (
              <p className="mb-3 text-sm text-red-400">{deleteError}</p>
            )}
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setDeleteConfirm(null);
                  setDeleteError(null);
                }}
                className="flex-1 rounded bg-gray-800 px-4 py-2 text-sm text-gray-400 transition-colors hover:text-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(deleteConfirm)}
                className="flex-1 rounded bg-red-600 px-4 py-2 text-sm text-white transition-colors hover:bg-red-500"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
