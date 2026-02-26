import { useEffect, useState, useMemo, useCallback } from "react";
import { listScenes } from "../api/client.ts";
import type { SceneListItem } from "../api/types.ts";
import { SceneCard } from "./SceneCard.tsx";

interface ScenePickerModalProps {
  productionId: string;
  linkedSceneIds: Set<string>;
  onConfirm: (sceneIds: string[]) => Promise<void>;
  onClose: () => void;
}

const STATUS_FILTERS = [
  { value: "", label: "All" },
  { value: "complete", label: "Complete" },
  { value: "draft", label: "Draft" },
  { value: "failed", label: "Failed" },
  { value: "stopped", label: "Stopped" },
] as const;

export function ScenePickerModal({ productionId, linkedSceneIds, onConfirm, onClose }: ScenePickerModalProps) {
  const [scenes, setScenes] = useState<SceneListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function fetchScenes() {
      setLoading(true);
      try {
        const data = await listScenes({ per_page: 96, view: "cards" });
        if (!cancelled) setScenes(data.items);
      } catch {
        // silently fail — modal will show empty state
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchScenes();
    return () => { cancelled = true; };
  }, []);

  // Filter out scenes already in this production or linked to other productions
  const availableScenes = useMemo(() => {
    return scenes.filter((s) => {
      // Already linked to this production
      if (linkedSceneIds.has(s.scene_id)) return false;
      // Linked to another production
      if (s.production_id && s.production_id !== productionId) return false;
      return true;
    });
  }, [scenes, linkedSceneIds, productionId]);

  // Apply client-side search and status filter
  const filteredScenes = useMemo(() => {
    let result = availableScenes;
    if (statusFilter) {
      result = result.filter((s) => s.status === statusFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      result = result.filter(
        (s) =>
          (s.title?.toLowerCase().includes(q)) ||
          s.prompt.toLowerCase().includes(q),
      );
    }
    return result;
  }, [availableScenes, statusFilter, search]);

  const toggleSelect = useCallback((sceneId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(sceneId)) {
        next.delete(sceneId);
      } else {
        next.add(sceneId);
      }
      return next;
    });
  }, []);

  async function handleConfirm() {
    if (selected.size === 0) return;
    setSubmitting(true);
    try {
      await onConfirm(Array.from(selected));
    } finally {
      setSubmitting(false);
    }
  }

  // Close on Escape
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/70"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="mt-4 flex h-[calc(100vh-2rem)] w-full max-w-5xl flex-col rounded-xl border border-gray-700 bg-gray-900 shadow-2xl sm:mt-8 sm:h-[calc(100vh-4rem)]">
        {/* Header */}
        <div className="flex shrink-0 items-center gap-3 border-b border-gray-700 px-4 py-3 sm:px-6">
          <h2 className="text-lg font-semibold text-white">Add Scenes</h2>
          <div className="flex-1">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search scenes..."
              className="w-full max-w-xs rounded-md border border-gray-600 bg-gray-800 px-3 py-1.5 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-gray-400 hover:bg-gray-800 hover:text-white"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Status filter chips */}
        <div className="flex shrink-0 flex-wrap gap-1.5 border-b border-gray-800 px-4 py-2 sm:px-6">
          {STATUS_FILTERS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setStatusFilter(opt.value)}
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors ${
                statusFilter === opt.value
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Scrollable card grid */}
        <div className="flex-1 overflow-y-auto px-4 py-4 sm:px-6">
          {loading ? (
            <div className="flex items-center justify-center py-16 text-gray-400">
              Loading scenes...
            </div>
          ) : filteredScenes.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-gray-500">
              <p>{availableScenes.length === 0 ? "No unlinked scenes available." : "No scenes match your filters."}</p>
              {(search || statusFilter) && availableScenes.length > 0 && (
                <button
                  onClick={() => { setSearch(""); setStatusFilter(""); }}
                  className="mt-2 text-sm text-blue-400 hover:text-blue-300"
                >
                  Clear filters
                </button>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {filteredScenes.map((scene) => (
                <SceneCard
                  key={scene.scene_id}
                  scene={scene}
                  onClick={() => toggleSelect(scene.scene_id)}
                  selectable
                  selected={selected.has(scene.scene_id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex shrink-0 items-center justify-between border-t border-gray-700 px-4 py-3 sm:px-6">
          <span className="text-sm text-gray-400">
            {selected.size > 0
              ? `${selected.size} scene${selected.size !== 1 ? "s" : ""} selected`
              : "Select scenes to add"}
          </span>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="rounded-md bg-gray-700 px-4 py-2 text-sm text-gray-300 hover:bg-gray-600"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={selected.size === 0 || submitting}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting
                ? "Adding..."
                : `Add ${selected.size || ""} Scene${selected.size !== 1 ? "s" : ""}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
