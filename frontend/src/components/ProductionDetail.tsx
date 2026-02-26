import { useEffect, useState, useMemo } from "react";
import {
  getProduction,
  updateProduction,
  listScenes,
  batchAddScenesToProduction,
  removeSceneFromProduction,
  type ProductionResponse,
} from "../api/client.ts";
import type { SceneListItem } from "../api/types.ts";
import { ScenePickerModal } from "./ScenePickerModal.tsx";

interface ProductionDetailProps {
  productionId: string;
  onViewScene: (id: string) => void;
}

export function ProductionDetail({ productionId, onViewScene }: ProductionDetailProps) {
  const [production, setProduction] = useState<ProductionResponse | null>(null);
  const [scenes, setScenes] = useState<SceneListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [showPicker, setShowPicker] = useState(false);

  async function load() {
    try {
      setLoading(true);
      const [prod, scenesData] = await Promise.all([
        getProduction(productionId),
        listScenes({ per_page: 96, view: "cards" }),
      ]);
      setProduction(prod);
      setEditName(prod.name);
      setEditDesc(prod.description || "");
      setScenes(scenesData.items.filter((s) => s.production_id === productionId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load production");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [productionId]);

  async function handleSave() {
    if (!editName.trim()) return;
    try {
      const updated = await updateProduction(productionId, {
        name: editName.trim(),
        description: editDesc.trim() || undefined,
      });
      setProduction(updated);
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update production");
    }
  }

  async function handleBatchAdd(sceneIds: string[]) {
    try {
      await batchAddScenesToProduction(productionId, sceneIds);
      setShowPicker(false);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add scenes");
    }
  }

  async function handleRemoveScene(sceneId: string) {
    try {
      await removeSceneFromProduction(productionId, sceneId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove scene");
    }
  }

  const linkedSceneIds = useMemo(
    () => new Set(scenes.map((s) => s.scene_id)),
    [scenes],
  );

  if (loading) {
    return <div className="text-center py-12 text-gray-400">Loading production...</div>;
  }

  if (!production) {
    return <div className="text-center py-12 text-red-400">Production not found</div>;
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-md bg-red-900/50 p-3 text-sm text-red-300">{error}</div>
      )}

      <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
        {editing ? (
          <div className="space-y-3">
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              className="w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white"
              autoFocus
            />
            <textarea
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              className="w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white"
              rows={2}
              placeholder="Description"
            />
            <div className="flex gap-2">
              <button onClick={handleSave} className="rounded-md bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-500">
                Save
              </button>
              <button onClick={() => setEditing(false)} className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600">
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-xl font-bold text-white">{production.name}</h1>
              {production.description && (
                <p className="text-sm text-gray-400 mt-1">{production.description}</p>
              )}
              <p className="text-xs text-gray-500 mt-2">
                {production.scene_count} scene{production.scene_count !== 1 ? "s" : ""} · Created{" "}
                {new Date(production.created_at).toLocaleDateString()}
              </p>
            </div>
            <button
              onClick={() => setEditing(true)}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600"
            >
              Edit
            </button>
          </div>
        )}
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Scenes</h2>
          <button
            onClick={() => setShowPicker(true)}
            className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500"
          >
            Add Scenes
          </button>
        </div>

        {scenes.length === 0 ? (
          <p className="text-sm text-gray-500 py-4">No scenes in this production yet.</p>
        ) : (
          <div className="space-y-2">
            {scenes.map((scene) => (
              <div
                key={scene.scene_id}
                className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800/50 p-3 hover:bg-gray-800 transition-colors cursor-pointer"
                onClick={() => onViewScene(scene.scene_id)}
              >
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-white truncate">
                    {scene.title || scene.prompt.slice(0, 80)}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {scene.status} · {new Date(scene.created_at).toLocaleDateString()}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveScene(scene.scene_id);
                  }}
                  className="ml-3 text-xs text-gray-500 hover:text-red-400"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {showPicker && (
        <ScenePickerModal
          productionId={productionId}
          linkedSceneIds={linkedSceneIds}
          onConfirm={handleBatchAdd}
          onClose={() => setShowPicker(false)}
        />
      )}
    </div>
  );
}
