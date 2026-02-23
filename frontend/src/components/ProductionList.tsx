import { useEffect, useState } from "react";
import {
  listProductions,
  createProduction,
  deleteProduction,
  type ProductionResponse,
} from "../api/client.ts";

interface ProductionListProps {
  onSelectProduction: (id: string) => void;
}

export function ProductionList({ onSelectProduction }: ProductionListProps) {
  const [productions, setProductions] = useState<ProductionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  async function load() {
    try {
      setLoading(true);
      const data = await listProductions();
      setProductions(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load productions");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate() {
    if (!newName.trim()) return;
    try {
      const prod = await createProduction({
        name: newName.trim(),
        description: newDesc.trim() || undefined,
      });
      setProductions((prev) => [prod, ...prev]);
      setShowCreate(false);
      setNewName("");
      setNewDesc("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create production");
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this production? Scenes will be unlinked but not deleted.")) return;
    try {
      await deleteProduction(id);
      setProductions((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete production");
    }
  }

  if (loading) {
    return <div className="text-center py-12 text-gray-400">Loading productions...</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Productions</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500"
        >
          New Production
        </button>
      </div>

      {error && (
        <div className="rounded-md bg-red-900/50 p-3 text-sm text-red-300">{error}</div>
      )}

      {showCreate && (
        <div className="rounded-lg border border-gray-700 bg-gray-800 p-4 space-y-3">
          <input
            type="text"
            placeholder="Production name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white placeholder-gray-500"
            autoFocus
          />
          <textarea
            placeholder="Description (optional)"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            className="w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white placeholder-gray-500"
            rows={2}
          />
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500"
            >
              Create
            </button>
            <button
              onClick={() => {
                setShowCreate(false);
                setNewName("");
                setNewDesc("");
              }}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-sm font-medium text-gray-300 hover:bg-gray-600"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {productions.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          No productions yet. Create one to group your scenes.
        </div>
      ) : (
        <div className="space-y-2">
          {productions.map((prod) => (
            <div
              key={prod.id}
              className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800/50 p-4 hover:bg-gray-800 transition-colors cursor-pointer"
              onClick={() => onSelectProduction(prod.id)}
            >
              <div>
                <h3 className="font-medium text-white">{prod.name}</h3>
                {prod.description && (
                  <p className="text-sm text-gray-400 mt-0.5">{prod.description}</p>
                )}
                <p className="text-xs text-gray-500 mt-1">
                  {prod.scene_count} scene{prod.scene_count !== 1 ? "s" : ""} · Created{" "}
                  {new Date(prod.created_at).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(prod.id);
                }}
                className="text-gray-500 hover:text-red-400 text-sm"
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
