import { useCallback, useEffect, useState } from "react";
import {
  DndContext,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import type { DragEndEvent } from "@dnd-kit/core";
import {
  listSequences,
  createSequence,
  updateSequence,
  deleteSequence,
  assignSceneToSequence,
} from "../api/client.ts";
import type { SceneListItem, SequenceResponse, SequenceUpdate } from "../api/types.ts";
import { SortableSequenceSection } from "./SortableSequenceSection.tsx";
import { UnsequencedSection, UNSEQUENCED_ID } from "./UnsequencedSection.tsx";

interface SequencedSceneListProps {
  productionId: string;
  scenes: SceneListItem[];
  onViewScene: (id: string) => void;
  onRefresh?: () => void;
}

export function SequencedSceneList({
  productionId,
  scenes,
  onViewScene,
  onRefresh,
}: SequencedSceneListProps) {
  const [sequences, setSequences] = useState<SequenceResponse[]>([]);
  const [localScenes, setLocalScenes] = useState<SceneListItem[]>(scenes);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Keep local scenes in sync when parent refreshes
  useEffect(() => {
    setLocalScenes(scenes);
  }, [scenes]);

  const loadSequences = useCallback(async () => {
    try {
      const seqs = await listSequences(productionId);
      setSequences(seqs.sort((a, b) => a.order - b.order));
    } catch {
      setError("Failed to load sequences");
    } finally {
      setLoading(false);
    }
  }, [productionId]);

  useEffect(() => {
    loadSequences();
  }, [loadSequences]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
  );

  function scenesForSequence(seqId: string): SceneListItem[] {
    return localScenes.filter((s) => s.sequence_id === seqId);
  }

  const unsequencedScenes = localScenes.filter((s) => !s.sequence_id);

  async function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over) return;

    const sceneId = active.id as string;
    const targetId = over.id as string;

    // Determine target sequence (null = unsequenced)
    const targetSequenceId = targetId === UNSEQUENCED_ID ? null : targetId;

    // Find current scene
    const scene = localScenes.find((s) => s.scene_id === sceneId);
    if (!scene) return;

    // Skip no-op: already in target
    if (scene.sequence_id === targetSequenceId) return;

    // Optimistic update
    const previousScenes = localScenes;
    setLocalScenes((prev) =>
      prev.map((s) =>
        s.scene_id === sceneId ? { ...s, sequence_id: targetSequenceId } : s,
      ),
    );

    // Update sequence scene counts optimistically
    if (scene.sequence_id) {
      setSequences((prev) =>
        prev.map((seq) =>
          seq.id === scene.sequence_id
            ? { ...seq, scene_count: Math.max(0, seq.scene_count - 1) }
            : seq.id === targetSequenceId
              ? { ...seq, scene_count: seq.scene_count + 1 }
              : seq,
        ),
      );
    }

    try {
      await assignSceneToSequence(sceneId, { sequence_id: targetSequenceId });
      // Refresh parent if callback provided
      onRefresh?.();
    } catch {
      // Revert on failure
      setLocalScenes(previousScenes);
      setError("Failed to move scene");
    }
  }

  async function handleAddSequence() {
    const title = newTitle.trim();
    if (!title) return;

    try {
      const seq = await createSequence(productionId, { title });
      setSequences((prev) => [...prev, seq]);
      setNewTitle("");
      setAdding(false);
    } catch {
      setError("Failed to create sequence");
    }
  }

  async function handleUpdateSequence(id: string, updates: SequenceUpdate) {
    try {
      const updated = await updateSequence(id, updates);
      setSequences((prev) => prev.map((s) => (s.id === id ? updated : s)));
    } catch {
      setError("Failed to update sequence");
    }
  }

  async function handleDeleteSequence(id: string) {
    try {
      await deleteSequence(id);
      setSequences((prev) => prev.filter((s) => s.id !== id));
      // Move scenes from deleted sequence to unsequenced in local state
      setLocalScenes((prev) =>
        prev.map((s) => (s.sequence_id === id ? { ...s, sequence_id: null } : s)),
      );
      onRefresh?.();
    } catch {
      setError("Failed to delete sequence");
    }
  }

  function handleToggleCollapse(id: string) {
    setCollapsed((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  if (loading) {
    return <div className="text-center py-6 text-gray-500 text-sm">Loading sequences...</div>;
  }

  return (
    <div className="space-y-3">
      {error && (
        <div className="rounded-md bg-red-900/50 p-2 text-xs text-red-300 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200">
            Dismiss
          </button>
        </div>
      )}

      <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
        {/* Sequences in order */}
        {sequences.map((seq) => (
          <SortableSequenceSection
            key={seq.id}
            sequence={seq}
            scenes={scenesForSequence(seq.id)}
            isCollapsed={!!collapsed[seq.id]}
            onViewScene={onViewScene}
            onUpdate={handleUpdateSequence}
            onDelete={handleDeleteSequence}
            onToggleCollapse={handleToggleCollapse}
          />
        ))}

        {/* Unsequenced scenes */}
        <UnsequencedSection
          scenes={unsequencedScenes}
          onViewScene={onViewScene}
        />
      </DndContext>

      {/* Add sequence UI */}
      {adding ? (
        <div className="flex gap-2 items-center">
          <input
            type="text"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleAddSequence();
              if (e.key === "Escape") {
                setAdding(false);
                setNewTitle("");
              }
            }}
            placeholder="Sequence title..."
            autoFocus
            className="flex-1 rounded-md border border-gray-600 bg-gray-900 px-3 py-1.5 text-sm text-white placeholder:text-gray-600 outline-none focus:border-blue-500"
          />
          <button
            onClick={handleAddSequence}
            className="rounded-md bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-500"
          >
            Add
          </button>
          <button
            onClick={() => {
              setAdding(false);
              setNewTitle("");
            }}
            className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600"
          >
            Cancel
          </button>
        </div>
      ) : (
        <button
          onClick={() => setAdding(true)}
          className="w-full rounded-lg border border-dashed border-gray-700 py-2 text-sm text-gray-500 hover:text-gray-300 hover:border-gray-500 transition-colors"
        >
          + Add Sequence
        </button>
      )}
    </div>
  );
}
