import { useCallback, useEffect, useState } from "react";
import {
  DndContext,
  PointerSensor,
  closestCenter,
  pointerWithin,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import type { CollisionDetection, DragEndEvent } from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import {
  listSequences,
  createSequence,
  updateSequence,
  deleteSequence,
  assignSceneToSequence,
  reorderSequences,
  reorderScenesInSequence,
} from "../api/client.ts";
import type { SceneListItem, SequenceResponse, SequenceUpdate } from "../api/types.ts";
import { SortableSequenceSection } from "./SortableSequenceSection.tsx";
import { UnsequencedSection, UNSEQUENCED_ID } from "./UnsequencedSection.tsx";

/**
 * Custom collision detection: if the pointer is physically inside the
 * unsequenced drop zone, always prioritize it. Otherwise fall back to
 * closestCenter for sorting and cross-sequence assignment.
 */
const sequenceCollision: CollisionDetection = (args) => {
  const withinCollisions = pointerWithin(args);
  const unsequencedHit = withinCollisions.find((c) => c.id === UNSEQUENCED_ID);
  if (unsequencedHit) return [unsequencedHit];
  return closestCenter(args);
};

interface SequencedSceneListProps {
  productionId: string;
  scenes: SceneListItem[];
  onViewScene: (id: string) => void;
  onRefresh?: () => void;
  onRemoveScene?: (sceneId: string) => void;
}

export function SequencedSceneList({
  productionId,
  scenes,
  onViewScene,
  onRefresh,
  onRemoveScene,
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

  const sequenceIds = new Set(sequences.map((s) => s.id));

  async function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over) return;

    const activeType = active.data.current?.type as string | undefined;

    // --- Sequence reorder ---
    if (activeType === "sequence") {
      const activeId = active.id as string;
      const overId = over.id as string;
      if (activeId === overId) return;

      const oldIndex = sequences.findIndex((s) => s.id === activeId);
      const newIndex = sequences.findIndex((s) => s.id === overId);
      if (oldIndex === -1 || newIndex === -1) return;

      const reordered = arrayMove(sequences, oldIndex, newIndex);
      const previousSequences = sequences;
      setSequences(reordered);

      try {
        await reorderSequences(productionId, {
          sequence_ids: reordered.map((s) => s.id),
        });
      } catch {
        setSequences(previousSequences);
        setError("Failed to reorder sequences");
      }
      return;
    }

    // --- Within-sequence scene reorder ---
    if (
      activeType === "scene-within-sequence" &&
      over.data.current?.type === "scene-within-sequence" &&
      active.data.current?.sequenceId === over.data.current?.sequenceId
    ) {
      const sequenceId = active.data.current?.sequenceId as string;
      const seqScenes = localScenes
        .filter((s) => s.sequence_id === sequenceId)
        .sort((a, b) => (a.scene_order ?? 0) - (b.scene_order ?? 0));

      const activeId = active.id as string;
      const overId = over.id as string;
      if (activeId === overId) return;

      const oldIndex = seqScenes.findIndex((s) => s.scene_id === activeId);
      const newIndex = seqScenes.findIndex((s) => s.scene_id === overId);
      if (oldIndex === -1 || newIndex === -1) return;

      const reorderedScenes = arrayMove(seqScenes, oldIndex, newIndex);
      const reorderedSceneIds = reorderedScenes.map((s) => s.scene_id);

      // Optimistic update: reassign scene_order values
      const previousScenes = localScenes;
      setLocalScenes((prev) =>
        prev.map((s) => {
          const idx = reorderedSceneIds.indexOf(s.scene_id);
          if (idx !== -1) {
            return { ...s, scene_order: idx };
          }
          return s;
        }),
      );

      try {
        await reorderScenesInSequence(sequenceId, {
          scene_ids: reorderedSceneIds,
        });
      } catch {
        setLocalScenes(previousScenes);
        setError("Failed to reorder scenes");
      }
      return;
    }

    // --- Cross-sequence scene drag (existing behavior) ---
    const sceneId = active.id as string;
    const targetId = over.id as string;

    // Determine target sequence (null = unsequenced)
    let resolvedTarget: string | null;
    if (targetId === UNSEQUENCED_ID) {
      // Explicitly unsequencing — intentional null
      resolvedTarget = null;
    } else if (sequenceIds.has(targetId)) {
      // Dropped on a sequence header (sortable)
      resolvedTarget = targetId;
    } else if (over.data.current?.type === "sequence-drop") {
      // Dropped on a sequence body drop zone (e.g. "Drop scenes here")
      resolvedTarget = over.data.current.sequenceId as string;
    } else {
      // Dropped on a scene row — inherit its sequence (or null if unsequenced)
      resolvedTarget = (over.data.current?.sequenceId as string) ?? null;
    }

    // Find current scene
    const scene = localScenes.find((s) => s.scene_id === sceneId);
    if (!scene) return;

    // Skip no-op: already in target
    if (scene.sequence_id === resolvedTarget) return;

    // Optimistic update
    const previousScenes = localScenes;
    setLocalScenes((prev) =>
      prev.map((s) =>
        s.scene_id === sceneId ? { ...s, sequence_id: resolvedTarget } : s,
      ),
    );

    // Update sequence scene counts optimistically
    if (scene.sequence_id || resolvedTarget) {
      setSequences((prev) =>
        prev.map((seq) =>
          seq.id === scene.sequence_id
            ? { ...seq, scene_count: Math.max(0, seq.scene_count - 1) }
            : seq.id === resolvedTarget
              ? { ...seq, scene_count: seq.scene_count + 1 }
              : seq,
        ),
      );
    }

    try {
      await assignSceneToSequence(sceneId, { sequence_id: resolvedTarget });
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

      <DndContext sensors={sensors} collisionDetection={sequenceCollision} onDragEnd={handleDragEnd}>
        {/* Sequences in order — wrapped in SortableContext for sequence reorder */}
        <SortableContext
          items={sequences.map((s) => s.id)}
          strategy={verticalListSortingStrategy}
        >
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
              onRemoveScene={onRemoveScene}
            />
          ))}
        </SortableContext>

        {/* Unsequenced scenes */}
        <UnsequencedSection
          scenes={unsequencedScenes}
          onViewScene={onViewScene}
          onRemoveScene={onRemoveScene}
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
