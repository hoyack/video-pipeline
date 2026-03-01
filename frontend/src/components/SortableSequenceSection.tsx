import { useDroppable } from "@dnd-kit/core";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { SceneListItem, SequenceResponse, SequenceUpdate } from "../api/types.ts";
import { SequenceHeader } from "./SequenceHeader.tsx";

interface SortableSequenceSectionProps {
  sequence: SequenceResponse;
  scenes: SceneListItem[];
  isCollapsed: boolean;
  onViewScene: (id: string) => void;
  onUpdate: (id: string, updates: SequenceUpdate) => void;
  onDelete: (id: string) => void;
  onToggleCollapse: (id: string) => void;
}

/** A scene row within a sequence that is draggable to other sequences. */
function DraggableSceneRow({
  scene,
  onViewScene,
}: {
  scene: SceneListItem;
  onViewScene: (id: string) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: scene.scene_id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : undefined,
    position: "relative" as const,
    zIndex: isDragging ? 20 : undefined,
  };

  return (
    <div ref={setNodeRef} style={style}>
      <div className="flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-800/50 px-2 py-2 hover:bg-gray-800 transition-colors">
        {/* Drag handle */}
        <button
          {...attributes}
          {...listeners}
          className="flex-shrink-0 text-gray-600 hover:text-gray-400 cursor-grab active:cursor-grabbing p-0.5"
          title="Drag to reorder"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M7 2a2 2 0 110 4 2 2 0 010-4zM13 2a2 2 0 110 4 2 2 0 010-4zM7 8a2 2 0 110 4 2 2 0 010-4zM13 8a2 2 0 110 4 2 2 0 010-4zM7 14a2 2 0 110 4 2 2 0 010-4zM13 14a2 2 0 110 4 2 2 0 010-4z" />
          </svg>
        </button>

        {scene.thumbnail_url ? (
          <img
            src={scene.thumbnail_url}
            alt=""
            className="w-12 h-8 object-cover rounded flex-shrink-0"
          />
        ) : (
          <div className="w-12 h-8 bg-gray-700 rounded flex-shrink-0" />
        )}

        <div
          className="min-w-0 flex-1 cursor-pointer"
          onClick={() => onViewScene(scene.scene_id)}
        >
          <p className="text-sm text-white truncate">
            {scene.title || scene.prompt.slice(0, 80)}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">{scene.status}</p>
        </div>
      </div>
    </div>
  );
}

export function SortableSequenceSection({
  sequence,
  scenes,
  isCollapsed,
  onViewScene,
  onUpdate,
  onDelete,
  onToggleCollapse,
}: SortableSequenceSectionProps) {
  // Make the section itself a droppable target
  const { setNodeRef, isOver } = useDroppable({ id: sequence.id });

  // Sort scenes by scene_order
  const sortedScenes = [...scenes].sort((a, b) => {
    const aOrder = (a as SceneListItem & { scene_order?: number }).scene_order ?? 0;
    const bOrder = (b as SceneListItem & { scene_order?: number }).scene_order ?? 0;
    return aOrder - bOrder;
  });

  return (
    <div className="space-y-1">
      <SequenceHeader
        sequence={sequence}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => onToggleCollapse(sequence.id)}
        onUpdate={(updates) => onUpdate(sequence.id, updates)}
        onDelete={onDelete}
      />

      {!isCollapsed && (
        <div
          ref={setNodeRef}
          className={`ml-4 space-y-1 min-h-[40px] rounded-lg transition-colors ${
            isOver ? "bg-blue-900/20 border border-blue-700/50" : ""
          }`}
        >
          {sortedScenes.length === 0 ? (
            <div className="py-3 text-center text-xs text-gray-600">
              Drop scenes here
            </div>
          ) : (
            sortedScenes.map((scene) => (
              <DraggableSceneRow
                key={scene.scene_id}
                scene={scene}
                onViewScene={onViewScene}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}
