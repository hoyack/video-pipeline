import { useDroppable } from "@dnd-kit/core";
import { SortableContext, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
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
  sequenceId,
  onViewScene,
}: {
  scene: SceneListItem;
  sequenceId: string;
  onViewScene: (id: string) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({
    id: scene.scene_id,
    data: { type: "scene-within-sequence", sequenceId },
  });

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
  // Make the sequence header sortable for sequence reorder
  const {
    attributes: seqAttributes,
    listeners: seqListeners,
    setNodeRef: setSortableRef,
    transform: seqTransform,
    transition: seqTransition,
    isDragging: seqIsDragging,
  } = useSortable({
    id: sequence.id,
    data: { type: "sequence" },
  });

  const sortableStyle: React.CSSProperties = {
    transform: CSS.Transform.toString(seqTransform),
    transition: seqTransition,
    opacity: seqIsDragging ? 0.5 : undefined,
    position: "relative" as const,
    zIndex: seqIsDragging ? 30 : undefined,
  };

  // Make the section itself a droppable target for cross-sequence scene drag
  const { setNodeRef: setDropRef, isOver } = useDroppable({ id: sequence.id });

  // Sort scenes by scene_order
  const sortedScenes = [...scenes].sort((a, b) => {
    const aOrder = a.scene_order ?? 0;
    const bOrder = b.scene_order ?? 0;
    return aOrder - bOrder;
  });

  return (
    <div ref={setSortableRef} style={sortableStyle} className="space-y-1">
      <div className="flex items-center gap-1">
        {/* Sequence drag handle */}
        <button
          {...seqAttributes}
          {...seqListeners}
          className="flex-shrink-0 text-gray-600 hover:text-gray-400 cursor-grab active:cursor-grabbing p-0.5"
          title="Drag to reorder sequence"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M7 2a2 2 0 110 4 2 2 0 010-4zM13 2a2 2 0 110 4 2 2 0 010-4zM7 8a2 2 0 110 4 2 2 0 010-4zM13 8a2 2 0 110 4 2 2 0 010-4zM7 14a2 2 0 110 4 2 2 0 010-4zM13 14a2 2 0 110 4 2 2 0 010-4z" />
          </svg>
        </button>
        <div className="flex-1 min-w-0">
          <SequenceHeader
            sequence={sequence}
            isCollapsed={isCollapsed}
            onToggleCollapse={() => onToggleCollapse(sequence.id)}
            onUpdate={(updates) => onUpdate(sequence.id, updates)}
            onDelete={onDelete}
          />
        </div>
      </div>

      {!isCollapsed && (
        <div
          ref={setDropRef}
          className={`ml-4 space-y-1 min-h-[40px] rounded-lg transition-colors ${
            isOver ? "bg-blue-900/20 border border-blue-700/50" : ""
          }`}
        >
          <SortableContext
            items={sortedScenes.map((s) => s.scene_id)}
            strategy={verticalListSortingStrategy}
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
                  sequenceId={sequence.id}
                  onViewScene={onViewScene}
                />
              ))
            )}
          </SortableContext>
        </div>
      )}
    </div>
  );
}
