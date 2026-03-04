import { useDraggable, useDroppable } from "@dnd-kit/core";
import { CSS } from "@dnd-kit/utilities";
import type { SceneListItem } from "../api/types.ts";

interface UnsequencedSectionProps {
  scenes: SceneListItem[];
  onViewScene: (id: string) => void;
  onRemoveScene?: (sceneId: string) => void;
}

const UNSEQUENCED_ID = "__unsequenced__";

/** A draggable scene row for the unsequenced zone. */
function DraggableUnsequencedRow({
  scene,
  onViewScene,
  onRemoveScene,
}: {
  scene: SceneListItem;
  onViewScene: (id: string) => void;
  onRemoveScene?: (sceneId: string) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    isDragging,
  } = useDraggable({
    id: scene.scene_id,
    data: { type: "scene-unsequenced" },
  });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
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
          title="Drag to sequence"
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
        {onRemoveScene && (
          <button
            onClick={(e) => { e.stopPropagation(); onRemoveScene(scene.scene_id); }}
            className="flex-shrink-0 text-gray-500 hover:text-red-400 p-0.5"
            title="Remove from production"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

export function UnsequencedSection({ scenes, onViewScene, onRemoveScene }: UnsequencedSectionProps) {
  const { setNodeRef, isOver } = useDroppable({ id: UNSEQUENCED_ID });

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/40 border border-dashed border-gray-700">
        <div className="w-3 h-3 rounded-full bg-gray-600 flex-shrink-0" />
        <span className="text-sm font-semibold text-gray-400 flex-1">Unsequenced</span>
        {scenes.length > 0 && (
          <span className="text-xs text-gray-500 bg-gray-700/50 rounded-full px-2 py-0.5">
            {scenes.length} {scenes.length === 1 ? "scene" : "scenes"}
          </span>
        )}
      </div>

      {/* Droppable area — always rendered so scenes can be dropped here */}
      <div
        ref={setNodeRef}
        className={`space-y-1 min-h-[48px] rounded-lg transition-colors ${
          isOver ? "bg-blue-900/20 border border-blue-700/50" : ""
        }`}
      >
        {scenes.length === 0 ? (
          <div className="py-3 text-center text-xs text-gray-600 border border-dashed border-gray-700/50 rounded-lg">
            Drop scenes here to unsequence
          </div>
        ) : (
          scenes.map((scene) => (
            <DraggableUnsequencedRow
              key={scene.scene_id}
              scene={scene}
              onViewScene={onViewScene}
              onRemoveScene={onRemoveScene}
            />
          ))
        )}
      </div>
    </div>
  );
}

export { UNSEQUENCED_ID };
