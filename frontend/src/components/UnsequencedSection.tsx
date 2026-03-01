import { useDroppable } from "@dnd-kit/core";
import type { SceneListItem } from "../api/types.ts";

interface UnsequencedSectionProps {
  scenes: SceneListItem[];
  onViewScene: (id: string) => void;
}

const UNSEQUENCED_ID = "__unsequenced__";

export function UnsequencedSection({ scenes, onViewScene }: UnsequencedSectionProps) {
  const { setNodeRef, isOver } = useDroppable({ id: UNSEQUENCED_ID });

  if (scenes.length === 0) return null;

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/40 border border-dashed border-gray-700">
        <div className="w-3 h-3 rounded-full bg-gray-600 flex-shrink-0" />
        <span className="text-sm font-semibold text-gray-400 flex-1">Unsequenced</span>
        <span className="text-xs text-gray-500 bg-gray-700/50 rounded-full px-2 py-0.5">
          {scenes.length} {scenes.length === 1 ? "scene" : "scenes"}
        </span>
      </div>

      {/* Droppable area */}
      <div
        ref={setNodeRef}
        className={`space-y-1 min-h-[48px] rounded-lg transition-colors ${
          isOver ? "bg-blue-900/20 border border-blue-700/50" : ""
        }`}
      >
        {scenes.map((scene) => (
          <div
            key={scene.scene_id}
            onClick={() => onViewScene(scene.scene_id)}
            className="flex items-center gap-3 rounded-lg border border-gray-700 bg-gray-800/50 px-3 py-2.5 hover:bg-gray-800 transition-colors cursor-pointer"
          >
            {scene.thumbnail_url ? (
              <img
                src={scene.thumbnail_url}
                alt=""
                className="w-12 h-8 object-cover rounded flex-shrink-0"
              />
            ) : (
              <div className="w-12 h-8 bg-gray-700 rounded flex-shrink-0" />
            )}
            <div className="min-w-0 flex-1">
              <p className="text-sm text-white truncate">
                {scene.title || scene.prompt.slice(0, 80)}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">{scene.status}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export { UNSEQUENCED_ID };
