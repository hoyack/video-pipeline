import clsx from "clsx";
import type { SceneDetail } from "../api/types.ts";

function Dot({ filled, color }: { filled: boolean; color: string }) {
  return (
    <span
      className={clsx(
        "inline-block h-2 w-2 rounded-full",
        filled ? color : "bg-gray-700",
      )}
      title={filled ? "Ready" : "Pending"}
    />
  );
}

export function SceneCard({ scene }: { scene: SceneDetail }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400">
          Scene {scene.scene_index + 1}
        </span>
        <span
          className={clsx(
            "text-[10px] font-medium uppercase",
            scene.has_clip ? "text-green-400" : "text-gray-500",
          )}
        >
          {scene.clip_status ?? scene.status}
        </span>
      </div>
      <p className="mb-2 text-sm leading-snug text-gray-300 line-clamp-2">
        {scene.description}
      </p>
      <div className="flex items-center gap-1.5">
        <Dot filled={scene.has_start_keyframe} color="bg-blue-400" />
        <Dot filled={scene.has_end_keyframe} color="bg-indigo-400" />
        <Dot filled={scene.has_clip} color="bg-green-400" />
        <span className="ml-1 text-[10px] text-gray-600">
          KF start / KF end / clip
        </span>
      </div>
    </div>
  );
}
