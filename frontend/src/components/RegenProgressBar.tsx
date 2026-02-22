interface RegenProgressBarProps {
  phase: string | null;
  totalScenes: number;
  completedScenes: number;
  currentSceneIndex: number | null;
  currentStatus: string | null;
  wsConnected: boolean;
}

const PHASE_LABELS: Record<string, string> = {
  storyboard: "Storyboard",
  keyframes: "Keyframes",
  clips: "Video Clips",
  stitch: "Stitching",
};

const STATUS_LABELS: Record<string, string> = {
  generating_text: "Generating text",
  generating_start_kf: "Generating start keyframe",
  generating_end_kf: "Generating end keyframe",
  generating_clip: "Generating clip",
};

export function RegenProgressBar({
  phase,
  totalScenes,
  completedScenes,
  currentSceneIndex,
  currentStatus,
  wsConnected,
}: RegenProgressBarProps) {
  const phaseLabel = phase ? (PHASE_LABELS[phase] || phase) : "Processing";
  const statusLabel = currentStatus ? (STATUS_LABELS[currentStatus] || currentStatus) : null;
  const pct = totalScenes > 0 ? Math.round((completedScenes / totalScenes) * 100) : 0;

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 px-4 py-3 space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-gray-200">
          {phaseLabel}
          {currentSceneIndex !== null && (
            <span className="text-gray-400 ml-1">
              — Scene {currentSceneIndex + 1}
              {statusLabel && <span className="ml-1 text-gray-500">({statusLabel})</span>}
            </span>
          )}
        </span>
        <span className="text-gray-400 tabular-nums">
          {completedScenes}/{totalScenes} ({pct}%)
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-700 overflow-hidden">
        <div
          className="h-full rounded-full bg-indigo-500 transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
      {!wsConnected && (
        <p className="text-xs text-amber-400">
          Live updates unavailable — falling back to polling
        </p>
      )}
    </div>
  );
}
