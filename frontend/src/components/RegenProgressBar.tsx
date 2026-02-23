interface RegenProgressBarProps {
  scope: string | null;
  phase: string | null;
  totalScenes: number;
  completedScenes: number;
  currentSceneIndex: number | null;
  currentStatus: string | null;
  wsConnected: boolean;
  completedPhases: string[];
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

const SCOPE_PHASES: Record<string, string[]> = {
  all_phases: ["storyboard", "keyframes", "clips", "stitch"],
  storyboard: ["storyboard"],
  keyframes: ["keyframes"],
  clips: ["clips"],
  stitch_only: ["stitch"],
  stitch: ["stitch"],
};

const PHASE_WEIGHTS: Record<string, number> = {
  storyboard: 1,
  keyframes: 3,
  clips: 5,
  stitch: 1,
};

export function RegenProgressBar({
  scope,
  phase,
  totalScenes,
  completedScenes,
  currentSceneIndex,
  currentStatus,
  wsConnected,
  completedPhases,
}: RegenProgressBarProps) {
  const phases = scope ? (SCOPE_PHASES[scope] ?? [phase ?? ""].filter(Boolean)) : [phase ?? ""].filter(Boolean);
  const isMultiPhase = phases.length > 1;

  const phaseLabel = phase ? (PHASE_LABELS[phase] || phase) : "Processing";
  const statusLabel = currentStatus ? (STATUS_LABELS[currentStatus] || currentStatus) : null;

  // Compute overall progress across all phases
  const totalWeight = phases.reduce((sum, p) => sum + (PHASE_WEIGHTS[p] ?? 1), 0);
  const completedWeight = completedPhases.reduce(
    (sum, p) => sum + (phases.includes(p) ? (PHASE_WEIGHTS[p] ?? 1) : 0),
    0,
  );
  const currentPhaseWeight = phase ? (PHASE_WEIGHTS[phase] ?? 1) : 0;
  const currentPhasePct = totalScenes > 0 ? completedScenes / totalScenes : 0;
  const overallPct = totalWeight > 0
    ? Math.round(((completedWeight + currentPhaseWeight * currentPhasePct) / totalWeight) * 100)
    : 0;

  const phaseIndex = phase ? phases.indexOf(phase) : -1;

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 px-4 py-3 space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-gray-200">
          {isMultiPhase && phaseIndex >= 0 && (
            <span className="text-gray-400 mr-1">
              Phase {phaseIndex + 1}/{phases.length}:
            </span>
          )}
          {phaseLabel}
          {currentSceneIndex !== null && (
            <span className="text-gray-400 ml-1">
              — Scene {currentSceneIndex + 1}
              {statusLabel && <span className="ml-1 text-gray-500">({statusLabel})</span>}
            </span>
          )}
        </span>
        <span className="text-gray-400 tabular-nums">
          {overallPct}%
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-700 overflow-hidden">
        <div
          className="h-full rounded-full bg-indigo-500 transition-all duration-500 ease-out"
          style={{ width: `${overallPct}%` }}
        />
      </div>
      {/* Phase pills for multi-phase operations */}
      {isMultiPhase && (
        <div className="flex items-center gap-1.5 pt-0.5">
          {phases.map((p) => {
            const isDone = completedPhases.includes(p);
            const isActive = p === phase;
            return (
              <span
                key={p}
                className={
                  isDone
                    ? "rounded-full px-2 py-0.5 text-[10px] font-medium bg-green-900/60 text-green-300 border border-green-700"
                    : isActive
                      ? "rounded-full px-2 py-0.5 text-[10px] font-medium bg-indigo-900/60 text-indigo-300 border border-indigo-600 animate-pulse"
                      : "rounded-full px-2 py-0.5 text-[10px] font-medium bg-gray-800 text-gray-500 border border-gray-700"
                }
              >
                {PHASE_LABELS[p] ?? p}
              </span>
            );
          })}
        </div>
      )}
      {!wsConnected && (
        <p className="text-xs text-amber-400">
          Live updates unavailable — falling back to polling
        </p>
      )}
    </div>
  );
}
