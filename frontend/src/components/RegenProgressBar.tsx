import { useState } from "react";

export interface RegenTaskLogEntry {
  id: string;
  ts: number;
  phase: string | null;
  shotIndex: number | null;
  message: string;
  tone: "info" | "success" | "error";
  detail?: string | null;
  source?: string | null;
  kind?: string | null;
}

interface RegenProgressBarProps {
  scope: string | null;
  phase: string | null;
  totalShots: number;
  completedShots: number;
  activeShots: number;
  currentShotIndex: number | null;
  currentStatus: string | null;
  statusMessage: string | null;
  wsConnected: boolean;
  completedPhases: string[];
  messages: RegenTaskLogEntry[];
  isActive: boolean;
}

const PHASE_LABELS: Record<string, string> = {
  storyboard: "Storyboard",
  keyframes: "Keyframes",
  clips: "Video Clips",
  stitch: "Stitching",
};

const STATUS_LABELS: Record<string, string> = {
  generating_text: "Generating text",
  generating_manifest: "Generating manifest",
  generating_prompts: "Writing prompts",
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

function normalizeTimestamp(ts: number): number {
  return ts > 10_000_000_000 ? ts : ts * 1000;
}

function formatLogTime(ts: number): string {
  return new Date(normalizeTimestamp(ts)).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatLogContext(entry: RegenTaskLogEntry): string {
  const parts: string[] = [];
  if (entry.source) {
    parts.push(entry.source);
  }
  if (entry.phase) {
    parts.push(PHASE_LABELS[entry.phase] ?? entry.phase);
  }
  if (entry.shotIndex !== null) {
    parts.push(`Shot ${entry.shotIndex + 1}`);
  }
  return parts.join(" • ") || "Update";
}

export function RegenProgressBar({
  scope,
  phase,
  totalShots,
  completedShots,
  activeShots,
  currentShotIndex,
  currentStatus,
  statusMessage,
  wsConnected,
  completedPhases,
  messages,
  isActive,
}: RegenProgressBarProps) {
  const [traceExpanded, setTraceExpanded] = useState(true);
  const visibleMessages = messages.slice(-5).reverse();
  const phases = scope ? (SCOPE_PHASES[scope] ?? [phase ?? ""].filter(Boolean)) : [phase ?? ""].filter(Boolean);
  const isMultiPhase = phases.length > 1;

  const phaseLabel = phase
    ? (PHASE_LABELS[phase] || phase)
    : messages.length > 0
      ? "Recent Task Updates"
      : "Processing";
  const statusLabel = currentStatus ? (STATUS_LABELS[currentStatus] || currentStatus) : null;
  const progressLabel = statusMessage || (
    totalShots > 0
      ? `${completedShots}/${totalShots} shots complete${activeShots > 0 ? ` • ${activeShots} active` : ""}`
      : null
  );

  // Compute overall progress across all phases
  const totalWeight = phases.reduce((sum, p) => sum + (PHASE_WEIGHTS[p] ?? 1), 0);
  const completedWeight = completedPhases.reduce(
    (sum, p) => sum + (phases.includes(p) ? (PHASE_WEIGHTS[p] ?? 1) : 0),
    0,
  );
  const currentPhaseWeight = phase ? (PHASE_WEIGHTS[phase] ?? 1) : 0;
  const currentPhasePct = totalShots > 0 ? completedShots / totalShots : 0;
  const overallPct = totalWeight > 0
    ? Math.round(((completedWeight + currentPhaseWeight * currentPhasePct) / totalWeight) * 100)
    : 0;

  const phaseIndex = phase ? phases.indexOf(phase) : -1;
  const showOverallPct = totalWeight > 0;

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
          {currentShotIndex !== null && (
            <span className="text-gray-400 ml-1">
              — Shot {currentShotIndex + 1}
              {statusLabel && <span className="ml-1 text-gray-500">({statusLabel})</span>}
            </span>
          )}
        </span>
        {showOverallPct && (
          <span className="text-gray-400 tabular-nums">
            {overallPct}%
          </span>
        )}
      </div>
      {progressLabel && (
        <p className="text-xs text-gray-400">
          {progressLabel}
        </p>
      )}
      {showOverallPct && (
        <div className="h-2 w-full rounded-full bg-gray-700 overflow-hidden">
          <div
            className="h-full rounded-full bg-indigo-500 transition-all duration-500 ease-out"
            style={{ width: `${overallPct}%` }}
          />
        </div>
      )}
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
      {(visibleMessages.length > 0 || isActive) && (
        <div className="rounded-md border border-gray-700/80 bg-gray-950/80">
          <button
            type="button"
            onClick={() => setTraceExpanded((prev) => !prev)}
            className="flex w-full items-center justify-between border-b border-gray-700/80 px-3 py-2 text-left"
          >
            <span className="font-mono text-[11px] font-medium uppercase tracking-wide text-gray-300">
              Verbose Task Trace
            </span>
            <span className="font-mono text-[10px] uppercase tracking-wide text-gray-500">
              Last {visibleMessages.length || 1} event{visibleMessages.length === 1 ? "" : "s"} · {traceExpanded ? "Collapse" : "Expand"}
            </span>
          </button>
          {traceExpanded && (
            <div className="space-y-2 px-3 py-3 font-mono">
              {visibleMessages.length === 0 && (
                <div className="rounded-md border border-dashed border-gray-800 bg-black/50 px-3 py-2 text-xs text-gray-400">
                  {wsConnected
                    ? "Task started. Waiting for live prompt and status events..."
                    : "Task started. Waiting for WebSocket reconnect and live events..."}
                </div>
              )}
              {visibleMessages.map((entry, depth) => {
                const toneClasses = entry.tone === "error"
                  ? "border-red-800/80 bg-red-950/30 text-red-100"
                  : entry.tone === "success"
                    ? "border-emerald-900/60 bg-emerald-950/20 text-emerald-100"
                    : "border-gray-800 bg-black/60 text-gray-100";
                const stackStyle = {
                  opacity: Math.max(0.45, 1 - depth * 0.14),
                  transform: `translateX(${depth * 10}px) scale(${1 - depth * 0.02})`,
                  zIndex: visibleMessages.length - depth,
                  marginTop: depth === 0 ? undefined : "-0.2rem",
                } satisfies React.CSSProperties;
                if (!entry.detail) {
                  return (
                    <div
                      key={entry.id}
                      className={`relative rounded-md border px-2.5 py-2 transition-all duration-300 ease-out ${toneClasses}`}
                      style={stackStyle}
                    >
                      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-wide">
                        <span className="text-gray-500">{formatLogContext(entry)}</span>
                        <span className="shrink-0 text-gray-600">{formatLogTime(entry.ts)}</span>
                      </div>
                      <p className="mt-1 text-xs leading-5">
                        {entry.message}
                      </p>
                    </div>
                  );
                }
                return (
                  <details
                    key={entry.id}
                    className={`relative rounded-md border px-2.5 py-2 transition-all duration-300 ease-out ${toneClasses}`}
                    style={stackStyle}
                  >
                    <summary className="cursor-pointer list-none">
                      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-wide">
                        <span className="text-gray-500">{formatLogContext(entry)}</span>
                        <span className="shrink-0 text-gray-600">{formatLogTime(entry.ts)}</span>
                      </div>
                      <p className="mt-1 pr-6 text-xs leading-5">
                        {entry.message}
                      </p>
                    </summary>
                    <pre className="mt-2 overflow-x-auto rounded border border-gray-800 bg-black/70 px-2.5 py-2 text-[11px] leading-5 text-gray-200 whitespace-pre-wrap">
                      {entry.detail}
                    </pre>
                  </details>
                );
              })}
            </div>
          )}
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
