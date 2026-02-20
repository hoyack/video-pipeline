import clsx from "clsx";
import { PIPELINE_STAGES, STAGE_LABELS } from "../lib/constants.ts";

// Steps shown in stepper (exclude "pending" — it's the pre-start state)
const STEPS = PIPELINE_STAGES.filter((s) => s !== "pending");

function stageIndex(status: string): number {
  return PIPELINE_STAGES.indexOf(status as (typeof PIPELINE_STAGES)[number]);
}

// Map run_through value to the pipeline stage that was completed
const STAGED_COMPLETE_MAP: Record<string, string> = {
  storyboard: "storyboarding",
  keyframes: "keyframing",
  video: "video_gen",
};

interface PipelineStepperProps {
  status: string;
  runThrough?: string | null;
}

export function PipelineStepper({ status, runThrough }: PipelineStepperProps) {
  const isStaged = status === "staged";
  const isFailed = status === "failed";

  // For staged projects, resolve the effective stage from run_through
  const effectiveStatus = isStaged && runThrough
    ? STAGED_COMPLETE_MAP[runThrough] ?? status
    : status;
  const currentIdx = stageIndex(effectiveStatus);

  return (
    <div className="flex items-center gap-2">
      {STEPS.map((step, i) => {
        const stepIdx = stageIndex(step);
        const isComplete = isStaged
          ? stepIdx <= currentIdx  // staged: mark the boundary stage as complete too
          : currentIdx > stepIdx;
        const isCurrent = isStaged
          ? step === effectiveStatus  // the stage we paused at
          : status === step;

        return (
          <div key={step} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={clsx(
                  "h-0.5 w-8",
                  isComplete ? (isStaged ? "bg-cyan-500" : "bg-green-500") : "bg-gray-700",
                )}
              />
            )}
            <div className="flex flex-col items-center gap-1">
              <div
                className={clsx(
                  "flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold",
                  isComplete && !isCurrent && (isStaged ? "bg-green-600 text-white" : "bg-green-600 text-white"),
                  isCurrent && isStaged && "bg-cyan-600 text-white",
                  isCurrent && !isStaged && !isFailed && "bg-blue-500 text-white animate-pulse",
                  isCurrent && isFailed && "bg-red-600 text-white",
                  !isComplete && !isCurrent && "bg-gray-800 text-gray-500",
                )}
              >
                {isComplete && !isCurrent ? "\u2713" : isCurrent && isStaged ? "\u23F8" : isComplete ? "\u2713" : i + 1}
              </div>
              <span
                className={clsx(
                  "text-[10px]",
                  isCurrent ? (isStaged ? "text-cyan-300 font-medium" : "text-gray-200 font-medium") : "text-gray-500",
                )}
              >
                {STAGE_LABELS[step]}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
