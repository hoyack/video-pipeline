import clsx from "clsx";
import { PIPELINE_STAGES, STAGE_LABELS } from "../lib/constants.ts";

// Steps shown in stepper (exclude "pending" — it's the pre-start state)
const STEPS = PIPELINE_STAGES.filter((s) => s !== "pending");

function stageIndex(status: string): number {
  return PIPELINE_STAGES.indexOf(status as (typeof PIPELINE_STAGES)[number]);
}

export function PipelineStepper({ status }: { status: string }) {
  const currentIdx = stageIndex(status);
  const isFailed = status === "failed";

  return (
    <div className="flex items-center gap-2">
      {STEPS.map((step, i) => {
        const stepIdx = stageIndex(step);
        const isComplete = currentIdx > stepIdx;
        const isCurrent = status === step;

        return (
          <div key={step} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={clsx(
                  "h-0.5 w-8",
                  isComplete ? "bg-green-500" : "bg-gray-700",
                )}
              />
            )}
            <div className="flex flex-col items-center gap-1">
              <div
                className={clsx(
                  "flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold",
                  isComplete && "bg-green-600 text-white",
                  isCurrent && !isFailed && "bg-blue-500 text-white animate-pulse",
                  isCurrent && isFailed && "bg-red-600 text-white",
                  !isComplete && !isCurrent && "bg-gray-800 text-gray-500",
                )}
              >
                {isComplete ? "\u2713" : i + 1}
              </div>
              <span
                className={clsx(
                  "text-[10px]",
                  isCurrent ? "text-gray-200 font-medium" : "text-gray-500",
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
