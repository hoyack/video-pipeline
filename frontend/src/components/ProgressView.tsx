import { useEffect, useState } from "react";
import { getProjectDetail, getDownloadUrl } from "../api/client.ts";
import type { ProjectDetail } from "../api/types.ts";
import { useProjectStatus } from "../hooks/useProjectStatus.ts";
import { TERMINAL_STATUSES } from "../lib/constants.ts";
import { PipelineStepper } from "./PipelineStepper.tsx";
import { SceneCard } from "./SceneCard.tsx";

interface ProgressViewProps {
  projectId: string;
  onViewDetail: (projectId: string) => void;
}

export function ProgressView({ projectId, onViewDetail }: ProgressViewProps) {
  const { status, error: pollError, isTerminal } = useProjectStatus(projectId);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);

  // Fetch full detail periodically to get scene info
  useEffect(() => {
    if (!projectId) return;

    let cancelled = false;

    async function fetchDetail() {
      try {
        const d = await getProjectDetail(projectId);
        if (!cancelled) setDetail(d);
      } catch {
        // Status polling handles errors
      }
    }

    fetchDetail();
    const id = setInterval(fetchDetail, 3000);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [projectId, isTerminal]);

  // Stop detail polling once terminal
  useEffect(() => {
    if (isTerminal && projectId) {
      getProjectDetail(projectId).then(setDetail).catch(() => {});
    }
  }, [isTerminal, projectId]);

  const currentStatus = status?.status ?? "pending";

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="mb-1 text-2xl font-bold text-white">Generation Progress</h1>
        <p className="text-sm text-gray-400">
          Project {projectId.slice(0, 8)}...
        </p>
      </div>

      {/* Stepper */}
      <div className="flex justify-center py-4">
        <PipelineStepper status={currentStatus} />
      </div>

      {/* Error message */}
      {status?.error_message && (
        <div className="rounded-md border border-red-800 bg-red-900/50 px-3 py-2 text-sm text-red-300">
          {status.error_message}
        </div>
      )}

      {pollError && (
        <div className="rounded-md border border-amber-800 bg-amber-900/50 px-3 py-2 text-sm text-amber-300">
          Polling error: {pollError}
        </div>
      )}

      {/* Scenes grid */}
      {detail && detail.scenes.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-medium text-gray-400">
            Scenes ({detail.scenes.length})
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {detail.scenes.map((scene) => (
              <SceneCard key={scene.scene_index} scene={scene} />
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      {isTerminal && (
        <div className="flex gap-3">
          {currentStatus === "complete" && (
            <a
              href={getDownloadUrl(projectId)}
              className="rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-500 transition-colors"
            >
              Download Video
            </a>
          )}
          <button
            onClick={() => onViewDetail(projectId)}
            className="rounded-lg border border-gray-700 px-4 py-2 text-sm font-medium text-gray-300 hover:border-gray-600 transition-colors"
          >
            View Details
          </button>
        </div>
      )}

      {/* Loading indicator for active pipelines */}
      {!isTerminal && !TERMINAL_STATUSES.has(currentStatus) && (
        <p className="text-center text-sm text-gray-500 animate-pulse">
          Processing...
        </p>
      )}
    </div>
  );
}
