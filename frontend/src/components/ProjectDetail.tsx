import { useEffect, useState } from "react";
import { getProjectDetail, resumeProject, getDownloadUrl } from "../api/client.ts";
import type { ProjectDetail as ProjectDetailType } from "../api/types.ts";
import { TERMINAL_STATUSES, TEXT_MODELS, IMAGE_MODELS, VIDEO_MODELS, estimateCost } from "../lib/constants.ts";
import { StatusBadge } from "./StatusBadge.tsx";
import { PipelineStepper } from "./PipelineStepper.tsx";
import { SceneCard } from "./SceneCard.tsx";
import { EditForkPanel } from "./EditForkPanel.tsx";
import { CopyButton } from "./CopyButton.tsx";
import type { SceneDetail } from "../api/types.ts";

function CopyAllScenesButton({ scenes }: { scenes: SceneDetail[] }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    const text = scenes
      .map((s) => {
        const lines = [`Scene ${s.scene_index + 1}:`];
        lines.push(`  Description: ${s.description}`);
        if (s.start_frame_prompt) lines.push(`  Start Frame: ${s.start_frame_prompt}`);
        if (s.end_frame_prompt) lines.push(`  End Frame: ${s.end_frame_prompt}`);
        if (s.video_motion_prompt) lines.push(`  Motion: ${s.video_motion_prompt}`);
        if (s.transition_notes) lines.push(`  Transition: ${s.transition_notes}`);
        return lines.join("\n");
      })
      .join("\n\n");

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1 rounded border border-gray-700 px-2 py-1 text-xs text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors"
    >
      {copied ? (
        <>
          <svg className="h-3 w-3 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          Copied!
        </>
      ) : (
        <>
          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Copy All Scenes
        </>
      )}
    </button>
  );
}

function modelLabel(
  catalogs: { id: string; label: string }[][],
  id: string | null | undefined,
): string | null {
  if (!id) return null;
  for (const catalog of catalogs) {
    const found = catalog.find((m) => m.id === id);
    if (found) return found.label;
  }
  return id; // fallback to raw ID
}

interface ProjectDetailProps {
  projectId: string;
  onViewProgress: (projectId: string) => void;
  onForked?: (newProjectId: string) => void;
  onViewProject?: (projectId: string) => void;
  onViewManifest?: (manifestId: string) => void;
}

export function ProjectDetail({ projectId, onViewProgress, onForked, onViewProject, onViewManifest }: ProjectDetailProps) {
  const [detail, setDetail] = useState<ProjectDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    getProjectDetail(projectId)
      .then((d) => {
        if (!cancelled) {
          setDetail(d);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [projectId]);

  async function handleResume() {
    setResuming(true);
    try {
      await resumeProject(projectId);
      onViewProgress(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Resume failed");
    } finally {
      setResuming(false);
    }
  }

  if (loading) {
    return <p className="text-center text-sm text-gray-500">Loading...</p>;
  }

  if (error && !detail) {
    return <p className="text-center text-sm text-red-400">{error}</p>;
  }

  if (!detail) return null;

  const isTerminal = TERMINAL_STATUSES.has(detail.status);
  const canResume = detail.status === "failed" || detail.status === "stopped";
  const canFork = isTerminal;
  const isRunning = !isTerminal;

  if (editing && detail) {
    return (
      <div className="mx-auto max-w-3xl space-y-6">
        <EditForkPanel
          detail={detail}
          onForked={(newId) => {
            setEditing(false);
            if (onForked) onForked(newId);
          }}
          onCancel={() => setEditing(false)}
        />
      </div>
    );
  }

  const costEstimate = detail.total_duration && detail.clip_duration && detail.text_model && detail.image_model && detail.video_model
    ? estimateCost(detail.total_duration, detail.clip_duration, detail.text_model, detail.image_model, detail.video_model, detail.audio_enabled ?? false)
    : null;

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* Header */}
      <div>
        <h1 className="mb-1 text-2xl font-bold text-white">Project Detail</h1>
        <div className="flex items-center gap-1">
          <p className="text-sm text-gray-500 font-mono">{detail.project_id}</p>
          <CopyButton text={detail.project_id} />
        </div>
        {detail.forked_from && (
          <p className="mt-1 text-xs text-gray-500">
            Forked from{" "}
            <button
              onClick={() => onViewProject?.(detail.forked_from!)}
              className="font-mono text-blue-400 hover:text-blue-300 underline"
            >
              {detail.forked_from.slice(0, 8)}...
            </button>
          </p>
        )}
      </div>

      {/* Meta */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <span className="text-xs text-gray-500">Status</span>
          <div className="mt-1">
            <StatusBadge status={detail.status} />
          </div>
        </div>
        <div>
          <span className="text-xs text-gray-500">Style</span>
          <p className="mt-1 text-sm capitalize text-gray-200">
            {detail.style.replace("_", " ")}
          </p>
        </div>
        <div>
          <span className="text-xs text-gray-500">Aspect Ratio</span>
          <p className="mt-1 text-sm text-gray-200">{detail.aspect_ratio}</p>
        </div>
        <div>
          <span className="text-xs text-gray-500">Scenes</span>
          <p className="mt-1 text-sm text-gray-200">{detail.scene_count}</p>
        </div>
        {detail.total_duration && (
          <div>
            <span className="text-xs text-gray-500">Total Duration</span>
            <p className="mt-1 text-sm text-gray-200">{detail.total_duration}s</p>
          </div>
        )}
        {detail.text_model && (
          <div>
            <span className="text-xs text-gray-500">Text Model</span>
            <p className="mt-1 text-sm text-gray-200">
              {modelLabel([TEXT_MODELS], detail.text_model)}
            </p>
          </div>
        )}
        {detail.image_model && (
          <div>
            <span className="text-xs text-gray-500">Image Model</span>
            <p className="mt-1 text-sm text-gray-200">
              {modelLabel([IMAGE_MODELS], detail.image_model)}
            </p>
          </div>
        )}
        {detail.video_model && (
          <div>
            <span className="text-xs text-gray-500">Video Model</span>
            <p className="mt-1 text-sm text-gray-200">
              {modelLabel([VIDEO_MODELS], detail.video_model)}
            </p>
          </div>
        )}
        {detail.audio_enabled != null && (
          <div>
            <span className="text-xs text-gray-500">Audio</span>
            <p className="mt-1 text-sm text-gray-200">
              {detail.audio_enabled ? "Enabled" : "Disabled"}
            </p>
          </div>
        )}
        {costEstimate != null && (
          <div>
            <span className="text-xs text-gray-500">Est. Cost</span>
            <p className="mt-1 text-sm font-mono text-gray-200">
              ${costEstimate.toFixed(2)}
            </p>
          </div>
        )}
      </div>

      {/* Prompt */}
      <div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500">Prompt</span>
          <CopyButton text={detail.prompt} />
        </div>
        <p className="mt-1 text-sm leading-relaxed text-gray-300">
          {detail.prompt}
        </p>
      </div>

      {/* Pipeline stepper */}
      <div className="flex justify-center py-2">
        <PipelineStepper status={detail.status} />
      </div>

      {/* Error */}
      {detail.error_message && (
        <div className="rounded-md border border-red-800 bg-red-900/50 px-3 py-2 text-sm text-red-300">
          {detail.error_message}
        </div>
      )}

      {error && (
        <div className="rounded-md border border-amber-800 bg-amber-900/50 px-3 py-2 text-sm text-amber-300">
          {error}
        </div>
      )}

      {/* Final video player */}
      {detail.status === "complete" && (
        <div>
          <h2 className="mb-3 text-sm font-medium text-gray-400">Final Video</h2>
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            src={getDownloadUrl(projectId)}
            className="w-full rounded-lg border border-gray-800"
            controls
            preload="metadata"
          />
        </div>
      )}

      {/* Scenes */}
      {detail.scenes.length > 0 && (
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-400">
              Scenes ({detail.scenes.length})
            </h2>
            <CopyAllScenesButton scenes={detail.scenes} />
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {detail.scenes.map((scene) => (
              <SceneCard
                key={scene.scene_index}
                scene={scene}
                defaultExpanded
                projectId={detail.project_id}
                qualityMode={detail.quality_mode}
                onViewManifest={onViewManifest}
                manifestId={detail.manifest_id}
              />
            ))}
          </div>
        </div>
      )}

      {/* Timestamps */}
      <div className="flex gap-6 text-xs text-gray-600">
        <span>Created: {new Date(detail.created_at).toLocaleString()}</span>
        <span>Updated: {new Date(detail.updated_at).toLocaleString()}</span>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        {detail.status === "complete" && (
          <a
            href={getDownloadUrl(projectId)}
            className="rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-500 transition-colors"
          >
            Download Video
          </a>
        )}
        {canResume && (
          <button
            onClick={handleResume}
            disabled={resuming}
            className="rounded-lg bg-amber-600 px-4 py-2 text-sm font-semibold text-white hover:bg-amber-500 transition-colors disabled:opacity-50"
          >
            {resuming ? "Resuming..." : "Resume Pipeline"}
          </button>
        )}
        {canFork && (
          <button
            onClick={() => setEditing(true)}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-500 transition-colors"
          >
            Edit & Fork
          </button>
        )}
        {isRunning && (
          <button
            onClick={() => onViewProgress(projectId)}
            className="rounded-lg border border-gray-700 px-4 py-2 text-sm font-medium text-gray-300 hover:border-gray-600 transition-colors"
          >
            View Progress
          </button>
        )}
      </div>
    </div>
  );
}
