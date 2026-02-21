import { useEffect, useState } from "react";
import { getProjectDetail, resumeProject, getDownloadUrl, updateProject } from "../api/client.ts";
import type { ProjectDetail as ProjectDetailType } from "../api/types.ts";
import { TERMINAL_STATUSES, TEXT_MODELS, IMAGE_MODELS, VIDEO_MODELS, estimateCost } from "../lib/constants.ts";
import { StatusBadge } from "./StatusBadge.tsx";
import { PipelineStepper } from "./PipelineStepper.tsx";
import { SceneCard } from "./SceneCard.tsx";
import { EditForkPanel } from "./EditForkPanel.tsx";
import { ContinuePanel } from "./ContinuePanel.tsx";
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
  const [continuing, setContinuing] = useState(false);
  const [continueTarget, setContinueTarget] = useState<string | null | undefined>(undefined); // undefined=closed, null|string=open with run_through value
  const [editing, setEditing] = useState(false);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");

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

  async function handleContinue(runThrough: string | null) {
    setContinuing(true);
    try {
      await resumeProject(projectId, { run_through: runThrough ?? "all" });
      onViewProgress(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Continue failed");
    } finally {
      setContinuing(false);
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
  const canContinue = detail.status === "staged" && detail.run_through != null;
  const canFork = isTerminal;
  const isRunning = !isTerminal;

  const NEXT_STAGE: Record<string, { run_through: string | null; label: string }> = {
    storyboard: { run_through: "keyframes", label: "Continue to Keyframes" },
    keyframes: { run_through: "video", label: "Continue to Video Gen" },
    video: { run_through: null, label: "Continue to Completion" },
  };

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
        {editingTitle ? (
          <form
            className="mb-1 flex items-center gap-2"
            onSubmit={async (e) => {
              e.preventDefault();
              const trimmed = titleDraft.trim();
              if (!trimmed) return;
              try {
                await updateProject(detail.project_id, { title: trimmed });
                setDetail({ ...detail, title: trimmed });
              } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to update title");
              }
              setEditingTitle(false);
            }}
          >
            <input
              autoFocus
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Escape") setEditingTitle(false); }}
              className="w-full rounded border border-gray-600 bg-gray-800 px-2 py-1 text-2xl font-bold text-white focus:border-blue-500 focus:outline-none"
              maxLength={200}
            />
            <button type="submit" className="rounded bg-blue-600 px-3 py-1 text-sm text-white hover:bg-blue-500">Save</button>
            <button type="button" onClick={() => setEditingTitle(false)} className="rounded border border-gray-600 px-3 py-1 text-sm text-gray-300 hover:border-gray-500">Cancel</button>
          </form>
        ) : (
          <h1
            className="mb-1 text-2xl font-bold text-white cursor-pointer hover:text-gray-300 transition-colors"
            onClick={() => { setTitleDraft(detail.title || ""); setEditingTitle(true); }}
            title="Click to edit title"
          >
            {detail.title || "Untitled Project"}
            <svg className="ml-2 inline h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          </h1>
        )}
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
        {detail.run_through && (
          <div>
            <span className="text-xs text-gray-500">Generate Through</span>
            <p className="mt-1 text-sm capitalize text-cyan-300">{detail.run_through}</p>
          </div>
        )}
      </div>

      {/* Prompt */}
      <div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500">Prompt</span>
          <CopyButton text={detail.prompt} />
        </div>
        {detail.prompt.length <= 150 ? (
          <p className="mt-1 text-sm leading-relaxed text-gray-300">
            {detail.prompt}
          </p>
        ) : (
          <button
            onClick={() => setPromptExpanded(!promptExpanded)}
            className="mt-1 w-full text-left"
          >
            <p className="text-sm leading-relaxed text-gray-300">
              {promptExpanded ? detail.prompt : `${detail.prompt.slice(0, 150)}...`}
            </p>
            <span className="text-xs text-gray-500 hover:text-gray-400 transition-colors">
              {promptExpanded ? "Show less" : "Show more"}
            </span>
          </button>
        )}
      </div>

      {/* Pipeline stepper */}
      <div className="flex justify-center py-2">
        <PipelineStepper status={detail.status} runThrough={detail.run_through} />
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

      {/* Continue Panel — shown when user clicks a Continue button */}
      {continueTarget !== undefined && canContinue && (
        <ContinuePanel
          detail={detail}
          nextRunThrough={continueTarget}
          onContinued={(pid) => {
            setContinueTarget(undefined);
            onViewProgress(pid);
          }}
          onCancel={() => setContinueTarget(undefined)}
        />
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
                defaultExpanded={false}
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
        {canContinue && detail.run_through && NEXT_STAGE[detail.run_through] && (
          detail.run_through === "video" ? (
            // Stitching needs no config — continue directly
            <button
              onClick={() => handleContinue(null)}
              disabled={continuing}
              className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 transition-colors disabled:opacity-50"
            >
              {continuing ? "Continuing..." : "Continue to Completion"}
            </button>
          ) : (
            // storyboard/keyframes need model config — open panel
            <>
              <button
                onClick={() => setContinueTarget(NEXT_STAGE[detail.run_through!].run_through)}
                className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 transition-colors"
              >
                {NEXT_STAGE[detail.run_through].label}
              </button>
              <button
                onClick={() => setContinueTarget(null)}
                className="rounded-lg border border-cyan-700 px-4 py-2 text-sm font-medium text-cyan-300 hover:border-cyan-600 transition-colors"
              >
                Run to Completion
              </button>
            </>
          )
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
