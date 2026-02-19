import type { ProjectListItem } from "../api/types.ts";
import { estimateCost } from "../lib/constants.ts";
import { TERMINAL_STATUSES } from "../lib/constants.ts";
import { StatusBadge } from "./StatusBadge.tsx";

function formatCost(item: ProjectListItem): string | null {
  if (!item.total_duration || !item.clip_duration || !item.text_model || !item.image_model || !item.video_model) {
    return null;
  }
  const cost = estimateCost(
    item.total_duration,
    item.clip_duration,
    item.text_model,
    item.image_model,
    item.video_model,
    item.audio_enabled ?? false,
  );
  return `$${cost.toFixed(2)}`;
}

/** Short display name for a video model ID */
function shortModel(modelId: string | null | undefined): string | null {
  if (!modelId) return null;
  // e.g. "veo-3.1-generate-preview" → "Veo 3.1"
  const match = modelId.match(/veo[- ]?([\d.]+)/i);
  if (match) return `Veo ${match[1]}`;
  // comfyui models
  if (modelId.startsWith("comfyui/")) return modelId.split("/")[1];
  return modelId;
}

interface ProjectCardProps {
  project: ProjectListItem;
  onClick: () => void;
  onDelete: () => void;
}

export function ProjectCard({ project, onClick, onDelete }: ProjectCardProps) {
  const cost = formatCost(project);
  const canDelete = TERMINAL_STATUSES.has(project.status);

  return (
    <div
      onClick={onClick}
      className="group cursor-pointer rounded-lg border border-gray-800 bg-gray-900/50 hover:border-gray-700 transition-colors overflow-hidden"
    >
      {/* Thumbnail area */}
      <div className="relative aspect-video bg-gray-800">
        {project.thumbnail_url ? (
          <img
            src={project.thumbnail_url}
            alt=""
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-gray-600">
            <svg className="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
            </svg>
          </div>
        )}
        {/* Status badge overlaid */}
        <div className="absolute top-2 right-2">
          <StatusBadge status={project.status} />
        </div>
      </div>

      {/* Body */}
      <div className="p-3 space-y-2">
        {/* Prompt (2-line clamp) */}
        <p className="text-sm text-gray-200 line-clamp-2 leading-snug">
          {project.prompt}
        </p>

        {/* Metadata chips */}
        <div className="flex flex-wrap gap-1.5">
          {project.aspect_ratio && (
            <span className="inline-block rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-400">
              {project.aspect_ratio}
            </span>
          )}
          {project.audio_enabled && (
            <span className="inline-block rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-400">
              Audio
            </span>
          )}
          {shortModel(project.video_model) && (
            <span className="inline-block rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-400">
              {shortModel(project.video_model)}
            </span>
          )}
        </div>

        {/* Cost + date + delete */}
        <div className="flex items-center justify-between text-xs">
          <span className="font-mono text-gray-400">{cost ?? ""}</span>
          <div className="flex items-center gap-2">
            <span className="text-gray-500">
              {new Date(project.created_at).toLocaleDateString()}
            </span>
            {canDelete && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="text-gray-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                title="Delete project"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
