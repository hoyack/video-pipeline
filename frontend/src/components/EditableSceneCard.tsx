import clsx from "clsx";
import type { SceneDetail } from "../api/types.ts";

interface EditableSceneCardProps {
  scene: SceneDetail;
  edits: Record<string, string>;
  onChange: (sceneIndex: number, field: string, value: string) => void;
  deleted: boolean;
  keyframesCleared: boolean;
  onDelete: (sceneIndex: number) => void;
  onRestoreScene: (sceneIndex: number) => void;
  onClearKeyframes: (sceneIndex: number) => void;
  onRestoreKeyframes: (sceneIndex: number) => void;
  canDelete: boolean;
}

function EditableField({
  label,
  value,
  originalValue,
  onChange,
  onClear,
}: {
  label: string;
  value: string;
  originalValue: string;
  onChange: (value: string) => void;
  onClear: () => void;
}) {
  const isModified = value !== originalValue;
  return (
    <div className="mt-2">
      <div className="flex items-center justify-between">
        <span
          className={clsx(
            "text-[10px] font-semibold uppercase tracking-wide",
            isModified ? "text-amber-400" : "text-gray-500",
          )}
        >
          {label} {isModified && "(edited)"}
        </span>
        {value && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onClear();
            }}
            className="text-gray-600 hover:text-red-400 transition-colors"
            title={`Clear ${label}`}
          >
            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
      <textarea
        rows={2}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        className={clsx(
          "mt-0.5 w-full rounded border bg-gray-950 px-2 py-1 text-xs leading-relaxed text-gray-300 focus:outline-none focus:ring-1",
          isModified
            ? "border-amber-600 focus:ring-amber-500"
            : "border-gray-700 focus:ring-blue-500",
        )}
      />
    </div>
  );
}

export function EditableSceneCard({
  scene,
  edits,
  onChange,
  deleted,
  keyframesCleared,
  onDelete,
  onRestoreScene,
  onClearKeyframes,
  onRestoreKeyframes,
  canDelete,
}: EditableSceneCardProps) {
  const idx = scene.scene_index;

  const getValue = (field: string, original: string | null | undefined) =>
    edits[field] ?? original ?? "";

  const getOriginal = (field: string, original: string | null | undefined) =>
    original ?? "";

  // Collapsed deleted state
  if (deleted) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-gray-900/50 p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-500 line-through">
            Scene {idx + 1} &mdash; {scene.description?.slice(0, 60)}
            {(scene.description?.length ?? 0) > 60 ? "..." : ""}
          </span>
          <button
            type="button"
            onClick={() => onRestoreScene(idx)}
            className="rounded px-2 py-0.5 text-[11px] font-medium text-blue-400 hover:bg-blue-500/10 transition-colors"
          >
            Restore
          </button>
        </div>
      </div>
    );
  }

  const hasKeyframes = scene.start_keyframe_url || scene.end_keyframe_url;

  return (
    <div
      className={clsx(
        "rounded-lg border bg-gray-900 p-3",
        Object.keys(edits).length > 0 ? "border-amber-700" : "border-gray-800",
      )}
    >
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400">
          Scene {idx + 1}
        </span>
        <div className="flex items-center gap-2">
          {Object.keys(edits).length > 0 && (
            <span className="text-[10px] font-medium uppercase text-amber-400">
              Modified
            </span>
          )}
          {canDelete && (
            <button
              type="button"
              onClick={() => onDelete(idx)}
              className="text-gray-600 hover:text-red-400 transition-colors"
              title="Delete scene"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Keyframe previews with clear overlay */}
      {hasKeyframes && !keyframesCleared && (
        <div className="mb-2 flex gap-2">
          {scene.start_keyframe_url && (
            <div className="relative flex-1 group">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                Start KF
              </span>
              <img
                src={scene.start_keyframe_url}
                alt={`Scene ${idx + 1} start`}
                className="mt-0.5 w-full rounded border border-gray-700"
                loading="lazy"
              />
              <button
                type="button"
                onClick={() => onClearKeyframes(idx)}
                className="absolute top-5 right-1 rounded-full bg-black/70 p-0.5 text-gray-400 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-all"
                title="Clear keyframes (will regenerate)"
              >
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
          {scene.end_keyframe_url && (
            <div className="relative flex-1 group">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                End KF
              </span>
              <img
                src={scene.end_keyframe_url}
                alt={`Scene ${idx + 1} end`}
                className="mt-0.5 w-full rounded border border-gray-700"
                loading="lazy"
              />
              <button
                type="button"
                onClick={() => onClearKeyframes(idx)}
                className="absolute top-5 right-1 rounded-full bg-black/70 p-0.5 text-gray-400 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-all"
                title="Clear keyframes (will regenerate)"
              >
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Keyframes cleared message */}
      {keyframesCleared && hasKeyframes && (
        <div className="mb-2 flex items-center justify-between rounded border border-amber-800/50 bg-amber-950/30 px-2 py-1.5">
          <span className="text-[11px] text-amber-400">
            Keyframes will be regenerated
          </span>
          <button
            type="button"
            onClick={() => onRestoreKeyframes(idx)}
            className="text-[11px] font-medium text-blue-400 hover:text-blue-300 transition-colors"
          >
            Restore
          </button>
        </div>
      )}

      <EditableField
        label="Description"
        value={getValue("scene_description", scene.description)}
        originalValue={getOriginal("scene_description", scene.description)}
        onChange={(v) => onChange(idx, "scene_description", v)}
        onClear={() => onChange(idx, "scene_description", "")}
      />
      <EditableField
        label="Start Frame Prompt"
        value={getValue("start_frame_prompt", scene.start_frame_prompt)}
        originalValue={getOriginal("start_frame_prompt", scene.start_frame_prompt)}
        onChange={(v) => onChange(idx, "start_frame_prompt", v)}
        onClear={() => onChange(idx, "start_frame_prompt", "")}
      />
      <EditableField
        label="End Frame Prompt"
        value={getValue("end_frame_prompt", scene.end_frame_prompt)}
        originalValue={getOriginal("end_frame_prompt", scene.end_frame_prompt)}
        onChange={(v) => onChange(idx, "end_frame_prompt", v)}
        onClear={() => onChange(idx, "end_frame_prompt", "")}
      />
      <EditableField
        label="Motion Prompt"
        value={getValue("video_motion_prompt", scene.video_motion_prompt)}
        originalValue={getOriginal("video_motion_prompt", scene.video_motion_prompt)}
        onChange={(v) => onChange(idx, "video_motion_prompt", v)}
        onClear={() => onChange(idx, "video_motion_prompt", "")}
      />
      <EditableField
        label="Transition Notes"
        value={getValue("transition_notes", scene.transition_notes)}
        originalValue={getOriginal("transition_notes", scene.transition_notes)}
        onChange={(v) => onChange(idx, "transition_notes", v)}
        onClear={() => onChange(idx, "transition_notes", "")}
      />
    </div>
  );
}
