import { useState } from "react";
import clsx from "clsx";
import type { SceneDetail } from "../api/types.ts";

function Dot({ filled, color }: { filled: boolean; color: string }) {
  return (
    <span
      className={clsx(
        "inline-block h-2 w-2 rounded-full",
        filled ? color : "bg-gray-700",
      )}
      title={filled ? "Ready" : "Pending"}
    />
  );
}

function PromptSection({ label, text }: { label: string; text: string }) {
  return (
    <div className="mt-2">
      <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
        {label}
      </span>
      <p className="mt-0.5 text-xs leading-relaxed text-gray-400">{text}</p>
    </div>
  );
}

export function SceneCard({
  scene,
  defaultExpanded = false,
}: {
  scene: SceneDetail;
  defaultExpanded?: boolean;
}) {
  const hasExpandableContent = !!(
    scene.start_frame_prompt ||
    scene.end_frame_prompt ||
    scene.video_motion_prompt ||
    scene.transition_notes ||
    scene.start_keyframe_url ||
    scene.end_keyframe_url ||
    scene.clip_url
  );
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div
      className={clsx(
        "rounded-lg border border-gray-800 bg-gray-900 p-3",
        hasExpandableContent && "cursor-pointer hover:border-gray-700",
      )}
      onClick={() => hasExpandableContent && setExpanded((prev) => !prev)}
    >
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400">
          Scene {scene.scene_index + 1}
        </span>
        <div className="flex items-center gap-1.5">
          {hasExpandableContent && (
            <svg
              className={clsx(
                "h-3.5 w-3.5 text-gray-500 transition-transform",
                expanded && "rotate-180",
              )}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          )}
          <span
            className={clsx(
              "text-[10px] font-medium uppercase",
              scene.has_clip ? "text-green-400" : "text-gray-500",
            )}
          >
            {scene.clip_status ?? scene.status}
          </span>
        </div>
      </div>
      <p
        className={clsx(
          "mb-2 text-sm leading-snug text-gray-300",
          !expanded && "line-clamp-2",
        )}
      >
        {scene.description}
      </p>
      {expanded && (
        <div className="border-t border-gray-800 pt-1">
          {(scene.start_keyframe_url || scene.end_keyframe_url) && (
            <div className="mt-2 flex gap-2">
              {scene.start_keyframe_url && (
                <div className="flex-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                    Start Keyframe
                  </span>
                  <img
                    src={scene.start_keyframe_url}
                    alt={`Scene ${scene.scene_index + 1} start`}
                    className="mt-0.5 w-full rounded border border-gray-700"
                    loading="lazy"
                  />
                </div>
              )}
              {scene.end_keyframe_url && (
                <div className="flex-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                    End Keyframe
                  </span>
                  <img
                    src={scene.end_keyframe_url}
                    alt={`Scene ${scene.scene_index + 1} end`}
                    className="mt-0.5 w-full rounded border border-gray-700"
                    loading="lazy"
                  />
                </div>
              )}
            </div>
          )}
          {scene.start_frame_prompt && (
            <PromptSection label="Start Frame Prompt" text={scene.start_frame_prompt} />
          )}
          {scene.end_frame_prompt && (
            <PromptSection label="End Frame Prompt" text={scene.end_frame_prompt} />
          )}
          {scene.video_motion_prompt && (
            <PromptSection label="Motion" text={scene.video_motion_prompt} />
          )}
          {scene.transition_notes && (
            <PromptSection label="Transition" text={scene.transition_notes} />
          )}
        </div>
      )}

      {/* Clip video player */}
      {scene.clip_url && (
        <div className="mt-2">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
            Clip
          </span>
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            src={scene.clip_url}
            className="mt-0.5 w-full rounded border border-gray-700"
            controls
            preload="metadata"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      <div className="mt-2 flex items-center gap-1.5">
        <Dot filled={scene.has_start_keyframe} color="bg-blue-400" />
        <Dot filled={scene.has_end_keyframe} color="bg-indigo-400" />
        <Dot filled={scene.has_clip} color="bg-green-400" />
        <span className="ml-1 text-[10px] text-gray-600">
          KF start / KF end / clip
        </span>
      </div>
    </div>
  );
}
