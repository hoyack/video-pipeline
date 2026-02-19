import { useState, useEffect } from "react";
import clsx from "clsx";
import type { SceneDetail, CandidateScore } from "../api/types.ts";
import { listCandidates, selectCandidate } from "../api/client.ts";
import { CopyButton } from "./CopyButton.tsx";
import { ImageLightbox } from "./ImageLightbox.tsx";

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

function PromptSection({ label, text, copyable }: { label: string; text: string; copyable?: boolean }) {
  return (
    <div className="mt-2">
      <div className="flex items-center gap-1">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
          {label}
        </span>
        {copyable && <CopyButton text={text} />}
      </div>
      <p className="mt-0.5 text-xs leading-relaxed text-gray-400">{text}</p>
    </div>
  );
}

export function SceneCard({
  scene,
  defaultExpanded = false,
  projectId,
  qualityMode = false,
  onViewManifest,
  manifestId,
}: {
  scene: SceneDetail;
  defaultExpanded?: boolean;
  projectId?: string;
  qualityMode?: boolean;
  onViewManifest?: (manifestId: string) => void;
  manifestId?: string | null;
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
  const [candidates, setCandidates] = useState<CandidateScore[]>([]);
  const [candidatesLoaded, setCandidatesLoaded] = useState(false);
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    if (qualityMode && projectId && expanded && !candidatesLoaded) {
      listCandidates(projectId, scene.scene_index)
        .then((data) => {
          setCandidates(data);
          setCandidatesLoaded(true);
        })
        .catch(() => {}); // Non-critical — scene card still shows normally
    }
  }, [qualityMode, projectId, expanded, candidatesLoaded, scene.scene_index]);

  const sceneLabel = `Scene ${scene.scene_index + 1}`;

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
          {sceneLabel}
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
                    alt={`${sceneLabel} start`}
                    className="mt-0.5 w-full cursor-zoom-in rounded border border-gray-700 hover:border-gray-500 transition-colors"
                    loading="lazy"
                    onClick={(e) => {
                      e.stopPropagation();
                      setLightboxImage({ src: scene.start_keyframe_url!, title: `${sceneLabel} — Start Keyframe` });
                    }}
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
                    alt={`${sceneLabel} end`}
                    className="mt-0.5 w-full cursor-zoom-in rounded border border-gray-700 hover:border-gray-500 transition-colors"
                    loading="lazy"
                    onClick={(e) => {
                      e.stopPropagation();
                      setLightboxImage({ src: scene.end_keyframe_url!, title: `${sceneLabel} — End Keyframe` });
                    }}
                  />
                </div>
              )}
            </div>
          )}
          {/* Identity References (Phase 8) */}
          {scene.selected_references && scene.selected_references.length > 0 && (
            <div className="mt-2">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                Identity References
              </span>
              <div className="mt-1 flex flex-wrap gap-2">
                {scene.selected_references.map((ref) => {
                  const canNavigate = !!(onViewManifest && manifestId);
                  return (
                    <button
                      key={ref.asset_id}
                      className={clsx(
                        "flex items-center gap-1.5 rounded border bg-gray-800/50 px-2 py-1",
                        canNavigate
                          ? "border-gray-700 cursor-pointer hover:border-blue-500 transition-colors"
                          : "border-gray-700 cursor-default",
                      )}
                      title={canNavigate ? `${ref.name} — Click to view manifest` : ref.name}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (canNavigate) onViewManifest!(manifestId!);
                      }}
                    >
                      {(ref.thumbnail_url || ref.reference_image_url) && (
                        <img
                          src={ref.thumbnail_url || ref.reference_image_url || ""}
                          alt={ref.manifest_tag}
                          className="h-6 w-6 rounded object-cover"
                          loading="lazy"
                        />
                      )}
                      <span className="text-[10px] font-medium text-blue-400">
                        {ref.manifest_tag}
                      </span>
                      {ref.quality_score != null && (
                        <span className="text-[10px] text-gray-500">
                          {ref.quality_score.toFixed(1)}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
          {scene.start_frame_prompt && (
            <PromptSection label="Start Frame Prompt" text={scene.start_frame_prompt} copyable />
          )}
          {scene.end_frame_prompt && (
            <PromptSection label="End Frame Prompt" text={scene.end_frame_prompt} copyable />
          )}
          {scene.video_motion_prompt && (
            <PromptSection label="Motion" text={scene.video_motion_prompt} copyable />
          )}
          {scene.transition_notes && (
            <PromptSection label="Transition" text={scene.transition_notes} copyable />
          )}
          {/* Quality Mode — Candidate Comparison (Phase 11) */}
          {candidates.length > 1 && (
            <div className="mt-3 border-t border-gray-800 pt-3">
              <h4 className="text-[10px] font-semibold uppercase tracking-wide text-gray-500 mb-2">
                Quality Mode — {candidates.length} Candidates
              </h4>
              <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
                {candidates.map((c) => (
                  <button
                    key={c.candidate_id}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (!c.is_selected && projectId) {
                        selectCandidate(projectId, scene.scene_index, c.candidate_id)
                          .then(() => {
                            setCandidates((prev) =>
                              prev.map((p) => ({
                                ...p,
                                is_selected: p.candidate_id === c.candidate_id,
                                selected_by: p.candidate_id === c.candidate_id ? "user" : p.selected_by,
                              }))
                            );
                          })
                          .catch(() => {});
                      }
                    }}
                    className={clsx(
                      "rounded-lg border p-2 text-left transition-colors text-xs",
                      c.is_selected
                        ? "border-amber-500 bg-amber-500/10"
                        : "border-gray-800 bg-gray-900/50 hover:border-gray-700"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-gray-300">
                        Take {c.candidate_number + 1}
                      </span>
                      {c.is_selected && (
                        <span className="text-[9px] font-bold uppercase text-amber-400">
                          {c.selected_by === "user" ? "User Pick" : "Best"}
                        </span>
                      )}
                    </div>
                    {c.composite_score != null && (
                      <div className="text-lg font-bold text-gray-200 mb-1">
                        {c.composite_score.toFixed(1)}
                      </div>
                    )}
                    <div className="space-y-0.5 text-[10px] text-gray-500">
                      {c.manifest_adherence_score != null && (
                        <div className="flex justify-between">
                          <span>Manifest</span>
                          <span className="text-gray-400">{c.manifest_adherence_score.toFixed(1)}</span>
                        </div>
                      )}
                      {c.visual_quality_score != null && (
                        <div className="flex justify-between">
                          <span>Quality</span>
                          <span className="text-gray-400">{c.visual_quality_score.toFixed(1)}</span>
                        </div>
                      )}
                      {c.continuity_score != null && (
                        <div className="flex justify-between">
                          <span>Continuity</span>
                          <span className="text-gray-400">{c.continuity_score.toFixed(1)}</span>
                        </div>
                      )}
                      {c.prompt_adherence_score != null && (
                        <div className="flex justify-between">
                          <span>Prompt</span>
                          <span className="text-gray-400">{c.prompt_adherence_score.toFixed(1)}</span>
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
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

      {/* Image Lightbox */}
      {lightboxImage && (
        <ImageLightbox
          src={lightboxImage.src}
          title={lightboxImage.title}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </div>
  );
}
