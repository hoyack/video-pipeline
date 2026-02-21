import clsx from "clsx";
import { useState, useRef, useEffect, useCallback } from "react";
import type { SceneDetail } from "../api/types.ts";
import {
  regenerateScene,
  regenerateSceneText,
  generateSceneFields,
  uploadKeyframe,
  uploadClip,
  deleteSceneKeyframe,
  deleteSceneClip,
} from "../api/client.ts";
import { usePolling } from "../hooks/usePolling.ts";
import { CopyButton } from "./CopyButton.tsx";
import { MarkdownEditorModal } from "./MarkdownEditorModal.tsx";

interface SceneEditorCardProps {
  scene: SceneDetail;
  edits: Record<string, string>;
  onChange: (sceneIndex: number, field: string, value: string) => void;
  onRemove: (sceneIndex: number) => void;
  removed: boolean;
  onRestore: (sceneIndex: number) => void;
  canRemove: boolean;
  projectId?: string;
  onAssetChanged?: () => void;
  /** Called when a regen fires — passes the pre-regen head_sha for revert-on-cancel */
  onRegenStarted?: (headSha: string | null) => void;
  /** Current text model selection from edit mode (used for text regen) */
  textModel?: string;
  /** Current video model selection from edit mode (used for clip regen) */
  videoModel?: string;
  /** Current image model selection from edit mode (used for keyframe regen) */
  imageModel?: string;
  /** All scene edits across the project — for generate-scene-fields neighbor context */
  allSceneEdits?: Record<number, Record<string, string>>;
  /** Called to generate a complete new scene (text + assets) from an empty slot */
  onGenerateScene?: (sceneIndex: number) => Promise<void>;
  /** True when background asset generation is in progress for this scene */
  isGeneratingAssets?: boolean;
}

function StalenessBadge({ staleness }: { staleness: string | null | undefined }) {
  if (!staleness || staleness === "fresh") return null;
  if (staleness === "missing") {
    return (
      <span className="inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] font-medium bg-gray-800 text-gray-400">
        Missing
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] font-medium bg-amber-900/50 text-amber-400">
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-400" />
      Stale
    </span>
  );
}

function Spinner({ className }: { className?: string }) {
  return (
    <svg className={clsx("animate-spin", className ?? "h-3 w-3")} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

function EditableField({
  label,
  value,
  originalValue,
  onChange,
  onClear,
  staleness,
  onRegen,
  regenerating,
  showContext,
  onToggleContext,
  contextValue,
  onContextChange,
  onOpenEditor,
}: {
  label: string;
  value: string;
  originalValue: string;
  onChange: (value: string) => void;
  onClear: () => void;
  staleness?: string | null;
  onRegen?: () => void;
  regenerating?: boolean;
  showContext?: boolean;
  onToggleContext?: () => void;
  contextValue?: string;
  onContextChange?: (v: string) => void;
  onOpenEditor?: () => void;
}) {
  const isModified = value !== originalValue;
  return (
    <div className="mt-2">
      <div className="flex items-center justify-between gap-1">
        <span
          className={clsx(
            "text-[10px] font-semibold uppercase tracking-wide",
            isModified ? "text-amber-400" : "text-gray-500",
          )}
        >
          {label} {isModified && "(edited)"}
        </span>
        <div className="flex items-center gap-1">
          {staleness && <StalenessBadge staleness={staleness} />}
          {onRegen && (
            <>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onRegen(); }}
                disabled={regenerating}
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
                title={`Regenerate ${label}`}
              >
                {regenerating && <Spinner className="h-2.5 w-2.5 text-indigo-400" />}
                {regenerating ? "Generating..." : "Regen"}
              </button>
              {onToggleContext && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onToggleContext(); }}
                  className={clsx(
                    "rounded px-1 py-0.5 text-[10px] transition-colors",
                    showContext ? "text-indigo-300 bg-indigo-900/30" : "text-gray-500 hover:text-gray-400",
                  )}
                  title="Add direction"
                >
                  +
                </button>
              )}
            </>
          )}
          <CopyButton text={value} />
          {onOpenEditor && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onOpenEditor(); }}
              className="inline-flex items-center justify-center h-5 w-5 rounded hover:bg-gray-700/50 transition-colors"
              title="Edit in markdown editor"
            >
              <svg className="h-3 w-3 text-gray-500 hover:text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
              </svg>
            </button>
          )}
          {value && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onClear(); }}
              className="text-gray-600 hover:text-red-400 transition-colors"
              title={`Clear ${label}`}
            >
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
      {showContext && onContextChange && (
        <input
          type="text"
          placeholder="Extra direction (optional)..."
          value={contextValue ?? ""}
          onChange={(e) => onContextChange(e.target.value)}
          onClick={(e) => e.stopPropagation()}
          className="mt-0.5 w-full rounded border border-gray-700 bg-gray-950 px-2 py-0.5 text-[10px] text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
      )}
      <div className="relative mt-0.5">
        <textarea
          rows={2}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onClick={(e) => e.stopPropagation()}
          className={clsx(
            "w-full rounded border bg-gray-950 px-2 py-1 text-xs leading-relaxed text-gray-300 focus:outline-none focus:ring-1",
            regenerating
              ? "border-indigo-600/50 text-gray-500"
              : isModified
                ? "border-amber-600 focus:ring-amber-500"
                : "border-gray-700 focus:ring-blue-500",
          )}
        />
        {regenerating && (
          <div className="absolute inset-0 flex items-center justify-center rounded bg-gray-950/60">
            <div className="flex items-center gap-1.5 text-[10px] text-indigo-400">
              <Spinner className="h-3.5 w-3.5 text-indigo-400" />
              <span>Generating...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function SceneEditorCard({
  scene,
  edits,
  onChange,
  onRemove,
  removed,
  onRestore,
  canRemove,
  projectId,
  onAssetChanged,
  onRegenStarted,
  textModel,
  videoModel,
  imageModel,
  allSceneEdits,
  onGenerateScene,
  isGeneratingAssets,
}: SceneEditorCardProps) {
  const idx = scene.scene_index;
  const [promptDetailsOpen, setPromptDetailsOpen] = useState(false);
  const [regenerating, setRegenerating] = useState<string | null>(null);
  const [regenQueued, setRegenQueued] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const startKfInput = useRef<HTMLInputElement>(null);
  const endKfInput = useRef<HTMLInputElement>(null);
  const clipInput = useRef<HTMLInputElement>(null);

  // Polling state for regen completion
  const [pollingTarget, setPollingTarget] = useState<string | null>(null);
  const baselineUrl = useRef<string | null>(null);
  const pollCount = useRef(0);
  const POLL_INTERVAL = 5000;
  const MAX_POLLS = 120; // 120 × 5s = 600s (10 min)

  // Extra context state (for keyframe/clip regen)
  const [showContextFor, setShowContextFor] = useState<string | null>(null);
  const [extraContext, setExtraContext] = useState<Record<string, string>>({});

  // Text field regen state (Set allows multiple concurrent regens)
  const [regenTextFields, setRegenTextFields] = useState<Set<string>>(new Set());
  const [showTextContextFor, setShowTextContextFor] = useState<string | null>(null);
  const [textExtraContext, setTextExtraContext] = useState<Record<string, string>>({});

  // Lightbox state
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [lightboxLabel, setLightboxLabel] = useState("");

  // Markdown editor modal state
  const [editorField, setEditorField] = useState<string | null>(null);

  // Empty slot generation state (must be unconditional to satisfy rules of hooks)
  const [emptySlotGenerating, setEmptySlotGenerating] = useState(false);
  const [emptySlotGenError, setEmptySlotGenError] = useState<string | null>(null);

  // Map target → scene URL field
  function getUrlForTarget(target: string): string | null | undefined {
    if (target === "start_keyframe") return scene.start_keyframe_url;
    if (target === "end_keyframe") return scene.end_keyframe_url;
    if (target === "video_clip") return scene.clip_url;
    return undefined;
  }

  // Poll callback: trigger parent refresh
  const handlePollTick = useCallback(() => {
    pollCount.current += 1;
    if (pollCount.current > MAX_POLLS) {
      setPollingTarget(null);
      setRegenQueued(null);
      setActionError("Regeneration may still be running — click Refresh");
      return;
    }
    onAssetChanged?.();
  }, [onAssetChanged]);

  usePolling(handlePollTick, POLL_INTERVAL, !!pollingTarget);

  // Detect completion: URL changed from baseline
  useEffect(() => {
    if (!pollingTarget) return;
    const currentUrl = getUrlForTarget(pollingTarget);
    if (currentUrl && currentUrl !== baselineUrl.current) {
      setPollingTarget(null);
      setRegenQueued(null);
      pollCount.current = 0;
    }
  }, [pollingTarget, scene.start_keyframe_url, scene.end_keyframe_url, scene.clip_url]);

  async function handleRegenerate(target: string) {
    if (!projectId) return;
    setRegenerating(target);
    setRegenQueued(null);
    setActionError(null);

    // Build prompt_overrides from extra context
    const context = extraContext[target]?.trim();
    let promptOverrides: Record<string, string> | undefined;
    if (context) {
      let basePrompt: string;
      if (target === "video_clip") {
        basePrompt = edits["video_motion_prompt"] ?? scene.rewritten_video_prompt ?? scene.video_motion_prompt ?? "";
      } else if (target === "start_keyframe") {
        basePrompt = edits["start_frame_prompt"] ?? scene.rewritten_keyframe_prompt ?? scene.start_frame_prompt ?? "";
      } else {
        basePrompt = edits["end_frame_prompt"] ?? scene.rewritten_keyframe_prompt ?? scene.end_frame_prompt ?? "";
      }
      promptOverrides = { [target]: basePrompt + "\n\n[Additional direction: " + context + "]" };
    }

    try {
      // Snapshot baseline URL before firing
      baselineUrl.current = getUrlForTarget(target) ?? null;
      pollCount.current = 0;

      const resp = await regenerateScene(projectId, idx, {
        targets: [target],
        skip_checkpoint: true,
        ...(promptOverrides && { prompt_overrides: promptOverrides }),
        video_model: videoModel,
        image_model: imageModel,
        scene_edits: Object.keys(edits).length > 0 ? edits : undefined,
      });
      // Notify parent of baseline sha for revert-on-cancel
      onRegenStarted?.(resp.head_sha ?? null);
      // 202 accepted — start polling for completion
      setRegenQueued(target);
      setPollingTarget(target);
      // Clear extra context input on success
      if (context) {
        setExtraContext((prev) => { const next = { ...prev }; delete next[target]; return next; });
        setShowContextFor(null);
      }
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Regenerate failed");
    } finally {
      setRegenerating(null);
    }
  }

  async function handleUploadKeyframe(position: string, file: File) {
    if (!projectId) return;
    setActionError(null);
    try {
      await uploadKeyframe(projectId, idx, position, file);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  async function handleUploadClip(file: File) {
    if (!projectId) return;
    setActionError(null);
    try {
      await uploadClip(projectId, idx, file);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  async function handleDeleteKeyframe(position: string) {
    if (!projectId) return;
    setActionError(null);
    try {
      await deleteSceneKeyframe(projectId, idx, position);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function handleDeleteClip() {
    if (!projectId) return;
    setActionError(null);
    try {
      await deleteSceneClip(projectId, idx);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function handleTextRegen(field: string) {
    if (!projectId) return;
    setRegenTextFields((prev) => new Set(prev).add(field));
    setActionError(null);
    try {
      const resp = await regenerateSceneText(projectId, idx, {
        field,
        extra_context: textExtraContext[field]?.trim() || undefined,
        text_model: textModel,
        scene_edits: Object.keys(edits).length > 0 ? edits : undefined,
      });
      onChange(idx, field, resp.text);
      // Clear extra context on success
      if (textExtraContext[field]) {
        setTextExtraContext((prev) => { const next = { ...prev }; delete next[field]; return next; });
        setShowTextContextFor(null);
      }
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Text regen failed");
    } finally {
      setRegenTextFields((prev) => { const next = new Set(prev); next.delete(field); return next; });
    }
  }

  const getValue = (field: string, original: string | null | undefined) =>
    edits[field] ?? original ?? "";
  const getOriginal = (_field: string, original: string | null | undefined) =>
    original ?? "";

  // Empty slot — full card with all 5 editable fields + Generate button
  if (scene.is_empty_slot) {
    async function handleGenerateScene() {
      if (!onGenerateScene) return;
      setEmptySlotGenerating(true);
      setEmptySlotGenError(null);
      try {
        await onGenerateScene(idx);
      } catch (err) {
        setEmptySlotGenError(err instanceof Error ? err.message : "Generation failed");
      } finally {
        setEmptySlotGenerating(false);
      }
    }

    async function handleGenerateTextOnly() {
      if (!projectId) return;
      setEmptySlotGenerating(true);
      setEmptySlotGenError(null);
      try {
        const resp = await generateSceneFields(projectId, {
          scene_index: idx,
          all_scene_edits: allSceneEdits,
          text_model: textModel,
        });
        onChange(idx, "scene_description", resp.scene_description);
        onChange(idx, "start_frame_prompt", resp.start_frame_prompt);
        onChange(idx, "end_frame_prompt", resp.end_frame_prompt);
        onChange(idx, "video_motion_prompt", resp.video_motion_prompt);
        onChange(idx, "transition_notes", resp.transition_notes);
      } catch (err) {
        setEmptySlotGenError(err instanceof Error ? err.message : "Generation failed");
      } finally {
        setEmptySlotGenerating(false);
      }
    }

    return (
      <div className="rounded-lg border border-dashed border-gray-700 bg-gray-900/50 p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-emerald-400">
            Scene {idx + 1} — New
          </span>
          <div className="flex items-center gap-2">
            {projectId && onGenerateScene && (
              <button
                type="button"
                onClick={handleGenerateScene}
                disabled={emptySlotGenerating}
                className="flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
              >
                {emptySlotGenerating && <Spinner className="h-2.5 w-2.5 text-indigo-400" />}
                {emptySlotGenerating ? "Generating..." : "Generate Scene"}
              </button>
            )}
            {projectId && (
              <button
                type="button"
                onClick={handleGenerateTextOnly}
                disabled={emptySlotGenerating}
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-gray-500 hover:text-gray-400 hover:bg-gray-800/50 transition-colors disabled:opacity-50"
                title="Generate text fields only (no keyframes or clip)"
              >
                Text Only
              </button>
            )}
            {canRemove && (
              <button
                type="button"
                onClick={() => onRemove(idx)}
                className="text-gray-600 hover:text-red-400 transition-colors"
                title="Remove scene"
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
        {emptySlotGenError && (
          <div className="mt-1 rounded border border-red-800 bg-red-900/50 px-2 py-1 text-[11px] text-red-300">
            {emptySlotGenError}
          </div>
        )}
        <EditableField
          label="Description"
          value={getValue("scene_description", scene.description)}
          originalValue=""
          onChange={(v) => onChange(idx, "scene_description", v)}
          onClear={() => onChange(idx, "scene_description", "")}
        />
        <EditableField
          label="Start Frame Prompt"
          value={getValue("start_frame_prompt", scene.start_frame_prompt)}
          originalValue=""
          onChange={(v) => onChange(idx, "start_frame_prompt", v)}
          onClear={() => onChange(idx, "start_frame_prompt", "")}
        />
        <EditableField
          label="End Frame Prompt"
          value={getValue("end_frame_prompt", scene.end_frame_prompt)}
          originalValue=""
          onChange={(v) => onChange(idx, "end_frame_prompt", v)}
          onClear={() => onChange(idx, "end_frame_prompt", "")}
        />
        <EditableField
          label="Motion Prompt"
          value={getValue("video_motion_prompt", scene.video_motion_prompt)}
          originalValue=""
          onChange={(v) => onChange(idx, "video_motion_prompt", v)}
          onClear={() => onChange(idx, "video_motion_prompt", "")}
        />
        <EditableField
          label="Transition Notes"
          value={getValue("transition_notes", scene.transition_notes)}
          originalValue=""
          onChange={(v) => onChange(idx, "transition_notes", v)}
          onClear={() => onChange(idx, "transition_notes", "")}
        />
      </div>
    );
  }

  // Removed scene
  if (removed) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-gray-900/50 p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-500 line-through">
            Scene {idx + 1} &mdash; {scene.description?.slice(0, 60)}
            {(scene.description?.length ?? 0) > 60 ? "..." : ""}
          </span>
          <button
            type="button"
            onClick={() => onRestore(idx)}
            className="rounded px-2 py-0.5 text-[11px] font-medium text-blue-400 hover:bg-blue-500/10 transition-colors"
          >
            Restore
          </button>
        </div>
      </div>
    );
  }

  const hasEdits = Object.keys(edits).length > 0;

  return (
    <div
      className={clsx(
        "rounded-lg border bg-gray-900 p-3",
        hasEdits ? "border-amber-700" : "border-gray-800",
      )}
    >
      {/* Header */}
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400">
          Scene {idx + 1}
        </span>
        <div className="flex items-center gap-2">
          {hasEdits && (
            <span className="text-[10px] font-medium uppercase text-amber-400">Modified</span>
          )}
          {canRemove && (
            <button
              type="button"
              onClick={() => onRemove(idx)}
              className="text-gray-600 hover:text-red-400 transition-colors"
              title="Remove scene"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Action error */}
      {actionError && (
        <div className="mb-2 rounded border border-red-800 bg-red-900/50 px-2 py-1 text-[11px] text-red-300">
          {actionError}
        </div>
      )}

      {/* Generating assets banner (shown after scene created, before assets arrive) */}
      {isGeneratingAssets && (
        <div className="mb-2 flex items-center gap-1.5 rounded border border-indigo-800 bg-indigo-900/50 px-2 py-1 text-[11px] text-indigo-300">
          <Spinner className="h-2.5 w-2.5 text-indigo-400" />
          Generating keyframes & video clip...
        </div>
      )}

      {/* Regen queued feedback with animated indicator */}
      {regenQueued && (
        <div className="mb-2 flex items-center justify-between rounded border border-green-800 bg-green-900/50 px-2 py-1 text-[11px] text-green-300">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
            Regenerating {regenQueued.replace(/_/g, " ")}...
          </span>
          <button onClick={() => { setRegenQueued(null); setPollingTarget(null); }} className="text-green-400 hover:text-green-300 ml-1">
            &times;
          </button>
        </div>
      )}

      {/* Hidden file inputs */}
      <input ref={startKfInput} type="file" accept="image/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUploadKeyframe("start", f); e.target.value = ""; }} />
      <input ref={endKfInput} type="file" accept="image/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUploadKeyframe("end", f); e.target.value = ""; }} />
      <input ref={clipInput} type="file" accept="video/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUploadClip(f); e.target.value = ""; }} />

      {/* Keyframe previews with staleness badges + actions */}
      <div className="mb-2 flex gap-2">
        {/* Start keyframe */}
        <div className="relative flex-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
              Start KF
            </span>
            <StalenessBadge staleness={scene.start_keyframe_staleness} />
          </div>
          {scene.start_keyframe_url ? (
            <img
              src={scene.start_keyframe_url}
              alt={`Scene ${idx + 1} start`}
              onClick={() => { setLightboxUrl(scene.start_keyframe_url!); setLightboxLabel(`Scene ${idx + 1} — Start Keyframe`); }}
              className={clsx(
                "mt-0.5 w-full rounded border cursor-pointer hover:brightness-110 transition",
                scene.start_keyframe_staleness === "stale" ? "border-amber-700" : "border-gray-700",
              )}
              loading="lazy"
            />
          ) : (
            <div className="mt-0.5 flex h-16 items-center justify-center rounded border border-dashed border-gray-700 bg-gray-950 text-[10px] text-gray-600">
              No keyframe
            </div>
          )}
          {projectId && (
            <div className="mt-1">
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => handleRegenerate("start_keyframe")}
                  disabled={regenerating === "start_keyframe" || pollingTarget === "start_keyframe"}
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
                >
                  {(regenerating === "start_keyframe" || regenQueued === "start_keyframe") && <Spinner className="h-2.5 w-2.5 text-indigo-400" />}
                  {regenerating === "start_keyframe" ? "Sending..." : regenQueued === "start_keyframe" ? "Generating..." : "Regen"}
                </button>
                <button
                  type="button"
                  onClick={() => setShowContextFor(showContextFor === "start_keyframe" ? null : "start_keyframe")}
                  className={clsx(
                    "rounded px-1 py-0.5 text-[10px] transition-colors",
                    showContextFor === "start_keyframe" ? "text-indigo-300 bg-indigo-900/30" : "text-gray-500 hover:text-gray-400",
                  )}
                  title="Add direction"
                >
                  +
                </button>
                <button
                  type="button"
                  onClick={() => startKfInput.current?.click()}
                  className="rounded px-1.5 py-0.5 text-[10px] text-gray-400 hover:bg-gray-800 transition-colors"
                >
                  Upload
                </button>
                {scene.has_start_keyframe && (
                  <button
                    type="button"
                    onClick={() => handleDeleteKeyframe("start")}
                    className="rounded px-1.5 py-0.5 text-[10px] text-red-400 hover:bg-red-900/30 transition-colors"
                  >
                    Del
                  </button>
                )}
              </div>
              {showContextFor === "start_keyframe" && (
                <input
                  type="text"
                  placeholder="Extra direction (optional)..."
                  value={extraContext["start_keyframe"] ?? ""}
                  onChange={(e) => setExtraContext((prev) => ({ ...prev, start_keyframe: e.target.value }))}
                  onClick={(e) => e.stopPropagation()}
                  className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-0.5 text-[10px] text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              )}
            </div>
          )}
        </div>
        {/* End keyframe */}
        <div className="relative flex-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
              End KF
            </span>
            <StalenessBadge staleness={scene.end_keyframe_staleness} />
          </div>
          {scene.end_keyframe_url ? (
            <img
              src={scene.end_keyframe_url}
              alt={`Scene ${idx + 1} end`}
              onClick={() => { setLightboxUrl(scene.end_keyframe_url!); setLightboxLabel(`Scene ${idx + 1} — End Keyframe`); }}
              className={clsx(
                "mt-0.5 w-full rounded border cursor-pointer hover:brightness-110 transition",
                scene.end_keyframe_staleness === "stale" ? "border-amber-700" : "border-gray-700",
              )}
              loading="lazy"
            />
          ) : (
            <div className="mt-0.5 flex h-16 items-center justify-center rounded border border-dashed border-gray-700 bg-gray-950 text-[10px] text-gray-600">
              No keyframe
            </div>
          )}
          {projectId && (
            <div className="mt-1">
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => handleRegenerate("end_keyframe")}
                  disabled={regenerating === "end_keyframe" || pollingTarget === "end_keyframe"}
                  className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
                >
                  {(regenerating === "end_keyframe" || regenQueued === "end_keyframe") && <Spinner className="h-2.5 w-2.5 text-indigo-400" />}
                  {regenerating === "end_keyframe" ? "Sending..." : regenQueued === "end_keyframe" ? "Generating..." : "Regen"}
                </button>
                <button
                  type="button"
                  onClick={() => setShowContextFor(showContextFor === "end_keyframe" ? null : "end_keyframe")}
                  className={clsx(
                    "rounded px-1 py-0.5 text-[10px] transition-colors",
                    showContextFor === "end_keyframe" ? "text-indigo-300 bg-indigo-900/30" : "text-gray-500 hover:text-gray-400",
                  )}
                  title="Add direction"
                >
                  +
                </button>
                <button
                  type="button"
                  onClick={() => endKfInput.current?.click()}
                  className="rounded px-1.5 py-0.5 text-[10px] text-gray-400 hover:bg-gray-800 transition-colors"
                >
                  Upload
                </button>
                {scene.has_end_keyframe && (
                  <button
                    type="button"
                    onClick={() => handleDeleteKeyframe("end")}
                    className="rounded px-1.5 py-0.5 text-[10px] text-red-400 hover:bg-red-900/30 transition-colors"
                  >
                    Del
                  </button>
                )}
              </div>
              {showContextFor === "end_keyframe" && (
                <input
                  type="text"
                  placeholder="Extra direction (optional)..."
                  value={extraContext["end_keyframe"] ?? ""}
                  onChange={(e) => setExtraContext((prev) => ({ ...prev, end_keyframe: e.target.value }))}
                  onClick={(e) => e.stopPropagation()}
                  className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-0.5 text-[10px] text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Clip preview */}
      {scene.clip_url && (
        <div className="mb-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
              Clip
            </span>
            <StalenessBadge staleness={scene.clip_staleness} />
          </div>
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            src={scene.clip_url}
            className={clsx(
              "mt-0.5 w-full rounded border",
              scene.clip_staleness === "stale" ? "border-amber-700" : "border-gray-700",
            )}
            controls
            preload="metadata"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      {/* Clip staleness + actions */}
      {scene.has_clip && scene.clip_staleness === "stale" && !scene.clip_url && (
        <div className="mb-2 flex items-center gap-1 rounded border border-amber-800/50 bg-amber-950/30 px-2 py-1">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-400" />
          <span className="text-[11px] text-amber-400">Video clip is stale — prompt has changed</span>
        </div>
      )}
      {projectId && (
        <div className="mb-2">
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => handleRegenerate("video_clip")}
              disabled={regenerating === "video_clip" || pollingTarget === "video_clip"}
              className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
            >
              {(regenerating === "video_clip" || regenQueued === "video_clip") && <Spinner className="h-2.5 w-2.5 text-indigo-400" />}
              {regenerating === "video_clip" ? "Sending..." : regenQueued === "video_clip" ? "Generating..." : "Regen Clip"}
            </button>
            <button
              type="button"
              onClick={() => setShowContextFor(showContextFor === "video_clip" ? null : "video_clip")}
              className={clsx(
                "rounded px-1 py-0.5 text-[10px] transition-colors",
                showContextFor === "video_clip" ? "text-indigo-300 bg-indigo-900/30" : "text-gray-500 hover:text-gray-400",
              )}
              title="Add direction"
            >
              +
            </button>
            <button
              type="button"
              onClick={() => clipInput.current?.click()}
              className="rounded px-1.5 py-0.5 text-[10px] text-gray-400 hover:bg-gray-800 transition-colors"
            >
              Upload Clip
            </button>
            {scene.has_clip && (
              <button
                type="button"
                onClick={handleDeleteClip}
                className="rounded px-1.5 py-0.5 text-[10px] text-red-400 hover:bg-red-900/30 transition-colors"
              >
                Del Clip
              </button>
            )}
          </div>
          {showContextFor === "video_clip" && (
            <input
              type="text"
              placeholder="Extra direction (optional)..."
              value={extraContext["video_clip"] ?? ""}
              onChange={(e) => setExtraContext((prev) => ({ ...prev, video_clip: e.target.value }))}
              onClick={(e) => e.stopPropagation()}
              className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-0.5 text-[10px] text-gray-300 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          )}
        </div>
      )}

      {/* Editable fields */}
      <EditableField
        label="Description"
        value={getValue("scene_description", scene.description)}
        originalValue={getOriginal("scene_description", scene.description)}
        onChange={(v) => onChange(idx, "scene_description", v)}
        onClear={() => onChange(idx, "scene_description", "")}
        onRegen={projectId ? () => handleTextRegen("scene_description") : undefined}
        regenerating={regenTextFields.has("scene_description")}
        showContext={showTextContextFor === "scene_description"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "scene_description" ? null : "scene_description")}
        contextValue={textExtraContext["scene_description"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, scene_description: v }))}
        onOpenEditor={() => setEditorField("scene_description")}
      />
      <EditableField
        label="Start Frame Prompt"
        value={getValue("start_frame_prompt", scene.start_frame_prompt)}
        originalValue={getOriginal("start_frame_prompt", scene.start_frame_prompt)}
        onChange={(v) => onChange(idx, "start_frame_prompt", v)}
        onClear={() => onChange(idx, "start_frame_prompt", "")}
        staleness={scene.start_keyframe_staleness}
        onRegen={projectId ? () => handleTextRegen("start_frame_prompt") : undefined}
        regenerating={regenTextFields.has("start_frame_prompt")}
        showContext={showTextContextFor === "start_frame_prompt"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "start_frame_prompt" ? null : "start_frame_prompt")}
        contextValue={textExtraContext["start_frame_prompt"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, start_frame_prompt: v }))}
        onOpenEditor={() => setEditorField("start_frame_prompt")}
      />
      <EditableField
        label="End Frame Prompt"
        value={getValue("end_frame_prompt", scene.end_frame_prompt)}
        originalValue={getOriginal("end_frame_prompt", scene.end_frame_prompt)}
        onChange={(v) => onChange(idx, "end_frame_prompt", v)}
        onClear={() => onChange(idx, "end_frame_prompt", "")}
        staleness={scene.end_keyframe_staleness}
        onRegen={projectId ? () => handleTextRegen("end_frame_prompt") : undefined}
        regenerating={regenTextFields.has("end_frame_prompt")}
        showContext={showTextContextFor === "end_frame_prompt"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "end_frame_prompt" ? null : "end_frame_prompt")}
        contextValue={textExtraContext["end_frame_prompt"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, end_frame_prompt: v }))}
        onOpenEditor={() => setEditorField("end_frame_prompt")}
      />
      <EditableField
        label="Motion Prompt"
        value={getValue("video_motion_prompt", scene.video_motion_prompt)}
        originalValue={getOriginal("video_motion_prompt", scene.video_motion_prompt)}
        onChange={(v) => onChange(idx, "video_motion_prompt", v)}
        onClear={() => onChange(idx, "video_motion_prompt", "")}
        staleness={scene.clip_staleness}
        onRegen={projectId ? () => handleTextRegen("video_motion_prompt") : undefined}
        regenerating={regenTextFields.has("video_motion_prompt")}
        showContext={showTextContextFor === "video_motion_prompt"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "video_motion_prompt" ? null : "video_motion_prompt")}
        contextValue={textExtraContext["video_motion_prompt"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, video_motion_prompt: v }))}
        onOpenEditor={() => setEditorField("video_motion_prompt")}
      />
      <EditableField
        label="Transition Notes"
        value={getValue("transition_notes", scene.transition_notes)}
        originalValue={getOriginal("transition_notes", scene.transition_notes)}
        onChange={(v) => onChange(idx, "transition_notes", v)}
        onClear={() => onChange(idx, "transition_notes", "")}
        onRegen={projectId ? () => handleTextRegen("transition_notes") : undefined}
        regenerating={regenTextFields.has("transition_notes")}
        showContext={showTextContextFor === "transition_notes"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "transition_notes" ? null : "transition_notes")}
        contextValue={textExtraContext["transition_notes"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, transition_notes: v }))}
        onOpenEditor={() => setEditorField("transition_notes")}
      />

      {/* Prompt details (collapsible) */}
      {(scene.rewritten_keyframe_prompt || scene.rewritten_video_prompt ||
        scene.start_keyframe_prompt_used || scene.clip_prompt_used) && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setPromptDetailsOpen(!promptDetailsOpen)}
            className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-400 transition-colors"
          >
            <svg
              className={clsx("h-3 w-3 transition-transform", promptDetailsOpen && "rotate-90")}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            Prompt details
          </button>
          {promptDetailsOpen && (
            <div className="mt-1 space-y-1.5 rounded border border-gray-800 bg-gray-950 p-2 text-[11px]">
              {scene.rewritten_keyframe_prompt && (
                <div>
                  <span className="font-medium text-gray-500">Rewritten KF:</span>
                  <p className="text-gray-400">{scene.rewritten_keyframe_prompt}</p>
                </div>
              )}
              {scene.rewritten_video_prompt && (
                <div>
                  <span className="font-medium text-gray-500">Rewritten Video:</span>
                  <p className="text-gray-400">{scene.rewritten_video_prompt}</p>
                </div>
              )}
              {scene.start_keyframe_prompt_used && (
                <div>
                  <span className="font-medium text-gray-500">Start KF sent:</span>
                  <p className="text-gray-400">{scene.start_keyframe_prompt_used}</p>
                </div>
              )}
              {scene.end_keyframe_prompt_used && (
                <div>
                  <span className="font-medium text-gray-500">End KF sent:</span>
                  <p className="text-gray-400">{scene.end_keyframe_prompt_used}</p>
                </div>
              )}
              {scene.clip_prompt_used && (
                <div>
                  <span className="font-medium text-gray-500">Video sent:</span>
                  <p className="text-gray-400">{scene.clip_prompt_used}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Lightbox modal */}
      {lightboxUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setLightboxUrl(null)}
        >
          <div
            className="relative max-h-[90vh] max-w-[90vw]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm font-medium text-gray-300">{lightboxLabel}</span>
              <button
                onClick={() => setLightboxUrl(null)}
                className="rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition-colors"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <img
              src={lightboxUrl}
              alt={lightboxLabel}
              className="max-h-[85vh] max-w-[90vw] rounded-lg border border-gray-700 object-contain"
            />
          </div>
        </div>
      )}

      {/* Markdown editor modal */}
      {editorField && (() => {
        const fieldLabelMap: Record<string, string> = {
          scene_description: "Description",
          start_frame_prompt: "Start Frame Prompt",
          end_frame_prompt: "End Frame Prompt",
          video_motion_prompt: "Motion Prompt",
          transition_notes: "Transition Notes",
        };
        const fieldOriginalMap: Record<string, string | null | undefined> = {
          scene_description: scene.description,
          start_frame_prompt: scene.start_frame_prompt,
          end_frame_prompt: scene.end_frame_prompt,
          video_motion_prompt: scene.video_motion_prompt,
          transition_notes: scene.transition_notes,
        };
        return (
          <MarkdownEditorModal
            label={`Scene ${idx + 1} — ${fieldLabelMap[editorField] ?? editorField}`}
            value={getValue(editorField, fieldOriginalMap[editorField])}
            onChange={(v) => onChange(idx, editorField, v)}
            onClose={() => setEditorField(null)}
            onRegen={projectId ? () => handleTextRegen(editorField) : undefined}
            regenerating={regenTextFields.has(editorField)}
            extraContext={textExtraContext[editorField] ?? ""}
            onExtraContextChange={(v) => setTextExtraContext(prev => ({ ...prev, [editorField]: v }))}
          />
        );
      })()}
    </div>
  );
}
