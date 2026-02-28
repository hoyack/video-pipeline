import clsx from "clsx";
import { useState, useRef, useEffect, useCallback } from "react";
import type { ShotDetail } from "../api/types.ts";
import {
  regenerateShot,
  regenerateShotText,
  generateShotFields,
  uploadKeyframe,
  uploadClip,
  deleteShotKeyframe,
  deleteShotClip,
} from "../api/client.ts";
import { usePolling } from "../hooks/usePolling.ts";
import { CopyButton } from "./CopyButton.tsx";
import { MarkdownEditorModal } from "./MarkdownEditorModal.tsx";

interface ShotEditorCardProps {
  shot: ShotDetail;
  edits: Record<string, string>;
  onChange: (shotIndex: number, field: string, value: string) => void;
  onRemove: (shotIndex: number) => void;
  removed: boolean;
  onRestore: (shotIndex: number) => void;
  canRemove: boolean;
  sceneId?: string;
  onAssetChanged?: () => void;
  /** Called when a regen fires — passes the pre-regen head_sha for revert-on-cancel */
  onRegenStarted?: (headSha: string | null) => void;
  /** Current text model selection from edit mode (used for text regen) */
  textModel?: string;
  /** Current video model selection from edit mode (used for clip regen) */
  videoModel?: string;
  /** Current image model selection from edit mode (used for keyframe regen) */
  imageModel?: string;
  /** All shot edits across the scene — for generate-shot-fields neighbor context */
  allShotEdits?: Record<number, Record<string, string>>;
  /** Current scene prompt from edit form (may differ from saved DB value) */
  prompt?: string;
  /** Called to generate a complete new shot (text + assets) from an empty slot */
  onGenerateShot?: (shotIndex: number) => Promise<void>;
  /** True when background asset generation is in progress for this shot */
  isGeneratingAssets?: boolean;
  /** When true, parent WS handles refreshes — suppress per-shot polling */
  wsConnected?: boolean;
  /** Whether this card is expanded (accordion) */
  expanded?: boolean;
  /** Toggle expand/collapse */
  onToggleExpand?: () => void;
  /** 1-based display position (from drag order) */
  displayIndex?: number;
  /** Drag handle listeners from useSortable */
  dragHandleListeners?: Record<string, Function>;
  /** Drag handle attributes from useSortable */
  dragHandleAttributes?: Record<string, unknown>;
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

function AutoResizeTextarea(
  props: React.TextareaHTMLAttributes<HTMLTextAreaElement>,
) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const resize = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.max(el.scrollHeight, 40)}px`;
  }, []);
  useEffect(resize, [props.value, resize]);
  return <textarea ref={ref} rows={2} {...props} onInput={resize} />;
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
        <AutoResizeTextarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onClick={(e) => e.stopPropagation()}
          className={clsx(
            "w-full rounded border bg-gray-950 px-2 py-1 text-xs leading-relaxed text-gray-300 focus:outline-none focus:ring-1 resize-none overflow-hidden",
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

export function ShotEditorCard({
  shot,
  edits,
  onChange,
  onRemove,
  removed,
  onRestore,
  canRemove,
  sceneId,
  onAssetChanged,
  onRegenStarted,
  textModel,
  videoModel,
  imageModel,
  allShotEdits,
  prompt,
  onGenerateShot,
  isGeneratingAssets,
  wsConnected,
  expanded,
  onToggleExpand,
  displayIndex,
  dragHandleListeners,
  dragHandleAttributes,
}: ShotEditorCardProps) {
  const idx = shot.shot_index;
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
  const MAX_POLLS = 120; // 120 * 5s = 600s (10 min)

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

  // Derived: pipeline generation_status from DB (set during bulk regen)
  const genStatus = shot.generation_status ?? null;

  // Map target -> shot URL field
  function getUrlForTarget(target: string): string | null | undefined {
    if (target === "start_keyframe") return shot.start_keyframe_url;
    if (target === "end_keyframe") return shot.end_keyframe_url;
    if (target === "video_clip") return shot.clip_url;
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

  usePolling(handlePollTick, POLL_INTERVAL, !!pollingTarget && !wsConnected);

  // Detect completion: URL changed from baseline
  useEffect(() => {
    if (!pollingTarget) return;
    const currentUrl = getUrlForTarget(pollingTarget);
    if (currentUrl && currentUrl !== baselineUrl.current) {
      setPollingTarget(null);
      setRegenQueued(null);
      pollCount.current = 0;
    }
  }, [pollingTarget, shot.start_keyframe_url, shot.end_keyframe_url, shot.clip_url]);

  async function handleRegenerate(target: string) {
    if (!sceneId) return;
    setRegenerating(target);
    setRegenQueued(null);
    setActionError(null);

    // Build prompt_overrides from extra context
    const context = extraContext[target]?.trim();
    let promptOverrides: Record<string, string> | undefined;
    if (context) {
      let basePrompt: string;
      if (target === "video_clip") {
        basePrompt = edits["video_motion_prompt"] ?? shot.rewritten_video_prompt ?? shot.video_motion_prompt ?? "";
      } else if (target === "start_keyframe") {
        basePrompt = edits["start_frame_prompt"] ?? shot.rewritten_keyframe_prompt ?? shot.start_frame_prompt ?? "";
      } else {
        basePrompt = edits["end_frame_prompt"] ?? shot.rewritten_keyframe_prompt ?? shot.end_frame_prompt ?? "";
      }
      promptOverrides = { [target]: basePrompt + "\n\n[Additional direction: " + context + "]" };
    }

    try {
      // Snapshot baseline URL before firing
      baselineUrl.current = getUrlForTarget(target) ?? null;
      pollCount.current = 0;

      const resp = await regenerateShot(sceneId, idx, {
        targets: [target],
        skip_checkpoint: true,
        ...(promptOverrides && { prompt_overrides: promptOverrides }),
        video_model: videoModel,
        image_model: imageModel,
        shot_edits: Object.keys(edits).length > 0 ? edits : undefined,
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
    if (!sceneId) return;
    setActionError(null);
    try {
      await uploadKeyframe(sceneId, idx, position, file);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  async function handleUploadClip(file: File) {
    if (!sceneId) return;
    setActionError(null);
    try {
      await uploadClip(sceneId, idx, file);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  async function handleDeleteKeyframe(position: string) {
    if (!sceneId) return;
    setActionError(null);
    try {
      await deleteShotKeyframe(sceneId, idx, position);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function handleDeleteClip() {
    if (!sceneId) return;
    setActionError(null);
    try {
      await deleteShotClip(sceneId, idx);
      onAssetChanged?.();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  async function handleTextRegen(field: string) {
    if (!sceneId) return;
    setRegenTextFields((prev) => new Set(prev).add(field));
    setActionError(null);
    try {
      const resp = await regenerateShotText(sceneId, idx, {
        field,
        extra_context: textExtraContext[field]?.trim() || undefined,
        text_model: textModel,
        shot_edits: Object.keys(edits).length > 0 ? edits : undefined,
        prompt: prompt || undefined,
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

  // Treat any shot with no content (no text, no assets) as "new" — not just synthetic empty slots
  const isNewShot = shot.is_empty_slot || (
    !shot.description &&
    !shot.start_frame_prompt &&
    !shot.end_frame_prompt &&
    !shot.video_motion_prompt &&
    !shot.has_start_keyframe &&
    !shot.has_end_keyframe &&
    !shot.has_clip
  );

  async function handleGenerateFullShot() {
    if (!onGenerateShot) return;
    setEmptySlotGenerating(true);
    setEmptySlotGenError(null);
    try {
      await onGenerateShot(idx);
    } catch (err) {
      setEmptySlotGenError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setEmptySlotGenerating(false);
    }
  }

  // Removed shot
  if (removed) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-gray-900/50 p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-500 line-through">
            Shot {idx + 1} &mdash; {shot.description?.slice(0, 60)}
            {(shot.description?.length ?? 0) > 60 ? "..." : ""}
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
        "rounded-lg border p-3",
        isNewShot
          ? "border-dashed border-gray-700 bg-gray-900/50"
          : hasEdits ? "border-amber-700 bg-gray-900" : "border-gray-800 bg-gray-900",
      )}
    >
      {/* Header — clickable to toggle expand/collapse */}
      <div
        className={clsx("flex items-center justify-between", onToggleExpand && "cursor-pointer select-none")}
        onClick={onToggleExpand}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          {dragHandleListeners && dragHandleAttributes && (
            <button
              type="button"
              className="flex-shrink-0 cursor-grab active:cursor-grabbing text-gray-600 hover:text-gray-400 transition-colors touch-none"
              onClick={(e) => e.stopPropagation()}
              onPointerDown={(e) => e.stopPropagation()}
              {...dragHandleAttributes}
              {...dragHandleListeners}
            >
              <svg className="h-4 w-4" viewBox="0 0 16 16" fill="currentColor">
                <circle cx="5" cy="3" r="1.5" /><circle cx="11" cy="3" r="1.5" />
                <circle cx="5" cy="8" r="1.5" /><circle cx="11" cy="8" r="1.5" />
                <circle cx="5" cy="13" r="1.5" /><circle cx="11" cy="13" r="1.5" />
              </svg>
            </button>
          )}
          {onToggleExpand && (
            <svg
              className={clsx("h-3.5 w-3.5 flex-shrink-0 text-gray-500 transition-transform", expanded && "rotate-180")}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          )}
          <span className={clsx("text-xs font-medium flex-shrink-0", isNewShot ? "text-emerald-400" : "text-gray-400")}>
            Shot {displayIndex ?? (idx + 1)}{isNewShot ? " — New" : ""}
          </span>
          {expanded === false && (
            <>
              <span className="text-xs text-gray-500 truncate min-w-0">
                {getValue("shot_description", shot.description).slice(0, 80) || "No description"}
              </span>
              {/* Compact asset dots */}
              <span className="flex items-center gap-1 flex-shrink-0 ml-1">
                <span className={clsx("inline-block h-1.5 w-1.5 rounded-full", shot.has_start_keyframe ? "bg-blue-400" : "bg-gray-700")} title="Start KF" />
                <span className={clsx("inline-block h-1.5 w-1.5 rounded-full", shot.has_end_keyframe ? "bg-indigo-400" : "bg-gray-700")} title="End KF" />
                <span className={clsx("inline-block h-1.5 w-1.5 rounded-full", shot.has_clip ? "bg-green-400" : "bg-gray-700")} title="Clip" />
              </span>
              {(shot.start_keyframe_staleness === "stale" || shot.end_keyframe_staleness === "stale" || shot.clip_staleness === "stale") && (
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-400 flex-shrink-0" title="Has stale assets" />
              )}
            </>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {isNewShot && sceneId && onGenerateShot && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); handleGenerateFullShot(); }}
              disabled={emptySlotGenerating}
              className="flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium text-indigo-400 hover:bg-indigo-900/30 transition-colors disabled:opacity-50"
            >
              {emptySlotGenerating && <Spinner className="h-2.5 w-2.5 text-indigo-400" />}
              {emptySlotGenerating ? "Generating..." : "Generate Shot"}
            </button>
          )}
          {hasEdits && (
            <span className="text-[10px] font-medium uppercase text-amber-400">Modified</span>
          )}
          {canRemove && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onRemove(idx); }}
              className="text-gray-600 hover:text-red-400 transition-colors"
              title="Remove shot"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Hidden file inputs (always rendered so refs work) */}
      <input ref={startKfInput} type="file" accept="image/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUploadKeyframe("start", f); e.target.value = ""; }} />
      <input ref={endKfInput} type="file" accept="image/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUploadKeyframe("end", f); e.target.value = ""; }} />
      <input ref={clipInput} type="file" accept="video/*" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUploadClip(f); e.target.value = ""; }} />

      {expanded !== false && (<>
      {/* Action error */}
      {actionError && (
        <div className="mt-2 mb-2 rounded border border-red-800 bg-red-900/50 px-2 py-1 text-[11px] text-red-300">
          {actionError}
        </div>
      )}
      {emptySlotGenError && (
        <div className="mt-2 mb-2 rounded border border-red-800 bg-red-900/50 px-2 py-1 text-[11px] text-red-300">
          {emptySlotGenError}
        </div>
      )}

      {/* Generating assets banner (shown after shot created, before assets arrive) */}
      {isGeneratingAssets && (
        <div className="mt-2 mb-2 flex items-center gap-1.5 rounded border border-indigo-800 bg-indigo-900/50 px-2 py-1 text-[11px] text-indigo-300">
          <Spinner className="h-2.5 w-2.5 text-indigo-400" />
          Generating keyframes & video clip...
        </div>
      )}

      {/* Regen queued feedback with animated indicator */}
      {regenQueued && (
        <div className="mt-2 mb-2 flex items-center justify-between rounded border border-green-800 bg-green-900/50 px-2 py-1 text-[11px] text-green-300">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
            Regenerating {regenQueued.replace(/_/g, " ")}...
          </span>
          <button onClick={() => { setRegenQueued(null); setPollingTarget(null); }} className="text-green-400 hover:text-green-300 ml-1">
            &times;
          </button>
        </div>
      )}

      {/* Keyframe previews with staleness badges + actions */}
      <div className="mb-2 flex gap-2">
        {/* Start keyframe */}
        <div className="relative flex-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
              Start KF
            </span>
            <StalenessBadge staleness={shot.start_keyframe_staleness} />
          </div>
          <div className="relative mt-0.5">
            {shot.start_keyframe_url ? (
              <img
                src={shot.start_keyframe_url}
                alt={`Shot ${idx + 1} start`}
                onClick={() => { setLightboxUrl(shot.start_keyframe_url!); setLightboxLabel(`Shot ${idx + 1} — Start Keyframe`); }}
                className={clsx(
                  "w-full rounded border cursor-pointer hover:brightness-110 transition",
                  shot.start_keyframe_staleness === "stale" ? "border-amber-700" : "border-gray-700",
                )}
                loading="lazy"
              />
            ) : (
              <div className="flex h-16 items-center justify-center rounded border border-dashed border-gray-700 bg-gray-950 text-[10px] text-gray-600">
                {genStatus === "generating_start_kf" ? (
                  <span className="flex items-center gap-1 text-indigo-400"><Spinner className="h-3 w-3" /> Generating...</span>
                ) : "No keyframe"}
              </div>
            )}
            {genStatus === "generating_start_kf" && shot.start_keyframe_url && (
              <div className="absolute inset-0 flex items-center justify-center rounded bg-gray-950/60">
                <span className="flex items-center gap-1 text-[10px] text-indigo-400"><Spinner className="h-3 w-3" /> Regenerating...</span>
              </div>
            )}
          </div>
          {sceneId && (
            <div className="mt-1">
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => handleRegenerate("start_keyframe")}
                  disabled={regenerating === "start_keyframe" || pollingTarget === "start_keyframe" || !imageModel}
                  className={clsx(
                    "flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-colors disabled:opacity-50",
                    !imageModel ? "text-gray-600 cursor-not-allowed" : "text-indigo-400 hover:bg-indigo-900/30",
                  )}
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
                {shot.has_start_keyframe && (
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
            <StalenessBadge staleness={shot.end_keyframe_staleness} />
          </div>
          <div className="relative mt-0.5">
            {shot.end_keyframe_url ? (
              <img
                src={shot.end_keyframe_url}
                alt={`Shot ${idx + 1} end`}
                onClick={() => { setLightboxUrl(shot.end_keyframe_url!); setLightboxLabel(`Shot ${idx + 1} — End Keyframe`); }}
                className={clsx(
                  "w-full rounded border cursor-pointer hover:brightness-110 transition",
                  shot.end_keyframe_staleness === "stale" ? "border-amber-700" : "border-gray-700",
                )}
                loading="lazy"
              />
            ) : (
              <div className="flex h-16 items-center justify-center rounded border border-dashed border-gray-700 bg-gray-950 text-[10px] text-gray-600">
                {genStatus === "generating_end_kf" ? (
                  <span className="flex items-center gap-1 text-indigo-400"><Spinner className="h-3 w-3" /> Generating...</span>
                ) : "No keyframe"}
              </div>
            )}
            {genStatus === "generating_end_kf" && shot.end_keyframe_url && (
              <div className="absolute inset-0 flex items-center justify-center rounded bg-gray-950/60">
                <span className="flex items-center gap-1 text-[10px] text-indigo-400"><Spinner className="h-3 w-3" /> Regenerating...</span>
              </div>
            )}
          </div>
          {sceneId && (
            <div className="mt-1">
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => handleRegenerate("end_keyframe")}
                  disabled={regenerating === "end_keyframe" || pollingTarget === "end_keyframe" || !imageModel}
                  className={clsx(
                    "flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-colors disabled:opacity-50",
                    !imageModel ? "text-gray-600 cursor-not-allowed" : "text-indigo-400 hover:bg-indigo-900/30",
                  )}
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
                {shot.has_end_keyframe && (
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
      <div className="mb-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
            Clip
          </span>
          <StalenessBadge staleness={shot.clip_staleness} />
        </div>
        <div className="relative mt-0.5">
          {shot.clip_url ? (
            /* eslint-disable-next-line jsx-a11y/media-has-caption */
            <video
              src={shot.clip_url}
              className={clsx(
                "w-full rounded border",
                shot.clip_staleness === "stale" ? "border-amber-700" : "border-gray-700",
              )}
              controls
              preload="metadata"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <div className="flex h-16 items-center justify-center rounded border border-dashed border-gray-700 bg-gray-950 text-[10px] text-gray-600">
              {genStatus === "generating_clip" ? (
                <span className="flex items-center gap-1 text-indigo-400"><Spinner className="h-3 w-3" /> Generating clip...</span>
              ) : "No clip"}
            </div>
          )}
          {genStatus === "generating_clip" && shot.clip_url && (
            <div className="absolute inset-0 flex items-center justify-center rounded bg-gray-950/60">
              <span className="flex items-center gap-1 text-[10px] text-indigo-400"><Spinner className="h-3 w-3" /> Regenerating clip...</span>
            </div>
          )}
        </div>
      </div>

      {/* Clip staleness + actions */}
      {shot.has_clip && shot.clip_staleness === "stale" && !shot.clip_url && (
        <div className="mb-2 flex items-center gap-1 rounded border border-amber-800/50 bg-amber-950/30 px-2 py-1">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-400" />
          <span className="text-[11px] text-amber-400">Video clip is stale — prompt has changed</span>
        </div>
      )}
      {sceneId && (
        <div className="mb-2">
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => handleRegenerate("video_clip")}
              disabled={regenerating === "video_clip" || pollingTarget === "video_clip" || !videoModel}
              className={clsx(
                "flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition-colors disabled:opacity-50",
                !videoModel ? "text-gray-600 cursor-not-allowed" : "text-indigo-400 hover:bg-indigo-900/30",
              )}
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
            {shot.has_clip && (
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
        value={getValue("shot_description", shot.description)}
        originalValue={getOriginal("shot_description", shot.description)}
        onChange={(v) => onChange(idx, "shot_description", v)}
        onClear={() => onChange(idx, "shot_description", "")}
        onRegen={sceneId && textModel ? () => handleTextRegen("shot_description") : undefined}
        regenerating={regenTextFields.has("shot_description") || genStatus === "generating_text"}
        showContext={showTextContextFor === "shot_description"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "shot_description" ? null : "shot_description")}
        contextValue={textExtraContext["shot_description"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, shot_description: v }))}
        onOpenEditor={() => setEditorField("shot_description")}
      />
      <EditableField
        label="Start Frame Prompt"
        value={getValue("start_frame_prompt", shot.start_frame_prompt)}
        originalValue={getOriginal("start_frame_prompt", shot.start_frame_prompt)}
        onChange={(v) => onChange(idx, "start_frame_prompt", v)}
        onClear={() => onChange(idx, "start_frame_prompt", "")}
        staleness={shot.start_keyframe_staleness}
        onRegen={sceneId && textModel ? () => handleTextRegen("start_frame_prompt") : undefined}
        regenerating={regenTextFields.has("start_frame_prompt") || genStatus === "generating_text"}
        showContext={showTextContextFor === "start_frame_prompt"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "start_frame_prompt" ? null : "start_frame_prompt")}
        contextValue={textExtraContext["start_frame_prompt"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, start_frame_prompt: v }))}
        onOpenEditor={() => setEditorField("start_frame_prompt")}
      />
      <EditableField
        label="End Frame Prompt"
        value={getValue("end_frame_prompt", shot.end_frame_prompt)}
        originalValue={getOriginal("end_frame_prompt", shot.end_frame_prompt)}
        onChange={(v) => onChange(idx, "end_frame_prompt", v)}
        onClear={() => onChange(idx, "end_frame_prompt", "")}
        staleness={shot.end_keyframe_staleness}
        onRegen={sceneId && textModel ? () => handleTextRegen("end_frame_prompt") : undefined}
        regenerating={regenTextFields.has("end_frame_prompt") || genStatus === "generating_text"}
        showContext={showTextContextFor === "end_frame_prompt"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "end_frame_prompt" ? null : "end_frame_prompt")}
        contextValue={textExtraContext["end_frame_prompt"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, end_frame_prompt: v }))}
        onOpenEditor={() => setEditorField("end_frame_prompt")}
      />
      <EditableField
        label="Motion Prompt"
        value={getValue("video_motion_prompt", shot.video_motion_prompt)}
        originalValue={getOriginal("video_motion_prompt", shot.video_motion_prompt)}
        onChange={(v) => onChange(idx, "video_motion_prompt", v)}
        onClear={() => onChange(idx, "video_motion_prompt", "")}
        staleness={shot.clip_staleness}
        onRegen={sceneId && textModel ? () => handleTextRegen("video_motion_prompt") : undefined}
        regenerating={regenTextFields.has("video_motion_prompt") || genStatus === "generating_text"}
        showContext={showTextContextFor === "video_motion_prompt"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "video_motion_prompt" ? null : "video_motion_prompt")}
        contextValue={textExtraContext["video_motion_prompt"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, video_motion_prompt: v }))}
        onOpenEditor={() => setEditorField("video_motion_prompt")}
      />
      <EditableField
        label="Transition Notes"
        value={getValue("transition_notes", shot.transition_notes)}
        originalValue={getOriginal("transition_notes", shot.transition_notes)}
        onChange={(v) => onChange(idx, "transition_notes", v)}
        onClear={() => onChange(idx, "transition_notes", "")}
        onRegen={sceneId && textModel ? () => handleTextRegen("transition_notes") : undefined}
        regenerating={regenTextFields.has("transition_notes") || genStatus === "generating_text"}
        showContext={showTextContextFor === "transition_notes"}
        onToggleContext={() => setShowTextContextFor((prev) => prev === "transition_notes" ? null : "transition_notes")}
        contextValue={textExtraContext["transition_notes"] ?? ""}
        onContextChange={(v) => setTextExtraContext((prev) => ({ ...prev, transition_notes: v }))}
        onOpenEditor={() => setEditorField("transition_notes")}
      />

      {/* Prompt details (collapsible) */}
      {(shot.rewritten_keyframe_prompt || shot.rewritten_video_prompt ||
        shot.start_keyframe_prompt_used || shot.clip_prompt_used) && (
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
              {shot.rewritten_keyframe_prompt && (
                <div>
                  <span className="font-medium text-gray-500">Rewritten KF:</span>
                  <p className="text-gray-400">{shot.rewritten_keyframe_prompt}</p>
                </div>
              )}
              {shot.rewritten_video_prompt && (
                <div>
                  <span className="font-medium text-gray-500">Rewritten Video:</span>
                  <p className="text-gray-400">{shot.rewritten_video_prompt}</p>
                </div>
              )}
              {shot.start_keyframe_prompt_used && (
                <div>
                  <span className="font-medium text-gray-500">Start KF sent:</span>
                  <p className="text-gray-400">{shot.start_keyframe_prompt_used}</p>
                </div>
              )}
              {shot.end_keyframe_prompt_used && (
                <div>
                  <span className="font-medium text-gray-500">End KF sent:</span>
                  <p className="text-gray-400">{shot.end_keyframe_prompt_used}</p>
                </div>
              )}
              {shot.clip_prompt_used && (
                <div>
                  <span className="font-medium text-gray-500">Video sent:</span>
                  <p className="text-gray-400">{shot.clip_prompt_used}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      </>)}

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
          shot_description: "Description",
          start_frame_prompt: "Start Frame Prompt",
          end_frame_prompt: "End Frame Prompt",
          video_motion_prompt: "Motion Prompt",
          transition_notes: "Transition Notes",
        };
        const fieldOriginalMap: Record<string, string | null | undefined> = {
          shot_description: shot.description,
          start_frame_prompt: shot.start_frame_prompt,
          end_frame_prompt: shot.end_frame_prompt,
          video_motion_prompt: shot.video_motion_prompt,
          transition_notes: shot.transition_notes,
        };
        return (
          <MarkdownEditorModal
            label={`Shot ${idx + 1} — ${fieldLabelMap[editorField] ?? editorField}`}
            value={getValue(editorField, fieldOriginalMap[editorField])}
            onChange={(v) => onChange(idx, editorField, v)}
            onClose={() => setEditorField(null)}
            onRegen={sceneId && textModel ? () => handleTextRegen(editorField) : undefined}
            regenerating={regenTextFields.has(editorField)}
            extraContext={textExtraContext[editorField] ?? ""}
            onExtraContextChange={(v) => setTextExtraContext(prev => ({ ...prev, [editorField]: v }))}
          />
        );
      })()}
    </div>
  );
}
