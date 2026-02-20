import { useState, useMemo, useEffect } from "react";
import clsx from "clsx";
import { getEnabledModels, resumeProject } from "../api/client.ts";
import type { ProjectDetail } from "../api/types.ts";
import type { EnabledModelsResponse } from "../api/types.ts";
import {
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
} from "../lib/constants.ts";

/**
 * Which settings are relevant when continuing FROM a given stage boundary:
 * - storyboard → next is keyframes: needs image_model, vision_model
 * - keyframes  → next is video_gen: needs video_model, audio, clip_duration
 * - video      → next is stitching: nothing to configure
 */
type StageTarget = "keyframes" | "video" | "completion";

interface ContinuePanelProps {
  detail: ProjectDetail;
  /** run_through value for the next leg (null = all/completion) */
  nextRunThrough: string | null;
  onContinued: (projectId: string) => void;
  onCancel: () => void;
}

export function ContinuePanel({ detail, nextRunThrough, onContinued, onCancel }: ContinuePanelProps) {
  const currentStage = detail.run_through; // what we staged at
  const target: StageTarget = nextRunThrough === "keyframes"
    ? "keyframes"
    : nextRunThrough === "video"
      ? "video"
      : "completion";

  // Determine which config sections to show
  const needsImageConfig = currentStage === "storyboard";
  const needsVideoConfig = currentStage === "storyboard" || currentStage === "keyframes";

  // State for model selections, seeded from project
  const [imageModel, setImageModel] = useState(detail.image_model ?? IMAGE_MODELS[0].id);
  const [visionModel, setVisionModel] = useState(detail.vision_model ?? "");
  const [videoModel, setVideoModel] = useState(detail.video_model ?? VIDEO_MODELS[0].id);
  const [enableAudio, setEnableAudio] = useState(detail.audio_enabled ?? true);
  const [clipDuration, setClipDuration] = useState(detail.clip_duration ?? 6);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelSettings, setModelSettings] = useState<EnabledModelsResponse | null>(null);

  // Fetch enabled models
  useEffect(() => {
    getEnabledModels()
      .then(setModelSettings)
      .catch(() => {});
  }, []);

  // Filter models based on settings
  const filteredTextModels = useMemo(() => {
    if (!modelSettings?.enabled_text_models) return TEXT_MODELS;
    const enabled = new Set(modelSettings.enabled_text_models);
    return TEXT_MODELS.filter((m) => enabled.has(m.id));
  }, [modelSettings]);

  const filteredImageModels = useMemo(() => {
    if (!modelSettings?.enabled_image_models) return IMAGE_MODELS;
    const enabled = new Set(modelSettings.enabled_image_models);
    return IMAGE_MODELS.filter((m) => enabled.has(m.id));
  }, [modelSettings]);

  const filteredVideoModels = useMemo(() => {
    if (!modelSettings?.enabled_video_models) return VIDEO_MODELS;
    const enabled = new Set(modelSettings.enabled_video_models);
    return VIDEO_MODELS.filter((m) => enabled.has(m.id));
  }, [modelSettings]);

  // Merge Ollama vision models
  const allVisionModels = useMemo(() => {
    const base = filteredTextModels;
    const ollamaVision = (modelSettings?.ollama_models ?? [])
      .filter((m) => m.enabled && m.vision)
      .map((m) => ({ id: m.id, label: `${m.label} (Ollama)`, costPerCall: 0 }));
    return [...base, ...ollamaVision];
  }, [filteredTextModels, modelSettings]);

  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel) ?? VIDEO_MODELS[0];
  const allowedDurations = selectedVideoModel.allowedDurations;

  // Snap clip duration when video model changes
  useEffect(() => {
    if (!allowedDurations.includes(clipDuration)) {
      const nearest = allowedDurations.reduce((a, b) =>
        Math.abs(b - clipDuration) < Math.abs(a - clipDuration) ? b : a
      );
      setClipDuration(nearest);
    }
  }, [allowedDurations, clipDuration]);

  function handleVideoModelChange(id: string) {
    setVideoModel(id);
    const model = VIDEO_MODELS.find((m) => m.id === id) ?? VIDEO_MODELS[0];
    if (!model.allowedDurations.includes(clipDuration)) {
      const nearest = model.allowedDurations.reduce((a, b) =>
        Math.abs(b - clipDuration) < Math.abs(a - clipDuration) ? b : a
      );
      setClipDuration(nearest);
    }
    setEnableAudio(model.supportsAudio);
  }

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        run_through: nextRunThrough ?? "all",
      };
      if (needsImageConfig) {
        body.image_model = imageModel;
        body.vision_model = visionModel || undefined;
      }
      if (needsVideoConfig) {
        body.video_model = videoModel;
        body.audio_enabled = enableAudio && selectedVideoModel.supportsAudio;
        body.clip_duration = clipDuration;
      }
      await resumeProject(detail.project_id, body as Parameters<typeof resumeProject>[1]);
      onContinued(detail.project_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Continue failed");
    } finally {
      setSubmitting(false);
    }
  }

  // Target label for the button
  const targetLabel = target === "keyframes"
    ? "Continue to Keyframes"
    : target === "video"
      ? "Continue to Video Gen"
      : "Continue to Completion";

  // Nothing to configure for stitching-only
  if (!needsImageConfig && !needsVideoConfig) {
    return null; // caller should just call resume directly
  }

  return (
    <div className="rounded-lg border border-cyan-800 bg-cyan-950/20 p-4 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-cyan-200">
          Configure Next Stage
        </h3>
        <button
          onClick={onCancel}
          className="text-gray-500 hover:text-gray-300 transition-colors text-sm"
        >
          Cancel
        </button>
      </div>

      {/* Image Model — when continuing from storyboard */}
      {needsImageConfig && (
        <>
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">
              Image Model
            </label>
            <div className="flex flex-wrap gap-2">
              {filteredImageModels.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setImageModel(m.id)}
                  className={clsx(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                    imageModel === m.id
                      ? "border-blue-500 bg-blue-500/20 text-blue-300"
                      : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">
              Vision Model
              <span className="ml-2 text-xs text-gray-500 font-normal">
                For image analysis & scoring
              </span>
            </label>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setVisionModel("")}
                className={clsx(
                  "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                  visionModel === ""
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                )}
              >
                Same as Text
              </button>
              {allVisionModels.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setVisionModel(m.id)}
                  className={clsx(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                    visionModel === m.id
                      ? "border-blue-500 bg-blue-500/20 text-blue-300"
                      : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Video Model + Audio + Clip Duration — when continuing to video_gen */}
      {needsVideoConfig && (
        <>
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">
              Video Model
            </label>
            <div className="flex flex-wrap gap-2">
              {filteredVideoModels.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => handleVideoModelChange(m.id)}
                  className={clsx(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                    videoModel === m.id
                      ? "border-blue-500 bg-blue-500/20 text-blue-300"
                      : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">
              Scene Length
            </label>
            <div className="flex gap-2">
              {allowedDurations.map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => setClipDuration(d)}
                  className={clsx(
                    "rounded-md border px-4 py-1.5 text-sm font-medium transition-colors",
                    clipDuration === d
                      ? "border-blue-500 bg-blue-500/20 text-blue-300"
                      : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                  )}
                >
                  {d}s
                </button>
              ))}
            </div>
          </div>

          {selectedVideoModel.supportsAudio && (
            <div className="flex items-center gap-3">
              <label className="text-sm font-medium text-gray-300">Audio</label>
              <button
                type="button"
                onClick={() => setEnableAudio(!enableAudio)}
                className={clsx(
                  "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                  enableAudio ? "bg-blue-600" : "bg-gray-700",
                )}
              >
                <span
                  className={clsx(
                    "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                    enableAudio ? "translate-x-6" : "translate-x-1",
                  )}
                />
              </button>
              <span className="text-sm text-gray-400">
                {enableAudio ? "Enabled" : "Disabled"}
              </span>
            </div>
          )}
        </>
      )}

      {error && (
        <div className="rounded-md bg-red-900/50 border border-red-800 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={handleSubmit}
          disabled={submitting}
          className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 transition-colors disabled:opacity-50"
        >
          {submitting ? "Starting..." : targetLabel}
        </button>
        <button
          onClick={onCancel}
          className="rounded-lg border border-gray-700 px-4 py-2 text-sm font-medium text-gray-300 hover:border-gray-600 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
