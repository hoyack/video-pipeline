import { useState } from "react";
import clsx from "clsx";
import { generateVideo } from "../api/client.ts";
import {
  STYLE_OPTIONS,
  ASPECT_RATIOS,
  DURATION_DEFAULT,
  TOTAL_DURATION_MIN,
  TOTAL_DURATION_MAX,
  TOTAL_DURATION_DEFAULT,
  TOTAL_DURATION_STEP,
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
  estimateCost,
} from "../lib/constants.ts";

interface GenerateFormProps {
  onGenerated: (projectId: string) => void;
}

export function GenerateForm({ onGenerated }: GenerateFormProps) {
  const [prompt, setPrompt] = useState("");
  const [style, setStyle] = useState<string>(STYLE_OPTIONS[0]);
  const [aspectRatio, setAspectRatio] = useState<string>(ASPECT_RATIOS[0]);
  const [clipDuration, setClipDuration] = useState(DURATION_DEFAULT);
  const [totalDuration, setTotalDuration] = useState(TOTAL_DURATION_DEFAULT);
  const [textModel, setTextModel] = useState(TEXT_MODELS[0].id);
  const [imageModel, setImageModel] = useState(IMAGE_MODELS[0].id);
  const [videoModel, setVideoModel] = useState(VIDEO_MODELS[0].id);
  const [enableAudio, setEnableAudio] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel) ?? VIDEO_MODELS[0];
  const allowedDurations = selectedVideoModel.allowedDurations;
  const sceneCount = Math.ceil(totalDuration / clipDuration);
  const audioActive = enableAudio && selectedVideoModel.supportsAudio;
  const cost = estimateCost(totalDuration, clipDuration, textModel, imageModel, videoModel, audioActive);

  function handleVideoModelChange(id: string) {
    setVideoModel(id);
    const model = VIDEO_MODELS.find((m) => m.id === id) ?? VIDEO_MODELS[0];
    if (!model.allowedDurations.includes(clipDuration)) {
      // Snap to nearest allowed duration
      const nearest = model.allowedDurations.reduce((a, b) =>
        Math.abs(b - clipDuration) < Math.abs(a - clipDuration) ? b : a
      );
      setClipDuration(nearest);
    }
    // Auto-toggle audio based on model support
    setEnableAudio(model.supportsAudio);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || submitting) return;

    setSubmitting(true);
    setError(null);

    try {
      const res = await generateVideo({
        prompt: prompt.trim(),
        style,
        aspect_ratio: aspectRatio,
        clip_duration: clipDuration,
        total_duration: totalDuration,
        text_model: textModel,
        image_model: imageModel,
        video_model: videoModel,
        enable_audio: audioActive,
      });
      onGenerated(res.project_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setSubmitting(false);
    }
  }

  const videoCostPerSecond = audioActive
    ? selectedVideoModel.costPerSecondAudio
    : selectedVideoModel.costPerSecond;

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="mb-1 text-2xl font-bold text-white">Generate Video</h1>
        <p className="text-sm text-gray-400">
          Describe the video you want to create.
        </p>
      </div>

      {/* Prompt */}
      <div>
        <label htmlFor="prompt" className="mb-1 block text-sm font-medium text-gray-300">
          Prompt
        </label>
        <textarea
          id="prompt"
          rows={4}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="A sweeping aerial shot of a bioluminescent forest at night..."
          className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {/* Style */}
      <div>
        <label className="mb-2 block text-sm font-medium text-gray-300">
          Style
        </label>
        <div className="flex flex-wrap gap-2">
          {STYLE_OPTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStyle(s)}
              className={clsx(
                "rounded-md border px-3 py-1.5 text-sm font-medium capitalize transition-colors",
                style === s
                  ? "border-blue-500 bg-blue-500/20 text-blue-300"
                  : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
              )}
            >
              {s.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Aspect Ratio */}
      <div>
        <label className="mb-2 block text-sm font-medium text-gray-300">
          Aspect Ratio
        </label>
        <div className="flex gap-2">
          {ASPECT_RATIOS.map((ar) => (
            <button
              key={ar}
              type="button"
              onClick={() => setAspectRatio(ar)}
              className={clsx(
                "rounded-md border px-4 py-1.5 text-sm font-medium transition-colors",
                aspectRatio === ar
                  ? "border-blue-500 bg-blue-500/20 text-blue-300"
                  : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
              )}
            >
              {ar}
            </button>
          ))}
        </div>
      </div>

      {/* Scene Length (clip duration) */}
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

      {/* Total Duration */}
      <div>
        <label htmlFor="totalDuration" className="mb-2 block text-sm font-medium text-gray-300">
          Total Duration: {totalDuration}s ({sceneCount} scenes)
        </label>
        <input
          id="totalDuration"
          type="range"
          min={TOTAL_DURATION_MIN}
          max={TOTAL_DURATION_MAX}
          step={TOTAL_DURATION_STEP}
          value={totalDuration}
          onChange={(e) => setTotalDuration(Number(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="mt-1 flex justify-between text-xs text-gray-600">
          <span>{TOTAL_DURATION_MIN}s</span>
          <span>{TOTAL_DURATION_MAX}s</span>
        </div>
      </div>

      {/* Model Selection */}
      <div className="space-y-4">
        {/* Text Model */}
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">
            Text Model
          </label>
          <div className="flex flex-wrap gap-2">
            {TEXT_MODELS.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setTextModel(m.id)}
                className={clsx(
                  "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                  textModel === m.id
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Image Model */}
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">
            Image Model
          </label>
          <div className="flex flex-wrap gap-2">
            {IMAGE_MODELS.map((m) => (
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

        {/* Video Model */}
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">
            Video Model
          </label>
          <div className="flex flex-wrap gap-2">
            {VIDEO_MODELS.map((m) => (
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

        {/* Audio Toggle — only shown for models that support audio */}
        {selectedVideoModel.supportsAudio && (
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">
              Audio
            </label>
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
            <span className="ml-2 text-sm text-gray-400">
              {enableAudio ? "Enabled" : "Disabled"}
            </span>
          </div>
        )}
      </div>

      {/* Cost Estimate */}
      <div className="rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-300">
        <div>Estimated cost: ~${cost.toFixed(2)}</div>
        <div className="mt-1 text-xs text-gray-500">
          {sceneCount} scenes &middot; ${videoCostPerSecond.toFixed(2)}/s video{audioActive ? " (with audio)" : ""} &middot; ${(IMAGE_MODELS.find((m) => m.id === imageModel)?.costPerImage ?? 0).toFixed(2)}/img
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-md bg-red-900/50 border border-red-800 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={!prompt.trim() || submitting}
        className={clsx(
          "w-full rounded-lg py-2.5 text-sm font-semibold transition-colors",
          prompt.trim() && !submitting
            ? "bg-blue-600 text-white hover:bg-blue-500"
            : "bg-gray-800 text-gray-500 cursor-not-allowed",
        )}
      >
        {submitting ? "Starting..." : "Generate Video"}
      </button>
    </form>
  );
}
