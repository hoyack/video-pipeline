import { useState } from "react";
import clsx from "clsx";
import { generateVideo } from "../api/client.ts";
import {
  STYLE_OPTIONS,
  ASPECT_RATIOS,
  DURATION_MIN,
  DURATION_MAX,
  DURATION_DEFAULT,
} from "../lib/constants.ts";

interface GenerateFormProps {
  onGenerated: (projectId: string) => void;
}

export function GenerateForm({ onGenerated }: GenerateFormProps) {
  const [prompt, setPrompt] = useState("");
  const [style, setStyle] = useState<string>(STYLE_OPTIONS[0]);
  const [aspectRatio, setAspectRatio] = useState<string>(ASPECT_RATIOS[0]);
  const [clipDuration, setClipDuration] = useState(DURATION_DEFAULT);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      });
      onGenerated(res.project_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setSubmitting(false);
    }
  }

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

      {/* Duration */}
      <div>
        <label htmlFor="duration" className="mb-2 block text-sm font-medium text-gray-300">
          Clip Duration: {clipDuration}s
        </label>
        <input
          id="duration"
          type="range"
          min={DURATION_MIN}
          max={DURATION_MAX}
          step={1}
          value={clipDuration}
          onChange={(e) => setClipDuration(Number(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="mt-1 flex justify-between text-xs text-gray-600">
          <span>{DURATION_MIN}s</span>
          <span>{DURATION_MAX}s</span>
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
