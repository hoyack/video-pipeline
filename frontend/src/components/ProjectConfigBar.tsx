import { useState, useEffect, useMemo } from "react";
import clsx from "clsx";
import { getEnabledModels } from "../api/client.ts";
import type { EnabledModelsResponse } from "../api/types.ts";
import { ManifestSelector } from "./ManifestSelector.tsx";
import {
  STYLE_OPTIONS,
  ASPECT_RATIOS,
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
} from "../lib/constants.ts";

interface ProjectConfigBarProps {
  title: string; setTitle: (s: string) => void;
  prompt: string; setPrompt: (s: string) => void;
  style: string; setStyle: (s: string) => void;
  aspectRatio: string; setAspectRatio: (s: string) => void;
  textModel: string; setTextModel: (s: string) => void;
  imageModel: string; setImageModel: (s: string) => void;
  videoModel: string; setVideoModel: (s: string) => void;
  visionModel: string; setVisionModel: (s: string) => void;
  enableAudio: boolean; setEnableAudio: (b: boolean) => void;
  manifestId: string | null; setManifestId: (s: string | null) => void;
  qualityMode: boolean; setQualityMode: (b: boolean) => void;
  candidateCount: number; setCandidateCount: (n: number) => void;
  expanded: boolean; setExpanded: (b: boolean) => void;
  disabled?: boolean;
}

export function ProjectConfigBar(props: ProjectConfigBarProps) {
  const {
    title, setTitle,
    prompt, setPrompt,
    style, setStyle,
    aspectRatio, setAspectRatio,
    textModel, setTextModel,
    imageModel, setImageModel,
    videoModel, setVideoModel,
    visionModel, setVisionModel,
    enableAudio, setEnableAudio,
    manifestId, setManifestId,
    qualityMode, setQualityMode,
    candidateCount, setCandidateCount,
    expanded, setExpanded,
    disabled,
  } = props;

  const [modelSettings, setModelSettings] = useState<EnabledModelsResponse | null>(null);

  useEffect(() => {
    getEnabledModels()
      .then((ms) => setModelSettings(ms))
      .catch(() => {});
  }, []);

  // Filtered models based on settings
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

  // Merge Ollama text models
  const allTextModels = useMemo(() => {
    const base = filteredTextModels;
    const ollamaText = (modelSettings?.ollama_models ?? [])
      .filter((m) => m.enabled)
      .map((m) => ({ id: m.id, label: `${m.label} (Ollama)`, costPerCall: 0 }));
    return [...base, ...ollamaText];
  }, [filteredTextModels, modelSettings]);

  // Vision models: Gemini text models + Ollama vision models
  const allVisionModels = useMemo(() => {
    const base = filteredTextModels;
    const ollamaVision = (modelSettings?.ollama_models ?? [])
      .filter((m) => m.enabled && m.vision)
      .map((m) => ({ id: m.id, label: `${m.label} (Ollama)`, costPerCall: 0 }));
    return [...base, ...ollamaVision];
  }, [filteredTextModels, modelSettings]);

  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel) ?? VIDEO_MODELS[0];

  function handleVideoModelChange(id: string) {
    setVideoModel(id);
    const model = VIDEO_MODELS.find((m) => m.id === id) ?? VIDEO_MODELS[0];
    setEnableAudio(model.supportsAudio);
  }

  // Collapsed summary
  if (!expanded) {
    return (
      <div
        className={clsx(
          "rounded-lg border bg-gray-900 px-4 py-2 cursor-pointer transition-colors hover:bg-gray-800/80",
          disabled ? "border-gray-800 opacity-70" : "border-gray-700",
        )}
        onClick={() => !disabled && setExpanded(true)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-semibold text-white truncate">
              {title || "Untitled Project"}
            </span>
            <span className="text-xs text-gray-500 truncate">
              {style} | {textModel.split("-").slice(0, 2).join("-")} | {videoModel.split("-").slice(0, 2).join("-")}
            </span>
          </div>
          <svg className="h-4 w-4 text-gray-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className={clsx("rounded-lg border bg-gray-900 p-4 space-y-4", disabled ? "border-gray-800 opacity-70 pointer-events-none" : "border-gray-700")}>
      {/* Header with collapse button */}
      <div className="flex items-center justify-between">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Untitled Project"
          className="flex-1 bg-transparent text-xl font-bold text-white placeholder-gray-600 border-none outline-none"
        />
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="ml-2 p-1 rounded hover:bg-gray-800 transition-colors"
        >
          <svg className="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
          </svg>
        </button>
      </div>

      {/* Prompt */}
      <div>
        <label className="mb-1 block text-xs font-medium text-gray-400">Prompt</label>
        <textarea
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the video you want to create..."
          className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {/* Style + Aspect Ratio row */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-400">Style</label>
          <div className="flex flex-wrap gap-1">
            {STYLE_OPTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setStyle(s)}
                className={clsx(
                  "rounded border px-2 py-0.5 text-[11px] font-medium capitalize transition-colors",
                  style === s
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                )}
              >
                {s.replace("_", " ")}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-400">Aspect Ratio</label>
          <div className="flex gap-1">
            {ASPECT_RATIOS.map((ar) => (
              <button
                key={ar}
                type="button"
                onClick={() => setAspectRatio(ar)}
                className={clsx(
                  "rounded border px-3 py-0.5 text-[11px] font-medium transition-colors",
                  aspectRatio === ar
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                )}
              >
                {ar}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Models row */}
      <div className="space-y-3">
        {/* Text Model */}
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-400">Text Model</label>
          <div className="flex flex-wrap gap-1">
            {allTextModels.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setTextModel(m.id)}
                className={clsx(
                  "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
                  textModel === m.id
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Vision Model */}
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-400">
            Vision Model
            <span className="ml-1 text-[10px] text-gray-600 font-normal">For analysis & scoring</span>
          </label>
          <div className="flex flex-wrap gap-1">
            <button
              type="button"
              onClick={() => setVisionModel("")}
              className={clsx(
                "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
                visionModel === ""
                  ? "border-blue-500 bg-blue-500/20 text-blue-300"
                  : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
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
                  "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
                  visionModel === m.id
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Image Model */}
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-400">Image Model</label>
          <div className="flex flex-wrap gap-1">
            {filteredImageModels.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setImageModel(m.id)}
                className={clsx(
                  "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
                  imageModel === m.id
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Video Model */}
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-400">Video Model</label>
          <div className="flex flex-wrap gap-1">
            {filteredVideoModels.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => handleVideoModelChange(m.id)}
                className={clsx(
                  "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
                  videoModel === m.id
                    ? "border-blue-500 bg-blue-500/20 text-blue-300"
                    : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Audio Toggle */}
        {selectedVideoModel.supportsAudio && (
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium text-gray-400">Audio</label>
            <button
              type="button"
              onClick={() => setEnableAudio(!enableAudio)}
              className={clsx(
                "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
                enableAudio ? "bg-blue-600" : "bg-gray-700",
              )}
            >
              <span
                className={clsx(
                  "inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform",
                  enableAudio ? "translate-x-4.5" : "translate-x-0.5",
                )}
              />
            </button>
            <span className="text-xs text-gray-500">{enableAudio ? "On" : "Off"}</span>
          </div>
        )}
      </div>

      {/* Manifest Selector */}
      <div>
        <label className="mb-1 block text-xs font-medium text-gray-400">Asset Manifest</label>
        <ManifestSelector
          selectedManifestId={manifestId}
          onManifestSelect={setManifestId}
        />
      </div>

      {/* Quality Mode */}
      <div className="rounded border border-gray-800 bg-gray-950 p-3">
        <div className="flex items-center justify-between">
          <div>
            <label className="text-xs font-medium text-gray-300">Quality Mode</label>
            <p className="text-[10px] text-gray-600 mt-0.5">
              Generate multiple candidates per scene
            </p>
          </div>
          <button
            type="button"
            onClick={() => setQualityMode(!qualityMode)}
            className={clsx(
              "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
              qualityMode ? "bg-amber-600" : "bg-gray-700",
            )}
          >
            <span
              className={clsx(
                "inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform",
                qualityMode ? "translate-x-4.5" : "translate-x-0.5",
              )}
            />
          </button>
        </div>
        {qualityMode && (
          <div className="mt-2 flex items-center gap-2">
            <span className="text-[10px] text-gray-500">Candidates:</span>
            {[2, 3, 4].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setCandidateCount(n)}
                className={clsx(
                  "px-2 py-0.5 rounded text-[10px] font-medium transition-colors",
                  candidateCount === n
                    ? "bg-amber-600 text-white"
                    : "bg-gray-800 text-gray-500 hover:bg-gray-700",
                )}
              >
                {n}x
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
