/** Pipeline stages in execution order */
export const PIPELINE_STAGES = [
  "pending",
  "storyboarding",
  "keyframing",
  "video_gen",
  "stitching",
  "complete",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];

/** Human-readable labels for pipeline stages */
export const STAGE_LABELS: Record<string, string> = {
  pending: "Pending",
  storyboarding: "Storyboard",
  keyframing: "Keyframes",
  video_gen: "Video Gen",
  stitching: "Stitch",
  complete: "Complete",
};

/** Available style presets */
export const STYLE_OPTIONS = [
  "cinematic",
  "anime",
  "watercolor",
  "photorealistic",
  "3d_render",
  "pixel_art",
] as const;

/** Available aspect ratios (Veo supports 16:9 and 9:16 only) */
export const ASPECT_RATIOS = ["16:9", "9:16"] as const;

/** Default clip duration */
export const DURATION_DEFAULT = 6;

/** Total duration range */
export const TOTAL_DURATION_MIN = 10;
export const TOTAL_DURATION_MAX = 300;
export const TOTAL_DURATION_DEFAULT = 15;
export const TOTAL_DURATION_STEP = 5;

// ---------------------------------------------------------------------------
// Model catalog types
// ---------------------------------------------------------------------------

export interface TextModelOption {
  id: string;
  label: string;
  costPerCall: number;
}

export interface ImageModelOption {
  id: string;
  label: string;
  costPerImage: number;
}

export interface VideoModelOption {
  id: string;
  label: string;
  costPerSecond: number;
  costPerSecondAudio: number;
  supportsAudio: boolean;
  /** Allowed clip durations in seconds for image-to-video */
  allowedDurations: number[];
}

// ---------------------------------------------------------------------------
// Model catalogs
// ---------------------------------------------------------------------------

export const TEXT_MODELS: TextModelOption[] = [
  { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash", costPerCall: 0.006 },
  { id: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite", costPerCall: 0.001 },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro", costPerCall: 0.023 },
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash", costPerCall: 0.007 },
  { id: "gemini-3-pro-preview", label: "Gemini 3 Pro", costPerCall: 0.028 },
];

export const IMAGE_MODELS: ImageModelOption[] = [
  { id: "gemini-2.5-flash-image", label: "Nano Banana", costPerImage: 0.04 },
  { id: "gemini-3-pro-image-preview", label: "Nano Banana Pro", costPerImage: 0.13 },
  { id: "qwen-fast", label: "Qwen Fast", costPerImage: 0.00 },
];

export const VIDEO_MODELS: VideoModelOption[] = [
  { id: "veo-2.0-generate-001", label: "Veo 2", costPerSecond: 0.35, costPerSecondAudio: 0.35, supportsAudio: false, allowedDurations: [5, 6, 7, 8] },
  { id: "veo-3.0-generate-001", label: "Veo 3", costPerSecond: 0.40, costPerSecondAudio: 0.40, supportsAudio: true, allowedDurations: [4, 6, 8] },
  { id: "veo-3.0-fast-generate-001", label: "Veo 3 Fast", costPerSecond: 0.15, costPerSecondAudio: 0.15, supportsAudio: true, allowedDurations: [4, 6, 8] },
  { id: "veo-3.1-generate-preview", label: "Veo 3.1", costPerSecond: 0.40, costPerSecondAudio: 0.40, supportsAudio: true, allowedDurations: [4, 6, 8] },
  { id: "veo-3.1-generate-001", label: "Veo 3.1 GA", costPerSecond: 0.40, costPerSecondAudio: 0.40, supportsAudio: true, allowedDurations: [4, 6, 8] },
  { id: "veo-3.1-fast-generate-preview", label: "Veo 3.1 Fast", costPerSecond: 0.10, costPerSecondAudio: 0.15, supportsAudio: true, allowedDurations: [4, 6, 8] },
  { id: "veo-3.1-fast-generate-001", label: "Veo 3.1 Fast GA", costPerSecond: 0.10, costPerSecondAudio: 0.15, supportsAudio: true, allowedDurations: [4, 6, 8] },
  { id: "wan-2.2-ref-i2v", label: "Wan 2.2 Ref", costPerSecond: 0, costPerSecondAudio: 0, supportsAudio: false, allowedDurations: [5] },
  { id: "wan-2.2-i2v", label: "Wan 2.2", costPerSecond: 0, costPerSecondAudio: 0, supportsAudio: false, allowedDurations: [5] },
];

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

export function estimateCost(
  totalDuration: number,
  clipDuration: number,
  textModelId: string,
  imageModelId: string,
  videoModelId: string,
  enableAudio: boolean = false,
): number {
  const sceneCount = Math.ceil(totalDuration / clipDuration);
  const videoModel = VIDEO_MODELS.find((m) => m.id === videoModelId);
  const imageModel = IMAGE_MODELS.find((m) => m.id === imageModelId);
  const textModel = TEXT_MODELS.find((m) => m.id === textModelId);

  const videoCostPerSecond = enableAudio && videoModel?.supportsAudio
    ? (videoModel?.costPerSecondAudio ?? 0.40)
    : (videoModel?.costPerSecond ?? 0.40);
  const videoCost = sceneCount * clipDuration * videoCostPerSecond;
  const imageCost = (sceneCount + 1) * (imageModel?.costPerImage ?? 0.04);
  const textCost = textModel?.costPerCall ?? 0.01;

  return videoCost + imageCost + textCost;
}

/** Estimate quality mode cost multiplier */
export function qualityModeCostMultiplier(candidateCount: number): number {
  return candidateCount; // Linear scaling — N candidates = Nx video gen cost
}

/** Terminal statuses — pipeline is no longer running */
export const TERMINAL_STATUSES = new Set(["complete", "failed", "stopped"]);

/** Slow stages where polling can back off */
export const SLOW_STAGES = new Set(["video_gen"]);
