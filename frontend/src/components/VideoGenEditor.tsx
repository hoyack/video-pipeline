import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import clsx from "clsx";
import type { View } from "./Layout.tsx";
import type { ProjectDetail, SceneDetail } from "../api/types.ts";
import {
  createDraftProject,
  startGeneration,
  getProjectDetail,
  stopProject,
  uploadFinalVideo,
  generateNewScene,
  getEnabledModels,
} from "../api/client.ts";
import { usePolling } from "../hooks/usePolling.ts";
import { SceneEditorCard } from "./SceneEditorCard.tsx";
import { ProjectConfigBar } from "./ProjectConfigBar.tsx";
import { GenerateThroughSlider, sliderToRunThrough } from "./GenerateThroughSlider.tsx";
import {
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
  estimatePartialCost,
} from "../lib/constants.ts";
import type { EnabledModelsResponse } from "../api/types.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type EditorMode = "drafting" | "running" | "editing";

interface VideoGenEditorProps {
  projectId?: string | null;
  onNavigate: (view: View, projectId?: string) => void;
}

// Running statuses where the pipeline is actively processing
const RUNNING_STATUSES = new Set([
  "pending",
  "storyboarding",
  "keyframing",
  "video_gen",
  "stitching",
]);

function getEditorMode(projectId: string | null, detail: ProjectDetail | null): EditorMode {
  if (!projectId || !detail) return "drafting";
  if (RUNNING_STATUSES.has(detail.status)) return "running";
  return "editing"; // draft, stopped, staged, failed, complete
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function VideoGenEditor({ projectId: initialProjectId, onNavigate }: VideoGenEditorProps) {
  // Core state
  const [projectId, setProjectId] = useState<string | null>(initialProjectId ?? null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);

  // Config state
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [style, setStyle] = useState("cinematic");
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [clipDuration, setClipDuration] = useState(6);
  const [sceneCount, setSceneCount] = useState(3);
  const [textModel, setTextModel] = useState("gemini-2.5-flash");
  const [imageModel, setImageModel] = useState("gemini-2.5-flash-image");
  const [videoModel, setVideoModel] = useState("veo-3.1-fast-generate-001");
  const [visionModel, setVisionModel] = useState("");
  const [enableAudio, setEnableAudio] = useState(true);
  const [manifestId, setManifestId] = useState<string | null>(null);
  const [qualityMode, setQualityMode] = useState(false);
  const [candidateCount, setCandidateCount] = useState(2);

  // Generation control
  const [generateThrough, setGenerateThrough] = useState(4); // 0-4, default "All"
  const [configExpanded, setConfigExpanded] = useState(true);

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Scene edit tracking (for SceneEditorCard)
  const [sceneEdits, setSceneEdits] = useState<Record<number, Record<string, string>>>({});
  const [removedScenes, setRemovedScenes] = useState<Set<number>>(new Set());

  // Polling / new-asset detection
  const prevScenesRef = useRef<SceneDetail[] | null>(null);
  const [newAssetFlags, setNewAssetFlags] = useState<Record<number, string[]>>({});

  // Generating scenes (background asset generation for empty slots)
  const [generatingSceneIndices, setGeneratingSceneIndices] = useState<Set<number>>(new Set());

  // Final video upload
  const finalVideoInput = useRef<HTMLInputElement>(null);
  const [uploadingFinalVideo, setUploadingFinalVideo] = useState(false);

  // Model settings for cost estimation
  const [modelSettings, setModelSettings] = useState<EnabledModelsResponse | null>(null);
  useEffect(() => {
    getEnabledModels().then(setModelSettings).catch(() => {});
  }, []);

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------
  const editorMode = getEditorMode(projectId, detail);
  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel) ?? VIDEO_MODELS[0];
  const allowedDurations = selectedVideoModel.allowedDurations;
  const audioActive = enableAudio && selectedVideoModel.supportsAudio;
  const runThrough = sliderToRunThrough(generateThrough);
  const scenes = detail?.scenes ?? [];

  // Cost estimation
  const effectiveTotalDuration = sceneCount * clipDuration;
  const comfyuiCostOverride = modelSettings?.comfyui_cost_per_second;
  const cost = useMemo(() => {
    const base = estimatePartialCost(
      effectiveTotalDuration,
      clipDuration,
      textModel,
      imageModel,
      videoModel,
      audioActive,
      runThrough,
    );
    if (videoModel === "wan-2.2-ref-i2v" && comfyuiCostOverride != null && comfyuiCostOverride > 0 && runThrough !== "storyboard" && runThrough !== "keyframes") {
      const scenes = Math.ceil(effectiveTotalDuration / clipDuration);
      return base + scenes * clipDuration * comfyuiCostOverride;
    }
    return base;
  }, [effectiveTotalDuration, clipDuration, textModel, imageModel, videoModel, audioActive, comfyuiCostOverride, runThrough]);

  // Stage completion for slider
  const stageCompletion = useMemo(() => {
    if (!detail || scenes.length === 0) return [false, false, false, false, false];
    const hasStoryboard = scenes.some((s) => s.description && s.description.trim().length > 0);
    const hasKeyframes = scenes.some((s) => s.has_start_keyframe || s.has_end_keyframe);
    const hasClips = scenes.some((s) => s.has_clip);
    const hasAllClips = scenes.every((s) => s.has_clip || s.is_empty_slot);
    return [hasStoryboard, hasKeyframes, hasClips, hasAllClips, hasAllClips];
  }, [detail, scenes]);

  // ---------------------------------------------------------------------------
  // Load existing project on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!projectId) return;
    getProjectDetail(projectId)
      .then((d) => {
        setDetail(d);
        prevScenesRef.current = d.scenes;
        // Populate config from existing project
        if (d.title) setTitle(d.title);
        setPrompt(d.prompt ?? "");
        setStyle(d.style ?? "cinematic");
        setAspectRatio(d.aspect_ratio ?? "16:9");
        setClipDuration(d.clip_duration ?? 6);
        setSceneCount(d.scene_count);
        setTextModel(d.text_model ?? TEXT_MODELS[0].id);
        setImageModel(d.image_model ?? IMAGE_MODELS[0].id);
        setVideoModel(d.video_model ?? VIDEO_MODELS[0].id);
        setVisionModel(d.vision_model ?? "");
        setEnableAudio(d.audio_enabled ?? true);
        setManifestId(d.manifest_id ?? null);
        setQualityMode(d.quality_mode ?? false);
        setCandidateCount(d.candidate_count ?? 2);
        // If project is running or has content, collapse config
        if (RUNNING_STATUSES.has(d.status) || d.scenes.some((s) => !s.is_empty_slot && s.description.trim())) {
          setConfigExpanded(false);
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load project");
      });
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Polling during "running" mode
  // ---------------------------------------------------------------------------
  const pollDetail = useCallback(async () => {
    if (!projectId) return;
    try {
      const d = await getProjectDetail(projectId);
      // Detect newly arrived assets
      if (prevScenesRef.current) {
        const flags: Record<number, string[]> = {};
        for (const scene of d.scenes) {
          const prev = prevScenesRef.current.find((s) => s.scene_index === scene.scene_index);
          if (!prev) continue;
          const arrivals: string[] = [];
          if (!prev.has_start_keyframe && scene.has_start_keyframe) arrivals.push("start_kf");
          if (!prev.has_end_keyframe && scene.has_end_keyframe) arrivals.push("end_kf");
          if (!prev.has_clip && scene.has_clip) arrivals.push("clip");
          if (arrivals.length > 0) flags[scene.scene_index] = arrivals;
        }
        if (Object.keys(flags).length > 0) {
          setNewAssetFlags((prev) => ({ ...prev, ...flags }));
          // Clear highlights after 2 seconds
          setTimeout(() => {
            setNewAssetFlags((prev) => {
              const next = { ...prev };
              for (const key of Object.keys(flags)) delete next[Number(key)];
              return next;
            });
          }, 2000);
        }
      }
      prevScenesRef.current = d.scenes;
      setDetail(d);
      setSceneCount(d.scene_count);
      setError(null);

      // Detect generating scenes completion
      if (generatingSceneIndices.size > 0) {
        setGeneratingSceneIndices((prev) => {
          const next = new Set(prev);
          let changed = false;
          for (const idx of prev) {
            const scene = d.scenes.find((s) => s.scene_index === idx);
            if (scene && scene.has_end_keyframe && scene.has_clip) {
              next.delete(idx);
              changed = true;
            }
          }
          return changed ? next : prev;
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Polling failed");
    }
  }, [projectId, generatingSceneIndices]);

  const isPolling = editorMode === "running" || generatingSceneIndices.size > 0;
  usePolling(pollDetail, 2000, isPolling);

  // ---------------------------------------------------------------------------
  // Auto-collapse config when generation starts
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (editorMode === "running") setConfigExpanded(false);
  }, [editorMode]);

  // ---------------------------------------------------------------------------
  // Snap clip duration to allowed values when video model changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!allowedDurations.includes(clipDuration)) {
      const nearest = allowedDurations.reduce((a, b) =>
        Math.abs(b - clipDuration) < Math.abs(a - clipDuration) ? b : a,
      );
      setClipDuration(nearest);
    }
  }, [allowedDurations, clipDuration]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  async function handleGenerate() {
    if (submitting) return;
    setSubmitting(true);
    setError(null);

    try {
      let pid = projectId;

      // Create draft project if needed
      if (!pid) {
        const res = await createDraftProject({
          prompt: prompt.trim() || undefined,
          title: title.trim() || undefined,
          style,
          aspect_ratio: aspectRatio,
          clip_duration: clipDuration,
          scene_count: sceneCount,
          text_model: textModel,
          image_model: imageModel,
          video_model: videoModel,
          enable_audio: audioActive,
          manifest_id: manifestId ?? undefined,
          quality_mode: qualityMode,
          candidate_count: qualityMode ? candidateCount : undefined,
          vision_model: visionModel || undefined,
        });
        pid = res.project_id;
        setProjectId(pid);
        // Fetch the newly created project to get scene data
        const d = await getProjectDetail(pid);
        setDetail(d);
        prevScenesRef.current = d.scenes;
      }

      // Start generation
      await startGeneration(pid, {
        run_through: runThrough,
        text_model: textModel,
        image_model: imageModel,
        video_model: videoModel,
        vision_model: visionModel || undefined,
        clip_duration: clipDuration,
        enable_audio: audioActive,
      });

      // Polling will pick up the new status
      if (projectId) {
        // Trigger an immediate refresh
        const d = await getProjectDetail(pid);
        setDetail(d);
        prevScenesRef.current = d.scenes;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleStop() {
    if (!projectId) return;
    try {
      await stopProject(projectId);
    } catch {
      // Polling will show actual state
    }
  }

  function handleSceneChange(sceneIndex: number, field: string, value: string) {
    setSceneEdits((prev) => ({
      ...prev,
      [sceneIndex]: { ...(prev[sceneIndex] ?? {}), [field]: value },
    }));
  }

  function handleRemoveScene(sceneIndex: number) {
    setRemovedScenes((prev) => new Set(prev).add(sceneIndex));
  }

  function handleRestoreScene(sceneIndex: number) {
    setRemovedScenes((prev) => {
      const next = new Set(prev);
      next.delete(sceneIndex);
      return next;
    });
  }

  function handleAssetChanged() {
    // Trigger a refresh
    if (projectId) {
      getProjectDetail(projectId).then((d) => {
        setDetail(d);
        prevScenesRef.current = d.scenes;
      }).catch(() => {});
    }
  }

  async function handleGenerateScene(sceneIndex: number) {
    if (!projectId) return;
    setGeneratingSceneIndices((prev) => new Set(prev).add(sceneIndex));
    await generateNewScene(projectId, {
      scene_index: sceneIndex,
      all_scene_edits: sceneEdits,
      text_model: textModel,
      image_model: imageModel,
      video_model: videoModel,
    });
    // Polling will detect completion and clear the generating flag
  }

  async function handleUploadFinalVideo(file: File) {
    if (!projectId) return;
    setUploadingFinalVideo(true);
    try {
      await uploadFinalVideo(projectId, file);
      handleAssetChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadingFinalVideo(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const generateButtonLabel = (() => {
    if (submitting) return "Starting...";
    if (editorMode === "drafting") {
      return runThrough ? `Generate Through ${runThrough.charAt(0).toUpperCase() + runThrough.slice(1)}` : "Generate";
    }
    // editing mode — resume / re-generate
    return runThrough ? `Resume Through ${runThrough.charAt(0).toUpperCase() + runThrough.slice(1)}` : "Resume / Generate";
  })();

  const canGenerate = editorMode !== "running" && !submitting;
  const isPartialMode = runThrough === "storyboard" || runThrough === "keyframes";

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-4">
      {/* Project Config Bar */}
      <ProjectConfigBar
        title={title} setTitle={setTitle}
        prompt={prompt} setPrompt={setPrompt}
        style={style} setStyle={setStyle}
        aspectRatio={aspectRatio} setAspectRatio={setAspectRatio}
        textModel={textModel} setTextModel={setTextModel}
        imageModel={imageModel} setImageModel={setImageModel}
        videoModel={videoModel} setVideoModel={setVideoModel}
        visionModel={visionModel} setVisionModel={setVisionModel}
        enableAudio={enableAudio} setEnableAudio={setEnableAudio}
        manifestId={manifestId} setManifestId={setManifestId}
        qualityMode={qualityMode} setQualityMode={setQualityMode}
        candidateCount={candidateCount} setCandidateCount={setCandidateCount}
        expanded={configExpanded} setExpanded={setConfigExpanded}
        disabled={editorMode === "running"}
      />

      {/* Generation Controls */}
      <div className="rounded-lg border border-gray-700 bg-gray-900 p-4 space-y-3">
        <GenerateThroughSlider
          value={generateThrough}
          onChange={setGenerateThrough}
          stageCompletion={stageCompletion}
          disabled={editorMode === "running"}
        />

        <div className="flex flex-wrap items-center gap-3">
          {/* Scene count (only in drafting mode or when editable) */}
          {editorMode === "drafting" && (
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400">Scenes:</label>
              <input
                type="number"
                min={1}
                max={20}
                value={sceneCount}
                onChange={(e) => setSceneCount(Math.max(1, Math.min(20, Number(e.target.value))))}
                className="w-14 rounded border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-200 text-center focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
          )}

          {/* Clip duration buttons */}
          {!isPartialMode && (
            <div className="flex items-center gap-1">
              <label className="text-xs text-gray-400 mr-1">Clip:</label>
              {allowedDurations.map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => setClipDuration(d)}
                  disabled={editorMode === "running"}
                  className={clsx(
                    "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
                    clipDuration === d
                      ? "border-blue-500 bg-blue-500/20 text-blue-300"
                      : "border-gray-700 bg-gray-950 text-gray-500 hover:border-gray-600",
                    editorMode === "running" && "opacity-50 cursor-not-allowed",
                  )}
                >
                  {d}s
                </button>
              ))}
            </div>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* Cost estimate */}
          <span className="text-xs text-gray-500">
            ~${(qualityMode ? cost * candidateCount : cost).toFixed(2)}
            {runThrough && <span className="ml-1 text-cyan-500">({runThrough})</span>}
          </span>

          {/* Generate / Pause / Resume button */}
          {editorMode === "running" ? (
            <button
              type="button"
              onClick={handleStop}
              className="rounded-lg border border-red-700 bg-red-900/30 px-4 py-1.5 text-xs font-medium text-red-300 hover:bg-red-900/60 transition-colors"
            >
              Pause
            </button>
          ) : (
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className={clsx(
                "rounded-lg px-4 py-1.5 text-xs font-semibold transition-colors",
                canGenerate
                  ? "bg-blue-600 text-white hover:bg-blue-500"
                  : "bg-gray-800 text-gray-500 cursor-not-allowed",
              )}
            >
              {generateButtonLabel}
            </button>
          )}
        </div>

        {/* Status indicator */}
        {editorMode === "running" && detail && (
          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
            <span className="text-xs text-blue-400">
              {detail.status === "pending" ? "Starting..." : detail.status.replace("_", " ")}
              {detail.scenes.length > 0 && ` (${detail.scenes.filter((s) => s.has_clip).length}/${detail.scenes.length} clips)`}
            </span>
          </div>
        )}

        {/* Stopped/Staged/Failed banners */}
        {detail?.status === "stopped" && editorMode === "editing" && (
          <div className="rounded border border-amber-800 bg-amber-900/50 px-2 py-1 text-[11px] text-amber-300">
            Pipeline paused. Click Generate to resume from where it left off.
          </div>
        )}
        {detail?.status === "staged" && editorMode === "editing" && (
          <div className="rounded border border-cyan-800 bg-cyan-900/50 px-2 py-1 text-[11px] text-cyan-300">
            Completed through requested stage. Adjust the slider and Generate to continue.
          </div>
        )}
        {detail?.status === "failed" && (
          <div className="rounded border border-red-800 bg-red-900/50 px-2 py-1 text-[11px] text-red-300">
            Generation failed{detail.error_message ? `: ${detail.error_message}` : ""}. Fix issues and Generate to retry.
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-800 bg-red-900/50 px-3 py-2 text-sm text-red-300">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-400 hover:text-red-300">&times;</button>
        </div>
      )}

      {/* Scene Grid */}
      <div>
        <h2 className="mb-2 text-sm font-medium text-gray-400">
          Scenes ({detail ? scenes.length : sceneCount})
        </h2>

        {/* Drafting mode placeholder cards */}
        {editorMode === "drafting" && !detail && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: sceneCount }, (_, i) => (
              <div
                key={i}
                className="rounded-lg border border-dashed border-gray-700 bg-gray-900/30 p-4 flex items-center justify-center min-h-[120px]"
              >
                <span className="text-xs text-gray-600">Scene {i + 1}</span>
              </div>
            ))}
          </div>
        )}

        {/* Actual scene cards */}
        {detail && scenes.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {scenes.map((scene) => (
              <div
                key={scene.scene_index}
                className={clsx(
                  "transition-all duration-300",
                  newAssetFlags[scene.scene_index]?.length
                    ? "ring-2 ring-green-400 rounded-lg"
                    : "",
                )}
              >
                <SceneEditorCard
                  scene={scene}
                  edits={sceneEdits[scene.scene_index] ?? {}}
                  onChange={handleSceneChange}
                  onRemove={handleRemoveScene}
                  removed={removedScenes.has(scene.scene_index)}
                  onRestore={handleRestoreScene}
                  canRemove={scenes.length > 1}
                  projectId={projectId ?? undefined}
                  onAssetChanged={handleAssetChanged}
                  textModel={textModel}
                  videoModel={videoModel}
                  imageModel={imageModel}
                  allSceneEdits={sceneEdits}
                  onGenerateScene={handleGenerateScene}
                  isGeneratingAssets={
                    generatingSceneIndices.has(scene.scene_index) ||
                    (scene.generation_status != null && scene.generation_status !== "")
                  }
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Final Video Section */}
      {detail && editorMode !== "drafting" && (
        <div className="rounded-lg border border-gray-700 bg-gray-900 p-4">
          <h3 className="mb-2 text-sm font-medium text-gray-400">Final Video</h3>
          {detail.status === "complete" ? (
            <div className="space-y-2">
              <a
                href={`/api/projects/${projectId}/download`}
                className="inline-block rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-500 transition-colors"
              >
                Download Video
              </a>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-gray-500">
                Upload a final stitched video or complete generation to produce one automatically.
              </p>
              <input
                ref={finalVideoInput}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleUploadFinalVideo(f);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                onClick={() => finalVideoInput.current?.click()}
                disabled={uploadingFinalVideo}
                className="rounded-lg border border-gray-700 px-3 py-1.5 text-xs text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors disabled:opacity-50"
              >
                {uploadingFinalVideo ? "Uploading..." : "Upload Final Video"}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Navigation back to list */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onNavigate("list")}
          className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
        >
          &larr; Back to Projects
        </button>
        {projectId && (
          <button
            type="button"
            onClick={() => onNavigate("detail", projectId)}
            className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
          >
            View Details
          </button>
        )}
      </div>
    </div>
  );
}
