import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import clsx from "clsx";
import { editProject, getEnabledModels, regenerateProject, revertToCheckpoint, createCheckpoint, generateNewScene, getDownloadUrl, deleteProject } from "../api/client.ts";
import type { ProjectDetail, SceneDetail, SceneEditPayload, EditProjectRequest, EnabledModelsResponse, SceneReference } from "../api/types.ts";
import { usePolling } from "../hooks/usePolling.ts";
import {
  STYLE_OPTIONS,
  ASPECT_RATIOS,
  TOTAL_DURATION_MIN,
  TOTAL_DURATION_MAX,
  TOTAL_DURATION_STEP,
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
  estimatePartialCost,
} from "../lib/constants.ts";
import { SceneEditorCard } from "./SceneEditorCard.tsx";
import { CopyButton } from "./CopyButton.tsx";
import { MarkdownEditorModal } from "./MarkdownEditorModal.tsx";
import { ManifestSelector } from "./ManifestSelector.tsx";
import { RegenProgressBar } from "./RegenProgressBar.tsx";
import { useProjectWebSocket } from "../hooks/useProjectWebSocket.ts";
import type { WsEvent } from "../api/wsTypes.ts";

/** Schema for project export/import */
interface ProjectSchema {
  version: 1;
  exported_at: string;
  project: {
    title?: string | null;
    prompt: string;
    style: string;
    aspect_ratio: string;
    clip_duration: number;
    scene_count: number;
    text_model?: string | null;
    image_model?: string | null;
    video_model?: string | null;
    vision_model?: string | null;
    audio_enabled?: boolean;
    manifest_id?: string | null;
    quality_mode?: boolean;
    candidate_count?: number;
  };
  scenes: Array<{
    scene_index: number;
    description: string;
    start_frame_prompt?: string | null;
    end_frame_prompt?: string | null;
    video_motion_prompt?: string | null;
    transition_notes?: string | null;
    start_keyframe_url?: string | null;
    end_keyframe_url?: string | null;
    clip_url?: string | null;
    rewritten_keyframe_prompt?: string | null;
    rewritten_video_prompt?: string | null;
    selected_references?: SceneReference[];
  }>;
}

interface EditModeOverlayProps {
  detail: ProjectDetail;
  onCommitted: () => void;
  onCancel: () => void;
  /** Refresh detail data without exiting edit mode */
  onRefresh?: () => void;
}

export function EditModeOverlay({ detail, onCommitted, onCancel, onRefresh }: EditModeOverlayProps) {
  // Project-level state
  const [title, setTitle] = useState(detail.title ?? "");
  const [prompt, setPrompt] = useState(detail.prompt);
  const [style, setStyle] = useState(detail.style ?? "");
  const [aspectRatio, setAspectRatio] = useState(detail.aspect_ratio ?? "");
  const [clipDuration, setClipDuration] = useState(detail.clip_duration ?? 0);
  const [sceneCount, setSceneCount] = useState(detail.scene_count);
  const [textModel, setTextModel] = useState(detail.text_model ?? "");
  const [imageModel, setImageModel] = useState(detail.image_model ?? "");
  const [videoModel, setVideoModel] = useState(detail.video_model ?? "");
  const [visionModel, setVisionModel] = useState(detail.vision_model ?? "");
  const [enableAudio, setEnableAudio] = useState(detail.audio_enabled ?? false);
  const [runThrough, setRunThrough] = useState<string | null>(null);
  const [totalDuration, setTotalDuration] = useState(detail.scene_count * (detail.clip_duration ?? 6));
  const [manifestId, setManifestId] = useState<string | null>(detail.manifest_id ?? null);

  const isPartialMode = runThrough === "storyboard" || runThrough === "keyframes";

  // Scene edits
  const [sceneEdits, setSceneEdits] = useState<Record<number, Record<string, string>>>({});
  const [removedScenes, setRemovedScenes] = useState<Set<number>>(new Set());

  // Accordion state — all collapsed by default
  const [expandedScenes, setExpandedScenes] = useState<Set<number>>(new Set());

  const [commitMessage, setCommitMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [regenScope, setRegenScope] = useState<string | null>(null);
  const [regenMessage, setRegenMessage] = useState<string | null>(null);
  const [stitching, setStitching] = useState(false);
  const [stitchMessage, setStitchMessage] = useState<string | null>(null);
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);
  const importFileRef = useRef<HTMLInputElement>(null);
  const [importMessage, setImportMessage] = useState<string | null>(null);

  // Background operation tracking: regen ("all"/"stale") or stitch
  // When set, polling is enabled and we watch for head_sha to change.
  const [bgOpPending, setBgOpPending] = useState<string | null>(null);
  const bgOpBaselineSha = useRef<string | null>(null);

  // Track scenes currently generating assets in background
  const [generatingSceneIndices, setGeneratingSceneIndices] = useState<Set<number>>(new Set());

  // WebSocket progress state
  const [wsProgress, setWsProgress] = useState<{
    phase: string | null;
    totalScenes: number;
    completedScenes: number;
    currentSceneIndex: number | null;
    currentStatus: string | null;
  }>({ phase: null, totalScenes: 0, completedScenes: 0, currentSceneIndex: null, currentStatus: null });

  const handleWsEvent = useCallback((event: WsEvent) => {
    switch (event.type) {
      case "phase_started":
        setWsProgress({ phase: event.phase, totalScenes: event.total_scenes, completedScenes: 0, currentSceneIndex: null, currentStatus: null });
        break;
      case "phase_completed":
        setWsProgress(prev => ({ ...prev, phase: null, currentSceneIndex: null, currentStatus: null }));
        break;
      case "scene_status":
        setWsProgress(prev => ({ ...prev, currentSceneIndex: event.scene_index, currentStatus: event.status }));
        onRefresh?.();
        break;
      case "scene_keyframe_ready":
      case "scene_clip_ready":
        setWsProgress(prev => ({ ...prev, completedScenes: prev.completedScenes + 1 }));
        onRefresh?.();
        break;
      case "scene_text_ready":
      case "stitch_ready":
      case "refresh":
        onRefresh?.();
        break;
      case "checkpoint_created":
        // Intermediate progress — just refresh data, don't clear bgOpPending
        onRefresh?.();
        break;
      case "regen_complete": {
        // Final signal: all phases done — clear bgOpPending and show feedback
        const op = bgOpPending;
        setBgOpPending(null);
        bgOpBaselineSha.current = null;
        setWsProgress({ phase: null, totalScenes: 0, completedScenes: 0, currentSceneIndex: null, currentStatus: null });
        if (op === "stitch") {
          setStitchMessage("Re-stitch complete — video updated.");
        } else if (op) {
          setRegenMessage(`Regeneration (${op}) complete.`);
        }
        onRefresh?.();
        break;
      }
      case "error":
        setError(event.message);
        if (bgOpPending) {
          setBgOpPending(null);
          bgOpBaselineSha.current = null;
          setWsProgress({ phase: null, totalScenes: 0, completedScenes: 0, currentSceneIndex: null, currentStatus: null });
        }
        break;
      case "scene_regen_started":
      case "scene_regen_done":
        if (event.type === "scene_regen_done") {
          onRefresh?.();
        }
        break;
    }
  }, [bgOpPending, onRefresh]);

  const wsEnabled = bgOpPending !== null || generatingSceneIndices.size > 0;
  const { connected: wsConnected } = useProjectWebSocket({
    projectId: detail.project_id,
    enabled: wsEnabled,
    onEvent: handleWsEvent,
  });

  // Poll for completion when any background operation is running (fallback when WS is not connected)
  usePolling(
    () => { onRefresh?.(); },
    5000,
    (generatingSceneIndices.size > 0 || bgOpPending !== null) && !wsConnected,
  );

  // Detect background operation completion: head_sha changes after checkpoint
  useEffect(() => {
    if (!bgOpPending || !bgOpBaselineSha.current) return;
    if (wsConnected) return;  // WS handles completion via regen_complete
    if (detail.head_sha && detail.head_sha !== bgOpBaselineSha.current) {
      const op = bgOpPending;
      setBgOpPending(null);
      bgOpBaselineSha.current = null;
      if (op === "stitch") {
        setStitchMessage("Re-stitch complete — video updated.");
      } else {
        setRegenMessage(`Regeneration (${op}) complete.`);
      }
    }
  }, [detail.head_sha, bgOpPending, wsConnected]);

  // Completion detection: remove from generating set when assets arrive
  useEffect(() => {
    if (generatingSceneIndices.size === 0) return;
    setGeneratingSceneIndices((prev) => {
      const next = new Set(prev);
      let changed = false;
      for (const idx of prev) {
        const scene = detail.scenes.find((s) => s.scene_index === idx);
        if (scene && scene.has_end_keyframe && scene.has_clip) {
          next.delete(idx);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [detail.scenes, generatingSceneIndices]);

  // Auto-mark trailing scenes as removed when scene count is reduced
  useEffect(() => {
    const realScenes = detail.scenes.filter(s => !s.is_empty_slot);
    const newRemoved = new Set(removedScenes);
    let changed = false;
    for (const s of realScenes) {
      if (s.scene_index >= sceneCount && !newRemoved.has(s.scene_index)) {
        newRemoved.add(s.scene_index);
        changed = true;
      } else if (s.scene_index < sceneCount && newRemoved.has(s.scene_index)) {
        newRemoved.delete(s.scene_index);
        changed = true;
      }
    }
    if (changed) setRemovedScenes(newRemoved);
  }, [sceneCount]); // eslint-disable-line react-hooks/exhaustive-deps

  function buildSchema(): ProjectSchema {
    // Merge current edits with scene data to get effective values
    function effective(scene: SceneDetail, field: string, original: string | null | undefined): string {
      return sceneEdits[scene.scene_index]?.[field] ?? original ?? "";
    }

    return {
      version: 1,
      exported_at: new Date().toISOString(),
      project: {
        title: title || null,
        prompt,
        style,
        aspect_ratio: aspectRatio,
        clip_duration: clipDuration,
        scene_count: sceneCount,
        text_model: textModel,
        image_model: imageModel,
        video_model: videoModel,
        vision_model: visionModel || null,
        audio_enabled: enableAudio,
        manifest_id: manifestId,
        quality_mode: detail.quality_mode,
        candidate_count: detail.candidate_count,
      },
      scenes: detail.scenes
        .filter((s) => !s.is_empty_slot && !removedScenes.has(s.scene_index))
        .map((s) => ({
          scene_index: s.scene_index,
          description: effective(s, "scene_description", s.description),
          start_frame_prompt: effective(s, "start_frame_prompt", s.start_frame_prompt) || null,
          end_frame_prompt: effective(s, "end_frame_prompt", s.end_frame_prompt) || null,
          video_motion_prompt: effective(s, "video_motion_prompt", s.video_motion_prompt) || null,
          transition_notes: effective(s, "transition_notes", s.transition_notes) || null,
          start_keyframe_url: s.start_keyframe_url ?? null,
          end_keyframe_url: s.end_keyframe_url ?? null,
          clip_url: s.clip_url ?? null,
          rewritten_keyframe_prompt: s.rewritten_keyframe_prompt ?? null,
          rewritten_video_prompt: s.rewritten_video_prompt ?? null,
          selected_references: s.selected_references ?? [],
        })),
    };
  }

  function handleExportSchema() {
    const schema = buildSchema();
    const json = JSON.stringify(schema, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const slug = (detail.title ?? detail.prompt ?? "project").slice(0, 40).replace(/[^a-zA-Z0-9]+/g, "-").replace(/-+$/, "");
    a.download = `${slug}-schema.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleImportSchema(file: File) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const schema = JSON.parse(e.target?.result as string) as ProjectSchema;
        if (!schema.version || !schema.project) {
          setError("Invalid schema file: missing version or project data");
          return;
        }

        // Apply project-level settings
        const p = schema.project;
        if (p.prompt != null) setPrompt(p.prompt);
        if (p.style != null) setStyle(p.style);
        if (p.aspect_ratio != null) setAspectRatio(p.aspect_ratio);
        if (p.clip_duration != null) setClipDuration(p.clip_duration);
        if (p.scene_count != null) setSceneCount(p.scene_count);
        if (p.text_model != null) setTextModel(p.text_model);
        if (p.image_model != null) setImageModel(p.image_model);
        if (p.video_model != null) {
          handleVideoModelChange(p.video_model);
          // Override audio since handleVideoModelChange sets it based on model
          if (p.audio_enabled != null) setEnableAudio(p.audio_enabled);
        }
        if (p.vision_model !== undefined) setVisionModel(p.vision_model ?? "");
        if (p.audio_enabled != null && !p.video_model) setEnableAudio(p.audio_enabled);

        // Apply scene text edits
        if (schema.scenes?.length) {
          const textFields = [
            "scene_description",
            "start_frame_prompt",
            "end_frame_prompt",
            "video_motion_prompt",
            "transition_notes",
          ] as const;
          const fieldToSchemaKey: Record<string, keyof (typeof schema.scenes)[0]> = {
            scene_description: "description",
            start_frame_prompt: "start_frame_prompt",
            end_frame_prompt: "end_frame_prompt",
            video_motion_prompt: "video_motion_prompt",
            transition_notes: "transition_notes",
          };

          const newEdits: Record<number, Record<string, string>> = {};
          let appliedCount = 0;

          for (const importedScene of schema.scenes) {
            const existingScene = detail.scenes.find((s) => s.scene_index === importedScene.scene_index);
            if (!existingScene) continue;

            for (const field of textFields) {
              const importedValue = (importedScene[fieldToSchemaKey[field]] as string | null | undefined) ?? "";
              const origMap: Record<string, string | null | undefined> = {
                scene_description: existingScene.description,
                start_frame_prompt: existingScene.start_frame_prompt,
                end_frame_prompt: existingScene.end_frame_prompt,
                video_motion_prompt: existingScene.video_motion_prompt,
                transition_notes: existingScene.transition_notes,
              };
              const original = origMap[field] ?? "";

              if (importedValue !== original) {
                if (!newEdits[importedScene.scene_index]) newEdits[importedScene.scene_index] = {};
                newEdits[importedScene.scene_index][field] = importedValue;
                appliedCount++;
              }
            }
          }

          if (Object.keys(newEdits).length > 0) {
            setSceneEdits((prev) => {
              const merged = { ...prev };
              for (const [idx, fields] of Object.entries(newEdits)) {
                merged[Number(idx)] = { ...(merged[Number(idx)] || {}), ...fields };
              }
              return merged;
            });
          }

          setImportMessage(`Imported: project settings + ${appliedCount} scene field edit${appliedCount !== 1 ? "s" : ""} across ${schema.scenes.length} scene${schema.scenes.length !== 1 ? "s" : ""}`);
        } else {
          setImportMessage("Imported: project settings (no scene data in schema)");
        }

        setError(null);
      } catch {
        setError("Failed to parse schema file — ensure it is valid JSON");
      }
    };
    reader.readAsText(file);
  }

  // Baseline SHA for revert-on-cancel when regens are done in edit mode
  const baselineSha = useRef<string | null>(detail.head_sha ?? null);
  const regenDone = useRef(false);
  const [cancelling, setCancelling] = useState(false);

  const handleRegenStarted = useCallback((headSha: string | null) => {
    // Record the first baseline SHA we see (before any regens modify state)
    if (!regenDone.current && headSha) {
      baselineSha.current = headSha;
    }
    regenDone.current = true;
  }, []);

  const handleGenerateScene = useCallback(async (sceneIndex: number) => {
    const resp = await generateNewScene(detail.project_id, {
      scene_index: sceneIndex,
      all_scene_edits: Object.keys(sceneEdits).length > 0 ? sceneEdits : undefined,
      text_model: textModel,
      image_model: imageModel,
      video_model: videoModel,
      prompt: prompt || undefined,
    });
    // Record baseline SHA for revert-on-cancel
    handleRegenStarted(resp.head_sha ?? null);
    // Clear any edits the user had typed for this scene index (now real scene has them)
    setSceneEdits((prev) => {
      const next = { ...prev };
      delete next[sceneIndex];
      return next;
    });
    // Track as generating
    setGeneratingSceneIndices((prev) => new Set(prev).add(sceneIndex));
    // Refresh to pick up the new DB scene
    onRefresh?.();
  }, [detail.project_id, sceneEdits, textModel, imageModel, videoModel, prompt, handleRegenStarted, onRefresh]);

  async function handleCancel() {
    if (regenDone.current && baselineSha.current && detail.project_id) {
      setCancelling(true);
      try {
        await revertToCheckpoint(detail.project_id, baselineSha.current);
        onRefresh?.();
      } catch (err) {
        console.error("Failed to revert on cancel:", err);
      } finally {
        setCancelling(false);
      }
    }
    onCancel();
  }

  // Model settings
  const [modelSettings, setModelSettings] = useState<EnabledModelsResponse | null>(null);
  useEffect(() => {
    getEnabledModels().then(setModelSettings).catch(() => {});
  }, []);

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

  const allTextModels = useMemo(() => {
    const ollamaText = (modelSettings?.ollama_models ?? [])
      .filter((m) => m.enabled)
      .map((m) => ({ id: m.id, label: `${m.label} (Ollama)`, costPerCall: 0 }));
    return [...filteredTextModels, ...ollamaText];
  }, [filteredTextModels, modelSettings]);

  const allVisionModels = useMemo(() => {
    const ollamaVision = (modelSettings?.ollama_models ?? [])
      .filter((m) => m.enabled && m.vision)
      .map((m) => ({ id: m.id, label: `${m.label} (Ollama)`, costPerCall: 0 }));
    return [...filteredTextModels, ...ollamaVision];
  }, [filteredTextModels, modelSettings]);

  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel);
  const allowedDurations = selectedVideoModel?.allowedDurations ?? [];

  function handleClipDurationChange(newClip: number) {
    setClipDuration(newClip);
  }

  function handleVideoModelChange(id: string) {
    setVideoModel(id);
    const model = VIDEO_MODELS.find((m) => m.id === id);
    if (!model) return;
    if (clipDuration && !model.allowedDurations.includes(clipDuration)) {
      const nearest = model.allowedDurations.reduce((a, b) =>
        Math.abs(b - clipDuration) < Math.abs(a - clipDuration) ? b : a
      );
      handleClipDurationChange(nearest);
    }
    setEnableAudio(model.supportsAudio);
  }

  function handleSceneChange(sceneIndex: number, field: string, value: string) {
    setSceneEdits((prev) => {
      const scene = detail.scenes.find((s) => s.scene_index === sceneIndex);

      // For synthetic (empty slot) scenes, all edits are new — no original to compare
      if (!scene) {
        const editsForIdx = { ...(prev[sceneIndex] || {}) };
        if (value === "") {
          delete editsForIdx[field];
        } else {
          editsForIdx[field] = value;
        }
        const next = { ...prev };
        if (Object.keys(editsForIdx).length === 0) {
          delete next[sceneIndex];
        } else {
          next[sceneIndex] = editsForIdx;
        }
        return next;
      }

      const origMap: Record<string, string | null | undefined> = {
        scene_description: scene.description,
        start_frame_prompt: scene.start_frame_prompt,
        end_frame_prompt: scene.end_frame_prompt,
        video_motion_prompt: scene.video_motion_prompt,
        transition_notes: scene.transition_notes,
      };
      const original = origMap[field] ?? "";

      const editsForIdx = { ...(prev[sceneIndex] || {}) };
      if (value === original) {
        delete editsForIdx[field];
      } else {
        editsForIdx[field] = value;
      }

      const next = { ...prev };
      if (Object.keys(editsForIdx).length === 0) {
        delete next[sceneIndex];
      } else {
        next[sceneIndex] = editsForIdx;
      }
      return next;
    });
  }

  function handleRemoveScene(idx: number) {
    setRemovedScenes((prev) => new Set(prev).add(idx));
  }

  function handleRestoreScene(idx: number) {
    setRemovedScenes((prev) => {
      const next = new Set(prev);
      next.delete(idx);
      return next;
    });
  }

  function toggleScene(idx: number) {
    setExpandedScenes((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  function expandAllScenes() {
    setExpandedScenes(new Set(allScenes.map((s) => s.scene_index)));
  }

  function collapseAllScenes() {
    setExpandedScenes(new Set());
  }

  function buildEditRequest(): EditProjectRequest {
    const req: EditProjectRequest = {};

    if (title !== (detail.title ?? "")) req.title = title || undefined;
    if (prompt !== detail.prompt) req.prompt = prompt;
    if (style !== (detail.style ?? "")) req.style = style;
    if (aspectRatio !== (detail.aspect_ratio ?? "")) req.aspect_ratio = aspectRatio;
    if (clipDuration !== (detail.clip_duration ?? 0)) req.clip_duration = clipDuration || undefined;
    if (sceneCount !== detail.scene_count) req.target_scene_count = sceneCount;
    if (textModel !== (detail.text_model ?? "")) req.text_model = textModel || undefined;
    if (imageModel !== (detail.image_model ?? "")) req.image_model = imageModel || undefined;
    if (videoModel !== (detail.video_model ?? "")) req.video_model = videoModel || undefined;
    if ((visionModel || undefined) !== (detail.vision_model || undefined)) req.vision_model = visionModel || undefined;
    if (enableAudio !== (detail.audio_enabled ?? false)) req.audio_enabled = enableAudio;
    if (manifestId !== (detail.manifest_id ?? null)) req.manifest_id = manifestId;

    if (Object.keys(sceneEdits).length > 0) {
      const converted: Record<number, SceneEditPayload> = {};
      for (const [idx, edits] of Object.entries(sceneEdits)) {
        converted[Number(idx)] = edits as SceneEditPayload;
      }
      req.scene_edits = converted;
    }

    if (removedScenes.size > 0) {
      req.removed_scenes = [...removedScenes];
    }

    if (commitMessage.trim()) {
      req.commit_message = commitMessage.trim();
    }

    if (detail.head_sha) {
      req.expected_sha = detail.head_sha;
    }

    return req;
  }

  function hasChanges(): boolean {
    // Regens done in this edit session count as changes
    if (regenDone.current) return true;
    const req = buildEditRequest();
    // Exclude expected_sha and commit_message from change detection
    const { expected_sha: _e, commit_message: _c, ...rest } = req;
    return Object.keys(rest).length > 0;
  }

  // Count stale assets
  const staleCount = detail.scenes.reduce((count, s) => {
    let n = count;
    if (s.start_keyframe_staleness === "stale") n++;
    if (s.end_keyframe_staleness === "stale") n++;
    if (s.clip_staleness === "stale") n++;
    return n;
  }, 0);

  // Cost estimate
  const costEstimate = useMemo(() => {
    const effectiveTotalDuration = isPartialMode
      ? sceneCount * clipDuration
      : totalDuration;
    return estimatePartialCost(
      effectiveTotalDuration, clipDuration,
      textModel, imageModel, videoModel,
      enableAudio, runThrough,
    );
  }, [sceneCount, clipDuration, totalDuration, textModel, imageModel, videoModel, enableAudio, runThrough, isPartialMode]);

  const videoCostPerSecond = useMemo(() => {
    const vm = VIDEO_MODELS.find((m) => m.id === videoModel);
    return enableAudio && vm?.supportsAudio
      ? (vm?.costPerSecondAudio ?? 0.40)
      : (vm?.costPerSecond ?? 0.40);
  }, [videoModel, enableAudio]);

  async function handleCommit() {
    if (!hasChanges()) return;
    setSubmitting(true);
    setError(null);
    try {
      const req = buildEditRequest();
      const { expected_sha: _e, commit_message: _c, ...fieldChanges } = req;
      if (Object.keys(fieldChanges).length > 0) {
        // Text/field edits present — use the edit endpoint
        await editProject(detail.project_id, req);
      } else {
        // Regen-only changes — create a checkpoint of current state
        await createCheckpoint(detail.project_id);
      }
      onCommitted();
    } catch (err: unknown) {
      if (err instanceof Error && err.message.includes("Conflict")) {
        setError("Conflict: another edit was committed. Please refresh and try again.");
      } else {
        setError(err instanceof Error ? err.message : "Edit failed");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRegenerate(
    scope: "storyboard" | "keyframes" | "clips" | "stitch_only" | "all_phases",
  ) {
    setRegenScope(scope);
    setError(null);
    try {
      // Auto-save pending edits so the backend uses the latest state
      let currentSha = detail.head_sha ?? null;
      const req = buildEditRequest();
      const { expected_sha: _e, commit_message: _c, ...fieldChanges } = req;
      if (Object.keys(fieldChanges).length > 0) {
        const editResp = await editProject(detail.project_id, req);
        currentSha = editResp.head_sha;
        setSceneEdits({});  // Clear — edits now persisted in DB
        onRefresh?.();
      }

      bgOpBaselineSha.current = currentSha;
      await regenerateProject(detail.project_id, {
        scope,
        ...(textModel ? { text_model: textModel } : {}),
        ...(imageModel ? { image_model: imageModel } : {}),
        ...(videoModel ? { video_model: videoModel } : {}),
        ...(scope === "all_phases" ? { run_through: runThrough } : {}),
      });
      handleRegenStarted(detail.head_sha ?? null);
      setBgOpPending(scope);
      setRegenMessage(`Regeneration (${scope}) started — running in background.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Regeneration failed");
    } finally {
      setRegenScope(null);
    }
  }

  async function handleRestitch() {
    setStitching(true);
    setStitchMessage(null);
    setError(null);
    try {
      bgOpBaselineSha.current = detail.head_sha ?? null;
      await regenerateProject(detail.project_id, { scope: "stitch_only" });
      setBgOpPending("stitch");
      setStitchMessage("Re-stitching started — video will update when complete.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Re-stitch failed");
    } finally {
      setStitching(false);
    }
  }

  function handleAssetChanged() {
    // Refresh detail data in-place without exiting edit mode
    onRefresh?.();
  }

  /** Compute CSS --fill percentage for dark-slider range inputs */
  function sliderFill(value: number, min: number, max: number): React.CSSProperties {
    const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
    return { "--fill": `${pct}%` } as React.CSSProperties;
  }

  const activeScenes = detail.scenes.filter((s) => !s.is_empty_slot && !removedScenes.has(s.scene_index));

  // Synthetic empty slots when sceneCount exceeds active real scenes
  const maxExistingIdx = detail.scenes.length > 0
    ? Math.max(...detail.scenes.map(s => s.scene_index))
    : -1;
  const activeRealCount = detail.scenes.filter(
    s => !s.is_empty_slot && !removedScenes.has(s.scene_index)
  ).length;
  const syntheticCount = Math.max(0, sceneCount - activeRealCount);

  const syntheticScenes: SceneDetail[] = Array.from({ length: syntheticCount }, (_, i) => ({
    scene_index: maxExistingIdx + 1 + i,
    description: "",
    status: "pending",
    has_start_keyframe: false,
    has_end_keyframe: false,
    has_clip: false,
    clip_status: null,
    is_empty_slot: true,
  }));

  const allScenes = [...detail.scenes, ...syntheticScenes];

  return (
    <div className="space-y-6">
      {/* Hidden file input for schema import */}
      <input
        ref={importFileRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleImportSchema(f);
          e.target.value = "";
        }}
      />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex-1 mr-4">
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Untitled Project"
            className={clsx(
              "w-full bg-transparent text-lg font-bold focus:outline-none border-b pb-1 transition-colors",
              title !== (detail.title ?? "")
                ? "text-indigo-300 border-indigo-500"
                : "text-white border-transparent hover:border-gray-700",
            )}
          />
          <div className="flex items-center gap-1.5 mt-1">
            <code className="text-xs text-gray-500 font-mono">{detail.project_id}</code>
            <CopyButton text={detail.project_id} />
          </div>
          {staleCount > 0 && (
            <p className="mt-1 text-xs text-amber-400">
              {staleCount} stale asset{staleCount !== 1 ? "s" : ""} detected.
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleExportSchema}
            className="flex items-center gap-1.5 rounded-md border border-gray-700 px-3 py-1.5 text-sm text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors"
            title="Export project schema as JSON"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export
          </button>
          <button
            type="button"
            onClick={() => importFileRef.current?.click()}
            className="flex items-center gap-1.5 rounded-md border border-gray-700 px-3 py-1.5 text-sm text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors"
            title="Import project schema from JSON"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Import
          </button>
        </div>
      </div>

      {/* Import feedback */}
      {importMessage && (
        <div className="flex items-center justify-between rounded-md border border-blue-800 bg-blue-900/50 px-3 py-2 text-sm text-blue-300">
          <span>{importMessage}</span>
          <button onClick={() => setImportMessage(null)} className="text-blue-400 hover:text-blue-300 text-xs ml-2">
            Dismiss
          </button>
        </div>
      )}

      {/* Prompt */}
      <div>
        <div className="mb-1 flex items-center justify-between">
          <label htmlFor="edit-prompt" className="text-sm font-medium text-gray-300">
            Prompt
          </label>
          <div className="flex items-center gap-1">
            <CopyButton text={prompt} />
            <button
              type="button"
              onClick={() => setPromptEditorOpen(true)}
              className="inline-flex items-center justify-center h-5 w-5 rounded hover:bg-gray-700/50 transition-colors"
              title="Edit in markdown editor"
            >
              <svg className="h-3.5 w-3.5 text-gray-500 hover:text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
              </svg>
            </button>
          </div>
        </div>
        <textarea
          id="edit-prompt"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className={clsx(
            "w-full rounded-lg border bg-gray-900 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-1",
            prompt !== detail.prompt
              ? "border-amber-600 focus:ring-amber-500"
              : "border-gray-700 focus:ring-blue-500",
          )}
        />
        {promptEditorOpen && (
          <MarkdownEditorModal
            label="Project Prompt"
            value={prompt}
            onChange={setPrompt}
            onClose={() => setPromptEditorOpen(false)}
          />
        )}
      </div>

      {/* Final Video */}
      <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-400">Final Video</h3>
          <div className="flex items-center gap-2">
            {detail.status === "complete" && (
              <a
                href={`${getDownloadUrl(detail.project_id)}?dl=1`}
                className="rounded px-2.5 py-1 text-[11px] font-medium text-green-300 bg-green-900/50 hover:bg-green-800/50 transition-colors"
              >
                Download
              </a>
            )}
            <button
              type="button"
              onClick={handleRestitch}
              disabled={stitching || regenScope !== null || bgOpPending !== null}
              className={clsx(
                "rounded px-2.5 py-1 text-[11px] font-medium transition-colors",
                stitching
                  ? "bg-gray-800 text-gray-500"
                  : "bg-green-900/50 text-green-300 hover:bg-green-800/50",
              )}
            >
              {stitching ? "Stitching..." : detail.status === "complete" ? "Re-stitch" : "Stitch"}
            </button>
          </div>
        </div>
        {stitchMessage && (
          <div className="mb-2 flex items-center justify-between rounded border border-green-800 bg-green-900/50 px-2 py-1 text-[11px] text-green-300">
            <span>{stitchMessage}</span>
            <button onClick={() => setStitchMessage(null)} className="text-green-400 hover:text-green-300 text-xs ml-2">
              &times;
            </button>
          </div>
        )}
        {detail.status === "complete" ? (
          <video
            src={`${getDownloadUrl(detail.project_id)}?v=${detail.head_sha ?? ""}`}
            className="w-full rounded-lg border border-gray-700"
            controls
            preload="metadata"
          />
        ) : (
          <div className="flex h-24 items-center justify-center rounded-lg border border-dashed border-gray-700 bg-gray-950 text-xs text-gray-600">
            No final video yet — stitch when all scenes have clips
          </div>
        )}
      </div>

      {/* Asset Manifest */}
      <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-4">
        <h3 className="mb-2 text-sm font-medium text-gray-400">Asset Manifest</h3>
        <ManifestSelector
          selectedManifestId={manifestId}
          onManifestSelect={setManifestId}
        />
      </div>

      {/* Scene Count / Total Duration (dual mode) */}
      {isPartialMode || !clipDuration ? (
        <div>
          <label htmlFor="edit-sceneCount" className="mb-2 block text-sm font-medium text-gray-300">
            Scenes: {sceneCount}
            {clipDuration
              ? <span className="text-gray-500"> · {sceneCount * clipDuration}s total</span>
              : <span className="text-gray-500"> · 0s total (select a scene length)</span>
            }
            {activeScenes.length !== sceneCount && <span className="text-gray-500"> · {activeScenes.length} active</span>}
            {syntheticCount > 0 && <span className="text-gray-500">, {syntheticCount} new</span>}
            {removedScenes.size > 0 && <span className="text-gray-500">, {removedScenes.size} removed</span>}
          </label>
          <input
            id="edit-sceneCount"
            type="range"
            min={1}
            max={50}
            step={1}
            value={sceneCount}
            onChange={(e) => {
              const count = Number(e.target.value);
              setSceneCount(count);
              setTotalDuration(count * clipDuration);
            }}
            className="dark-slider w-full"
            style={sliderFill(sceneCount, 1, 50)}
          />
          <div className="mt-1 flex justify-between text-xs text-gray-600">
            <span>1</span>
            <span>50</span>
          </div>
        </div>
      ) : (
        <div>
          <label htmlFor="edit-totalDuration" className="mb-2 block text-sm font-medium text-gray-300">
            Total Duration: {totalDuration}s ({sceneCount} scenes{activeScenes.length !== sceneCount ? ` · ${activeScenes.length} active` : ""}{removedScenes.size > 0 ? ` · ${removedScenes.size} removed` : ""})
          </label>
          <input
            id="edit-totalDuration"
            type="range"
            min={clipDuration}
            max={TOTAL_DURATION_MAX}
            step={TOTAL_DURATION_STEP}
            value={totalDuration}
            onChange={(e) => {
              const dur = Number(e.target.value);
              setTotalDuration(dur);
              setSceneCount(Math.ceil(dur / clipDuration));
            }}
            className="dark-slider w-full"
            style={sliderFill(totalDuration, clipDuration, TOTAL_DURATION_MAX)}
          />
          <div className="mt-1 flex justify-between text-xs text-gray-600">
            <span>{clipDuration}s</span>
            <span>{TOTAL_DURATION_MAX}s</span>
          </div>
        </div>
      )}

      {/* Scene Edits */}
      {allScenes.length > 0 && (
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-400">
              Scenes ({detail.scenes.length}{syntheticCount > 0 ? ` + ${syntheticCount} new` : ""})
              {removedScenes.size > 0 && (
                <span className="ml-1 text-red-400">
                  ({removedScenes.size} removed)
                </span>
              )}
            </h3>
            <button
              type="button"
              onClick={expandedScenes.size > 0 ? collapseAllScenes : expandAllScenes}
              className="text-[11px] text-gray-500 hover:text-gray-300 transition-colors"
            >
              {expandedScenes.size > 0 ? "Collapse All" : "Expand All"}
            </button>
          </div>
          <div className="grid gap-3">
            {allScenes.map((scene) => (
              <SceneEditorCard
                key={scene.scene_index}
                scene={scene}
                edits={sceneEdits[scene.scene_index] || {}}
                onChange={handleSceneChange}
                removed={removedScenes.has(scene.scene_index)}
                onRemove={handleRemoveScene}
                onRestore={handleRestoreScene}
                canRemove={activeScenes.length + syntheticCount > 1}
                projectId={detail.project_id}
                onAssetChanged={handleAssetChanged}
                onRegenStarted={handleRegenStarted}
                textModel={textModel}
                videoModel={videoModel}
                imageModel={imageModel}
                allSceneEdits={sceneEdits}
                prompt={prompt}
                onGenerateScene={handleGenerateScene}
                isGeneratingAssets={generatingSceneIndices.has(scene.scene_index)}
                wsConnected={wsConnected}
                expanded={expandedScenes.has(scene.scene_index)}
                onToggleExpand={() => toggleScene(scene.scene_index)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Regeneration toolbar */}
      {(() => {
        const busy = regenScope !== null || bgOpPending !== null;
        const showKeyframes = runThrough !== "storyboard";
        const showClips = runThrough !== "storyboard" && runThrough !== "keyframes";
        const showVideo = runThrough === null;

        const storyboardDisabled = busy || !textModel;
        const keyframesDisabled = busy || !textModel || !imageModel;
        const clipsDisabled = busy || !textModel || !imageModel || !videoModel;
        const videoDisabled = busy || !videoModel;
        const allPhasesDisabled = busy || !textModel
          || (showKeyframes && !imageModel)
          || (showClips && !videoModel);

        const chips: Array<{
          scope: "storyboard" | "keyframes" | "clips" | "stitch_only" | "all_phases";
          label: string;
          disabled: boolean;
          activeClass: string;
          visible: boolean;
        }> = [
          {
            scope: "storyboard", label: "Storyboard", disabled: storyboardDisabled,
            activeClass: "bg-violet-900/50 text-violet-300 hover:bg-violet-800/50",
            visible: true,
          },
          {
            scope: "keyframes", label: "Keyframes", disabled: keyframesDisabled,
            activeClass: "bg-blue-900/50 text-blue-300 hover:bg-blue-800/50",
            visible: showKeyframes,
          },
          {
            scope: "clips", label: "Clips", disabled: clipsDisabled,
            activeClass: "bg-teal-900/50 text-teal-300 hover:bg-teal-800/50",
            visible: showClips,
          },
          {
            scope: "stitch_only", label: "Video", disabled: videoDisabled,
            activeClass: "bg-green-900/50 text-green-300 hover:bg-green-800/50",
            visible: showVideo,
          },
          {
            scope: "all_phases", label: "All Phases", disabled: allPhasesDisabled,
            activeClass: "bg-indigo-900/50 text-indigo-300 hover:bg-indigo-800/50",
            visible: true,
          },
        ];

        return (
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-gray-800 bg-gray-900/50 px-3 py-2">
            <span className="text-[11px] font-medium text-gray-500">Regenerate:</span>
            {chips.filter(c => c.visible).map(({ scope, label, disabled, activeClass }) => {
              const isActive = regenScope === scope || bgOpPending === scope;
              return (
                <button
                  key={scope}
                  type="button"
                  onClick={() => handleRegenerate(scope)}
                  disabled={disabled}
                  className={clsx(
                    "rounded px-2.5 py-1 text-[11px] font-medium transition-colors",
                    isActive
                      ? "bg-gray-800 text-gray-500"
                      : disabled
                        ? "bg-gray-800 text-gray-600 cursor-not-allowed"
                        : activeClass,
                  )}
                >
                  {isActive ? "Regenerating..." : label}
                </button>
              );
            })}
          </div>
        );
      })()}

      {/* WebSocket progress bar — shown during background operations */}
      {bgOpPending && (
        <RegenProgressBar
          phase={wsProgress.phase}
          totalScenes={wsProgress.totalScenes}
          completedScenes={wsProgress.completedScenes}
          currentSceneIndex={wsProgress.currentSceneIndex}
          currentStatus={wsProgress.currentStatus}
          wsConnected={wsConnected}
        />
      )}

      {/* Cost Estimate */}
      <div className="rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-300">
        <div>
          Estimated cost: ~${costEstimate.toFixed(2)}
          {runThrough && <span className="ml-1 text-xs text-indigo-400">(through {runThrough})</span>}
        </div>
        <div className="mt-1 text-xs text-gray-500">
          {sceneCount} scene{sceneCount !== 1 ? "s" : ""}
          {runThrough !== "storyboard" && <> &middot; ${(IMAGE_MODELS.find((m) => m.id === imageModel)?.costPerImage ?? 0).toFixed(2)}/img</>}
          {runThrough !== "storyboard" && runThrough !== "keyframes" && <> &middot; ${videoCostPerSecond.toFixed(2)}/s video{enableAudio ? " (with audio)" : ""}</>}
        </div>
      </div>

      {/* Generate Through (4-position slider) */}
      <div>
        <label className="mb-2 block text-sm font-medium text-gray-300">
          Generate Through: <span className="text-indigo-300">{
            runThrough === "storyboard" ? "Storyboard" :
            runThrough === "keyframes" ? "Keyframes" :
            runThrough === "video" ? "Clips" : "Video"
          }</span>
        </label>
        <input
          type="range"
          min={1}
          max={4}
          step={1}
          value={
            runThrough === "storyboard" ? 1 :
            runThrough === "keyframes" ? 2 :
            runThrough === "video" ? 3 : 4
          }
          onChange={(e) => {
            const v = Number(e.target.value);
            setRunThrough(v === 1 ? "storyboard" : v === 2 ? "keyframes" : v === 3 ? "video" : null);
          }}
          className="dark-slider w-full"
          style={sliderFill(
            runThrough === "storyboard" ? 1 : runThrough === "keyframes" ? 2 : runThrough === "video" ? 3 : 4,
            1, 4,
          )}
        />
        <div className="mt-1 flex justify-between text-xs text-gray-500">
          <span>Storyboard</span>
          <span>Keyframes</span>
          <span>Clips</span>
          <span>Video</span>
        </div>
      </div>

      {/* Models & Settings */}
      <div className="space-y-4">
        {/* Text Model + Vision Model (side by side) */}
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">Text Model</label>
            <div className="flex flex-wrap gap-2">
              {allTextModels.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setTextModel(textModel === m.id ? "" : m.id)}
                  className={clsx(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                    textModel === m.id
                      ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
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
                For image analysis &amp; scoring
              </span>
            </label>
            <div className="flex flex-wrap gap-2">
              {allVisionModels.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setVisionModel(visionModel === m.id ? "" : m.id)}
                  className={clsx(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                    visionModel === m.id
                      ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                      : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                  )}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Image Model + Aspect Ratio (visible from Keyframes+) */}
        {runThrough !== "storyboard" && (
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-2 block text-sm font-medium text-gray-300">Image Model</label>
              <div className="flex flex-wrap gap-2">
                {filteredImageModels.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setImageModel(imageModel === m.id ? "" : m.id)}
                    className={clsx(
                      "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                      imageModel === m.id
                        ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                        : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                    )}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-gray-300">Aspect Ratio</label>
              <div className="flex gap-2">
                {ASPECT_RATIOS.map((ar) => (
                  <button
                    key={ar}
                    type="button"
                    onClick={() => setAspectRatio(aspectRatio === ar ? "" : ar)}
                    className={clsx(
                      "rounded-md border px-4 py-1.5 text-sm font-medium transition-colors",
                      aspectRatio === ar
                        ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                        : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                    )}
                  >
                    {ar}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Video Model + Scene Length + Audio (visible from Clips+) */}
        {runThrough !== "storyboard" && runThrough !== "keyframes" && (
          <>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="mb-2 block text-sm font-medium text-gray-300">Video Model</label>
                <div className="flex flex-wrap gap-2">
                  {filteredVideoModels.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => videoModel === m.id ? setVideoModel("") : handleVideoModelChange(m.id)}
                      className={clsx(
                        "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                        videoModel === m.id
                          ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                          : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                      )}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-gray-300">Scene Length</label>
                <div className="flex gap-2">
                  {allowedDurations.map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => handleClipDurationChange(clipDuration === d ? 0 : d)}
                      className={clsx(
                        "rounded-md border px-4 py-1.5 text-sm font-medium transition-colors",
                        clipDuration === d
                          ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                          : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                      )}
                    >
                      {d}s
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Audio Toggle */}
            {selectedVideoModel?.supportsAudio && (
              <div>
                <label className="mb-2 block text-sm font-medium text-gray-300">Audio</label>
                <button
                  type="button"
                  onClick={() => setEnableAudio(!enableAudio)}
                  className={clsx(
                    "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                    enableAudio ? "bg-indigo-600" : "bg-gray-700",
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
          </>
        )}

        {/* Style (always visible) */}
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">Style</label>
          <div className="flex flex-wrap gap-2">
            {STYLE_OPTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setStyle(style === s ? "" : s)}
                className={clsx(
                  "rounded-md border px-3 py-1.5 text-sm font-medium capitalize transition-colors",
                  style === s
                    ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                    : "border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600",
                )}
              >
                {s.replace("_", " ")}
              </button>
            ))}
          </div>
        </div>

        {/* Quality Mode (read-only) */}
        {detail.quality_mode && (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-full bg-amber-900/50 border border-amber-700 px-2.5 py-0.5 text-xs font-medium text-amber-300">
              Quality Mode: {detail.candidate_count ?? 2}x candidates
            </span>
          </div>
        )}
      </div>

      {/* Commit message */}
      <div>
        <label htmlFor="edit-message" className="mb-1 block text-sm font-medium text-gray-300">
          Commit Message (optional)
        </label>
        <input
          id="edit-message"
          type="text"
          value={commitMessage}
          onChange={(e) => setCommitMessage(e.target.value)}
          placeholder="Describe your changes..."
          className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-md border border-red-800 bg-red-900/50 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Regen feedback */}
      {regenMessage && (
        <div className="flex items-center justify-between rounded-md border border-green-800 bg-green-900/50 px-3 py-2 text-sm text-green-300">
          <span>{regenMessage}</span>
          <button onClick={() => setRegenMessage(null)} className="text-green-400 hover:text-green-300 text-xs ml-2">
            Dismiss
          </button>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleCommit}
          disabled={submitting || !hasChanges()}
          className={clsx(
            "rounded-lg px-6 py-2.5 text-sm font-semibold transition-colors",
            hasChanges() && !submitting
              ? "bg-indigo-600 text-white hover:bg-indigo-500"
              : "bg-gray-800 text-gray-500 cursor-not-allowed",
          )}
        >
          {submitting ? "Committing..." : "Commit Changes"}
        </button>
        <button
          onClick={handleCancel}
          disabled={cancelling}
          className="rounded-lg border border-gray-700 px-4 py-2.5 text-sm font-medium text-gray-300 hover:border-gray-600 transition-colors disabled:opacity-50"
        >
          {cancelling ? "Reverting..." : "Cancel"}
        </button>
        <button
          onClick={async () => {
            if (!confirm("Delete this project? This cannot be undone.")) return;
            try {
              await deleteProject(detail.project_id);
              onCancel();
            } catch (err) {
              setError(err instanceof Error ? err.message : "Delete failed");
            }
          }}
          className="ml-auto rounded-lg border border-red-800 px-4 py-2.5 text-sm font-medium text-red-400 hover:bg-red-900/50 hover:border-red-700 transition-colors"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
