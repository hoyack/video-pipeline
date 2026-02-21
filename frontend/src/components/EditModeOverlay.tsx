import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import clsx from "clsx";
import { editProject, getEnabledModels, regenerateProject, revertToCheckpoint, createCheckpoint, generateNewScene, getDownloadUrl } from "../api/client.ts";
import type { ProjectDetail, SceneDetail, SceneEditPayload, EditProjectRequest, EnabledModelsResponse, SceneReference } from "../api/types.ts";
import { usePolling } from "../hooks/usePolling.ts";
import {
  STYLE_OPTIONS,
  ASPECT_RATIOS,
  TOTAL_DURATION_MAX,
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
} from "../lib/constants.ts";
import { SceneEditorCard } from "./SceneEditorCard.tsx";
import { CopyButton } from "./CopyButton.tsx";
import { MarkdownEditorModal } from "./MarkdownEditorModal.tsx";

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
  const [prompt, setPrompt] = useState(detail.prompt);
  const [style, setStyle] = useState(detail.style);
  const [aspectRatio, setAspectRatio] = useState(detail.aspect_ratio);
  const [clipDuration, setClipDuration] = useState(detail.clip_duration ?? 6);
  const [sceneCount, setSceneCount] = useState(detail.scene_count);
  const [textModel, setTextModel] = useState(detail.text_model ?? TEXT_MODELS[0].id);
  const [imageModel, setImageModel] = useState(detail.image_model ?? IMAGE_MODELS[0].id);
  const [videoModel, setVideoModel] = useState(detail.video_model ?? VIDEO_MODELS[0].id);
  const [visionModel, setVisionModel] = useState(detail.vision_model ?? "");
  const [enableAudio, setEnableAudio] = useState(detail.audio_enabled ?? false);

  // Scene edits
  const [sceneEdits, setSceneEdits] = useState<Record<number, Record<string, string>>>({});
  const [removedScenes, setRemovedScenes] = useState<Set<number>>(new Set());

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

  // Poll for completion when any background operation is running
  usePolling(
    () => { onRefresh?.(); },
    5000,
    generatingSceneIndices.size > 0 || bgOpPending !== null,
  );

  // Detect background operation completion: head_sha changes after checkpoint
  useEffect(() => {
    if (!bgOpPending || !bgOpBaselineSha.current) return;
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
  }, [detail.head_sha, bgOpPending]);

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

  function buildSchema(): ProjectSchema {
    // Merge current edits with scene data to get effective values
    function effective(scene: SceneDetail, field: string, original: string | null | undefined): string {
      return sceneEdits[scene.scene_index]?.[field] ?? original ?? "";
    }

    return {
      version: 1,
      exported_at: new Date().toISOString(),
      project: {
        title: detail.title,
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
        manifest_id: detail.manifest_id,
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
  }, [detail.project_id, sceneEdits, textModel, imageModel, videoModel, handleRegenStarted, onRefresh]);

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

  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel) ?? VIDEO_MODELS[0];
  const allowedDurations = selectedVideoModel.allowedDurations;

  function handleClipDurationChange(newClip: number) {
    setClipDuration(newClip);
  }

  function handleVideoModelChange(id: string) {
    setVideoModel(id);
    const model = VIDEO_MODELS.find((m) => m.id === id) ?? VIDEO_MODELS[0];
    if (!model.allowedDurations.includes(clipDuration)) {
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

  function buildEditRequest(): EditProjectRequest {
    const req: EditProjectRequest = {};

    if (prompt !== detail.prompt) req.prompt = prompt;
    if (style !== detail.style) req.style = style;
    if (aspectRatio !== detail.aspect_ratio) req.aspect_ratio = aspectRatio;
    if (clipDuration !== (detail.clip_duration ?? 6)) req.clip_duration = clipDuration;
    if (sceneCount !== detail.scene_count) req.target_scene_count = sceneCount;
    if (textModel !== (detail.text_model ?? TEXT_MODELS[0].id)) req.text_model = textModel;
    if (imageModel !== (detail.image_model ?? IMAGE_MODELS[0].id)) req.image_model = imageModel;
    if (videoModel !== (detail.video_model ?? VIDEO_MODELS[0].id)) req.video_model = videoModel;
    if ((visionModel || undefined) !== (detail.vision_model || undefined)) req.vision_model = visionModel || undefined;
    if (enableAudio !== (detail.audio_enabled ?? false)) req.audio_enabled = enableAudio;

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

  async function handleRegenerate(scope: "stale" | "all") {
    setRegenScope(scope);
    setError(null);
    try {
      bgOpBaselineSha.current = detail.head_sha ?? null;
      await regenerateProject(detail.project_id, { scope });
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
        <div>
          <h2 className="text-lg font-bold text-white">Edit Project</h2>
          <div className="flex items-center gap-1.5 mt-0.5">
            <code className="text-xs text-gray-500 font-mono">{detail.project_id}</code>
            <CopyButton text={detail.project_id} />
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Modify settings or scene prompts in-place. Changes are saved as a versioned checkpoint.
          </p>
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
          <button
            onClick={handleCancel}
            disabled={cancelling}
            className="rounded-md border border-gray-700 px-3 py-1.5 text-sm text-gray-400 hover:border-gray-600 transition-colors disabled:opacity-50"
          >
            {cancelling ? "Reverting..." : "Cancel"}
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

      {/* Regeneration toolbar */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-gray-800 bg-gray-900/50 px-3 py-2">
        <span className="text-[11px] font-medium text-gray-500">Regenerate:</span>
        {staleCount > 0 && (
          <button
            type="button"
            onClick={() => handleRegenerate("stale")}
            disabled={regenScope !== null || bgOpPending !== null}
            className={clsx(
              "rounded px-2.5 py-1 text-[11px] font-medium transition-colors",
              regenScope === "stale"
                ? "bg-gray-800 text-gray-500"
                : "bg-amber-900/50 text-amber-300 hover:bg-amber-800/50",
            )}
          >
            {regenScope === "stale" ? "Regenerating..." : bgOpPending === "stale" ? "Regenerating..." : `Stale (${staleCount})`}
          </button>
        )}
        <button
          type="button"
          onClick={() => handleRegenerate("all")}
          disabled={regenScope !== null}
          className={clsx(
            "rounded px-2.5 py-1 text-[11px] font-medium transition-colors",
            regenScope === "all"
              ? "bg-gray-800 text-gray-500"
              : "bg-indigo-900/50 text-indigo-300 hover:bg-indigo-800/50",
          )}
        >
          {regenScope === "all" ? "Regenerating..." : bgOpPending === "all" ? "Regenerating..." : "All Assets"}
        </button>
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

      {/* Style */}
      <div>
        <label className="mb-2 block text-sm font-medium text-gray-300">Style</label>
        <div className="flex flex-wrap gap-2">
          {STYLE_OPTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStyle(s)}
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

      {/* Aspect Ratio */}
      <div>
        <label className="mb-2 block text-sm font-medium text-gray-300">Aspect Ratio</label>
        <div className="flex gap-2">
          {ASPECT_RATIOS.map((ar) => (
            <button
              key={ar}
              type="button"
              onClick={() => setAspectRatio(ar)}
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

      {/* Scene Length */}
      <div>
        <label className="mb-2 block text-sm font-medium text-gray-300">Scene Length</label>
        <div className="flex gap-2">
          {allowedDurations.map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => handleClipDurationChange(d)}
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

      {/* Scene Count */}
      <div>
        <label htmlFor="edit-sceneCount" className="mb-2 block text-sm font-medium text-gray-300">
          Scene Count: {sceneCount} ({activeScenes.length} active{syntheticCount > 0 ? `, ${syntheticCount} new` : ""}{removedScenes.size > 0 ? `, ${removedScenes.size} removed` : ""})
        </label>
        <input
          id="edit-sceneCount"
          type="range"
          min={1}
          max={Math.ceil(TOTAL_DURATION_MAX / clipDuration)}
          step={1}
          value={sceneCount}
          onChange={(e) => setSceneCount(Number(e.target.value))}
          className="w-full accent-indigo-500"
        />
      </div>

      {/* Models */}
      <div className="space-y-4">
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">Text Model</label>
          <div className="flex flex-wrap gap-2">
            {allTextModels.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setTextModel(m.id)}
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
              For image analysis, reverse-prompting, and scoring
            </span>
          </label>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setVisionModel("")}
              className={clsx(
                "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                visionModel === ""
                  ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
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
          <label className="mb-2 block text-sm font-medium text-gray-300">Image Model</label>
          <div className="flex flex-wrap gap-2">
            {filteredImageModels.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setImageModel(m.id)}
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
          <label className="mb-2 block text-sm font-medium text-gray-300">Video Model</label>
          <div className="flex flex-wrap gap-2">
            {filteredVideoModels.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => handleVideoModelChange(m.id)}
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

        {/* Audio Toggle */}
        {selectedVideoModel.supportsAudio && (
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
      </div>

      {/* Scene Edits */}
      {allScenes.length > 0 && (
        <div>
          <h3 className="mb-3 text-sm font-medium text-gray-400">
            Scenes ({detail.scenes.length}{syntheticCount > 0 ? ` + ${syntheticCount} new` : ""})
            {removedScenes.size > 0 && (
              <span className="ml-1 text-red-400">
                ({removedScenes.size} removed)
              </span>
            )}
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
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
                onGenerateScene={handleGenerateScene}
                isGeneratingAssets={generatingSceneIndices.has(scene.scene_index)}
              />
            ))}
          </div>
        </div>
      )}

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
      <div className="flex gap-3">
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
      </div>
    </div>
  );
}
