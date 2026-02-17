import { useState, useEffect } from "react";
import clsx from "clsx";
import { forkProject, fetchManifestAssets } from "../api/client.ts";
import type { ProjectDetail } from "../api/types.ts";
import type { ForkRequest, AssetResponse, AssetChanges, ModifiedAsset, NewForkUpload } from "../api/types.ts";
import {
  STYLE_OPTIONS,
  ASPECT_RATIOS,
  TOTAL_DURATION_MAX,
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
  estimateCost,
} from "../lib/constants.ts";
import { EditableSceneCard } from "./EditableSceneCard.tsx";

interface EditForkPanelProps {
  detail: ProjectDetail;
  onForked: (newProjectId: string) => void;
  onCancel: () => void;
}

export function EditForkPanel({ detail, onForked, onCancel }: EditForkPanelProps) {
  // Project-level state (initialized from existing detail)
  const [prompt, setPrompt] = useState(detail.prompt);
  const [style, setStyle] = useState(detail.style);
  const [aspectRatio, setAspectRatio] = useState(detail.aspect_ratio);
  const [clipDuration, setClipDuration] = useState(detail.clip_duration ?? 6);
  const [totalDuration, setTotalDuration] = useState(() => {
    const raw = detail.total_duration ?? 15;
    const clip = detail.clip_duration ?? 6;
    return Math.ceil(raw / clip) * clip;
  });
  const [textModel, setTextModel] = useState(detail.text_model ?? TEXT_MODELS[0].id);
  const [imageModel, setImageModel] = useState(detail.image_model ?? IMAGE_MODELS[0].id);
  const [videoModel, setVideoModel] = useState(detail.video_model ?? VIDEO_MODELS[0].id);
  const [enableAudio, setEnableAudio] = useState(detail.audio_enabled ?? false);

  // Scene edits: { sceneIndex: { field: value } }
  const [sceneEdits, setSceneEdits] = useState<Record<number, Record<string, string>>>({});

  // Deletion state
  const [deletedScenes, setDeletedScenes] = useState<Set<number>>(new Set());
  const [clearedKeyframes, setClearedKeyframes] = useState<Set<number>>(new Set());

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Asset management state (Phase 12)
  const [assets, setAssets] = useState<AssetResponse[]>([]);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [modifiedAssets, setModifiedAssets] = useState<Record<string, ModifiedAsset>>({});
  const [removedAssetIds, setRemovedAssetIds] = useState<Set<string>>(new Set());
  const [newUploads, setNewUploads] = useState<NewForkUpload[]>([]);

  useEffect(() => {
    if (!detail.manifest_id) return;
    setAssetsLoading(true);
    fetchManifestAssets(detail.manifest_id)
      .then(setAssets)
      .catch(() => {}) // Silent fail — asset section just won't show
      .finally(() => setAssetsLoading(false));
  }, [detail.manifest_id]);

  const selectedVideoModel = VIDEO_MODELS.find((m) => m.id === videoModel) ?? VIDEO_MODELS[0];
  const allowedDurations = selectedVideoModel.allowedDurations;
  const sceneCount = Math.round(totalDuration / clipDuration);
  const audioActive = enableAudio && selectedVideoModel.supportsAudio;

  function handleClipDurationChange(newClip: number) {
    setClipDuration(newClip);
    setTotalDuration((prev) => Math.max(newClip, Math.round(prev / newClip) * newClip));
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
      if (!scene) return prev;

      // Get original value
      const origMap: Record<string, string | null | undefined> = {
        scene_description: scene.description,
        start_frame_prompt: scene.start_frame_prompt,
        end_frame_prompt: scene.end_frame_prompt,
        video_motion_prompt: scene.video_motion_prompt,
        transition_notes: scene.transition_notes,
      };
      const original = origMap[field] ?? "";

      const sceneEditsForIdx = { ...(prev[sceneIndex] || {}) };

      if (value === original) {
        // Remove edit if back to original
        delete sceneEditsForIdx[field];
      } else {
        sceneEditsForIdx[field] = value;
      }

      const next = { ...prev };
      if (Object.keys(sceneEditsForIdx).length === 0) {
        delete next[sceneIndex];
      } else {
        next[sceneIndex] = sceneEditsForIdx;
      }
      return next;
    });
  }

  function handleDeleteScene(idx: number) {
    setDeletedScenes((prev) => new Set(prev).add(idx));
    setTotalDuration((prev) => Math.max(clipDuration, prev - clipDuration));
  }

  function handleRestoreScene(idx: number) {
    setDeletedScenes((prev) => {
      const next = new Set(prev);
      next.delete(idx);
      return next;
    });
    setTotalDuration((prev) => Math.min(TOTAL_DURATION_MAX, prev + clipDuration));
  }

  function handleClearKeyframes(idx: number) {
    setClearedKeyframes((prev) => new Set(prev).add(idx));
  }

  function handleRestoreKeyframes(idx: number) {
    setClearedKeyframes((prev) => {
      const next = new Set(prev);
      next.delete(idx);
      return next;
    });
  }

  // Asset management helpers (Phase 12)
  function getAssetStatus(assetId: string): "locked" | "modified" | "removed" {
    if (removedAssetIds.has(assetId)) return "removed";
    if (modifiedAssets[assetId]) return "modified";
    return "locked";
  }

  function handleRemoveAsset(assetId: string) {
    setRemovedAssetIds((prev) => new Set(prev).add(assetId));
  }

  function handleRestoreAsset(assetId: string) {
    setRemovedAssetIds((prev) => {
      const next = new Set(prev);
      next.delete(assetId);
      return next;
    });
    // Also clear any modifications
    setModifiedAssets((prev) => {
      const next = { ...prev };
      delete next[assetId];
      return next;
    });
  }

  function handleEditAssetField(assetId: string, field: string, value: string) {
    const asset = assets.find((a) => a.asset_id === assetId);
    if (!asset) return;
    const original = field === "reverse_prompt" ? asset.reverse_prompt ?? "" : asset.name;
    setModifiedAssets((prev) => {
      const existing = prev[assetId]?.changes || {};
      const updated = { ...existing, [field]: value };
      // Remove field if back to original
      if (value === original) delete updated[field];
      if (Object.keys(updated).length === 0) {
        const next = { ...prev };
        delete next[assetId];
        return next;
      }
      return { ...prev, [assetId]: { changes: updated } };
    });
  }

  function handleAddUpload(files: FileList | null) {
    if (!files) return;
    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(",")[1]; // strip data:...;base64, prefix
        setNewUploads((prev) => [
          ...prev,
          {
            image_data: base64,
            name: file.name.replace(/\.[^.]+$/, ""),
            asset_type: "CHARACTER",
            description: undefined,
            tags: undefined,
          },
        ]);
      };
      reader.readAsDataURL(file);
    });
  }

  function handleRemoveUpload(index: number) {
    setNewUploads((prev) => prev.filter((_, i) => i !== index));
  }

  function buildForkRequest(): ForkRequest {
    const req: ForkRequest = {};

    // Only include changed project-level fields
    if (prompt !== detail.prompt) req.prompt = prompt;
    if (style !== detail.style) req.style = style;
    if (aspectRatio !== detail.aspect_ratio) req.aspect_ratio = aspectRatio;
    if (clipDuration !== (detail.clip_duration ?? 6)) req.clip_duration = clipDuration;
    const origClip = detail.clip_duration ?? 6;
    const origSnapped = Math.ceil((detail.total_duration ?? 15) / origClip) * origClip;
    if (totalDuration !== origSnapped || clipDuration !== origClip || deletedScenes.size > 0) req.total_duration = totalDuration;
    if (textModel !== (detail.text_model ?? TEXT_MODELS[0].id)) req.text_model = textModel;
    if (imageModel !== (detail.image_model ?? IMAGE_MODELS[0].id)) req.image_model = imageModel;
    if (videoModel !== (detail.video_model ?? VIDEO_MODELS[0].id)) req.video_model = videoModel;
    if (enableAudio !== (detail.audio_enabled ?? false)) req.audio_enabled = enableAudio;

    if (Object.keys(sceneEdits).length > 0) {
      req.scene_edits = sceneEdits;
    }

    if (deletedScenes.size > 0) {
      req.deleted_scenes = [...deletedScenes];
    }

    if (clearedKeyframes.size > 0) {
      const filtered = [...clearedKeyframes].filter((i) => !deletedScenes.has(i));
      if (filtered.length > 0) {
        req.clear_keyframes = filtered;
      }
    }

    if (detail.manifest_id && (Object.keys(modifiedAssets).length > 0 || removedAssetIds.size > 0 || newUploads.length > 0)) {
      req.asset_changes = {
        modified_assets: Object.keys(modifiedAssets).length > 0 ? modifiedAssets : undefined,
        removed_asset_ids: removedAssetIds.size > 0 ? [...removedAssetIds] : undefined,
        new_uploads: newUploads.length > 0 ? newUploads : undefined,
      };
    }

    return req;
  }

  const hasChanges = () => {
    const req = buildForkRequest();
    return Object.keys(req).length > 0;
  };

  async function handleFork() {
    setSubmitting(true);
    setError(null);
    try {
      const req = buildForkRequest();
      const res = await forkProject(detail.project_id, req);
      onForked(res.project_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Fork failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Edit & Fork</h2>
          <p className="text-sm text-gray-400">
            Modify settings or scene prompts, then fork to create a new project.
          </p>
        </div>
        <button
          onClick={onCancel}
          className="rounded-md border border-gray-700 px-3 py-1.5 text-sm text-gray-400 hover:border-gray-600 transition-colors"
        >
          Cancel
        </button>
      </div>

      {/* Prompt */}
      <div>
        <label htmlFor="fork-prompt" className="mb-1 block text-sm font-medium text-gray-300">
          Prompt
        </label>
        <textarea
          id="fork-prompt"
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
                  ? "border-blue-500 bg-blue-500/20 text-blue-300"
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
        <label htmlFor="fork-totalDuration" className="mb-2 block text-sm font-medium text-gray-300">
          Total Duration: {totalDuration}s ({sceneCount} scene{sceneCount !== 1 ? "s" : ""})
        </label>
        <input
          id="fork-totalDuration"
          type="range"
          min={clipDuration}
          max={TOTAL_DURATION_MAX}
          step={clipDuration}
          value={totalDuration}
          onChange={(e) => setTotalDuration(Number(e.target.value))}
          className="w-full accent-blue-500"
        />
      </div>

      {/* Models */}
      <div className="space-y-4">
        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">Text Model</label>
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

        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">Image Model</label>
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

        <div>
          <label className="mb-2 block text-sm font-medium text-gray-300">Video Model</label>
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

        {/* Audio Toggle */}
        {selectedVideoModel.supportsAudio && (
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-300">Audio</label>
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
        <div>Estimated cost for fork: ~${estimateCost(
          sceneCount * clipDuration, clipDuration, textModel, imageModel, videoModel, audioActive
        ).toFixed(2)}</div>
        <div className="mt-1 text-xs text-gray-500">
          {sceneCount} scene{sceneCount !== 1 ? "s" : ""} &middot; Full regeneration cost shown (inherited assets reduce actual cost)
        </div>
      </div>

      {/* Asset Registry (Phase 12) */}
      {detail.manifest_id && (
        <div>
          <h3 className="mb-3 text-sm font-medium text-gray-400">
            Asset Registry
            {assets.length > 0 && (
              <span className="ml-2 text-gray-500">
                ({assets.length - removedAssetIds.size} active
                {removedAssetIds.size > 0 && `, ${removedAssetIds.size} removed`}
                {newUploads.length > 0 && `, ${newUploads.length} new`})
              </span>
            )}
          </h3>

          {assetsLoading ? (
            <div className="text-sm text-gray-500">Loading assets...</div>
          ) : assets.length === 0 ? (
            <div className="text-sm text-gray-500">No assets in manifest</div>
          ) : (
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {assets.map((asset) => {
                const status = getAssetStatus(asset.asset_id);
                return (
                  <div
                    key={asset.asset_id}
                    className={clsx(
                      "flex items-center gap-3 rounded-lg border px-3 py-2",
                      status === "removed"
                        ? "border-red-800/50 bg-red-900/20 opacity-60"
                        : status === "modified"
                        ? "border-amber-700 bg-amber-900/20"
                        : "border-gray-700 bg-gray-900",
                    )}
                  >
                    {/* Thumbnail */}
                    {asset.thumbnail_url || asset.reference_image_url ? (
                      <img
                        src={asset.thumbnail_url || asset.reference_image_url || ""}
                        alt={asset.name}
                        className="h-10 w-10 rounded object-cover"
                      />
                    ) : (
                      <div className="flex h-10 w-10 items-center justify-center rounded bg-gray-800 text-xs text-gray-500">
                        {asset.manifest_tag}
                      </div>
                    )}

                    {/* Info */}
                    <div className="min-w-0 flex-1">
                      <div className={clsx("text-sm font-medium", status === "removed" ? "text-gray-500 line-through" : "text-gray-200")}>
                        {asset.manifest_tag} — {asset.name}
                      </div>
                      {status !== "removed" && (
                        <input
                          type="text"
                          value={modifiedAssets[asset.asset_id]?.changes?.reverse_prompt ?? asset.reverse_prompt ?? ""}
                          onChange={(e) => handleEditAssetField(asset.asset_id, "reverse_prompt", e.target.value)}
                          placeholder="Reverse prompt..."
                          className={clsx(
                            "mt-1 w-full rounded border bg-transparent px-2 py-0.5 text-xs text-gray-400 focus:outline-none focus:ring-1",
                            status === "modified" ? "border-amber-700 focus:ring-amber-500" : "border-gray-700 focus:ring-blue-500",
                          )}
                        />
                      )}
                    </div>

                    {/* Status badge */}
                    <span className={clsx(
                      "shrink-0 rounded px-1.5 py-0.5 text-xs font-medium",
                      status === "locked" && "bg-gray-800 text-gray-400",
                      status === "modified" && "bg-amber-900/50 text-amber-300",
                      status === "removed" && "bg-red-900/50 text-red-400",
                    )}>
                      {status === "locked" ? "Inherited" : status === "modified" ? "Edited" : "Removed"}
                    </span>

                    {/* Actions */}
                    <div className="flex shrink-0 gap-1">
                      {status === "removed" ? (
                        <button
                          onClick={() => handleRestoreAsset(asset.asset_id)}
                          className="rounded px-2 py-1 text-xs text-blue-400 hover:bg-blue-900/30"
                          title="Restore"
                        >
                          Restore
                        </button>
                      ) : (
                        <button
                          onClick={() => handleRemoveAsset(asset.asset_id)}
                          className="rounded px-2 py-1 text-xs text-red-400 hover:bg-red-900/30"
                          title="Remove from fork"
                        >
                          Remove
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* New Uploads */}
          {newUploads.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="text-xs font-medium text-green-400">New Uploads</div>
              {newUploads.map((upload, i) => (
                <div key={i} className="flex items-center gap-2 rounded-lg border border-green-800/50 bg-green-900/20 px-3 py-2">
                  <div className="text-sm text-gray-300">{upload.name}</div>
                  <span className="text-xs text-gray-500">{upload.asset_type}</span>
                  <div className="flex-1" />
                  <button
                    onClick={() => handleRemoveUpload(i)}
                    className="rounded px-2 py-1 text-xs text-red-400 hover:bg-red-900/30"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Add New button */}
          <label className="mt-3 flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-gray-700 px-3 py-2 text-sm text-gray-400 hover:border-gray-600 hover:text-gray-300 transition-colors">
            <span>+ Add New Reference Images</span>
            <input
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => handleAddUpload(e.target.files)}
            />
          </label>
        </div>
      )}

      {/* Scene Edits */}
      {detail.scenes.length > 0 && (
        <div>
          <h3 className="mb-3 text-sm font-medium text-gray-400">
            Scenes ({deletedScenes.size > 0
              ? `${detail.scenes.length} \u2192 ${sceneCount}`
              : detail.scenes.length})
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {detail.scenes.map((scene) => (
              <EditableSceneCard
                key={scene.scene_index}
                scene={scene}
                edits={sceneEdits[scene.scene_index] || {}}
                onChange={handleSceneChange}
                deleted={deletedScenes.has(scene.scene_index)}
                keyframesCleared={clearedKeyframes.has(scene.scene_index)}
                onDelete={handleDeleteScene}
                onRestoreScene={handleRestoreScene}
                onClearKeyframes={handleClearKeyframes}
                onRestoreKeyframes={handleRestoreKeyframes}
                canDelete={sceneCount > 1}
              />
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-md border border-red-800 bg-red-900/50 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={handleFork}
          disabled={submitting}
          className={clsx(
            "rounded-lg px-6 py-2.5 text-sm font-semibold transition-colors",
            !submitting
              ? "bg-blue-600 text-white hover:bg-blue-500"
              : "bg-gray-800 text-gray-500 cursor-not-allowed",
          )}
        >
          {submitting ? "Forking..." : hasChanges() ? "Fork & Resume" : "Fork (exact copy)"}
        </button>
        <button
          onClick={onCancel}
          className="rounded-lg border border-gray-700 px-4 py-2.5 text-sm font-medium text-gray-300 hover:border-gray-600 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
