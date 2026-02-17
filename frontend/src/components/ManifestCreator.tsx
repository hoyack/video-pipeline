import { useState, useEffect } from "react";
import type {
  ManifestDetail,
  AssetResponse,
  UpdateAssetRequest,
  ProcessingProgress,
} from "../api/types.ts";
import {
  createManifest,
  getManifestDetail,
  updateManifest,
  createAsset,
  updateAsset,
  deleteAsset,
  uploadAssetImage,
  processManifest,
  getProcessingProgress,
  reprocessAsset,
  uploadVideoForManifest,
  getExtractionProgress,
} from "../api/client.ts";
import { AssetUploader } from "./AssetUploader.tsx";
import { AssetEditor } from "./AssetEditor.tsx";

interface ManifestCreatorProps {
  manifestId?: string | null;
  onSaved: (manifestId: string) => void;
  onCancel: () => void;
}

const CATEGORIES = [
  "CUSTOM",
  "CHARACTERS",
  "ENVIRONMENTS",
  "OBJECTS",
  "PROPS",
  "STYLES",
];

const ASSET_TYPES = ["CHARACTER", "OBJECT", "VEHICLE", "ENVIRONMENT", "PROP", "STYLE", "OTHER"];

// Human-readable step labels
const STEP_LABELS: Record<string, string> = {
  contact_sheet: "Assembling contact sheet...",
  yolo_detection: "Detecting objects and faces...",
  face_matching: "Cross-matching faces...",
  reverse_prompting: "Generating asset descriptions...",
  finalizing: "Assigning tags and populating registry...",
  done: "Processing complete!",
};

// Extraction step labels
const EXTRACTION_STEP_LABELS: Record<string, string> = {
  initializing: "Initializing...",
  analyzing: "Analyzing video...",
  sampling: "Sampling frames...",
  deduplicating: "Deduplicating with CLIP...",
  saving: "Saving assets...",
  complete: "Extraction complete!",
};

export function ManifestCreator({
  manifestId,
  onSaved,
  onCancel,
}: ManifestCreatorProps) {
  const [manifest, setManifest] = useState<ManifestDetail | null>(null);
  const [assets, setAssets] = useState<AssetResponse[]>([]);
  const [pendingFiles, setPendingFiles] = useState<Map<string, File>>(
    new Map()
  );
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("CUSTOM");
  const [tags, setTags] = useState("");
  const [saving, setSaving] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadingAssets, setUploadingAssets] = useState<Set<string>>(
    new Set()
  );
  const [progress, setProgress] = useState<ProcessingProgress | null>(null);
  const [reprocessingAssets, setReprocessingAssets] = useState<Set<string>>(
    new Set()
  );
  const [editingFields, setEditingFields] = useState<
    Record<string, Record<string, boolean>>
  >({});

  // Video extraction state
  const [extracting, setExtracting] = useState(false);
  const [extractionProgress, setExtractionProgress] =
    useState<ProcessingProgress | null>(null);

  const isNewManifest = !manifestId;

  // Determine current stage based on manifest status
  const currentStage = (() => {
    if (extracting) return 1; // Stay on stage 1 but show extraction UI
    if (!manifest || manifest.status === "DRAFT") return 1;
    if (manifest.status === "EXTRACTING") return 1;
    if (manifest.status === "PROCESSING") return 2;
    if (manifest.status === "READY" || manifest.status === "ERROR") return 3;
    return 1;
  })();

  // Load existing manifest in edit mode
  useEffect(() => {
    if (manifestId) {
      getManifestDetail(manifestId)
        .then((data) => {
          setManifest(data);
          setName(data.name);
          setDescription(data.description || "");
          setCategory(data.category);
          setTags(data.tags ? data.tags.join(", ") : "");
          setAssets(data.assets);

          // If manifest is already PROCESSING, start polling
          if (data.status === "PROCESSING") {
            setProcessing(true);
          }
          // If manifest is EXTRACTING, start extraction polling
          if (data.status === "EXTRACTING") {
            setExtracting(true);
          }
        })
        .catch((err) => {
          setError(`Failed to load manifest: ${err.message}`);
        });
    }
  }, [manifestId]);

  // Polling for Stage 2 (PROCESSING)
  useEffect(() => {
    if (!processing || !manifest?.manifest_id) return;

    const pollInterval = setInterval(async () => {
      try {
        const progressData = await getProcessingProgress(manifest.manifest_id);
        setProgress(progressData);

        if (progressData.status === "complete") {
          clearInterval(pollInterval);
          setProcessing(false);
          // Reload manifest to get updated status and assets
          const updated = await getManifestDetail(manifest.manifest_id);
          setManifest(updated);
          setAssets(updated.assets);
        } else if (progressData.status === "error") {
          clearInterval(pollInterval);
          setProcessing(false);
          setError(progressData.error || "Processing failed");
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        console.error("Failed to fetch progress:", errorMessage);
        // Don't clear interval on network errors - keep polling
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [processing, manifest?.manifest_id]);

  // Polling for video frame extraction
  useEffect(() => {
    if (!extracting || !manifest?.manifest_id) return;

    const pollInterval = setInterval(async () => {
      try {
        const progressData = await getExtractionProgress(manifest.manifest_id);
        setExtractionProgress(progressData);

        if (progressData.status === "complete") {
          clearInterval(pollInterval);
          setExtracting(false);
          setExtractionProgress(null);
          // Reload manifest to get extracted frame assets
          const updated = await getManifestDetail(manifest.manifest_id);
          setManifest(updated);
          setAssets(updated.assets);
        } else if (progressData.status === "error") {
          clearInterval(pollInterval);
          setExtracting(false);
          setExtractionProgress(null);
          setError(progressData.error || "Video extraction failed");
          // Reload manifest to get current state
          const updated = await getManifestDetail(manifest.manifest_id);
          setManifest(updated);
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        console.error("Failed to fetch extraction progress:", errorMessage);
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [extracting, manifest?.manifest_id]);

  // Lazy-create manifest helper (shared between image and video upload)
  const ensureManifestExists = async (): Promise<string | null> => {
    const currentManifestId = manifestId || manifest?.manifest_id;
    if (currentManifestId) return currentManifestId;

    try {
      const newManifest = await createManifest({
        name: name || "Untitled Manifest",
        description: description || undefined,
        category,
        tags: tags
          ? tags
              .split(",")
              .map((t) => t.trim())
              .filter(Boolean)
          : undefined,
      });
      setManifest(newManifest as unknown as ManifestDetail);
      return newManifest.manifest_id;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to create manifest: ${errorMessage}`);
      return null;
    }
  };

  const handleFilesSelected = async (files: File[]) => {
    if (files.length === 0) return;

    const currentManifestId = await ensureManifestExists();
    if (!currentManifestId) return;

    // Process files sequentially to avoid overwhelming API
    for (const file of files) {
      try {
        // Create asset
        const asset = await createAsset(currentManifestId, {
          name: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
          asset_type: "CHARACTER", // Default type
        });

        // Store file for preview
        setPendingFiles((prev) => new Map(prev).set(asset.asset_id, file));

        // Mark as uploading
        setUploadingAssets((prev) => new Set(prev).add(asset.asset_id));

        // Upload image
        const updatedAsset = await uploadAssetImage(asset.asset_id, file);

        // Add to assets list
        setAssets((prev) => [...prev, updatedAsset]);

        // Clear uploading state
        setUploadingAssets((prev) => {
          const next = new Set(prev);
          next.delete(asset.asset_id);
          return next;
        });
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        setError(`Failed to upload ${file.name}: ${errorMessage}`);
      }
    }
  };

  const handleVideoSelected = async (file: File) => {
    const currentManifestId = await ensureManifestExists();
    if (!currentManifestId) return;

    setError(null);
    setExtracting(true);

    try {
      await uploadVideoForManifest(currentManifestId, file);
      // Update local manifest status
      setManifest((prev) =>
        prev ? { ...prev, status: "EXTRACTING" } : null
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to upload video: ${errorMessage}`);
      setExtracting(false);
    }
  };

  const handleAssetUpdate = async (
    assetId: string,
    updates: UpdateAssetRequest
  ) => {
    try {
      const updatedAsset = await updateAsset(assetId, updates);
      setAssets((prev) =>
        prev.map((a) => (a.asset_id === assetId ? updatedAsset : a))
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to update asset: ${errorMessage}`);
    }
  };

  const handleAssetDelete = async (assetId: string) => {
    if (!confirm("Remove this asset from the manifest?")) return;

    try {
      await deleteAsset(assetId);
      setAssets((prev) => prev.filter((a) => a.asset_id !== assetId));
      setPendingFiles((prev) => {
        const next = new Map(prev);
        next.delete(assetId);
        return next;
      });
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to delete asset: ${errorMessage}`);
    }
  };

  const handleProcess = async () => {
    if (!manifest?.manifest_id) return;

    setProcessing(true);
    setError(null);

    try {
      await processManifest(manifest.manifest_id);
      // Update local status to transition to Stage 2
      setManifest((prev) => (prev ? { ...prev, status: "PROCESSING" } : null));
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to start processing: ${errorMessage}`);
      setProcessing(false);
    }
  };

  const handleReprocess = async (assetId: string) => {
    setReprocessingAssets((prev) => new Set(prev).add(assetId));
    setError(null);

    try {
      const updatedAsset = await reprocessAsset(assetId);
      setAssets((prev) =>
        prev.map((a) => (a.asset_id === assetId ? updatedAsset : a))
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to reprocess asset: ${errorMessage}`);
    } finally {
      setReprocessingAssets((prev) => {
        const next = new Set(prev);
        next.delete(assetId);
        return next;
      });
    }
  };

  const handleReprocessAll = async () => {
    if (!manifest?.manifest_id) return;
    if (
      !confirm(
        "Re-run full processing pipeline? This will regenerate all asset descriptions."
      )
    )
      return;

    setProcessing(true);
    setError(null);

    try {
      await processManifest(manifest.manifest_id);
      setManifest((prev) => (prev ? { ...prev, status: "PROCESSING" } : null));
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to start processing: ${errorMessage}`);
      setProcessing(false);
    }
  };

  const handleSave = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const currentManifestId = manifestId || manifest?.manifest_id;

      if (currentManifestId) {
        // Update existing manifest
        await updateManifest(currentManifestId, {
          name,
          description: description || undefined,
          category,
          tags: tags
            ? tags
                .split(",")
                .map((t) => t.trim())
                .filter(Boolean)
            : undefined,
        });
        onSaved(currentManifestId);
      } else {
        // Create new manifest (no assets uploaded yet)
        const newManifest = await createManifest({
          name,
          description: description || undefined,
          category,
          tags: tags
            ? tags
                .split(",")
                .map((t) => t.trim())
                .filter(Boolean)
            : undefined,
        });
        onSaved(newManifest.manifest_id);
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to save manifest: ${errorMessage}`);
    } finally {
      setSaving(false);
    }
  };

  const toggleFieldEdit = (assetId: string, field: string) => {
    setEditingFields((prev) => ({
      ...prev,
      [assetId]: {
        ...prev[assetId],
        [field]: !prev[assetId]?.[field],
      },
    }));
  };

  const isFieldEditing = (assetId: string, field: string) => {
    return editingFields[assetId]?.[field] || false;
  };

  const getQualityBadgeColor = (score: number | null) => {
    if (score === null) return "bg-gray-700 text-gray-400";
    if (score >= 7) return "bg-green-700 text-green-300";
    if (score >= 4) return "bg-yellow-700 text-yellow-300";
    return "bg-red-700 text-red-300";
  };

  // Extraction progress UI (shown during video frame extraction)
  const renderExtractionProgress = () => {
    const stepLabel =
      EXTRACTION_STEP_LABELS[extractionProgress?.current_step || ""] ||
      "Extracting frames...";
    const candidateFrames =
      (extractionProgress?.progress as Record<string, number> | null)
        ?.candidate_frames || 0;
    const uniqueFrames =
      (extractionProgress?.progress as Record<string, number> | null)
        ?.unique_frames || 0;

    return (
      <div className="mb-8 rounded-lg border border-purple-800 bg-purple-900/20 p-8">
        <div className="flex flex-col items-center gap-4">
          {/* Spinner */}
          <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div>

          {/* Step label */}
          <p className="text-lg text-purple-300">{stepLabel}</p>

          {/* Frame counts */}
          <div className="flex gap-6 text-sm">
            {candidateFrames > 0 && (
              <div>
                <span className="text-gray-400">Sampled:</span>{" "}
                <span className="text-white font-medium">
                  {candidateFrames} frames
                </span>
              </div>
            )}
            {uniqueFrames > 0 && (
              <div>
                <span className="text-gray-400">Unique:</span>{" "}
                <span className="text-purple-300 font-medium">
                  {uniqueFrames} frames
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Video info banner (shown in Stage 1 after extraction)
  const renderVideoInfoBanner = () => {
    if (!manifest?.source_video_duration) return null;

    const videoFrameAssets = assets.filter((a) => a.source === "video_frame");
    if (videoFrameAssets.length === 0) return null;

    return (
      <div className="mb-4 rounded-lg border border-purple-800 bg-purple-900/20 px-4 py-3">
        <p className="text-sm text-purple-300">
          Video uploaded ({manifest.source_video_duration.toFixed(1)}s) —{" "}
          {videoFrameAssets.length} frames extracted
        </p>
      </div>
    );
  };

  // STAGE 1: Upload + tag UI (DRAFT)
  const renderStage1 = () => (
    <>
      {/* Header form */}
      <div className="mb-8 rounded-lg border border-gray-800 bg-gray-900/50 p-6">
        <h2 className="text-xl font-bold text-white mb-4">
          {isNewManifest ? "New Manifest" : "Edit Manifest"}
        </h2>

        {/* Name */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            Name <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none"
            placeholder="My Manifest"
          />
        </div>

        {/* Description */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none resize-none"
            placeholder="Describe your manifest..."
          />
        </div>

        {/* Category + Tags */}
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-1">
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-1">Tags</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none"
              placeholder="tag1, tag2, tag3"
            />
          </div>
        </div>
      </div>

      {/* Extraction progress (shown during video extraction) */}
      {extracting && renderExtractionProgress()}

      {/* Upload zone (hidden during extraction) */}
      {!extracting && (
        <div className="mb-8">
          <AssetUploader
            onFilesSelected={handleFilesSelected}
            onVideoSelected={handleVideoSelected}
            disabled={saving}
          />
        </div>
      )}

      {/* Video info banner */}
      {!extracting && renderVideoInfoBanner()}

      {/* Asset list */}
      <div className="mb-8">
        <h3 className="text-lg font-medium text-gray-200 mb-4">
          Assets ({assets.length})
        </h3>
        {assets.length === 0 && !extracting ? (
          <p className="text-sm text-gray-500 text-center py-8">
            No assets yet. Drop images or a video above to add assets.
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {assets.map((asset) => (
              <AssetEditor
                key={asset.asset_id}
                asset={asset}
                imageFile={pendingFiles.get(asset.asset_id)}
                onUpdate={handleAssetUpdate}
                onDelete={handleAssetDelete}
                isUploading={uploadingAssets.has(asset.asset_id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="flex justify-end gap-3 pt-6 border-t border-gray-800">
        <button
          onClick={onCancel}
          className="text-gray-400 hover:text-gray-300 px-4 py-2 text-sm"
        >
          Cancel
        </button>
        {assets.length > 0 && !extracting && (
          <button
            onClick={handleProcess}
            disabled={processing}
            className="bg-green-600 hover:bg-green-500 text-white rounded-lg px-6 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {processing ? "Starting..." : "Process"}
          </button>
        )}
        <button
          onClick={handleSave}
          disabled={saving || extracting || !name.trim()}
          className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? "Saving..." : "Save Draft"}
        </button>
      </div>
    </>
  );

  // STAGE 2: Processing progress UI
  const renderStage2 = () => {
    const stepLabel =
      STEP_LABELS[progress?.current_step || ""] || "Processing...";
    const uploadsTotal = progress?.progress?.uploads_total || 0;
    const uploadsProcessed = progress?.progress?.uploads_processed || 0;
    const cropsTotal = progress?.progress?.crops_total || 0;
    const cropsReversePrompted =
      progress?.progress?.crops_reverse_prompted || 0;

    const uploadsProgress =
      uploadsTotal > 0 ? (uploadsProcessed / uploadsTotal) * 100 : 0;
    const cropsProgress =
      cropsTotal > 0 ? (cropsReversePrompted / cropsTotal) * 100 : 0;

    return (
      <div className="flex flex-col items-center justify-center py-16">
        <div className="w-full max-w-lg">
          {/* Status header */}
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-white mb-2">
              Processing Manifest
            </h2>
            <p className="text-sm text-gray-400">{manifest?.name}</p>
          </div>

          {/* Spinner */}
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>

          {/* Current step */}
          <div className="text-center mb-8">
            <p className="text-lg text-gray-300">{stepLabel}</p>
          </div>

          {/* Progress bars */}
          <div className="space-y-6">
            {/* Images processed */}
            <div>
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Images processed</span>
                <span>
                  {uploadsProcessed} / {uploadsTotal}
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-500 h-full transition-all duration-300"
                  style={{ width: `${uploadsProgress}%` }}
                ></div>
              </div>
            </div>

            {/* Assets described */}
            {cropsTotal > 0 && (
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-2">
                  <span>Assets described</span>
                  <span>
                    {cropsReversePrompted} / {cropsTotal}
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-green-500 h-full transition-all duration-300"
                    style={{ width: `${cropsProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          {/* Face merges info */}
          {progress?.progress?.face_merges !== undefined &&
            progress.progress.face_merges > 0 && (
              <div className="mt-6 text-center text-sm text-gray-400">
                <p>
                  Merged {progress.progress.face_merges} duplicate face
                  {progress.progress.face_merges === 1 ? "" : "s"}
                </p>
              </div>
            )}
        </div>
      </div>
    );
  };

  // STAGE 3: Review/refine UI
  const renderStage3 = () => {
    const uploadedAssets = assets.filter((a) => a.source === "uploaded");
    const extractedAssets = assets.filter((a) => a.source === "extracted");
    const videoFrameAssets = assets.filter((a) => a.source === "video_frame");

    const assetsByType = assets.reduce(
      (acc, asset) => {
        acc[asset.asset_type] = (acc[asset.asset_type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    return (
      <>
        {/* Header */}
        <div className="mb-8 rounded-lg border border-gray-800 bg-gray-900/50 p-6">
          <h2 className="text-xl font-bold text-white mb-2">
            Review & Refine Assets
          </h2>
          <p className="text-sm text-gray-400 mb-4">{manifest?.name}</p>

          {/* Summary stats */}
          <div className="flex gap-6 text-sm flex-wrap">
            <div>
              <span className="text-gray-400">Total assets:</span>{" "}
              <span className="text-white font-medium">{assets.length}</span>
            </div>
            {uploadedAssets.length > 0 && (
              <div>
                <span className="text-gray-400">Uploaded:</span>{" "}
                <span className="text-white font-medium">
                  {uploadedAssets.length}
                </span>
              </div>
            )}
            {videoFrameAssets.length > 0 && (
              <div>
                <span className="text-gray-400">Video frames:</span>{" "}
                <span className="text-purple-300 font-medium">
                  {videoFrameAssets.length}
                </span>
              </div>
            )}
            {extractedAssets.length > 0 && (
              <div>
                <span className="text-gray-400">Extracted:</span>{" "}
                <span className="text-white font-medium">
                  {extractedAssets.length}
                </span>
              </div>
            )}
            {Object.entries(assetsByType).map(([type, count]) => (
              <div key={type}>
                <span className="text-gray-400">{type}:</span>{" "}
                <span className="text-white font-medium">{count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Add more assets */}
        {!extracting && (
          <div className="mb-8">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Add more references
            </h3>
            <AssetUploader
              onFilesSelected={handleFilesSelected}
              onVideoSelected={handleVideoSelected}
              disabled={processing}
            />
          </div>
        )}

        {extracting && extractionProgress && renderExtractionProgress()}

        {/* Asset list */}
        <div className="mb-8 space-y-4">
          {assets.map((asset) => (
            <div
              key={asset.asset_id}
              className="rounded-lg border border-gray-800 bg-gray-900/50 p-4"
            >
              <div className="flex gap-4">
                {/* Thumbnail */}
                <div className="flex-shrink-0">
                  {asset.reference_image_url ? (
                    <img
                      src={asset.reference_image_url}
                      alt={asset.name}
                      className="w-32 h-32 object-cover rounded border border-gray-700"
                    />
                  ) : (
                    <div className="w-32 h-32 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-600 text-xs">
                      No image
                    </div>
                  )}
                </div>

                {/* Details */}
                <div className="flex-1 min-w-0">
                  {/* Name + badges row */}
                  <div className="flex items-start gap-2 mb-2">
                    <input
                      type="text"
                      value={asset.name}
                      onChange={(e) =>
                        handleAssetUpdate(asset.asset_id, {
                          name: e.target.value,
                        })
                      }
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white text-sm font-medium flex-1 min-w-0 focus:border-blue-500 outline-none"
                    />
                    <span className="px-2 py-1 text-xs font-medium bg-blue-900 text-blue-300 rounded">
                      {asset.manifest_tag}
                    </span>
                    <select
                      value={asset.asset_type}
                      onChange={(e) =>
                        handleAssetUpdate(asset.asset_id, {
                          asset_type: e.target.value,
                        })
                      }
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 focus:border-blue-500 outline-none"
                    >
                      {ASSET_TYPES.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Quality score + detection info */}
                  <div className="flex items-center gap-3 mb-3 text-xs">
                    {asset.quality_score !== null && (
                      <span
                        className={`px-2 py-1 rounded font-medium ${getQualityBadgeColor(asset.quality_score)}`}
                      >
                        Quality: {asset.quality_score.toFixed(1)}
                      </span>
                    )}
                    {asset.detection_class && (
                      <span className="text-gray-500">
                        {asset.detection_class}{" "}
                        {asset.detection_confidence !== null &&
                          `(${(asset.detection_confidence * 100).toFixed(0)}%)`}
                      </span>
                    )}
                    <span
                      className={`px-2 py-1 rounded ${
                        asset.source === "uploaded"
                          ? "bg-purple-900 text-purple-300"
                          : asset.source === "video_frame"
                            ? "bg-purple-900 text-purple-300"
                            : "bg-gray-800 text-gray-400"
                      }`}
                    >
                      {asset.source === "video_frame" ? "video frame" : asset.source}
                    </span>
                    {asset.is_face_crop && (
                      <span className="px-2 py-1 rounded bg-indigo-900 text-indigo-300">
                        face
                      </span>
                    )}
                  </div>

                  {/* Reverse prompt */}
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs font-medium text-gray-400">
                        Reverse Prompt
                      </label>
                      <button
                        onClick={() =>
                          toggleFieldEdit(asset.asset_id, "reverse_prompt")
                        }
                        className="text-xs text-blue-400 hover:text-blue-300"
                      >
                        {isFieldEditing(asset.asset_id, "reverse_prompt")
                          ? "Save"
                          : "Edit"}
                      </button>
                    </div>
                    {isFieldEditing(asset.asset_id, "reverse_prompt") ? (
                      <textarea
                        value={asset.reverse_prompt || ""}
                        onChange={(e) =>
                          handleAssetUpdate(asset.asset_id, {
                            reverse_prompt: e.target.value,
                          })
                        }
                        rows={3}
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
                      />
                    ) : (
                      <p className="text-sm text-gray-300">
                        {asset.reverse_prompt || (
                          <span className="text-gray-600 italic">
                            No reverse prompt
                          </span>
                        )}
                      </p>
                    )}
                  </div>

                  {/* Visual description */}
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-xs font-medium text-gray-400">
                        Visual Description
                      </label>
                      <button
                        onClick={() =>
                          toggleFieldEdit(asset.asset_id, "visual_description")
                        }
                        className="text-xs text-blue-400 hover:text-blue-300"
                      >
                        {isFieldEditing(asset.asset_id, "visual_description")
                          ? "Save"
                          : "Edit"}
                      </button>
                    </div>
                    {isFieldEditing(asset.asset_id, "visual_description") ? (
                      <textarea
                        value={asset.visual_description || ""}
                        onChange={(e) =>
                          handleAssetUpdate(asset.asset_id, {
                            visual_description: e.target.value,
                          })
                        }
                        rows={2}
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
                      />
                    ) : (
                      <p className="text-sm text-gray-300">
                        {asset.visual_description || (
                          <span className="text-gray-600 italic">
                            No visual description
                          </span>
                        )}
                      </p>
                    )}
                  </div>

                  {/* Action buttons */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleReprocess(asset.asset_id)}
                      disabled={reprocessingAssets.has(asset.asset_id)}
                      className="text-xs px-3 py-1 rounded bg-yellow-900 text-yellow-300 hover:bg-yellow-800 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {reprocessingAssets.has(asset.asset_id)
                        ? "Processing..."
                        : "Re-process"}
                    </button>
                    <button
                      onClick={() => handleAssetDelete(asset.asset_id)}
                      className="text-xs px-3 py-1 rounded bg-red-900 text-red-300 hover:bg-red-800"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer actions */}
        <div className="flex justify-between pt-6 border-t border-gray-800">
          <button
            onClick={handleReprocessAll}
            disabled={processing}
            className="text-sm px-4 py-2 rounded bg-yellow-900 text-yellow-300 hover:bg-yellow-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {processing ? "Processing..." : "Reprocess All"}
          </button>
          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="text-gray-400 hover:text-gray-300 px-4 py-2 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => manifest?.manifest_id && onSaved(manifest.manifest_id)}
              className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-2 text-sm font-medium"
            >
              Done
            </button>
          </div>
        </div>
      </>
    );
  };

  return (
    <div>
      {/* Back link */}
      <button
        onClick={onCancel}
        className="text-sm text-gray-400 hover:text-gray-300 mb-6"
      >
        ← Back to Library
      </button>

      {/* Error banner */}
      {error && (
        <div className="mb-6 rounded-lg bg-red-900/30 border border-red-700 px-4 py-3 flex items-start justify-between">
          <p className="text-sm text-red-300">{error}</p>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-300 ml-4"
          >
            ✕
          </button>
        </div>
      )}

      {/* Render current stage */}
      {currentStage === 1 && renderStage1()}
      {currentStage === 2 && renderStage2()}
      {currentStage === 3 && renderStage3()}
    </div>
  );
}
