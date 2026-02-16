import { useState, useEffect } from "react";
import type {
  ManifestDetail,
  AssetResponse,
  UpdateAssetRequest,
} from "../api/types.ts";
import {
  createManifest,
  getManifestDetail,
  updateManifest,
  createAsset,
  updateAsset,
  deleteAsset,
  uploadAssetImage,
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
  const [error, setError] = useState<string | null>(null);
  const [uploadingAssets, setUploadingAssets] = useState<Set<string>>(
    new Set()
  );

  const isNewManifest = !manifestId;

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
        })
        .catch((err) => {
          setError(`Failed to load manifest: ${err.message}`);
        });
    }
  }, [manifestId]);

  const handleFilesSelected = async (files: File[]) => {
    if (files.length === 0) return;

    // Lazy create manifest on first upload if creating new
    let currentManifestId = manifestId || manifest?.manifest_id;

    if (!currentManifestId) {
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
        currentManifestId = newManifest.manifest_id;
        setManifest(newManifest as unknown as ManifestDetail);
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        setError(`Failed to create manifest: ${errorMessage}`);
        return;
      }
    }

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

      {/* Upload zone */}
      <div className="mb-8">
        <AssetUploader
          onFilesSelected={handleFilesSelected}
          disabled={saving}
        />
      </div>

      {/* Asset list */}
      <div className="mb-8">
        <h3 className="text-lg font-medium text-gray-200 mb-4">
          Assets ({assets.length})
        </h3>
        {assets.length === 0 ? (
          <p className="text-sm text-gray-500 text-center py-8">
            No assets yet. Drop images above to add assets.
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
        <button
          onClick={handleSave}
          disabled={saving || !name.trim()}
          className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? "Saving..." : "Save Draft"}
        </button>
      </div>
    </div>
  );
}
