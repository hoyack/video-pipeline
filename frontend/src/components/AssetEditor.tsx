import { useState, useEffect } from "react";
import type { AssetResponse, UpdateAssetRequest } from "../api/types.ts";

interface AssetEditorProps {
  asset: AssetResponse;
  imageFile?: File;
  onUpdate: (assetId: string, updates: UpdateAssetRequest) => void;
  onDelete: (assetId: string) => void;
  isUploading?: boolean;
}

const ASSET_TYPES = [
  "CHARACTER",
  "OBJECT",
  "ENVIRONMENT",
  "PROP",
  "STYLE",
  "OTHER",
];

export function AssetEditor({
  asset,
  imageFile,
  onUpdate,
  onDelete,
  isUploading = false,
}: AssetEditorProps) {
  const [localName, setLocalName] = useState(asset.name);
  const [localType, setLocalType] = useState(asset.asset_type);
  const [localDescription, setLocalDescription] = useState(asset.description || "");
  const [localTags, setLocalTags] = useState(
    asset.user_tags ? asset.user_tags.join(", ") : ""
  );
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Create preview URL for local file
  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [imageFile]);

  const handleNameBlur = () => {
    if (localName !== asset.name) {
      onUpdate(asset.asset_id, { name: localName });
    }
  };

  const handleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = e.target.value;
    setLocalType(newType);
    onUpdate(asset.asset_id, { asset_type: newType });
  };

  const handleDescriptionBlur = () => {
    if (localDescription !== (asset.description || "")) {
      onUpdate(asset.asset_id, { description: localDescription });
    }
  };

  const handleTagsBlur = () => {
    const tagsArray = localTags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const currentTags = asset.user_tags || [];
    if (JSON.stringify(tagsArray) !== JSON.stringify(currentTags)) {
      onUpdate(asset.asset_id, { user_tags: tagsArray });
    }
  };

  const handleDelete = () => {
    if (confirm(`Delete asset "${asset.name}"?`)) {
      onDelete(asset.asset_id);
    }
  };

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-3 flex items-start gap-4">
      {/* Thumbnail */}
      <div className="w-24 h-24 flex-shrink-0 rounded overflow-hidden bg-gray-800">
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={asset.name}
            className="w-full h-full object-cover"
          />
        ) : asset.reference_image_url ? (
          <div className="w-full h-full flex items-center justify-center text-gray-600">
            <span className="text-4xl">🖼️</span>
          </div>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-600">
            <span className="text-4xl">📷</span>
          </div>
        )}
      </div>

      {/* Metadata fields */}
      <div className="flex-1 flex flex-col gap-2">
        {/* Name */}
        <input
          type="text"
          value={localName}
          onChange={(e) => setLocalName(e.target.value)}
          onBlur={handleNameBlur}
          className="bg-transparent border-b border-gray-700 text-gray-200 text-sm w-full focus:border-blue-500 outline-none pb-1"
          placeholder="Asset name"
        />

        {/* Type */}
        <select
          value={localType}
          onChange={handleTypeChange}
          className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200 text-sm focus:border-blue-500 outline-none"
        >
          {ASSET_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>

        {/* Description */}
        <textarea
          value={localDescription}
          onChange={(e) => setLocalDescription(e.target.value)}
          onBlur={handleDescriptionBlur}
          rows={2}
          className="bg-transparent border-b border-gray-700 text-gray-200 text-sm w-full focus:border-blue-500 outline-none resize-none"
          placeholder="Description (optional)"
        />

        {/* Tags */}
        <div>
          {asset.user_tags && asset.user_tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-1">
              {asset.user_tags.map((tag, idx) => (
                <span
                  key={idx}
                  className="text-xs bg-gray-800 rounded px-2 py-0.5 text-gray-400"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
          <input
            type="text"
            value={localTags}
            onChange={(e) => setLocalTags(e.target.value)}
            onBlur={handleTagsBlur}
            className="bg-transparent border-b border-gray-700 text-gray-200 text-sm w-full focus:border-blue-500 outline-none"
            placeholder="Tags (comma-separated)"
          />
        </div>
      </div>

      {/* Right side */}
      <div className="flex-shrink-0 flex flex-col items-end gap-2">
        {/* Manifest tag badge */}
        <span className="text-xs font-mono bg-gray-800 rounded px-2 py-0.5 text-gray-400">
          {asset.manifest_tag}
        </span>

        {/* Upload status */}
        {isUploading && (
          <span className="text-xs text-blue-400">Uploading...</span>
        )}

        {/* Delete button */}
        <button
          onClick={handleDelete}
          className="text-sm text-red-400 hover:text-red-300 transition-colors"
        >
          Remove
        </button>
      </div>
    </div>
  );
}
