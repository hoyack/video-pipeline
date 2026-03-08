import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useLocation } from "wouter";
import type { LibraryProp, LibraryPropRef, EnabledModelsResponse } from "../api/types.ts";
import {
  getLibraryProp,
  updateLibraryProp,
  deleteLibraryProp,
  uploadLibraryPropRef,
  deleteLibraryPropRef,
  generatePropMetadata,
  generatePropImage,
  getEnabledModels,
} from "../api/client.ts";
import { IMAGE_MODELS, REFERENCE_IMAGE_MODELS } from "../lib/constants.ts";

interface PropLibraryDetailProps {
  propId: string;
}

type Tab = "overview" | "refs";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "refs", label: "Reference Images" },
];

export function PropLibraryDetail({ propId }: PropLibraryDetailProps) {
  const [, navigate] = useLocation();
  const [prop, setProp] = useState<LibraryProp | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getLibraryProp(propId);
      setProp(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [propId]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async () => {
    if (!confirm("Delete this prop? This cannot be undone.")) return;
    try {
      await deleteLibraryProp(propId);
      navigate("/asset-library");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!prop) {
    return (
      <div className="text-center py-12">
        <p className="text-sm text-red-400">{error ?? "Prop not found"}</p>
        <button onClick={() => navigate("/asset-library")} className="mt-4 text-sm text-blue-400 hover:text-blue-300">
          Back to Asset Library
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/asset-library")}
            className="text-gray-400 hover:text-gray-200 text-sm"
          >
            Props /
          </button>
          <h1 className="text-xl font-bold text-white">{prop.name}</h1>
        </div>
        <button
          onClick={handleDelete}
          className="text-xs px-3 py-1.5 rounded bg-red-900 text-red-300 hover:bg-red-800"
        >
          Delete
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded bg-red-900/30 border border-red-700 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-6 border-b border-gray-800 pb-2">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
              activeTab === tab.id
                ? "text-blue-400 border-b-2 border-blue-400"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex gap-6">
        {/* Sidebar preview */}
        <div className="w-48 flex-shrink-0">
          {prop.refs.length > 0 ? (
            <img
              src={prop.refs.find((r) => r.is_primary)?.image_url ?? prop.refs[0].image_url}
              alt={prop.name}
              className="w-full rounded-lg border border-gray-700 object-cover aspect-square"
            />
          ) : (
            <div className="w-full rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center aspect-square">
              <span className="text-gray-600 text-2xl">PROP</span>
            </div>
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          {activeTab === "overview" && (
            <OverviewTab
              prop={prop}
              onRefresh={load}
              onError={(msg) => setError(msg)}
            />
          )}
          {activeTab === "refs" && (
            <RefsTab
              prop={prop}
              onRefresh={load}
              onError={(msg) => setError(msg)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({
  prop,
  onRefresh,
  onError,
}: {
  prop: LibraryProp;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const [name, setName] = useState(prop.name);
  const [description, setDescription] = useState(prop.description ?? "");
  const [appearancePrompt, setAppearancePrompt] = useState(prop.appearance_prompt ?? "");
  const [promptTags, setPromptTags] = useState((prop.prompt_tags ?? []).join(", "));
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const result = await generatePropMetadata(prop.id);
      setDescription(result.description);
      setAppearancePrompt(result.appearance_prompt);
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    setName(prop.name);
    setDescription(prop.description ?? "");
    setAppearancePrompt(prop.appearance_prompt ?? "");
    setPromptTags((prop.prompt_tags ?? []).join(", "));
  }, [prop]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const parseTags = (s: string) => {
        const tags = s.split(",").map((t) => t.trim()).filter(Boolean);
        return tags.length > 0 ? tags : undefined;
      };
      await updateLibraryProp(prop.id, {
        name: name.trim() || prop.name,
        description: description.trim() || undefined,
        appearance_prompt: appearancePrompt.trim() || undefined,
        prompt_tags: parseTags(promptTags),
      });
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const inputClass =
    "w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none";

  const hasRefs = prop.refs.length > 0;

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} className={inputClass} />
      </div>
      <div>
        <button
          onClick={handleGenerate}
          disabled={generating || !hasRefs}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-700 text-purple-100 hover:bg-purple-600 disabled:opacity-50 transition-colors"
        >
          {generating ? "Generating..." : "Auto-Generate Description & Appearance Prompt"}
        </button>
        {!hasRefs && (
          <p className="text-xs text-gray-500 mt-1">Upload or generate a reference image first</p>
        )}
        {generating && (
          <p className="text-xs text-gray-500 mt-1">Analyzing reference image with vision model...</p>
        )}
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Description</label>
        <textarea rows={3} value={description} onChange={(e) => setDescription(e.target.value)} className={inputClass} placeholder="Physical description of the prop..." />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Appearance Prompt</label>
        <textarea rows={3} value={appearancePrompt} onChange={(e) => setAppearancePrompt(e.target.value)} className={inputClass} placeholder="Text-to-image prompt to recreate this prop..." />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Prompt Tags</label>
        <input type="text" value={promptTags} onChange={(e) => setPromptTags(e.target.value)} className={inputClass} placeholder="tag1, tag2, ..." />
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs text-gray-500">
          Bindings: {prop.binding_count ?? 0}
        </span>
      </div>
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Refs Tab
// ============================================================================

type PropPromptSource = "description" | "appearance_prompt" | "custom";

function RefsTab({
  prop,
  onRefresh,
  onError,
}: {
  prop: LibraryProp;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [showGenerate, setShowGenerate] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [promptSource, setPromptSource] = useState<PropPromptSource>("appearance_prompt");
  const [customPrompt, setCustomPrompt] = useState("");
  const [modelSettings, setModelSettings] = useState<EnabledModelsResponse | null>(null);
  const [selectedImageModel, setSelectedImageModel] = useState("");
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [selectedRefId, setSelectedRefId] = useState<string | null>(null);

  const supportsRefs = REFERENCE_IMAGE_MODELS.has(selectedImageModel);

  useEffect(() => {
    getEnabledModels()
      .then((ms) => {
        setModelSettings(ms);
        if (ms.default_image_model) setSelectedImageModel(ms.default_image_model);
        else if (ms.enabled_image_models?.length) setSelectedImageModel(ms.enabled_image_models[0]);
        else setSelectedImageModel(IMAGE_MODELS[0].id);
      })
      .catch(() => { setSelectedImageModel(IMAGE_MODELS[0].id); });
  }, []);

  const filteredImageModels = useMemo(() => {
    if (!modelSettings?.enabled_image_models) return IMAGE_MODELS;
    const enabled = new Set(modelSettings.enabled_image_models);
    return IMAGE_MODELS.filter((m) => enabled.has(m.id));
  }, [modelSettings]);

  const getPromptText = (): string => {
    switch (promptSource) {
      case "description":
        return prop.description ?? "";
      case "appearance_prompt":
        return prop.appearance_prompt ?? "";
      case "custom":
        return customPrompt;
    }
  };

  const handleGenerate = async () => {
    const prompt = getPromptText().trim();
    if (!prompt) {
      onError(
        promptSource === "custom"
          ? "Enter a prompt first."
          : `No ${promptSource === "description" ? "description" : "appearance prompt"} set. Generate metadata first or enter a custom prompt.`,
      );
      return;
    }
    setGenerating(true);
    try {
      await generatePropImage(prop.id, prompt, selectedImageModel || undefined, selectedRefId ?? undefined);
      onRefresh();
      setShowGenerate(false);
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadLibraryPropRef(prop.id, file);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDeleteRef = async (ref: LibraryPropRef) => {
    if (!confirm("Delete this reference image?")) return;
    try {
      await deleteLibraryPropRef(ref.id);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const inputClass =
    "w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-300">
          Reference Images ({prop.refs.length})
        </h4>
        <div className="flex gap-2">
          <button
            onClick={() => setShowGenerate(!showGenerate)}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-700 text-purple-100 hover:bg-purple-600 transition-colors"
          >
            Generate Image
          </button>
          <label className="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 cursor-pointer">
            {uploading ? "Uploading..." : "Upload"}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleUpload}
              disabled={uploading}
            />
          </label>
        </div>
      </div>

      {showGenerate && (
        <div className="rounded-lg border border-purple-800/50 bg-purple-950/20 p-4 space-y-3">
          <h5 className="text-xs font-medium text-purple-300">Generate from:</h5>
          <div className="flex gap-2">
            {([
              { id: "appearance_prompt" as const, label: "Appearance Prompt" },
              { id: "description" as const, label: "Description" },
              { id: "custom" as const, label: "New Prompt" },
            ]).map((opt) => (
              <button
                key={opt.id}
                onClick={() => setPromptSource(opt.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                  promptSource === opt.id
                    ? "bg-purple-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:text-gray-200"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {promptSource === "custom" ? (
            <textarea
              rows={3}
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              className={inputClass}
              placeholder="Enter a prompt to generate the prop image..."
            />
          ) : (
            <div className="rounded bg-gray-900/60 border border-gray-700 px-3 py-2 text-xs text-gray-400 max-h-24 overflow-y-auto">
              {getPromptText() || (
                <span className="italic text-gray-600">
                  No {promptSource === "description" ? "description" : "appearance prompt"} set yet.
                </span>
              )}
            </div>
          )}

          <div>
            <h5 className="text-xs font-medium text-purple-300 mb-2">Image model:</h5>
            <div className="flex flex-wrap gap-2">
              {filteredImageModels.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedImageModel(m.id)}
                  disabled={generating}
                  className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                    selectedImageModel === m.id
                      ? "bg-purple-600 text-white"
                      : "bg-gray-800 text-gray-400 hover:text-gray-200"
                  } disabled:opacity-50`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {supportsRefs && prop.refs.length > 0 && (
            <p className="text-xs text-purple-300">
              {selectedRefId
                ? "Reference image selected below — it will be used as input."
                : "Click an image below to use it as a reference for generation."}
            </p>
          )}

          <div className="flex items-center gap-3">
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="px-4 py-2 text-xs font-medium rounded-lg bg-purple-600 text-white hover:bg-purple-500 disabled:opacity-50 transition-colors"
            >
              {generating ? "Generating..." : "Generate"}
            </button>
            {generating && (
              <span className="text-xs text-gray-500">This may take a few seconds...</span>
            )}
          </div>
        </div>
      )}

      {prop.refs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-500">No reference images yet</p>
          <p className="text-xs text-gray-600 mt-1">Upload images to define this prop's look</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {prop.refs.map((ref) => {
            const isSelectable = showGenerate && supportsRefs;
            const isSelected = isSelectable && selectedRefId === ref.id;
            return (
              <div
                key={ref.id}
                className={`relative group rounded-lg overflow-hidden border-2 transition-colors ${
                  isSelected
                    ? "border-purple-500 ring-1 ring-purple-500/50"
                    : "border-gray-700"
                } ${isSelectable ? "cursor-pointer" : ""}`}
                onClick={() => {
                  if (isSelectable) {
                    setSelectedRefId(isSelected ? null : ref.id);
                  } else {
                    setLightboxUrl(ref.image_url);
                  }
                }}
              >
                <img
                  src={ref.image_url}
                  alt={ref.label ?? "Prop ref"}
                  className="w-full aspect-square object-cover"
                />
                {isSelected && (
                  <div className="absolute inset-0 bg-purple-500/20 flex items-center justify-center pointer-events-none">
                    <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center shadow-lg">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  </div>
                )}
                {ref.is_primary && (
                  <span className="absolute top-1.5 left-1.5 text-[10px] px-1.5 py-0.5 rounded bg-blue-600 text-white">
                    Primary
                  </span>
                )}
                {isSelectable && !isSelected && (
                  <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-purple-500/10 transition-opacity pointer-events-none" />
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); handleDeleteRef(ref); }}
                  className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 text-[10px] px-1.5 py-0.5 rounded bg-red-900/80 text-red-300 hover:bg-red-800 transition-opacity"
                >
                  Delete
                </button>
              </div>
            );
          })}
        </div>
      )}

      {lightboxUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 cursor-pointer"
          onClick={() => setLightboxUrl(null)}
        >
          <img
            src={lightboxUrl}
            alt="Full size"
            className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
