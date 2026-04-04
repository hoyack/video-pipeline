import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useLocation } from "wouter";
import type { LibrarySet, LibrarySetRef, EnabledModelsResponse } from "../api/types.ts";
import {
  getLibrarySet,
  updateLibrarySet,
  deleteLibrarySet,
  uploadLibrarySetRef,
  deleteLibrarySetRef,
  generateLibrarySetMetadata,
  generateLibrarySetImage,
  getEnabledModels,
} from "../api/client.ts";
import { IMAGE_MODELS, REFERENCE_IMAGE_MODELS } from "../lib/constants.ts";

interface LibrarySetDetailProps {
  setId: string;
}

type Tab = "overview" | "refs";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "refs", label: "Reference Images" },
];

export function LibrarySetDetail({ setId }: LibrarySetDetailProps) {
  const [, navigate] = useLocation();
  const [libSet, setLibSet] = useState<LibrarySet | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getLibrarySet(setId);
      setLibSet(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [setId]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async () => {
    if (!confirm("Delete this set? This cannot be undone.")) return;
    try {
      await deleteLibrarySet(setId);
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

  if (!libSet) {
    return (
      <div className="text-center py-12">
        <p className="text-sm text-red-400">{error ?? "Set not found"}</p>
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
            Sets /
          </button>
          <h1 className="text-xl font-bold text-white">{libSet.name}</h1>
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
          {libSet.refs.length > 0 ? (
            <img
              src={libSet.refs.find((r) => r.is_primary)?.image_url ?? libSet.refs[0].image_url}
              alt={libSet.name}
              className="w-full rounded-lg border border-gray-700 object-cover aspect-square"
            />
          ) : (
            <div className="w-full rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center aspect-square">
              <span className="text-gray-600 text-2xl">SET</span>
            </div>
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          {activeTab === "overview" && (
            <OverviewTab
              libSet={libSet}
              onRefresh={load}
              onError={(msg) => setError(msg)}
            />
          )}
          {activeTab === "refs" && (
            <RefsTab
              libSet={libSet}
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
  libSet,
  onRefresh,
  onError,
}: {
  libSet: LibrarySet;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const [name, setName] = useState(libSet.name);
  const [description, setDescription] = useState(libSet.description ?? "");
  const [reversePrompt, setReversePrompt] = useState(libSet.reverse_prompt ?? "");
  const [lightingNotes, setLightingNotes] = useState(libSet.lighting_notes ?? "");
  const [promptTags, setPromptTags] = useState((libSet.prompt_tags ?? []).join(", "));
  const [styleTags, setStyleTags] = useState((libSet.style_tags ?? []).join(", "));
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const result = await generateLibrarySetMetadata(libSet.id);
      setDescription(result.description);
      setReversePrompt(result.reverse_prompt);
      setLightingNotes(result.lighting_notes);
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    setName(libSet.name);
    setDescription(libSet.description ?? "");
    setReversePrompt(libSet.reverse_prompt ?? "");
    setLightingNotes(libSet.lighting_notes ?? "");
    setPromptTags((libSet.prompt_tags ?? []).join(", "));
    setStyleTags((libSet.style_tags ?? []).join(", "));
  }, [libSet]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const parseTags = (s: string) => {
        const tags = s.split(",").map((t) => t.trim()).filter(Boolean);
        return tags.length > 0 ? tags : null;
      };
      await updateLibrarySet(libSet.id, {
        name: name.trim() || libSet.name,
        description: description.trim() || undefined,
        reverse_prompt: reversePrompt.trim() || undefined,
        lighting_notes: lightingNotes.trim() || undefined,
        prompt_tags: parseTags(promptTags) ?? undefined,
        style_tags: parseTags(styleTags) ?? undefined,
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

  const hasRefs = libSet.refs.length > 0;

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} className={inputClass} />
      </div>
      {hasRefs && (
        <div>
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-700 text-purple-100 hover:bg-purple-600 disabled:opacity-50 transition-colors"
          >
            {generating ? "Generating..." : "Auto-Generate Description, Prompt & Lighting"}
          </button>
          {generating && (
            <p className="text-xs text-gray-500 mt-1">Analyzing reference image with vision model...</p>
          )}
        </div>
      )}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Description</label>
        <textarea rows={3} value={description} onChange={(e) => setDescription(e.target.value)} className={inputClass} placeholder="Visual description of the set..." />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Reverse Prompt</label>
        <textarea rows={2} value={reversePrompt} onChange={(e) => setReversePrompt(e.target.value)} className={inputClass} placeholder="Prompt used to recreate this set..." />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Lighting Notes</label>
        <input type="text" value={lightingNotes} onChange={(e) => setLightingNotes(e.target.value)} className={inputClass} placeholder="e.g. warm golden hour, neon-lit..." />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Prompt Tags</label>
          <input type="text" value={promptTags} onChange={(e) => setPromptTags(e.target.value)} className={inputClass} placeholder="tag1, tag2, ..." />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Style Tags</label>
          <input type="text" value={styleTags} onChange={(e) => setStyleTags(e.target.value)} className={inputClass} placeholder="cyberpunk, noir, ..." />
        </div>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs text-gray-500">
          Bindings: {libSet.binding_count ?? 0}
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

type PromptSource = "description" | "reverse_prompt" | "custom";

function RefsTab({
  libSet,
  onRefresh,
  onError,
}: {
  libSet: LibrarySet;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [showGenerate, setShowGenerate] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [promptSource, setPromptSource] = useState<PromptSource>("reverse_prompt");
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
        return libSet.description ?? "";
      case "reverse_prompt":
        return libSet.reverse_prompt ?? "";
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
          : `No ${promptSource === "description" ? "description" : "reverse prompt"} set. Generate metadata first or enter a custom prompt.`,
      );
      return;
    }
    setGenerating(true);
    try {
      await generateLibrarySetImage(libSet.id, prompt, selectedImageModel || undefined, selectedRefId ?? undefined);
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
      await uploadLibrarySetRef(libSet.id, file);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDeleteRef = async (ref: LibrarySetRef) => {
    if (!confirm("Delete this reference image?")) return;
    try {
      await deleteLibrarySetRef(ref.id);
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
          Reference Images ({libSet.refs.length})
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
              { id: "reverse_prompt" as const, label: "Reverse Prompt" },
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
              placeholder="Enter a prompt to generate the set image..."
            />
          ) : (
            <div className="rounded bg-gray-900/60 border border-gray-700 px-3 py-2 text-xs text-gray-400 max-h-24 overflow-y-auto">
              {getPromptText() || (
                <span className="italic text-gray-600">
                  No {promptSource === "description" ? "description" : "reverse prompt"} set yet.
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

          {supportsRefs && libSet.refs.length > 0 && (
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

      {libSet.refs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-500">No reference images yet</p>
          <p className="text-xs text-gray-600 mt-1">Upload images to define this set's look</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {libSet.refs.map((ref) => {
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
                  alt={ref.label ?? "Set ref"}
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
