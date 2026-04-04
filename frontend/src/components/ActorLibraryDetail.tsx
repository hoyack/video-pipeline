import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useLocation } from "wouter";
import type {
  Actor,
  ActorRef,
  ActorVoiceProfile,
  ActorWardrobePreset,
  EnabledModelsResponse,
} from "../api/types.ts";
import {
  getActor,
  updateActor,
  deleteActor,
  uploadActorLibraryRef,
  deleteActorLibraryRef,
  createActorVoiceProfile,
  updateActorVoiceProfile,
  deleteActorVoiceProfile,
  testActorVoiceProfile,
  createActorWardrobePreset,
  updateActorWardrobePreset,
  deleteActorWardrobePreset,
  generateWardrobeImage,
  deleteWardrobeImage,
  uploadWardrobePresetImage,
  getEnabledModels,
  generateActorMetadata,
  generateActorImage,
  trainActorLora,
  getActorLoraStatus,
} from "../api/client.ts";
import { IMAGE_MODELS, REFERENCE_IMAGE_MODELS, STYLE_OPTIONS } from "../lib/constants.ts";

interface ActorLibraryDetailProps {
  actorId: string;
}

type Tab = "overview" | "refs" | "voice" | "wardrobe" | "usage";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "refs", label: "Appearance Refs" },
  { id: "voice", label: "Voice Profiles" },
  { id: "wardrobe", label: "Wardrobe Presets" },
  { id: "usage", label: "Usage" },
];

export function ActorLibraryDetail({ actorId }: ActorLibraryDetailProps) {
  const [, navigate] = useLocation();
  const [actor, setActor] = useState<Actor | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getActor(actorId);
      setActor(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [actorId]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async () => {
    if (!confirm("Delete this actor? This cannot be undone.")) return;
    try {
      await deleteActor(actorId);
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

  if (!actor) {
    return (
      <div className="text-center py-12">
        <p className="text-sm text-red-400">{error ?? "Actor not found"}</p>
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
            Actor /
          </button>
          <h1 className="text-xl font-bold text-white">{actor.name}</h1>
        </div>
        <button
          onClick={handleDelete}
          className="px-3 py-1.5 text-xs rounded-lg bg-red-900 text-red-300 hover:bg-red-800"
        >
          Delete Actor
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-red-900/30 border border-red-700 px-4 py-2 flex items-center justify-between">
          <p className="text-sm text-red-300">{error}</p>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 ml-4 text-xs">dismiss</button>
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 mb-4 border-b border-gray-800 pb-2">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
              activeTab === tab.id
                ? "bg-gray-800 text-white border-b-2 border-blue-500"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {tab.label}
            {tab.id === "refs" && actor.refs.length > 0 && (
              <span className="ml-1 text-[10px] bg-gray-700 rounded-full px-1.5">{actor.refs.length}</span>
            )}
            {tab.id === "voice" && actor.voice_profiles.length > 0 && (
              <span className="ml-1 text-[10px] bg-gray-700 rounded-full px-1.5">{actor.voice_profiles.length}</span>
            )}
            {tab.id === "wardrobe" && actor.wardrobe_presets.length > 0 && (
              <span className="ml-1 text-[10px] bg-gray-700 rounded-full px-1.5">{actor.wardrobe_presets.length}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="rounded-lg border border-gray-800 bg-gray-900/50 p-4">
        {activeTab === "overview" && (
          <OverviewTab actor={actor} onUpdate={(data) => {
            updateActor(actorId, data)
              .then(() => load())
              .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
          }} onError={(msg) => setError(msg)} onRefresh={load} />
        )}
        {activeTab === "refs" && (
          <RefsTab actor={actor} onRefresh={load} onError={(msg) => setError(msg)} />
        )}
        {activeTab === "voice" && (
          <VoiceProfilesTab actor={actor} onRefresh={load} onError={(msg) => setError(msg)} />
        )}
        {activeTab === "wardrobe" && (
          <WardrobeTab actor={actor} onRefresh={load} onError={(msg) => setError(msg)} />
        )}
        {activeTab === "usage" && (
          <UsageTab actor={actor} />
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({
  actor,
  onUpdate,
  onError,
  onRefresh,
}: {
  actor: Actor;
  onUpdate: (data: Partial<{ name: string; description: string; base_appearance_prompt: string; prompt_tags: string[] }>) => void;
  onError: (msg: string) => void;
  onRefresh: () => void;
}) {
  const [name, setName] = useState(actor.name);
  const [description, setDescription] = useState(actor.description ?? "");
  const [baseAppearance, setBaseAppearance] = useState(actor.base_appearance_prompt ?? "");
  const [promptTags, setPromptTags] = useState(actor.prompt_tags?.join(", ") ?? "");
  const [newTag, setNewTag] = useState("");
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [loraTraining, setLoraTraining] = useState(false);
  const [loraError, setLoraError] = useState<string | null>(null);

  // Poll for LoRA training status when in progress
  useEffect(() => {
    const status = actor.lora_training_status;
    if (status !== "QUEUED" && status !== "TRAINING") return;

    const interval = setInterval(async () => {
      try {
        const result = await getActorLoraStatus(actor.id);
        if (result.status === "COMPLETED" || result.status === "FAILED") {
          clearInterval(interval);
          onRefresh();
        }
      } catch {
        // Silently continue polling on error
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [actor.id, actor.lora_training_status, onRefresh]);

  const handleTrainLora = async () => {
    setLoraTraining(true);
    setLoraError(null);
    try {
      await trainActorLora(actor.id);
      onRefresh();
    } catch (e: unknown) {
      setLoraError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoraTraining(false);
    }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const result = await generateActorMetadata(actor.id);
      setDescription(result.description);
      setBaseAppearance(result.base_appearance_prompt);
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    setName(actor.name);
    setDescription(actor.description ?? "");
    setBaseAppearance(actor.base_appearance_prompt ?? "");
    setPromptTags(actor.prompt_tags?.join(", ") ?? "");
  }, [actor]);

  const parsedTags = promptTags ? promptTags.split(",").map((t) => t.trim()).filter(Boolean) : [];

  const handleRemoveTag = (tag: string) => {
    const updated = parsedTags.filter((t) => t !== tag);
    setPromptTags(updated.join(", "));
  };

  const handleAddTag = () => {
    if (!newTag.trim()) return;
    const updated = [...parsedTags, newTag.trim()];
    setPromptTags(updated.join(", "));
    setNewTag("");
  };

  const handleSave = async () => {
    setSaving(true);
    onUpdate({
      name: name.trim(),
      description: description || undefined,
      base_appearance_prompt: baseAppearance || undefined,
      prompt_tags: parsedTags.length > 0 ? parsedTags : undefined,
    });
    // Small delay for UX feedback
    setTimeout(() => setSaving(false), 500);
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
        />
      </div>

      <div>
        <button
          onClick={handleGenerate}
          disabled={generating || actor.refs.length === 0}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-700 text-purple-100 hover:bg-purple-600 disabled:opacity-50 transition-colors"
        >
          {generating ? "Generating..." : "Auto-Generate Description & Appearance Prompt"}
        </button>
        {actor.refs.length === 0 && (
          <p className="text-xs text-gray-500 mt-1">Upload or generate a reference image first</p>
        )}
        {generating && (
          <p className="text-xs text-gray-500 mt-1">Analyzing reference image with vision model...</p>
        )}
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Actor description..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Base Appearance Prompt</label>
        <textarea
          value={baseAppearance}
          onChange={(e) => setBaseAppearance(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Visual appearance for prompting..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Prompt Tags</label>
        <div className="flex flex-wrap gap-1 mb-2">
          {parsedTags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded bg-blue-900/50 text-blue-300"
            >
              {tag}
              <button onClick={() => handleRemoveTag(tag)} className="text-blue-400 hover:text-blue-200">x</button>
            </span>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddTag()}
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
            placeholder="Add a tag..."
          />
          <button
            onClick={handleAddTag}
            disabled={!newTag.trim()}
            className="text-xs px-3 py-1.5 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-50"
          >
            Add
          </button>
        </div>
      </div>

      {/* LoRA Identity Model Section */}
      <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-gray-200">LoRA Identity Model</h4>
            <p className="text-xs text-gray-500 mt-0.5">
              Train a LoRA model from reference images to improve identity consistency in generated images.
            </p>
          </div>
          <LoraStatusBadge status={actor.lora_training_status} trainedAt={actor.lora_trained_at} />
        </div>

        {actor.lora_training_status === "COMPLETED" && actor.lora_trained_at && (
          <p className="text-xs text-gray-400">
            Last trained: {new Date(actor.lora_trained_at).toLocaleDateString()}
          </p>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={handleTrainLora}
            disabled={
              actor.refs.length < 5 ||
              loraTraining ||
              actor.lora_training_status === "QUEUED" ||
              actor.lora_training_status === "TRAINING"
            }
            className="px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loraTraining
              ? "Starting..."
              : actor.lora_training_status === "QUEUED" || actor.lora_training_status === "TRAINING"
              ? "Training in Progress..."
              : actor.lora_training_status === "COMPLETED"
              ? "Retrain Model"
              : "Train Identity Model"}
          </button>
          {actor.refs.length < 5 && (
            <span className="text-xs text-gray-500">
              Need at least 5 reference images ({actor.refs.length}/5)
            </span>
          )}
        </div>

        {loraError && (
          <p className="text-xs text-red-400">{loraError}</p>
        )}
      </div>

      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-1.5 text-sm rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// LoRA Status Badge
// ============================================================================

function LoraStatusBadge({
  status,
  trainedAt: _trainedAt,
}: {
  status: string | null;
  trainedAt: string | null;
}) {
  if (!status) {
    return (
      <span className="inline-flex items-center text-[10px] px-2 py-0.5 rounded-full bg-gray-700 text-gray-400">
        No Model
      </span>
    );
  }
  if (status === "QUEUED") {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-yellow-900/50 text-yellow-300">
        Queued...
      </span>
    );
  }
  if (status === "TRAINING") {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-yellow-900/50 text-yellow-300">
        <span className="w-3 h-3 border border-yellow-400 border-t-transparent rounded-full animate-spin" />
        Training...
      </span>
    );
  }
  if (status === "COMPLETED") {
    return (
      <span className="inline-flex items-center text-[10px] px-2 py-0.5 rounded-full bg-green-900/50 text-green-300">
        Model Ready
      </span>
    );
  }
  if (status === "FAILED") {
    return (
      <span className="inline-flex items-center text-[10px] px-2 py-0.5 rounded-full bg-red-900/50 text-red-300">
        Training Failed
      </span>
    );
  }
  return null;
}

// ============================================================================
// Appearance Refs Tab
// ============================================================================

type ActorPromptSource = "description" | "base_appearance_prompt" | "custom";

function RefsTab({
  actor,
  onRefresh,
  onError,
}: {
  actor: Actor;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [showGenerate, setShowGenerate] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [promptSource, setPromptSource] = useState<ActorPromptSource>("base_appearance_prompt");
  const [customPrompt, setCustomPrompt] = useState("");
  const [modelSettings, setModelSettings] = useState<EnabledModelsResponse | null>(null);
  const [selectedImageModel, setSelectedImageModel] = useState("");
  const [selectedRefId, setSelectedRefId] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState("");
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);

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
        return actor.description ?? "";
      case "base_appearance_prompt":
        return actor.base_appearance_prompt ?? "";
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
          : `No ${promptSource === "description" ? "description" : "base appearance prompt"} set. Generate metadata first or enter a custom prompt.`,
      );
      return;
    }
    setGenerating(true);
    try {
      await generateActorImage(actor.id, prompt, selectedImageModel || undefined, selectedRefId ?? undefined, selectedStyle || undefined);
      onRefresh();
      // Keep the panel open so the user can generate more images with the same settings.
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
      await uploadActorLibraryRef(actor.id, file);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDeleteRef = async (refId: string) => {
    if (!confirm("Delete this reference image?")) return;
    try {
      await deleteActorLibraryRef(refId);
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
          Reference Images ({actor.refs.length})
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
              { id: "base_appearance_prompt" as const, label: "Appearance Prompt" },
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
              placeholder="Enter a prompt to generate the actor image..."
            />
          ) : (
            <div className="rounded bg-gray-900/60 border border-gray-700 px-3 py-2 text-xs text-gray-400 max-h-24 overflow-y-auto">
              {getPromptText() || (
                <span className="italic text-gray-600">
                  No {promptSource === "description" ? "description" : "base appearance prompt"} set yet.
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

          <div>
            <h5 className="text-xs font-medium text-purple-300 mb-2">Style (optional):</h5>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setSelectedStyle("")}
                disabled={generating}
                className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                  !selectedStyle
                    ? "bg-purple-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:text-gray-200"
                } disabled:opacity-50`}
              >
                Default
              </button>
              {STYLE_OPTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => setSelectedStyle(s)}
                  disabled={generating}
                  className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${
                    selectedStyle === s
                      ? "bg-purple-600 text-white"
                      : "bg-gray-800 text-gray-400 hover:text-gray-200"
                  } disabled:opacity-50`}
                >
                  {s.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>

          {supportsRefs && actor.refs.length > 0 && (
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

      {actor.refs.length === 0 ? (
        <div className="flex items-center justify-center py-12 rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-500">No reference images yet. Upload photos above.</p>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {actor.refs.map((ref: ActorRef) => {
            const isSelectable = showGenerate && supportsRefs;
            const isSelected = isSelectable && selectedRefId === ref.id;
            return (
              <div
                key={ref.id}
                className={`relative group rounded-lg overflow-hidden border-2 transition-colors cursor-pointer ${
                  isSelected
                    ? "border-purple-500 ring-1 ring-purple-500/50"
                    : "border-gray-700"
                }`}
                onClick={() => setLightboxUrl(ref.image_url)}
              >
                <img
                  src={ref.image_url}
                  alt={ref.label ?? "Actor ref"}
                  className="w-full aspect-square object-cover"
                />
                {isSelected && (
                  <div
                    className="absolute inset-0 bg-purple-500/20 flex items-center justify-center cursor-pointer"
                    onClick={(e) => { e.stopPropagation(); setSelectedRefId(null); }}
                  >
                    <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center shadow-lg">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  </div>
                )}
                <div className="absolute top-1 right-1 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {isSelectable && !isSelected && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setSelectedRefId(ref.id); }}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-purple-800 text-purple-200 hover:bg-purple-700"
                    >
                      Use as ref
                    </button>
                  )}
                  {ref.is_primary && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-900 text-yellow-300">Primary</span>
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteRef(ref.id); }}
                    className="text-xs px-1.5 py-0.5 rounded bg-red-900/80 text-red-300 hover:bg-red-800"
                  >
                    x
                  </button>
                </div>
                {ref.label && (
                  <span className="absolute bottom-1 left-1 text-[10px] px-1.5 py-0.5 rounded bg-gray-900/80 text-gray-300">
                    {ref.label}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Lightbox */}
      {lightboxUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={() => setLightboxUrl(null)}
        >
          <img
            src={lightboxUrl}
            alt="Reference image"
            className="max-w-[90vw] max-h-[90vh] rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Voice Profiles Tab
// ============================================================================

function VoiceProfilesTab({
  actor,
  onRefresh,
  onError,
}: {
  actor: Actor;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [voiceId, setVoiceId] = useState("");
  const [adapterType, setAdapterType] = useState("ELEVENLABS");
  const [styleNotes, setStyleNotes] = useState("");
  const [saving, setSaving] = useState(false);

  const handleAdd = async () => {
    setSaving(true);
    try {
      await createActorVoiceProfile(actor.id, {
        voice_id: voiceId || undefined,
        adapter_type: adapterType,
        style_notes: styleNotes || undefined,
      });
      setShowAdd(false);
      setVoiceId("");
      setStyleNotes("");
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleUpdate = async (id: string) => {
    setSaving(true);
    try {
      await updateActorVoiceProfile(id, {
        voice_id: voiceId || undefined,
        adapter_type: adapterType,
        style_notes: styleNotes || undefined,
      });
      setEditingId(null);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this voice profile?")) return;
    try {
      await deleteActorVoiceProfile(id);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const startEdit = (vp: ActorVoiceProfile) => {
    setEditingId(vp.id);
    setVoiceId(vp.voice_id ?? "");
    setAdapterType(vp.adapter_type ?? "ELEVENLABS");
    setStyleNotes(vp.style_notes ?? "");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-300">Voice Profiles</h4>
        <button
          onClick={() => {
            setShowAdd(true);
            setVoiceId("");
            setStyleNotes("");
            getEnabledModels()
              .then((ms) => { if (ms.default_voice_id) setVoiceId(ms.default_voice_id); })
              .catch(() => {});
          }}
          className="text-xs px-3 py-1.5 rounded bg-blue-600 text-white hover:bg-blue-500"
        >
          + Add
        </button>
      </div>

      {showAdd && (
        <VoiceProfileForm
          voiceId={voiceId}
          setVoiceId={setVoiceId}
          adapterType={adapterType}
          setAdapterType={setAdapterType}
          styleNotes={styleNotes}
          setStyleNotes={setStyleNotes}
          saving={saving}
          onSave={handleAdd}
          onCancel={() => setShowAdd(false)}
        />
      )}

      {actor.voice_profiles.length === 0 && !showAdd && (
        <p className="text-xs text-gray-500 text-center py-6">No voice profiles yet</p>
      )}

      {actor.voice_profiles.map((vp) =>
        editingId === vp.id ? (
          <VoiceProfileForm
            key={vp.id}
            voiceId={voiceId}
            setVoiceId={setVoiceId}
            adapterType={adapterType}
            setAdapterType={setAdapterType}
            styleNotes={styleNotes}
            setStyleNotes={setStyleNotes}
            saving={saving}
            onSave={() => handleUpdate(vp.id)}
            onCancel={() => setEditingId(null)}
          />
        ) : (
          <VoiceProfileCard
            key={vp.id}
            vp={vp}
            onEdit={() => startEdit(vp)}
            onDelete={() => handleDelete(vp.id)}
            onRefresh={onRefresh}
            onError={onError}
          />
        )
      )}
    </div>
  );
}

function VoiceProfileCard({
  vp,
  onEdit,
  onDelete,
  onRefresh,
  onError,
}: {
  vp: ActorVoiceProfile;
  onEdit: () => void;
  onDelete: () => void;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const [testing, setTesting] = useState(false);
  const [testText, setTestText] = useState("Hello, this is a sample of my voice. How does it sound?");
  const [showTestInput, setShowTestInput] = useState(false);
  const [sampleUrl, setSampleUrl] = useState(vp.sample_url);

  const handleTest = async () => {
    if (!vp.voice_id) {
      onError("Voice ID not set. Edit the profile to add one.");
      return;
    }
    setTesting(true);
    try {
      const updated = await testActorVoiceProfile(vp.id, testText);
      setSampleUrl(updated.sample_url);
      setShowTestInput(false);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="rounded border border-gray-700 bg-gray-800/50 p-3 space-y-2">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-300 font-mono">{vp.voice_id ?? "No voice ID"}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-400">
              {vp.adapter_type ?? "ELEVENLABS"}
            </span>
          </div>
          {vp.style_notes && (
            <p className="text-xs text-gray-400 mt-1">{vp.style_notes}</p>
          )}
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setShowTestInput(!showTestInput)}
            disabled={!vp.voice_id || testing}
            className="text-xs px-2 py-1 rounded bg-green-900 text-green-300 hover:bg-green-800 disabled:opacity-50"
          >
            {testing ? "Generating..." : "Test Voice"}
          </button>
          <button
            onClick={onEdit}
            className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
          >
            Edit
          </button>
          <button
            onClick={onDelete}
            className="text-xs px-2 py-1 rounded bg-red-900 text-red-300 hover:bg-red-800"
          >
            Delete
          </button>
        </div>
      </div>

      {showTestInput && (
        <div className="flex gap-2">
          <input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleTest()}
            placeholder="Text to speak..."
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 focus:border-green-500 outline-none"
          />
          <button
            onClick={handleTest}
            disabled={testing || !testText.trim()}
            className="text-xs px-3 py-1.5 rounded bg-green-600 text-white hover:bg-green-500 disabled:opacity-50"
          >
            {testing ? "..." : "Generate"}
          </button>
        </div>
      )}

      {sampleUrl && (
        <audio controls preload="none" className="w-full h-8" src={sampleUrl} key={sampleUrl} />
      )}
    </div>
  );
}


function VoiceProfileForm({
  voiceId,
  setVoiceId,
  adapterType,
  setAdapterType,
  styleNotes,
  setStyleNotes,
  saving,
  onSave,
  onCancel,
}: {
  voiceId: string;
  setVoiceId: (v: string) => void;
  adapterType: string;
  setAdapterType: (v: string) => void;
  styleNotes: string;
  setStyleNotes: (v: string) => void;
  saving: boolean;
  onSave: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="rounded border border-blue-700 bg-gray-800/50 p-3 space-y-2">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Voice ID</label>
        <input
          type="text"
          value={voiceId}
          onChange={(e) => setVoiceId(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
          placeholder="ElevenLabs voice ID"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Adapter Type</label>
        <select
          value={adapterType}
          onChange={(e) => setAdapterType(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none"
        >
          <option value="ELEVENLABS">ELEVENLABS</option>
          <option value="GOOGLE_TTS">GOOGLE_TTS</option>
        </select>
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Style Notes</label>
        <textarea
          value={styleNotes}
          onChange={(e) => setStyleNotes(e.target.value)}
          rows={2}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Voice style direction..."
        />
      </div>
      <div className="flex gap-1 justify-end">
        <button onClick={onCancel} className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600">Cancel</button>
        <button onClick={onSave} disabled={saving} className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50">
          {saving ? "..." : "Save"}
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Wardrobe Presets Tab
// ============================================================================

function WardrobeTab({
  actor,
  onRefresh,
  onError,
}: {
  actor: Actor;
  onRefresh: () => void;
  onError: (msg: string) => void;
}) {
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [description, setDescription] = useState("");
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Image generation state
  const [genPresetId, setGenPresetId] = useState<string | null>(null);
  const [genRefId, setGenRefId] = useState("");
  const [genModel, setGenModel] = useState("");
  const [genExtraPrompt, setGenExtraPrompt] = useState("");
  const [genStyle, setGenStyle] = useState("");
  const [generating, setGenerating] = useState(false);

  // Image upload state
  const uploadRef = useRef<HTMLInputElement>(null);
  const [uploadPresetId, setUploadPresetId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const handleAdd = async () => {
    if (!label.trim()) return;
    setSaving(true);
    try {
      await createActorWardrobePreset(actor.id, {
        label: label.trim(),
        description: description || undefined,
      });
      setShowAdd(false);
      setLabel("");
      setDescription("");
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleUpdate = async (id: string) => {
    setSaving(true);
    try {
      await updateActorWardrobePreset(id, {
        label: label.trim(),
        description: description || undefined,
      });
      setEditingId(null);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this wardrobe preset?")) return;
    try {
      await deleteActorWardrobePreset(id);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const startEdit = (wp: ActorWardrobePreset) => {
    setEditingId(wp.id);
    setLabel(wp.label);
    setDescription(wp.description ?? "");
  };

  const handleGenerateImage = async (presetId: string) => {
    if (!genRefId) return;
    setGenerating(true);
    try {
      await generateWardrobeImage(presetId, genRefId, genModel || undefined, genExtraPrompt || undefined, genStyle || undefined);
      // Keep the panel open so the user can generate more images with the same settings.
      // Only clear the extra prompt — style, model, and ref persist.
      setGenExtraPrompt("");
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  };

  const handleDeleteImage = async (presetId: string, index: number) => {
    try {
      await deleteWardrobeImage(presetId, index);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUploadImage = async (presetId: string, file: File) => {
    setUploading(true);
    try {
      await uploadWardrobePresetImage(presetId, file);
      onRefresh();
    } catch (err: unknown) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      setUploadPresetId(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-300">Wardrobe Presets</h4>
        <button
          onClick={() => { setShowAdd(true); setLabel(""); setDescription(""); }}
          className="text-xs px-3 py-1.5 rounded bg-blue-600 text-white hover:bg-blue-500"
        >
          + Add
        </button>
      </div>

      {showAdd && (
        <div className="rounded border border-blue-700 bg-gray-800/50 p-3 space-y-2">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Label</label>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
              placeholder="e.g. Casual, Formal, Combat"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
              placeholder="Describe the wardrobe..."
            />
          </div>
          <div className="flex gap-1 justify-end">
            <button onClick={() => setShowAdd(false)} className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600">Cancel</button>
            <button onClick={handleAdd} disabled={saving || !label.trim()} className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50">
              {saving ? "..." : "Create"}
            </button>
          </div>
        </div>
      )}

      {actor.wardrobe_presets.length === 0 && !showAdd && (
        <p className="text-xs text-gray-500 text-center py-6">No wardrobe presets yet</p>
      )}

      {/* Hidden file input for image upload */}
      <input
        ref={uploadRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file && uploadPresetId) {
            handleUploadImage(uploadPresetId, file);
          }
          e.target.value = "";
        }}
      />

      {actor.wardrobe_presets.map((wp) =>
        editingId === wp.id ? (
          <div key={wp.id} className="rounded border border-blue-700 bg-gray-800/50 p-3 space-y-2">
            <div className="flex gap-3">
              {/* Images on left in edit view */}
              {wp.reference_images && wp.reference_images.length > 0 && (
                <div className="flex flex-col gap-1.5 shrink-0">
                  {wp.reference_images.map((url, idx) => (
                    <div key={idx} className="relative group">
                      <img
                        src={url}
                        alt={`Ref ${idx + 1}`}
                        className="w-16 h-16 object-cover rounded border border-gray-700 cursor-pointer hover:border-blue-500 transition-colors"
                        onClick={() => setLightboxUrl(url)}
                      />
                      <button
                        onClick={() => handleDeleteImage(wp.id, idx)}
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-600 text-white text-xs leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Remove image"
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex-1 min-w-0 space-y-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Label</label>
                  <input
                    type="text"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Description</label>
                  <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={2}
                    className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
                  />
                </div>
                <div className="flex gap-1 justify-end">
                  <button onClick={() => setEditingId(null)} className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600">Cancel</button>
                  <button onClick={() => handleUpdate(wp.id)} disabled={saving} className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50">
                    {saving ? "..." : "Save"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div key={wp.id} className="rounded border border-gray-700 bg-gray-800/50 p-3">
            <div className="flex items-start gap-3">
              {/* Images on left */}
              {wp.reference_images && wp.reference_images.length > 0 && (
                <div className="flex flex-col gap-1.5 shrink-0">
                  {wp.reference_images.map((url, idx) => (
                    <div key={idx} className="relative group">
                      <img
                        src={url}
                        alt={`Ref ${idx + 1}`}
                        className="w-16 h-16 object-cover rounded border border-gray-700 cursor-pointer hover:border-blue-500 transition-colors"
                        onClick={() => setLightboxUrl(url)}
                      />
                      <button
                        onClick={() => handleDeleteImage(wp.id, idx)}
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-600 text-white text-xs leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Remove image"
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium text-gray-200">{wp.label}</span>
                {wp.description && (
                  <p className="text-xs text-gray-400 mt-1">{wp.description}</p>
                )}

                {/* Generate image inline form */}
                {genPresetId === wp.id && (
                  <div className="mt-3 rounded border border-purple-700/50 bg-gray-900/50 p-2 space-y-2">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Source Face Ref</label>
                      <select
                        value={genRefId}
                        onChange={(e) => setGenRefId(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 outline-none"
                      >
                        <option value="">Select reference...</option>
                        {actor.refs.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.label || "Ref"} {r.is_primary ? "(primary)" : ""}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Image Model (optional)</label>
                      <select
                        value={genModel}
                        onChange={(e) => setGenModel(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 outline-none"
                      >
                        <option value="">Default</option>
                        {IMAGE_MODELS.filter((m) => REFERENCE_IMAGE_MODELS.has(m.id)).map((m) => (
                          <option key={m.id} value={m.id}>{m.label}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Extra Prompt (optional)</label>
                      <input
                        type="text"
                        value={genExtraPrompt}
                        onChange={(e) => setGenExtraPrompt(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 outline-none"
                        placeholder="Additional pose/setting details..."
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Style (optional)</label>
                      <div className="flex flex-wrap gap-1.5">
                        <button
                          onClick={() => setGenStyle("")}
                          disabled={generating}
                          className={`px-2 py-1 text-[11px] font-medium rounded-full transition-colors ${
                            !genStyle
                              ? "bg-purple-600 text-white"
                              : "bg-gray-800 text-gray-400 hover:text-gray-200"
                          } disabled:opacity-50`}
                        >
                          Default
                        </button>
                        {STYLE_OPTIONS.map((s) => (
                          <button
                            key={s}
                            onClick={() => setGenStyle(s)}
                            disabled={generating}
                            className={`px-2 py-1 text-[11px] font-medium rounded-full transition-colors ${
                              genStyle === s
                                ? "bg-purple-600 text-white"
                                : "bg-gray-800 text-gray-400 hover:text-gray-200"
                            } disabled:opacity-50`}
                          >
                            {s.replace(/_/g, " ")}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-1 justify-end">
                      <button
                        onClick={() => setGenPresetId(null)}
                        className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => handleGenerateImage(wp.id)}
                        disabled={generating || !genRefId}
                        className="text-xs px-3 py-1 rounded bg-purple-600 text-white hover:bg-purple-500 disabled:opacity-50"
                      >
                        {generating ? "Generating..." : "Generate"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
              <div className="flex gap-1 shrink-0 ml-2">
                {genPresetId !== wp.id && (
                  <>
                    <button
                      onClick={() => {
                        setUploadPresetId(wp.id);
                        uploadRef.current?.click();
                      }}
                      disabled={uploading}
                      className="text-xs px-2 py-1 rounded bg-green-900/50 text-green-300 hover:bg-green-800/50 disabled:opacity-30"
                      title="Upload wardrobe image"
                    >
                      {uploading && uploadPresetId === wp.id ? "..." : "Upload"}
                    </button>
                    <button
                      onClick={() => {
                        setGenPresetId(wp.id);
                        setGenRefId(actor.refs.find((r) => r.is_primary)?.id || actor.refs[0]?.id || "");
                        setGenModel("");
                        setGenExtraPrompt("");
                        setGenStyle("");
                      }}
                      disabled={actor.refs.length === 0}
                      className="text-xs px-2 py-1 rounded bg-purple-900/50 text-purple-300 hover:bg-purple-800/50 disabled:opacity-30"
                      title={actor.refs.length === 0 ? "Upload a face ref first" : "Generate wardrobe image"}
                    >
                      Gen
                    </button>
                  </>
                )}
                <button
                  onClick={() => startEdit(wp)}
                  className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDelete(wp.id)}
                  className="text-xs px-2 py-1 rounded bg-red-900 text-red-300 hover:bg-red-800"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )
      )}

      {/* Lightbox */}
      {lightboxUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={() => setLightboxUrl(null)}
        >
          <img
            src={lightboxUrl}
            alt="Wardrobe reference"
            className="max-w-[90vw] max-h-[90vh] rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Usage Tab
// ============================================================================

function UsageTab({ actor }: { actor: Actor }) {
  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-gray-300">Usage</h4>
      {actor.binding_count && actor.binding_count > 0 ? (
        <p className="text-xs text-gray-400">
          This actor is used in {actor.binding_count} production bible binding{actor.binding_count !== 1 ? "s" : ""}.
        </p>
      ) : null}
      <div className="flex items-center justify-center py-12 rounded-lg border border-dashed border-gray-700">
        <p className="text-sm text-gray-500">Usage tracking coming soon</p>
      </div>
    </div>
  );
}
