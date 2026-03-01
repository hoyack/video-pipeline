import { useState, useEffect, useCallback } from "react";
import type {
  SetResponse,
  PropResponse,
  CharacterResponse,
} from "../api/types.ts";
import {
  listSets,
  createSet,
  updateSet,
  deleteSet,
  uploadSetReference,
  upsertSonicIdentity,
  deleteSonicIdentity,
  listProps,
  createProp,
  updateProp,
  deleteProp,
  listCharacters,
  migrateEntities,
} from "../api/client.ts";

interface SetDetailProps {
  productionBibleId: string;
}

type ViewMode = "sets" | "props";
type SetSubTab = "visual" | "sonic_identity";

export function SetDetail({ productionBibleId }: SetDetailProps) {
  const [sets, setSets] = useState<SetResponse[]>([]);
  const [props, setProps] = useState<PropResponse[]>([]);
  const [characters, setCharacters] = useState<CharacterResponse[]>([]);
  const [selectedSetId, setSelectedSetId] = useState<string | null>(null);
  const [selectedPropId, setSelectedPropId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("sets");
  const [activeSubTab, setActiveSubTab] = useState<SetSubTab>("visual");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create forms
  const [showCreateSetForm, setShowCreateSetForm] = useState(false);
  const [newSetName, setNewSetName] = useState("");
  const [creatingSet, setCreatingSet] = useState(false);

  const [showCreatePropForm, setShowCreatePropForm] = useState(false);
  const [newPropName, setNewPropName] = useState("");
  const [newPropDescription, setNewPropDescription] = useState("");
  const [creatingProp, setCreatingProp] = useState(false);

  const selectedSet = sets.find((s) => s.set_id === selectedSetId) ?? null;
  const selectedProp = props.find((p) => p.prop_id === selectedPropId) ?? null;

  const load = useCallback(async () => {
    try {
      setLoading(true);
      migrateEntities(productionBibleId).catch(() => {});
      const [setsData, propsData, charsData] = await Promise.all([
        listSets(productionBibleId),
        listProps(productionBibleId),
        listCharacters(productionBibleId),
      ]);
      setSets(setsData);
      setProps(propsData);
      setCharacters(charsData);
      if (setsData.length > 0 && !selectedSetId) {
        setSelectedSetId(setsData[0].set_id);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [productionBibleId, selectedSetId]);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productionBibleId]);

  // --- Set handlers ---
  const handleCreateSet = async () => {
    if (!newSetName.trim()) return;
    setCreatingSet(true);
    setError(null);
    try {
      const s = await createSet(productionBibleId, { name: newSetName.trim() });
      setSets((prev) => [...prev, s]);
      setSelectedSetId(s.set_id);
      setNewSetName("");
      setShowCreateSetForm(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingSet(false);
    }
  };

  const handleUpdateSet = async (setId: string, data: Record<string, unknown>) => {
    try {
      const updated = await updateSet(setId, data);
      setSets((prev) => prev.map((s) => (s.set_id === setId ? updated : s)));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteSet = async (setId: string) => {
    if (!confirm("Delete this set? This cannot be undone.")) return;
    try {
      await deleteSet(setId);
      setSets((prev) => prev.filter((s) => s.set_id !== setId));
      if (selectedSetId === setId) {
        setSelectedSetId(sets.find((s) => s.set_id !== setId)?.set_id ?? null);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUploadSetRef = async (setId: string, file: File) => {
    try {
      const updated = await uploadSetReference(setId, file);
      setSets((prev) => prev.map((s) => (s.set_id === setId ? updated : s)));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUpsertSonic = async (setId: string, data: { ambience_description?: string; generation_prompt?: string }) => {
    try {
      const sonic = await upsertSonicIdentity(setId, data);
      setSets((prev) =>
        prev.map((s) => (s.set_id === setId ? { ...s, sonic_identity: sonic } : s)),
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteSonic = async (setId: string) => {
    try {
      await deleteSonicIdentity(setId);
      setSets((prev) =>
        prev.map((s) => (s.set_id === setId ? { ...s, sonic_identity: null } : s)),
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // --- Prop handlers ---
  const handleCreateProp = async () => {
    if (!newPropName.trim()) return;
    setCreatingProp(true);
    setError(null);
    try {
      const p = await createProp(productionBibleId, {
        name: newPropName.trim(),
        description: newPropDescription || undefined,
      });
      setProps((prev) => [...prev, p]);
      setNewPropName("");
      setNewPropDescription("");
      setShowCreatePropForm(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingProp(false);
    }
  };

  const handleUpdateProp = async (propId: string, data: Record<string, unknown>) => {
    try {
      const updated = await updateProp(propId, data);
      setProps((prev) => prev.map((p) => (p.prop_id === propId ? updated : p)));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteProp = async (propId: string) => {
    if (!confirm("Delete this prop? This cannot be undone.")) return;
    try {
      await deleteProp(propId);
      setProps((prev) => prev.filter((p) => p.prop_id !== propId));
      if (selectedPropId === propId) setSelectedPropId(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (loading) {
    return <div className="flex justify-center py-12"><div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" /></div>;
  }

  return (
    <div>
      {error && (
        <div className="mb-4 rounded-lg bg-red-900/30 border border-red-700 px-4 py-2 flex items-center justify-between">
          <p className="text-sm text-red-300">{error}</p>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 ml-4 text-xs">dismiss</button>
        </div>
      )}

      {/* View mode toggle */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setViewMode("sets")}
          className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
            viewMode === "sets" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Sets
          {sets.length > 0 && <span className="ml-1.5 rounded-full bg-gray-700 px-1.5 py-0.5 text-xs">{sets.length}</span>}
        </button>
        <button
          onClick={() => setViewMode("props")}
          className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
            viewMode === "props" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Props
          {props.length > 0 && <span className="ml-1.5 rounded-full bg-gray-700 px-1.5 py-0.5 text-xs">{props.length}</span>}
        </button>
      </div>

      {/* Sets view */}
      {viewMode === "sets" && (
        <div className="flex gap-4 min-h-[400px]">
          {/* Left panel: set list */}
          <div className="w-56 flex-shrink-0 rounded-lg border border-gray-800 bg-gray-900/50 p-3">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-300">Sets</h3>
              <button
                onClick={() => setShowCreateSetForm(true)}
                className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
              >
                + Add
              </button>
            </div>

            {showCreateSetForm && (
              <div className="mb-3 rounded border border-gray-700 bg-gray-800 p-2 space-y-2">
                <input
                  type="text"
                  value={newSetName}
                  onChange={(e) => setNewSetName(e.target.value)}
                  placeholder="Set name"
                  className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
                  autoFocus
                />
                <div className="flex gap-1">
                  <button
                    onClick={handleCreateSet}
                    disabled={creatingSet || !newSetName.trim()}
                    className="flex-1 text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
                  >
                    {creatingSet ? "..." : "Create"}
                  </button>
                  <button
                    onClick={() => { setShowCreateSetForm(false); setNewSetName(""); }}
                    className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            <div className="space-y-1">
              {sets.length === 0 && <p className="text-xs text-gray-500 text-center py-4">No sets yet</p>}
              {sets.map((s) => (
                <div
                  key={s.set_id}
                  onClick={() => { setSelectedSetId(s.set_id); setActiveSubTab("visual"); }}
                  className={`flex items-center justify-between px-2 py-1.5 rounded cursor-pointer group ${
                    selectedSetId === s.set_id
                      ? "bg-blue-600/20 border border-blue-600/40"
                      : "hover:bg-gray-800"
                  }`}
                >
                  <span className="text-sm text-gray-200 truncate">{s.name}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteSet(s.set_id); }}
                    className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 text-xs ml-1"
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Right panel: set detail */}
          <div className="flex-1 rounded-lg border border-gray-800 bg-gray-900/50 p-4">
            {!selectedSet ? (
              <div className="flex items-center justify-center h-full">
                <p className="text-sm text-gray-500">Select a set to edit</p>
              </div>
            ) : (
              <>
                {/* Sub-tabs */}
                <div className="flex gap-1 mb-4 border-b border-gray-800 pb-2">
                  {([
                    { id: "visual" as SetSubTab, label: "Visual" },
                    { id: "sonic_identity" as SetSubTab, label: "Sonic Identity" },
                  ] as const).map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveSubTab(tab.id)}
                      className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
                        activeSubTab === tab.id
                          ? "bg-gray-800 text-white border-b-2 border-blue-500"
                          : "text-gray-400 hover:text-gray-200"
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {activeSubTab === "visual" && (
                  <SetVisualTab
                    set={selectedSet}
                    onUpdate={handleUpdateSet}
                    onUploadRef={handleUploadSetRef}
                  />
                )}

                {activeSubTab === "sonic_identity" && (
                  <SetSonicTab
                    set={selectedSet}
                    onUpsert={handleUpsertSonic}
                    onDelete={handleDeleteSonic}
                  />
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Props view */}
      {viewMode === "props" && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Props</h3>
            <button
              onClick={() => setShowCreatePropForm(true)}
              className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
            >
              + Add Prop
            </button>
          </div>

          {showCreatePropForm && (
            <div className="mb-4 rounded border border-gray-700 bg-gray-800 p-3 space-y-2">
              <input
                type="text"
                value={newPropName}
                onChange={(e) => setNewPropName(e.target.value)}
                placeholder="Prop name"
                className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
                autoFocus
              />
              <input
                type="text"
                value={newPropDescription}
                onChange={(e) => setNewPropDescription(e.target.value)}
                placeholder="Description (optional)"
                className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
              />
              <div className="flex gap-1">
                <button
                  onClick={handleCreateProp}
                  disabled={creatingProp || !newPropName.trim()}
                  className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {creatingProp ? "..." : "Create"}
                </button>
                <button
                  onClick={() => { setShowCreatePropForm(false); setNewPropName(""); setNewPropDescription(""); }}
                  className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {props.length === 0 ? (
            <div className="flex items-center justify-center py-12 rounded-lg border border-dashed border-gray-700">
              <p className="text-sm text-gray-500">No props yet</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              {props.map((p) => (
                <div
                  key={p.prop_id}
                  onClick={() => setSelectedPropId(selectedPropId === p.prop_id ? null : p.prop_id)}
                  className={`rounded-lg border bg-gray-900/50 p-3 cursor-pointer transition-colors ${
                    selectedPropId === p.prop_id
                      ? "border-blue-600/60"
                      : "border-gray-800 hover:border-gray-700"
                  }`}
                >
                  {/* Thumbnail */}
                  {p.reference_image ? (
                    <img
                      src={p.reference_image}
                      alt={p.name}
                      className="w-full aspect-square object-cover rounded border border-gray-700 mb-2"
                    />
                  ) : (
                    <div className="w-full aspect-square bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-600 text-xs mb-2">
                      No image
                    </div>
                  )}
                  <h4 className="text-sm font-medium text-gray-200 truncate">{p.name}</h4>
                  {p.description && (
                    <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{p.description}</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Prop inline editor */}
          {selectedProp && (
            <PropEditor
              prop={selectedProp}
              characters={characters}
              onUpdate={handleUpdateProp}
              onDelete={handleDeleteProp}
              onClose={() => setSelectedPropId(null)}
            />
          )}
        </div>
      )}
    </div>
  );
}

// --- Sub-components ---

function SetVisualTab({
  set,
  onUpdate,
  onUploadRef,
}: {
  set: SetResponse;
  onUpdate: (id: string, data: Record<string, unknown>) => Promise<void>;
  onUploadRef: (id: string, file: File) => Promise<void>;
}) {
  const [name, setName] = useState(set.name);
  const [styleTags, setStyleTags] = useState(set.style_tags?.join(", ") ?? "");
  const [lightingNotes, setLightingNotes] = useState(set.lighting_notes ?? "");
  const [promptTags, setPromptTags] = useState(set.prompt_tags?.join(", ") ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(set.name);
    setStyleTags(set.style_tags?.join(", ") ?? "");
    setLightingNotes(set.lighting_notes ?? "");
    setPromptTags(set.prompt_tags?.join(", ") ?? "");
  }, [set]);

  const handleSave = async () => {
    setSaving(true);
    await onUpdate(set.set_id, {
      name: name.trim(),
      style_tags: styleTags ? styleTags.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
      lighting_notes: lightingNotes || undefined,
      prompt_tags: promptTags ? promptTags.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
    });
    setSaving(false);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUploadRef(set.set_id, file);
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

      {/* Reference image */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Reference Image</label>
        {set.reference_image ? (
          <img
            src={set.reference_image}
            alt={set.name}
            className="w-48 h-32 object-cover rounded border border-gray-700 mb-2"
          />
        ) : (
          <div className="w-48 h-32 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-600 text-xs mb-2">
            No image
          </div>
        )}
        <label className="inline-block text-xs px-3 py-1.5 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 cursor-pointer">
          Upload Reference
          <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
        </label>
      </div>

      {/* Reverse prompt (read-only) */}
      {set.reverse_prompt && (
        <div>
          <label className="block text-xs text-gray-400 mb-1">Reverse Prompt (auto-generated)</label>
          <p className="text-sm text-gray-400 bg-gray-800 rounded px-2 py-1.5">{set.reverse_prompt}</p>
        </div>
      )}

      <div>
        <label className="block text-xs text-gray-400 mb-1">Style Tags (comma-separated)</label>
        <input
          type="text"
          value={styleTags}
          onChange={(e) => setStyleTags(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
          placeholder="cinematic, noir, moody"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Lighting Notes</label>
        <textarea
          value={lightingNotes}
          onChange={(e) => setLightingNotes(e.target.value)}
          rows={2}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Lighting direction..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Prompt Tags (comma-separated)</label>
        <input
          type="text"
          value={promptTags}
          onChange={(e) => setPromptTags(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
          placeholder="tag1, tag2"
        />
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

function SetSonicTab({
  set,
  onUpsert,
  onDelete,
}: {
  set: SetResponse;
  onUpsert: (setId: string, data: { ambience_description?: string; generation_prompt?: string }) => Promise<void>;
  onDelete: (setId: string) => Promise<void>;
}) {
  const [ambienceDescription, setAmbienceDescription] = useState(set.sonic_identity?.ambience_description ?? "");
  const [generationPrompt, setGenerationPrompt] = useState(set.sonic_identity?.generation_prompt ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setAmbienceDescription(set.sonic_identity?.ambience_description ?? "");
    setGenerationPrompt(set.sonic_identity?.generation_prompt ?? "");
  }, [set.sonic_identity]);

  const handleSave = async () => {
    setSaving(true);
    await onUpsert(set.set_id, {
      ambience_description: ambienceDescription || undefined,
      generation_prompt: generationPrompt || undefined,
    });
    setSaving(false);
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Ambience Description</label>
        <textarea
          value={ambienceDescription}
          onChange={(e) => setAmbienceDescription(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Describe the ambient sound of this location..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Generation Prompt</label>
        <textarea
          value={generationPrompt}
          onChange={(e) => setGenerationPrompt(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Audio generation prompt..."
        />
      </div>

      <div className="flex items-center justify-between">
        <button
          disabled
          title="Audio adapter coming soon"
          className="text-xs px-3 py-1.5 rounded bg-gray-700 text-gray-500 cursor-not-allowed"
        >
          Generate Audio
        </button>

        <div className="flex gap-2">
          {set.sonic_identity && (
            <button
              onClick={() => onDelete(set.set_id)}
              className="text-xs px-3 py-1.5 rounded bg-red-900 text-red-300 hover:bg-red-800"
            >
              Delete Sonic Identity
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-1.5 text-sm rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

function PropEditor({
  prop,
  characters,
  onUpdate,
  onDelete,
  onClose,
}: {
  prop: PropResponse;
  characters: CharacterResponse[];
  onUpdate: (id: string, data: Record<string, unknown>) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onClose: () => void;
}) {
  const [name, setName] = useState(prop.name);
  const [description, setDescription] = useState(prop.description ?? "");
  const [promptTags, setPromptTags] = useState(prop.prompt_tags?.join(", ") ?? "");
  const [associatedChars, setAssociatedChars] = useState<string[]>(prop.associated_characters ?? []);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(prop.name);
    setDescription(prop.description ?? "");
    setPromptTags(prop.prompt_tags?.join(", ") ?? "");
    setAssociatedChars(prop.associated_characters ?? []);
  }, [prop]);

  const handleSave = async () => {
    setSaving(true);
    await onUpdate(prop.prop_id, {
      name: name.trim(),
      description: description || undefined,
      prompt_tags: promptTags ? promptTags.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
      associated_characters: associatedChars.length > 0 ? associatedChars : undefined,
    });
    setSaving(false);
  };

  const toggleCharacter = (charId: string) => {
    setAssociatedChars((prev) =>
      prev.includes(charId)
        ? prev.filter((id) => id !== charId)
        : [...prev, charId],
    );
  };

  return (
    <div className="mt-4 rounded-lg border border-blue-700/50 bg-gray-900/80 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-200">Edit Prop</h4>
        <button onClick={onClose} className="text-xs text-gray-400 hover:text-gray-200">close</button>
      </div>

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
        <label className="block text-xs text-gray-400 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
        />
      </div>

      {characters.length > 0 && (
        <div>
          <label className="block text-xs text-gray-400 mb-1">Associated Characters</label>
          <div className="flex flex-wrap gap-1">
            {characters.map((c) => (
              <button
                key={c.character_id}
                onClick={() => toggleCharacter(c.character_id)}
                className={`text-xs px-2 py-1 rounded transition-colors ${
                  associatedChars.includes(c.character_id)
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
              >
                {c.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <div>
        <label className="block text-xs text-gray-400 mb-1">Prompt Tags (comma-separated)</label>
        <input
          type="text"
          value={promptTags}
          onChange={(e) => setPromptTags(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
        />
      </div>

      <div className="flex justify-between">
        <button
          onClick={() => onDelete(prop.prop_id)}
          className="text-xs px-3 py-1.5 rounded bg-red-900 text-red-300 hover:bg-red-800"
        >
          Delete
        </button>
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
