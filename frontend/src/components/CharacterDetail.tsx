import { useState, useEffect, useCallback } from "react";
import type {
  CharacterResponse,
  WardrobeResponse,
  VoiceProfileResponse,
} from "../api/types.ts";
import {
  listCharacters,
  createCharacter,
  updateCharacter,
  deleteCharacter,
  createWardrobe,
  updateWardrobe,
  deleteWardrobe,
  upsertVoiceProfile,
  deleteVoiceProfile,
  migrateEntities,
} from "../api/client.ts";

interface CharacterDetailProps {
  productionBibleId: string;
}

const ROLES = ["PROTAGONIST", "ANTAGONIST", "SUPPORTING", "EXTRA"] as const;

type SubTab = "overview" | "actor_refs" | "wardrobe" | "voice_profile";

export function CharacterDetail({ productionBibleId }: CharacterDetailProps) {
  const [characters, setCharacters] = useState<CharacterResponse[]>([]);
  const [selectedCharacterId, setSelectedCharacterId] = useState<string | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<SubTab>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create form state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newRole, setNewRole] = useState<string>("SUPPORTING");
  const [creating, setCreating] = useState(false);

  // Wardrobe create form
  const [showWardrobeForm, setShowWardrobeForm] = useState(false);
  const [newWardrobeLabel, setNewWardrobeLabel] = useState("");
  const [creatingWardrobe, setCreatingWardrobe] = useState(false);

  const selectedCharacter = characters.find((c) => c.character_id === selectedCharacterId) ?? null;

  const load = useCallback(async () => {
    try {
      setLoading(true);
      // Fire-and-forget migration (idempotent)
      migrateEntities(productionBibleId).catch(() => {});
      const chars = await listCharacters(productionBibleId);
      setCharacters(chars);
      if (chars.length > 0 && !selectedCharacterId) {
        setSelectedCharacterId(chars[0].character_id);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [productionBibleId, selectedCharacterId]);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productionBibleId]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const char = await createCharacter(productionBibleId, { name: newName.trim(), role: newRole });
      setCharacters((prev) => [...prev, char]);
      setSelectedCharacterId(char.character_id);
      setNewName("");
      setNewRole("SUPPORTING");
      setShowCreateForm(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (charId: string) => {
    if (!confirm("Delete this character? This cannot be undone.")) return;
    try {
      await deleteCharacter(charId);
      setCharacters((prev) => prev.filter((c) => c.character_id !== charId));
      if (selectedCharacterId === charId) {
        setSelectedCharacterId(characters.find((c) => c.character_id !== charId)?.character_id ?? null);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUpdate = async (charId: string, data: Record<string, unknown>) => {
    try {
      const updated = await updateCharacter(charId, data);
      setCharacters((prev) => prev.map((c) => (c.character_id === charId ? updated : c)));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleCreateWardrobe = async () => {
    if (!selectedCharacter || !newWardrobeLabel.trim()) return;
    setCreatingWardrobe(true);
    try {
      const w = await createWardrobe(selectedCharacter.character_id, { label: newWardrobeLabel.trim() });
      setCharacters((prev) =>
        prev.map((c) =>
          c.character_id === selectedCharacter.character_id
            ? { ...c, wardrobe: [...(c.wardrobe ?? []), w] }
            : c,
        ),
      );
      setNewWardrobeLabel("");
      setShowWardrobeForm(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingWardrobe(false);
    }
  };

  const handleUpdateWardrobe = async (wId: string, data: Record<string, unknown>) => {
    if (!selectedCharacter) return;
    try {
      const updated = await updateWardrobe(wId, data);
      setCharacters((prev) =>
        prev.map((c) =>
          c.character_id === selectedCharacter.character_id
            ? { ...c, wardrobe: (c.wardrobe ?? []).map((w) => (w.wardrobe_id === wId ? updated : w)) }
            : c,
        ),
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteWardrobe = async (wId: string) => {
    if (!selectedCharacter || !confirm("Delete this wardrobe item?")) return;
    try {
      await deleteWardrobe(wId);
      setCharacters((prev) =>
        prev.map((c) =>
          c.character_id === selectedCharacter.character_id
            ? { ...c, wardrobe: (c.wardrobe ?? []).filter((w) => w.wardrobe_id !== wId) }
            : c,
        ),
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUpsertVoice = async (data: { voice_id?: string; adapter_type?: string; style_notes?: string }) => {
    if (!selectedCharacter) return;
    try {
      const vp = await upsertVoiceProfile(selectedCharacter.character_id, data);
      setCharacters((prev) =>
        prev.map((c) =>
          c.character_id === selectedCharacter.character_id
            ? { ...c, voice_profile: vp }
            : c,
        ),
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteVoice = async () => {
    if (!selectedCharacter) return;
    try {
      await deleteVoiceProfile(selectedCharacter.character_id);
      setCharacters((prev) =>
        prev.map((c) =>
          c.character_id === selectedCharacter.character_id
            ? { ...c, voice_profile: null }
            : c,
        ),
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const roleBadgeColor = (role: string) => {
    switch (role) {
      case "PROTAGONIST": return "bg-yellow-900 text-yellow-300";
      case "ANTAGONIST": return "bg-red-900 text-red-300";
      case "SUPPORTING": return "bg-blue-900 text-blue-300";
      case "EXTRA": return "bg-gray-700 text-gray-300";
      default: return "bg-gray-700 text-gray-300";
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

      <div className="flex gap-4 min-h-[400px]">
        {/* Left panel: character list */}
        <div className="w-64 flex-shrink-0 rounded-lg border border-gray-800 bg-gray-900/50 p-3">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">Characters</h3>
            <button
              onClick={() => setShowCreateForm(true)}
              className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
            >
              + Add
            </button>
          </div>

          {showCreateForm && (
            <div className="mb-3 rounded border border-gray-700 bg-gray-800 p-2 space-y-2">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Character name"
                className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
                autoFocus
              />
              <select
                value={newRole}
                onChange={(e) => setNewRole(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none"
              >
                {ROLES.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
              <div className="flex gap-1">
                <button
                  onClick={handleCreate}
                  disabled={creating || !newName.trim()}
                  className="flex-1 text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {creating ? "..." : "Create"}
                </button>
                <button
                  onClick={() => { setShowCreateForm(false); setNewName(""); }}
                  className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          <div className="space-y-1">
            {characters.length === 0 && (
              <p className="text-xs text-gray-500 text-center py-4">No characters yet</p>
            )}
            {characters.map((char) => (
              <div
                key={char.character_id}
                onClick={() => { setSelectedCharacterId(char.character_id); setActiveSubTab("overview"); }}
                className={`flex items-center justify-between px-2 py-1.5 rounded cursor-pointer group ${
                  selectedCharacterId === char.character_id
                    ? "bg-blue-600/20 border border-blue-600/40"
                    : "hover:bg-gray-800"
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-sm text-gray-200 truncate">{char.name}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${roleBadgeColor(char.role)}`}>
                    {char.role}
                  </span>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(char.character_id); }}
                  className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 text-xs ml-1"
                  title="Delete character"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Right panel: character detail */}
        <div className="flex-1 rounded-lg border border-gray-800 bg-gray-900/50 p-4">
          {!selectedCharacter ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-sm text-gray-500">Select a character to edit</p>
            </div>
          ) : (
            <>
              {/* Sub-tabs */}
              <div className="flex gap-1 mb-4 border-b border-gray-800 pb-2">
                {([
                  { id: "overview" as SubTab, label: "Overview" },
                  { id: "actor_refs" as SubTab, label: "Actor Refs" },
                  { id: "wardrobe" as SubTab, label: "Wardrobe" },
                  { id: "voice_profile" as SubTab, label: "Voice Profile" },
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
                    {tab.id === "wardrobe" && (selectedCharacter.wardrobe?.length ?? 0) > 0 && (
                      <span className="ml-1 text-[10px] bg-gray-700 rounded-full px-1.5">{selectedCharacter.wardrobe.length}</span>
                    )}
                  </button>
                ))}
              </div>

              {/* Overview tab */}
              {activeSubTab === "overview" && (
                <OverviewTab character={selectedCharacter} onUpdate={handleUpdate} />
              )}

              {/* Actor References tab */}
              {activeSubTab === "actor_refs" && (
                <ActorRefsTab character={selectedCharacter} />
              )}

              {/* Wardrobe tab */}
              {activeSubTab === "wardrobe" && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-medium text-gray-300">Wardrobe Items</h4>
                    <button
                      onClick={() => setShowWardrobeForm(true)}
                      className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
                    >
                      + Add Wardrobe
                    </button>
                  </div>

                  {showWardrobeForm && (
                    <div className="mb-3 rounded border border-gray-700 bg-gray-800 p-3 space-y-2">
                      <input
                        type="text"
                        value={newWardrobeLabel}
                        onChange={(e) => setNewWardrobeLabel(e.target.value)}
                        placeholder="Wardrobe label (e.g. 'Casual', 'Formal')"
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
                        autoFocus
                      />
                      <div className="flex gap-1">
                        <button
                          onClick={handleCreateWardrobe}
                          disabled={creatingWardrobe || !newWardrobeLabel.trim()}
                          className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
                        >
                          {creatingWardrobe ? "..." : "Create"}
                        </button>
                        <button
                          onClick={() => { setShowWardrobeForm(false); setNewWardrobeLabel(""); }}
                          className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {(selectedCharacter.wardrobe?.length ?? 0) === 0 ? (
                    <p className="text-xs text-gray-500 text-center py-6">No wardrobe items yet</p>
                  ) : (
                    <div className="space-y-2">
                      {(selectedCharacter.wardrobe ?? []).map((w) => (
                        <WardrobeItem
                          key={w.wardrobe_id}
                          wardrobe={w}
                          onUpdate={handleUpdateWardrobe}
                          onDelete={handleDeleteWardrobe}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Voice Profile tab */}
              {activeSubTab === "voice_profile" && (
                <VoiceProfileTab
                  voiceProfile={selectedCharacter.voice_profile}
                  onUpsert={handleUpsertVoice}
                  onDelete={handleDeleteVoice}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Sub-components ---

function OverviewTab({
  character,
  onUpdate,
}: {
  character: CharacterResponse;
  onUpdate: (id: string, data: Record<string, unknown>) => Promise<void>;
}) {
  const [name, setName] = useState(character.name);
  const [role, setRole] = useState(character.role);
  const [description, setDescription] = useState(character.description ?? "");
  const [arc, setArc] = useState(character.arc ?? "");
  const [baseAppearance, setBaseAppearance] = useState(character.base_appearance ?? "");
  const [promptTags, setPromptTags] = useState(character.prompt_tags?.join(", ") ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(character.name);
    setRole(character.role);
    setDescription(character.description ?? "");
    setArc(character.arc ?? "");
    setBaseAppearance(character.base_appearance ?? "");
    setPromptTags(character.prompt_tags?.join(", ") ?? "");
  }, [character]);

  const handleSave = async () => {
    setSaving(true);
    await onUpdate(character.character_id, {
      name: name.trim(),
      role,
      description: description || undefined,
      arc: arc || undefined,
      base_appearance: baseAppearance || undefined,
      prompt_tags: promptTags ? promptTags.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
    });
    setSaving(false);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
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
          <label className="block text-xs text-gray-400 mb-1">Role</label>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as CharacterResponse["role"])}
            className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
          >
            {ROLES.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Character description..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Character Arc</label>
        <textarea
          value={arc}
          onChange={(e) => setArc(e.target.value)}
          rows={2}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Character arc..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Base Appearance</label>
        <textarea
          value={baseAppearance}
          onChange={(e) => setBaseAppearance(e.target.value)}
          rows={2}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Visual appearance for prompting..."
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Prompt Tags (comma-separated)</label>
        <input
          type="text"
          value={promptTags}
          onChange={(e) => setPromptTags(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
          placeholder="tag1, tag2, tag3"
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

function ActorRefsTab({ character }: { character: CharacterResponse }) {
  const refs = character.actor_refs ?? [];

  if (refs.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-sm text-gray-500">
          No actor references yet. Upload endpoint coming in a future phase.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 gap-3">
      {refs.map((url, idx) => (
        <img
          key={idx}
          src={url}
          alt={`Actor ref ${idx + 1}`}
          className="w-full aspect-square object-cover rounded border border-gray-700"
        />
      ))}
    </div>
  );
}

function WardrobeItem({
  wardrobe,
  onUpdate,
  onDelete,
}: {
  wardrobe: WardrobeResponse;
  onUpdate: (id: string, data: Record<string, unknown>) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(wardrobe.label);
  const [promptDescriptor, setPromptDescriptor] = useState(wardrobe.prompt_descriptor ?? "");
  const [sceneContext, setSceneContext] = useState(wardrobe.scene_context ?? "");
  const [isDefault, setIsDefault] = useState(wardrobe.is_default);

  useEffect(() => {
    setLabel(wardrobe.label);
    setPromptDescriptor(wardrobe.prompt_descriptor ?? "");
    setSceneContext(wardrobe.scene_context ?? "");
    setIsDefault(wardrobe.is_default);
  }, [wardrobe]);

  const handleSave = async () => {
    await onUpdate(wardrobe.wardrobe_id, {
      label: label.trim(),
      prompt_descriptor: promptDescriptor || undefined,
      scene_context: sceneContext || undefined,
      is_default: isDefault,
    });
    setEditing(false);
  };

  if (!editing) {
    return (
      <div className="rounded border border-gray-700 bg-gray-800/50 p-3 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-200">{wardrobe.label}</span>
            {wardrobe.is_default && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-900 text-green-300">default</span>
            )}
          </div>
          {wardrobe.prompt_descriptor && (
            <p className="text-xs text-gray-400 mt-1">{wardrobe.prompt_descriptor}</p>
          )}
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setEditing(true)}
            className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
          >
            Edit
          </button>
          <button
            onClick={() => onDelete(wardrobe.wardrobe_id)}
            className="text-xs px-2 py-1 rounded bg-red-900 text-red-300 hover:bg-red-800"
          >
            Delete
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded border border-blue-700 bg-gray-800/50 p-3 space-y-2">
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
        <label className="block text-xs text-gray-400 mb-1">Prompt Descriptor</label>
        <input
          type="text"
          value={promptDescriptor}
          onChange={(e) => setPromptDescriptor(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
          placeholder="e.g. wearing a red leather jacket"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Scene Context</label>
        <input
          type="text"
          value={sceneContext}
          onChange={(e) => setSceneContext(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:border-blue-500 outline-none"
          placeholder="e.g. Act 2, night scenes"
        />
      </div>
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={isDefault}
          onChange={(e) => setIsDefault(e.target.checked)}
          id={`default-${wardrobe.wardrobe_id}`}
          className="accent-blue-500"
        />
        <label htmlFor={`default-${wardrobe.wardrobe_id}`} className="text-xs text-gray-400">Default wardrobe</label>
      </div>
      <div className="flex gap-1 justify-end">
        <button
          onClick={() => setEditing(false)}
          className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
        >
          Save
        </button>
      </div>
    </div>
  );
}

function VoiceProfileTab({
  voiceProfile,
  onUpsert,
  onDelete,
}: {
  voiceProfile: VoiceProfileResponse | null;
  onUpsert: (data: { voice_id?: string; adapter_type?: string; style_notes?: string }) => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [voiceId, setVoiceId] = useState(voiceProfile?.voice_id ?? "");
  const [styleNotes, setStyleNotes] = useState(voiceProfile?.style_notes ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setVoiceId(voiceProfile?.voice_id ?? "");
    setStyleNotes(voiceProfile?.style_notes ?? "");
  }, [voiceProfile]);

  const handleSave = async () => {
    setSaving(true);
    await onUpsert({
      voice_id: voiceId || undefined,
      adapter_type: "ELEVENLABS",
      style_notes: styleNotes || undefined,
    });
    setSaving(false);
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Voice ID</label>
        <input
          type="text"
          value={voiceId}
          onChange={(e) => setVoiceId(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
          placeholder="ElevenLabs voice ID"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Adapter Type</label>
        <input
          type="text"
          value="ELEVENLABS"
          readOnly
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-500 cursor-not-allowed"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Style Notes</label>
        <textarea
          value={styleNotes}
          onChange={(e) => setStyleNotes(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
          placeholder="Voice style direction..."
        />
      </div>

      <div className="flex items-center justify-between">
        <button
          disabled
          title="ElevenLabs adapter coming soon"
          className="text-xs px-3 py-1.5 rounded bg-gray-700 text-gray-500 cursor-not-allowed"
        >
          Generate Sample
        </button>

        <div className="flex gap-2">
          {voiceProfile && (
            <button
              onClick={onDelete}
              className="text-xs px-3 py-1.5 rounded bg-red-900 text-red-300 hover:bg-red-800"
            >
              Delete Profile
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
