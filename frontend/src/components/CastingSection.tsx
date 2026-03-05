import { useState, useEffect } from "react";
import type { CastBinding } from "../api/types.ts";
import {
  listCastBindings,
  createCastBinding,
  updateCastBinding,
  deleteCastBinding,
} from "../api/client.ts";
import { AssetPicker } from "./AssetPicker.tsx";

interface CastingSectionProps {
  bibleId: string;
}

const ROLES = ["LEAD", "SUPPORTING", "EXTRA", "NARRATOR"] as const;

const ROLE_COLORS: Record<string, string> = {
  LEAD: "bg-amber-900/50 text-amber-300",
  SUPPORTING: "bg-blue-900/50 text-blue-300",
  EXTRA: "bg-gray-700 text-gray-300",
  NARRATOR: "bg-purple-900/50 text-purple-300",
};

function generateTag(name: string): string {
  return name
    .toUpperCase()
    .replace(/[^A-Z0-9\s]/g, "")
    .trim()
    .replace(/\s+/g, "_");
}

export function CastingSection({ bibleId }: CastingSectionProps) {
  const [bindings, setBindings] = useState<CastBinding[]>([]);
  const [loading, setLoading] = useState(true);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [castFormOpen, setCastFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);

  // Form state for new cast binding
  const [selectedActor, setSelectedActor] = useState<{
    id: string;
    name: string;
    primary_ref_url?: string | null;
  } | null>(null);
  const [characterName, setCharacterName] = useState("");
  const [role, setRole] = useState<string>("LEAD");
  const [tag, setTag] = useState("");
  const [characterDescription, setCharacterDescription] = useState("");
  const [characterArc, setCharacterArc] = useState("");
  const [behavioralNotes, setBehavioralNotes] = useState("");

  // Edit form state
  const [editCharacterName, setEditCharacterName] = useState("");
  const [editRole, setEditRole] = useState<string>("LEAD");
  const [editTag, setEditTag] = useState("");
  const [editBehavioralNotes, setEditBehavioralNotes] = useState("");

  const [saving, setSaving] = useState(false);

  const fetchBindings = async () => {
    try {
      const data = await listCastBindings(bibleId);
      setBindings(data);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBindings();
  }, [bibleId]);

  const handleActorSelected = (asset: { id: string; name: string; primary_ref_url?: string | null }) => {
    setSelectedActor(asset);
    setCharacterName(asset.name);
    setTag(generateTag(asset.name));
    setRole("LEAD");
    setCharacterDescription("");
    setCharacterArc("");
    setBehavioralNotes("");
    setCastFormOpen(true);
  };

  const handleCharacterNameChange = (name: string) => {
    setCharacterName(name);
    setTag(generateTag(name));
  };

  const handleSaveCastBinding = async () => {
    if (!selectedActor || !characterName.trim() || !tag.trim()) return;
    setSaving(true);
    try {
      await createCastBinding(bibleId, {
        actor_id: selectedActor.id,
        tag: tag.trim(),
        character_name: characterName.trim(),
        role,
        character_description: characterDescription.trim() || undefined,
        character_arc: characterArc.trim() || undefined,
        behavioral_notes: behavioralNotes.trim() || undefined,
      });
      setCastFormOpen(false);
      setSelectedActor(null);
      await fetchBindings();
    } catch {
      // silently fail
    } finally {
      setSaving(false);
    }
  };

  const handleStartEdit = (binding: CastBinding) => {
    setEditingId(binding.id);
    setEditCharacterName(binding.character_name);
    setEditRole(binding.role);
    setEditTag(binding.tag);
    setEditBehavioralNotes(binding.behavioral_notes || "");
  };

  const handleSaveEdit = async () => {
    if (!editingId || !editCharacterName.trim() || !editTag.trim()) return;
    setSaving(true);
    try {
      await updateCastBinding(editingId, {
        character_name: editCharacterName.trim(),
        role: editRole,
        tag: editTag.trim(),
        behavioral_notes: editBehavioralNotes.trim() || undefined,
      });
      setEditingId(null);
      await fetchBindings();
    } catch {
      // silently fail
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteCastBinding(id);
      setBindings((prev) => prev.filter((b) => b.id !== id));
    } catch {
      // silently fail
    }
  };

  const excludeActorIds = bindings.map((b) => b.actor_id);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <span className="text-sm text-gray-400">Loading cast...</span>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Cast</h3>
        <button
          onClick={() => setPickerOpen(true)}
          className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
        >
          + Add Cast Member
        </button>
      </div>

      {/* Empty state */}
      {bindings.length === 0 && !castFormOpen && (
        <div className="flex min-h-[100px] items-center justify-center rounded-lg border border-dashed border-gray-700">
          <p className="text-sm text-gray-500">
            No cast members yet. Add actors from the Asset Library.
          </p>
        </div>
      )}

      {/* Cast form (shown after picker selection) */}
      {castFormOpen && selectedActor && (
        <div className="mb-4 rounded-lg border border-blue-800 bg-gray-900/80 p-4">
          <h4 className="text-sm font-medium text-white mb-3">
            Cast <span className="text-blue-400">{selectedActor.name}</span> as Character
          </h4>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Character Name</label>
              <input
                type="text"
                value={characterName}
                onChange={(e) => handleCharacterNameChange(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500"
              >
                {ROLES.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Tag</label>
              <input
                type="text"
                value={tag}
                onChange={(e) => setTag(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ""))}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-blue-500"
              />
            </div>
          </div>
          <div className="space-y-2 mb-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Character Description (optional)</label>
              <textarea
                value={characterDescription}
                onChange={(e) => setCharacterDescription(e.target.value)}
                rows={2}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Character Arc (optional)</label>
              <textarea
                value={characterArc}
                onChange={(e) => setCharacterArc(e.target.value)}
                rows={2}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Behavioral Notes (optional)</label>
              <textarea
                value={behavioralNotes}
                onChange={(e) => setBehavioralNotes(e.target.value)}
                rows={2}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleSaveCastBinding}
              disabled={saving || !characterName.trim() || !tag.trim()}
              className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors disabled:opacity-50"
            >
              {saving ? "Saving..." : "Add to Cast"}
            </button>
            <button
              onClick={() => { setCastFormOpen(false); setSelectedActor(null); }}
              className="rounded bg-gray-700 px-4 py-1.5 text-sm text-gray-300 hover:text-white transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Binding cards */}
      <div className="space-y-3">
        {bindings.map((binding) =>
          editingId === binding.id ? (
            /* Inline edit form */
            <div key={binding.id} className="rounded-lg border border-yellow-800 bg-gray-900/80 p-4">
              <div className="grid grid-cols-3 gap-3 mb-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Character Name</label>
                  <input
                    type="text"
                    value={editCharacterName}
                    onChange={(e) => setEditCharacterName(e.target.value)}
                    className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Role</label>
                  <select
                    value={editRole}
                    onChange={(e) => setEditRole(e.target.value)}
                    className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500"
                  >
                    {ROLES.map((r) => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Tag</label>
                  <input
                    type="text"
                    value={editTag}
                    onChange={(e) => setEditTag(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ""))}
                    className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-blue-500"
                  />
                </div>
              </div>
              <div className="mb-3">
                <label className="block text-xs text-gray-400 mb-1">Behavioral Notes</label>
                <textarea
                  value={editBehavioralNotes}
                  onChange={(e) => setEditBehavioralNotes(e.target.value)}
                  rows={2}
                  className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleSaveEdit}
                  disabled={saving}
                  className="rounded bg-blue-600 px-3 py-1 text-sm text-white hover:bg-blue-500 transition-colors disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Save"}
                </button>
                <button
                  onClick={() => setEditingId(null)}
                  className="rounded bg-gray-700 px-3 py-1 text-sm text-gray-300 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            /* Display card */
            <div
              key={binding.id}
              className="flex items-start gap-3 rounded-lg border border-gray-800 bg-gray-900/50 p-3"
            >
              {/* Thumbnail */}
              {binding.primary_ref_url ? (
                <img
                  src={binding.primary_ref_url}
                  alt={binding.actor_name || "Actor"}
                  className="w-14 h-14 rounded-lg object-cover border border-gray-700 shrink-0"
                />
              ) : (
                <div className="w-14 h-14 rounded-lg bg-gray-800 border border-gray-700 shrink-0 flex items-center justify-center text-gray-600 text-xs">
                  No ref
                </div>
              )}

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-semibold text-white">{binding.character_name}</span>
                  <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${ROLE_COLORS[binding.role] || "bg-gray-700 text-gray-300"}`}>
                    {binding.role}
                  </span>
                  <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs font-mono text-gray-300">
                    {binding.tag}
                  </span>
                </div>
                {binding.actor_name && (
                  <div className="text-xs text-gray-400 mt-0.5">
                    played by <span className="text-gray-300">{binding.actor_name}</span>
                  </div>
                )}
                {binding.behavioral_notes && (
                  <div className="text-xs text-gray-500 mt-1 truncate">{binding.behavioral_notes}</div>
                )}
              </div>

              {/* Actions */}
              <div className="flex gap-1 shrink-0">
                <button
                  onClick={() => handleStartEdit(binding)}
                  className="rounded px-2 py-1 text-xs text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDelete(binding.id)}
                  className="rounded px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-gray-800 transition-colors"
                >
                  Remove
                </button>
              </div>
            </div>
          )
        )}
      </div>

      {/* Asset Picker Modal */}
      <AssetPicker
        isOpen={pickerOpen}
        onClose={() => setPickerOpen(false)}
        assetType="actor"
        onSelect={handleActorSelected}
        excludeIds={excludeActorIds}
      />
    </div>
  );
}
