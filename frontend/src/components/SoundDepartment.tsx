import { useState, useEffect, useCallback } from "react";
import type {
  ScoreThemeResponse,
  SFXItemResponse,
} from "../api/types.ts";
import {
  listScoreThemes,
  createScoreTheme,
  updateScoreTheme,
  deleteScoreTheme,
  listSFXItems,
  createSFXItem,
  updateSFXItem,
  deleteSFXItem,
} from "../api/client.ts";

interface SoundDepartmentProps {
  productionBibleId: string;
}

const SFX_CATEGORIES = ["IMPACT", "MECHANICAL", "NATURAL", "UI", "FOLEY", "AMBIENCE"] as const;

export function SoundDepartment({ productionBibleId }: SoundDepartmentProps) {
  const [scoreThemes, setScoreThemes] = useState<ScoreThemeResponse[]>([]);
  const [sfxItems, setSfxItems] = useState<SFXItemResponse[]>([]);
  const [selectedThemeId, setSelectedThemeId] = useState<string | null>(null);
  const [selectedSfxId, setSelectedSfxId] = useState<string | null>(null);
  const [sfxCategoryFilter, setSfxCategoryFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create forms
  const [showCreateTheme, setShowCreateTheme] = useState(false);
  const [newThemeName, setNewThemeName] = useState("");
  const [creatingTheme, setCreatingTheme] = useState(false);

  const [showCreateSfx, setShowCreateSfx] = useState(false);
  const [newSfxName, setNewSfxName] = useState("");
  const [newSfxCategory, setNewSfxCategory] = useState<string>("FOLEY");
  const [creatingSfx, setCreatingSfx] = useState(false);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const [themes, sfx] = await Promise.all([
        listScoreThemes(productionBibleId),
        listSFXItems(productionBibleId),
      ]);
      setScoreThemes(themes);
      setSfxItems(sfx);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [productionBibleId]);

  useEffect(() => {
    load();
  }, [load]);

  // Reload SFX when category filter changes
  const handleCategoryFilter = async (cat: string | null) => {
    setSfxCategoryFilter(cat);
    try {
      const sfx = await listSFXItems(productionBibleId, cat ?? undefined);
      setSfxItems(sfx);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // --- Score Theme handlers ---
  const handleCreateTheme = async () => {
    if (!newThemeName.trim()) return;
    setCreatingTheme(true);
    setError(null);
    try {
      const theme = await createScoreTheme(productionBibleId, { name: newThemeName.trim() });
      setScoreThemes((prev) => [...prev, theme]);
      setSelectedThemeId(theme.score_theme_id);
      setNewThemeName("");
      setShowCreateTheme(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingTheme(false);
    }
  };

  const handleUpdateTheme = async (themeId: string, data: Record<string, unknown>) => {
    try {
      const updated = await updateScoreTheme(themeId, data);
      setScoreThemes((prev) => prev.map((t) => (t.score_theme_id === themeId ? updated : t)));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteTheme = async (themeId: string) => {
    if (!confirm("Delete this score theme?")) return;
    try {
      await deleteScoreTheme(themeId);
      setScoreThemes((prev) => prev.filter((t) => t.score_theme_id !== themeId));
      if (selectedThemeId === themeId) setSelectedThemeId(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  // --- SFX handlers ---
  const handleCreateSfx = async () => {
    if (!newSfxName.trim()) return;
    setCreatingSfx(true);
    setError(null);
    try {
      const sfx = await createSFXItem(productionBibleId, { name: newSfxName.trim(), category: newSfxCategory });
      setSfxItems((prev) => [...prev, sfx]);
      setSelectedSfxId(sfx.sfx_item_id);
      setNewSfxName("");
      setNewSfxCategory("FOLEY");
      setShowCreateSfx(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingSfx(false);
    }
  };

  const handleUpdateSfx = async (sfxId: string, data: Record<string, unknown>) => {
    try {
      const updated = await updateSFXItem(sfxId, data);
      setSfxItems((prev) => prev.map((s) => (s.sfx_item_id === sfxId ? updated : s)));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteSfx = async (sfxId: string) => {
    if (!confirm("Delete this SFX item?")) return;
    try {
      await deleteSFXItem(sfxId);
      setSfxItems((prev) => prev.filter((s) => s.sfx_item_id !== sfxId));
      if (selectedSfxId === sfxId) setSelectedSfxId(null);
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

      {/* Score Themes Section */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-medium text-gray-200">Score Themes</h3>
          <button
            onClick={() => setShowCreateTheme(true)}
            className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
          >
            + Add Theme
          </button>
        </div>

        {showCreateTheme && (
          <div className="mb-3 rounded border border-gray-700 bg-gray-800 p-3 space-y-2">
            <input
              type="text"
              value={newThemeName}
              onChange={(e) => setNewThemeName(e.target.value)}
              placeholder="Theme name"
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
              autoFocus
            />
            <div className="flex gap-1">
              <button
                onClick={handleCreateTheme}
                disabled={creatingTheme || !newThemeName.trim()}
                className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
              >
                {creatingTheme ? "..." : "Create"}
              </button>
              <button
                onClick={() => { setShowCreateTheme(false); setNewThemeName(""); }}
                className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {scoreThemes.length === 0 ? (
          <div className="flex items-center justify-center py-8 rounded-lg border border-dashed border-gray-700">
            <p className="text-sm text-gray-500">No score themes yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {scoreThemes.map((theme) => (
              <ScoreThemeItem
                key={theme.score_theme_id}
                theme={theme}
                isExpanded={selectedThemeId === theme.score_theme_id}
                onToggle={() => setSelectedThemeId(selectedThemeId === theme.score_theme_id ? null : theme.score_theme_id)}
                onUpdate={handleUpdateTheme}
                onDelete={handleDeleteTheme}
              />
            ))}
          </div>
        )}
      </div>

      {/* SFX Library Section */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-medium text-gray-200">SFX Library</h3>
          <button
            onClick={() => setShowCreateSfx(true)}
            className="text-xs px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-500"
          >
            + Add SFX
          </button>
        </div>

        {showCreateSfx && (
          <div className="mb-3 rounded border border-gray-700 bg-gray-800 p-3 space-y-2">
            <input
              type="text"
              value={newSfxName}
              onChange={(e) => setNewSfxName(e.target.value)}
              placeholder="SFX name"
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:border-blue-500 outline-none"
              autoFocus
            />
            <select
              value={newSfxCategory}
              onChange={(e) => setNewSfxCategory(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
            >
              {SFX_CATEGORIES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <div className="flex gap-1">
              <button
                onClick={handleCreateSfx}
                disabled={creatingSfx || !newSfxName.trim()}
                className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
              >
                {creatingSfx ? "..." : "Create"}
              </button>
              <button
                onClick={() => { setShowCreateSfx(false); setNewSfxName(""); }}
                className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Category filter pills */}
        <div className="flex gap-1.5 mb-3 flex-wrap">
          <button
            onClick={() => handleCategoryFilter(null)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              sfxCategoryFilter === null ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            All
          </button>
          {SFX_CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => handleCategoryFilter(cat)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                sfxCategoryFilter === cat ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {sfxItems.length === 0 ? (
          <div className="flex items-center justify-center py-8 rounded-lg border border-dashed border-gray-700">
            <p className="text-sm text-gray-500">
              {sfxCategoryFilter ? `No ${sfxCategoryFilter} SFX items` : "No SFX items yet"}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {sfxItems.map((sfx) => (
              <SFXItemRow
                key={sfx.sfx_item_id}
                sfx={sfx}
                isExpanded={selectedSfxId === sfx.sfx_item_id}
                onToggle={() => setSelectedSfxId(selectedSfxId === sfx.sfx_item_id ? null : sfx.sfx_item_id)}
                onUpdate={handleUpdateSfx}
                onDelete={handleDeleteSfx}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// --- Sub-components ---

function ScoreThemeItem({
  theme,
  isExpanded,
  onToggle,
  onUpdate,
  onDelete,
}: {
  theme: ScoreThemeResponse;
  isExpanded: boolean;
  onToggle: () => void;
  onUpdate: (id: string, data: Record<string, unknown>) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [name, setName] = useState(theme.name);
  const [moodDescriptors, setMoodDescriptors] = useState(theme.mood_descriptors?.join(", ") ?? "");
  const [tempoNotes, setTempoNotes] = useState(theme.tempo_notes ?? "");
  const [usageNotes, setUsageNotes] = useState(theme.usage_notes ?? "");
  const [generationPrompt, setGenerationPrompt] = useState(theme.generation_prompt ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(theme.name);
    setMoodDescriptors(theme.mood_descriptors?.join(", ") ?? "");
    setTempoNotes(theme.tempo_notes ?? "");
    setUsageNotes(theme.usage_notes ?? "");
    setGenerationPrompt(theme.generation_prompt ?? "");
  }, [theme]);

  const handleSave = async () => {
    setSaving(true);
    await onUpdate(theme.score_theme_id, {
      name: name.trim(),
      mood_descriptors: moodDescriptors ? moodDescriptors.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
      tempo_notes: tempoNotes || undefined,
      usage_notes: usageNotes || undefined,
      generation_prompt: generationPrompt || undefined,
    });
    setSaving(false);
  };

  const categoryColor = (cat: string) => {
    const colors: Record<string, string> = {
      tense: "bg-red-900 text-red-300",
      hopeful: "bg-green-900 text-green-300",
      melancholic: "bg-indigo-900 text-indigo-300",
      energetic: "bg-yellow-900 text-yellow-300",
    };
    return colors[cat.toLowerCase()] ?? "bg-gray-700 text-gray-300";
  };

  return (
    <div className={`rounded-lg border bg-gray-900/50 transition-colors ${isExpanded ? "border-blue-700/50" : "border-gray-800"}`}>
      {/* Header row */}
      <div
        onClick={onToggle}
        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-800/50"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium text-gray-200">{theme.name}</span>
          {theme.mood_descriptors && theme.mood_descriptors.length > 0 && (
            <div className="flex gap-1 flex-wrap">
              {theme.mood_descriptors.slice(0, 3).map((d, i) => (
                <span key={i} className={`text-[10px] px-1.5 py-0.5 rounded ${categoryColor(d)}`}>{d}</span>
              ))}
              {theme.mood_descriptors.length > 3 && (
                <span className="text-[10px] text-gray-500">+{theme.mood_descriptors.length - 3}</span>
              )}
            </div>
          )}
        </div>
        <span className="text-xs text-gray-500">{isExpanded ? "collapse" : "expand"}</span>
      </div>

      {/* Expanded editor */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-gray-800 pt-3">
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
            <label className="block text-xs text-gray-400 mb-1">Mood Descriptors (comma-separated)</label>
            <input
              type="text"
              value={moodDescriptors}
              onChange={(e) => setMoodDescriptors(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
              placeholder="tense, hopeful, melancholic"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Tempo Notes</label>
            <input
              type="text"
              value={tempoNotes}
              onChange={(e) => setTempoNotes(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
              placeholder="e.g. 120 BPM, building"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Usage Notes</label>
            <input
              type="text"
              value={usageNotes}
              onChange={(e) => setUsageNotes(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
              placeholder="e.g. Act 2 climax, transitions"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Generation Prompt</label>
            <textarea
              value={generationPrompt}
              onChange={(e) => setGenerationPrompt(e.target.value)}
              rows={2}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
              placeholder="Music generation prompt..."
            />
          </div>

          <div className="flex items-center justify-between">
            <button
              disabled
              title="Music adapter coming soon"
              className="text-xs px-3 py-1.5 rounded bg-gray-700 text-gray-500 cursor-not-allowed"
            >
              Generate Music
            </button>
            <div className="flex gap-2">
              <button
                onClick={() => onDelete(theme.score_theme_id)}
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
        </div>
      )}
    </div>
  );
}

function SFXItemRow({
  sfx,
  isExpanded,
  onToggle,
  onUpdate,
  onDelete,
}: {
  sfx: SFXItemResponse;
  isExpanded: boolean;
  onToggle: () => void;
  onUpdate: (id: string, data: Record<string, unknown>) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [name, setName] = useState(sfx.name);
  const [category, setCategory] = useState(sfx.category);
  const [generationPrompt, setGenerationPrompt] = useState(sfx.generation_prompt ?? "");
  const [tags, setTags] = useState(sfx.tags?.join(", ") ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(sfx.name);
    setCategory(sfx.category);
    setGenerationPrompt(sfx.generation_prompt ?? "");
    setTags(sfx.tags?.join(", ") ?? "");
  }, [sfx]);

  const handleSave = async () => {
    setSaving(true);
    await onUpdate(sfx.sfx_item_id, {
      name: name.trim(),
      category,
      generation_prompt: generationPrompt || undefined,
      tags: tags ? tags.split(",").map((t) => t.trim()).filter(Boolean) : undefined,
    });
    setSaving(false);
  };

  const categoryBadgeColor = (cat: string) => {
    const colors: Record<string, string> = {
      IMPACT: "bg-red-900 text-red-300",
      MECHANICAL: "bg-orange-900 text-orange-300",
      NATURAL: "bg-green-900 text-green-300",
      UI: "bg-blue-900 text-blue-300",
      FOLEY: "bg-purple-900 text-purple-300",
      AMBIENCE: "bg-teal-900 text-teal-300",
    };
    return colors[cat] ?? "bg-gray-700 text-gray-300";
  };

  return (
    <div className={`rounded-lg border bg-gray-900/50 transition-colors ${isExpanded ? "border-blue-700/50" : "border-gray-800"}`}>
      {/* Header row */}
      <div
        onClick={onToggle}
        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-800/50"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium text-gray-200">{sfx.name}</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded ${categoryBadgeColor(sfx.category)}`}>{sfx.category}</span>
          {sfx.tags && sfx.tags.length > 0 && (
            <div className="flex gap-1">
              {sfx.tags.slice(0, 2).map((t, i) => (
                <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-400">{t}</span>
              ))}
              {sfx.tags.length > 2 && <span className="text-[10px] text-gray-500">+{sfx.tags.length - 2}</span>}
            </div>
          )}
        </div>
        <span className="text-xs text-gray-500">{isExpanded ? "collapse" : "expand"}</span>
      </div>

      {/* Expanded editor */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-gray-800 pt-3">
          <div className="grid grid-cols-2 gap-3">
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
              <label className="block text-xs text-gray-400 mb-1">Category</label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value as SFXItemResponse["category"])}
                className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
              >
                {SFX_CATEGORIES.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Generation Prompt</label>
            <textarea
              value={generationPrompt}
              onChange={(e) => setGenerationPrompt(e.target.value)}
              rows={2}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
              placeholder="SFX generation prompt..."
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Tags (comma-separated)</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-300 focus:border-blue-500 outline-none"
              placeholder="door, slam, impact"
            />
          </div>

          <div className="flex justify-between">
            <button
              onClick={() => onDelete(sfx.sfx_item_id)}
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
      )}
    </div>
  );
}
