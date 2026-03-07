import { useState, useEffect, useRef } from "react";
import { useLocation } from "wouter";
import type {
  ActorListItem,
  LibrarySetListItem,
  LibraryPropListItem,
  SoundAssetListItem,
} from "../api/types.ts";
import {
  listActors,
  createActor,
  listLibrarySets,
  createLibrarySet,
  listLibraryProps,
  createLibraryProp,
  listSoundAssets,
  createSoundAsset,
} from "../api/client.ts";

type Tab = "actors" | "sets" | "props" | "sounds";

const TABS: { id: Tab; label: string }[] = [
  { id: "actors", label: "Actors" },
  { id: "sets", label: "Sets" },
  { id: "props", label: "Props" },
  { id: "sounds", label: "Sound Assets" },
];

const SOUND_CATEGORIES = ["ALL", "SCORE_THEME", "SFX", "AMBIENCE", "FOLEY", "UI"] as const;

export function AssetLibrary() {
  const [, navigate] = useLocation();
  const [activeTab, setActiveTab] = useState<Tab>("actors");
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  // Debounce search
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setDebouncedSearch(searchQuery), 300);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [searchQuery]);

  // Reset search on tab change
  useEffect(() => {
    setSearchQuery("");
    setDebouncedSearch("");
  }, [activeTab]);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Asset Library</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 transition-colors"
        >
          + Create
        </button>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-4">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium rounded-full transition-colors ${
              activeTab === tab.id
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="mb-4">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder={`Search ${TABS.find((t) => t.id === activeTab)?.label ?? ""}...`}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:border-blue-500 outline-none placeholder-gray-500"
        />
      </div>

      {/* Tab content */}
      {activeTab === "actors" && (
        <ActorsGrid search={debouncedSearch} refreshKey={refreshKey} onSelect={(id) => navigate(`/asset-library/actors/${id}`)} />
      )}
      {activeTab === "sets" && (
        <SetsGrid search={debouncedSearch} refreshKey={refreshKey} onSelect={(id) => navigate(`/asset-library/sets/${id}`)} />
      )}
      {activeTab === "props" && (
        <PropsGrid search={debouncedSearch} refreshKey={refreshKey} onSelect={(id) => navigate(`/asset-library/props/${id}`)} />
      )}
      {activeTab === "sounds" && (
        <SoundsList search={debouncedSearch} refreshKey={refreshKey} />
      )}

      {/* Create modal */}
      {showCreateModal && (
        <CreateModal
          activeTab={activeTab}
          onClose={() => setShowCreateModal(false)}
          onCreated={(id) => {
            setShowCreateModal(false);
            if (activeTab === "actors") {
              navigate(`/asset-library/actors/${id}`);
            } else if (activeTab === "sets") {
              navigate(`/asset-library/sets/${id}`);
            } else if (activeTab === "props") {
              navigate(`/asset-library/props/${id}`);
            } else {
              setRefreshKey((k) => k + 1);
            }
          }}
        />
      )}
    </div>
  );
}

// ============================================================================
// Actors Grid
// ============================================================================

function ActorsGrid({ search, refreshKey, onSelect }: { search: string; refreshKey: number; onSelect: (id: string) => void }) {
  const [actors, setActors] = useState<ActorListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listActors(search || undefined)
      .then((data) => { if (!cancelled) setActors(data); })
      .catch(() => { if (!cancelled) setActors([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [search, refreshKey]);

  if (loading) return <LoadingSpinner />;
  if (actors.length === 0) return <EmptyState entity="actors" />;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {actors.map((actor) => (
        <button
          key={actor.id}
          onClick={() => onSelect(actor.id)}
          className="text-left rounded-lg border border-gray-800 bg-gray-900/50 p-4 hover:border-gray-600 transition-colors"
        >
          <div className="flex gap-3">
            {actor.primary_ref_url ? (
              <img
                src={actor.primary_ref_url}
                alt={actor.name}
                className="w-16 h-16 rounded-lg object-cover border border-gray-700 flex-shrink-0"
              />
            ) : (
              <div className="w-16 h-16 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center flex-shrink-0">
                <span className="text-gray-600 text-xl">?</span>
              </div>
            )}
            <div className="min-w-0 flex-1">
              <h3 className="text-sm font-medium text-gray-200 truncate">{actor.name}</h3>
              {actor.description && (
                <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{actor.description}</p>
              )}
              <div className="flex items-center gap-2 mt-2">
                {actor.ref_count > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">
                    {actor.ref_count} ref{actor.ref_count !== 1 ? "s" : ""}
                  </span>
                )}
                {actor.binding_count > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300">
                    {actor.binding_count} binding{actor.binding_count !== 1 ? "s" : ""}
                  </span>
                )}
              </div>
              {actor.prompt_tags && actor.prompt_tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {actor.prompt_tags.slice(0, 3).map((tag) => (
                    <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-300">
                      {tag}
                    </span>
                  ))}
                  {actor.prompt_tags.length > 3 && (
                    <span className="text-[10px] text-gray-500">+{actor.prompt_tags.length - 3}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Sets Grid
// ============================================================================

function SetsGrid({ search, refreshKey, onSelect }: { search: string; refreshKey: number; onSelect: (id: string) => void }) {
  const [sets, setSets] = useState<LibrarySetListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listLibrarySets(search || undefined)
      .then((data) => { if (!cancelled) setSets(data); })
      .catch(() => { if (!cancelled) setSets([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [search, refreshKey]);

  if (loading) return <LoadingSpinner />;
  if (sets.length === 0) return <EmptyState entity="sets" />;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {sets.map((set) => (
        <button
          key={set.id}
          onClick={() => onSelect(set.id)}
          className="text-left rounded-lg border border-gray-800 bg-gray-900/50 p-4 hover:border-gray-600 transition-colors"
        >
          <div className="flex gap-3">
            {set.primary_ref_url ? (
              <img
                src={set.primary_ref_url}
                alt={set.name}
                className="w-16 h-16 rounded-lg object-cover border border-gray-700 flex-shrink-0"
              />
            ) : (
              <div className="w-16 h-16 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center flex-shrink-0">
                <span className="text-gray-600 text-lg">SET</span>
              </div>
            )}
            <div className="min-w-0 flex-1">
              <h3 className="text-sm font-medium text-gray-200 truncate">{set.name}</h3>
              {set.description && (
                <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{set.description}</p>
              )}
              <div className="flex items-center gap-2 mt-2">
                {set.ref_count > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">
                    {set.ref_count} ref{set.ref_count !== 1 ? "s" : ""}
                  </span>
                )}
                {set.binding_count > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300">
                    {set.binding_count} binding{set.binding_count !== 1 ? "s" : ""}
                  </span>
                )}
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Props Grid
// ============================================================================

function PropsGrid({ search, refreshKey, onSelect }: { search: string; refreshKey: number; onSelect: (id: string) => void }) {
  const [props, setProps] = useState<LibraryPropListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listLibraryProps(search || undefined)
      .then((data) => { if (!cancelled) setProps(data); })
      .catch(() => { if (!cancelled) setProps([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [search, refreshKey]);

  if (loading) return <LoadingSpinner />;
  if (props.length === 0) return <EmptyState entity="props" />;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {props.map((prop) => (
        <button
          key={prop.id}
          onClick={() => onSelect(prop.id)}
          className="text-left rounded-lg border border-gray-800 bg-gray-900/50 p-4 hover:border-gray-600 transition-colors"
        >
          <div className="flex gap-3">
            {prop.primary_ref_url ? (
              <img
                src={prop.primary_ref_url}
                alt={prop.name}
                className="w-16 h-16 rounded-lg object-cover border border-gray-700 flex-shrink-0"
              />
            ) : (
              <div className="w-16 h-16 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center flex-shrink-0">
                <span className="text-gray-600 text-lg">PROP</span>
              </div>
            )}
            <div className="min-w-0 flex-1">
              <h3 className="text-sm font-medium text-gray-200 truncate">{prop.name}</h3>
              {prop.description && (
                <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{prop.description}</p>
              )}
              <div className="flex items-center gap-2 mt-2">
                {prop.binding_count > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300">
                    {prop.binding_count} binding{prop.binding_count !== 1 ? "s" : ""}
                  </span>
                )}
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Sound Assets List
// ============================================================================

function SoundsList({ search, refreshKey }: { search: string; refreshKey: number }) {
  const [sounds, setSounds] = useState<SoundAssetListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const cat = categoryFilter !== "ALL" ? categoryFilter : undefined;
    listSoundAssets(search || undefined, cat)
      .then((data) => { if (!cancelled) setSounds(data); })
      .catch(() => { if (!cancelled) setSounds([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [search, categoryFilter, refreshKey]);

  const categoryBadgeColor = (cat: string) => {
    switch (cat) {
      case "SCORE_THEME": return "bg-yellow-900/50 text-yellow-300";
      case "SFX": return "bg-red-900/50 text-red-300";
      case "AMBIENCE": return "bg-green-900/50 text-green-300";
      case "FOLEY": return "bg-orange-900/50 text-orange-300";
      case "UI": return "bg-cyan-900/50 text-cyan-300";
      default: return "bg-gray-700 text-gray-300";
    }
  };

  return (
    <div>
      {/* Category filter pills */}
      <div className="flex gap-1 mb-4">
        {SOUND_CATEGORIES.map((cat) => (
          <button
            key={cat}
            onClick={() => setCategoryFilter(cat)}
            className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
              categoryFilter === cat
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            {cat.replace("_", " ")}
          </button>
        ))}
      </div>

      {loading ? (
        <LoadingSpinner />
      ) : sounds.length === 0 ? (
        <EmptyState entity="sound assets" />
      ) : (
        <div className="rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-900/80 text-left">
                <th className="px-4 py-2 text-xs font-medium text-gray-400">Name</th>
                <th className="px-4 py-2 text-xs font-medium text-gray-400">Category</th>
                <th className="px-4 py-2 text-xs font-medium text-gray-400">Subcategory</th>
                <th className="px-4 py-2 text-xs font-medium text-gray-400 text-center">Audio</th>
                <th className="px-4 py-2 text-xs font-medium text-gray-400 text-right">Bindings</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {sounds.map((sound) => (
                <tr key={sound.id} className="hover:bg-gray-800/50">
                  <td className="px-4 py-2.5">
                    <span className="text-sm text-gray-200">{sound.name}</span>
                    {sound.description && (
                      <p className="text-xs text-gray-500 mt-0.5 truncate max-w-xs">{sound.description}</p>
                    )}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={`text-[10px] px-2 py-0.5 rounded ${categoryBadgeColor(sound.category)}`}>
                      {sound.category}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-xs text-gray-400">
                    {sound.subcategory ?? "--"}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {sound.has_audio ? (
                      <span className="text-green-400 text-xs">Yes</span>
                    ) : (
                      <span className="text-gray-600 text-xs">--</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-right text-xs text-gray-400">
                    {sound.binding_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Create Modal
// ============================================================================

function CreateModal({
  activeTab,
  onClose,
  onCreated,
}: {
  activeTab: Tab;
  onClose: () => void;
  onCreated: (id: string) => void;
}) {
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sound-specific
  const [soundCategory, setSoundCategory] = useState("SFX");

  const entityLabel = activeTab === "actors" ? "Actor" : activeTab === "sets" ? "Set" : activeTab === "props" ? "Prop" : "Sound Asset";

  const handleCreate = async () => {
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      let id: string;
      switch (activeTab) {
        case "actors": {
          const res = await createActor({ name: name.trim() });
          id = res.id;
          break;
        }
        case "sets": {
          const res = await createLibrarySet({ name: name.trim() });
          id = res.id;
          break;
        }
        case "props": {
          const res = await createLibraryProp({ name: name.trim() });
          id = res.id;
          break;
        }
        case "sounds": {
          const res = await createSoundAsset({ name: name.trim(), category: soundCategory });
          id = res.id;
          break;
        }
      }
      onCreated(id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative bg-gray-900 border border-gray-700 rounded-xl p-6 w-full max-w-md shadow-xl">
        <h2 className="text-lg font-semibold text-white mb-4">Create {entityLabel}</h2>

        {error && (
          <div className="mb-3 rounded bg-red-900/30 border border-red-700 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              placeholder={`${entityLabel} name`}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:border-blue-500 outline-none"
              autoFocus
            />
          </div>

          {activeTab === "sounds" && (
            <div>
              <label className="block text-xs text-gray-400 mb-1">Category</label>
              <select
                value={soundCategory}
                onChange={(e) => setSoundCategory(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 focus:border-blue-500 outline-none"
              >
                {SOUND_CATEGORIES.filter((c) => c !== "ALL").map((c) => (
                  <option key={c} value={c}>{c.replace("_", " ")}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={creating || !name.trim()}
            className="px-4 py-2 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {creating ? "Creating..." : "Create"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Shared components
// ============================================================================

function LoadingSpinner() {
  return (
    <div className="flex justify-center py-12">
      <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

function EmptyState({ entity }: { entity: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 rounded-lg border border-dashed border-gray-700">
      <p className="text-sm text-gray-500">No {entity} found</p>
      <p className="text-xs text-gray-600 mt-1">Create one to get started</p>
    </div>
  );
}
