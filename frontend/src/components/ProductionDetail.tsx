import { useCallback, useEffect, useState, useMemo } from "react";
import {
  getProduction,
  updateProduction,
  getProductionMaster,
  renderProductionMaster,
  getSoundDeck,
  generateSoundDeck,
  generateSoundDeckAudio,
  mixSoundDeck,
  generateSoundCueAudio,
  listScenes,
  listSequences,
  createSequence,
  batchAddScenesToProduction,
  removeSceneFromProduction,
  type ProductionMasterResponse,
  type ProductionResponse,
} from "../api/client.ts";
import type { SceneListItem, SequenceResponse, SoundDeckResponse, SoundEffectCueResponse } from "../api/types.ts";
import { ScenePickerModal } from "./ScenePickerModal.tsx";
import { SequencedSceneList } from "./SequencedSceneList.tsx";
import { ScreenplayEditor } from "./ScreenplayEditor.tsx";

type ProductionTab = "scenes" | "screenplay" | "sound";

interface ProductionDetailProps {
  productionId: string;
  onViewScene: (id: string) => void;
}

function statusCode(error: unknown): number | null {
  if (typeof error !== "object" || error === null || !("status" in error)) {
    return null;
  }
  const value = (error as { status?: unknown }).status;
  return typeof value === "number" ? value : null;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) {
    return "Unknown duration";
  }
  const rounded = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(rounded / 60);
  const remaining = rounded % 60;
  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

export function ProductionDetail({ productionId, onViewScene }: ProductionDetailProps) {
  const [production, setProduction] = useState<ProductionResponse | null>(null);
  const [master, setMaster] = useState<ProductionMasterResponse | null>(null);
  const [scenes, setScenes] = useState<SceneListItem[]>([]);
  const [sequences, setSequences] = useState<SequenceResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [renderingMaster, setRenderingMaster] = useState(false);
  const [masterError, setMasterError] = useState<string | null>(null);
  const [soundDeck, setSoundDeck] = useState<SoundDeckResponse | null>(null);
  const [soundDeckBusy, setSoundDeckBusy] = useState<string | null>(null);
  const [soundDeckError, setSoundDeckError] = useState<string | null>(null);
  const [videoVersion, setVideoVersion] = useState(() => Date.now());
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [showPicker, setShowPicker] = useState(false);
  const [activeTab, setActiveTab] = useState<ProductionTab>("scenes");

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const [prod, scenesData, seqs, masterData, soundData] = await Promise.all([
        getProduction(productionId),
        listScenes({ per_page: 96, view: "cards" }),
        listSequences(productionId).catch(() => [] as SequenceResponse[]),
        getProductionMaster(productionId).catch((err) => {
          if (statusCode(err) === 404) {
            return null;
          }
          throw err;
        }),
        getSoundDeck(productionId).catch(() => null),
      ]);
      setProduction(prod);
      setMaster(masterData);
      setSoundDeck(soundData);
      setEditName(prod.name);
      setEditDesc(prod.description || "");
      setScenes(scenesData.items.filter((s) => s.production_id === productionId));
      setSequences(seqs);
      setError(null);
      setMasterError(null);
      setSoundDeckError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load production");
    } finally {
      setLoading(false);
    }
  }, [productionId]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleSave() {
    if (!editName.trim()) return;
    try {
      const updated = await updateProduction(productionId, {
        name: editName.trim(),
        description: editDesc.trim() || undefined,
      });
      setProduction(updated);
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update production");
    }
  }

  async function handleBatchAdd(sceneIds: string[]) {
    try {
      await batchAddScenesToProduction(productionId, sceneIds);
      setShowPicker(false);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add scenes");
    }
  }

  async function handleRemoveScene(sceneId: string) {
    try {
      await removeSceneFromProduction(productionId, sceneId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove scene");
    }
  }

  async function handleRenderMaster() {
    try {
      setRenderingMaster(true);
      setMasterError(null);
      const rendered = await renderProductionMaster(productionId);
      setMaster(rendered);
      setVideoVersion(Date.now());
    } catch (err) {
      setMasterError(err instanceof Error ? err.message : "Failed to render final master");
    } finally {
      setRenderingMaster(false);
    }
  }

  async function runSoundDeckAction(
    label: string,
    action: () => Promise<SoundDeckResponse>,
  ) {
    try {
      setSoundDeckBusy(label);
      setSoundDeckError(null);
      const nextDeck = await action();
      setSoundDeck(nextDeck);
    } catch (err) {
      setSoundDeckError(err instanceof Error ? err.message : `Failed to ${label}`);
    } finally {
      setSoundDeckBusy(null);
    }
  }

  async function handleGenerateSoundDeck() {
    await runSoundDeckAction("generate", async () => {
      const result = await generateSoundDeck(productionId);
      return result.sound_deck;
    });
  }

  async function handleGenerateSoundAudio() {
    await runSoundDeckAction("generate audio", async () => {
      const result = await generateSoundDeckAudio(productionId);
      return result.sound_deck;
    });
  }

  async function handleMixSoundDeck() {
    await runSoundDeckAction("mix", async () => {
      const result = await mixSoundDeck(productionId);
      return result.sound_deck;
    });
  }

  async function handleGenerateCue(cue: SoundEffectCueResponse) {
    try {
      setSoundDeckBusy(cue.id);
      setSoundDeckError(null);
      const updated = await generateSoundCueAudio(cue.id);
      setSoundDeck((current) => current ? {
        ...current,
        cues: current.cues.map((item) => item.id === updated.id ? updated : item),
      } : current);
    } catch (err) {
      setSoundDeckError(err instanceof Error ? err.message : "Failed to generate cue audio");
    } finally {
      setSoundDeckBusy(null);
    }
  }

  const linkedSceneIds = useMemo(
    () => new Set(scenes.map((s) => s.scene_id)),
    [scenes],
  );

  if (loading) {
    return <div className="text-center py-12 text-gray-400">Loading production...</div>;
  }

  if (!production) {
    return <div className="text-center py-12 text-red-400">Production not found</div>;
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-md bg-red-900/50 p-3 text-sm text-red-300">{error}</div>
      )}

      <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
        {editing ? (
          <div className="space-y-3">
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              className="w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white"
              autoFocus
            />
            <textarea
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              className="w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white"
              rows={2}
              placeholder="Description"
            />
            <div className="flex gap-2">
              <button onClick={handleSave} className="rounded-md bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-500">
                Save
              </button>
              <button onClick={() => setEditing(false)} className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600">
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-xl font-bold text-white">{production.name}</h1>
              {production.description && (
                <p className="text-sm text-gray-400 mt-1">{production.description}</p>
              )}
              <p className="text-xs text-gray-500 mt-2">
                {production.scene_count} scene{production.scene_count !== 1 ? "s" : ""} · Created{" "}
                {new Date(production.created_at).toLocaleDateString()}
              </p>
            </div>
            <button
              onClick={() => setEditing(true)}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600"
            >
              Edit
            </button>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-white">Final Master</h2>
            <p className="mt-1 text-sm text-gray-400">
              {master
                ? `${formatDuration(master.duration_seconds)} · ${master.scene_count} scene${master.scene_count !== 1 ? "s" : ""} · ${master.audio_stem_count} audio stem${master.audio_stem_count !== 1 ? "s" : ""}`
                : "Render the mixed production video after scenes, voice stems, and sound stems are ready."}
            </p>
          </div>
          <div className="flex gap-2">
            {master && (
              <a
                href={master.video_url}
                target="_blank"
                rel="noreferrer"
                className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600"
              >
                Open MP4
              </a>
            )}
            <button
              onClick={handleRenderMaster}
              disabled={renderingMaster}
              className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {renderingMaster ? "Rendering..." : master ? "Re-render Master" : "Render Master"}
            </button>
          </div>
        </div>
        {masterError && (
          <div className="mt-3 rounded-md bg-red-900/50 p-3 text-sm text-red-300">{masterError}</div>
        )}
        {master && (
          <video
            className="mt-4 aspect-video w-full rounded-md bg-black"
            controls
            preload="metadata"
            src={`${master.video_url}?v=${videoVersion}`}
          />
        )}
      </div>

      {/* Tab navigation: Scenes | Screenplay | Sound */}
      <div className="flex gap-1">
        <button
          onClick={() => setActiveTab("scenes")}
          className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
            activeTab === "scenes"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Scenes
        </button>
        <button
          onClick={() => setActiveTab("screenplay")}
          className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
            activeTab === "screenplay"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Screenplay
        </button>
        <button
          onClick={() => setActiveTab("sound")}
          className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
            activeTab === "sound"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Sound
        </button>
      </div>

      {/* Tab content */}
      {activeTab === "scenes" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Scenes</h2>
            <div className="flex gap-2">
              {sequences.length === 0 && (
                <button
                  onClick={async () => {
                    try {
                      const seq = await createSequence(productionId, { title: "Chapter 1" });
                      setSequences([seq]);
                    } catch {
                      setError("Failed to create sequence");
                    }
                  }}
                  className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-600"
                >
                  + Create Sequence
                </button>
              )}
              <button
                onClick={() => setShowPicker(true)}
                className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500"
              >
                Add Scenes
              </button>
            </div>
          </div>

          {scenes.length === 0 ? (
            <p className="text-sm text-gray-500 py-4">No scenes in this production yet.</p>
          ) : sequences.length > 0 ? (
            <SequencedSceneList
              productionId={productionId}
              scenes={scenes}
              onViewScene={onViewScene}
              onRefresh={load}
              onRemoveScene={handleRemoveScene}
            />
          ) : (
            <div className="space-y-2">
              {scenes.map((scene) => (
                <div
                  key={scene.scene_id}
                  className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800/50 p-3 hover:bg-gray-800 transition-colors cursor-pointer"
                  onClick={() => onViewScene(scene.scene_id)}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-white truncate">
                        {scene.title || scene.prompt.slice(0, 80)}
                      </p>
                      {scene.screenplay_breakdown_index != null && (
                        <span className="inline-flex items-center rounded-full bg-blue-900 px-2 py-0.5 text-xs font-medium text-blue-300 whitespace-nowrap">
                          Screenplay
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {scene.status} · {new Date(scene.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveScene(scene.scene_id);
                    }}
                    className="ml-3 text-xs text-gray-500 hover:text-red-400"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === "sound" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-white">Sound Deck</h2>
              <p className="mt-1 text-sm text-gray-400">
                {soundDeck
                  ? `${soundDeck.cues.length} cue${soundDeck.cues.length !== 1 ? "s" : ""} · ${soundDeck.mix_artifacts.length} SFX stem${soundDeck.mix_artifacts.length !== 1 ? "s" : ""}`
                  : "Generate and mix sound effects for the production timeline."}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleGenerateSoundDeck}
                disabled={soundDeckBusy !== null}
                className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {soundDeckBusy === "generate" ? "Generating..." : "Generate Cues"}
              </button>
              <button
                onClick={handleGenerateSoundAudio}
                disabled={soundDeckBusy !== null || !soundDeck?.cues.length}
                className="rounded-md bg-gray-700 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {soundDeckBusy === "generate audio" ? "Generating..." : "Generate Audio"}
              </button>
              <button
                onClick={handleMixSoundDeck}
                disabled={soundDeckBusy !== null || !soundDeck?.cues.some((cue) => cue.audio_path)}
                className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {soundDeckBusy === "mix" ? "Mixing..." : "Mix Stems"}
              </button>
            </div>
          </div>

          {soundDeckError && (
            <div className="rounded-md bg-red-900/50 p-3 text-sm text-red-300">{soundDeckError}</div>
          )}

          {!soundDeck || soundDeck.cues.length === 0 ? (
            <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 text-sm text-gray-400">
              No sound cues yet.
            </div>
          ) : (
            <div className="space-y-2">
              {soundDeck.cues.map((cue) => (
                <div key={cue.id} className="rounded-lg border border-gray-700 bg-gray-800/50 p-3">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded bg-gray-700 px-2 py-0.5 text-xs text-gray-300">{cue.cue_type}</span>
                        <h3 className="text-sm font-medium text-white">{cue.name}</h3>
                        <span className="text-xs text-gray-500">
                          Scene {cue.scene_number ?? "-"} / Shot {cue.shot_number ?? "-"}
                        </span>
                        <span className={`text-xs ${cue.generation_status === "READY" ? "text-green-400" : cue.generation_status === "FAILED" ? "text-red-400" : "text-gray-400"}`}>
                          {cue.generation_status}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-400">{cue.prompt}</p>
                      <p className="mt-1 text-xs text-gray-500">
                        {formatDuration(cue.start_time_seconds)} · {formatDuration(cue.duration_seconds)} · {cue.volume_db ?? -18} dB
                      </p>
                      {cue.error_message && <p className="mt-1 text-xs text-red-300">{cue.error_message}</p>}
                      {cue.audio_url && (
                        <audio className="mt-2 h-8 w-full" controls src={cue.audio_url} />
                      )}
                    </div>
                    <button
                      onClick={() => handleGenerateCue(cue)}
                      disabled={soundDeckBusy !== null}
                      className="rounded-md bg-gray-700 px-2 py-1 text-xs text-gray-200 hover:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {soundDeckBusy === cue.id ? "Generating..." : cue.audio_url ? "Regenerate" : "Generate"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {soundDeck?.mix_artifacts.length ? (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-300">SFX Stems</h3>
              {soundDeck.mix_artifacts.map((artifact) => (
                <div key={artifact.id} className="rounded-lg border border-gray-700 bg-gray-800/50 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-sm text-gray-300">
                      {artifact.artifact_type} · {formatDuration(artifact.duration_seconds)} · {artifact.status}
                    </p>
                    {artifact.audio_url && <audio className="h-8 min-w-64" controls src={artifact.audio_url} />}
                  </div>
                  {artifact.error_message && <p className="mt-1 text-xs text-red-300">{artifact.error_message}</p>}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}

      {activeTab === "screenplay" && (
        <ScreenplayEditor
          productionId={productionId}
          onScenesGenerated={load}
        />
      )}

      {showPicker && (
        <ScenePickerModal
          productionId={productionId}
          linkedSceneIds={linkedSceneIds}
          onConfirm={handleBatchAdd}
          onClose={() => setShowPicker(false)}
        />
      )}
    </div>
  );
}
