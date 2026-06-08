import { useCallback, useEffect, useState } from "react";
import type { ScreenplayResponse, VoiceLineResponse, VoiceScriptResponse } from "../api/types.ts";
import {
  deleteVoiceLine,
  generateVoiceLineAudio,
  generateVoiceScript,
  generateVoiceScriptAudio,
  getVoiceScript,
  lipSyncVoiceScript,
  mixVoiceScript,
  resolveVoiceBindings,
  updateVoiceLine,
} from "../api/client.ts";

interface VoiceScriptTabProps {
  productionId: string;
  screenplay: ScreenplayResponse;
}

export function VoiceScriptTab({ productionId, screenplay }: VoiceScriptTabProps) {
  const [voiceScript, setVoiceScript] = useState<VoiceScriptResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [editingLineId, setEditingLineId] = useState<string | null>(null);
  const [draftText, setDraftText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const loadVoiceScript = useCallback(async () => {
    setLoading(true);
    try {
      setVoiceScript(await getVoiceScript(productionId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load voice script");
    } finally {
      setLoading(false);
    }
  }, [productionId]);

  useEffect(() => {
    loadVoiceScript();
  }, [loadVoiceScript]);

  async function runAction(action: string, task: () => Promise<VoiceScriptResponse>) {
    setBusy(action);
    setError(null);
    try {
      setVoiceScript(await task());
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action}`);
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerate() {
    await runAction("generate", async () => {
      const result = await generateVoiceScript(productionId, screenplay.text_model ?? undefined);
      return result.voice_script;
    });
  }

  async function handleResolve() {
    if (!voiceScript) return;
    await runAction("resolve", async () => (await resolveVoiceBindings(voiceScript.id)).voice_script);
  }

  async function handleGenerateAudio() {
    if (!voiceScript) return;
    await runAction("audio", async () => (await generateVoiceScriptAudio(voiceScript.id)).voice_script);
  }

  async function handleMix() {
    if (!voiceScript) return;
    await runAction("mix", async () => (await mixVoiceScript(voiceScript.id)).voice_script);
  }

  async function handleLipSync() {
    if (!voiceScript) return;
    await runAction("lip-sync", async () => (await lipSyncVoiceScript(voiceScript.id)).voice_script);
  }

  async function handleLineAudio(lineId: string) {
    if (!voiceScript) return;
    setBusy(`line-${lineId}`);
    try {
      const updatedLine = await generateVoiceLineAudio(lineId);
      setVoiceScript({
        ...voiceScript,
        lines: voiceScript.lines.map((line) => (line.id === lineId ? updatedLine : line)),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate line audio");
    } finally {
      setBusy(null);
    }
  }

  async function saveLineText(line: VoiceLineResponse) {
    if (draftText === line.text) {
      setEditingLineId(null);
      return;
    }
    const updatedLine = await updateVoiceLine(line.id, { text: draftText });
    setVoiceScript((current) => current
      ? { ...current, lines: current.lines.map((item) => (item.id === line.id ? updatedLine : item)) }
      : current);
    setEditingLineId(null);
  }

  async function removeLine(lineId: string) {
    await deleteVoiceLine(lineId);
    setVoiceScript((current) => current
      ? { ...current, lines: current.lines.filter((line) => line.id !== lineId) }
      : current);
  }

  if (loading) {
    return <div className="py-8 text-center text-sm text-gray-400">Loading voice script...</div>;
  }

  const lines = voiceScript?.lines ?? [];

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded-md bg-red-900/50 p-3 text-sm text-red-300 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200 ml-3">Dismiss</button>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <StatusPill label="Script" value={voiceScript?.status ?? "DRAFT"} />
          <StatusPill label="TTS" value={voiceScript?.voice_generation_status ?? "PENDING"} />
          <StatusPill label="Mix" value={voiceScript?.mix_status ?? "PENDING"} />
          <StatusPill label="Lip" value={voiceScript?.lip_sync_status ?? "PENDING"} />
        </div>
        <div className="flex flex-wrap gap-2">
          <ActionButton label="Generate Script" active={busy === "generate"} disabled={busy !== null} onClick={handleGenerate} />
          <ActionButton label="Resolve Voices" active={busy === "resolve"} disabled={busy !== null || !voiceScript} onClick={handleResolve} />
          <ActionButton label="Generate Audio" active={busy === "audio"} disabled={busy !== null || lines.length === 0} onClick={handleGenerateAudio} />
          <ActionButton label="Mix" active={busy === "mix"} disabled={busy !== null || lines.every((line) => !line.audio_path)} onClick={handleMix} />
          <ActionButton label="Lip Sync" active={busy === "lip-sync"} disabled={busy !== null || lines.every((line) => !line.audio_path)} onClick={handleLipSync} />
        </div>
      </div>

      {lines.length === 0 ? (
        <div className="rounded-md border border-gray-700 bg-gray-900 p-6 text-center text-sm text-gray-400">
          No voice lines yet.
        </div>
      ) : (
        <div className="space-y-3">
          {lines.map((line) => (
            <div key={line.id} className="rounded-md border border-gray-700 bg-gray-900 p-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                    <span className="rounded bg-gray-800 px-2 py-0.5 font-medium text-gray-300">{line.line_type}</span>
                    {line.speaker_tag && <span className="text-blue-300">{line.speaker_tag}</span>}
                    {line.speaker_name && <span className="text-gray-400">{line.speaker_name}</span>}
                    <span className="text-gray-600">Scene {line.scene_number ?? "-"} / Shot {line.shot_number ?? "-"}</span>
                    <StatusPill label="Audio" value={line.generation_status} />
                    {line.lip_sync_status && <StatusPill label="Lip" value={line.lip_sync_status} />}
                  </div>

                  {editingLineId === line.id ? (
                    <textarea
                      value={draftText}
                      onChange={(event) => setDraftText(event.target.value)}
                      onBlur={() => saveLineText(line)}
                      rows={3}
                      autoFocus
                      className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-200 outline-none focus:border-blue-500"
                    />
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        setEditingLineId(line.id);
                        setDraftText(line.text);
                      }}
                      className="block w-full text-left text-sm leading-6 text-gray-200 hover:text-white"
                    >
                      {line.text}
                    </button>
                  )}

                  {line.delivery_notes && <p className="mt-2 text-xs text-gray-500">{line.delivery_notes}</p>}
                  {line.warnings.length > 0 && (
                    <div className="mt-2 text-xs text-yellow-300">{line.warnings.join(" · ")}</div>
                  )}
                  {line.audio_url && (
                    <audio controls src={line.audio_url} className="mt-3 h-8 w-full max-w-md" />
                  )}
                </div>

                <div className="flex shrink-0 gap-2">
                  <button
                    onClick={() => handleLineAudio(line.id)}
                    disabled={busy !== null || !line.voice_id}
                    className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-gray-700 disabled:text-gray-500"
                  >
                    {busy === `line-${line.id}` ? "Generating..." : "Audio"}
                  </button>
                  <button
                    onClick={() => removeLine(line.id)}
                    disabled={busy !== null}
                    className="rounded-md bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-700 disabled:opacity-50"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {(voiceScript?.mix_artifacts.length ?? 0) > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-200">Voice Stems</h3>
          {voiceScript?.mix_artifacts.map((artifact) => (
            <div key={artifact.id} className="rounded-md border border-gray-700 bg-gray-900 p-3">
              <div className="mb-2 flex items-center gap-2 text-xs text-gray-400">
                <StatusPill label={artifact.artifact_type} value={artifact.status} />
                <span>{artifact.duration_seconds ? `${artifact.duration_seconds.toFixed(1)}s` : "duration unknown"}</span>
              </div>
              {artifact.audio_url && <audio controls src={artifact.audio_url} className="h-8 w-full max-w-md" />}
              {artifact.error_message && <p className="mt-2 text-xs text-red-300">{artifact.error_message}</p>}
            </div>
          ))}
        </div>
      )}

      {(voiceScript?.lip_sync_jobs.length ?? 0) > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-200">Lip-sync Jobs</h3>
          <div className="overflow-x-auto rounded-md border border-gray-700">
            <table className="min-w-full divide-y divide-gray-800 text-left text-xs">
              <thead className="bg-gray-900 text-gray-500">
                <tr>
                  <th className="px-3 py-2 font-medium">Shot</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Adapter</th>
                  <th className="px-3 py-2 font-medium">Output</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800 bg-gray-950 text-gray-300">
                {voiceScript?.lip_sync_jobs.map((job) => (
                  <tr key={job.id}>
                    <td className="px-3 py-2">{job.shot_id.slice(0, 8)}</td>
                    <td className="px-3 py-2">{job.status}</td>
                    <td className="px-3 py-2">{job.adapter_type}</td>
                    <td className="px-3 py-2">{job.output_clip_path ?? job.error_message ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function ActionButton({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-gray-700 disabled:text-gray-500"
    >
      {active ? "Working..." : label}
    </button>
  );
}

function StatusPill({ label, value }: { label: string; value: string }) {
  const color = value === "READY" || value === "LOCKED"
    ? "bg-green-900/60 text-green-300"
    : value === "FAILED"
      ? "bg-red-900/60 text-red-300"
      : value === "SKIPPED" || value === "PARTIAL"
        ? "bg-yellow-900/60 text-yellow-300"
        : "bg-gray-800 text-gray-400";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs ${color}`}>
      <span className="text-gray-500">{label}</span>
      <span>{value}</span>
    </span>
  );
}
