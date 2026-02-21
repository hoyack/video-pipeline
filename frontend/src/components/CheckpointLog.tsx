import { useEffect, useState } from "react";
import clsx from "clsx";
import { listCheckpoints, getCheckpointDiff, revertToCheckpoint, deleteCheckpoint } from "../api/client.ts";
import type { CheckpointListItem, CheckpointDiff } from "../api/types.ts";

interface CheckpointLogProps {
  projectId: string;
  headSha: string | null | undefined;
  onReverted: () => void;
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function CheckpointLog({ projectId, headSha, onReverted }: CheckpointLogProps) {
  const [checkpoints, setCheckpoints] = useState<CheckpointListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedSha, setExpandedSha] = useState<string | null>(null);
  const [diff, setDiff] = useState<CheckpointDiff | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const [reverting, setReverting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listCheckpoints(projectId)
      .then(setCheckpoints)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [projectId, headSha]);

  async function handleExpand(sha: string) {
    if (expandedSha === sha) {
      setExpandedSha(null);
      setDiff(null);
      return;
    }
    setExpandedSha(sha);
    setDiffLoading(true);
    try {
      const d = await getCheckpointDiff(projectId, sha);
      setDiff(d);
    } catch {
      setDiff(null);
    } finally {
      setDiffLoading(false);
    }
  }

  async function handleRevert(sha: string) {
    setReverting(sha);
    setError(null);
    try {
      await revertToCheckpoint(projectId, sha);
      onReverted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Revert failed");
    } finally {
      setReverting(null);
    }
  }

  async function handleDelete(sha: string) {
    try {
      await deleteCheckpoint(projectId, sha);
      setCheckpoints((prev) => prev.filter((cp) => cp.sha !== sha));
      if (expandedSha === sha) {
        setExpandedSha(null);
        setDiff(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  if (loading) {
    return <p className="text-sm text-gray-500">Loading history...</p>;
  }

  if (checkpoints.length === 0) {
    return <p className="text-sm text-gray-500">No checkpoints yet.</p>;
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-gray-400">
        Version History ({checkpoints.length})
      </h3>

      {error && (
        <div className="rounded border border-red-800 bg-red-900/50 px-2 py-1 text-xs text-red-300">
          {error}
        </div>
      )}

      <div className="max-h-96 space-y-1 overflow-y-auto">
        {checkpoints.map((cp, i) => {
          const isHead = cp.sha === headSha;
          const isExpanded = expandedSha === cp.sha;

          return (
            <div key={cp.sha} className="rounded border border-gray-800 bg-gray-900">
              {/* Summary row */}
              <button
                onClick={() => handleExpand(cp.sha)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-gray-800/50 transition-colors"
              >
                <span className="font-mono text-xs text-indigo-400">{cp.sha.slice(0, 8)}</span>
                <span className="flex-1 truncate text-xs text-gray-300">{cp.message}</span>
                {cp.changes_count > 0 && (
                  <span className="shrink-0 rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-400">
                    {cp.changes_count}
                  </span>
                )}
                <span className="shrink-0 text-[10px] text-gray-500">{timeAgo(cp.created_at)}</span>
                {isHead && (
                  <span className="shrink-0 rounded bg-indigo-900/50 px-1.5 py-0.5 text-[10px] font-medium text-indigo-300">
                    HEAD
                  </span>
                )}
              </button>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="border-t border-gray-800 px-3 py-2 space-y-2">
                  {diffLoading ? (
                    <p className="text-xs text-gray-500">Loading diff...</p>
                  ) : diff && diff.changes.length > 0 ? (
                    <div className="space-y-1">
                      {diff.changes.map((change, ci) => (
                        <div key={ci} className="text-[11px] text-gray-400">
                          {change.type === "project_field" && (
                            <span>
                              Changed <span className="text-gray-300">{change.field}</span>
                              {change.old && <span className="text-red-400"> {change.old}</span>}
                              {change.new && <span className="text-green-400"> {change.new}</span>}
                            </span>
                          )}
                          {change.type === "scene_field" && (
                            <span>
                              Scene {(change.scene_index ?? 0) + 1}: changed{" "}
                              <span className="text-gray-300">{change.field}</span>
                            </span>
                          )}
                          {change.type === "scene_added" && (
                            <span className="text-green-400">
                              Added scene {(change.scene_index ?? 0) + 1}
                            </span>
                          )}
                          {change.type === "scene_removed" && (
                            <span className="text-red-400">
                              Removed scene {(change.scene_index ?? 0) + 1}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">No structured diff available</p>
                  )}

                  {/* Actions */}
                  <div className="flex gap-2 pt-1">
                    {!isHead && (
                      <button
                        onClick={() => handleRevert(cp.sha)}
                        disabled={reverting === cp.sha}
                        className={clsx(
                          "rounded px-2 py-1 text-[11px] font-medium transition-colors",
                          reverting === cp.sha
                            ? "bg-gray-800 text-gray-500"
                            : "bg-amber-900/50 text-amber-300 hover:bg-amber-800/50",
                        )}
                      >
                        {reverting === cp.sha ? "Reverting..." : "Revert to this"}
                      </button>
                    )}
                    {!isHead && i < checkpoints.length - 1 && (
                      <button
                        onClick={() => handleDelete(cp.sha)}
                        className="rounded px-2 py-1 text-[11px] text-red-400 hover:bg-red-900/30 transition-colors"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
