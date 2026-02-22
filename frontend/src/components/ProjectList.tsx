import { useCallback, useEffect, useState } from "react";
import { listProjects, deleteProject } from "../api/client.ts";
import type { ProjectListItem } from "../api/types.ts";
import { estimateCost, TERMINAL_STATUSES } from "../lib/constants.ts";
import { StatusBadge } from "./StatusBadge.tsx";
import { ProjectCard } from "./ProjectCard.tsx";

type ViewMode = "list" | "cards";

const STORAGE_KEY = "vidpipe_projects_view";

function getInitialViewMode(): ViewMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "list" || stored === "cards") return stored;
  } catch { /* ignore */ }
  return "list";
}

function formatCost(item: ProjectListItem): string | null {
  if (!item.total_duration || !item.clip_duration || !item.text_model || !item.image_model || !item.video_model) {
    return null;
  }
  const cost = estimateCost(
    item.total_duration,
    item.clip_duration,
    item.text_model,
    item.image_model,
    item.video_model,
    item.audio_enabled ?? false,
  );
  return `$${cost.toFixed(2)}`;
}

interface ProjectListProps {
  onSelectProject: (projectId: string) => void;
  onNewProject: () => void;
}

export function ProjectList({ onSelectProject, onNewProject }: ProjectListProps) {
  const [viewMode, setViewMode] = useState<ViewMode>(getInitialViewMode);
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(10);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const totalPages = Math.max(1, Math.ceil(total / perPage));

  const fetchProjects = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listProjects({
        page,
        per_page: perPage,
        view: viewMode === "cards" ? "cards" : undefined,
        status: statusFilter || undefined,
      });
      setProjects(data.items);
      setTotal(data.total);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load projects");
    } finally {
      setLoading(false);
    }
  }, [page, perPage, viewMode, statusFilter]);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  // Persist view mode
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, viewMode); } catch { /* ignore */ }
  }, [viewMode]);

  // Reset page when perPage or filter changes
  const handlePerPageChange = (newPerPage: number) => {
    setPerPage(newPerPage);
    setPage(1);
  };

  const handleStatusFilterChange = (newStatus: string) => {
    setStatusFilter(newStatus);
    setPage(1);
  };

  // Delete flow
  const handleDelete = async (projectId: string) => {
    setDeleteError(null);
    try {
      await deleteProject(projectId);
      // Optimistically remove from list
      setProjects((prev) => prev.filter((p) => p.project_id !== projectId));
      setTotal((prev) => prev - 1);
      setDeleteConfirm(null);
      // If we deleted the last item on this page, go back
      if (projects.length === 1 && page > 1) {
        setPage((p) => p - 1);
      }
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  // View toggle icons
  const ListIcon = () => (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  );
  const GridIcon = () => (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm10 0a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zm10 0a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
    </svg>
  );

  if (loading && projects.length === 0) {
    return <p className="text-center text-sm text-gray-500">Loading projects...</p>;
  }

  if (error && projects.length === 0) {
    return (
      <div className="text-center">
        <p className="text-sm text-red-400">{error}</p>
        <button
          onClick={fetchProjects}
          className="mt-2 text-sm text-blue-400 hover:text-blue-300"
        >
          Retry
        </button>
      </div>
    );
  }

  if (total === 0 && !loading && !statusFilter) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No projects yet.</p>
        <p className="mt-1 text-sm text-gray-600">
          Create one from the Generate tab.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Header bar */}
      <div className="mb-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">Projects</h1>
            <button
              onClick={onNewProject}
              className="rounded-md bg-blue-600 px-3 py-1 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
            >
              + New
            </button>
          </div>
          <div className="flex items-center gap-3">
            {/* Per-page dropdown */}
            <select
              value={perPage}
              onChange={(e) => handlePerPageChange(Number(e.target.value))}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-300"
            >
              <option value={10}>10 / page</option>
              <option value={50}>50 / page</option>
              <option value={100}>100 / page</option>
            </select>

          {/* View toggle */}
          <div className="flex rounded border border-gray-700 overflow-hidden">
            <button
              onClick={() => setViewMode("list")}
              className={`px-2 py-1 transition-colors ${viewMode === "list" ? "bg-gray-700 text-white" : "bg-gray-800 text-gray-500 hover:text-gray-300"}`}
              title="List view"
            >
              <ListIcon />
            </button>
            <button
              onClick={() => setViewMode("cards")}
              className={`px-2 py-1 transition-colors ${viewMode === "cards" ? "bg-gray-700 text-white" : "bg-gray-800 text-gray-500 hover:text-gray-300"}`}
              title="Cards view"
            >
              <GridIcon />
            </button>
          </div>
        </div>
        </div>

        {/* Status filter chips */}
        <div className="flex flex-wrap gap-1.5">
          {[
            { value: "", label: "All" },
            { value: "complete", label: "Complete" },
            { value: "failed", label: "Failed" },
            { value: "stopped", label: "Stopped" },
            { value: "video_gen", label: "Video Gen" },
            { value: "keyframing", label: "Keyframing" },
            { value: "storyboarding", label: "Storyboarding" },
            { value: "stitching", label: "Stitching" },
            { value: "pending", label: "Pending" },
          ].map((opt) => (
            <button
              key={opt.value}
              onClick={() => handleStatusFilterChange(opt.value)}
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors ${
                statusFilter === opt.value
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* No results with filter active */}
      {total === 0 && !loading && statusFilter && (
        <div className="text-center py-12">
          <p className="text-gray-500">No projects with status "{statusFilter}".</p>
          <button
            onClick={() => handleStatusFilterChange("")}
            className="mt-2 text-sm text-blue-400 hover:text-blue-300"
          >
            Clear filter
          </button>
        </div>
      )}

      {/* Loading overlay for page transitions */}
      {loading && (
        <p className="mb-2 text-center text-xs text-gray-500">Loading...</p>
      )}

      {/* Cards view */}
      {viewMode === "cards" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {projects.map((p) => (
            <ProjectCard
              key={p.project_id}
              project={p}
              onClick={() => onSelectProject(p.project_id)}
              onDelete={() => setDeleteConfirm(p.project_id)}
            />
          ))}
        </div>
      )}

      {/* List view */}
      {viewMode === "list" && (
        <div className="overflow-hidden rounded-lg border border-gray-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-900/50">
                <th className="px-4 py-2.5 text-left font-medium text-gray-400">
                  Prompt
                </th>
                <th className="px-4 py-2.5 text-left font-medium text-gray-400">
                  Status
                </th>
                <th className="px-4 py-2.5 text-right font-medium text-gray-400">
                  Est. Cost
                </th>
                <th className="px-4 py-2.5 text-left font-medium text-gray-400">
                  Created
                </th>
                <th className="w-10 px-2 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => {
                const cost = formatCost(p);
                const canDelete = TERMINAL_STATUSES.has(p.status);
                return (
                  <tr
                    key={p.project_id}
                    onClick={() => onSelectProject(p.project_id)}
                    className="group cursor-pointer border-b border-gray-800/50 hover:bg-gray-900/80 transition-colors"
                  >
                    <td className="px-4 py-2.5 text-gray-200 max-w-md truncate">
                      {p.prompt}
                    </td>
                    <td className="px-4 py-2.5">
                      <StatusBadge status={p.status} />
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-400">
                      {cost ?? "\u2014"}
                    </td>
                    <td className="px-4 py-2.5 text-gray-500">
                      {new Date(p.created_at).toLocaleString()}
                    </td>
                    <td className="px-2 py-2.5">
                      {canDelete && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteConfirm(p.project_id);
                          }}
                          className="text-gray-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                          title="Delete project"
                        >
                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                          </svg>
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination bar */}
      {totalPages > 1 && (
        <div className="mt-4 flex items-center justify-center gap-2 text-sm">
          <button
            onClick={() => setPage(1)}
            disabled={page <= 1}
            className="rounded px-2 py-1 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            &laquo;
          </button>
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="rounded px-2 py-1 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            &lsaquo;
          </button>
          <span className="px-3 text-gray-300">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="rounded px-2 py-1 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            &rsaquo;
          </button>
          <button
            onClick={() => setPage(totalPages)}
            disabled={page >= totalPages}
            className="rounded px-2 py-1 text-gray-400 hover:text-white hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            &raquo;
          </button>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 w-full max-w-md rounded-lg border border-gray-700 bg-gray-900 p-6 shadow-xl">
            <h3 className="text-lg font-semibold text-white">Delete project?</h3>
            <p className="mt-2 text-sm text-gray-400">
              This will permanently remove all generated keyframes, video clips, and output
              files. The project record will be kept for cost tracking.
            </p>
            {deleteError && (
              <p className="mt-2 text-sm text-red-400">{deleteError}</p>
            )}
            <div className="mt-4 flex justify-end gap-3">
              <button
                onClick={() => { setDeleteConfirm(null); setDeleteError(null); }}
                className="rounded px-3 py-1.5 text-sm text-gray-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(deleteConfirm)}
                className="rounded bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-500"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
