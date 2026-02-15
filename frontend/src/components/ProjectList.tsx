import { useCallback, useEffect, useState } from "react";
import { listProjects } from "../api/client.ts";
import type { ProjectListItem } from "../api/types.ts";
import { StatusBadge } from "./StatusBadge.tsx";

interface ProjectListProps {
  onSelectProject: (projectId: string) => void;
}

export function ProjectList({ onSelectProject }: ProjectListProps) {
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProjects = useCallback(async () => {
    try {
      const data = await listProjects();
      setProjects(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load projects");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  if (loading) {
    return <p className="text-center text-sm text-gray-500">Loading projects...</p>;
  }

  if (error) {
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

  if (projects.length === 0) {
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
      <h1 className="mb-4 text-2xl font-bold text-white">Projects</h1>
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
              <th className="px-4 py-2.5 text-left font-medium text-gray-400">
                Created
              </th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => (
              <tr
                key={p.project_id}
                onClick={() => onSelectProject(p.project_id)}
                className="cursor-pointer border-b border-gray-800/50 hover:bg-gray-900/80 transition-colors"
              >
                <td className="px-4 py-2.5 text-gray-200 max-w-md truncate">
                  {p.prompt}
                </td>
                <td className="px-4 py-2.5">
                  <StatusBadge status={p.status} />
                </td>
                <td className="px-4 py-2.5 text-gray-500">
                  {new Date(p.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
