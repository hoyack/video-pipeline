import type {
  GenerateRequest,
  GenerateResponse,
  StatusResponse,
  ProjectDetail,
  ProjectListItem,
  ResumeResponse,
  StopResponse,
} from "./types.ts";

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

/** POST /api/generate — start a new video generation job */
export function generateVideo(body: GenerateRequest): Promise<GenerateResponse> {
  return request<GenerateResponse>("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/projects/{id}/status — lightweight polling endpoint */
export function getProjectStatus(projectId: string): Promise<StatusResponse> {
  return request<StatusResponse>(`/api/projects/${projectId}/status`);
}

/** GET /api/projects/{id} — full project detail with scenes */
export function getProjectDetail(projectId: string): Promise<ProjectDetail> {
  return request<ProjectDetail>(`/api/projects/${projectId}`);
}

/** GET /api/projects — list all projects */
export function listProjects(): Promise<ProjectListItem[]> {
  return request<ProjectListItem[]>("/api/projects");
}

/** POST /api/projects/{id}/resume — resume a failed/interrupted job */
export function resumeProject(projectId: string): Promise<ResumeResponse> {
  return request<ResumeResponse>(`/api/projects/${projectId}/resume`, {
    method: "POST",
  });
}

/** POST /api/projects/{id}/stop — stop a running pipeline */
export function stopProject(projectId: string): Promise<StopResponse> {
  return request<StopResponse>(`/api/projects/${projectId}/stop`, {
    method: "POST",
  });
}

/** GET /api/projects/{id}/download — returns download URL (not JSON) */
export function getDownloadUrl(projectId: string): string {
  return `/api/projects/${projectId}/download`;
}

export { ApiError };
