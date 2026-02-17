import type {
  GenerateRequest,
  GenerateResponse,
  StatusResponse,
  ProjectDetail,
  ProjectListItem,
  ResumeResponse,
  StopResponse,
  ForkRequest,
  ForkResponse,
  MetricsResponse,
  ManifestListItem,
  ManifestDetail,
  CreateManifestRequest,
  UpdateManifestRequest,
  CreateAssetRequest,
  UpdateAssetRequest,
  AssetResponse,
  ProcessingProgress,
  CandidateScore,
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

/** POST /api/projects/{id}/fork — fork a project with optional edits */
export function forkProject(projectId: string, body: ForkRequest): Promise<ForkResponse> {
  return request<ForkResponse>(`/api/projects/${projectId}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/projects/{id}/download — returns download URL (not JSON) */
export function getDownloadUrl(projectId: string): string {
  return `/api/projects/${projectId}/download`;
}

/** GET /api/metrics — aggregate metrics across all projects */
export function getMetrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/api/metrics");
}

/** GET /api/manifests — list manifests with optional filters */
export function listManifests(params?: {
  category?: string;
  sort_by?: string;
  sort_order?: string;
}): Promise<ManifestListItem[]> {
  const searchParams = new URLSearchParams();
  if (params?.category) searchParams.set("category", params.category);
  if (params?.sort_by) searchParams.set("sort_by", params.sort_by);
  if (params?.sort_order) searchParams.set("sort_order", params.sort_order);
  const qs = searchParams.toString();
  return request<ManifestListItem[]>(`/api/manifests${qs ? `?${qs}` : ""}`);
}

/** POST /api/manifests — create new manifest */
export function createManifest(body: CreateManifestRequest): Promise<ManifestListItem> {
  return request<ManifestListItem>("/api/manifests", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/manifests/{id} — get manifest with assets */
export function getManifestDetail(manifestId: string): Promise<ManifestDetail> {
  return request<ManifestDetail>(`/api/manifests/${manifestId}`);
}

/** PUT /api/manifests/{id} — update manifest */
export function updateManifest(manifestId: string, body: UpdateManifestRequest): Promise<ManifestListItem> {
  return request<ManifestListItem>(`/api/manifests/${manifestId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** DELETE /api/manifests/{id} — soft delete manifest */
export function deleteManifest(manifestId: string): Promise<{ status: string; manifest_id: string }> {
  return request<{ status: string; manifest_id: string }>(`/api/manifests/${manifestId}`, {
    method: "DELETE",
  });
}

/** POST /api/manifests/{id}/duplicate — duplicate manifest */
export function duplicateManifest(manifestId: string, name?: string): Promise<ManifestListItem> {
  const qs = name ? `?name=${encodeURIComponent(name)}` : "";
  return request<ManifestListItem>(`/api/manifests/${manifestId}/duplicate${qs}`, {
    method: "POST",
  });
}

/** POST /api/manifests/{id}/assets — create asset in manifest */
export function createAsset(manifestId: string, body: CreateAssetRequest): Promise<AssetResponse> {
  return request<AssetResponse>(`/api/manifests/${manifestId}/assets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** PUT /api/assets/{id} — update asset metadata */
export function updateAsset(assetId: string, body: UpdateAssetRequest): Promise<AssetResponse> {
  return request<AssetResponse>(`/api/assets/${assetId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** DELETE /api/assets/{id} — delete asset */
export function deleteAsset(assetId: string): Promise<{ status: string; asset_id: string }> {
  return request<{ status: string; asset_id: string }>(`/api/assets/${assetId}`, {
    method: "DELETE",
  });
}

/** POST /api/assets/{id}/upload — upload image for asset */
export async function uploadAssetImage(assetId: string, file: File): Promise<AssetResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/assets/${assetId}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<AssetResponse>;
}

/** POST /api/manifests/{id}/upload-video — upload video for frame extraction */
export async function uploadVideoForManifest(
  manifestId: string,
  file: File,
): Promise<{ task_id: string; status: string; manifest_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/manifests/${manifestId}/upload-video`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<{ task_id: string; status: string; manifest_id: string }>;
}

/** GET /api/manifests/{id}/extraction-progress — poll extraction progress */
export function getExtractionProgress(manifestId: string): Promise<ProcessingProgress> {
  return request<ProcessingProgress>(`/api/manifests/${manifestId}/extraction-progress`);
}

/** POST /api/manifests/{id}/process — trigger background processing */
export function processManifest(manifestId: string): Promise<{ task_id: string; status: string }> {
  return request<{ task_id: string; status: string }>(`/api/manifests/${manifestId}/process`, {
    method: "POST",
  });
}

/** GET /api/manifests/{id}/progress — poll processing progress */
export function getProcessingProgress(manifestId: string): Promise<ProcessingProgress> {
  return request<ProcessingProgress>(`/api/manifests/${manifestId}/progress`);
}

/** POST /api/assets/{id}/reprocess — re-run detection + reverse-prompting for single asset */
export function reprocessAsset(assetId: string): Promise<AssetResponse> {
  return request<AssetResponse>(`/api/assets/${assetId}/reprocess`, {
    method: "POST",
  });
}

/** GET /api/projects/{id}/scenes/{idx}/candidates */
export function listCandidates(
  projectId: string,
  sceneIdx: number,
): Promise<CandidateScore[]> {
  return request<CandidateScore[]>(
    `/api/projects/${projectId}/scenes/${sceneIdx}/candidates`,
  );
}

/** PUT /api/projects/{id}/scenes/{idx}/candidates/{cid}/select */
export function selectCandidate(
  projectId: string,
  sceneIdx: number,
  candidateId: string,
): Promise<{ selected: string; selected_by: string }> {
  return request<{ selected: string; selected_by: string }>(
    `/api/projects/${projectId}/scenes/${sceneIdx}/candidates/${candidateId}/select`,
    { method: "PUT" },
  );
}

/** GET /api/manifests/{id} — fetch assets for a manifest (used in EditForkPanel) */
export async function fetchManifestAssets(manifestId: string): Promise<AssetResponse[]> {
  const res = await fetch(`/api/manifests/${manifestId}`);
  if (!res.ok) throw new ApiError(res.status, await res.text());
  const data = await res.json();
  return data.assets;
}

export { ApiError };
