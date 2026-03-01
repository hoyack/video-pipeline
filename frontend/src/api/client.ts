import type {
  GenerateRequest,
  GenerateResponse,
  StatusResponse,
  SceneDetail,
  PaginatedScenes,
  ResumeResponse,
  StopResponse,
  ForkRequest,
  ForkResponse,
  EditSceneRequest,
  EditSceneResponse,
  CheckpointListItem,
  CheckpointDiff,
  RegenerateShotRequest,
  RegenerateSceneRequest,
  RegenerateTextRequest,
  RegenerateTextResponse,
  GenerateShotFieldsRequest,
  GenerateShotFieldsResponse,
  GenerateNewShotRequest,
  GenerateNewShotResponse,
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
  UserSettingsResponse,
  UserSettingsUpdate,
  EnabledModelsResponse,
  CreateDraftSceneRequest,
  CreateDraftSceneResponse,
  StartGenerationRequest,
  StartGenerationResponse,
  SequenceResponse,
  SequenceWithScenes,
  SequenceCreate,
  SequenceUpdate,
  SequenceReorderRequest,
  AssignSequenceRequest,
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

/** POST /api/scenes — create a draft scene (no pipeline execution) */
export function createDraftScene(body: CreateDraftSceneRequest = {}): Promise<CreateDraftSceneResponse> {
  return request<CreateDraftSceneResponse>("/api/scenes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** POST /api/scenes/{id}/generate — start pipeline on an existing scene */
export function startGeneration(sceneId: string, body?: StartGenerationRequest): Promise<StartGenerationResponse> {
  return request<StartGenerationResponse>(`/api/scenes/${sceneId}/generate`, {
    method: "POST",
    ...(body
      ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : {}),
  });
}

/** POST /api/generate — start a new video generation job */
export function generateVideo(body: GenerateRequest): Promise<GenerateResponse> {
  return request<GenerateResponse>("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/scenes/{id}/status — lightweight polling endpoint */
export function getSceneStatus(sceneId: string): Promise<StatusResponse> {
  return request<StatusResponse>(`/api/scenes/${sceneId}/status`);
}

/** GET /api/scenes/{id} — full scene detail with shots */
export function getSceneDetail(sceneId: string): Promise<SceneDetail> {
  return request<SceneDetail>(`/api/scenes/${sceneId}`);
}

/** GET /api/scenes — list scenes with pagination */
export function listScenes(params?: {
  page?: number;
  per_page?: number;
  view?: string;
  status?: string;
}): Promise<PaginatedScenes> {
  const searchParams = new URLSearchParams();
  if (params?.page) searchParams.set("page", String(params.page));
  if (params?.per_page) searchParams.set("per_page", String(params.per_page));
  if (params?.view) searchParams.set("view", params.view);
  if (params?.status) searchParams.set("status", params.status);
  const qs = searchParams.toString();
  return request<PaginatedScenes>(`/api/scenes${qs ? `?${qs}` : ""}`);
}

/** PATCH /api/scenes/{id} — update scene title */
export function updateScene(sceneId: string, body: { title: string }): Promise<{ scene_id: string; title: string }> {
  return request<{ scene_id: string; title: string }>(`/api/scenes/${sceneId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** DELETE /api/scenes/{id} — soft delete scene */
export function deleteScene(sceneId: string): Promise<{ status: string; scene_id: string }> {
  return request<{ status: string; scene_id: string }>(`/api/scenes/${sceneId}`, {
    method: "DELETE",
  });
}

/** POST /api/scenes/{id}/resume — resume a failed/interrupted job */
export function resumeScene(
  sceneId: string,
  body?: {
    run_through?: string | null;
    image_model?: string;
    vision_model?: string;
    video_model?: string;
    audio_enabled?: boolean;
    clip_duration?: number;
  },
): Promise<ResumeResponse> {
  return request<ResumeResponse>(`/api/scenes/${sceneId}/resume`, {
    method: "POST",
    ...(body
      ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : {}),
  });
}

/** POST /api/scenes/{id}/stop — stop a running pipeline */
export function stopScene(sceneId: string): Promise<StopResponse> {
  return request<StopResponse>(`/api/scenes/${sceneId}/stop`, {
    method: "POST",
  });
}

/** POST /api/scenes/{id}/fork — fork a scene with optional edits */
export function forkScene(sceneId: string, body: ForkRequest): Promise<ForkResponse> {
  return request<ForkResponse>(`/api/scenes/${sceneId}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** PATCH /api/scenes/{id}/edit — edit scene in-place (PipeSVN) */
export function editScene(sceneId: string, body: EditSceneRequest): Promise<EditSceneResponse> {
  return request<EditSceneResponse>(`/api/scenes/${sceneId}/edit`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/scenes/{id}/download — returns download URL (not JSON) */
export function getDownloadUrl(sceneId: string): string {
  return `/api/scenes/${sceneId}/download`;
}

/** GET /api/metrics — aggregate metrics across all scenes */
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

/** POST /api/manifests/from-scene — create manifest from scene storyboard */
export function importSceneToManifest(
  sceneId: string,
  name?: string,
): Promise<ManifestDetail> {
  return request<ManifestDetail>("/api/manifests/from-scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: sceneId, name }),
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

/** GET /api/scenes/{id}/shots/{idx}/candidates */
export function listCandidates(
  sceneId: string,
  shotIdx: number,
): Promise<CandidateScore[]> {
  return request<CandidateScore[]>(
    `/api/scenes/${sceneId}/shots/${shotIdx}/candidates`,
  );
}

/** PUT /api/scenes/{id}/shots/{idx}/candidates/{cid}/select */
export function selectCandidate(
  sceneId: string,
  shotIdx: number,
  candidateId: string,
): Promise<{ selected: string; selected_by: string }> {
  return request<{ selected: string; selected_by: string }>(
    `/api/scenes/${sceneId}/shots/${shotIdx}/candidates/${candidateId}/select`,
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

/** GET /api/settings — get user settings */
export function getSettings(): Promise<UserSettingsResponse> {
  return request<UserSettingsResponse>("/api/settings");
}

/** PUT /api/settings — update user settings */
export function updateSettings(body: UserSettingsUpdate): Promise<UserSettingsResponse> {
  return request<UserSettingsResponse>("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/settings/models — lightweight enabled models for GenerateForm */
export function getEnabledModels(): Promise<EnabledModelsResponse> {
  return request<EnabledModelsResponse>("/api/settings/models");
}

// ============================================================================
// PipeSVN: Checkpoint API
// ============================================================================

/** GET /api/scenes/{id}/checkpoints — list checkpoints */
export function listCheckpoints(sceneId: string): Promise<CheckpointListItem[]> {
  return request<CheckpointListItem[]>(`/api/scenes/${sceneId}/checkpoints`);
}

/** GET /api/scenes/{id}/checkpoints/{sha}/diff — get checkpoint diff */
export function getCheckpointDiff(sceneId: string, sha: string): Promise<CheckpointDiff> {
  return request<CheckpointDiff>(`/api/scenes/${sceneId}/checkpoints/${sha}/diff`);
}

/** POST /api/scenes/{id}/checkpoints — create manual checkpoint */
export function createCheckpoint(sceneId: string): Promise<{ sha: string; message: string }> {
  return request<{ sha: string; message: string }>(`/api/scenes/${sceneId}/checkpoints`, {
    method: "POST",
  });
}

/** DELETE /api/scenes/{id}/checkpoints/{sha} — delete checkpoint */
export function deleteCheckpoint(sceneId: string, sha: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/scenes/${sceneId}/checkpoints/${sha}`, {
    method: "DELETE",
  });
}

/** POST /api/scenes/{id}/revert — revert to checkpoint */
export function revertToCheckpoint(
  sceneId: string,
  sha: string,
): Promise<{ status: string; head_sha: string; reverted_to: string }> {
  return request<{ status: string; head_sha: string; reverted_to: string }>(
    `/api/scenes/${sceneId}/revert`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sha }),
    },
  );
}

// ============================================================================
// PipeSVN: Regeneration API
// ============================================================================

/** POST /api/scenes/{id}/shots/{idx}/regenerate — regenerate shot assets */
export function regenerateShot(
  sceneId: string,
  shotIdx: number,
  body: RegenerateShotRequest,
): Promise<{ status: string; head_sha?: string | null }> {
  return request<{ status: string; head_sha?: string | null }>(
    `/api/scenes/${sceneId}/shots/${shotIdx}/regenerate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/** POST /api/scenes/{id}/shots/{idx}/regenerate-text — regenerate a text field via LLM */
export function regenerateShotText(
  sceneId: string,
  shotIdx: number,
  body: RegenerateTextRequest,
): Promise<RegenerateTextResponse> {
  return request<RegenerateTextResponse>(
    `/api/scenes/${sceneId}/shots/${shotIdx}/regenerate-text`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
  );
}

/** POST /api/scenes/{id}/generate-shot-fields — generate all 5 text fields for a new shot */
export function generateShotFields(
  sceneId: string,
  body: GenerateShotFieldsRequest,
): Promise<GenerateShotFieldsResponse> {
  return request<GenerateShotFieldsResponse>(
    `/api/scenes/${sceneId}/generate-shot-fields`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
  );
}

/** POST /api/scenes/{id}/generate-new-shot — generate complete shot (text sync + assets background) */
export function generateNewShot(
  sceneId: string,
  body: GenerateNewShotRequest,
): Promise<GenerateNewShotResponse> {
  return request<GenerateNewShotResponse>(
    `/api/scenes/${sceneId}/generate-new-shot`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
  );
}

/** POST /api/scenes/{id}/regenerate — scene-wide regeneration */
export function regenerateScene(
  sceneId: string,
  body: RegenerateSceneRequest,
): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/scenes/${sceneId}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** PUT /api/scenes/{id}/shots/{idx}/keyframes/{pos} — upload keyframe */
export async function uploadKeyframe(
  sceneId: string,
  shotIdx: number,
  position: string,
  file: File,
): Promise<{ status: string; file_path: string; keyframe_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(
    `/api/scenes/${sceneId}/shots/${shotIdx}/keyframes/${position}`,
    { method: "PUT", body: formData },
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

/** PUT /api/scenes/{id}/shots/{idx}/clip — upload clip */
export async function uploadClip(
  sceneId: string,
  shotIdx: number,
  file: File,
): Promise<{ status: string; file_path: string; clip_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(
    `/api/scenes/${sceneId}/shots/${shotIdx}/clip`,
    { method: "PUT", body: formData },
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

/** DELETE /api/scenes/{id}/shots/{idx}/clip — delete clip */
export function deleteShotClip(
  sceneId: string,
  shotIdx: number,
): Promise<{ status: string }> {
  return request<{ status: string }>(
    `/api/scenes/${sceneId}/shots/${shotIdx}/clip`,
    { method: "DELETE" },
  );
}

/** DELETE /api/scenes/{id}/shots/{idx}/keyframes/{pos} — delete keyframe */
export function deleteShotKeyframe(
  sceneId: string,
  shotIdx: number,
  position: string,
): Promise<{ status: string }> {
  return request<{ status: string }>(
    `/api/scenes/${sceneId}/shots/${shotIdx}/keyframes/${position}`,
    { method: "DELETE" },
  );
}

// ============================================================================
// Productions API
// ============================================================================

export interface ProductionResponse {
  id: string;
  name: string;
  description: string | null;
  tags: string[] | null;
  scene_count: number;
  created_at: string;
  updated_at: string;
}

export interface ProductionCreate {
  name: string;
  description?: string;
  tags?: string[];
}

export interface ProductionUpdate {
  name?: string;
  description?: string;
  tags?: string[];
}

/** GET /api/productions — list all productions */
export function listProductions(): Promise<ProductionResponse[]> {
  return request<ProductionResponse[]>("/api/productions");
}

/** POST /api/productions — create a production */
export function createProduction(body: ProductionCreate): Promise<ProductionResponse> {
  return request<ProductionResponse>("/api/productions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/productions/{id} — get production detail */
export function getProduction(productionId: string): Promise<ProductionResponse> {
  return request<ProductionResponse>(`/api/productions/${productionId}`);
}

/** PUT /api/productions/{id} — update production */
export function updateProduction(productionId: string, body: ProductionUpdate): Promise<ProductionResponse> {
  return request<ProductionResponse>(`/api/productions/${productionId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** DELETE /api/productions/{id} — delete production */
export function deleteProduction(productionId: string): Promise<{ status: string; production_id: string }> {
  return request<{ status: string; production_id: string }>(`/api/productions/${productionId}`, {
    method: "DELETE",
  });
}

/** POST /api/productions/{id}/scenes/{sceneId} — add scene to production */
export function addSceneToProduction(productionId: string, sceneId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/productions/${productionId}/scenes/${sceneId}`, {
    method: "POST",
  });
}

/** POST /api/productions/{id}/scenes — batch add scenes to production */
export function batchAddScenesToProduction(
  productionId: string,
  sceneIds: string[],
): Promise<{ status: string; production_id: string; scene_ids: string[]; count: number }> {
  return request<{ status: string; production_id: string; scene_ids: string[]; count: number }>(
    `/api/productions/${productionId}/scenes`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_ids: sceneIds }),
    },
  );
}

/** DELETE /api/productions/{id}/scenes/{sceneId} — remove scene from production */
export function removeSceneFromProduction(productionId: string, sceneId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/productions/${productionId}/scenes/${sceneId}`, {
    method: "DELETE",
  });
}

// ============================================================================
// Sequences API (Issue #24 — Sequence Grouping Layer)
// ============================================================================

/** GET /api/productions/{id}/sequences — list sequences for a production */
export function listSequences(productionId: string): Promise<SequenceResponse[]> {
  return request<SequenceResponse[]>(`/api/productions/${productionId}/sequences`);
}

/** POST /api/productions/{id}/sequences — create a new sequence */
export function createSequence(productionId: string, body: SequenceCreate): Promise<SequenceResponse> {
  return request<SequenceResponse>(`/api/productions/${productionId}/sequences`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GET /api/sequences/{id} — get sequence with scenes */
export function getSequence(sequenceId: string): Promise<SequenceWithScenes> {
  return request<SequenceWithScenes>(`/api/sequences/${sequenceId}`);
}

/** PUT /api/sequences/{id} — update sequence */
export function updateSequence(sequenceId: string, body: SequenceUpdate): Promise<SequenceResponse> {
  return request<SequenceResponse>(`/api/sequences/${sequenceId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** DELETE /api/sequences/{id} — delete sequence (unsequences children) */
export function deleteSequence(sequenceId: string): Promise<{ status: string; unsequenced_scenes: number }> {
  return request<{ status: string; unsequenced_scenes: number }>(`/api/sequences/${sequenceId}`, {
    method: "DELETE",
  });
}

/** PUT /api/productions/{id}/sequences/reorder — bulk reorder sequences */
export function reorderSequences(productionId: string, body: SequenceReorderRequest): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/productions/${productionId}/sequences/reorder`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** PUT /api/scenes/{id}/sequence — assign/unassign scene to sequence */
export function assignSceneToSequence(
  sceneId: string,
  body: AssignSequenceRequest,
): Promise<{ status: string; sequence_id: string | null; scene_order: number }> {
  return request<{ status: string; sequence_id: string | null; scene_order: number }>(
    `/api/scenes/${sceneId}/sequence`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export { ApiError };
