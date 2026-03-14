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
  ProductionBibleListItem,
  ProductionBibleDetail,
  CreateProductionBibleRequest,
  UpdateProductionBibleRequest,
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
  SceneReorderInSequenceRequest,
  AssignSequenceRequest,
  CharacterResponse,
  WardrobeResponse,
  VoiceProfileResponse,
  SetResponse,
  SonicIdentityResponse,
  PropResponse,
  ScoreThemeResponse,
  SFXItemResponse,
  ScreenplayResponse,
  ScreenplayUpdate,
  GeneratedSceneResult,
  VoiceInfo,
  ActorListItem,
  ActorDetail,
  ActorRefResponse,
  ActorVoiceProfileResponse,
  ActorWardrobePresetResponse,
  LibrarySetListItem,
  LibrarySetDetail,
  LibrarySetRefResponse,
  LibraryPropListItem,
  LibraryPropDetail,
  LibraryPropRefResponse,
  SoundAssetListItem,
  SoundAssetDetail,
  CastBindingResponse,
  SetBindingResponse,
  PropBindingResponse,
  SoundBindingResponse,
  BoundAssetSummary,
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
  if (res.status === 204) return undefined as T;
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

/** GET /api/production-bibles — list production bibles with optional filters */
export function listProductionBibles(params?: {
  category?: string;
  sort_by?: string;
  sort_order?: string;
}): Promise<ProductionBibleListItem[]> {
  const searchParams = new URLSearchParams();
  if (params?.category) searchParams.set("category", params.category);
  if (params?.sort_by) searchParams.set("sort_by", params.sort_by);
  if (params?.sort_order) searchParams.set("sort_order", params.sort_order);
  const qs = searchParams.toString();
  return request<ProductionBibleListItem[]>(`/api/production-bibles${qs ? `?${qs}` : ""}`);
}

/** POST /api/production-bibles — create new production bible */
export function createProductionBible(body: CreateProductionBibleRequest): Promise<ProductionBibleListItem> {
  return request<ProductionBibleListItem>("/api/production-bibles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** POST /api/production-bibles/from-scene — create production bible from scene storyboard */
export function importSceneToProductionBible(
  sceneId: string,
  name?: string,
): Promise<ProductionBibleDetail> {
  return request<ProductionBibleDetail>("/api/production-bibles/from-scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: sceneId, name }),
  });
}

/** GET /api/production-bibles/{id} — get production bible with assets */
export function getProductionBibleDetail(productionBibleId: string): Promise<ProductionBibleDetail> {
  return request<ProductionBibleDetail>(`/api/production-bibles/${productionBibleId}`);
}

/** PUT /api/production-bibles/{id} — update production bible */
export function updateProductionBible(productionBibleId: string, body: UpdateProductionBibleRequest): Promise<ProductionBibleListItem> {
  return request<ProductionBibleListItem>(`/api/production-bibles/${productionBibleId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** DELETE /api/production-bibles/{id} — soft delete production bible */
export function deleteProductionBible(productionBibleId: string): Promise<{ status: string; production_bible_id: string }> {
  return request<{ status: string; production_bible_id: string }>(`/api/production-bibles/${productionBibleId}`, {
    method: "DELETE",
  });
}

/** POST /api/production-bibles/{id}/duplicate — duplicate production bible */
export function duplicateProductionBible(productionBibleId: string, name?: string): Promise<ProductionBibleListItem> {
  const qs = name ? `?name=${encodeURIComponent(name)}` : "";
  return request<ProductionBibleListItem>(`/api/production-bibles/${productionBibleId}/duplicate${qs}`, {
    method: "POST",
  });
}

/** POST /api/production-bibles/{id}/assets — create asset in production bible */
export function createAsset(productionBibleId: string, body: CreateAssetRequest): Promise<AssetResponse> {
  return request<AssetResponse>(`/api/production-bibles/${productionBibleId}/assets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ============================================================================
// Deprecated aliases for backward-compat (will be removed in future cleanup)
// ============================================================================

/** @deprecated Use listProductionBibles */
export const listManifests = listProductionBibles;
/** @deprecated Use createProductionBible */
export const createManifest = createProductionBible;
/** @deprecated Use importSceneToProductionBible */
export const importSceneToManifest = importSceneToProductionBible;
/** @deprecated Use getProductionBibleDetail */
export const getManifestDetail = getProductionBibleDetail;
/** @deprecated Use updateProductionBible */
export const updateManifest = updateProductionBible;
/** @deprecated Use deleteProductionBible */
export const deleteManifest = deleteProductionBible;
/** @deprecated Use duplicateProductionBible */
export const duplicateManifest = duplicateProductionBible;

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

/** POST /api/production-bibles/{id}/upload-video — upload video for frame extraction */
export async function uploadVideoForProductionBible(
  productionBibleId: string,
  file: File,
): Promise<{ task_id: string; status: string; production_bible_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/production-bibles/${productionBibleId}/upload-video`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<{ task_id: string; status: string; production_bible_id: string }>;
}

/** GET /api/production-bibles/{id}/extraction-progress — poll extraction progress */
export function getExtractionProgress(productionBibleId: string): Promise<ProcessingProgress> {
  return request<ProcessingProgress>(`/api/production-bibles/${productionBibleId}/extraction-progress`);
}

/** POST /api/production-bibles/{id}/process — trigger background processing */
export function processProductionBible(productionBibleId: string): Promise<{ task_id: string; status: string }> {
  return request<{ task_id: string; status: string }>(`/api/production-bibles/${productionBibleId}/process`, {
    method: "POST",
  });
}

/** GET /api/production-bibles/{id}/progress — poll processing progress */
export function getProcessingProgress(productionBibleId: string): Promise<ProcessingProgress> {
  return request<ProcessingProgress>(`/api/production-bibles/${productionBibleId}/progress`);
}

/** @deprecated Use uploadVideoForProductionBible */
export const uploadVideoForManifest = uploadVideoForProductionBible;
/** @deprecated Use processProductionBible */
export const processManifest = processProductionBible;

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

/** GET /api/production-bibles/{id} — fetch assets for a production bible (used in EditForkPanel) */
export async function fetchProductionBibleAssets(productionBibleId: string): Promise<AssetResponse[]> {
  const res = await fetch(`/api/production-bibles/${productionBibleId}`);
  if (!res.ok) throw new ApiError(res.status, await res.text());
  const data = await res.json();
  return data.assets;
}

/** @deprecated Use fetchProductionBibleAssets */
export const fetchManifestAssets = fetchProductionBibleAssets;

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

/** PUT /api/sequences/{id}/scenes/reorder — bulk reorder scenes within a sequence */
export function reorderScenesInSequence(sequenceId: string, body: SceneReorderInSequenceRequest): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/sequences/${sequenceId}/scenes/reorder`, {
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

// ============================================================================
// Phase 17: Production Bible Entity CRUD
// ============================================================================

// --- Characters ---

/** GET /api/production-bibles/{id}/characters — list characters */
export function listCharacters(bibleId: string): Promise<CharacterResponse[]> {
  return request<CharacterResponse[]>(`/api/production-bibles/${bibleId}/characters`);
}

/** POST /api/production-bibles/{id}/characters — create character */
export function createCharacter(
  bibleId: string,
  data: { name: string; role: string; description?: string; arc?: string; base_appearance?: string; prompt_tags?: string[] },
): Promise<CharacterResponse> {
  return request<CharacterResponse>(`/api/production-bibles/${bibleId}/characters`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** GET /api/characters/{id} — get character detail */
export function getCharacter(characterId: string): Promise<CharacterResponse> {
  return request<CharacterResponse>(`/api/characters/${characterId}`);
}

/** PUT /api/characters/{id} — update character */
export function updateCharacter(
  characterId: string,
  data: Partial<{ name: string; role: string; description: string; arc: string; base_appearance: string; prompt_tags: string[] }>,
): Promise<CharacterResponse> {
  return request<CharacterResponse>(`/api/characters/${characterId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/characters/{id} — delete character */
export function deleteCharacter(characterId: string): Promise<void> {
  return request<void>(`/api/characters/${characterId}`, { method: "DELETE" });
}

/** POST /api/characters/{id}/wardrobes — create wardrobe item */
export function createWardrobe(
  characterId: string,
  data: { label: string; scene_context?: string; prompt_descriptor?: string; is_default?: boolean },
): Promise<WardrobeResponse> {
  return request<WardrobeResponse>(`/api/characters/${characterId}/wardrobes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/wardrobes/{id} — update wardrobe item */
export function updateWardrobe(
  wardrobeId: string,
  data: Partial<{ label: string; scene_context: string; prompt_descriptor: string; is_default: boolean }>,
): Promise<WardrobeResponse> {
  return request<WardrobeResponse>(`/api/wardrobes/${wardrobeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/wardrobes/{id} — delete wardrobe item */
export function deleteWardrobe(wardrobeId: string): Promise<void> {
  return request<void>(`/api/wardrobes/${wardrobeId}`, { method: "DELETE" });
}

/** PUT /api/characters/{id}/voice-profile — upsert voice profile */
export function upsertVoiceProfile(
  characterId: string,
  data: { voice_id?: string; adapter_type?: string; style_notes?: string },
): Promise<VoiceProfileResponse> {
  return request<VoiceProfileResponse>(`/api/characters/${characterId}/voice-profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/characters/{id}/voice-profile — delete voice profile */
export function deleteVoiceProfile(characterId: string): Promise<void> {
  return request<void>(`/api/characters/${characterId}/voice-profile`, { method: "DELETE" });
}

// --- Sets ---

/** GET /api/production-bibles/{id}/sets — list sets */
export function listSets(bibleId: string): Promise<SetResponse[]> {
  return request<SetResponse[]>(`/api/production-bibles/${bibleId}/sets`);
}

/** POST /api/production-bibles/{id}/sets — create set */
export function createSet(
  bibleId: string,
  data: { name: string; style_tags?: string[]; lighting_notes?: string; prompt_tags?: string[] },
): Promise<SetResponse> {
  return request<SetResponse>(`/api/production-bibles/${bibleId}/sets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** GET /api/sets/{id} — get set detail */
export function getSet(setId: string): Promise<SetResponse> {
  return request<SetResponse>(`/api/sets/${setId}`);
}

/** PUT /api/sets/{id} — update set */
export function updateSet(
  setId: string,
  data: Partial<{ name: string; style_tags: string[]; lighting_notes: string; prompt_tags: string[] }>,
): Promise<SetResponse> {
  return request<SetResponse>(`/api/sets/${setId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/sets/{id} — delete set */
export function deleteSet(setId: string): Promise<void> {
  return request<void>(`/api/sets/${setId}`, { method: "DELETE" });
}

/** POST /api/sets/{id}/upload-reference — upload reference image for set */
export async function uploadSetReference(setId: string, file: File): Promise<SetResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/sets/${setId}/upload-reference`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<SetResponse>;
}

/** PUT /api/sets/{id}/sonic-identity — upsert sonic identity */
export function upsertSonicIdentity(
  setId: string,
  data: { ambience_description?: string; generation_prompt?: string },
): Promise<SonicIdentityResponse> {
  return request<SonicIdentityResponse>(`/api/sets/${setId}/sonic-identity`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/sets/{id}/sonic-identity — delete sonic identity */
export function deleteSonicIdentity(setId: string): Promise<void> {
  return request<void>(`/api/sets/${setId}/sonic-identity`, { method: "DELETE" });
}

// --- Props ---

/** GET /api/production-bibles/{id}/props — list props */
export function listProps(bibleId: string): Promise<PropResponse[]> {
  return request<PropResponse[]>(`/api/production-bibles/${bibleId}/props`);
}

/** POST /api/production-bibles/{id}/props — create prop */
export function createProp(
  bibleId: string,
  data: { name: string; description?: string; associated_characters?: string[]; prompt_tags?: string[] },
): Promise<PropResponse> {
  return request<PropResponse>(`/api/production-bibles/${bibleId}/props`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/props/{id} — update prop */
export function updateProp(
  propId: string,
  data: Partial<{ name: string; description: string; associated_characters: string[]; prompt_tags: string[] }>,
): Promise<PropResponse> {
  return request<PropResponse>(`/api/props/${propId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/props/{id} — delete prop */
export function deleteProp(propId: string): Promise<void> {
  return request<void>(`/api/props/${propId}`, { method: "DELETE" });
}

// --- Score Themes ---

/** GET /api/production-bibles/{id}/score-themes — list score themes */
export function listScoreThemes(bibleId: string): Promise<ScoreThemeResponse[]> {
  return request<ScoreThemeResponse[]>(`/api/production-bibles/${bibleId}/score-themes`);
}

/** POST /api/production-bibles/{id}/score-themes — create score theme */
export function createScoreTheme(
  bibleId: string,
  data: { name: string; mood_descriptors?: string[]; tempo_notes?: string; usage_notes?: string; generation_prompt?: string },
): Promise<ScoreThemeResponse> {
  return request<ScoreThemeResponse>(`/api/production-bibles/${bibleId}/score-themes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/score-themes/{id} — update score theme */
export function updateScoreTheme(
  themeId: string,
  data: Partial<{ name: string; mood_descriptors: string[]; tempo_notes: string; usage_notes: string; generation_prompt: string }>,
): Promise<ScoreThemeResponse> {
  return request<ScoreThemeResponse>(`/api/score-themes/${themeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/score-themes/{id} — delete score theme */
export function deleteScoreTheme(themeId: string): Promise<void> {
  return request<void>(`/api/score-themes/${themeId}`, { method: "DELETE" });
}

// --- SFX Items ---

/** GET /api/production-bibles/{id}/sfx — list SFX items (optional category filter) */
export function listSFXItems(bibleId: string, category?: string): Promise<SFXItemResponse[]> {
  const qs = category ? `?category=${encodeURIComponent(category)}` : "";
  return request<SFXItemResponse[]>(`/api/production-bibles/${bibleId}/sfx${qs}`);
}

/** POST /api/production-bibles/{id}/sfx — create SFX item */
export function createSFXItem(
  bibleId: string,
  data: { name: string; category: string; generation_prompt?: string; tags?: string[] },
): Promise<SFXItemResponse> {
  return request<SFXItemResponse>(`/api/production-bibles/${bibleId}/sfx`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/sfx/{id} — update SFX item */
export function updateSFXItem(
  sfxId: string,
  data: Partial<{ name: string; category: string; generation_prompt: string; tags: string[] }>,
): Promise<SFXItemResponse> {
  return request<SFXItemResponse>(`/api/sfx/${sfxId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/sfx/{id} — delete SFX item */
export function deleteSFXItem(sfxId: string): Promise<void> {
  return request<void>(`/api/sfx/${sfxId}`, { method: "DELETE" });
}

// --- Character uploads ---

/** POST /api/characters/{id}/actor-refs -- upload actor reference image */
export async function uploadActorRef(characterId: string, file: File): Promise<CharacterResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/characters/${characterId}/actor-refs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<CharacterResponse>;
}

/** POST /api/characters/{id}/generate-appearance -- generate base appearance from actor refs */
export async function generateAppearance(characterId: string): Promise<CharacterResponse> {
  const res = await fetch(`/api/characters/${characterId}/generate-appearance`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<CharacterResponse>;
}

/** POST /api/wardrobes/{id}/upload-reference -- upload wardrobe reference image */
export async function uploadWardrobeReference(wardrobeId: string, file: File): Promise<WardrobeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/wardrobes/${wardrobeId}/upload-reference`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<WardrobeResponse>;
}

/** POST /api/props/{id}/upload-reference -- upload prop reference image */
export async function uploadPropReference(propId: string, file: File): Promise<PropResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/props/${propId}/upload-reference`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<PropResponse>;
}

// --- Audio uploads ---

/** POST /api/sonic-identities/{id}/upload-audio -- upload sonic identity reference audio */
export async function uploadSonicIdentityAudio(sonicIdentityId: string, file: File): Promise<SonicIdentityResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/sonic-identities/${sonicIdentityId}/upload-audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<SonicIdentityResponse>;
}

/** POST /api/score-themes/{id}/upload-audio -- upload score theme reference audio */
export async function uploadScoreThemeAudio(scoreThemeId: string, file: File): Promise<ScoreThemeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/score-themes/${scoreThemeId}/upload-audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<ScoreThemeResponse>;
}

/** POST /api/sfx/{id}/upload-audio -- upload SFX item source audio */
export async function uploadSFXAudio(sfxItemId: string, file: File): Promise<SFXItemResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/sfx/${sfxItemId}/upload-audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<SFXItemResponse>;
}

// --- Migration ---

/** POST /api/production-bibles/{id}/migrate-entities — migrate assets to structured entities */
export function migrateEntities(
  bibleId: string,
): Promise<{ characters_created: number; sets_created: number }> {
  return request<{ characters_created: number; sets_created: number }>(
    `/api/production-bibles/${bibleId}/migrate-entities`,
    { method: "POST" },
  );
}

// ============================================================================
// Screenplay API (Phase 18)
// ============================================================================

/** GET /api/productions/{id}/screenplay — get or auto-create screenplay */
export function getScreenplay(productionId: string): Promise<ScreenplayResponse> {
  return request<ScreenplayResponse>(`/api/productions/${productionId}/screenplay`);
}

/** PUT /api/productions/{id}/screenplay — update screenplay fields */
export function updateScreenplay(productionId: string, data: ScreenplayUpdate): Promise<ScreenplayResponse> {
  return request<ScreenplayResponse>(`/api/productions/${productionId}/screenplay`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** POST /api/productions/{id}/screenplay/generate — full chain background task (202) */
export async function generateScreenplayFull(productionId: string, textModel?: string): Promise<void> {
  const body: Record<string, string> = {};
  if (textModel) body.text_model = textModel;
  const res = await fetch(`/api/productions/${productionId}/screenplay/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, data.detail ?? res.statusText);
  }
  // 202 response — no body needed
}

/** POST /api/productions/{id}/screenplay/generate-{step} — regenerate single step */
export function generateScreenplayStep(
  productionId: string,
  step: string,
  textModel?: string,
): Promise<ScreenplayResponse> {
  const body: Record<string, string> = {};
  if (textModel) body.text_model = textModel;
  return request<ScreenplayResponse>(
    `/api/productions/${productionId}/screenplay/generate-${step}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/** PATCH /api/productions/{id}/screenplay/status — transition screenplay status */
export function updateScreenplayStatus(productionId: string, status: string): Promise<ScreenplayResponse> {
  return request<ScreenplayResponse>(
    `/api/productions/${productionId}/screenplay/status`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    },
  );
}

/** POST /api/productions/{id}/screenplay/generate-scenes — create scenes from locked breakdown */
export function generateScenesFromScreenplay(
  productionId: string,
  force?: boolean,
): Promise<GeneratedSceneResult[]> {
  const qs = force ? "?force=true" : "";
  return request<GeneratedSceneResult[]>(
    `/api/productions/${productionId}/screenplay/generate-scenes${qs}`,
    { method: "POST" },
  );
}

// ============================================================================
// Audio generation (ElevenLabs)
// ============================================================================

/** GET /api/elevenlabs/voices — search or list ElevenLabs voices */
export function searchElevenLabsVoices(query?: string): Promise<VoiceInfo[]> {
  const qs = query ? `?search=${encodeURIComponent(query)}` : "";
  return request<VoiceInfo[]>(`/api/elevenlabs/voices${qs}`);
}

/** GET /api/elevenlabs/voices/{voiceId} — resolve voice ID to name + labels */
export function resolveElevenLabsVoice(voiceId: string): Promise<VoiceInfo> {
  return request<VoiceInfo>(`/api/elevenlabs/voices/${encodeURIComponent(voiceId)}`);
}

/** POST /api/characters/{id}/generate-voice-sample — generate TTS sample */
export function generateVoiceSample(
  characterId: string,
  text?: string,
): Promise<VoiceProfileResponse> {
  const body: Record<string, string> = {};
  if (text) body.text = text;
  return request<VoiceProfileResponse>(
    `/api/characters/${characterId}/generate-voice-sample`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/** POST /api/sfx/{id}/generate — generate SFX audio from generation_prompt */
export function generateSFX(sfxItemId: string): Promise<SFXItemResponse> {
  return request<SFXItemResponse>(`/api/sfx/${sfxItemId}/generate`, {
    method: "POST",
  });
}

// ============================================================================
// Phase 22: Asset Library — Actors
// ============================================================================

/** GET /api/asset-library/actors — list all actors */
export function listActors(search?: string): Promise<ActorListItem[]> {
  const qs = search ? `?search=${encodeURIComponent(search)}` : "";
  return request<ActorListItem[]>(`/api/asset-library/actors${qs}`);
}

/** POST /api/asset-library/actors — create actor */
export function createActor(
  data: { name: string; description?: string; base_appearance_prompt?: string; prompt_tags?: string[] },
): Promise<ActorListItem> {
  return request<ActorListItem>("/api/asset-library/actors", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** GET /api/asset-library/actors/{id} — get actor detail */
export function getActor(id: string): Promise<ActorDetail> {
  return request<ActorDetail>(`/api/asset-library/actors/${id}`);
}

/** PUT /api/asset-library/actors/{id} — update actor */
export function updateActor(
  id: string,
  data: Partial<{ name: string; description: string; base_appearance_prompt: string; prompt_tags: string[] }>,
): Promise<ActorListItem> {
  return request<ActorListItem>(`/api/asset-library/actors/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/asset-library/actors/{id} — delete actor */
export function deleteActor(id: string): Promise<void> {
  return request<void>(`/api/asset-library/actors/${id}`, { method: "DELETE" });
}

/** POST /api/asset-library/actors/{id}/refs — upload actor reference image */
export async function uploadActorLibraryRef(actorId: string, file: File, label?: string): Promise<ActorRefResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (label) formData.append("label", label);
  const res = await fetch(`/api/asset-library/actors/${actorId}/refs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<ActorRefResponse>;
}

/** DELETE /api/asset-library/actor-refs/{id} — delete actor ref */
export function deleteActorLibraryRef(refId: string): Promise<void> {
  return request<void>(`/api/asset-library/actor-refs/${refId}`, { method: "DELETE" });
}

/** POST /api/asset-library/actors/{id}/voice-profiles — create voice profile */
export function createActorVoiceProfile(
  actorId: string,
  data: Partial<ActorVoiceProfileResponse>,
): Promise<ActorVoiceProfileResponse> {
  return request<ActorVoiceProfileResponse>(`/api/asset-library/actors/${actorId}/voice-profiles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/asset-library/actor-voice-profiles/{id} — update voice profile */
export function updateActorVoiceProfile(
  id: string,
  data: Partial<ActorVoiceProfileResponse>,
): Promise<ActorVoiceProfileResponse> {
  return request<ActorVoiceProfileResponse>(`/api/asset-library/actor-voice-profiles/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/asset-library/actor-voice-profiles/{id} — delete voice profile */
export function deleteActorVoiceProfile(id: string): Promise<void> {
  return request<void>(`/api/asset-library/actor-voice-profiles/${id}`, { method: "DELETE" });
}

/** POST /api/asset-library/actor-voice-profiles/{id}/test — generate TTS sample */
export function testActorVoiceProfile(id: string, text?: string): Promise<ActorVoiceProfile> {
  return request<ActorVoiceProfile>(`/api/asset-library/actor-voice-profiles/${id}/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(text ? { text } : {}),
  });
}

/** POST /api/asset-library/actors/{id}/wardrobe-presets — create wardrobe preset */
export function createActorWardrobePreset(
  actorId: string,
  data: { label: string; description?: string },
): Promise<ActorWardrobePresetResponse> {
  return request<ActorWardrobePresetResponse>(`/api/asset-library/actors/${actorId}/wardrobe-presets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/asset-library/actor-wardrobe-presets/{id} — update wardrobe preset */
export function updateActorWardrobePreset(
  id: string,
  data: Partial<{ label: string; description: string }>,
): Promise<ActorWardrobePresetResponse> {
  return request<ActorWardrobePresetResponse>(`/api/asset-library/actor-wardrobe-presets/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/asset-library/actor-wardrobe-presets/{id} — delete wardrobe preset */
export function deleteActorWardrobePreset(id: string): Promise<void> {
  return request<void>(`/api/asset-library/actor-wardrobe-presets/${id}`, { method: "DELETE" });
}

// ============================================================================
// Phase 22: Asset Library — Library Sets
// ============================================================================

/** GET /api/asset-library/sets — list library sets */
export function listLibrarySets(search?: string): Promise<LibrarySetListItem[]> {
  const qs = search ? `?search=${encodeURIComponent(search)}` : "";
  return request<LibrarySetListItem[]>(`/api/asset-library/sets${qs}`);
}

/** POST /api/asset-library/sets — create library set */
export function createLibrarySet(
  data: { name: string; description?: string; reverse_prompt?: string; style_tags?: string[]; prompt_tags?: string[]; lighting_notes?: string },
): Promise<LibrarySetListItem> {
  return request<LibrarySetListItem>("/api/asset-library/sets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** GET /api/asset-library/sets/{id} — get library set detail */
export function getLibrarySet(id: string): Promise<LibrarySetDetail> {
  return request<LibrarySetDetail>(`/api/asset-library/sets/${id}`);
}

/** PUT /api/asset-library/sets/{id} — update library set */
export function updateLibrarySet(
  id: string,
  data: Partial<{ name: string; description: string; reverse_prompt: string; style_tags: string[]; prompt_tags: string[]; lighting_notes: string }>,
): Promise<LibrarySetListItem> {
  return request<LibrarySetListItem>(`/api/asset-library/sets/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/asset-library/sets/{id} — delete library set */
export function deleteLibrarySet(id: string): Promise<void> {
  return request<void>(`/api/asset-library/sets/${id}`, { method: "DELETE" });
}

/** POST /api/asset-library/sets/{id}/refs — upload set reference image */
export async function uploadLibrarySetRef(setId: string, file: File, label?: string): Promise<LibrarySetRefResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (label) formData.append("label", label);
  const res = await fetch(`/api/asset-library/sets/${setId}/refs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<LibrarySetRefResponse>;
}

/** POST /api/asset-library/sets/{id}/generate-metadata — auto-generate description, reverse_prompt, lighting_notes */
export function generateLibrarySetMetadata(
  setId: string,
): Promise<{ description: string; reverse_prompt: string; lighting_notes: string }> {
  return request<{ description: string; reverse_prompt: string; lighting_notes: string }>(
    `/api/asset-library/sets/${setId}/generate-metadata`,
    { method: "POST" },
  );
}

/** POST /api/asset-library/sets/{id}/generate-image — generate reference image from prompt */
export function generateLibrarySetImage(
  setId: string,
  prompt: string,
  imageModel?: string,
  referenceImageId?: string,
): Promise<{ id: string; library_set_id: string; image_url: string; label: string | null; is_primary: boolean; created_at: string }> {
  return request(`/api/asset-library/sets/${setId}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      image_model: imageModel || undefined,
      reference_image_id: referenceImageId || undefined,
    }),
  });
}

/** DELETE /api/asset-library/set-refs/{id} — delete set ref */
export function deleteLibrarySetRef(refId: string): Promise<void> {
  return request<void>(`/api/asset-library/set-refs/${refId}`, { method: "DELETE" });
}

// ============================================================================
// Phase 22: Asset Library — Library Props
// ============================================================================

/** GET /api/asset-library/props — list library props */
export function listLibraryProps(search?: string): Promise<LibraryPropListItem[]> {
  const qs = search ? `?search=${encodeURIComponent(search)}` : "";
  return request<LibraryPropListItem[]>(`/api/asset-library/props${qs}`);
}

/** POST /api/asset-library/props — create library prop */
export function createLibraryProp(
  data: { name: string; description?: string; appearance_prompt?: string; prompt_tags?: string[] },
): Promise<LibraryPropListItem> {
  return request<LibraryPropListItem>("/api/asset-library/props", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** GET /api/asset-library/props/{id} — get library prop detail */
export function getLibraryProp(id: string): Promise<LibraryPropDetail> {
  return request<LibraryPropDetail>(`/api/asset-library/props/${id}`);
}

/** PUT /api/asset-library/props/{id} — update library prop */
export function updateLibraryProp(
  id: string,
  data: Partial<{ name: string; description: string; appearance_prompt: string; prompt_tags: string[] }>,
): Promise<LibraryPropListItem> {
  return request<LibraryPropListItem>(`/api/asset-library/props/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/asset-library/props/{id} — delete library prop */
export function deleteLibraryProp(id: string): Promise<void> {
  return request<void>(`/api/asset-library/props/${id}`, { method: "DELETE" });
}

/** POST /api/asset-library/props/{id}/refs — upload prop reference image */
export async function uploadLibraryPropRef(propId: string, file: File, label?: string): Promise<LibraryPropRefResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (label) formData.append("label", label);
  const res = await fetch(`/api/asset-library/props/${propId}/refs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<LibraryPropRefResponse>;
}

/** DELETE /api/asset-library/prop-refs/{id} — delete prop ref */
export function deleteLibraryPropRef(refId: string): Promise<void> {
  return request<void>(`/api/asset-library/prop-refs/${refId}`, { method: "DELETE" });
}

/** POST /api/asset-library/actors/{id}/generate-metadata — auto-generate description, base_appearance_prompt */
export function generateActorMetadata(
  actorId: string,
): Promise<{ description: string; base_appearance_prompt: string }> {
  return request<{ description: string; base_appearance_prompt: string }>(
    `/api/asset-library/actors/${actorId}/generate-metadata`,
    { method: "POST" },
  );
}

/** POST /api/asset-library/actors/{id}/generate-image — generate reference image from prompt */
export function generateActorImage(
  actorId: string,
  prompt: string,
  imageModel?: string,
  referenceImageId?: string,
): Promise<{ id: string; actor_id: string; image_url: string; label: string | null; is_primary: boolean; created_at: string }> {
  return request(`/api/asset-library/actors/${actorId}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      image_model: imageModel || undefined,
      reference_image_id: referenceImageId || undefined,
    }),
  });
}

/** POST /api/asset-library/props/{id}/generate-metadata — auto-generate description, appearance_prompt */
export function generatePropMetadata(
  propId: string,
): Promise<{ description: string; appearance_prompt: string }> {
  return request<{ description: string; appearance_prompt: string }>(
    `/api/asset-library/props/${propId}/generate-metadata`,
    { method: "POST" },
  );
}

/** POST /api/asset-library/props/{id}/generate-image — generate reference image from prompt */
export function generatePropImage(
  propId: string,
  prompt: string,
  imageModel?: string,
  referenceImageId?: string,
): Promise<{ id: string; library_prop_id: string; image_url: string; label: string | null; is_primary: boolean; created_at: string }> {
  return request(`/api/asset-library/props/${propId}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      image_model: imageModel || undefined,
      reference_image_id: referenceImageId || undefined,
    }),
  });
}

// ============================================================================
// Phase 22: Asset Library — Sound Assets
// ============================================================================

/** GET /api/asset-library/sounds — list sound assets */
export function listSoundAssets(search?: string, category?: string): Promise<SoundAssetListItem[]> {
  const params: string[] = [];
  if (search) params.push(`search=${encodeURIComponent(search)}`);
  if (category) params.push(`category=${encodeURIComponent(category)}`);
  const qs = params.length ? `?${params.join("&")}` : "";
  return request<SoundAssetListItem[]>(`/api/asset-library/sounds${qs}`);
}

/** POST /api/asset-library/sounds — create sound asset */
export function createSoundAsset(
  data: { name: string; category: string; subcategory?: string; description?: string; generation_prompt?: string; tags?: string[] },
): Promise<SoundAssetDetail> {
  return request<SoundAssetDetail>("/api/asset-library/sounds", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** GET /api/asset-library/sounds/{id} — get sound asset detail */
export function getSoundAsset(id: string): Promise<SoundAssetDetail> {
  return request<SoundAssetDetail>(`/api/asset-library/sounds/${id}`);
}

/** PUT /api/asset-library/sounds/{id} — update sound asset */
export function updateSoundAsset(
  id: string,
  data: Partial<{ name: string; category: string; subcategory: string; description: string; generation_prompt: string; tags: string[] }>,
): Promise<SoundAssetDetail> {
  return request<SoundAssetDetail>(`/api/asset-library/sounds/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/asset-library/sounds/{id} — delete sound asset */
export function deleteSoundAsset(id: string): Promise<void> {
  return request<void>(`/api/asset-library/sounds/${id}`, { method: "DELETE" });
}

/** POST /api/asset-library/sounds/{id}/upload-audio — upload audio file */
export async function uploadSoundAssetAudio(id: string, file: File): Promise<SoundAssetDetail> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`/api/asset-library/sounds/${id}/upload-audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<SoundAssetDetail>;
}

// ============================================================================
// Phase 22: Bindings — Cast, Set, Prop, Sound
// ============================================================================

// --- Cast Bindings ---

/** GET /api/production-bibles/{id}/cast — list cast bindings */
export function listCastBindings(bibleId: string): Promise<CastBindingResponse[]> {
  return request<CastBindingResponse[]>(`/api/production-bibles/${bibleId}/cast`);
}

/** POST /api/production-bibles/{id}/cast — create cast binding */
export function createCastBinding(
  bibleId: string,
  data: {
    actor_id: string;
    tag: string;
    character_name: string;
    role: string;
    character_description?: string;
    character_arc?: string;
    wardrobe_override?: Record<string, unknown>;
    voice_profile_id?: string;
    behavioral_notes?: string;
    prompt_tags?: string[];
  },
): Promise<CastBindingResponse> {
  return request<CastBindingResponse>(`/api/production-bibles/${bibleId}/cast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/cast-bindings/{id} — update cast binding */
export function updateCastBinding(
  id: string,
  data: Partial<{
    tag: string;
    character_name: string;
    character_description: string;
    character_arc: string;
    role: string;
    wardrobe_override: Record<string, unknown>;
    voice_profile_id: string;
    behavioral_notes: string;
    prompt_tags: string[];
  }>,
): Promise<CastBindingResponse> {
  return request<CastBindingResponse>(`/api/cast-bindings/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/cast-bindings/{id} — delete cast binding */
export function deleteCastBinding(id: string): Promise<void> {
  return request<void>(`/api/cast-bindings/${id}`, { method: "DELETE" });
}

// --- Set Bindings ---

/** GET /api/production-bibles/{id}/set-bindings — list set bindings */
export function listSetBindings(bibleId: string): Promise<SetBindingResponse[]> {
  return request<SetBindingResponse[]>(`/api/production-bibles/${bibleId}/set-bindings`);
}

/** POST /api/production-bibles/{id}/set-bindings — create set binding */
export function createSetBinding(
  bibleId: string,
  data: {
    library_set_id: string;
    tag: string;
    production_name?: string;
    lighting_override?: string;
    sonic_override?: string;
    prompt_tags?: string[];
  },
): Promise<SetBindingResponse> {
  return request<SetBindingResponse>(`/api/production-bibles/${bibleId}/set-bindings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/set-bindings/{id} — update set binding */
export function updateSetBinding(
  id: string,
  data: Partial<{
    tag: string;
    production_name: string;
    lighting_override: string;
    sonic_override: string;
    prompt_tags: string[];
  }>,
): Promise<SetBindingResponse> {
  return request<SetBindingResponse>(`/api/set-bindings/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/set-bindings/{id} — delete set binding */
export function deleteSetBinding(id: string): Promise<void> {
  return request<void>(`/api/set-bindings/${id}`, { method: "DELETE" });
}

// --- Prop Bindings ---

/** GET /api/production-bibles/{id}/prop-bindings — list prop bindings */
export function listPropBindings(bibleId: string): Promise<PropBindingResponse[]> {
  return request<PropBindingResponse[]>(`/api/production-bibles/${bibleId}/prop-bindings`);
}

/** POST /api/production-bibles/{id}/prop-bindings — create prop binding */
export function createPropBinding(
  bibleId: string,
  data: {
    library_prop_id: string;
    tag: string;
    production_name?: string;
    notes?: string;
    prompt_tags?: string[];
  },
): Promise<PropBindingResponse> {
  return request<PropBindingResponse>(`/api/production-bibles/${bibleId}/prop-bindings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/prop-bindings/{id} — update prop binding */
export function updatePropBinding(
  id: string,
  data: Partial<{
    tag: string;
    production_name: string;
    notes: string;
    prompt_tags: string[];
  }>,
): Promise<PropBindingResponse> {
  return request<PropBindingResponse>(`/api/prop-bindings/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/prop-bindings/{id} — delete prop binding */
export function deletePropBinding(id: string): Promise<void> {
  return request<void>(`/api/prop-bindings/${id}`, { method: "DELETE" });
}

// --- Sound Bindings ---

/** GET /api/production-bibles/{id}/sound-bindings — list sound bindings */
export function listSoundBindings(bibleId: string): Promise<SoundBindingResponse[]> {
  return request<SoundBindingResponse[]>(`/api/production-bibles/${bibleId}/sound-bindings`);
}

/** POST /api/production-bibles/{id}/sound-bindings — create sound binding */
export function createSoundBinding(
  bibleId: string,
  data: {
    sound_asset_id: string;
    tag?: string;
    usage_notes?: string;
    prompt_tags?: string[];
  },
): Promise<SoundBindingResponse> {
  return request<SoundBindingResponse>(`/api/production-bibles/${bibleId}/sound-bindings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** PUT /api/sound-bindings/{id} — update sound binding */
export function updateSoundBinding(
  id: string,
  data: Partial<{
    tag: string;
    usage_notes: string;
    prompt_tags: string[];
  }>,
): Promise<SoundBindingResponse> {
  return request<SoundBindingResponse>(`/api/sound-bindings/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/** DELETE /api/sound-bindings/{id} — delete sound binding */
export function deleteSoundBinding(id: string): Promise<void> {
  return request<void>(`/api/sound-bindings/${id}`, { method: "DELETE" });
}

// --- Bound Assets Summary (Phase 23) ---

/** GET /api/production-bibles/{id}/bound-assets/summary — flat list of all bindings */
export function getBoundAssetsSummary(bibleId: string): Promise<BoundAssetSummary[]> {
  return request<BoundAssetSummary[]>(`/api/production-bibles/${bibleId}/bound-assets/summary`);
}

export { ApiError };
