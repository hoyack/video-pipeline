/** Candidate score data from GET /api/projects/{id}/scenes/{idx}/candidates */
export interface CandidateScore {
  candidate_id: string;
  candidate_number: number;
  local_path: string | null;
  manifest_adherence_score: number | null;
  visual_quality_score: number | null;
  continuity_score: number | null;
  prompt_adherence_score: number | null;
  composite_score: number | null;
  is_selected: boolean;
  selected_by: string;
  generation_cost: number;
  scoring_cost: number;
}

/** Request body for POST /api/generate */
export interface GenerateRequest {
  prompt: string;
  style: string;
  aspect_ratio: string;
  clip_duration: number;
  total_duration: number;
  text_model: string;
  image_model: string;
  video_model: string;
  enable_audio: boolean;
  manifest_id?: string;
  quality_mode?: boolean;
  candidate_count?: number;
}

/** Response from POST /api/generate */
export interface GenerateResponse {
  project_id: string;
  status: string;
  status_url: string;
}

/** Response from GET /api/projects/{id}/status */
export interface StatusResponse {
  project_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  error_message: string | null;
}

/** Reference image selected for a scene's video generation */
export interface SceneReference {
  asset_id: string;
  manifest_tag: string;
  name: string;
  asset_type: string;
  thumbnail_url: string | null;
  reference_image_url: string | null;
  quality_score: number | null;
  is_face_crop: boolean;
}

/** Scene detail within ProjectDetail */
export interface SceneDetail {
  scene_index: number;
  description: string;
  status: string;
  has_start_keyframe: boolean;
  has_end_keyframe: boolean;
  has_clip: boolean;
  clip_status: string | null;
  start_frame_prompt?: string | null;
  end_frame_prompt?: string | null;
  video_motion_prompt?: string | null;
  transition_notes?: string | null;
  start_keyframe_url?: string | null;
  end_keyframe_url?: string | null;
  clip_url?: string | null;
  selected_references?: SceneReference[];
}

/** Response from GET /api/projects/{id} */
export interface ProjectDetail {
  project_id: string;
  prompt: string;
  style: string;
  aspect_ratio: string;
  status: string;
  created_at: string;
  updated_at: string;
  scene_count: number;
  scenes: SceneDetail[];
  error_message: string | null;
  total_duration?: number | null;
  clip_duration?: number | null;
  text_model?: string | null;
  image_model?: string | null;
  video_model?: string | null;
  audio_enabled?: boolean | null;
  forked_from?: string | null;
  manifest_id?: string | null;
  quality_mode?: boolean;
  candidate_count?: number;
}

/** Item in GET /api/projects list */
export interface ProjectListItem {
  project_id: string;
  prompt: string;
  status: string;
  created_at: string;
  total_duration?: number | null;
  clip_duration?: number | null;
  text_model?: string | null;
  image_model?: string | null;
  video_model?: string | null;
  audio_enabled?: boolean | null;
}

/** Response from POST /api/projects/{id}/resume */
export interface ResumeResponse {
  project_id: string;
  status: string;
  status_url: string;
}

/** Response from POST /api/projects/{id}/stop */
export interface StopResponse {
  project_id: string;
  status: string;
}

/** Asset modification in a fork */
export interface ModifiedAsset {
  changes: {
    reverse_prompt?: string;
    name?: string;
    visual_description?: string;
  };
}

/** New reference upload to add in fork */
export interface NewForkUpload {
  image_data: string;  // base64-encoded
  name: string;
  asset_type: string;
  description?: string;
  tags?: string[];
}

/** Asset changes for fork request */
export interface AssetChanges {
  modified_assets?: Record<string, ModifiedAsset>;
  removed_asset_ids?: string[];
  new_uploads?: NewForkUpload[];
}

/** Request body for POST /api/projects/{id}/fork */
export interface ForkRequest {
  prompt?: string;
  style?: string;
  aspect_ratio?: string;
  clip_duration?: number;
  total_duration?: number;
  text_model?: string;
  image_model?: string;
  video_model?: string;
  audio_enabled?: boolean;
  scene_edits?: Record<number, Record<string, string>>;
  deleted_scenes?: number[];
  clear_keyframes?: number[];
  asset_changes?: AssetChanges;
}

/** Response from POST /api/projects/{id}/fork */
export interface ForkResponse {
  project_id: string;
  forked_from: string;
  status: string;
  status_url: string;
  copied_scenes: number;
  resume_from: string;
}

/** Response from GET /api/metrics */
export interface MetricsResponse {
  total_projects: number;
  status_counts: Record<string, number>;
  style_counts: Record<string, number>;
  aspect_ratio_counts: Record<string, number>;
  text_model_counts: Record<string, number>;
  image_model_counts: Record<string, number>;
  video_model_counts: Record<string, number>;
  audio_counts: Record<string, number>;
  scene_count_counts: Record<string, number>;
  total_estimated_cost: number;
  total_video_seconds: number;
  avg_clip_duration: number | null;
}

/** Item in GET /api/manifests list */
export interface ManifestListItem {
  manifest_id: string;
  name: string;
  description: string | null;
  thumbnail_url: string | null;
  category: string;
  tags: string[] | null;
  status: string;
  asset_count: number;
  times_used: number;
  last_used_at: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

/** Response from GET /api/manifests/{id} */
export interface ManifestDetail {
  manifest_id: string;
  name: string;
  description: string | null;
  thumbnail_url: string | null;
  category: string;
  tags: string[] | null;
  status: string;
  processing_progress: Record<string, unknown> | null;
  contact_sheet_url: string | null;
  asset_count: number;
  total_processing_cost: number;
  times_used: number;
  last_used_at: string | null;
  version: number;
  parent_manifest_id: string | null;
  source_video_duration: number | null;
  created_at: string;
  updated_at: string;
  assets: AssetResponse[];
}

/** Asset within a manifest */
export interface AssetResponse {
  asset_id: string;
  manifest_id: string;
  asset_type: string;
  name: string;
  manifest_tag: string;
  user_tags: string[] | null;
  reference_image_url: string | null;
  thumbnail_url: string | null;
  description: string | null;
  source: string;
  sort_order: number;
  created_at: string;
  // Phase 5 fields
  reverse_prompt: string | null;
  visual_description: string | null;
  detection_class: string | null;
  detection_confidence: number | null;
  is_face_crop: boolean;
  quality_score: number | null;
}

/** Request body for POST /api/manifests */
export interface CreateManifestRequest {
  name: string;
  description?: string;
  category?: string;
  tags?: string[];
}

/** Request body for PUT /api/manifests/{id} */
export interface UpdateManifestRequest {
  name?: string;
  description?: string;
  category?: string;
  tags?: string[];
}

/** Request body for POST /api/manifests/{id}/assets */
export interface CreateAssetRequest {
  name: string;
  asset_type: string;
  description?: string;
  user_tags?: string[];
}

/** Request body for PUT /api/assets/{id} */
export interface UpdateAssetRequest {
  name?: string;
  description?: string;
  asset_type?: string;
  user_tags?: string[];
  sort_order?: number;
  reverse_prompt?: string;
  visual_description?: string;
}

/** Processing progress response */
export interface ProcessingProgress {
  status: "processing" | "complete" | "error" | "not_started";
  current_step: string | null;
  progress: {
    uploads_total?: number;
    uploads_processed?: number;
    crops_total?: number;
    crops_reverse_prompted?: number;
    face_merges?: number;
  } | null;
  error: string | null;
}
