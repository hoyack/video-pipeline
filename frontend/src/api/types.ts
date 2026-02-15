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

/** Scene detail within ProjectDetail */
export interface SceneDetail {
  scene_index: number;
  description: string;
  status: string;
  has_start_keyframe: boolean;
  has_end_keyframe: boolean;
  has_clip: boolean;
  clip_status: string | null;
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
  text_model?: string | null;
  image_model?: string | null;
  video_model?: string | null;
  audio_enabled?: boolean | null;
}

/** Item in GET /api/projects list */
export interface ProjectListItem {
  project_id: string;
  prompt: string;
  status: string;
  created_at: string;
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
