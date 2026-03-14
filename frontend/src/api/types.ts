/** Candidate score data from GET /api/scenes/{id}/shots/{idx}/candidates */
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

/** Ollama model entry stored in user settings */
export interface OllamaModelEntry {
  id: string;       // e.g. "ollama/llama3.1"
  label: string;    // e.g. "Llama 3.1"
  enabled: boolean;
  vision: boolean;  // true if this model supports vision
}

/** Request body for POST /api/generate */
export interface GenerateRequest {
  title?: string | null;
  prompt: string;
  style: string;
  aspect_ratio: string;
  clip_duration: number;
  total_duration: number;
  text_model: string;
  image_model: string;
  video_model: string;
  enable_audio: boolean;
  production_bible_id?: string;
  quality_mode?: boolean;
  candidate_count?: number;
  vision_model?: string;
  run_through?: string | null;
}

/** Response from POST /api/generate */
export interface GenerateResponse {
  scene_id: string;
  status: string;
  status_url: string;
}

/** Response from GET /api/scenes/{id}/status */
export interface StatusResponse {
  scene_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  error_message: string | null;
}

/** Reference image selected for a shot's video generation */
export interface ShotReference {
  asset_id: string;
  manifest_tag: string;
  name: string;
  asset_type: string;
  thumbnail_url: string | null;
  reference_image_url: string | null;
  quality_score: number | null;
  is_face_crop: boolean;
}

/** Shot detail within SceneDetail */
export interface ShotDetail {
  shot_index: number;
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
  selected_references?: ShotReference[];
  // PipeSVN staleness
  start_keyframe_staleness?: string | null;
  end_keyframe_staleness?: string | null;
  clip_staleness?: string | null;
  start_keyframe_prompt_used?: string | null;
  end_keyframe_prompt_used?: string | null;
  clip_prompt_used?: string | null;
  rewritten_keyframe_prompt?: string | null;
  rewritten_video_prompt?: string | null;
  is_empty_slot?: boolean;
  generation_status?: string | null;
}

/** Response from GET /api/scenes/{id} */
export interface SceneDetail {
  scene_id: string;
  title?: string | null;
  prompt: string;
  style?: string | null;
  aspect_ratio?: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  shot_count: number;
  shots: ShotDetail[];
  error_message: string | null;
  total_duration?: number | null;
  clip_duration?: number | null;
  text_model?: string | null;
  image_model?: string | null;
  video_model?: string | null;
  audio_enabled?: boolean | null;
  forked_from?: string | null;
  production_bible_id?: string | null;
  quality_mode?: boolean;
  candidate_count?: number;
  vision_model?: string | null;
  run_through?: string | null;
  // PipeSVN
  head_sha?: string | null;
}

/** Per-shot edit payload for in-place editing */
export interface ShotEditPayload {
  shot_description?: string;
  start_frame_prompt?: string;
  end_frame_prompt?: string;
  video_motion_prompt?: string;
  transition_notes?: string;
}

/** Request body for PATCH /api/scenes/{id}/edit */
export interface EditSceneRequest {
  prompt?: string;
  title?: string;
  style?: string;
  aspect_ratio?: string;
  clip_duration?: number;
  target_shot_count?: number;
  text_model?: string;
  image_model?: string;
  video_model?: string;
  vision_model?: string;
  audio_enabled?: boolean;
  production_bible_id?: string | null;
  shot_edits?: Record<number, ShotEditPayload>;
  removed_shots?: number[];
  shot_order?: number[];  // shot_indices in desired display order
  commit_message?: string;
  expected_sha?: string;
}

/** Response from PATCH /api/scenes/{id}/edit */
export interface EditSceneResponse {
  scene_id: string;
  head_sha: string;
  message: string;
  changes_count: number;
}

/** Checkpoint list item from GET /api/scenes/{id}/checkpoints */
export interface CheckpointListItem {
  sha: string;
  parent_sha: string | null;
  message: string;
  changes_count: number;
  created_at: string;
}

/** Checkpoint diff from GET /api/scenes/{id}/checkpoints/{sha}/diff */
export interface CheckpointDiff {
  sha: string;
  message: string;
  changes: Array<{
    type: string;
    field?: string;
    shot_index?: number;
    old?: string | null;
    new?: string | null;
  }>;
}

/** Request body for POST /api/scenes/{id}/shots/{idx}/regenerate */
export interface RegenerateShotRequest {
  targets: string[];
  prompt_overrides?: Record<string, string>;
  skip_checkpoint?: boolean;
  video_model?: string;
  image_model?: string;
  shot_edits?: Record<string, string>;
}

/** Request body for POST /api/scenes/{id}/regenerate */
export interface RegenerateSceneRequest {
  scope: "stale" | "all" | "stitch_only" | "storyboard" | "keyframes" | "clips" | "all_phases";
  shot_indices?: number[];
  text_model?: string;
  image_model?: string;
  video_model?: string;
  run_through?: string | null;
}

/** Item in GET /api/scenes list */
export interface SceneListItem {
  scene_id: string;
  title?: string | null;
  prompt: string;
  status: string;
  created_at: string;
  total_duration?: number | null;
  clip_duration?: number | null;
  text_model?: string | null;
  image_model?: string | null;
  video_model?: string | null;
  audio_enabled?: boolean | null;
  vision_model?: string | null;
  run_through?: string | null;
  style?: string | null;
  aspect_ratio?: string | null;
  thumbnail_url?: string | null;
  production_id?: string | null;
  sequence_id?: string | null;
  scene_order?: number | null;
  screenplay_breakdown_index?: number | null;
}

/** Paginated response envelope for GET /api/scenes */
export interface PaginatedScenes {
  items: SceneListItem[];
  total: number;
  page: number;
  per_page: number;
}

/** Response from POST /api/scenes/{id}/resume */
export interface ResumeResponse {
  scene_id: string;
  status: string;
  status_url: string;
}

/** Response from POST /api/scenes/{id}/stop */
export interface StopResponse {
  scene_id: string;
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

/** Request body for POST /api/scenes/{id}/fork */
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
  vision_model?: string;
  shot_edits?: Record<number, Record<string, string>>;
  deleted_shots?: number[];
  clear_keyframes?: number[];
  asset_changes?: AssetChanges;
}

/** Response from POST /api/scenes/{id}/fork */
export interface ForkResponse {
  scene_id: string;
  forked_from: string;
  status: string;
  status_url: string;
  copied_shots: number;
  resume_from: string;
}

/** Response from GET /api/metrics */
export interface MetricsResponse {
  total_scenes: number;
  status_counts: Record<string, number>;
  style_counts: Record<string, number>;
  aspect_ratio_counts: Record<string, number>;
  text_model_counts: Record<string, number>;
  image_model_counts: Record<string, number>;
  video_model_counts: Record<string, number>;
  audio_counts: Record<string, number>;
  shot_count_counts: Record<string, number>;
  total_estimated_cost: number;
  total_video_seconds: number;
  avg_clip_duration: number | null;
}

/** Item in GET /api/production-bibles list */
export interface ProductionBibleListItem {
  production_bible_id: string;
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

/** Response from GET /api/production-bibles/{id} */
export interface ProductionBibleDetail {
  production_bible_id: string;
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
  parent_production_bible_id: string | null;
  source_video_duration: number | null;
  created_at: string;
  updated_at: string;
  assets: AssetResponse[];
}

/** Asset within a production bible */
export interface AssetResponse {
  asset_id: string;
  production_bible_id: string;
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

/** Request body for POST /api/production-bibles */
export interface CreateProductionBibleRequest {
  name: string;
  description?: string;
  category?: string;
  tags?: string[];
}

/** Request body for PUT /api/production-bibles/{id} */
export interface UpdateProductionBibleRequest {
  name?: string;
  description?: string;
  category?: string;
  tags?: string[];
}

// ============================================================================
// Backward-compatibility aliases (deprecated — use Production Bible names)
// ============================================================================

/** @deprecated Use ProductionBibleListItem */
export type ManifestListItem = ProductionBibleListItem;
/** @deprecated Use ProductionBibleDetail */
export type ManifestDetail = ProductionBibleDetail;
/** @deprecated Use CreateProductionBibleRequest */
export type CreateManifestRequest = CreateProductionBibleRequest;
/** @deprecated Use UpdateProductionBibleRequest */
export type UpdateManifestRequest = UpdateProductionBibleRequest;

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

/** Response from GET /api/settings */
export interface UserSettingsResponse {
  enabled_text_models: string[] | null;
  enabled_image_models: string[] | null;
  enabled_video_models: string[] | null;
  default_text_model: string | null;
  default_image_model: string | null;
  default_video_model: string | null;
  default_vision_model: string | null;
  gcp_project_id: string | null;
  gcp_location: string | null;
  has_api_key: boolean;
  has_gcp_service_account: boolean;
  comfyui_host: string | null;
  has_comfyui_key: boolean;
  comfyui_cost_per_second: number | null;
  ollama_use_cloud: boolean;
  has_ollama_key: boolean;
  ollama_endpoint: string | null;
  ollama_models: OllamaModelEntry[] | null;
  show_cost: boolean;
  has_elevenlabs_key: boolean;
  default_voice_id: string | null;
}

/** Request body for PUT /api/settings */
export interface UserSettingsUpdate {
  enabled_text_models?: string[] | null;
  enabled_image_models?: string[] | null;
  enabled_video_models?: string[] | null;
  default_text_model?: string | null;
  default_image_model?: string | null;
  default_video_model?: string | null;
  default_vision_model?: string | null;
  gcp_project_id?: string | null;
  gcp_location?: string | null;
  vertex_api_key?: string | null;
  clear_api_key?: boolean;
  gcp_service_account_json?: string | null;
  clear_gcp_service_account?: boolean;
  comfyui_host?: string | null;
  comfyui_api_key?: string | null;
  clear_comfyui_key?: boolean;
  comfyui_cost_per_second?: number | null;
  ollama_use_cloud?: boolean;
  ollama_api_key?: string | null;
  clear_ollama_key?: boolean;
  ollama_endpoint?: string | null;
  ollama_models?: OllamaModelEntry[] | null;
  show_cost?: boolean;
  elevenlabs_api_key?: string | null;
  clear_elevenlabs_key?: boolean;
  default_voice_id?: string | null;
}

/** Lightweight response from GET /api/settings/models */
export interface EnabledModelsResponse {
  enabled_text_models: string[] | null;
  enabled_image_models: string[] | null;
  enabled_video_models: string[] | null;
  default_text_model: string | null;
  default_image_model: string | null;
  default_video_model: string | null;
  default_vision_model: string | null;
  comfyui_cost_per_second: number | null;
  ollama_models: OllamaModelEntry[] | null;
  show_cost: boolean;
  default_voice_id: string | null;
}

/** Request body for POST /api/scenes/{id}/shots/{idx}/regenerate-text */
export interface RegenerateTextRequest {
  field: string;
  extra_context?: string;
  text_model?: string;
  shot_edits?: Record<string, string>;
  prompt?: string;
}

/** Response from POST /api/scenes/{id}/shots/{idx}/regenerate-text */
export interface RegenerateTextResponse {
  field: string;
  text: string;
}

/** Request body for POST /api/scenes/{id}/generate-shot-fields */
export interface GenerateShotFieldsRequest {
  shot_index: number;
  all_shot_edits?: Record<number, Record<string, string>>;
  text_model?: string;
  prompt?: string;
}

/** Response from POST /api/scenes/{id}/generate-shot-fields */
export interface GenerateShotFieldsResponse {
  shot_description: string;
  start_frame_prompt: string;
  end_frame_prompt: string;
  video_motion_prompt: string;
  transition_notes: string;
}

/** Request body for POST /api/scenes/{id}/generate-new-shot */
export interface GenerateNewShotRequest {
  shot_index: number;
  all_shot_edits?: Record<number, Record<string, string>>;
  text_model?: string;
  image_model?: string;
  video_model?: string;
  prompt?: string;
}

/** Response from POST /api/scenes/{id}/generate-new-shot */
export interface GenerateNewShotResponse {
  shot_index: number;
  shot_description: string;
  start_frame_prompt: string;
  end_frame_prompt: string;
  video_motion_prompt: string;
  transition_notes: string;
  head_sha?: string | null;
}

/** Request body for POST /api/scenes (draft scene creation) */
export interface CreateDraftSceneRequest {
  prompt?: string;
  title?: string;
  style?: string;
  aspect_ratio?: string;
  clip_duration?: number;
  shot_count?: number;
  text_model?: string;
  image_model?: string;
  video_model?: string;
  enable_audio?: boolean;
  production_bible_id?: string | null;
  quality_mode?: boolean;
  candidate_count?: number;
  vision_model?: string | null;
}

/** Response from POST /api/scenes */
export interface CreateDraftSceneResponse {
  scene_id: string;
  status: string;
  shot_count: number;
}

/** Request body for POST /api/scenes/{id}/generate */
export interface StartGenerationRequest {
  run_through?: string | null;
  text_model?: string;
  image_model?: string;
  video_model?: string;
  vision_model?: string;
  clip_duration?: number;
  enable_audio?: boolean;
}

/** Response from POST /api/scenes/{id}/generate */
export interface StartGenerationResponse {
  scene_id: string;
  status: string;
}

// ============================================================================
// Sequence types (Issue #24 — Sequence Grouping Layer)
// ============================================================================

/** Response from GET /api/productions/{id}/sequences */
export interface SequenceResponse {
  id: string;
  production_id: string;
  title: string;
  description: string | null;
  order: number;
  act: string | null;  // ACT_1, ACT_2, ACT_3
  color: string | null;  // hex color
  scene_count: number;
  total_duration: number | null;
  created_at: string;
  updated_at: string;
}

/** Response from GET /api/sequences/{id} with scenes */
export interface SequenceWithScenes extends SequenceResponse {
  scenes: SceneListItem[];
}

/** Request body for POST /api/productions/{id}/sequences */
export interface SequenceCreate {
  title: string;
  description?: string;
  act?: string | null;
  color?: string | null;
}

/** Request body for PUT /api/sequences/{id} */
export interface SequenceUpdate {
  title?: string;
  description?: string;
  act?: string | null;
  color?: string | null;
}

/** Request body for PUT /api/productions/{id}/sequences/reorder */
export interface SequenceReorderRequest {
  sequence_ids: string[];
}

/** Request body for PUT /api/sequences/{id}/scenes/reorder */
export interface SceneReorderInSequenceRequest {
  scene_ids: string[];
}

/** Request body for PUT /api/scenes/{id}/sequence */
export interface AssignSequenceRequest {
  sequence_id: string | null;
  scene_order?: number;
}

// ============================================================================
// Phase 17: Production Bible Entity Types
// ============================================================================

export interface CharacterResponse {
  character_id: string;
  production_bible_id: string;
  name: string;
  role: "PROTAGONIST" | "ANTAGONIST" | "SUPPORTING" | "EXTRA";
  description: string | null;
  arc: string | null;
  actor_refs: string[] | null;
  base_appearance: string | null;
  prompt_tags: string[] | null;
  wardrobe: WardrobeResponse[];
  voice_profile: VoiceProfileResponse | null;
  created_at: string;
  updated_at: string;
}

export interface WardrobeResponse {
  wardrobe_id: string;
  character_id: string;
  label: string;
  reference_images: string[] | null;
  scene_context: string | null;
  prompt_descriptor: string | null;
  is_default: boolean;
  created_at: string;
}

export interface VoiceProfileResponse {
  voice_profile_id: string;
  character_id: string;
  voice_id: string | null;
  adapter_type: string;
  style_notes: string | null;
  sample_audio: string | null;
  created_at: string;
}

export interface SetResponse {
  set_id: string;
  production_bible_id: string;
  name: string;
  reference_image: string | null;
  reverse_prompt: string | null;
  style_tags: string[] | null;
  lighting_notes: string | null;
  prompt_tags: string[] | null;
  sonic_identity: SonicIdentityResponse | null;
  created_at: string;
  updated_at: string;
}

export interface SonicIdentityResponse {
  sonic_identity_id: string;
  set_id: string;
  ambience_description: string | null;
  reference_audio: string | null;
  generation_prompt: string | null;
  created_at: string;
}

export interface PropResponse {
  prop_id: string;
  production_bible_id: string;
  name: string;
  reference_image: string | null;
  description: string | null;
  associated_characters: string[] | null;
  prompt_tags: string[] | null;
  created_at: string;
  updated_at: string;
}

export interface ScoreThemeResponse {
  score_theme_id: string;
  production_bible_id: string;
  name: string;
  mood_descriptors: string[] | null;
  tempo_notes: string | null;
  usage_notes: string | null;
  reference_audio: string | null;
  generation_prompt: string | null;
  adapter_type: string;
  created_at: string;
  updated_at: string;
}

export interface SFXItemResponse {
  sfx_item_id: string;
  production_bible_id: string;
  name: string;
  category: "IMPACT" | "MECHANICAL" | "NATURAL" | "UI" | "FOLEY" | "AMBIENCE";
  source_audio: string | null;
  generation_prompt: string | null;
  tags: string[] | null;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// Phase 18: Screenplay types
// ============================================================================

/** Character breakdown entry within a screenplay */
export interface CharacterBreakdownEntry {
  name: string;
  role: string;
  arc: string;
  description: string;
  manifest_tag: string | null;
}

/** Scene breakdown entry within a screenplay */
export interface SceneBreakdownEntry {
  scene_number: number;
  slugline: string;
  intent: string;
  emotional_beat: string;
  story_state_in: string;
  story_state_out: string;
  characters_present: string[];
  set_ref: string | null;
  props_required: string[];
}

/** Shot list entry within a screenplay */
export interface ShotListEntry {
  scene_number: number;
  shot_number: number;
  shot_type: string;
  camera_movement: string;
  action_notes: string;
  duration_hint: string;
}

/** Response from GET /api/productions/{id}/screenplay */
export interface ScreenplayResponse {
  id: string;
  production_id: string;
  title: string | null;
  genre: string | null;
  status: string; // DRAFT | IN_REVIEW | LOCKED
  logline: string | null;
  treatment: string | null;
  character_breakdowns: CharacterBreakdownEntry[] | null;
  scene_breakdown: SceneBreakdownEntry[] | null;
  script: string | null;
  shot_list: ShotListEntry[] | null;
  text_model: string | null;
  generating_step: string | null;
  created_at: string;
  updated_at: string;
}

/** Request body for PUT /api/productions/{id}/screenplay */
export interface ScreenplayUpdate {
  title?: string;
  genre?: string;
  logline?: string;
  treatment?: string;
  character_breakdowns?: CharacterBreakdownEntry[];
  scene_breakdown?: SceneBreakdownEntry[];
  script?: string;
  shot_list?: ShotListEntry[];
  text_model?: string;
}

/** Scene created from screenplay breakdown */
export interface GeneratedSceneResult {
  scene_id: string;
  title: string;
  scene_number: number;
}

// ============================================================================
// Audio generation types (ElevenLabs adapter)
// ============================================================================

/** Voice info from ElevenLabs voice search/resolve */
export interface VoiceInfo {
  voice_id: string;
  name: string;
  labels: Record<string, string>;
  preview_url: string | null;
}

// ============================================================================
// Phase 22: Asset Library & Actor-Character Model
// ============================================================================

/** Actor reference image */
export interface ActorRef {
  id: string;
  actor_id: string;
  image_url: string;
  label: string | null;
  is_primary: boolean;
  created_at: string;
}

/** Actor voice profile (library-level) */
export interface ActorVoiceProfile {
  id: string;
  actor_id: string;
  voice_id: string | null;
  adapter_type: string | null;
  style_notes: string | null;
  sample_url: string | null;
  created_at: string;
}

/** Actor wardrobe preset (library-level) */
export interface ActorWardrobePreset {
  id: string;
  actor_id: string;
  label: string;
  description: string | null;
  reference_images: string[] | null;
  created_at: string;
}

/** Full actor detail with sub-entities */
export interface Actor {
  id: string;
  name: string;
  description: string | null;
  base_appearance_prompt: string | null;
  prompt_tags: string[] | null;
  refs: ActorRef[];
  voice_profiles: ActorVoiceProfile[];
  wardrobe_presets: ActorWardrobePreset[];
  binding_count?: number;
  lora_url: string | null;
  lora_trained_at: string | null;
  lora_training_status: 'QUEUED' | 'TRAINING' | 'COMPLETED' | 'FAILED' | null;
  created_at: string;
  updated_at: string;
}

/** Actor list item (compact, from GET /api/asset-library/actors) */
export interface ActorListItem {
  id: string;
  name: string;
  description: string | null;
  prompt_tags: string[] | null;
  ref_count: number;
  binding_count: number;
  primary_ref_url: string | null;
  lora_training_status: 'QUEUED' | 'TRAINING' | 'COMPLETED' | 'FAILED' | null;
  created_at: string;
}

/** Library set reference image */
export interface LibrarySetRef {
  id: string;
  library_set_id: string;
  image_url: string;
  label: string | null;
  is_primary: boolean;
  created_at: string;
}

/** Library sonic identity (1:1 with library set) */
export interface LibrarySonicIdentity {
  id: string;
  library_set_id: string;
  ambience_description: string | null;
  reference_audio_url: string | null;
  generation_prompt: string | null;
  created_at: string;
}

/** Full library set detail */
export interface LibrarySet {
  id: string;
  name: string;
  description: string | null;
  reverse_prompt: string | null;
  style_tags: string[] | null;
  prompt_tags: string[] | null;
  lighting_notes: string | null;
  refs: LibrarySetRef[];
  sonic_identity: LibrarySonicIdentity | null;
  binding_count?: number;
  created_at: string;
  updated_at: string;
}

/** Library set list item (compact) */
export interface LibrarySetListItem {
  id: string;
  name: string;
  description: string | null;
  ref_count: number;
  binding_count: number;
  primary_ref_url: string | null;
  created_at: string;
}

/** Library prop reference image */
export interface LibraryPropRef {
  id: string;
  library_prop_id: string;
  image_url: string;
  label: string | null;
  is_primary: boolean;
  created_at: string;
}

/** Full library prop detail */
export interface LibraryProp {
  id: string;
  name: string;
  description: string | null;
  appearance_prompt: string | null;
  prompt_tags: string[] | null;
  refs: LibraryPropRef[];
  binding_count?: number;
  created_at: string;
  updated_at: string;
}

/** Library prop list item (compact) */
export interface LibraryPropListItem {
  id: string;
  name: string;
  description: string | null;
  ref_count: number;
  binding_count: number;
  primary_ref_url: string | null;
  created_at: string;
}

/** Full sound asset detail */
export interface SoundAsset {
  id: string;
  name: string;
  category: "SCORE_THEME" | "SFX" | "AMBIENCE" | "FOLEY" | "UI";
  subcategory: string | null;
  description: string | null;
  audio_url: string | null;
  generation_prompt: string | null;
  tags: string[] | null;
  binding_count?: number;
  created_at: string;
  updated_at: string;
}

/** Sound asset list item (compact) */
export interface SoundAssetListItem {
  id: string;
  name: string;
  category: string;
  subcategory: string | null;
  description: string | null;
  has_audio: boolean;
  binding_count: number;
  created_at: string;
}

// === Binding Types ===

/** Cast binding (actor to production bible) */
export interface CastBinding {
  id: string;
  production_bible_id: string;
  actor_id: string;
  tag: string;
  character_name: string;
  character_description: string | null;
  character_arc: string | null;
  role: "LEAD" | "SUPPORTING" | "EXTRA" | "NARRATOR";
  wardrobe_override: Record<string, unknown>[] | null;
  voice_profile_id: string | null;
  behavioral_notes: string | null;
  prompt_tags: string[] | null;
  actor_name: string | null;
  primary_ref_url: string | null;
  actor?: Actor;
  created_at: string;
  updated_at: string;
}

/** Set binding (library set to production bible) */
export interface SetBinding {
  id: string;
  production_bible_id: string;
  library_set_id: string;
  tag: string;
  production_name: string | null;
  lighting_override: string | null;
  sonic_override: string | null;
  prompt_tags: string[] | null;
  set_name: string | null;
  primary_ref_url: string | null;
  library_set?: LibrarySet;
  created_at: string;
  updated_at: string;
}

/** Prop binding (library prop to production bible) */
export interface PropBinding {
  id: string;
  production_bible_id: string;
  library_prop_id: string;
  tag: string;
  production_name: string | null;
  notes: string | null;
  prompt_tags: string[] | null;
  prop_name: string | null;
  primary_ref_url: string | null;
  library_prop?: LibraryProp;
  created_at: string;
  updated_at: string;
}

/** Sound binding (sound asset to production bible) */
export interface SoundBinding {
  id: string;
  production_bible_id: string;
  sound_asset_id: string;
  tag: string | null;
  usage_notes: string | null;
  prompt_tags: string[] | null;
  sound_name: string | null;
  sound_category: string | null;
  sound_asset?: SoundAsset;
  created_at: string;
  updated_at: string;
}

// Backward-compat aliases for existing code using Response-suffixed names
/** @deprecated Use ActorRef */
export type ActorRefResponse = ActorRef;
/** @deprecated Use ActorVoiceProfile */
export type ActorVoiceProfileResponse = ActorVoiceProfile;
/** @deprecated Use ActorWardrobePreset */
export type ActorWardrobePresetResponse = ActorWardrobePreset;
/** @deprecated Use Actor */
export type ActorDetail = Actor;
/** @deprecated Use LibrarySetRef */
export type LibrarySetRefResponse = LibrarySetRef;
/** @deprecated Use LibrarySet */
export type LibrarySetDetail = LibrarySet;
/** @deprecated Use LibraryPropRef */
export type LibraryPropRefResponse = LibraryPropRef;
/** @deprecated Use LibraryProp */
export type LibraryPropDetail = LibraryProp;
/** @deprecated Use SoundAsset */
export type SoundAssetDetail = SoundAsset;
/** @deprecated Use CastBinding */
export type CastBindingResponse = CastBinding;
/** @deprecated Use SetBinding */
export type SetBindingResponse = SetBinding;
/** @deprecated Use PropBinding */
export type PropBindingResponse = PropBinding;
/** @deprecated Use SoundBinding */
export type SoundBindingResponse = SoundBinding;

// ============================================================================
// Phase 23: Bound Asset Summary
// ============================================================================

/** Single entry from GET /api/production-bibles/{id}/bound-assets/summary */
export interface BoundAssetSummary {
  tag: string;
  name: string;
  type: "CHARACTER" | "SET" | "PROP";
  primary_thumbnail_url: string | null;
  description: string | null;
}

// ============================================================================
// Phase 25: LoRA Training Types
// ============================================================================

/** Response from GET /api/asset-library/actors/{id}/lora-status */
export interface LoraStatusResponse {
  status: 'QUEUED' | 'TRAINING' | 'COMPLETED' | 'FAILED' | null;
  lora_url: string | null;
  trained_at: string | null;
  error: string | null;
}

/** Response from POST /api/asset-library/actors/{id}/train-lora */
export interface TrainLoraResponse {
  status: string;
  message: string;
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
