/** WebSocket event types pushed by the backend EventBus. */

export interface WsEventBase {
  type: string;
  ts: number;
}

export interface PhaseStartedEvent extends WsEventBase {
  type: "phase_started";
  phase: string;
  total_shots: number;
  message?: string;
}

export interface PhaseCompletedEvent extends WsEventBase {
  type: "phase_completed";
  phase: string;
  message?: string;
}

export interface PhaseProgressEvent extends WsEventBase {
  type: "phase_progress";
  phase: string;
  completed_shots: number;
  total_shots: number;
  active_shots: number;
  message: string;
}

export interface ShotStatusEvent extends WsEventBase {
  type: "shot_status";
  shot_index: number;
  status: string;
  phase: string;
  message?: string;
}

export interface ShotTextReadyEvent extends WsEventBase {
  type: "shot_text_ready";
  shot_index: number;
  message?: string;
}

export interface ShotKeyframeReadyEvent extends WsEventBase {
  type: "shot_keyframe_ready";
  shot_index: number;
  position: string;
  message?: string;
}

export interface ShotClipReadyEvent extends WsEventBase {
  type: "shot_clip_ready";
  shot_index: number;
  message?: string;
}

export interface StitchReadyEvent extends WsEventBase {
  type: "stitch_ready";
  message?: string;
}

export interface CheckpointCreatedEvent extends WsEventBase {
  type: "checkpoint_created";
  sha: string | null;
  message: string;
}

export interface ErrorEvent extends WsEventBase {
  type: "error";
  phase?: string;
  shot_index?: number;
  message: string;
}

export interface ShotRegenStartedEvent extends WsEventBase {
  type: "shot_regen_started";
  shot_index: number;
  targets: string[];
  message?: string;
}

export interface ShotRegenDoneEvent extends WsEventBase {
  type: "shot_regen_done";
  shot_index: number;
  message?: string;
}

export interface RefreshEvent extends WsEventBase {
  type: "refresh";
}

export interface RegenCompleteEvent extends WsEventBase {
  type: "regen_complete";
  scope: string;
  message?: string;
}

export interface TaskLogEvent extends WsEventBase {
  type: "task_log";
  phase?: string;
  shot_index?: number;
  level: "info" | "success" | "error";
  kind?: string;
  source?: string;
  summary: string;
  detail?: string;
}

export interface HeartbeatEvent extends WsEventBase {
  type: "heartbeat";
}

export type WsEvent =
  | PhaseStartedEvent
  | PhaseCompletedEvent
  | PhaseProgressEvent
  | ShotStatusEvent
  | ShotTextReadyEvent
  | ShotKeyframeReadyEvent
  | ShotClipReadyEvent
  | StitchReadyEvent
  | CheckpointCreatedEvent
  | ErrorEvent
  | ShotRegenStartedEvent
  | ShotRegenDoneEvent
  | RefreshEvent
  | RegenCompleteEvent
  | TaskLogEvent
  | HeartbeatEvent;
