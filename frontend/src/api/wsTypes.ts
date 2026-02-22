/** WebSocket event types pushed by the backend EventBus. */

export interface WsEventBase {
  type: string;
  ts: number;
}

export interface PhaseStartedEvent extends WsEventBase {
  type: "phase_started";
  phase: string;
  total_scenes: number;
}

export interface PhaseCompletedEvent extends WsEventBase {
  type: "phase_completed";
  phase: string;
}

export interface SceneStatusEvent extends WsEventBase {
  type: "scene_status";
  scene_index: number;
  status: string;
  phase: string;
}

export interface SceneTextReadyEvent extends WsEventBase {
  type: "scene_text_ready";
  scene_index: number;
}

export interface SceneKeyframeReadyEvent extends WsEventBase {
  type: "scene_keyframe_ready";
  scene_index: number;
  position: string;
}

export interface SceneClipReadyEvent extends WsEventBase {
  type: "scene_clip_ready";
  scene_index: number;
}

export interface StitchReadyEvent extends WsEventBase {
  type: "stitch_ready";
}

export interface CheckpointCreatedEvent extends WsEventBase {
  type: "checkpoint_created";
  sha: string | null;
  message: string;
}

export interface ErrorEvent extends WsEventBase {
  type: "error";
  phase?: string;
  scene_index?: number;
  message: string;
}

export interface SceneRegenStartedEvent extends WsEventBase {
  type: "scene_regen_started";
  scene_index: number;
  targets: string[];
}

export interface SceneRegenDoneEvent extends WsEventBase {
  type: "scene_regen_done";
  scene_index: number;
}

export interface RefreshEvent extends WsEventBase {
  type: "refresh";
}

export interface RegenCompleteEvent extends WsEventBase {
  type: "regen_complete";
  scope: string;
}

export interface HeartbeatEvent extends WsEventBase {
  type: "heartbeat";
}

export type WsEvent =
  | PhaseStartedEvent
  | PhaseCompletedEvent
  | SceneStatusEvent
  | SceneTextReadyEvent
  | SceneKeyframeReadyEvent
  | SceneClipReadyEvent
  | StitchReadyEvent
  | CheckpointCreatedEvent
  | ErrorEvent
  | SceneRegenStartedEvent
  | SceneRegenDoneEvent
  | RefreshEvent
  | RegenCompleteEvent
  | HeartbeatEvent;
