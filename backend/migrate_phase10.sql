-- Phase 10: Adaptive Prompt Rewriting — add rewritten prompt columns
ALTER TABLE scene_manifests ADD COLUMN rewritten_keyframe_prompt TEXT;
ALTER TABLE scene_manifests ADD COLUMN rewritten_video_prompt TEXT;
