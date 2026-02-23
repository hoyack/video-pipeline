-- Migration: Rename "Scene" → "Shot" across all database tables
-- Run: sqlite3 vidpipe.db < backend/migrate_scene_to_shot.sql

-- 1. Rename tables
ALTER TABLE scenes RENAME TO shots;
ALTER TABLE scene_manifests RENAME TO shot_manifests;
ALTER TABLE scene_audio_manifests RENAME TO shot_audio_manifests;

-- 2. Rename columns in shots (formerly scenes)
ALTER TABLE shots RENAME COLUMN scene_index TO shot_index;
ALTER TABLE shots RENAME COLUMN scene_description TO shot_description;

-- 3. Rename columns in shot_manifests (formerly scene_manifests)
ALTER TABLE shot_manifests RENAME COLUMN scene_index TO shot_index;

-- 4. Rename columns in shot_audio_manifests (formerly scene_audio_manifests)
ALTER TABLE shot_audio_manifests RENAME COLUMN scene_index TO shot_index;

-- 5. Rename columns in asset_appearances
ALTER TABLE asset_appearances RENAME COLUMN scene_index TO shot_index;

-- 6. Rename columns in generation_candidates
ALTER TABLE generation_candidates RENAME COLUMN scene_index TO shot_index;

-- 7. Rename columns in projects
ALTER TABLE projects RENAME COLUMN target_scene_count TO target_shot_count;

-- 8. Rename columns in keyframes
ALTER TABLE keyframes RENAME COLUMN scene_id TO shot_id;

-- 9. Rename columns in video_clips
ALTER TABLE video_clips RENAME COLUMN scene_id TO shot_id;

-- 10. Recreate index with new name
DROP INDEX IF EXISTS idx_candidates_project_scene;
CREATE INDEX idx_candidates_project_shot ON generation_candidates(project_id, shot_index);
