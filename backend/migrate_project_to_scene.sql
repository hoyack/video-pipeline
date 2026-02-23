-- Migration: Rename "Project" → "Scene" + Add "Productions" entity
-- Run: sqlite3 vidpipe.db < backend/migrate_project_to_scene.sql

-- 1. Rename main tables
ALTER TABLE projects RENAME TO scenes;
ALTER TABLE project_checkpoints RENAME TO scene_checkpoints;

-- 2. Rename project_id → scene_id in dependent tables
-- shots
ALTER TABLE shots RENAME COLUMN project_id TO scene_id;

-- shot_manifests
ALTER TABLE shot_manifests RENAME COLUMN project_id TO scene_id;

-- shot_audio_manifests
ALTER TABLE shot_audio_manifests RENAME COLUMN project_id TO scene_id;

-- asset_appearances
ALTER TABLE asset_appearances RENAME COLUMN project_id TO scene_id;

-- generation_candidates
ALTER TABLE generation_candidates RENAME COLUMN project_id TO scene_id;

-- pipeline_runs
ALTER TABLE pipeline_runs RENAME COLUMN project_id TO scene_id;

-- manifest_snapshots
ALTER TABLE manifest_snapshots RENAME COLUMN project_id TO scene_id;

-- scene_checkpoints (already renamed table)
ALTER TABLE scene_checkpoints RENAME COLUMN project_id TO scene_id;

-- 3. Rename inherited_from_project → inherited_from_scene in assets
ALTER TABLE assets RENAME COLUMN inherited_from_project TO inherited_from_scene;

-- 4. Recreate index for generation_candidates
DROP INDEX IF EXISTS idx_candidates_project_shot;
CREATE INDEX idx_candidates_scene_shot ON generation_candidates(scene_id, shot_index);

-- 5. Recreate indexes for scene_checkpoints
DROP INDEX IF EXISTS idx_checkpoint_project_created;
CREATE INDEX idx_checkpoint_scene_created ON scene_checkpoints(scene_id, created_at);

-- 6. Create productions table
CREATE TABLE IF NOT EXISTS productions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    tags TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Add production_id to scenes
ALTER TABLE scenes ADD COLUMN production_id TEXT REFERENCES productions(id);
