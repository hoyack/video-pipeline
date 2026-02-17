-- Migration: Add video source columns to manifests table
-- For video upload → frame extraction feature

ALTER TABLE manifests ADD COLUMN source_video_path VARCHAR(500);
ALTER TABLE manifests ADD COLUMN source_video_duration REAL;
