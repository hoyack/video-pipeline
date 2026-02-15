-- Migration: Add audio_enabled column to projects table
-- Run against existing vidpipe.db to add audio support
ALTER TABLE projects ADD COLUMN audio_enabled BOOLEAN;
