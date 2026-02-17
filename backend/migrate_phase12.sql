-- Phase 12: Fork System Integration with Manifests — add asset inheritance columns
ALTER TABLE assets ADD COLUMN is_inherited BOOLEAN DEFAULT FALSE;
ALTER TABLE assets ADD COLUMN inherited_from_asset TEXT;
ALTER TABLE assets ADD COLUMN inherited_from_project TEXT;
