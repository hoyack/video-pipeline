---
phase: 04-manifest-system-foundation
plan: 01
subsystem: manifest-system
tags: [database, api, backend, crud]
dependencies:
  requires: []
  provides: [manifest-crud-api, asset-crud-api, manifest-models]
  affects: [projects-table]
tech-stack:
  added: [manifest_service.py]
  patterns: [sqlalchemy-mapped-annotations, fastapi-async-sessions, pydantic-schemas, soft-delete]
key-files:
  created:
    - backend/vidpipe/services/manifest_service.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/api/routes.py
decisions:
  - Assets belong to manifests only (no project_id column) per V2 architecture
  - Soft delete for manifests (deleted_at column) prevents data loss
  - Auto-generate manifest_tag on asset creation (CHAR_01, CHAR_02, OBJ_01, etc.)
  - Explicit index on Asset.manifest_id for query performance on SQLite
  - Image upload saves to tmp/manifests/{manifest_id}/uploads/ directory structure
  - Return 409 Conflict when attempting to delete manifest referenced by projects
metrics:
  duration: 5.4
  completed: 2026-02-16T14:32:00Z
  tasks: 2
  files_created: 1
  files_modified: 3
  endpoints_added: 11
---

# Phase 04 Plan 01: Manifest System Backend Foundation Summary

**One-liner:** Complete backend for Manifest System with Manifest and Asset ORM models, business logic service layer, and 11 REST API endpoints for full CRUD plus image upload.

## Overview

Successfully implemented the complete backend foundation for the V2 Manifest System. This establishes manifests as standalone, reusable entities with database persistence, a service layer for business logic, and HTTP endpoints that the frontend can consume. The implementation follows existing codebase patterns (SQLAlchemy 2.0 Mapped annotations, FastAPI async routes, Pydantic schemas) and introduces no new dependencies.

## What Was Built

### Database Schema (Task 1)

Created two new ORM models in `backend/vidpipe/db/models.py`:

**Manifest Model** (`manifests` table):
- 18 fields: id, name, description, thumbnail_url, category, tags, status, processing_progress, contact_sheet_url, asset_count, total_processing_cost, times_used, last_used_at, version, parent_manifest_id, deleted_at, created_at, updated_at
- DRAFT status by default
- Soft delete support via deleted_at column
- Self-referential FK for duplication tracking (parent_manifest_id)

**Asset Model** (`assets` table):
- 12 fields: id, manifest_id, asset_type, name, manifest_tag, user_tags, reference_image_url, thumbnail_url, description, source, sort_order, created_at
- Belongs to manifests only (no project_id column) — critical architectural decision
- Explicit index on manifest_id for query performance on SQLite
- Auto-generated manifest_tag (CHAR_01, OBJ_02, ENV_03, etc.)

**Project Model Extension**:
- Added manifest_id (ForeignKey to manifests.id, nullable, indexed)
- Added manifest_version (Integer, nullable)
- Idempotent ALTER TABLE migrations in `backend/vidpipe/db/__init__.py`

### Service Layer (Task 2)

Created `backend/vidpipe/services/manifest_service.py` with 12 functions:

**Manifest Operations**:
1. `create_manifest` - Create with category validation
2. `list_manifests` - Filter by category/status, sort by updated_at/name/times_used/asset_count
3. `get_manifest` - Get single manifest (only non-deleted)
4. `update_manifest` - Update name/description/category/tags
5. `delete_manifest` - Soft delete with project reference check
6. `duplicate_manifest` - Create copy with all assets and parent_manifest_id

**Asset Operations**:
7. `create_asset` - Auto-generate manifest_tag, update manifest.asset_count
8. `list_assets` - Ordered by sort_order then created_at
9. `get_asset` - Get single asset by ID
10. `update_asset` - Regenerate manifest_tag if asset_type changes
11. `delete_asset` - Hard delete, decrement manifest.asset_count
12. `save_asset_image` - Pure filesystem I/O (wrap in asyncio.to_thread)

### REST API Endpoints (Task 2)

Added 11 endpoints to `backend/vidpipe/api/routes.py` under `/api/manifests`:

1. `POST /manifests` (201) - Create manifest
2. `GET /manifests` - List with optional category/status filters and sorting
3. `GET /manifests/{manifest_id}` - Get detail with assets list
4. `PUT /manifests/{manifest_id}` - Update manifest
5. `DELETE /manifests/{manifest_id}` - Soft delete (409 if used by projects)
6. `POST /manifests/{manifest_id}/duplicate` (201) - Duplicate with assets
7. `POST /manifests/{manifest_id}/assets` (201) - Create asset
8. `GET /manifests/{manifest_id}/assets` - List assets
9. `PUT /assets/{asset_id}` - Update asset
10. `DELETE /assets/{asset_id}` - Delete asset
11. `POST /assets/{asset_id}/upload` - Upload image (multipart/form-data)

All endpoints use fresh async sessions per request, convert ValueError to HTTPException(422), and follow existing FastAPI patterns.

## Deviations from Plan

None — plan executed exactly as written. No bugs discovered, no missing functionality encountered, no architectural changes required.

## Testing Results

All verification criteria passed:

1. **Model verification**: Manifest has 18 columns, Asset has 12 columns, Project has manifest_id column, Asset has NO project_id column ✓
2. **API server startup**: No errors, all routes registered ✓
3. **Manifest CRUD**: POST/GET/PUT/DELETE all return expected status codes ✓
4. **Asset CRUD**: POST/GET/PUT/DELETE work within manifest scope ✓
5. **Auto-tagging**: Creating two CHARACTER assets produces CHAR_01, CHAR_02 ✓
6. **Soft delete**: Deleted manifests excluded from list ✓
7. **Image upload**: POST /assets/{id}/upload with multipart file saves to tmp/manifests/{manifest_id}/uploads/ and updates reference_image_url ✓
8. **Duplication**: POST /manifests/{id}/duplicate creates copy with all assets and correct parent_manifest_id ✓

**Test commands executed**:
```bash
# Create manifest
curl -X POST http://localhost:8000/api/manifests -H "Content-Type: application/json" \
  -d '{"name":"Hero Characters","category":"CHARACTERS"}'
# Response: manifest_id, status=DRAFT, asset_count=0

# Create assets with auto-tagging
curl -X POST http://localhost:8000/api/manifests/{id}/assets -H "Content-Type: application/json" \
  -d '{"name":"Hero","asset_type":"CHARACTER"}'
# Response: manifest_tag=CHAR_01

curl -X POST http://localhost:8000/api/manifests/{id}/assets -H "Content-Type: application/json" \
  -d '{"name":"Sidekick","asset_type":"CHARACTER"}'
# Response: manifest_tag=CHAR_02

curl -X POST http://localhost:8000/api/manifests/{id}/assets -H "Content-Type: application/json" \
  -d '{"name":"Sword","asset_type":"OBJECT"}'
# Response: manifest_tag=OBJ_01

# Get manifest detail
curl http://localhost:8000/api/manifests/{id}
# Response: asset_count=3, assets array with all three assets

# Upload image
curl -X POST http://localhost:8000/api/assets/{asset_id}/upload -F "file=@test.png"
# Response: reference_image_url="tmp/manifests/{manifest_id}/uploads/{asset_id}_test.png"

# Soft delete
curl -X DELETE http://localhost:8000/api/manifests/{id}
# Response: status=deleted

# List manifests (deleted one not included)
curl http://localhost:8000/api/manifests
# Response: array without deleted manifest

# Duplicate
curl -X POST "http://localhost:8000/api/manifests/{id}/duplicate?name=My%20Copy"
# Response: new manifest with asset_count=1, parent_manifest_id set
```

All endpoints returned correct status codes and response shapes.

## Commits

| Task | Commit | Files | Description |
|------|--------|-------|-------------|
| 1 | 33d2145 | 2 | Add Manifest and Asset models with Project FK |
| 2 | de24bbe | 2 | Add manifest service layer and REST API endpoints |

## Key Decisions

1. **Assets belong to manifests only** - No project_id column on Asset table. This architectural decision from v2-manifest.md enables asset reuse across unlimited projects without retroactive updates.

2. **Soft delete for manifests** - Added deleted_at column instead of hard delete. Prevents data loss, allows "Restore" feature, projects continue working with deleted manifests.

3. **Auto-generate manifest_tag on creation** - Service layer generates CHAR_01, OBJ_02, ENV_03 tags sequentially by type. Makes manifest immediately readable even without processing (Stage 2).

4. **Explicit index on Asset.manifest_id** - SQLite doesn't auto-index foreign keys like PostgreSQL. Explicit index prevents slow queries as asset count grows.

5. **Image upload directory structure** - Save to `tmp/manifests/{manifest_id}/uploads/` with `{asset_id}_{filename}` naming. Ensures uniqueness, enables cleanup per manifest, mirrors existing project directory patterns from FileManager.

6. **409 Conflict on delete if referenced** - Check Project table for manifest_id references before deleting. Prevents breaking existing projects. Frontend can prompt user to unlink projects first.

## Performance Notes

- Soft delete query: `WHERE deleted_at IS NULL` — add index if performance degrades with thousands of manifests
- Asset count update: Done in application code (not trigger) for async session compatibility
- Image upload: Wrapped in `asyncio.to_thread()` to prevent blocking event loop during file I/O

## Future Integration Points

### Phase 05: Manifesting Engine
- Will consume POST /manifests/{id}/assets endpoint
- Will update manifest.status from DRAFT → PROCESSING → READY
- Will populate processing_progress JSON during CV analysis

### Phase 06: GenerateForm Integration
- Frontend form will call POST /manifests with user input
- Will display ManifestListItem responses in dropdown selector
- Will send manifest_id in POST /api/generate payload

### Phase 07: Manifest-Aware Storyboarding
- Storyboard generator will read manifest.assets via GET /manifests/{id}
- Will inject manifest_tag references (CHAR_01, OBJ_02) into scene prompts
- Will increment manifest.times_used on project creation

## Dependencies Satisfied

All must-haves from plan achieved:

- [x] Manifest records can be created, read, updated, and deleted via API
- [x] Asset records can be created within a manifest and listed via API
- [x] Projects can reference a manifest_id and manifest_version
- [x] Deleting a manifest used by active projects returns 409 Conflict
- [x] Manifest list supports category filtering and sort options
- [x] Image files can be uploaded to an asset within a manifest

All artifacts from must_haves section present:
- [x] `backend/vidpipe/db/models.py` contains `class Manifest` and `class Asset`
- [x] `backend/vidpipe/services/manifest_service.py` exports all required functions
- [x] `backend/vidpipe/api/routes.py` contains `@router.post("/manifests")`

All key links verified:
- [x] routes.py imports from manifest_service (`from vidpipe.services import manifest_service`)
- [x] manifest_service queries Manifest and Asset models (`select(Manifest)`, `select(Asset)`)
- [x] migrations add manifest_id columns to projects table (`ALTER TABLE projects ADD COLUMN manifest_id`)

## Next Steps

Phase 04 Plan 02 will build the React frontend components:
- ManifestLibrary.tsx - Grid view with filters/sort
- ManifestCard.tsx - Card component for library
- ManifestCreator.tsx - Stage 1 upload + tag UI
- AssetUploader.tsx - Drag-drop multi-file upload

These components will consume the 11 REST endpoints created in this plan.
