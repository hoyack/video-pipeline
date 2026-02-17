# Phase 4: Manifest System Foundation - Research

**Researched:** 2026-02-16
**Domain:** Database schema design, API architecture, React UI components for asset management
**Confidence:** HIGH

## Summary

Phase 4 implements the Manifest System Foundation as defined in V2 architecture docs. This phase transforms manifests from per-project artifacts into standalone, reusable entities with full CRUD lifecycle. The implementation requires three parallel tracks: (1) database schema additions for `manifests`, `assets`, and relationship tracking, (2) FastAPI REST endpoints for manifest management, and (3) React components for Manifest Library browsing and Manifest Creator Stage 1 (upload + tag, no processing).

The architecture draws heavily from existing patterns: SQLAlchemy 2.0 Mapped[Type] annotations (Phase 1), FastAPI APIRouter with /api prefix (Phase 3), and async database sessions. The critical architectural decision is the manifest-to-project relationship: manifests are standalone entities that projects reference (many-to-one), enabling reuse across unlimited projects without retroactive updates.

**Primary recommendation:** Implement database schema first, then API layer, then UI components. Follow existing project fork patterns for relationship copying/inheritance logic.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.0+ | ORM with async support | Already used, Mapped[Type] annotations for type safety |
| FastAPI | 0.115.0+ | REST API framework | Existing API layer, async-first design |
| Pydantic | 2.0+ | Data validation and serialization | Used throughout for request/response schemas |
| React | 18+ | Frontend UI framework | Existing frontend stack |
| TypeScript | 5+ | Type-safe frontend | Existing frontend language |
| Tailwind CSS | 3+ | Utility-first CSS | Existing styling approach |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiosqlite | 0.22.1+ | Async SQLite driver | Already in requirements.txt for async sessions |
| uuid | stdlib | Unique identifiers for manifests/assets | Standard for all entity IDs in codebase |
| datetime | stdlib | Timestamps for created_at/updated_at | Consistent with existing models |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SQLite | PostgreSQL | SQLite chosen in Phase 1 for local-first tool, sufficient for manifest storage |
| REST API | GraphQL | REST chosen for simplicity, manifest CRUD fits REST patterns well |
| Pydantic v2 | Pydantic v1 | V2 already adopted, performance benefits for JSON serialization |

**Installation:**
All dependencies already present in backend/requirements.txt. No new packages required for Phase 4.

## Architecture Patterns

### Recommended Project Structure
```
backend/vidpipe/
├── db/
│   └── models.py          # Add Manifest, Asset models
├── api/
│   ├── routes.py          # Add /api/manifests endpoints
│   └── schemas/           # NEW: manifest_schemas.py
├── services/
│   └── manifest_service.py  # NEW: Business logic layer
└── schemas/
    └── manifest.py        # NEW: Pydantic validation schemas

frontend/src/components/
├── ManifestLibrary.tsx    # NEW: Grid view with filters/sort
├── ManifestCard.tsx       # NEW: Card component for library
├── ManifestCreator.tsx    # NEW: Stage 1 upload + tag UI
└── AssetUploader.tsx      # NEW: Drag-drop multi-file upload
```

### Pattern 1: SQLAlchemy 2.0 Mapped Annotations
**What:** Type-safe ORM model definitions using `Mapped[Type]` annotations
**When to use:** All new database models
**Example:**
```python
# Source: Existing backend/vidpipe/db/models.py pattern
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, JSON, Integer, ForeignKey
import uuid
from datetime import datetime

class Manifest(Base):
    __tablename__ = "manifests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(50), default="CUSTOM")
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="DRAFT")
    asset_count: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )
```

### Pattern 2: FastAPI APIRouter with Async Sessions
**What:** Route handlers that create fresh async sessions per request
**When to use:** All new API endpoints
**Example:**
```python
# Source: Existing backend/vidpipe/api/routes.py pattern
from fastapi import APIRouter, HTTPException
from vidpipe.db import async_session
from sqlalchemy import select

router = APIRouter(prefix="/api")

@router.get("/manifests", response_model=list[ManifestListItem])
async def list_manifests():
    """List all manifests ordered by last_used_at."""
    async with async_session() as session:
        result = await session.execute(
            select(Manifest).order_by(Manifest.updated_at.desc())
        )
        manifests = result.scalars().all()
        return [ManifestListItem.model_validate(m) for m in manifests]
```

### Pattern 3: Pydantic Request/Response Schemas
**What:** Separate Pydantic models for API contract validation
**When to use:** All API endpoints for type safety and documentation
**Example:**
```python
# Source: Existing backend/vidpipe/api/routes.py schemas
from pydantic import BaseModel, Field
from typing import Optional

class CreateManifestRequest(BaseModel):
    name: str
    description: Optional[str] = None
    category: str = "CUSTOM"
    tags: Optional[list[str]] = None

class ManifestResponse(BaseModel):
    manifest_id: str
    name: str
    description: Optional[str]
    category: str
    status: str
    asset_count: int
    created_at: str
    updated_at: str
```

### Pattern 4: React Component Composition
**What:** Small, focused components composed into views
**When to use:** All new UI features
**Example:**
```tsx
// Source: Existing frontend patterns (ProjectList.tsx, ProjectDetail.tsx)
interface ManifestCardProps {
  manifest: Manifest;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
}

export function ManifestCard({ manifest, onEdit, onDelete, onDuplicate }: ManifestCardProps) {
  return (
    <div className="border rounded-lg p-4 shadow-sm hover:shadow-md transition">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-lg">{manifest.name}</h3>
        <StatusBadge status={manifest.status} />
      </div>
      <p className="text-sm text-gray-600 mb-4">{manifest.description}</p>
      <div className="flex gap-2">
        <button onClick={() => onEdit(manifest.id)} className="btn-primary">
          Edit
        </button>
        <button onClick={() => onDuplicate(manifest.id)} className="btn-secondary">
          Duplicate
        </button>
      </div>
    </div>
  );
}
```

### Anti-Patterns to Avoid
- **Shared async sessions across requests:** Each API handler must create its own session. Never share sessions between background tasks and request handlers.
- **Direct database queries in React components:** Always use API endpoints, never expose database connection strings to frontend.
- **Mutable default arguments in Pydantic:** Use `Field(default_factory=list)` instead of `tags: list[str] = []`
- **Missing foreign key indexes:** Always add indexes on foreign key columns for query performance.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image upload handling | Custom multipart parser | FastAPI `UploadFile` + `File()` | Handles streaming, validation, temp storage automatically |
| File drag-drop UI | Custom DOM event handlers | react-dropzone or HTML5 native | Cross-browser compatibility, mobile support, MIME validation |
| Date formatting | String manipulation | `datetime.isoformat()` (backend), `new Date().toISOString()` (frontend) | Timezone-aware, ISO 8601 standard |
| UUID generation | Random string generators | Python `uuid.uuid4()`, crypto.randomUUID() (frontend) | RFC 4122 compliant, collision-resistant |
| JSON serialization | Manual dict building | Pydantic `.model_dump()` | Type validation, optional field handling, nested models |
| Database migrations | Manual ALTER TABLE | Alembic (future) | Version control, rollback support, team collaboration |

**Key insight:** FastAPI and Pydantic handle 90% of API boilerplate. React's built-in hooks (`useState`, `useEffect`) are sufficient for manifest library state management in Stage 1 (no processing logic yet).

## Common Pitfalls

### Pitfall 1: Foreign Key Cascade Deletes
**What goes wrong:** Deleting a manifest accidentally deletes all associated assets, breaking projects that reference them
**Why it happens:** Default SQLAlchemy cascade behavior can be aggressive
**How to avoid:** Use `nullable=True` for `asset.manifest_id` and implement soft deletes with a `deleted_at` column. Warn users if manifest is referenced by active projects.
**Warning signs:** Projects fail to load after manifest deletion, orphaned asset records with NULL manifest_id

### Pitfall 2: Missing Database Indexes on Foreign Keys
**What goes wrong:** Queries like "get all assets for manifest X" become slow as asset count grows
**Why it happens:** SQLite doesn't auto-index foreign keys like PostgreSQL does
**How to avoid:** Explicitly add indexes in model definition or migration
**Warning signs:** API latency increases with asset count, EXPLAIN QUERY PLAN shows full table scans
**Prevention:**
```python
class Asset(Base):
    __tablename__ = "assets"
    manifest_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("manifests.id"),
        index=True  # CRITICAL: explicit index
    )
```

### Pitfall 3: Async/Await Inconsistency
**What goes wrong:** Mixing sync and async code causes "coroutine was never awaited" errors or blocking event loop
**Why it happens:** FastAPI supports both, easy to forget `await` keyword
**How to avoid:** Always use `async def` for route handlers that query database. Always `await` session.execute(), session.commit(), etc.
**Warning signs:** RuntimeWarning about unawaited coroutines, 500 errors with cryptic traceback
**Example:**
```python
# WRONG
@router.get("/manifests/{id}")
async def get_manifest(id: uuid.UUID):
    session = async_session()  # Missing async context manager
    result = session.execute(select(Manifest).where(Manifest.id == id))  # Missing await

# CORRECT
@router.get("/manifests/{id}")
async def get_manifest(id: uuid.UUID):
    async with async_session() as session:
        result = await session.execute(select(Manifest).where(Manifest.id == id))
```

### Pitfall 4: JSON Column Type Mismatch
**What goes wrong:** Storing Python lists/dicts in JSON columns fails with "not JSON serializable" or loads as strings
**Why it happens:** SQLAlchemy JSON type needs explicit typing and serialization
**How to avoid:** Use `Mapped[Optional[list]]` or `Mapped[Optional[dict]]` with proper JSON column type
**Warning signs:** Tags load as string `"['tag1', 'tag2']"` instead of actual list
**Example:**
```python
# WRONG
tags: Mapped[str] = mapped_column(JSON)  # Type mismatch

# CORRECT
tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
```

### Pitfall 5: Many-to-One Relationship Confusion
**What goes wrong:** Thinking one manifest belongs to one project (one-to-one) instead of many projects sharing one manifest
**Why it happens:** Traditional file-based thinking (one manifest per project)
**How to avoid:** Manifest table has NO project_id column. Project table has manifest_id column (foreign key to manifests). Assets have manifest_id (not project_id).
**Warning signs:** Attempting to filter manifests by project, schema design with manifest.project_id
**Correct schema:**
```
manifests (id, name, ...)           -- No project_id
projects (id, manifest_id, ...)     -- References manifests.id
assets (id, manifest_id, ...)       -- References manifests.id (NOT projects.id)
```

## Code Examples

Verified patterns from existing codebase and V2 docs:

### Database Model Definition
```python
# Source: backend/vidpipe/db/models.py pattern + v2-manifest.md schema
from sqlalchemy import String, Text, JSON, Integer, Float, Boolean, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
import uuid
from datetime import datetime
from typing import Optional

class Manifest(Base):
    """Manifest model representing a standalone, reusable asset collection."""
    __tablename__ = "manifests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    category: Mapped[str] = mapped_column(String(50), default="CUSTOM")
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="DRAFT")
    processing_progress: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    contact_sheet_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    asset_count: Mapped[int] = mapped_column(Integer, default=0)
    total_processing_cost: Mapped[float] = mapped_column(Float, default=0.0)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_manifest_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("manifests.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )

class Asset(Base):
    """Asset model representing tagged visual elements within a manifest."""
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    manifest_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("manifests.id"),
        index=True  # Critical for query performance
    )
    asset_type: Mapped[str] = mapped_column(String(50))  # CHARACTER, OBJECT, etc.
    name: Mapped[str] = mapped_column(Text)
    manifest_tag: Mapped[str] = mapped_column(String(50))  # CHAR_01, OBJ_02, etc.
    user_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Image references (Stage 1: uploaded, Stage 2+: processed)
    reference_image_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Metadata (Stage 1: user-provided, Stage 2+: AI-generated)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

# Add manifest_id to existing Project model
class Project(Base):
    __tablename__ = "projects"
    # ... existing fields ...
    manifest_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("manifests.id"),
        nullable=True,
        index=True
    )
    manifest_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
```

### FastAPI CRUD Endpoints
```python
# Source: backend/vidpipe/api/routes.py patterns
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

router = APIRouter(prefix="/api")

class CreateManifestRequest(BaseModel):
    name: str
    description: Optional[str] = None
    category: str = "CUSTOM"
    tags: Optional[list[str]] = None

class ManifestResponse(BaseModel):
    manifest_id: str
    name: str
    description: Optional[str]
    category: str
    status: str
    asset_count: int
    created_at: str
    updated_at: str

@router.post("/manifests", status_code=201, response_model=ManifestResponse)
async def create_manifest(request: CreateManifestRequest):
    """Create new manifest in DRAFT status."""
    async with async_session() as session:
        manifest = Manifest(
            name=request.name,
            description=request.description,
            category=request.category,
            tags=request.tags or [],
            status="DRAFT",
        )
        session.add(manifest)
        await session.commit()
        await session.refresh(manifest)

        return ManifestResponse(
            manifest_id=str(manifest.id),
            name=manifest.name,
            description=manifest.description,
            category=manifest.category,
            status=manifest.status,
            asset_count=manifest.asset_count,
            created_at=manifest.created_at.isoformat(),
            updated_at=manifest.updated_at.isoformat(),
        )

@router.get("/manifests", response_model=list[ManifestResponse])
async def list_manifests(category: Optional[str] = None):
    """List manifests with optional category filter."""
    async with async_session() as session:
        query = select(Manifest).order_by(Manifest.updated_at.desc())
        if category:
            query = query.where(Manifest.category == category)

        result = await session.execute(query)
        manifests = result.scalars().all()

        return [
            ManifestResponse(
                manifest_id=str(m.id),
                name=m.name,
                description=m.description,
                category=m.category,
                status=m.status,
                asset_count=m.asset_count,
                created_at=m.created_at.isoformat(),
                updated_at=m.updated_at.isoformat(),
            )
            for m in manifests
        ]

@router.get("/manifests/{manifest_id}", response_model=ManifestResponse)
async def get_manifest(manifest_id: uuid.UUID):
    """Get manifest details by ID."""
    async with async_session() as session:
        result = await session.execute(
            select(Manifest).where(Manifest.id == manifest_id)
        )
        manifest = result.scalar_one_or_none()

        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found")

        return ManifestResponse(
            manifest_id=str(manifest.id),
            name=manifest.name,
            description=manifest.description,
            category=manifest.category,
            status=manifest.status,
            asset_count=manifest.asset_count,
            created_at=manifest.created_at.isoformat(),
            updated_at=manifest.updated_at.isoformat(),
        )

@router.delete("/manifests/{manifest_id}")
async def delete_manifest(manifest_id: uuid.UUID):
    """Delete manifest. Returns 409 if used by active projects."""
    async with async_session() as session:
        # Check if manifest is used by any projects
        projects_result = await session.execute(
            select(Project).where(Project.manifest_id == manifest_id)
        )
        projects = projects_result.scalars().all()

        if projects:
            raise HTTPException(
                status_code=409,
                detail=f"Manifest is used by {len(projects)} project(s). Cannot delete."
            )

        # Delete manifest
        result = await session.execute(
            select(Manifest).where(Manifest.id == manifest_id)
        )
        manifest = result.scalar_one_or_none()

        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found")

        await session.delete(manifest)
        await session.commit()

        return {"status": "deleted", "manifest_id": str(manifest_id)}
```

### React Component for Manifest Library
```tsx
// Source: Frontend patterns from ProjectList.tsx, ProjectDetail.tsx
import { useState, useEffect } from 'react';

interface Manifest {
  manifest_id: string;
  name: string;
  description: string | null;
  category: string;
  status: string;
  asset_count: number;
  created_at: string;
  updated_at: string;
}

export function ManifestLibrary() {
  const [manifests, setManifests] = useState<Manifest[]>([]);
  const [loading, setLoading] = useState(true);
  const [categoryFilter, setCategoryFilter] = useState<string>('ALL');

  useEffect(() => {
    async function loadManifests() {
      try {
        const url = categoryFilter === 'ALL'
          ? '/api/manifests'
          : `/api/manifests?category=${categoryFilter}`;
        const response = await fetch(url);
        const data = await response.json();
        setManifests(data);
      } catch (error) {
        console.error('Failed to load manifests:', error);
      } finally {
        setLoading(false);
      }
    }
    loadManifests();
  }, [categoryFilter]);

  if (loading) {
    return <div className="p-8 text-center">Loading manifests...</div>;
  }

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Manifest Library</h1>
        <button className="btn-primary">+ New Manifest</button>
      </div>

      <div className="flex gap-2 mb-6">
        {['ALL', 'CHARACTERS', 'ENVIRONMENT', 'FULL_PRODUCTION'].map(cat => (
          <button
            key={cat}
            onClick={() => setCategoryFilter(cat)}
            className={`px-4 py-2 rounded ${
              categoryFilter === cat ? 'bg-blue-600 text-white' : 'bg-gray-200'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {manifests.map(manifest => (
          <div key={manifest.manifest_id} className="border rounded-lg p-4 shadow-sm">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-semibold text-lg">{manifest.name}</h3>
              <span className={`px-2 py-1 rounded text-xs ${
                manifest.status === 'READY' ? 'bg-green-100 text-green-800' : 'bg-gray-100'
              }`}>
                {manifest.status}
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-2">{manifest.description}</p>
            <p className="text-xs text-gray-500 mb-4">{manifest.asset_count} assets</p>
            <div className="flex gap-2">
              <button className="text-sm text-blue-600 hover:underline">Edit</button>
              <button className="text-sm text-blue-600 hover:underline">Duplicate</button>
              <button className="text-sm text-red-600 hover:underline">Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manifests generated per-project | Manifests as standalone reusable entities | V2 architecture (2026-02) | Enables asset reuse across projects, reduces processing cost/time |
| Assets scoped to projects | Assets belong to manifests | V2 architecture (2026-02) | Decouples asset management from video generation |
| No asset tagging | Manifest tags (CHAR_01, OBJ_02) | V2 architecture (2026-02) | Machine-readable asset references in scene manifests |
| Inline reference uploads | Manifest Creator workspace | V2 architecture (2026-02) | Dedicated asset refinement before generation |

**Deprecated/outdated:**
- **Per-project asset pools:** V1 architecture treated assets as project-scoped. V2 manifests are project-independent.
- **No asset versioning:** V1 had no manifest version tracking. V2 supports versioning and parent_manifest_id for duplication history.

## Open Questions

1. **Image upload storage strategy for Stage 1**
   - What we know: Need to store uploaded images before processing in Stage 2
   - What's unclear: Use GCS immediately or local temp storage until manifest finalized?
   - Recommendation: Upload to GCS immediately with manifest_id prefix (e.g., `manifests/{manifest_id}/uploads/{filename}`). Simplifies Stage 2 processing, no local cleanup needed.

2. **Tag validation and naming conventions**
   - What we know: User provides name and type (CHARACTER, OBJECT, etc.)
   - What's unclear: Should Stage 1 auto-generate manifest_tag (CHAR_01) or wait for Stage 2 processing?
   - Recommendation: Auto-generate on save in Stage 1. Simple sequential numbering by type. Makes manifest immediately readable even without processing.

3. **Contact sheet generation timing**
   - What we know: Contact sheet shows visual grid of all uploaded assets
   - What's unclear: Generate in Stage 1 on upload, or Stage 2 during processing?
   - Recommendation: Generate in Stage 1 using Pillow (already in requirements.txt). Gives immediate visual feedback, no API cost.

4. **Manifest deletion vs soft delete**
   - What we know: Deleting manifest should not break existing projects
   - What's unclear: Hard delete with foreign key checks, or soft delete with deleted_at flag?
   - Recommendation: Soft delete with deleted_at column. Prevents data loss, allows "Restore" feature, projects continue working with deleted manifests.

5. **Asset count updates**
   - What we know: Manifest.asset_count tracks total assets
   - What's unclear: Updated via database trigger, or application code on asset create/delete?
   - Recommendation: Application code (Python). SQLite triggers are complex, async sessions may not fire them correctly. Update in same transaction as asset changes.

## Sources

### Primary (HIGH confidence)
- `/home/ubuntu/work/video-pipeline/docs/v2-manifest.md` - Manifest entity schema, relationship rules, Manifest Creator workflow stages
- `/home/ubuntu/work/video-pipeline/docs/v2-pipe-optimization.md` - Complete V2 architecture with asset registry, scene manifests
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/db/models.py` - Existing SQLAlchemy 2.0 patterns, Mapped[Type] annotations
- `/home/ubuntu/work/video-pipeline/backend/vidpipe/api/routes.py` - Existing FastAPI patterns, async session usage, CRUD endpoints
- `/home/ubuntu/work/video-pipeline/backend/requirements.txt` - Current dependency versions

### Secondary (MEDIUM confidence)
- Existing frontend components (ProjectList.tsx, ProjectDetail.tsx) - React patterns, Tailwind styling conventions
- SQLAlchemy 2.0 documentation - Async ORM patterns, relationship definitions

### Tertiary (LOW confidence)
- None required for Phase 4 foundation. All patterns derived from existing codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies already in use, no new packages required
- Architecture: HIGH - Follows existing patterns (SQLAlchemy Mapped, FastAPI async routes, React hooks)
- Pitfalls: HIGH - Derived from existing codebase patterns and common SQLAlchemy/FastAPI gotchas

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (30 days - stable stack, V2 architecture finalized)
