# Coding Conventions

**Analysis Date:** 2025-02-16

## Naming Patterns

**Files:**
- React components: PascalCase (e.g., `ProjectList.tsx`, `StatusBadge.tsx`)
- Non-component TypeScript: camelCase (e.g., `client.ts`, `constants.ts`)
- Python modules: snake_case (e.g., `file_manager.py`, `vertex_client.py`)
- Python classes: PascalCase (e.g., `FileManager`, `Project`, `ApiError`)

**Functions:**
- TypeScript: camelCase (e.g., `generateVideo`, `getProjectStatus`, `formatCost`)
- Python: snake_case (e.g., `validate_dependencies`, `configure_sqlite_pragmas`, `get_project_dir`)
- React hooks: camelCase with `use` prefix (e.g., `useProjectStatus`, `usePolling`)

**Variables:**
- TypeScript: camelCase (e.g., `clipDuration`, `selectedVideoModel`, `isTerminal`)
- Python: snake_case (e.g., `project_id`, `clip_duration`, `async_session`)
- Constants: UPPER_SNAKE_CASE (e.g., `TERMINAL_STATUSES`, `FAST_INTERVAL`, `TOTAL_DURATION_MAX`)
- React component props: camelCase (e.g., `onSelectProject`, `projectId`, `onGenerated`)

**Types/Interfaces:**
- React/TypeScript interfaces: PascalCase (e.g., `GenerateRequest`, `StatusResponse`, `ProjectListItem`)
- Pydantic models: PascalCase (e.g., `GenerateRequest`, `ProjectDetail`, `VideoClip`)
- TypeScript type aliases: PascalCase (e.g., `View`, `PipelineStage`)
- Record/map types: PascalCase with clear domain (e.g., `STATUS_COLORS`, `ALLOWED_MODELS`)

## Code Style

**Formatting:**
- TypeScript: Handled by ESLint 9.39.1 with flat config
- Python: Black/Ruff via pyproject.toml dev dependencies
- Frontend: Vite-native formatting (no explicit Prettier config)

**Linting:**
- TypeScript: ESLint with @eslint/js, typescript-eslint, react-hooks, react-refresh plugins
- Config location: `frontend/eslint.config.js` (flat config format, not .eslintrc)
- Severity: Recommended rules from typescript-eslint and react-refresh
- Python: Black + Ruff configured in `backend/pyproject.toml`

**Line Length:**
- TypeScript: No explicit max-line-length rule enforced (ESLint defaults)
- Python: Conventional 88 chars for Black

## Import Organization

**TypeScript Order:**
1. React/React-DOM imports (from "react")
2. Third-party libraries (clsx, ui components)
3. Local API imports (relative, with .ts/.tsx extensions explicit)
4. Local hooks (relative imports)
5. Local constants (relative imports)
6. Local types (import type { ... })

**Example:**
```typescript
import { useState } from "react";
import clsx from "clsx";
import { generateVideo } from "../api/client.ts";
import type { StatusResponse } from "../api/types.ts";
import { usePolling } from "./usePolling.ts";
import { TERMINAL_STATUSES } from "../lib/constants.ts";
```

**Python Order:**
1. Standard library (logging, uuid, pathlib, etc.)
2. Third-party (fastapi, sqlalchemy, pydantic, google.genai)
3. Local imports (vidpipe.* modules)
4. Conditional/type-only imports

**Example:**
```python
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from pydantic import BaseModel

from vidpipe.db.models import Project
from vidpipe.services.file_manager import FileManager
```

**Path Aliases:**
- TypeScript: None configured in tsconfig.app.json (all relative imports)
- Python: Package is installed as `vidpipe`, imports use full paths from package root

**Extension Requirements:**
- TypeScript imports explicitly include `.ts` or `.tsx` extensions (enforced by tsconfig `allowImportingTsExtensions`)
- Allows tree-shaking and clarity without build-time path resolution

## Error Handling

**TypeScript Patterns:**
- Try-catch with error instanceof checks: `err instanceof Error ? err.message : "default message"`
- Custom error classes that extend Error (e.g., `ApiError` in `client.ts` extends Error and sets `this.name`)
- HTTP errors converted to ApiError with status code and message from response body
- Async/await with try-finally for cleanup (e.g., ProjectList.tsx sets `finally { setLoading(false) }`)

**React Error Boundaries:**
- Not currently implemented; errors surface to console and UI error states

**Python Patterns:**
- FastAPI HTTPException with status_code and detail (e.g., `raise HTTPException(status_code=404, detail="Not found")`)
- Custom exceptions that extend Exception (e.g., `PipelineStopped` in orchestrator)
- Logging errors with logger.error() including context (file, operation, exception type)
- Generic exception handler at app level that logs and returns JSON response

**Example (Python):**
```python
try:
    result = await session.execute(query)
except Exception as err:
    logger.error(f"Query failed: {err}")
    raise HTTPException(status_code=500, detail=str(err))
```

**Example (TypeScript):**
```typescript
try {
  const data = await getProjectStatus(projectId);
  setStatus(data);
  setError(null);
} catch (err) {
  setError(err instanceof Error ? err.message : "Polling failed");
}
```

## Logging

**Framework:**
- TypeScript: console.log/error via standard browser/Node APIs (no logging library)
- Python: `logging` module with `__name__` loggers

**Patterns:**
- Python: `logger = logging.getLogger(__name__)` at module level
- Python: Log at info for startup/shutdown, error for failures
- TypeScript: Errors logged to console only on catch in components
- Python: Structured logging with context (e.g., `f"Project {project.id}: expanding storyboard"`)

**When to Log:**
- Python: Application startup/shutdown, database operations starting, errors, significant state changes
- TypeScript: Not heavily logged in UI; errors captured in state
- Avoid: Logging every render cycle or API call in React

## Comments

**When to Comment:**
- Public function docstrings explaining purpose, args, returns, raises
- Complex business logic (cost estimation with multiple steps)
- Non-obvious design decisions (e.g., why expire_on_commit=False is needed)
- Pitfall references to known issues documented elsewhere (e.g., "Pitfall 5" path traversal)

**JSDoc/TSDoc:**
- Not consistently used in TypeScript components (JSDoc comments on utility functions)
- Python docstrings use triple-quote format with Args/Returns/Raises sections
- Example: `FileManager.get_project_dir()` has full docstring with ValueError raises

**Example (Python):**
```python
def get_project_dir(self, project_id: uuid.UUID) -> Path:
    """Get or create project directory with subdirectories.

    Creates:
    - {base_dir}/{project_id}/
    - {base_dir}/{project_id}/keyframes/
    - ...

    Args:
        project_id: UUID of the project

    Returns:
        Resolved Path to project directory

    Raises:
        ValueError: If project_id creates path outside base_dir (traversal attack)
    """
```

## Function Design

**Size:**
- TypeScript: 20-80 lines typical (e.g., component renders, hooks)
- Python: 30-100 lines typical (endpoint handlers, pipeline steps)
- Small focused functions preferred (e.g., `_estimate_project_cost`, `formatCost` are 40-70 lines)

**Parameters:**
- TypeScript: 2-4 params typical; use destructuring for props/options
- Python: 2-5 params typical; use type hints on all parameters
- Avoid positional args in public APIs; use keyword args or BaseModel schemas

**Return Values:**
- TypeScript: Return Promises from async functions; return early for guard clauses
- Python: Return typed values or None; async functions return awaitable
- React hooks return object/tuple of related state (e.g., `{ status, error, isTerminal }`)

## Module Design

**Exports:**
- TypeScript: Named exports preferred; default export only for main App/entry
- Utilities and constants exported as `export function`, `export const`, `export interface`
- Re-export types from `client.ts` and `types.ts` cleanly
- Python: Functions/classes defined at module level; import with full path

**Barrel Files:**
- TypeScript: Uses `index.ts` or direct imports (no barrel file pattern observed)
- Python: Modules use `__init__.py` for package initialization but not barrel pattern

**Directory-Level Conventions:**
- `api/`: API client functions (`client.ts`) and API contract types (`types.ts`)
- `hooks/`: React custom hooks with "use" prefix
- `components/`: React components (each in own file, PascalCase)
- `lib/`: Constants, utilities, helpers (non-component)
- Python `db/`: Database engine, models, migrations
- Python `api/`: FastAPI routes, request/response schemas, handler functions
- Python `services/`: Client wrappers (Vertex AI, File Manager), not business logic

## TypeScript Specific

**Strict Mode:** Enabled (tsconfig.app.json)
- `strict: true` enforces null/undefined checks
- `noUnusedLocals: true`, `noUnusedParameters: true` enforced
- `noFallthroughCasesInSwitch: true` required

**Type Imports:** Use `import type { ... }` for types to enable tree-shaking
- Example: `import type { StatusResponse } from "../api/types.ts"`
- Data imports use regular `import` (e.g., `import { getProjectStatus } from "..."`)

**Null Handling:**
- Explicit `null` checks in conditions (e.g., `if (projectId === null) return`)
- Optional properties marked with `?` in interfaces
- Nullish coalescing for defaults: `status ?? "unknown"`

## Python Specific

**Async Patterns:**
- Use `async def` for I/O operations (database, HTTP)
- AsyncSession from SQLAlchemy for all database ops
- Avoid blocking calls in async context

**Type Hints:**
- All function signatures include parameter types and return types
- Use `Optional[T]` for nullable, `Union[A, B]` for alternatives
- Generic types (e.g., `dict[str, float]`) preferred over `Dict`
- Avoid bare `Any` except where truly necessary

**Pydantic:**
- BaseModel for request/response validation
- Field() for detailed field documentation
- field_validator for custom validation logic
- Use frozen=True for immutable data classes where appropriate

---

*Convention analysis: 2025-02-16*
