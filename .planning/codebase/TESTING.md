# Testing Patterns

**Analysis Date:** 2025-02-16

## Test Framework

**Runner:**
- Python: pytest 7.0+ with pytest-asyncio 0.21+ for async support
- TypeScript: No test framework installed (tests not implemented)
- Config: `backend/pyproject.toml` lists test dependencies under `[project.optional-dependencies]`

**Assertion Library:**
- Python: pytest's built-in assertions
- TypeScript: N/A (no testing framework)

**Run Commands:**
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests (pytest will discover test_*.py or *_test.py files)
pytest

# Run with async support
pytest -v

# Coverage (requires pytest-cov)
pytest --cov=vidpipe
```

**Status:**
- No test files found in codebase
- test framework installed as dev dependency but not actively used
- Frontend has no test framework configured

## Test File Organization

**Location:**
- Not established - no test files currently exist
- Convention when implemented should be:
  - Python: `backend/tests/` directory parallel to `vidpipe/` source
  - TypeScript: Co-located with components (e.g., `Button.test.tsx` next to `Button.tsx`) OR `frontend/tests/`

**Naming:**
- Python: `test_*.py` or `*_test.py` (pytest standard)
- TypeScript: `*.test.ts`, `*.test.tsx`, `*.spec.ts` (industry standard)

**Structure:**
```
backend/tests/
├── conftest.py              # pytest fixtures and shared setup
├── test_api_routes.py       # API endpoint tests
├── test_db_models.py        # Database model tests
├── test_services/
│   ├── test_file_manager.py
│   └── test_vertex_client.py
└── test_orchestrator/
    └── test_pipeline.py
```

## Test Structure

**Suite Organization:**

For Python tests (recommended pattern based on project structure):

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.mark.asyncio
class TestGenerateEndpoint:
    """Test suite for POST /api/generate endpoint."""

    async def test_generate_video_success(self, async_session: AsyncSession):
        """Test successful video generation request."""
        # Arrange
        request_body = {
            "prompt": "A dog running",
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "clip_duration": 6,
            "total_duration": 15,
            "text_model": "gemini-2.5-flash",
            "image_model": "imagen-3.0-generate-002",
            "video_model": "veo-3.1-fast-generate-001",
            "enable_audio": True,
        }

        # Act
        response = await client.post("/api/generate", json=request_body)

        # Assert
        assert response.status_code == 200
        assert "project_id" in response.json()
```

**Patterns:**

- **Setup/Teardown:** Use pytest fixtures for database setup/teardown
- **Async Testing:** Mark tests with `@pytest.mark.asyncio`
- **AAA Pattern:** Arrange (setup), Act (execute), Assert (verify)
- **Mocking:** Use `unittest.mock` or `pytest-mock` for external dependencies

## Mocking

**Framework:**
- Python: `unittest.mock` (built-in) or `pytest-mock`
- Mock external API calls, database operations, file I/O

**Patterns:**

Recommended approach for mocking Google Genai calls:

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_storyboard_generation_with_mock():
    """Mock the Genai client to test pipeline without real API calls."""
    with patch('vidpipe.pipeline.storyboard.genai.generate_content') as mock_genai:
        # Configure mock to return test storyboard
        mock_genai.return_value = AsyncMock()
        mock_genai.return_value.text = '{"scenes": [...]}'

        # Execute pipeline step
        result = await generate_storyboard(...)

        # Verify mock was called
        mock_genai.assert_called_once()
```

**What to Mock:**
- External API calls (Google Genai, Vertex AI operations)
- File I/O (disk writes, reads)
- Database operations (when testing business logic in isolation)
- Time/timers (sleep, intervals)

**What NOT to Mock:**
- FastAPI request/response handling (use TestClient instead)
- Pydantic model validation (test real validation)
- SQLAlchemy ORM behavior (test with in-memory SQLite if needed)
- Core business logic calculations (e.g., cost estimation)

## Fixtures and Factories

**Test Data:**

Pattern for creating test projects:

```python
import pytest
from vidpipe.db.models import Project

@pytest.fixture
async def test_project(async_session: AsyncSession):
    """Create a test project in the database."""
    project = Project(
        prompt="Test video generation",
        style="cinematic",
        aspect_ratio="16:9",
        target_clip_duration=6,
        target_scene_count=3,
        status="pending",
    )
    async_session.add(project)
    await async_session.commit()
    return project

@pytest.fixture
async def test_project_with_scenes(test_project: Project, async_session: AsyncSession):
    """Create a project with pre-populated scenes."""
    # Create scenes for the project
    # Return enhanced project
    return test_project
```

**Location:**
- Python: `backend/tests/conftest.py` (pytest auto-discovers and loads fixtures)
- TypeScript: Not established; follow testing framework convention if/when added

## Coverage

**Requirements:**
- No coverage thresholds enforced currently
- Infrastructure in place: pytest-cov can be added
- No CI/CD checks enforcing coverage

**View Coverage:**
```bash
# Install coverage plugin
pip install pytest-cov

# Generate coverage report
pytest --cov=vidpipe --cov-report=html

# View in browser
open htmlcov/index.html
```

## Test Types

**Unit Tests:**
- Scope: Individual functions, models, utilities
- Approach: Fast, isolated, mocked dependencies
- Example: Test `estimateCost()` function with various inputs
- Target location: `backend/tests/test_lib/` or alongside utility modules

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Real database (SQLite), mocked external APIs
- Example: Test full /api/generate → project creation → pipeline start flow
- Target location: `backend/tests/integration/`

**E2E Tests:**
- Framework: Not currently implemented
- Recommendation: Use pytest with TestClient + mocked Google APIs for API E2E
- Frontend E2E: Would require Playwright/Cypress (not configured)
- Scope would cover: Generate request → status polling → completion

## Common Patterns

**Async Testing:**

```python
@pytest.mark.asyncio
async def test_async_database_operation():
    """Test async database operations."""
    async with async_session() as session:
        # Create and test
        result = await session.execute(select(Project).limit(1))
        projects = result.scalars().all()
        assert len(projects) >= 0
```

**Error Testing:**

```python
@pytest.mark.asyncio
async def test_generate_with_invalid_model():
    """Test that invalid model ID raises validation error."""
    invalid_request = {
        "prompt": "Test",
        "video_model": "invalid-model-999",
        # ... other fields
    }

    with pytest.raises(HTTPException) as exc_info:
        # Would test validation at route level
        pass

    assert exc_info.value.status_code == 422
```

**Database Transaction Testing:**

```python
@pytest.fixture
async def db_session_for_test(async_session_maker):
    """Provide a session that rolls back after each test."""
    async with async_session_maker() as session:
        async with session.begin():
            yield session
            # Automatic rollback on context exit
```

## Testing Async Code

**Async Patterns in Codebase:**
- All database operations use SQLAlchemy AsyncSession
- All API routes are `async def`
- Hooks use `useEffect` (React) which handles async implicitly

**Testing Strategy:**
- Mark tests with `@pytest.mark.asyncio`
- Use AsyncSession fixtures from conftest
- Test real async paths, not wrapped sync versions

Example from project:
```python
async def test_get_project_status():
    """Test polling endpoint returns current status."""
    project = await create_test_project(session)

    response = await client.get(f"/api/projects/{project.id}/status")

    assert response.status_code == 200
    data = response.json()
    assert data["project_id"] == str(project.id)
    assert data["status"] in ["pending", "storyboarding", "complete"]
```

## Frontend Testing (Not Yet Implemented)

**When implemented, recommended approach:**
- Framework: Vitest (already uses Vite)
- Component testing: React Testing Library
- Location: Co-locate with components or `frontend/tests/`
- Example patterns:

```typescript
// Button.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "./Button";

describe("Button", () => {
  it("renders with text", () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText("Click me")).toBeInTheDocument();
  });

  it("calls onClick when clicked", async () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);

    await userEvent.click(screen.getByText("Click"));
    expect(handleClick).toHaveBeenCalled();
  });
});
```

---

*Testing analysis: 2025-02-16*

## Notes

- **Current State:** Backend has pytest setup but no tests written. Frontend has no test framework.
- **Technical Debt:** No test coverage means refactoring risk is high; core pipeline logic should be tested before major changes.
- **Recommendation:** Start with unit tests for core utilities (cost estimation, state machines) and API route handlers before expanding to integration tests.
