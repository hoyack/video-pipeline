---
phase: 13-llm-provider-abstraction-ollama
plan: "01"
subsystem: backend
tags: [llm-abstraction, ollama, vertex-ai, db-schema, settings-api]
dependency_graph:
  requires: []
  provides:
    - "vidpipe.services.llm package with LLMAdapter ABC and two concrete adapters"
    - "vidpipe.schemas.llm_vision with three Pydantic vision schemas"
    - "UserSettings Ollama columns (ollama_use_cloud, ollama_api_key, ollama_endpoint, ollama_models)"
    - "Project vision_model column"
    - "Settings API GET/PUT/models with Ollama configuration support"
  affects:
    - "backend/vidpipe/api/routes.py (settings + generate endpoints)"
    - "backend/vidpipe/db/models.py"
    - "backend/vidpipe/db/__init__.py"
tech_stack:
  added:
    - "ollama==0.6.1 (AsyncClient for structured Ollama chat)"
  patterns:
    - "ABC-based adapter pattern with get_adapter() registry for provider routing"
    - "tenacity retry with exponential backoff on all LLM calls"
    - "TYPE_CHECKING import to avoid circular deps in registry.py"
    - "has_X_key pattern (never expose secret, return bool presence flag)"
key_files:
  created:
    - backend/vidpipe/services/llm/__init__.py
    - backend/vidpipe/services/llm/base.py
    - backend/vidpipe/services/llm/vertex_adapter.py
    - backend/vidpipe/services/llm/ollama_adapter.py
    - backend/vidpipe/services/llm/registry.py
    - backend/vidpipe/schemas/llm_vision.py
  modified:
    - backend/vidpipe/db/models.py
    - backend/vidpipe/db/__init__.py
    - backend/vidpipe/api/routes.py
decisions:
  - "LLMAdapter ABC uses async abstract methods — consistent with existing async pipeline pattern"
  - "OllamaAdapter always passes stream=False — ollama library defaults to True (async generator)"
  - "Ollama model ID prefix ollama/ stripped before client.chat() — library uses bare names"
  - "TYPE_CHECKING import for UserSettings in registry.py — avoids circular import via db.models"
  - "ollama_use_cloud flag: True = use cloud endpoint + api_key, False = localhost or custom endpoint with no key"
  - "vision_model accepted in GenerateRequest alongside text_model for per-project LLM routing"
  - "GenerateRequest validation: ollama/ prefix accepted for text_model and vision_model"
metrics:
  duration: "5 min"
  completed: "2026-02-19"
  tasks_completed: 2
  files_changed: 9
---

# Phase 13 Plan 01: LLM Provider Abstraction + DB Schema Summary

**One-liner:** LLM adapter ABC with VertexAI + Ollama implementations, provider registry, vision Pydantic schemas, and DB/API extensions for Ollama configuration.

## What Was Built

### Task 1: LLM Adapter Package + Vision Schemas (commit 8c2dc85)

Created `backend/vidpipe/services/llm/` package:

**base.py** — `LLMAdapter` ABC with two async abstract methods:
- `generate_text(prompt, schema, *, temperature, system_prompt, max_retries) -> BaseModel`
- `analyze_image(image_bytes, prompt, schema, *, mime_type, temperature, max_retries) -> BaseModel`

**vertex_adapter.py** — `VertexAIAdapter(LLMAdapter)`:
- Uses `get_vertex_client(location=location_for_model(model_id))` for location-aware routing
- Structured output via `response_mime_type="application/json"` + `response_schema=schema`
- tenacity retry with exponential backoff (2-30s)

**ollama_adapter.py** — `OllamaAdapter(LLMAdapter)`:
- Strips `ollama/` prefix before passing to `ollama.AsyncClient.chat()`
- Always passes `stream=False` to avoid async generator returns
- Vision: base64-encodes image bytes in user message `"images"` field
- tenacity retry on all failures

**registry.py** — `get_adapter()`:
- `ollama/*` → OllamaAdapter (cloud or localhost based on `user_settings.ollama_use_cloud`)
- `gemini-*` and everything else → VertexAIAdapter
- TYPE_CHECKING import for UserSettings avoids circular imports

**schemas/llm_vision.py** — Three Pydantic models:
- `ReversePromptOutput`: reverse_prompt, visual_description, quality_score, suggested_name
- `SemanticAnalysisOutput`: manifest_adherence, visual_quality, continuity_issues, new_entities_description, overall_scene_description
- `VisualPromptScoreOutput`: visual_quality, prompt_adherence

### Task 2: DB Schema Extensions + Settings API (commit 78f1899)

**db/models.py:**
- `UserSettings`: 4 new Ollama columns (`ollama_use_cloud`, `ollama_api_key`, `ollama_endpoint`, `ollama_models`)
- `Project`: `vision_model` column

**db/__init__.py:**
- 5 idempotent ALTER TABLE migrations for Phase 13 columns

**api/routes.py:**
- `UserSettingsResponse`: added `ollama_use_cloud`, `has_ollama_key`, `ollama_endpoint`, `ollama_models`
- `UserSettingsUpdate`: added `ollama_use_cloud`, `ollama_api_key`, `clear_ollama_key`, `ollama_endpoint`, `ollama_models`
- `EnabledModelsResponse`: added `ollama_models`
- `GenerateRequest`: added `vision_model: Optional[str]`
- `ProjectDetail` / `ProjectListItem`: added `vision_model: Optional[str]`
- Validation: `ollama/` prefix accepted for `text_model` and `vision_model`

## Verification Results

All checks passed:
1. `from vidpipe.services.llm import get_adapter, LLMAdapter` — OK
2. Three Pydantic vision schemas importable from `vidpipe.schemas.llm_vision` — OK
3. DB migrations run idempotently on existing database — OK
4. Settings API GET/PUT/models round-trips Ollama configuration — OK
5. `get_adapter("gemini-2.5-flash")` returns `VertexAIAdapter` — OK
6. `get_adapter("ollama/llama3.1")` returns `OllamaAdapter` — OK

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

Created files:
- [x] backend/vidpipe/services/llm/__init__.py
- [x] backend/vidpipe/services/llm/base.py
- [x] backend/vidpipe/services/llm/vertex_adapter.py
- [x] backend/vidpipe/services/llm/ollama_adapter.py
- [x] backend/vidpipe/services/llm/registry.py
- [x] backend/vidpipe/schemas/llm_vision.py

Commits:
- [x] 8c2dc85 — Task 1 (LLM adapter package)
- [x] 78f1899 — Task 2 (DB schema + settings API)

## Self-Check: PASSED
