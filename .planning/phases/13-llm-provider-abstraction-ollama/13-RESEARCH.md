# Phase 13: LLM Provider Abstraction & Ollama Integration - Research

**Researched:** 2026-02-19
**Domain:** Python adapter pattern for multi-provider LLM abstraction; Ollama REST API; provider registry; FastAPI settings extension; React settings UI
**Confidence:** HIGH (Ollama API), HIGH (adapter pattern), MEDIUM (Ollama cloud nuances)

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LLMA-01 | Abstract base LLMAdapter class with generate_text() and analyze_image() | ABC pattern via `abc.ABC + @abstractmethod`; async signatures documented |
| LLMA-02 | VertexAIAdapter wrapping existing Gemini calls across all five call sites | All five sites identified and their exact call signatures catalogued |
| LLMA-03 | OllamaAdapter with text + vision using Ollama REST API | Ollama `ollama` Python library v0.6.1 confirms AsyncClient, format=schema, images |
| LLMA-04 | Settings UI: Ollama API key, cloud/local toggle, endpoint URL override | Cloud=`https://ollama.com` with Bearer token; Local=`http://localhost:11434`, no auth |
| LLMA-05 | Model management: add/remove/toggle Ollama model names, appear in dropdowns | Requires DB column for custom Ollama model list; frontend dynamic model list pattern documented |
| LLMA-06 | GenerateForm: text_model and vision_model dropdowns; routing per call type | GenerateForm already has text_model selection; add vision_model dropdown with same pattern |
| LLMA-07 | Provider routing: model ID prefix → adapter; extensible registry | Dict-based registry with prefix matching; Gemini models → VertexAIAdapter; ollama/ prefix → OllamaAdapter |
</phase_requirements>

---

## Summary

This phase adds a provider abstraction layer over all LLM text and vision calls in the video pipeline. Currently, five call sites (storyboard, prompt rewriter, reverse prompting, CV semantic analysis, candidate scoring) all call Gemini directly via the `google-genai` SDK. The refactor introduces an abstract `LLMAdapter` base class with two async methods: `generate_text()` for structured text output and `analyze_image()` for vision + structured output. Two concrete adapters implement it: `VertexAIAdapter` (wrapping existing calls) and `OllamaAdapter` (new, using the `ollama` Python library v0.6.1).

The Ollama integration distinguishes local mode (no auth, `http://localhost:11434`) from cloud mode (`https://ollama.com` with a Bearer API key). Both use the same `ollama.AsyncClient` API, with the host and headers differing. Structured outputs are supported on both via the `format=<json-schema-dict>` parameter on `AsyncClient.chat()` — confirmed by Ollama's official structured outputs documentation. Vision models accept base64-encoded images or file paths via the `images` field on messages.

The provider registry is a simple dict mapping model ID prefixes to adapter factory functions. Ollama models are identified by a user-defined convention (e.g. `ollama/llama3.2-vision`) — the planner must decide on prefix convention. The frontend adds an Ollama settings section (cloud/local toggle, API key, endpoint URL) and model management UI (add/remove model names that appear in GenerateForm text/vision dropdowns). The existing DB `UserSettings` model needs three new columns for Ollama config and a JSON column for the custom Ollama model list. The Project model already has `text_model`; it needs a `vision_model` column.

**Primary recommendation:** Use the official `ollama` Python library (v0.6.1) with `AsyncClient` for the OllamaAdapter. Do NOT hand-roll raw httpx calls against the Ollama REST API — the library handles host normalization, auth headers, base64 image encoding, and streaming automatically.

---

## Standard Stack

### Core (backend)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `ollama` | 0.6.1 | Official Ollama Python client — AsyncClient + structured output + vision | Official library; handles auth headers, host, base64 images automatically |
| `abc` | stdlib | Abstract base class for LLMAdapter | Python stdlib; no added dependency |
| `tenacity` | already installed | Retry logic on LLM calls | Already used throughout codebase |
| `pydantic` | already installed | Schema extraction via `model_json_schema()` | Already used; works for both Gemini and Ollama schemas |
| `httpx` | already installed | Underlying transport for ollama library | Already present |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `google-genai` | already installed | Existing Gemini/VertexAI calls | Wrapped inside VertexAIAdapter; not directly called from pipeline |
| `sqlalchemy[asyncio]` | already installed | New DB columns for Ollama settings | Already present |
| `alembic` | check if present | DB migration for new columns | If alembic is wired up; otherwise manual migration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `ollama` Python library | Raw `httpx` to Ollama REST API | Library is thinner abstraction, handles auth/base64/host automatically; raw httpx only if library has gaps |
| `ollama` Python library | LiteLLM | LiteLLM is a large dependency; overkill for two providers |
| Dict-based provider registry | Plugin system (setuptools entry_points) | Entry points are for packaged plugins; dict is sufficient for 2 providers and easy to understand |

**Installation:**
```bash
pip install ollama==0.6.1
```

---

## Architecture Patterns

### Recommended Project Structure
```
backend/vidpipe/
├── services/
│   ├── llm/
│   │   ├── __init__.py          # exports get_adapter()
│   │   ├── base.py              # LLMAdapter ABC
│   │   ├── vertex_adapter.py    # VertexAIAdapter
│   │   ├── ollama_adapter.py    # OllamaAdapter
│   │   └── registry.py          # provider registry + get_adapter()
│   ├── vertex_client.py         # UNCHANGED - still used by VertexAIAdapter
│   ├── prompt_rewriter.py       # REFACTORED - uses get_adapter()
│   ├── reverse_prompt_service.py # REFACTORED - uses get_adapter()
│   ├── cv_analysis_service.py   # REFACTORED - uses get_adapter()
│   └── candidate_scoring.py     # REFACTORED - uses get_adapter()
└── pipeline/
    └── storyboard.py            # REFACTORED - uses get_adapter()
```

### Pattern 1: Abstract LLMAdapter Base Class

**What:** Python ABC defining the contract all LLM providers must implement.
**When to use:** Always. Every LLM call in the pipeline goes through an adapter.

```python
# Source: https://docs.python.org/3/library/abc.html + Ollama docs pattern
from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel


class LLMAdapter(ABC):
    """Abstract base for LLM provider adapters.

    Two capabilities:
    - generate_text: structured text generation (storyboard, prompt rewriting)
    - analyze_image: vision + structured output (reverse-prompt, CV analysis, scoring)
    """

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> BaseModel:
        """Generate structured text output from a prompt.

        Args:
            prompt: User prompt text
            schema: Pydantic model class; adapter extracts JSON schema
            temperature: Generation temperature (0.0-1.0)
            system_prompt: Optional system/role instruction
            max_retries: Retry attempts on failure

        Returns:
            Validated pydantic model instance
        """
        ...

    @abstractmethod
    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        schema: Type[BaseModel],
        *,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> BaseModel:
        """Analyze an image and return structured output.

        Args:
            image_bytes: Raw image bytes
            prompt: Text prompt / question about the image
            schema: Pydantic model for response structure
            mime_type: Image MIME type
            temperature: Generation temperature
            max_retries: Retry attempts on failure

        Returns:
            Validated pydantic model instance
        """
        ...
```

### Pattern 2: VertexAIAdapter

**What:** Wraps existing `client.aio.models.generate_content()` calls into the adapter interface. Extracts JSON schema from pydantic models, uses existing `get_vertex_client()` singleton.
**When to use:** When model ID is a Gemini model (any model in `ALLOWED_TEXT_MODELS` set).

```python
# Source: existing storyboard.py pattern + codebase
from google.genai import types
from vidpipe.services.vertex_client import get_vertex_client, location_for_model

class VertexAIAdapter(LLMAdapter):
    def __init__(self, model_id: str):
        self._model_id = model_id

    async def generate_text(self, prompt, schema, *, temperature=0.7,
                             system_prompt=None, max_retries=3) -> BaseModel:
        client = get_vertex_client(location=location_for_model(self._model_id))
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        @retry(stop=stop_after_attempt(max_retries),
               retry=retry_if_exception_type(Exception))
        async def _attempt():
            response = await client.aio.models.generate_content(
                model=self._model_id,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                )
            )
            return schema.model_validate_json(response.text)

        return await _attempt()

    async def analyze_image(self, image_bytes, prompt, schema, *,
                             mime_type="image/jpeg", temperature=0.2,
                             max_retries=3) -> BaseModel:
        # Uses Part.from_bytes() + GenerateContentConfig pattern
        # Same as existing reverse_prompt_service.py approach
        ...
```

### Pattern 3: OllamaAdapter

**What:** Uses `ollama.AsyncClient` with `format=schema.model_json_schema()` for structured text; `images=[base64(bytes)]` for vision.
**When to use:** When model ID has `ollama/` prefix (or whatever convention chosen).

```python
# Source: https://deepwiki.com/ollama/ollama-python/4.4-structured-outputs
# Source: https://github.com/ollama/ollama-python
import base64
import json
from ollama import AsyncClient
from pydantic import BaseModel

class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        model_id: str,           # e.g. "ollama/llama3.1" -> strips to "llama3.1"
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
    ):
        self._ollama_model = model_id.removeprefix("ollama/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = AsyncClient(host=base_url, headers=headers)

    async def generate_text(self, prompt, schema, *, temperature=0.7,
                             system_prompt=None, max_retries=3) -> BaseModel:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        @retry(stop=stop_after_attempt(max_retries),
               retry=retry_if_exception_type(Exception))
        async def _attempt():
            response = await self._client.chat(
                model=self._ollama_model,
                messages=messages,
                format=schema.model_json_schema(),   # JSON schema dict
                options={"temperature": temperature},
            )
            return schema.model_validate_json(response.message.content)

        return await _attempt()

    async def analyze_image(self, image_bytes, prompt, schema, *,
                             mime_type="image/jpeg", temperature=0.2,
                             max_retries=3) -> BaseModel:
        b64 = base64.b64encode(image_bytes).decode()
        messages = [{
            "role": "user",
            "content": prompt,
            "images": [b64],   # Ollama SDK accepts b64 strings in REST; SDK also accepts paths
        }]

        @retry(stop=stop_after_attempt(max_retries),
               retry=retry_if_exception_type(Exception))
        async def _attempt():
            response = await self._client.chat(
                model=self._ollama_model,
                messages=messages,
                format=schema.model_json_schema(),
                options={"temperature": temperature},
            )
            return schema.model_validate_json(response.message.content)

        return await _attempt()
```

**IMPORTANT:** The `ollama` Python library's AsyncClient accepts file paths OR base64 strings in the `images` field (SDK handles encoding). For raw bytes, encode to base64 first. The REST API requires base64; the Python library may accept file paths. Use base64 from bytes to be safe.

### Pattern 4: Provider Registry

**What:** Maps model ID to adapter factory. Model prefix determines provider.
**When to use:** Always — called at the start of every pipeline stage instead of calling `get_vertex_client()` directly.

```python
# Source: standard Python registry pattern
from vidpipe.db.models import UserSettings

# Provider identification by model ID prefix
def _is_ollama_model(model_id: str) -> bool:
    return model_id.startswith("ollama/")

def _is_gemini_model(model_id: str) -> bool:
    return model_id.startswith("gemini-")


def get_adapter(
    model_id: str,
    user_settings: Optional[UserSettings] = None,
) -> LLMAdapter:
    """Get the appropriate LLMAdapter for a model ID.

    Args:
        model_id: e.g. "gemini-2.5-flash" or "ollama/llama3.1"
        user_settings: UserSettings row (for Ollama endpoint/key)

    Returns:
        LLMAdapter instance
    """
    if _is_ollama_model(model_id):
        # Determine Ollama endpoint
        if user_settings and user_settings.ollama_use_cloud:
            base_url = user_settings.ollama_endpoint or "https://ollama.com"
            api_key = user_settings.ollama_api_key
        else:
            base_url = user_settings.ollama_endpoint or "http://localhost:11434"
            api_key = None  # local needs no auth
        return OllamaAdapter(model_id, base_url=base_url, api_key=api_key)

    # Default: Gemini via VertexAI
    return VertexAIAdapter(model_id)
```

### Pattern 5: Ollama Model ID Convention

The codebase must distinguish Ollama models from Gemini models. Use a prefix convention:
- Gemini models: `gemini-2.5-flash`, `gemini-3-flash-preview` (existing)
- Ollama models: `ollama/llama3.1`, `ollama/llama3.2-vision`, `ollama/qwen2.5`

This convention allows `ALLOWED_TEXT_MODELS` validation to accept any `ollama/` prefix. The registry strips the prefix before calling the Ollama API. Critically, this prefix lives in the DB `project.text_model` and `project.vision_model` columns.

### Pattern 6: DB Schema Extensions

The `UserSettings` model needs new columns (Alembic migration or `CREATE TABLE` if not using Alembic):

```python
# Add to UserSettings in models.py
ollama_use_cloud: Mapped[bool] = mapped_column(Boolean, default=False)
ollama_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
ollama_endpoint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
ollama_models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
# ^ list of {"id": "ollama/llama3.1", "label": "Llama 3.1", "enabled": True}
```

The `Project` model needs one new column:
```python
vision_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
```

**Note:** The project currently has `text_model` already. The `vision_model` field defaults to `None`, and pipeline stages that do vision work (reverse-prompting, CV analysis, candidate scoring) will fall back to the text_model if vision_model is null, then to `settings.models.storyboard_llm` if text_model is also null.

### Pattern 7: Call-site Migration

Each existing call site follows the same migration pattern:

**Before (storyboard.py):**
```python
client = get_vertex_client(location=location_for_model(model_id))
response = await client.aio.models.generate_content(
    model=model_id, contents=[full_prompt],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=StoryboardOutput, temperature=0.7
    )
)
storyboard = StoryboardOutput.model_validate_json(response.text)
```

**After:**
```python
from vidpipe.services.llm.registry import get_adapter
# user_settings loaded from DB in route/orchestrator, passed down
adapter = get_adapter(model_id, user_settings=user_settings)
storyboard = await adapter.generate_text(
    full_prompt, StoryboardOutput, temperature=0.7, system_prompt=system_prompt
)
```

### Anti-Patterns to Avoid
- **Hardcoded `"gemini-2.5-flash"` model IDs:** All five call sites currently hardcode the model. They must accept model_id from the calling context (project.text_model or project.vision_model).
- **Creating adapters inside pipeline internals:** Adapter creation (including DB lookup for Ollama settings) should happen in the orchestrator or route, then be passed down. Don't create adapters deep in `_call_rewriter()`.
- **Mixing Ollama text/vision into one model:** Ollama models can be text-only OR vision-capable. The user selects them in separate dropdowns (`text_model` for storyboard/rewriting, `vision_model` for reverse-prompt/CV/scoring).
- **Streaming:** The current pipeline does NOT use streaming (it waits for complete responses). Keep `stream=False` (the default for the Python library when not setting `stream=True`). The ollama library defaults to `stream=True` for `chat()` — explicitly pass `stream=False` to get a single response object.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ollama HTTP requests | Custom httpx client against Ollama REST | `ollama.AsyncClient` | Library handles host normalization, auth headers, base64 image encoding, streaming |
| JSON schema extraction from Pydantic | Custom schema builder | `model.model_json_schema()` | Pydantic already generates the correct JSON schema format Ollama accepts |
| Retry logic | Custom retry loops | `tenacity` (already used) | Already present in every call site; consistent with codebase pattern |
| Model ID → provider routing | Long if/elif chains | Dict-based registry with prefix matching | Open/closed principle; adding a new provider doesn't touch existing code |
| Ollama model list | Hardcoded catalog | DB-stored list in `UserSettings.ollama_models` | Models are user-added; can't be hardcoded |

**Key insight:** The Ollama Python library abstracts ALL the differences between local (no auth, port 11434) and cloud (Bearer token, ollama.com) into just two constructor parameters: `host` and `headers`. The rest of the code is identical.

---

## Common Pitfalls

### Pitfall 1: Ollama Streaming Default
**What goes wrong:** `AsyncClient.chat()` defaults to `stream=True`, returning an async generator instead of a response object. Calling `.message.content` on a generator raises `AttributeError`.
**Why it happens:** Ollama defaults to streaming; the Python library reflects this.
**How to avoid:** Always pass `stream=False` (or do not pass `stream` and instead collect the generator). Recommendation: always pass `stream=False` explicitly.
**Warning signs:** `AttributeError: 'AsyncGenerator' object has no attribute 'message'`

### Pitfall 2: Ollama Structured Output Model Support
**What goes wrong:** Not all Ollama models reliably honor the JSON schema in `format`. Small models or reasoning-trace models (like gpt-oss) may produce extra text around the JSON or fail to close brackets.
**Why it happens:** Ollama uses grammar-constrained generation for structured output, but model quality varies.
**How to avoid:** Default Ollama text/vision model recommendations should be `llama3.2-vision` for vision (11B), `llama3.1` (8B) or `qwen2.5` (7B) for text — these have good structured output support. Add `temperature: 0.0` note in UI for structured calls.
**Warning signs:** `json.JSONDecodeError` or `ValidationError` after 3 retries; falls back to unstructured.

### Pitfall 3: Ollama Local vs Cloud Endpoint Confusion
**What goes wrong:** User toggles "cloud" in settings but forgets to add an API key. Requests go to `https://ollama.com/api/...` without auth and get 401.
**Why it happens:** The UI allows setting cloud mode without requiring an API key.
**How to avoid:** Validate in the settings save endpoint: if `ollama_use_cloud=True` and no existing `ollama_api_key`, return a warning (not hard error). Log a warning at adapter creation time.
**Warning signs:** 401 errors from Ollama cloud endpoint.

### Pitfall 4: Hardcoded Model IDs in Five Call Sites
**What goes wrong:** `prompt_rewriter.py` hardcodes `"gemini-2.5-flash"`. After refactor, if the model_id lookup fails, it falls through to the hardcoded string and bypasses the adapter.
**Why it happens:** The original code had no provider abstraction.
**How to avoid:** In the VertexAIAdapter and OllamaAdapter, the `model_id` passed at construction time is the source of truth. Remove all hardcoded model string literals from the five call sites.
**Warning signs:** Ollama model selected in UI but Gemini is still being called (check server logs).

### Pitfall 5: `vision_model` Fallback Chain
**What goes wrong:** `project.vision_model` is `None` (not set in GenerateForm), and the pipeline breaks trying to call `get_adapter(None, ...)`.
**Why it happens:** New column is optional; old projects don't have it.
**How to avoid:** Implement explicit fallback: `vision_model_id = project.vision_model or project.text_model or settings.models.storyboard_llm`. Document this in the registry's `get_adapter()` docstring.
**Warning signs:** `None` passed to adapter registry; `AttributeError` or `KeyError`.

### Pitfall 6: `user_settings` Availability in Deep Pipeline Code
**What goes wrong:** `_run_semantic_analysis()` in `cv_analysis_service.py` creates `get_vertex_client()` inline. After refactor, it needs `user_settings` to create an adapter. Currently `user_settings` is not passed to that level.
**Why it happens:** Settings were never needed inside CV analysis before.
**How to avoid:** Pass `user_settings` (or just the pre-constructed adapter) down through the call chain. Simplest approach: create the vision_adapter in the orchestrator and pass the adapter instance into `CVAnalysisService.analyze_generated_content()` as a parameter.
**Warning signs:** Missing `user_settings` context inside deeply nested service methods.

### Pitfall 7: `ALLOWED_TEXT_MODELS` Validation in routes.py
**What goes wrong:** The route validation set `ALLOWED_TEXT_MODELS` hardcodes Gemini model IDs. An Ollama model like `ollama/llama3.1` will fail validation with 422.
**Why it happens:** The allowed set was designed for fixed catalog models.
**How to avoid:** Either (a) accept any `ollama/`-prefixed model ID in validation, or (b) merge the user's Ollama model list into the validation set dynamically at request time. Option (a) is simpler: `if not (model_id in ALLOWED_TEXT_MODELS or model_id.startswith("ollama/"))`.
**Warning signs:** 422 validation error when trying to use Ollama model for generation.

---

## Code Examples

### Ollama AsyncClient with Structured Output (text)
```python
# Source: https://deepwiki.com/ollama/ollama-python/4.4-structured-outputs
# Source: https://github.com/ollama/ollama-python
from ollama import AsyncClient
from pydantic import BaseModel

class MySchema(BaseModel):
    name: str
    value: int

client = AsyncClient(host="http://localhost:11434")

response = await client.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "Give me a name and value"}],
    format=MySchema.model_json_schema(),   # dict, not the class itself
    stream=False,                          # IMPORTANT: disable streaming
    options={"temperature": 0.0},
)
result = MySchema.model_validate_json(response.message.content)
```

### Ollama AsyncClient with Vision + Structured Output
```python
# Source: https://ollama.com/blog/structured-outputs (vision example)
# Source: https://docs.ollama.com/capabilities/vision
import base64
from ollama import AsyncClient

image_bytes = Path("image.jpg").read_bytes()
b64_image = base64.b64encode(image_bytes).decode()

response = await client.chat(
    model="llama3.2-vision",
    messages=[{
        "role": "user",
        "content": "Describe this image in JSON",
        "images": [b64_image],  # list of b64 strings
    }],
    format=MySchema.model_json_schema(),
    stream=False,
    options={"temperature": 0.0},
)
result = MySchema.model_validate_json(response.message.content)
```

### Ollama Cloud Authentication
```python
# Source: https://docs.ollama.com/api/authentication
# Source: https://github.com/ollama/ollama-python (README)
import os
from ollama import AsyncClient

# Cloud: requires API key from https://ollama.com/settings/api-keys
client = AsyncClient(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)

# Local: no auth needed
client_local = AsyncClient(host="http://localhost:11434")
```

### Existing VertexAI Call Pattern (for reference during refactor)
```python
# Source: existing storyboard.py (confirmed in codebase)
# Pattern used by all 5 call sites with slight variations
response = await client.aio.models.generate_content(
    model=model_id,
    contents=[full_prompt],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ResponseSchema,       # pydantic class
        temperature=0.7,
    )
)
result = ResponseSchema.model_validate_json(response.text)
```

### Provider Registry
```python
# Source: standard Python dict registry pattern
from typing import Optional
from vidpipe.db.models import UserSettings
from vidpipe.services.llm.base import LLMAdapter
from vidpipe.services.llm.vertex_adapter import VertexAIAdapter
from vidpipe.services.llm.ollama_adapter import OllamaAdapter

def get_adapter(
    model_id: str,
    user_settings: Optional[UserSettings] = None,
) -> LLMAdapter:
    if model_id.startswith("ollama/"):
        use_cloud = bool(user_settings and user_settings.ollama_use_cloud)
        if use_cloud:
            base_url = (user_settings.ollama_endpoint or "https://ollama.com")
            api_key = user_settings.ollama_api_key
        else:
            base_url = (user_settings and user_settings.ollama_endpoint) or "http://localhost:11434"
            api_key = None
        return OllamaAdapter(model_id, base_url=base_url, api_key=api_key)
    # Fallback: all Gemini models
    return VertexAIAdapter(model_id)
```

### Frontend: Ollama Settings Section (pattern consistent with existing ComfyUI section)
```tsx
// Source: existing SettingsPage.tsx pattern for ComfyUI section
{/* Ollama Configuration */}
<section className="space-y-4">
  <h2 className="text-lg font-semibold text-white">Ollama Configuration</h2>

  {/* Cloud vs Local toggle */}
  <div className="flex items-center gap-3">
    <span className="text-sm text-gray-300">Mode:</span>
    <button onClick={() => setOllamaUseCloud(false)}
      className={clsx("px-3 py-1 rounded text-sm", !ollamaUseCloud ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300")}>
      Local
    </button>
    <button onClick={() => setOllamaUseCloud(true)}
      className={clsx("px-3 py-1 rounded text-sm", ollamaUseCloud ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300")}>
      Cloud (ollama.com)
    </button>
  </div>

  {/* API Key - only visible in cloud mode */}
  {ollamaUseCloud && (
    <input type="password" value={ollamaApiKey}
      onChange={(e) => setOllamaApiKey(e.target.value)}
      placeholder="Ollama API key" ... />
  )}

  {/* Custom endpoint - always visible */}
  <input type="text" value={ollamaEndpoint}
    onChange={(e) => setOllamaEndpoint(e.target.value)}
    placeholder={ollamaUseCloud ? "https://ollama.com" : "http://localhost:11434"} ... />

  {/* Model management */}
  <OllamaModelManager models={ollamaModels} onModelsChange={setOllamaModels} />
</section>
```

### Frontend: Dynamic Ollama Model List in GenerateForm
```tsx
// Ollama models from user settings appear alongside Gemini models in dropdowns
// They are prefixed with "ollama/" to distinguish them

const allTextModels = useMemo(() => {
  const base = TEXT_MODELS;  // static Gemini list
  const ollamaModels = (modelSettings?.ollama_models ?? [])
    .filter(m => m.enabled)
    .map(m => ({ id: m.id, label: m.label, costPerCall: 0 }));
  return [...base, ...ollamaModels];
}, [modelSettings]);
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct `client.aio.models.generate_content()` calls scattered in 5 files | Calls go through `LLMAdapter.generate_text()` / `analyze_image()` | Phase 13 | Enables Ollama (or any future provider) without touching call sites |
| Hardcoded `"gemini-2.5-flash"` strings | Model ID from project.text_model / project.vision_model | Phase 13 | Per-project model selection |
| No Ollama support | OllamaAdapter via `ollama` Python library v0.6.1 | Phase 13 | Local inference option |
| No vision_model column | `project.vision_model` separate from `text_model` | Phase 13 | Different models for text vs vision tasks |
| Static model catalog only | Static Gemini catalog + dynamic Ollama model list in DB | Phase 13 | User-managed model list |

**Deprecated/outdated after this phase:**
- Direct calls to `get_vertex_client()` from pipeline stages (should only be called from inside `VertexAIAdapter`)
- Hardcoded `"gemini-2.5-flash"` string literals in `prompt_rewriter.py`, `reverse_prompt_service.py`, `cv_analysis_service.py`, `candidate_scoring.py`

---

## Open Questions

1. **Model ID naming convention for Ollama**
   - What we know: Need a prefix to distinguish Ollama from Gemini models
   - What's unclear: Whether `ollama/` prefix or another convention (e.g., `local:llama3.1`) is preferred
   - Recommendation: Use `ollama/` prefix (e.g., `ollama/llama3.1`, `ollama/llama3.2-vision`). Consistent with how other multi-provider tools (LiteLLM uses `ollama/model`) handle this.

2. **Vision model fallback when project.vision_model is None**
   - What we know: Most projects won't have vision_model set initially
   - What's unclear: Should it fall back to text_model (which may be Ollama) even for vision calls?
   - Recommendation: Fallback chain: `vision_model → text_model → settings.models.storyboard_llm`. If the selected model is Ollama but text-only, the analyze_image call will fail at inference time. Add model capability metadata to the Ollama model list (e.g., `{"id": "ollama/llama3.1", "vision": false}`).

3. **Alembic migration vs manual migration**
   - What we know: The codebase uses SQLAlchemy async; no Alembic was detected in requirements.txt
   - What's unclear: Whether `create_all()` in app startup is used or a migration script
   - Recommendation: Check app startup code. If using `create_all()`, adding columns to models.py will not apply to existing tables — need explicit `ALTER TABLE` statements or a migration script.

4. **Ollama cloud API rate limits and pricing**
   - What we know: Ollama cloud provides inference; API key is free to create
   - What's unclear: Rate limits and pricing for cloud models vs local
   - Recommendation: LOW confidence on specifics. The UI should show a note that Ollama cloud pricing/limits are defined by ollama.com account settings, not this app.

5. **Adapter caching / instantiation overhead**
   - What we know: VertexAI client is cached per-location. OllamaAdapter creates an AsyncClient per call.
   - What's unclear: Whether AsyncClient should be a singleton or per-request
   - Recommendation: Cache the adapter in the orchestrator per pipeline run (not per scene). One adapter instance per model_id per run is sufficient. `httpx.AsyncClient` inside `ollama.AsyncClient` is connection-pooling capable.

---

## Existing Call Sites - Migration Checklist

The five sites to refactor, with their current hardcoded model behavior:

| File | Method | Current Model | Uses Vision? | Schema Type |
|------|---------|--------------|-------------|-------------|
| `pipeline/storyboard.py` | `generate_storyboard()` | `project.text_model or settings.models.storyboard_llm` | No | `StoryboardOutput` / `EnhancedStoryboardOutput` |
| `services/prompt_rewriter.py` | `_call_rewriter()` | Hardcoded `"gemini-2.5-flash"` | No | `RewrittenKeyframePromptOutput` / `RewrittenVideoPromptOutput` |
| `services/reverse_prompt_service.py` | `reverse_prompt_asset()` | Hardcoded `"gemini-2.5-flash"` | **Yes** - image bytes + Part.from_bytes | Dict schema (inline) |
| `services/cv_analysis_service.py` | `_run_semantic_analysis()` | Hardcoded `"gemini-2.5-flash"` | **Yes** - multiple frames | Dict schema (inline) |
| `services/candidate_scoring.py` | `_score_visual_and_prompt()` | Hardcoded `"gemini-2.5-flash"` | **Yes** - first frame JPEG | Dict schema (inline) |

**Note:** Three of five call sites use vision (image input). These require `analyze_image()`. The dict-style schemas in the three vision sites should be converted to Pydantic models for consistency with the adapter interface.

---

## Sources

### Primary (HIGH confidence)
- https://github.com/ollama/ollama-python/blob/main/ollama/_client.py — `AsyncClient` constructor, `format` parameter type, `OLLAMA_API_KEY` env var auto-handling, auth header pattern
- https://deepwiki.com/ollama/ollama-python/4.4-structured-outputs — structured output code examples with Pydantic and AsyncClient
- https://docs.ollama.com/capabilities/structured-outputs — official structured outputs docs with vision + schema pattern
- https://docs.ollama.com/api/authentication — official authentication docs: local=no auth, cloud=Bearer token, `OLLAMA_API_KEY` env var
- https://docs.ollama.com/capabilities/vision — vision API: images parameter, base64 encoding, SDK file path support
- https://docs.python.org/3/library/abc.html — Python ABC for LLMAdapter base class
- https://pypi.org/project/ollama/ — version 0.6.1, released November 13, 2025
- Codebase (existing files) — all 5 call sites, UserSettings model, routes.py validation sets, constants.ts, SettingsPage.tsx, GenerateForm.tsx

### Secondary (MEDIUM confidence)
- https://ollama.com/blog/structured-outputs — vision + structured output blog post; confirmed same format parameter works for vision
- https://github.com/ollama/ollama/blob/main/docs/api.md — REST API docs: `/api/generate` and `/api/chat` field names confirmed
- https://docs.ollama.com/cloud — cloud model description: sign-in or API key, host is `https://ollama.com`

### Tertiary (LOW confidence)
- https://www.glukhov.org/post/2025/10/ollama-gpt-oss-structured-output-issues/ — model-specific structured output failures (single source; used only for Pitfall 2)
- Ollama cloud pricing/rate limits — not publicly documented in detail; marked LOW

---

## Metadata

**Confidence breakdown:**
- Standard stack (ollama library, ABC pattern): HIGH — official docs + PyPI + codebase verified
- Ollama REST API (format, images, streaming): HIGH — official GitHub docs + DeepWiki code examples
- Ollama Cloud authentication: HIGH — official authentication docs page
- Architecture patterns (adapter, registry): HIGH — standard Python patterns + existing codebase patterns
- Ollama structured output reliability per-model: MEDIUM — some models have issues; verified in multiple sources
- Ollama cloud pricing/limits: LOW — not documented publicly

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (Ollama is fast-moving; re-check if library > 0.6.1 is released)
