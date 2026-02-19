---
phase: 13-llm-provider-abstraction-ollama
verified: 2026-02-19T18:15:00Z
status: passed
score: 18/18 must-haves verified
re_verification: false
human_verification:
  - test: "Settings page Ollama Configuration section renders and cloud/local toggle works"
    expected: "Clicking Local hides API key field; clicking Cloud shows API key input and hides it for local"
    why_human: "Cannot verify conditional rendering behavior programmatically without running the browser"
  - test: "Add an Ollama model in settings, save, then open GenerateForm — model appears in Text Model and Vision Model button groups"
    expected: "A model added as e.g. 'llama3.1' appears in text model row and, if vision flag set, in vision model row"
    why_human: "End-to-end state flow from SettingsPage through API round-trip to GenerateForm useMemo — requires live browser session"
  - test: "Ollama-powered pipeline execution (when local Ollama is running)"
    expected: "Project with text_model='ollama/llama3.1' generates storyboard via OllamaAdapter.generate_text(), not Gemini"
    why_human: "Requires live Ollama server; cannot verify provider routing at runtime without executing a real pipeline run"
---

# Phase 13: LLM Provider Abstraction & Ollama Integration Verification Report

**Phase Goal:** Abstract all LLM text/vision calls behind a provider adapter interface, extract existing Vertex AI/Gemini calls into a Vertex adapter, implement an Ollama adapter for text and vision models, add settings UI for Ollama configuration (API key, cloud/local toggle, endpoint, model management), and wire the pipeline to route through the correct adapter based on selected model provider.

**Verified:** 2026-02-19T18:15:00Z
**Status:** PASSED (with 3 items flagged for human verification)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | LLMAdapter ABC defines generate_text() and analyze_image() with consistent async signatures | VERIFIED | `base.py` — abstract class with both async abstract methods, correct signatures matching plan spec |
| 2 | VertexAIAdapter wraps google-genai client with location-aware routing and structured output | VERIFIED | `vertex_adapter.py` — calls `get_vertex_client(location=location_for_model(...))`, uses `response_mime_type="application/json"` + `response_schema=schema` |
| 3 | OllamaAdapter connects via ollama.AsyncClient with auth headers, structured JSON output, and vision support | VERIFIED | `ollama_adapter.py` — uses `AsyncClient(host=base_url, headers=...)`, `format=schema.model_json_schema()`, `stream=False`, base64 image encoding in analyze_image() |
| 4 | get_adapter() routes model IDs to correct adapter based on prefix (ollama/ vs gemini-) | VERIFIED | `registry.py` — `_is_ollama_model()` checks `startswith("ollama/")`, routes to OllamaAdapter; everything else routes to VertexAIAdapter |
| 5 | UserSettings has Ollama config columns; Project has vision_model column | VERIFIED | `db/models.py` lines 183, 404-408 — vision_model on Project, four Ollama columns on UserSettings |
| 6 | Settings API returns and accepts Ollama configuration fields | VERIFIED | `routes.py` lines 2597-2600, 2620-2624, 2663-2666, 2728-2737, 2755-2758, 2780 — full round-trip: GET returns ollama fields, PUT saves them |
| 7 | Storyboard generation routes through adapter.generate_text() with project's text_model | VERIFIED | `storyboard.py` line 24: `from vidpipe.services.llm import get_adapter, LLMAdapter`; signature accepts `text_adapter: Optional[LLMAdapter] = None`; line 193 `adapter = text_adapter or get_adapter(model_id)`; line 254 `await adapter.generate_text(...)` |
| 8 | Prompt rewriting routes through adapter.generate_text() instead of hardcoded gemini-2.5-flash | VERIFIED | `prompt_rewriter.py` lines 26, 101-102, 188-189 — adapter injection pattern with `get_adapter("gemini-2.5-flash")` fallback |
| 9 | Reverse prompting routes through adapter.analyze_image() with ReversePromptOutput schema | VERIFIED | `reverse_prompt_service.py` lines 11-12, 20-27, 63-67 — adapter injection, `adapter.analyze_image(..., schema=ReversePromptOutput, ...)` |
| 10 | CV semantic analysis routes through adapter.analyze_image() with SemanticAnalysisOutput schema | VERIFIED | `cv_analysis_service.py` lines 24, 29, 80-87, 367, 422-425 — adapter injection, `adapter.analyze_image(..., schema=SemanticAnalysisOutput, ...)` |
| 11 | Candidate scoring routes through adapter.analyze_image() with VisualPromptScoreOutput schema | VERIFIED | `candidate_scoring.py` lines 18-19, 120-127, 350, 364-367 — adapter injection, `adapter.analyze_image(..., schema=VisualPromptScoreOutput, ...)` |
| 12 | Vision call sites use vision_adapter with fallback chain (vision_model -> text_model -> settings.models.storyboard_llm) | VERIFIED | `orchestrator/pipeline.py` lines 197-200: `vision_model_id = project.vision_model or project.text_model or app_settings.models.storyboard_llm`; both adapters created from this chain |
| 13 | generate_keyframes() and generate_videos() accept and thread adapter parameters | VERIFIED | `keyframes.py` line 368: `text_adapter: Optional[LLMAdapter] = None`; `video_gen.py` lines 799-800: `text_adapter + vision_adapter`; `video_gen.py` lines 821-822: per-call CVAnalysisService + CandidateScoringService with vision_adapter |
| 14 | Orchestrator creates adapters from project config + UserSettings and passes to all pipeline stages | VERIFIED | `orchestrator/pipeline.py` lines 189-200, 233, 253, 276 — loads UserSettings, creates both adapters, passes `text_adapter=text_adapter` to storyboard + keyframes and both adapters to generate_videos |
| 15 | Settings page shows Ollama section with cloud/local toggle, API key input (cloud only), endpoint URL, and model management | VERIFIED | `SettingsPage.tsx` lines 360-461 — full JSX section with all required controls; `ollamaUseCloud` gates API key visibility |
| 16 | Added Ollama models appear in GenerateForm text_model and vision_model dropdowns when enabled | VERIFIED | `GenerateForm.tsx` lines 76-91 — `allTextModels` useMemo merges `filteredTextModels` + enabled ollama models; `allVisionModels` useMemo includes enabled Ollama vision models |
| 17 | GenerateForm has separate vision_model dropdown with "Same as Text" default | VERIFIED | `GenerateForm.tsx` lines 322-358 — Vision Model section rendered with "Same as Text" button (visionModel="" state) and model buttons |
| 18 | ollama/ prefixed models accepted in route validation | VERIFIED | `routes.py` lines 571-581 — `request.text_model.startswith("ollama/")` accepted; `request.vision_model.startswith("ollama/")` accepted |

**Score:** 18/18 truths verified

---

## Required Artifacts

### Plan 01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/vidpipe/services/llm/base.py` | LLMAdapter ABC | VERIFIED | 70 lines; both abstract methods with correct signatures |
| `backend/vidpipe/services/llm/vertex_adapter.py` | VertexAIAdapter | VERIFIED | 133 lines; full implementation with tenacity retry |
| `backend/vidpipe/services/llm/ollama_adapter.py` | OllamaAdapter | VERIFIED | 141 lines; stream=False, base64 vision, JSON schema format |
| `backend/vidpipe/services/llm/registry.py` | get_adapter() registry | VERIFIED | 78 lines; prefix routing with cloud/local Ollama logic |
| `backend/vidpipe/services/llm/__init__.py` | Re-exports | VERIFIED | Exports `LLMAdapter` and `get_adapter` |
| `backend/vidpipe/schemas/llm_vision.py` | 3 Pydantic vision schemas | VERIFIED | ReversePromptOutput, SemanticAnalysisOutput, VisualPromptScoreOutput |

### Plan 02 Artifacts

| Artifact | Contains | Status | Details |
|----------|----------|--------|---------|
| `backend/vidpipe/pipeline/storyboard.py` | get_adapter | VERIFIED | Imports LLM package; signature includes text_adapter parameter |
| `backend/vidpipe/services/prompt_rewriter.py` | get_adapter | VERIFIED | Constructor accepts text_adapter; _call_rewriter uses adapter.generate_text() |
| `backend/vidpipe/services/reverse_prompt_service.py` | analyze_image | VERIFIED | Constructor accepts vision_adapter; uses adapter.analyze_image() with ReversePromptOutput |
| `backend/vidpipe/services/cv_analysis_service.py` | analyze_image | VERIFIED | Constructor accepts vision_adapter; _run_semantic_analysis uses adapter.analyze_image() |
| `backend/vidpipe/services/candidate_scoring.py` | analyze_image | VERIFIED | Constructor accepts vision_adapter; _score_visual_and_prompt uses adapter.analyze_image() |
| `backend/vidpipe/pipeline/keyframes.py` | text_adapter | VERIFIED | generate_keyframes() signature: `text_adapter: Optional[LLMAdapter] = None` |
| `backend/vidpipe/pipeline/video_gen.py` | text_adapter | VERIFIED | generate_videos() signature: both adapters; per-call CVAnalysisService + CandidateScoringService |

### Plan 03 Artifacts

| Artifact | Contains | Status | Details |
|----------|----------|--------|---------|
| `frontend/src/components/SettingsPage.tsx` | ollama | VERIFIED | Full Ollama Configuration section, state, helpers, JSX |
| `frontend/src/components/GenerateForm.tsx` | vision_model | VERIFIED | visionModel state, allTextModels/allVisionModels memos, Vision Model section JSX |
| `frontend/src/api/types.ts` | ollama_models | VERIFIED | OllamaModelEntry interface; all required Ollama fields on UserSettings types; vision_model on GenerateRequest |
| `frontend/src/api/client.ts` | (passthrough) | VERIFIED | generateVideo() uses JSON.stringify(body) — passes vision_model through without cherry-picking |

---

## Key Link Verification

### Plan 01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `registry.py` | `vertex_adapter.py` | VertexAIAdapter import + instantiation | WIRED | `from vidpipe.services.llm.vertex_adapter import VertexAIAdapter` inside get_adapter(); `return VertexAIAdapter(model_id=model_id)` |
| `registry.py` | `ollama_adapter.py` | OllamaAdapter import + instantiation | WIRED | `from vidpipe.services.llm.ollama_adapter import OllamaAdapter` inside get_adapter(); `return OllamaAdapter(...)` |
| `vertex_adapter.py` | `vertex_client.py` | get_vertex_client() call | WIRED | Line 15: `from vidpipe.services.vertex_client import get_vertex_client, location_for_model`; called in both _call() closures |
| `ollama_adapter.py` | `ollama.AsyncClient` | library import | WIRED | Line 11: `from ollama import AsyncClient`; used in __init__ |

### Plan 02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `storyboard.py` | `registry.py` | get_adapter() import + call | WIRED | Line 24: `from vidpipe.services.llm import get_adapter, LLMAdapter`; line 193: `adapter = text_adapter or get_adapter(model_id)` |
| `orchestrator/pipeline.py` | `registry.py` | Creates adapters, passes to pipeline | WIRED | Line 27: `from vidpipe.services.llm import get_adapter`; lines 199-200: adapter creation; lines 233, 253, 276: passed to all three stages |
| `keyframes.py` | `prompt_rewriter.py` | PromptRewriterService(text_adapter=text_adapter) | WIRED | Line 519: `rewriter = PromptRewriterService(text_adapter=text_adapter)` |
| `video_gen.py` | `prompt_rewriter.py` | PromptRewriterService(text_adapter=text_adapter) | WIRED | Line 1222: `rewriter = PromptRewriterService(text_adapter=text_adapter)` |
| `reverse_prompt_service.py` | `base.py` | adapter.analyze_image() | WIRED | Line 64: `result = await adapter.analyze_image(...)` after injection or fallback |

### Plan 03 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `SettingsPage.tsx` | `client.ts` | updateSettings() with Ollama fields | WIRED | Line 96-99: ollama fields in handleSave() body passed to updateSettings() |
| `GenerateForm.tsx` | `types.ts` | EnabledModelsResponse with ollama_models | WIRED | modelSettings?.ollama_models used in allTextModels/allVisionModels useMemo |
| `GenerateForm.tsx` | `client.ts` | generateVideo() with vision_model | WIRED | Line 163: `vision_model: visionModel \|\| undefined` in handleSubmit body |

---

## Requirements Coverage

| Requirement | Description | Plan | Status | Evidence |
|-------------|-------------|------|--------|---------|
| LLMA-01 | LLMAdapter ABC with generate_text() and analyze_image() | 13-01 | SATISFIED | `base.py` — full ABC implementation |
| LLMA-02 | VertexAIAdapter wraps all Gemini calls (storyboard, prompt rewriting, reverse-prompting, CV, scoring) | 13-01 + 13-02 | SATISFIED | VertexAIAdapter created; all 5 call sites migrated |
| LLMA-03 | OllamaAdapter with text JSON mode and vision analysis via Ollama REST API | 13-01 | SATISFIED | `ollama_adapter.py` — uses ollama.AsyncClient, model_json_schema() for structured output, base64 vision |
| LLMA-04 | Settings UI: API key, cloud/local toggle, custom endpoint URL | 13-01 + 13-03 | SATISFIED | Backend DB columns + API endpoints (13-01); SettingsPage Ollama section (13-03) |
| LLMA-05 | Model management: add/toggle/remove models; added models appear in GenerateForm dropdowns | 13-03 | SATISFIED | handleAddOllamaModel, toggleOllamaModelEnabled, removeOllamaModel helpers; allTextModels/allVisionModels merges Ollama models |
| LLMA-06 | GenerateForm: text_model + vision_model selectors; correct adapter used per operation | 13-02 + 13-03 | SATISFIED | Vision Model section in GenerateForm; orchestrator uses vision_model for CV/scoring adapters, text_model for storyboard/rewriting |
| LLMA-07 | Provider routing via registry; Gemini → VertexAIAdapter; Ollama → OllamaAdapter | 13-01 + 13-02 | SATISFIED | `registry.py` get_adapter() routing; future providers can register without modifying core pipeline |

**All 7 requirements: SATISFIED**

---

## Notable Finding: google.genai in keyframes.py and video_gen.py

`keyframes.py` and `video_gen.py` still import `from google.genai import types` and `from google.genai.errors import ClientError, ServerError`. This is NOT a gap — these imports are for:

- `keyframes.py`: Imagen image generation API (`types.Part.from_bytes`, `types.GenerateContentConfig`) — this is image generation, not LLM text/vision calls
- `video_gen.py`: Veo video generation API (`types.GenerateVideosOperation`, `types.GenerateVideosConfig`, `types.Image`) — this is video generation

The phase plan explicitly scoped migration to LLM text/vision calls only. Image generation (Imagen) and video generation (Veo) remain on the direct google.genai SDK — this is correct and intended behavior. The plan's success criteria states "No direct google.genai SDK calls in the five call site files" (storyboard, prompt_rewriter, reverse_prompt_service, cv_analysis_service, candidate_scoring) — all five are clean. keyframes.py and video_gen.py retain google.genai only for their Imagen/Veo API surface.

---

## Anti-Patterns Found

None. No TODO/FIXME/placeholder/stub patterns found in any of the 14+ files created or modified by this phase.

---

## Human Verification Required

### 1. Ollama Settings Section Visual Behavior

**Test:** Open Settings page. Verify Ollama Configuration section renders. Toggle between Local and Cloud mode.
**Expected:** In Local mode — API key input is hidden. In Cloud mode — API key input appears. Endpoint URL field visible in both modes.
**Why human:** Conditional JSX rendering (`{ollamaUseCloud && ...}`) requires browser execution to verify.

### 2. Ollama Model Round-Trip to GenerateForm

**Test:** In SettingsPage, add model "llama3.1" (non-vision) and "llava" (auto-detects vision). Save settings. Open GenerateForm.
**Expected:** "llama3.1 (Ollama)" appears in Text Model buttons. "llava (Ollama)" appears in both Text Model AND Vision Model buttons.
**Why human:** Requires live API round-trip: SettingsPage PUT → backend stores → GenerateForm GET /settings/models → useMemo merges. Cannot verify this state flow without a running server + browser.

### 3. Ollama Pipeline Execution

**Test:** Start a pipeline with `text_model="ollama/llama3.1"` and a running local Ollama server with llama3.1 installed.
**Expected:** Storyboard generation calls OllamaAdapter.generate_text() instead of VertexAIAdapter. Log shows "Routing ollama/llama3.1 to OllamaAdapter".
**Why human:** Requires Ollama server running; cannot verify at import-time that the correct adapter is invoked.

---

## Gaps Summary

No gaps found. All 18 observable truths verified. All 14 artifacts exist and are substantive (not stubs). All 12 key links are wired. All 7 requirements satisfied. TypeScript compiles cleanly. No anti-patterns. The phase goal is fully achieved.

---

_Verified: 2026-02-19T18:15:00Z_
_Verifier: Claude (gsd-verifier)_
