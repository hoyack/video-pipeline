---
phase: 13-llm-provider-abstraction-ollama
plan: "03"
subsystem: frontend
tags: [ollama, settings-ui, generate-form, typescript, vision-model]
dependency_graph:
  requires:
    - "13-01 (OllamaModelEntry type in backend routes, Ollama DB columns, settings API)"
  provides:
    - "SettingsPage Ollama Configuration section with full model management"
    - "GenerateForm vision_model dropdown with Same-as-Text default"
    - "Ollama model integration in Text Model and Vision Model button groups"
    - "OllamaModelEntry TypeScript interface in frontend types"
  affects:
    - "frontend/src/api/types.ts"
    - "frontend/src/components/SettingsPage.tsx"
    - "frontend/src/components/GenerateForm.tsx"
tech_stack:
  added: []
  patterns:
    - "useMemo for merged Ollama+Gemini model lists (allTextModels, allVisionModels)"
    - "Empty string sentinel for 'Same as Text' vision model selection"
    - "Vision heuristic: /vision|llava/i regex for auto-detecting vision capability on model name"
key_files:
  created: []
  modified:
    - frontend/src/api/types.ts
    - frontend/src/components/SettingsPage.tsx
    - frontend/src/components/GenerateForm.tsx
decisions:
  - "Empty string (not null/undefined) for visionModel state — simplifies controlled input and conditional omission from JSON"
  - "allTextModels merges filteredTextModels + enabled Ollama models — Ollama models flow through same filter path as Gemini"
  - "allVisionModels uses filteredTextModels (all Gemini text models support vision) + Ollama vision models"
  - "vision_model sent as undefined (not null) when empty — JSON.stringify omits undefined fields, backend uses fallback chain"
  - "Auto-detect vision flag via /vision|llava/i heuristic on model name — users can override via toggle"
metrics:
  duration: "3 min"
  completed: "2026-02-19"
  tasks_completed: 2
  files_changed: 3
---

# Phase 13 Plan 03: Frontend Ollama UI + Vision Model Dropdown Summary

**One-liner:** Ollama Settings section with cloud/local toggle, model management (add/toggle vision/enable/remove), and GenerateForm vision_model dropdown that merges Gemini and Ollama models.

## What Was Built

### Task 1: TypeScript types + API client + Ollama Settings UI (commit b52f2f2)

**types.ts** — Added 6 changes:
- `OllamaModelEntry` interface: `{ id, label, enabled, vision }` for Ollama model management
- `GenerateRequest`: added `vision_model?: string`
- `ProjectDetail` + `ProjectListItem`: added `vision_model?: string | null`
- `UserSettingsResponse`: added `ollama_use_cloud`, `has_ollama_key`, `ollama_endpoint`, `ollama_models`
- `UserSettingsUpdate`: added `ollama_use_cloud`, `ollama_api_key`, `clear_ollama_key`, `ollama_endpoint`, `ollama_models`
- `EnabledModelsResponse`: added `ollama_models: OllamaModelEntry[] | null`

**client.ts** — No changes needed: `generateVideo()` already uses `JSON.stringify(body)` which passes through all fields including `vision_model`.

**SettingsPage.tsx** — Full Ollama Configuration section:
- State: `ollamaUseCloud`, `ollamaApiKey`, `hasOllamaKey`, `ollamaEndpoint`, `ollamaModels`, `newModelName`
- `useEffect` hydrates all Ollama state from `getSettings()` response
- `handleSave()` includes all Ollama fields; updates `hasOllamaKey` and clears `ollamaApiKey` on save
- Helper functions: `handleAddOllamaModel`, `toggleOllamaModelEnabled`, `toggleOllamaModelVision`, `removeOllamaModel`
- JSX section: cloud/local mode toggle buttons, conditional API key input (cloud only), endpoint URL field, model list with enable/disable toggle + vision badge + remove button, empty state message

### Task 2: GenerateForm vision_model dropdown + Ollama model integration (commit 978a5f0)

**GenerateForm.tsx**:
- `visionModel` state (empty string = "Same as Text")
- `allTextModels` useMemo: `filteredTextModels` + enabled Ollama models as `{ id, label, costPerCall: 0 }`
- `allVisionModels` useMemo: `filteredTextModels` (Gemini text all support vision) + enabled Ollama vision models
- Text Model button group updated to use `allTextModels`
- Vision Model section added after Text Model: "Same as Text" default button + model buttons
- `handleSubmit` sends `vision_model: visionModel || undefined` — undefined omitted from JSON
- Model validation useEffect updated: uses `allTextModels` for text model validation; resets `visionModel` to `""` if selected model removed

## Verification Results

All checks passed:
1. `npx tsc --noEmit` — zero TypeScript errors after both tasks
2. Must-have artifacts confirmed: SettingsPage contains `ollama`, GenerateForm contains `vision_model`, types.ts contains `ollama_models`, client.ts passes through all fields

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

Modified files:
- [x] frontend/src/api/types.ts — OllamaModelEntry, vision_model, Ollama settings fields
- [x] frontend/src/components/SettingsPage.tsx — Ollama Configuration section
- [x] frontend/src/components/GenerateForm.tsx — Vision Model dropdown + allTextModels/allVisionModels

Commits:
- [x] b52f2f2 — Task 1 (types + SettingsPage)
- [x] 978a5f0 — Task 2 (GenerateForm)

## Self-Check: PASSED
