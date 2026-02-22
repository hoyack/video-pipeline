# Frontend Documentation

This document describes the frontend implementation in `frontend/` as currently implemented.

## 1. Purpose and Scope

The frontend is a React SPA that provides:

- Project creation (`GenerateForm`)
- Live pipeline monitoring (`ProgressView`)
- Project review, continuation, editing, and forking (`ProjectDetail`, `EditModeOverlay`, `EditForkPanel`)
- Manifest management (`ManifestLibrary`, `ManifestCreator`)
- Global model/provider settings (`SettingsPage`)
- Usage analytics (`Dashboard`)

It communicates with the backend through `/api/*` endpoints and does not use server-side rendering.

## 2. Stack and Runtime

## 2.1 Tooling and Libraries

- React 19 (`react`, `react-dom`)
- TypeScript
- Vite 7
- Tailwind CSS 4
- `clsx` for conditional classes
- `react-markdown` for markdown preview in the editor modal

## 2.2 Build and Dev Commands

From `frontend/package.json`:

- `npm run dev`: Start Vite dev server
- `npm run build`: Type-check and create production build
- `npm run lint`: Run ESLint
- `npm run preview`: Preview production build

## 2.3 Dev Proxy

`frontend/vite.config.ts` proxies `/api` to `http://localhost:8000`, so frontend calls stay same-origin in development.

## 3. Application Architecture

## 3.1 Entry and Global Styles

- `src/main.tsx` mounts `<App />` in `StrictMode`
- `src/index.css` imports Tailwind and sets global dark background/text

## 3.2 Navigation Model

The app uses local state routing in `src/App.tsx`, not React Router.

- `currentView` controls active screen
- `activeProjectId` holds current project in detail/progress screens
- `activeManifestId` holds selected manifest for creator screen

Views:

- `generate`
- `progress`
- `list`
- `detail`
- `dashboard`
- `manifests`
- `manifest-creator`
- `settings`

Implications:

- URL does not represent route/state
- Page refresh resets to default view (`list`)
- Browser back/forward does not handle in-app navigation

## 3.3 Layout Shell

`Layout.tsx` provides:

- Top nav (`Projects`, `Manifests`, `Dashboard`, `Settings`)
- Active tab styling
- Shared max-width content container
- Brand button (`vidpipe`) that navigates to project list

## 3.4 State Strategy

There is no global state store (Redux/Zustand/etc). State is local to components:

- Screen-level state for forms, loading/error flags, and mode toggles
- API data fetched per view
- Polling hooks for near real-time updates
- `localStorage` used only for project list view preference (`list` vs `cards`)

## 4. Shared Domain Constants and Hooks

## 4.1 Pipeline and Model Constants (`src/lib/constants.ts`)

Defines:

- Pipeline stage order: `pending`, `storyboarding`, `keyframing`, `video_gen`, `stitching`, `complete`
- Stage labels and terminal statuses (`complete`, `failed`, `stopped`, `staged`)
- Slow stage for polling backoff (`video_gen`)
- Style options and aspect ratios
- Duration defaults and limits
- Text/image/video model catalogs, costs, audio support, allowed durations
- Cost helpers:
  - `estimateCost(...)`
  - `estimatePartialCost(...)`
  - `qualityModeCostMultiplier(...)`

## 4.2 Polling Hooks

`usePolling.ts`:

- Generic interval hook
- Executes immediately, then on interval
- Stops when `enabled` is false

`useProjectStatus.ts`:

- Polls `GET /api/projects/{id}/status`
- Uses 2s interval normally, 5s during `video_gen`
- Stops polling on terminal statuses

## 5. API Layer (`src/api/client.ts`)

## 5.1 Request Wrapper

- `request<T>()` wraps `fetch`
- Throws `ApiError(status, message)` for non-2xx
- Error message prefers backend `detail`
- Most endpoints are strongly typed via `src/api/types.ts`

Multipart uploads use direct `fetch` with `FormData`:

- Asset image upload
- Manifest video upload
- Scene keyframe upload
- Scene clip upload

## 5.2 Endpoint Surface by Domain

## Generation and Project Lifecycle

- `generateVideo` -> `POST /api/generate`
- `getProjectStatus` -> `GET /api/projects/{id}/status`
- `getProjectDetail` -> `GET /api/projects/{id}`
- `listProjects` -> `GET /api/projects`
- `updateProject` -> `PATCH /api/projects/{id}`
- `deleteProject` -> `DELETE /api/projects/{id}`
- `resumeProject` -> `POST /api/projects/{id}/resume`
- `stopProject` -> `POST /api/projects/{id}/stop`
- `forkProject` -> `POST /api/projects/{id}/fork`
- `editProject` -> `PATCH /api/projects/{id}/edit`
- `getDownloadUrl` -> `GET /api/projects/{id}/download` (URL builder)
- `getMetrics` -> `GET /api/metrics`

## Manifests and Assets

- `listManifests` -> `GET /api/manifests`
- `createManifest` -> `POST /api/manifests`
- `importProjectToManifest` -> `POST /api/manifests/from-project`
- `getManifestDetail` -> `GET /api/manifests/{id}`
- `updateManifest` -> `PUT /api/manifests/{id}`
- `deleteManifest` -> `DELETE /api/manifests/{id}`
- `duplicateManifest` -> `POST /api/manifests/{id}/duplicate`
- `createAsset` -> `POST /api/manifests/{id}/assets`
- `updateAsset` -> `PUT /api/assets/{id}`
- `deleteAsset` -> `DELETE /api/assets/{id}`
- `uploadAssetImage` -> `POST /api/assets/{id}/upload`
- `uploadVideoForManifest` -> `POST /api/manifests/{id}/upload-video`
- `getExtractionProgress` -> `GET /api/manifests/{id}/extraction-progress`
- `processManifest` -> `POST /api/manifests/{id}/process`
- `getProcessingProgress` -> `GET /api/manifests/{id}/progress`
- `reprocessAsset` -> `POST /api/assets/{id}/reprocess`
- `fetchManifestAssets` -> `GET /api/manifests/{id}` (asset extraction helper for forking)

## Quality Mode

- `listCandidates` -> `GET /api/projects/{id}/scenes/{idx}/candidates`
- `selectCandidate` -> `PUT /api/projects/{id}/scenes/{idx}/candidates/{cid}/select`

## Settings

- `getSettings` -> `GET /api/settings`
- `updateSettings` -> `PUT /api/settings`
- `getEnabledModels` -> `GET /api/settings/models`

## PipeSVN / Checkpoints and Regeneration

- `listCheckpoints` -> `GET /api/projects/{id}/checkpoints`
- `getCheckpointDiff` -> `GET /api/projects/{id}/checkpoints/{sha}/diff`
- `createCheckpoint` -> `POST /api/projects/{id}/checkpoints`
- `deleteCheckpoint` -> `DELETE /api/projects/{id}/checkpoints/{sha}`
- `revertToCheckpoint` -> `POST /api/projects/{id}/revert`
- `regenerateScene` -> `POST /api/projects/{id}/scenes/{idx}/regenerate`
- `regenerateSceneText` -> `POST /api/projects/{id}/scenes/{idx}/regenerate-text`
- `generateSceneFields` -> `POST /api/projects/{id}/generate-scene-fields`
- `generateNewScene` -> `POST /api/projects/{id}/generate-new-scene`
- `regenerateProject` -> `POST /api/projects/{id}/regenerate`
- `uploadKeyframe` -> `PUT /api/projects/{id}/scenes/{idx}/keyframes/{position}`
- `uploadClip` -> `PUT /api/projects/{id}/scenes/{idx}/clip`
- `deleteSceneClip` -> `DELETE /api/projects/{id}/scenes/{idx}/clip`
- `deleteSceneKeyframe` -> `DELETE /api/projects/{id}/scenes/{idx}/keyframes/{position}`

## 6. Core Screens and Workflows

## 6.1 Projects List (`ProjectList.tsx`)

Behavior:

- Fetches paginated projects from `listProjects`
- Supports:
  - view mode toggle (`list`/`cards`)
  - status filter chips
  - per-page selection
  - page navigation
- Cards mode requests backend with `view=cards` (for thumbnail-optimized list payload)
- View preference persisted in `localStorage` key: `vidpipe_projects_view`
- Delete is allowed only for terminal statuses

Related:

- `ProjectCard.tsx` renders visual card with thumbnail/status/chips/cost
- `StatusBadge.tsx` centralizes status color/label mapping

## 6.2 Generate Workflow (`GenerateForm.tsx`)

Main responsibilities:

- Collect project prompt, style, aspect ratio, duration, models, audio, manifest, quality mode
- Load enabled/default models from `getEnabledModels`
- Merge enabled Ollama models into text/vision lists
- Validate model selections when enabled lists change
- Snap clip duration to selected video model allowed durations

Generation modes:

- Full pipeline (`run_through = null`)
- Stage-limited generation:
  - `storyboard`
  - `keyframes`
  - `video`

Partial mode behavior:

- If `run_through` is `storyboard` or `keyframes`, user sets scene count directly
- In this mode, video model section is hidden until relevant stage

Costing:

- Uses `estimatePartialCost`
- Applies ComfyUI cost override (`comfyui_cost_per_second`) when relevant
- Quality mode multiplies video-generation cost estimate by candidate count

Submit:

- Calls `generateVideo(...)`
- On success navigates to progress view for returned `project_id`

## 6.3 Progress Screen (`ProgressView.tsx`)

Data flow:

- Lightweight status polling via `useProjectStatus`
- Full detail polling every 3s via `getProjectDetail` for scene/asset progress
- On terminal transition, fetches final detail once more

UI:

- Pipeline stepper
- Latest activity card (latest clip or keyframe)
- Scene grid
- Stop action while active (`stopProject`)
- Resume action for `failed`/`stopped` (`resumeProject`)
- Staged action to jump into detail and continue

## 6.4 Pipeline Stepper (`PipelineStepper.tsx`)

Key detail:

- For `status = staged`, UI maps `run_through` to completed stage boundary:
  - `storyboard` -> `storyboarding`
  - `keyframes` -> `keyframing`
  - `video` -> `video_gen`

This makes staged projects appear complete up to pause point.

## 6.5 Project Detail (`ProjectDetail.tsx`)

Loads full project detail and exposes post-generation operations.

Capabilities:

- Editable title (`updateProject`)
- Prompt expand/collapse and copy helpers
- Model/metadata summary
- Pipeline stepper and error banners
- Final video preview for complete projects
- Scene grid with `SceneCard`
- Copy-all-scenes helper
- Timestamp display

Actions by project state:

- Running: `View Progress`
- Complete: `Download Video`
- Failed/Stopped: `Resume Pipeline`
- Staged:
  - Continue to next stage
  - Run to completion
  - Stage-specific config via `ContinuePanel` for storyboard/keyframes pauses

Terminal-only advanced actions:

- `Edit` (in-place PipeSVN edit mode)
- `Fork` (new project from modified copy)
- `History` (checkpoint list/diff/revert/delete via `CheckpointLog`)

## 6.6 Continue Panel (`ContinuePanel.tsx`)

Used for staged projects resumed from detail screen.

Behavior:

- Determines required config by current staged boundary:
  - from storyboard: image model + vision model
  - from keyframes: video model + audio + clip duration
  - from video: no config required (caller resumes directly)
- Loads enabled models from settings
- Merges Ollama vision models into vision choices
- Calls `resumeProject` with selected overrides and target `run_through`

## 6.7 Scene Card (`SceneCard.tsx`)

Responsibilities:

- Expand/collapse scene details
- Show keyframe previews, clip player, and selected references
- Open image lightbox
- Render prompt chain viewer (`PromptChainViewer`) for base/rewritten/sent prompts
- Copy scene text and prompt fields

Quality mode support:

- On expand, fetch candidates (`listCandidates`) if quality mode is enabled
- Allows manual candidate selection (`selectCandidate`)
- Displays score breakdowns and selected source (`user` or automatic best)

## 6.8 In-place Edit Mode (`EditModeOverlay.tsx`)

This is the largest frontend editing workflow.

Project-level editing:

- Prompt, style, aspect ratio
- Scene length and target scene count
- Text/image/video/vision models and audio
- Optional commit message

Scene-level editing:

- Uses `SceneEditorCard` for each existing and synthetic scene slot
- Supports remove/restore scene
- Supports creating new scenes when scene count is increased

Background operations:

- Regeneration toolbar:
  - stale assets only (`scope=stale`)
  - all assets (`scope=all`)
- Restitch final video (`scope=stitch_only`)
- Uses `usePolling` + `onRefresh` to watch progress
- Tracks baseline `head_sha` and completion by SHA change

Commit behavior:

- If text/field/model changes exist: `editProject(...)`
- If only regeneration happened: `createCheckpoint(...)`
- Uses optimistic concurrency with `expected_sha`

Cancel behavior:

- If regeneration happened during session, attempts `revertToCheckpoint` to baseline SHA before exiting

Schema export/import:

- Exports JSON schema with project settings and effective scene text
- Imports schema to apply project settings and merge scene text edits

## 6.9 Scene Editor (`SceneEditorCard.tsx`)

Per-scene edit and regeneration logic.

Asset controls:

- Per-target regeneration:
  - `start_keyframe`
  - `end_keyframe`
  - `video_clip`
- Optional extra direction appended to prompt overrides
- Manual upload:
  - start/end keyframes
  - video clip
- Manual delete:
  - keyframes
  - clip

Regen completion detection:

- Starts polling every 5s after regen request accepted
- Completion detected by URL change from baseline
- Timeout after 10 minutes (120 polls)

Text controls:

- Edits for 5 scene text fields
- Per-field LLM regeneration via `regenerateSceneText`
- Optional per-field extra context
- Markdown modal editing per field

Empty-slot behavior (`scene.is_empty_slot`):

- `Generate Scene`: full text + asset generation via parent `onGenerateScene` (`generateNewScene`)
- `Text Only`: generate five text fields (`generateSceneFields`)

## 6.10 Markdown Editor Modal (`MarkdownEditorModal.tsx`)

Used by project prompt and scene field editors.

Features:

- Full-screen editor with optional markdown preview pane
- Line-number gutter
- Tab key inserts spaces for indentation
- Copy button
- Optional text regeneration controls with extra context input
- Escape to close

## 6.11 Fork Workflow (`EditForkPanel.tsx`)

Forking creates a new project and can apply selective edits.

Project-level fork edits:

- Prompt/style/aspect/duration/models/audio/vision

Scene-level fork edits:

- Text edits per field
- Scene delete/restore
- Keyframe clear/restore flags

Manifest asset override support:

- Load source manifest assets (`fetchManifestAssets`)
- Mark assets as:
  - inherited
  - edited (reverse prompt changes)
  - removed
- Add new uploads (stored as base64 for fork request)

Request construction:

- `buildForkRequest()` only sends changed fields
- Includes `asset_changes` only when manifest edits exist
- Submit via `forkProject`, then navigate to new project

`EditableSceneCard.tsx` is a focused fork-only scene editor variant.

## 6.12 Manifest Library (`ManifestLibrary.tsx`)

Capabilities:

- List manifests with category filter and sorting
- Duplicate manifest
- Delete manifest with confirmation modal
- Navigate into creator for create/edit/view

## 6.13 Manifest Creator (`ManifestCreator.tsx`)

Multi-stage manifest workflow:

- Stage 1: Draft/upload/extract/import
- Stage 2: Processing progress
- Stage 3: Review and refine assets

Stage transitions depend on manifest status:

- `DRAFT`, `EXTRACTING` -> stage 1
- `PROCESSING` -> stage 2
- `READY`, `ERROR` -> stage 3

Stage 1 features:

- Create/update manifest metadata
- Upload images (`createAsset` + `uploadAssetImage`)
- Upload video for frame extraction (`uploadVideoForManifest`)
- Import assets from project ID (`importProjectToManifest`)

Stage 2 features:

- Poll processing progress (`getProcessingProgress`)
- Show step labels and progress bars

Stage 3 features:

- Asset review with editable fields and quality metadata
- Reprocess single asset (`reprocessAsset`) or all assets (`processManifest`)
- Lightbox preview and copy/download helpers for image/text fields

Supporting components:

- `AssetUploader`: drag-drop/file-select for images/videos with size/type guards
- `AssetEditor`: quick metadata editing for stage-1 asset list
- `ManifestCard`: reusable manifest summary card
- `ManifestSelector`: picker used in generation form (READY manifests only)

## 6.14 Settings (`SettingsPage.tsx`)

Settings page loads from `getSettings()` and saves via `updateSettings()`.

Sections:

- Enabled/default text/image/video model toggles
- GCP project/location fields
- Vertex API key set/clear
- ComfyUI host/API key/cost-per-second fields and clear action
- Ollama mode (local/cloud), endpoint, API key, and model registry

Ollama model handling:

- Add custom models
- Enable/disable per model
- Vision capability toggle per model
- Remove model entries

## 6.15 Dashboard (`Dashboard.tsx`)

Uses `getMetrics()` and renders:

- Summary cards
- Distribution bars and legends for status/style/models/audio/aspect/scenes

## 6.16 Checkpoint History (`CheckpointLog.tsx`)

Functions:

- List checkpoints
- Expand checkpoint to view structured diff
- Revert to checkpoint
- Delete non-head checkpoints

## 7. Data Contracts (Type Layer)

`src/api/types.ts` defines frontend contracts for all backend interactions.

Most important interfaces:

- `GenerateRequest` / `GenerateResponse`
- `ProjectDetail` and `SceneDetail`
- `ForkRequest` / `ForkResponse`
- `EditProjectRequest` / `EditProjectResponse`
- `ManifestDetail`, `ManifestListItem`, `AssetResponse`
- `UserSettingsResponse`, `UserSettingsUpdate`, `EnabledModelsResponse`
- `CheckpointListItem`, `CheckpointDiff`
- `RegenerateSceneRequest`, `RegenerateTextRequest`, `GenerateSceneFieldsRequest`, `GenerateNewSceneRequest`
- `CandidateScore` for quality mode

Notable scene-level fields used heavily by UI:

- Staleness fields: `start_keyframe_staleness`, `end_keyframe_staleness`, `clip_staleness`
- Prompt lineage fields: rewritten prompts and prompt-used fields
- `is_empty_slot` for synthetic scenes in edit mode

## 8. Error Handling and UX Patterns

Patterns used across screens:

- Local `error` state in each component with inline banners
- Retry buttons for failed list/dashboard/settings fetches
- Non-critical API failures often fail silently in optional UI areas
- `confirm(...)` dialogs used for destructive actions in some flows
- Loading flags disable controls during submissions/regeneration

## 9. Polling and Background Behavior Summary

Polling sources:

- `useProjectStatus`: status endpoint (2s fast / 5s slow)
- `ProgressView`: detail refresh every 3s while tracking
- `ManifestCreator`: progress polling every 1.5s for extraction/processing
- `EditModeOverlay`: polling while background regen/stitch/scene-generation is active
- `SceneEditorCard`: target-specific regen polling every 5s until URL change or timeout

Completion signals:

- Pipeline completion from terminal statuses
- Edit-mode regen/stitch completion from `head_sha` change
- Scene asset regen completion from asset URL replacement

## 10. Notable Implementation Constraints and Gaps

- No URL-based routing; state is not deep-linkable and is reset on refresh.
- Several very large components (`EditModeOverlay`, `SceneEditorCard`, `ManifestCreator`, `EditForkPanel`) concentrate extensive logic and UI.
- Multipart upload endpoints bypass the shared `request<T>` helper and have local error handling.
- Manifest category enums are not fully consistent between library filters and creator form options.
- Settings UI includes explicit clear actions for Vertex and ComfyUI keys, but no dedicated "clear Ollama key" button path.

## 11. High-level File Map

Core files:

- `frontend/src/App.tsx`: view-state router
- `frontend/src/components/Layout.tsx`: shell/navigation
- `frontend/src/api/client.ts`: API integration
- `frontend/src/api/types.ts`: API data contracts
- `frontend/src/lib/constants.ts`: shared pipeline/model/cost constants
- `frontend/src/hooks/usePolling.ts`: generic polling
- `frontend/src/hooks/useProjectStatus.ts`: project status polling

Primary workflow components:

- `GenerateForm.tsx`
- `ProgressView.tsx`
- `ProjectList.tsx`
- `ProjectDetail.tsx`
- `EditModeOverlay.tsx`
- `SceneEditorCard.tsx`
- `EditForkPanel.tsx`
- `ManifestLibrary.tsx`
- `ManifestCreator.tsx`
- `SettingsPage.tsx`
- `Dashboard.tsx`
