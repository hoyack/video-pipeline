# User Instructions

This guide covers general usage of the vidpipe app: setup, running, and day-to-day workflows in the UI.

## 1. What This App Does

vidpipe turns a text prompt into a multi-scene video by running:

1. Storyboarding
2. Keyframe generation
3. Video clip generation
4. Final stitching

You can pause/resume generation, continue from stage boundaries, edit completed projects, fork variants, and reuse assets via manifests.

## 2. Prerequisites

Before running:

- Python 3.11+
- Node.js 18+
- `ffmpeg` installed
- Google Cloud service account with Vertex AI access
- `.env` configured at repo root

Minimal `.env` values:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
VIDPIPE_GOOGLE_CLOUD__PROJECT_ID=your-gcp-project-id
```

## 3. First-Time Setup

From repo root:

```bash
# Backend install
pip install -e backend/

# Frontend install
cd frontend
npm install
npm run build
cd ..
```

## 4. Start the App

Run backend server (also serves built frontend):

```bash
uvicorn vidpipe.api.app:app --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000`

Optional frontend hot-reload mode:

```bash
cd frontend
npm run dev
```

Then open:

- `http://localhost:5173` (proxies `/api` to backend on `:8000`)

## 5. Basic Workflow

## 5.1 Create a Project

1. Go to `Projects` and click `+ New`.
2. Fill prompt, style, aspect ratio, duration, and model choices.
3. Optionally choose:
   - `Generate Through` (stage-limited run)
   - `Asset Manifest`
   - `Quality Mode` (multiple candidates per scene)
4. Click `Generate Video`.

## 5.2 Monitor Progress

In `Progress` view you can:

- Watch current stage and latest scene activity
- Stop active pipeline
- Resume failed/stopped runs
- Jump to project details

## 5.3 Continue Staged Projects

If you generated through `storyboard`, `keyframes`, or `video`, project status becomes `staged`.

From Project Detail you can:

- Continue to next stage
- Run to completion
- Configure needed model settings before continuing

## 5.4 Download Final Output

When project status is `complete`:

- Use `Download Video` from Detail or Progress page

## 6. Project Management

## 6.1 Projects View

In `Projects` you can:

- Switch between list and card view
- Filter by status
- Change pagination size
- Delete terminal projects (`complete`, `failed`, `stopped`, `staged`)

## 6.2 Edit In Place

For terminal projects, use `Edit` in Project Detail to:

- Modify prompt, style, models, scene count, and scene fields
- Regenerate stale/all assets
- Re-stitch final output
- Commit changes as a checkpoint

## 6.3 Fork Variants

Use `Fork` to create a new project from an existing one with optional edits:

- Project-level setting changes
- Scene text/deletions
- Keyframe reset flags
- Manifest asset overrides and additional uploads

## 6.4 Version History

Use `History` in Project Detail to:

- View checkpoints
- Inspect structured diffs
- Revert to a checkpoint
- Delete old checkpoints

## 7. Manifest Workflow

Go to `Manifests` to manage reusable asset collections.

## 7.1 Create and Process a Manifest

1. Click `+ New Manifest`.
2. Add metadata (name, category, tags).
3. Add assets by:
   - Uploading images
   - Uploading a video for frame extraction
   - Importing from an existing project ID
4. Click `Process` to run manifest analysis.
5. Review and refine asset tags/descriptions/prompts.

## 7.2 Use Manifest in Generation

In Generate form:

- Open `Asset Manifest`
- Select a `READY` manifest
- Start generation

## 8. Settings

Go to `Settings` to control:

- Enabled text/image/video models
- Default models
- GCP project/location
- Vertex API key
- ComfyUI host/key/cost override
- Ollama mode/endpoint/model list

Changes apply to model selectors in generation/edit/fork flows.

## 9. Dashboard

Go to `Dashboard` for aggregate metrics:

- Project totals
- Estimated cost
- Success rate
- Model/style/status distributions

## 10. Status Meanings

Common project statuses:

- `pending`: queued
- `storyboarding`: building scenes
- `keyframing`: generating keyframes
- `video_gen`: generating clips
- `stitching`: building final video
- `complete`: finished successfully
- `failed`: error occurred
- `stopped`: user stopped run
- `staged`: intentionally paused at selected stage

## 11. Troubleshooting

If the app does not work as expected:

1. Confirm backend is running on `:8000`.
2. Confirm `.env` credentials and GCP project are valid.
3. Confirm `ffmpeg` is installed and accessible.
4. If using frontend dev server, ensure it runs on `:5173`.
5. Open browser dev tools and check `/api` request failures.
6. For interrupted runs, use `Resume Pipeline` or `Continue`.
7. For stale final output after edits, use `Re-stitch`.

## 12. Useful CLI Commands

These can be used alongside the UI:

```bash
python -m vidpipe list
python -m vidpipe status <project-id>
python -m vidpipe resume <project-id>
python -m vidpipe stitch <project-id> --crossfade 0.5
```

