# vidpipe — Frontend

React SPA for the video pipeline. Provides a UI to submit prompts, monitor pipeline progress in real-time, and view/download generated videos.

## Setup

```bash
cd frontend
npm install
```

## Running

```bash
# Development (with hot reload)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

The dev server runs on `http://localhost:5173` and proxies `/api` requests to the backend on port 8000 (configured in `vite.config.ts`).

## Tech Stack

- **React 19** with TypeScript
- **Vite 7** — dev server and bundler
- **Tailwind CSS 4** — styling
- **ESLint** — linting

## Project Structure

```
src/
├── main.tsx              # Entry point
├── App.tsx               # Root component and routing
├── index.css             # Global styles / Tailwind imports
├── api/
│   ├── client.ts         # API client (fetch wrapper)
│   └── types.ts          # API response types
├── components/
│   ├── GenerateForm.tsx   # Prompt input form
│   ├── Layout.tsx         # Page layout shell
│   ├── PipelineStepper.tsx# Pipeline stage progress indicator
│   ├── ProgressView.tsx   # Real-time generation progress
│   ├── ProjectDetail.tsx  # Single project view with scenes
│   ├── ProjectList.tsx    # All projects listing
│   ├── SceneCard.tsx      # Individual scene display
│   └── StatusBadge.tsx    # Status indicator chip
├── hooks/
│   ├── usePolling.ts      # Generic polling hook
│   └── useProjectStatus.ts# Project status polling
└── lib/
    └── constants.ts       # Shared constants
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server with HMR |
| `npm run build` | Type-check and build for production |
| `npm run lint` | Run ESLint |
| `npm run preview` | Serve production build locally |

## Production

Running `npm run build` outputs static files to `dist/`. The backend serves these automatically when the `dist/` directory exists — no separate web server needed.
