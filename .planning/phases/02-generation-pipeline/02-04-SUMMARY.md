---
phase: 02-generation-pipeline
plan: 04
subsystem: pipeline
tags: [ffmpeg, video-stitching, concat-demuxer, xfade, crossfade, subprocess]

# Dependency graph
requires:
  - phase: 02-03
    provides: Video clips saved as MP4 files with VideoClip.local_path populated
  - phase: 01-03
    provides: FileManager for output path management
  - phase: 01-02
    provides: Configuration for crossfade_seconds and clip_duration settings
provides:
  - Video stitcher using ffmpeg concat demuxer for hard cuts
  - Crossfade transitions using ffmpeg xfade filter
  - Startup validation for ffmpeg availability with clear error messages
  - Final video output at tmp/{project_id}/output/final.mp4
affects: [pipeline-orchestration, deployment]

# Tech tracking
tech-stack:
  added: [ffmpeg system dependency, asyncio.to_thread for subprocess calls]
  patterns: [startup dependency validation, subprocess wrapper with async, concat list file pattern]

key-files:
  created:
    - vidpipe/pipeline/stitcher.py
  modified:
    - vidpipe/__init__.py

key-decisions:
  - "Used concat demuxer with -safe 0 flag for absolute path support in concat list"
  - "Implemented xfade filter with variable frame rate (-vsync vfr) for crossfade transitions"
  - "Stream copy (-c copy) for concat demuxer to preserve audio quality without re-encoding"
  - "Wrapped subprocess.run() in asyncio.to_thread() to prevent event loop blocking"
  - "Validate ffmpeg at startup rather than during pipeline execution for fail-fast error handling"

patterns-established:
  - "Pattern: System dependency validation in __init__.py with RuntimeError and installation instructions"
  - "Pattern: Concat list file with absolute paths written to output directory, cleaned up after use"
  - "Pattern: Single clip edge case handled with stream copy instead of crossfade"
  - "Pattern: Missing clip files detected before ffmpeg execution with clear error message"

# Metrics
duration: 1.4min
completed: 2026-02-14
---

# Phase 02 Plan 04: Video Stitching Summary

**ffmpeg-based video stitching with concat demuxer for hard cuts, xfade filter for crossfade transitions, and startup validation ensuring ffmpeg availability**

## Performance

- **Duration:** 1.4 min (83 seconds)
- **Started:** 2026-02-14T22:42:27Z
- **Completed:** 2026-02-14T22:43:50Z
- **Tasks:** 2
- **Files created:** 1
- **Files modified:** 1

## Accomplishments
- ffmpeg startup validation function that checks for ffmpeg availability and logs version (STCH-05)
- Concat demuxer implementation for hard cuts using stream copy to preserve audio quality (STCH-01, STCH-03)
- xfade filter implementation for smooth crossfade transitions with configurable duration (STCH-02)
- Final video output saved to tmp/{project_id}/output/final.mp4 (STCH-04)
- Absolute path support in concat list using -safe 0 flag to avoid path errors
- Async subprocess execution using asyncio.to_thread() to prevent event loop blocking
- Edge case handling: no clips, single clip, missing files
- Project status updated to "complete" on success or "failed" with error message on failure

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement ffmpeg startup validation** - `4d9f227` (feat)
2. **Task 2: Implement video stitching with concat demuxer and xfade support** - `e39e86a` (feat)

## Files Created/Modified
- `vidpipe/__init__.py` - Added validate_dependencies() function for ffmpeg startup validation with RuntimeError and installation instructions
- `vidpipe/pipeline/stitcher.py` - Video stitcher with concat demuxer (hard cuts) and xfade filter (crossfades), async subprocess execution, edge case handling

## Decisions Made

**Concat demuxer with -safe 0 flag:** ffmpeg concat demuxer requires -safe 0 flag to accept absolute paths in concat list file. Without this flag, ffmpeg rejects absolute paths as security risk, causing concatenation to fail. Using absolute paths (via Path.resolve()) is more reliable than relative paths.

**Stream copy for concat demuxer:** When using concat demuxer for hard cuts (crossfade_seconds=0.0), use -c copy to preserve original video and audio quality without re-encoding. This is fast and lossless. xfade filter requires re-encoding, so stream copy only applies to concat demuxer mode.

**xfade filter with variable frame rate:** When using xfade filter for crossfades, pass -vsync vfr flag to enable variable frame rate handling. This ensures correct timing during transitions. Crossfade offset calculation: (clip_duration * i) - (crossfade_duration * i) positions transition at overlap point.

**asyncio.to_thread() wrapper:** Subprocess calls are blocking I/O operations. Running subprocess.run() directly in async function would block the entire event loop, preventing other async operations. Wrapping in asyncio.to_thread() moves subprocess execution to thread pool, keeping event loop responsive.

**Startup validation pattern:** Validating ffmpeg availability during application startup (via validate_dependencies()) provides fail-fast behavior with clear error message before any generation work begins. Alternative would be to discover missing ffmpeg during pipeline execution, wasting time and partial work.

**Edge case handling:** Single clip case bypasses xfade filter (no transitions needed) and uses stream copy. No clips case marks project as failed with clear error message. Missing clip files detected before ffmpeg execution to provide clearer error than ffmpeg's "no such file" message.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - ffmpeg was already installed per user setup instructions, validation passed on first execution.

## User Setup Required

**ffmpeg system dependency required.** Installation verified during plan execution:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

Verification command: `ffmpeg -version`

Application will raise RuntimeError with installation instructions if ffmpeg is missing.

## Next Phase Readiness

**Generation pipeline complete.** All components implemented:
- Storyboard generation (02-01) ✓
- Keyframe generation (02-02) ✓
- Video generation (02-03) ✓
- Video stitching (02-04) ✓

Ready for:
- Phase 03: Pipeline orchestration to connect all components
- End-to-end testing with real Vertex AI calls
- Error recovery and resume testing
- Cost estimation and quota management

**Blockers:**
- Vertex AI authentication requires GOOGLE_APPLICATION_CREDENTIALS environment variable for production
- Rate limiting on free tier may require quota increase or billing enablement
- ffmpeg must be installed on deployment environment

## Self-Check: PASSED

All files verified:
- ✓ vidpipe/__init__.py exists
- ✓ vidpipe/pipeline/stitcher.py exists

All commits verified:
- ✓ 4d9f227 (Task 1: ffmpeg startup validation)
- ✓ e39e86a (Task 2: video stitcher implementation)

---
*Phase: 02-generation-pipeline*
*Completed: 2026-02-14*
