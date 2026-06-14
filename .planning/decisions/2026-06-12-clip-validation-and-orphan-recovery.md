# ADR: Degenerate-clip validation gate and restart-orphan recovery

**Date:** 2026-06-12
**Status:** Accepted

## Context

Two production failures surfaced after the new ComfyUI FLF2V models shipped:

1. **Corrupted clip stitched into a finished production** (production
   `37375853`, scene `2589fc91`, shot 1). The `wan-2.2-flf2v` job reported
   `success` on ComfyUI Cloud and its server-side QC PNG frames were clean,
   but the MP4 that `SaveVideo` encoded contained a static noise pattern for
   every frame between the two pinned conditioning frames. The downloaded
   bytes were byte-identical to the server's file — this is provider-side
   encode corruption we cannot prevent, only detect. The pipeline had no
   quality gate on the ComfyUI download path, so the garbage clip was marked
   `complete`, stitched into the scene, and rendered into the production
   master.

2. **Productions stranded mid-run by container rebuilds.** Pipelines run as
   in-process asyncio tasks; rebuilding the backend container kills them.
   Scenes were left in `generating_video` with clips stuck in `polling`
   forever (the underlying ComfyUI job had actually succeeded).
   `generating_video` — a transient status set by `generate_keyframes` and
   normally renamed by the orchestrator moments later — was neither in
   `RESUMABLE_STATES` nor in the stop endpoint's `ACTIVE_STATUSES`, so the
   stranded scenes could not be resumed *or* stopped. Three E2E productions
   on 2026-06-11 died this way.

## Decisions

### 1. Frequency-spectrum noise gate on downloaded clips (`services/clip_validation.py`)

Every ComfyUI clip download (main pipeline and `_regenerate_clip`) is checked
before being accepted:

- **Noise signature** — ratio of mid-frequency band energy (between 1/4 and
  1/16 scale) to low-frequency structure, per sampled frame. Natural frames
  follow a ~1/f spectrum (measured ≤ 0.30 on real clips, including a
  near-black lens-occlusion frame); the observed corruption scores ≥ 3.8.
- **Keyframe anchors** — first frame must correlate with the start keyframe;
  last frame with the end keyframe when FLF2V conditioning was used
  (Pearson on downscaled grayscale; catches wrong-video downloads).
- **Temporal continuity is diagnostic only.** A hard continuity gate was
  prototyped and rejected: a *good* LTX clip contained a legitimate
  near-discontinuity (object sweeping across the lens) that would have been
  a false positive.

On failure the shot is resubmitted with a **perturbed seed**
(`seed + attempt * 1000003`) — same seed would risk ComfyUI's node cache
serving the identical corrupted encode — up to
`pipeline.clip_corrupt_retry_max` (default 2) times, then the scene fails
loudly with the validation detail in `error_message`.

Config: `pipeline.clip_validation_enabled` (default true).

### 2. Startup auto-resume of orphaned scenes (`orchestrator/recovery.py`)

On startup the API spawns a background task that finds scenes in in-flight
statuses (`pending`, `storyboarding`, `keyframing`, `generating_video`,
`video_gen`, `stitching`) and re-runs `run_pipeline` for each, sequentially.
Every stage checkpoints to the DB, so resume is idempotent: completed stages
are skipped and in-flight ComfyUI jobs are re-polled via the persisted
`comfyui:` operation id. Config: `pipeline.auto_resume_on_startup`
(default true — interrupted runs were already paid for and user-requested).

`generating_video` was added to `RESUMABLE_STATES`/`ACTIVE_STATUSES`, maps to
`video_gen` in `get_resume_step`, and `run_pipeline` normalizes it at entry
(previously a scene entering in that status matched no step block and
no-opped straight to "completed successfully").

## Consequences

- Corrupted provider output can no longer silently reach a stitched scene;
  worst case is a clearly-reported scene failure after N retries.
- Container rebuilds mid-run self-heal on next startup instead of stranding
  productions.
- Validation thresholds were calibrated against the real corrupted clip and
  seven real good clips (WAN + LTX); unit tests pin the behavior with
  synthetic fixtures (`tests/test_clip_validation.py`,
  `tests/test_orphan_recovery.py`).
- Veo path is unchanged (no corruption observed there; it has its own
  quality-mode scoring).
