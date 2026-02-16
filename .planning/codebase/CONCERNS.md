# Codebase Concerns

**Analysis Date:** 2026-02-16

## Tech Debt

**Status name mismatch in keyframe generation:**
- Issue: `generate_keyframes()` sets project status to `"generating_video"` but the pipeline orchestrator expects `"video_gen"`. The pipeline corrects this at `vidpipe/orchestrator/pipeline.py:192-194`.
- Files: `vidpipe/orchestrator/pipeline.py`, `vidpipe/pipeline/keyframes.py`
- Impact: Inconsistent status naming across pipeline steps requires special handling. Future changes to state machine are fragile.
- Fix approach: Standardize on `"video_gen"` status. Update `generate_keyframes()` to use correct status name.

**Broad exception handling in background tasks:**
- Issue: `run_pipeline_background()` at `vidpipe/api/routes.py:355` catches all exceptions with `except Exception as e:`, logging and swallowing them. Pipeline errors are "already persisted" but the broad catch masks unexpected errors.
- Files: `vidpipe/api/routes.py:346-357`
- Impact: Silent failures of unexpected exceptions (not caught by orchestrator). Difficult to debug issues that occur outside the pipeline state machine.
- Fix approach: Catch specific exception types. Only swallow `PipelineStopped`. Re-raise or alert on unexpected exceptions.

**Routes module size and complexity:**
- Issue: `vidpipe/api/routes.py` is 1317 lines, mixing schema definitions, validation logic, cost estimation, file serving, and endpoint handlers.
- Files: `vidpipe/api/routes.py`
- Impact: Difficult to maintain, test, and navigate. Cost estimation logic is duplicated conceptually between `_estimate_project_cost()` and the frontend. Single large file harder to review.
- Fix approach: Extract schemas to dedicated module, move cost estimation to shared service, split endpoints into logical route groups.

## Known Bugs

**Scene expansion index offset bug:**
- Issue: In `_generate_expansion_if_needed()` at `vidpipe/orchestrator/pipeline.py:74`, the new scene's `scene_index` is set to `existing_count` instead of incrementing from the last existing scene. This causes duplicate indices if expansion happens mid-pipeline.
- Files: `vidpipe/orchestrator/pipeline.py:71-82`
- Impact: Fork delete-then-expand creates scenes with incorrect ordering. Stitcher or UI may display scenes in wrong order.
- Trigger: Create project → fork with delete → expand → resume pipeline
- Workaround: Ensure all deletions preserve sequential indices; don't use expand feature

**Cost estimation doesn't account for partial projects:**
- Issue: `_estimate_project_cost()` at `vidpipe/api/routes.py:127-198` has complex logic for "incomplete" projects using `billed_veo_submissions`, but this field is only set when Veo submissions complete. Failed submissions mid-pipeline don't accumulate billed_veo_submissions, causing inaccurate cost.
- Files: `vidpipe/api/routes.py:127-198`
- Impact: Cost estimates are inaccurate for failed/incomplete projects. Users see incorrect billing information.
- Trigger: Start generation → fail at video_gen step → check cost estimate
- Workaround: None; estimates will be wrong until step completes

## Security Considerations

**API CORS hardcoded to Vite dev server:**
- Risk: `vidpipe/api/app.py:52-57` allows requests from `http://localhost:5173` (Vite dev server). In production, this is a CORS vulnerability.
- Files: `vidpipe/api/app.py:52-57`
- Current mitigation: Only affects development environment if not changed.
- Recommendations: Load allowed origins from config. Validate environment at startup. Warn if production build uses dev origins.

**Database password in connection string:**
- Risk: SQLite doesn't use passwords, but if ever migrated to PostgreSQL/MySQL, connection string with password could be logged or exposed.
- Files: `vidpipe/config.py`, `vidpipe/db/engine.py`
- Current mitigation: Connection string loaded from `config.yaml` (not committed). `.env` file exists (not read by code, only by Pydantic settings).
- Recommendations: Never log `database_url`. Use separate config for database credentials. Document secrets handling in README.

**Path traversal protection in FileManager:**
- Risk: `vidpipe/services/file_manager.py:61` checks `is_relative_to()` but only after calling `.resolve()`. Could be exploited with symlinks.
- Files: `vidpipe/services/file_manager.py:58-62`
- Current mitigation: Check is present but incomplete.
- Recommendations: Validate path components don't contain `..`. Use `pathlib.PurePath` comparison instead of `.resolve()`. Add unit tests for traversal attempts.

**RAI filtering responses exposed in API:**
- Risk: `vidpipe/api/routes.py` serves keyframes and clips directly without checking content policy flags. User could serve RAI-filtered clip that was marked `"rai_filtered"`.
- Files: `vidpipe/api/routes.py:1130-1170`, `vidpipe/pipeline/video_gen.py:263-270`
- Current mitigation: VideoClip status distinguishes `"rai_filtered"` but endpoints don't check it.
- Recommendations: Add check in file serving endpoints. Return 403 Forbidden for `rai_filtered` clips. Document in API spec.

## Performance Bottlenecks

**Video polling blocks entire scene:**
- Problem: `_poll_video_operation()` at `vidpipe/pipeline/video_gen.py:237-327` polls one clip sequentially. With 100 scenes, each with 60+ second polls, total time is ~2 hours minimum.
- Files: `vidpipe/pipeline/video_gen.py:237-327`, `vidpipe/pipeline/video_gen.py:333-365`
- Cause: Scenes are processed serially in a loop. Polling waits synchronously for each operation.
- Improvement path: Use asyncio task concurrency. Poll all operations in parallel with `asyncio.gather()`. Respect `video_gen_concurrency` setting.

**Database queries in loop during project detail:**
- Problem: `get_project_detail()` at `vidpipe/api/routes.py:459-530` executes N+1 queries: one per scene for keyframes, one per scene for clips. With 20 scenes, that's 40+ queries.
- Files: `vidpipe/api/routes.py:475-499`
- Cause: No eager loading or batch queries. Loop executes query for each scene.
- Improvement path: Use SQLAlchemy `selectinload()` or `joinedload()` to batch fetch keyframes and clips. Single query with join.

**File reads for every keyframe/clip serve:**
- Problem: `vidpipe/api/routes.py:1130-1170` reads entire file from disk on every request. No caching, no streaming.
- Files: `vidpipe/api/routes.py:1130-1170`
- Cause: `FileResponse(Path(...).read_bytes())` reads full file into memory for each request.
- Improvement path: Return `FileResponse` with path only (FastAPI streams). Add ETag/cache headers. Implement CDN for static assets.

**Storyboard generation temperature reduction on retry:**
- Problem: `generate_storyboard()` at `vidpipe/pipeline/storyboard.py:116-138` retries with lower temperature on JSON parse failure, but doesn't validate that lower temperature fixes the issue. Could retry forever with useless temperatures.
- Files: `vidpipe/pipeline/storyboard.py:120-143`
- Cause: No check whether lower temperature actually helps JSON parsing.
- Improvement path: Detect specific parse failures. Add circuit breaker. Fail fast if temperature near 0. Log which retry succeeded.

## Fragile Areas

**Fork with scene edits:**
- Files: `vidpipe/api/routes.py:810-1070`
- Why fragile: Complex branching logic. Multiple conditional paths for deleting, editing, expanding scenes. If expansion code path is exercised with edits, scene indices can become inconsistent (see Known Bugs). Test coverage for fork permutations is likely incomplete.
- Safe modification: Add comprehensive tests for all edit+delete+expand combinations. Verify scene indices are sequential post-operation. Add validation query before pipeline resumes.
- Test coverage: Unknown (no test files found)

**Cost estimation logic:**
- Files: `vidpipe/api/routes.py:127-198`, implicitly used in `vidpipe/pipeline/video_gen.py` (submission counting)
- Why fragile: Logic is split across multiple places. Cost estimation has special cases for "complete" vs "pending" vs "partial". `billed_veo_submissions` is set in video_gen but used in routes. If video_gen changes how it counts, cost breaks. No validation that estimated cost matches actual cost.
- Safe modification: Extract cost estimation to dedicated service. Define cost model as interface. Add cost validation endpoint that compares estimate vs actual. Test with mock clip counts.
- Test coverage: No tests found

**Content policy escalation in video generation:**
- Files: `vidpipe/pipeline/video_gen.py:167-532`
- Why fragile: Three escalation levels with increasing intervention (safety prefix → keyframe regen → strong prefix). If any level fails in unexpected way, graceful degradation is unclear. End keyframe regen can fail silently (`new_bytes` is None), leaving stale keyframe. Escalation doesn't check if keyframe regen actually improved safety.
- Safe modification: Add logging at each escalation level. Return result enum with failure reason. Test keyframe regen independently. Add unit test for escalation path with mocked responses.
- Test coverage: No tests found

**Retry logic with tenacity decorators:**
- Files: `vidpipe/pipeline/video_gen.py:117-163`, `vidpipe/pipeline/storyboard.py:120-143`
- Why fragile: Multiple retry decorators with different strategies. `_submit_video_job()` retries transient errors. Storyboard retries JSON errors. No unified retry policy. Hard to understand which errors are retriable across modules.
- Safe modification: Create centralized retry policy. Define error classification consistently. Add timeout guards. Document retry guarantees.
- Test coverage: No tests found

## Scaling Limits

**SQLite database for concurrent requests:**
- Current capacity: Single writer at a time. WAL mode helps but still bottleneck for >10 concurrent projects.
- Limit: PRAGMA busy_timeout=5000ms. If lock contention exceeds 5s, requests fail with "database locked".
- Scaling path: Migrate to PostgreSQL with connection pooling. Move cost calculations to read-only replicas. Batch project status updates.

**Temporary file storage in single directory:**
- Current capacity: `/tmp/video-pipeline/` grows unbounded. Each project generates 100+ MB. No cleanup.
- Limit: Filesystem runs out of space. No garbage collection. Project cleanup is manual.
- Scaling path: Implement TTL-based cleanup. Move outputs to cloud storage (GCS). Archive old projects. Monitor disk usage. Set quota.

**Veo API rate limits and quotas:**
- Current capacity: Memory holds one operation per scene while polling. No backoff. Cost uncontrolled.
- Limit: Google's Veo quota limits (not documented in code). Cost scaling with scene count.
- Scaling path: Implement request budget per project. Add cost warnings. Cap scenes per project. Queue projects with priority. Monitor quota usage.

**In-memory session sharing across async boundaries:**
- Current capacity: Each route creates fresh `async_session()`. No connection pool tuning.
- Limit: SQLAlchemy's session factory can exhaust connections if requests pile up.
- Scaling path: Configure pool size based on expected concurrency. Add connection monitoring. Use FastAPI dependency injection for session management (already partially done).

## Dependencies at Risk

**Tenacity retry library usage:**
- Risk: Custom retry decorators (`_submit_video_job`, `generate_storyboard`) assume tenacity's behavior. If dependency is dropped/changed, retry logic breaks silently.
- Impact: Retries may stop working without error.
- Migration plan: Abstract retry logic behind custom decorator. Replace tenacity with built-in retry using asyncio. Test retry guarantees independently.

**Pydantic v2 settings and validators:**
- Risk: Code uses Pydantic v2 syntax (`field_validator`, `model_validate_json`). Downgrading to v1 breaks code.
- Impact: Dependency lock-in. Settings loading and schema validation fail.
- Migration plan: Version pins are correct (`pydantic>=2.0`). But test migration path to v3 when available.

**Google Vertex AI SDK (`google-genai`):**
- Risk: SDK is relatively new. API surface could change. No fallback if service goes down.
- Impact: Entire pipeline depends on this. No graceful degradation. Cost tied to Google's pricing.
- Migration plan: Consider alternative image/video generation APIs as future phases. Implement service abstraction layer.

## Missing Critical Features

**No progress/completion callbacks for long-running operations:**
- Problem: Pipeline takes 10+ minutes. Frontend polls status endpoint every 2s. User has no visibility into which step is running.
- Blocks: Can't show step-by-step progress. Can't estimate remaining time. User sees "keyframing" for 5 minutes with no breakdown.
- Recommendation: Add WebSocket or Server-Sent Events (SSE) for live progress. Emit events from each pipeline step.

**No pipeline stop with graceful shutdown:**
- Problem: User can request stop (`project.status = "stopped"`) but in-flight operations (Veo submissions, image generation) continue. Clips are generated but not used, incurring cost.
- Blocks: Can't cancel running jobs. No cost control. No request cleanup.
- Recommendation: Track operation IDs. Implement cancel endpoint that kills Veo operations before they complete.

**No cost limit or budget enforcement:**
- Problem: User can start 100 projects simultaneously. No quota. Cost scales unbounded.
- Blocks: No rate limiting. No budget alerts. Runaway costs possible.
- Recommendation: Add per-user budget. Implement cost checking before pipeline step. Queue projects if budget exceeded.

**No explicit test suite:**
- Problem: No test files found (`test_*.py`, `*_test.py`). No test configuration.
- Blocks: Refactoring is dangerous. Behavior is undocumented. Regressions are undetected.
- Recommendation: Create comprehensive test suite. Unit test utils, services. Integration test pipeline steps. E2E test API endpoints.

## Test Coverage Gaps

**Pipeline orchestration:**
- What's not tested: State machine transitions, resume logic, idempotent step execution, error propagation.
- Files: `vidpipe/orchestrator/pipeline.py`, `vidpipe/orchestrator/state.py`
- Risk: Changes to resume logic could break idempotence. Scene expansion could silently corrupt indices.
- Priority: High (core functionality)

**Video generation with escalation:**
- What's not tested: Content policy escalation, keyframe regen, transient retry, polling timeout.
- Files: `vidpipe/pipeline/video_gen.py`
- Risk: Safety remediations may fail silently. Escalation path not exercised.
- Priority: High (expensive operation)

**Fork and scene editing:**
- What's not tested: Combinations of delete/edit/expand. Scene index consistency. Keyframe/clip inheritance.
- Files: `vidpipe/api/routes.py:810-1070`
- Risk: Complex branching logic not validated. Known bugs likely exist.
- Priority: High (user-facing feature)

**Database queries:**
- What's not tested: N+1 query issues in `get_project_detail()`. Query performance with large scene counts.
- Files: `vidpipe/api/routes.py:459-530`
- Risk: Scaling to 100+ scenes will be slow. Performance regression undetected.
- Priority: Medium (scaling concern)

**Frontend component state and API integration:**
- What's not tested: Form submission, error handling, polling logic, fork UI.
- Files: `frontend/src/components/*.tsx`
- Risk: UI could become unresponsive on API failure. State corruption on race conditions.
- Priority: Medium (UX quality)

**File serving and access control:**
- What's not tested: Whether RAI-filtered clips are served. Path traversal defenses. Concurrent file access.
- Files: `vidpipe/api/routes.py:1090-1170`
- Risk: Security vulnerability. Concurrent deletes could race with serves.
- Priority: High (security)

---

*Concerns audit: 2026-02-16*
