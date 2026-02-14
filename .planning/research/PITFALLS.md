# Pitfalls Research

**Domain:** AI Video Generation Pipeline
**Researched:** 2026-02-14
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Naive Polling Causes Cost Explosions and API Bans

**What goes wrong:**
Developers implement tight polling loops (checking every few seconds) for long-running Veo operations, burning through API quota, incurring unnecessary costs, and potentially triggering rate limits. With Veo video generation taking 30-120+ seconds per clip, constant polling wastes resources.

**Why it happens:**
The desire for "real-time" feedback conflicts with the reality that video generation is inherently slow. Developers unfamiliar with long-running operations patterns default to simple polling loops without backoff strategies.

**How to avoid:**
Implement exponential backoff starting at 10-15 seconds and doubling up to 60 seconds maximum. Store operation state in SQLite with next_poll_time to prevent duplicate requests. For Veo specifically, initial poll should wait at least 30 seconds since generation rarely completes faster.

**Warning signs:**
- API quota warnings or 429 errors appearing
- Database shows hundreds of status checks for single operation
- Network logs show identical GET requests every 2-5 seconds
- Cost dashboard shows unexpected API call volume

**Phase to address:**
Phase 1 (Core Pipeline) - Build polling logic with backoff from the start, not as a later optimization.

---

### Pitfall 2: No Idempotency = Duplicate Video Charges on Retry

**What goes wrong:**
When a request fails (network timeout, transient error), retry logic submits a brand new video generation request instead of checking if the original operation is still running. At $2-3 per video, duplicate submissions quickly drain budgets.

**Why it happens:**
Developers treat Veo API like synchronous requests where retry = resend. They don't realize that the operation may have been accepted even if the initial response was lost, and the operation ID is the only way to resume tracking.

**How to avoid:**
Store operation metadata (operation_id, prompt, status) in SQLite immediately after submission, keyed by a content hash of the request parameters. Before submitting new requests, check if an in-progress operation exists for the same prompt. Use operation.get() to verify status before creating duplicates.

**Warning signs:**
- Cost spikes showing 2-3x expected video generation charges
- Database contains multiple "pending" operations with identical prompts
- Logs show "Request timeout" followed by new submission without operation check
- User reports getting multiple versions of the same scene

**Phase to address:**
Phase 1 (Core Pipeline) - Critical for cost control from day one.

---

### Pitfall 3: Synchronous FFmpeg Stitching Blocks the Entire Pipeline

**What goes wrong:**
When FFmpeg stitching runs in the main thread/process without async handling, the entire pipeline freezes for 10-30+ seconds while waiting for video encoding. With 5 scenes, that's 50-150 seconds of complete pipeline blockage where no other work can happen.

**Why it happens:**
FFmpeg is a subprocess call that appears simple (`ffmpeg -i ...`) but is CPU-intensive. Developers don't realize that `subprocess.run()` blocks Python entirely until encoding completes.

**How to avoid:**
Run FFmpeg operations in background threads or processes using `concurrent.futures.ThreadPoolExecutor` or asyncio subprocess. Store stitching jobs in SQLite with a "stitching" status so crash recovery knows to resume. Allow the pipeline to continue downloading/processing other clips while stitching runs.

**Warning signs:**
- CLI appears frozen/unresponsive during stitching phase
- CPU usage shows extended periods where only FFmpeg is running
- Pipeline throughput degrades linearly with scene count
- Logs show large gaps (30+ seconds) between log entries

**Phase to address:**
Phase 2 (Stitching) - Async stitching must be architected upfront, not retrofitted.

---

### Pitfall 4: Not Handling RAI Content Rejections = Lost Scenes and Budget

**What goes wrong:**
Veo's Responsible AI filters reject clips containing prohibited content (violence, certain people, sensitive topics), returning errors instead of videos. Without proper detection and retry logic, the pipeline either crashes or produces incomplete multi-scene videos with gaps, while still charging for the rejected generation attempts.

**Why it happens:**
Developers focus on the happy path where Veo succeeds, treating content filtering as an edge case. They don't realize RAI rejections are common (10-30% of requests depending on prompts) and need systematic handling.

**How to avoid:**
Parse Veo error responses to distinguish RAI rejections from transient failures. Store rejection metadata in SQLite (scene_id, rejection_reason, original_prompt). Implement automatic prompt rewriting: soften language, remove specific references, or fallback to generic descriptions. Set a max retry limit (3 attempts) before marking scene as "ungenerable" and notifying user.

**Warning signs:**
- Final videos missing random scenes (1, 3, 5 present but 2, 4 missing)
- Logs show "Request failed" without retry attempts
- Database shows operations stuck in "failed" status with no follow-up
- Cost matches full scene count despite incomplete output
- User prompts contain potentially sensitive terms (weapons, public figures, medical procedures)

**Phase to address:**
Phase 1 (Core Pipeline) - RAI handling is table-stakes for production reliability.

---

### Pitfall 5: SQLite Database Corruption from Concurrent Writes During Crash

**What goes wrong:**
When the pipeline crashes mid-operation (Ctrl+C, system reboot, OOM), concurrent status updates to SQLite can corrupt the database file, making all state recovery impossible. Entire project state is lost along with tracking for already-paid-for Veo operations.

**Why it happens:**
SQLite's single-writer model means write transactions lock the database. If a crash occurs while a write is in progress, the journal/WAL may be left in an inconsistent state. Default SQLite settings don't prioritize crash safety.

**How to avoid:**
Enable WAL mode (`PRAGMA journal_mode=WAL`) for better concurrent access and crash recovery. Set `PRAGMA synchronous=FULL` to ensure commits wait for disk sync. Wrap all state updates in explicit transactions with immediate rollback on error. Implement checkpoint logic to periodically sync WAL to main database. Keep transactions small (single-row updates) to minimize lock time.

**Warning signs:**
- "Database is locked" errors appearing intermittently
- After crash, database file shows corruption messages on next run
- SQLite CLI reports "malformed database" or "disk I/O error"
- State recovery finds empty or partially written records
- Operations that definitely completed show as "pending" after restart

**Phase to address:**
Phase 1 (Core Pipeline) - Database integrity must be guaranteed before any crash recovery.

---

### Pitfall 6: Ignoring FFmpeg Format Mismatches = Silent Frame Loss

**What goes wrong:**
When stitching multiple Veo-generated clips, FFmpeg silently drops frames or creates visual glitches if clips have mismatched codecs, frame rates, or resolutions. The pipeline "succeeds" but the final video has jarring jumps, missing content, or sync issues.

**Why it happens:**
Veo outputs are supposed to be consistent (720p, 24fps, MP4), but API variations, aspect ratio differences, or encoding parameters can introduce subtle mismatches. FFmpeg's concat filter assumes uniformity and handles mismatches poorly without explicit re-encoding.

**How to avoid:**
Before concatenation, validate every clip: check resolution (ffprobe -show_streams), framerate, codec, and duration. Normalize all clips to identical specs using FFmpeg re-encoding: `-vf scale=1280:720,fps=24 -c:v libx264 -preset fast`. Store normalization metadata in SQLite to skip re-encoding on retry. Add duration sanity checks (each Veo clip should be exactly requested duration ±0.5s).

**Warning signs:**
- Final video plays but has abrupt transitions between scenes
- Frame counts don't match expected (5 scenes × 8s × 24fps = 960 frames, but output shows 940)
- ffmpeg concat logs show "Non-monotonous DTS" warnings
- Video players show buffering/seeking issues at scene boundaries
- Audio/video desync appearing mid-video

**Phase to address:**
Phase 2 (Stitching) - Validation must precede concatenation logic.

---

### Pitfall 7: Hard-Coded Prompt Limits Cause Truncation Without Warning

**What goes wrong:**
Veo has undocumented/soft prompt length limits (typically 500-1000 characters). Developers building prompt generation logic don't validate length, causing Veo to silently truncate prompts or return cryptic errors. Critical context (scene transitions, style notes, character descriptions) gets dropped, producing wrong videos.

**Why it happens:**
API documentation doesn't clearly specify character limits. Developers test with short prompts during development, then encounter failures when users provide longer scene descriptions or when automatic prompt enhancement adds text.

**How to avoid:**
Validate prompt length before submission (cap at 500 chars for safety margin). If prompt exceeds limit, use summarization strategy: prioritize essential elements (subject, action, camera) and drop optional modifiers (style, lighting details). Log original vs. truncated prompts for debugging. Test with maximum-length prompts during development.

**Warning signs:**
- Veo returns "Invalid request" errors with no explanation
- Generated videos ignore specific instructions that appear late in prompt
- Prompt enhancement feature causes previously-working requests to fail
- User reports "It's not following my description" for longer prompts
- Logs show prompts >800 characters being sent

**Phase to address:**
Phase 1 (Core Pipeline) - Input validation prevents cryptic API errors.

---

### Pitfall 8: No Visual Continuity Strategy = Disjointed Scene Sequences

**What goes wrong:**
Each scene is generated independently without reference to previous scenes, resulting in jarring visual discontinuities: characters change appearance, locations shift randomly, lighting/color grading varies wildly. The "multi-scene video" feels like random clips stitched together, not a coherent sequence.

**Why it happens:**
Developers treat video generation like text generation where each prompt is independent. They don't realize that human-quality video requires careful keyframe management, style consistency, and transition planning.

**How to avoid:**
Implement sequential keyframe generation: extract the final frame from scene N and use it as the reference image for scene N+1 via Veo's image-to-video mode. Maintain a style guide in prompts (camera angle, lighting, color palette) and append it to every scene. Store keyframe metadata (dominant colors, detected objects, camera position) in SQLite to inform next scene's prompt. Consider adding explicit transition scenes ("camera slowly pans from X to Y").

**Warning signs:**
- Character clothing/appearance changes between consecutive scenes
- Background location completely different despite narrative continuity
- Lighting jumps from day to night to sunset randomly
- Color grading shifts from warm to cool tones between scenes
- User feedback: "The scenes don't feel connected"

**Phase to address:**
Phase 3 (Visual Continuity) - This is an enhancement phase after basic stitching works, but architecture must support it from Phase 1.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Polling every 5 seconds without backoff | Faster perceived response | API quota exhaustion, rate limits, unnecessary costs | Never - backoff is trivial to implement |
| Storing videos as base64 in database | Simple to implement, no file I/O | Database bloat (20-50MB per clip), slow queries, backup issues | Only for initial prototype/testing (<5 clips) |
| Synchronous operation submission (wait for each scene) | Simpler code flow, easier to reason about | 5x longer pipeline execution (5 scenes × 60s each = 5min vs 60s parallel) | Single-scene MVPs only |
| Skipping operation metadata persistence | Faster development | No crash recovery, duplicate charges on failure | Never - metadata is small and critical |
| Using generic error handling (catch Exception) | All errors caught | Can't distinguish RAI rejections from network errors, wrong retry logic | Early prototyping only, must fix before production |
| Hard-coding scene count to 5 | Simpler testing | Breaks when users want 3-scene or 10-scene videos | Initial fixed-demo only |
| Storing API credentials in code | Quick setup | Security risk, credential rotation breaks code | Local development only with `.env` file |
| Skipping ffmpeg output validation | Trusting ffmpeg always succeeds | Silent corruption, missing frames, bad stitches | Never - validation is cheap |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Vertex AI Auth | Using personal gcloud credentials in production | Use service account with minimal permissions (only Vertex AI prediction), store keyfile securely |
| Veo Model Versions | Hard-coding `veo-2.0-generate-001` in requests | Store model version in config/env, prepare for Veo 3 migration with different parameters (audio generation, resolution options) |
| Rate Limiting (Image API) | Submitting 10 parallel keyframe requests immediately | Implement token bucket rate limiter (10 req/min = 1 token/6s), queue requests and process sequentially |
| Long-Running Ops | Using `operation.get()` without checking `done` field | Always check `operation.done` before accessing `operation.result`, handle `operation.error` separately |
| FFmpeg Path | Assuming `ffmpeg` is in PATH | Check ffmpeg exists on startup (`shutil.which('ffmpeg')`), fail fast with clear error if missing |
| Veo Output URLs | Parsing GCS URLs from response without expiry handling | Download videos immediately after generation, don't rely on GCS URLs persisting indefinitely (typically 7-day expiry) |
| SQLite Connections | Opening database once globally | Use connection pooling or context managers, enable WAL mode, set busy timeout (`PRAGMA busy_timeout=5000`) |
| Prompt Encoding | Sending prompts with unescaped quotes/newlines | JSON-encode prompts properly, test with special characters (quotes, unicode, emoji) |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading all project state into memory | Simple dict/list operations | Use SQLite pagination (`LIMIT`/`OFFSET`), stream results, index frequently-queried columns | >100 scenes (500MB+ memory), slow startup |
| Sequential scene processing | Predictable execution order | Parallelize Veo submissions (up to 5 concurrent), track with operation IDs, aggregate results | >3 scenes (3min → 15min for 10 scenes) |
| Storing videos in SQLite BLOBs | No filesystem management | Store on filesystem with DB references, use temp directories with cleanup | >20 videos (~1GB database, query slowdown) |
| No database indexing | Simple schema | Add indexes on `status`, `operation_id`, `created_at` for frequent queries | >50 operations (queries >1s) |
| Downloading Veo outputs repeatedly | Simple retry logic | Cache downloaded videos in temp dir, check existence before re-download | >10 retries (~100MB wasted bandwidth) |
| Full project re-processing on crash | Stateless retry | Track per-scene completion status, skip completed operations on resume | Any crash with >2 completed scenes (wasted $4+) |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Logging full prompts containing sensitive info | PII/confidential data in logs, compliance violations | Sanitize prompts before logging, redact user inputs, log only first 50 chars with ellipsis |
| Storing API credentials in SQLite | Database backup = credential leak | Use environment variables or Google Cloud Secret Manager, never commit credentials to git |
| No input validation on user prompts | Prompt injection leading to RAI violations or cost attacks | Validate prompt length, block prohibited terms (scraped from RAI guidelines), rate-limit submissions per user |
| Exposing Veo operation IDs in URLs/logs | Operation enumeration, unauthorized status checks | Treat operation IDs as sensitive, require authentication to query status, use UUIDs for external references |
| Serving generated videos without content review | Publishing RAI-violating content, brand risk | Implement manual review queue for generated content before publishing, log all generations for audit |
| No cleanup of temporary files | Disk exhaustion, leaked generated videos | Implement aggressive temp file cleanup, use context managers (`with tempfile.TemporaryDirectory()`), set max retention (24 hours) |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No progress indication during generation | "Is it working or frozen?" anxiety, premature cancellations | Show per-scene status (pending/generating/complete), estimated time remaining based on history, real-time log streaming |
| Failing entire project on single scene error | 4/5 scenes succeed but user sees "FAILED", wasted $12 | Partial completion: deliver 4-scene video, clearly show which scene failed and why, offer re-generation of failed scene only |
| No preview before $15 commit | User discovers style mismatch after paying | Generate single preview scene first (cost: $3), get user approval, then proceed with full 5-scene project |
| Cryptic error messages ("Request failed") | User doesn't know if it's their fault or system issue | Parse Veo errors into user-friendly messages: "Content filter blocked scene 3 due to [reason]. Try rephrasing: [suggestion]" |
| No cost transparency | Sticker shock at bill, distrust of system | Show cost estimate upfront (5 scenes × $3 = $15), track spent vs budget in real-time, warn at 80% budget consumption |
| Long wait without cancellation option | User stuck committing resources to unwanted video | Allow cancellation during generation, refund unused scenes, show "Cancel" button with clear refund policy |
| No retry guidance for RAI rejections | User frustrated by repeated failures without help | Suggest alternative phrasing: "Instead of 'battle scene', try 'competition' or 'sporting event'", link to content policy |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Crash Recovery:** Often missing operation state persistence — verify SQLite updates happen BEFORE starting long-running operations, test with kill -9 during generation
- [ ] **Rate Limiting:** Often missing queue implementation — verify requests are throttled to API limits (10 req/min for images), test with 20 rapid submissions
- [ ] **Cost Tracking:** Often missing per-project budgets — verify total cost is calculated and checked before each operation, test exceeding budget
- [ ] **Error Classification:** Often missing distinction between retryable vs fatal errors — verify transient network errors retry but RAI rejections don't infinite-loop
- [ ] **Idempotency:** Often missing duplicate request detection — verify submitting same prompt twice reuses existing operation, test with network retry simulation
- [ ] **FFmpeg Validation:** Often missing output verification — verify stitched video has correct duration/framerate/resolution, test with intentionally corrupt input
- [ ] **Partial Completion:** Often missing graceful degradation — verify 4/5 successful scenes produces usable output, test by forcing one scene to fail
- [ ] **Cleanup Logic:** Often missing temp file removal — verify /tmp or working directory doesn't accumulate GB of data, test with 10 consecutive runs
- [ ] **Prompt Truncation:** Often missing length validation — verify >500 char prompts are rejected or shortened before submission, test with 1000-char input
- [ ] **Visual Continuity:** Often missing keyframe extraction/reuse — verify consecutive scenes share visual elements, test with multi-scene narrative

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Duplicate Veo requests submitted | MEDIUM ($3-15 wasted) | Query Vertex AI for all pending operations, cancel duplicates via operation.cancel(), update DB to mark originals as "active" |
| SQLite database corrupted | LOW (if WAL enabled) | Run `PRAGMA integrity_check`, restore from WAL file if available, worst case: re-parse Vertex AI operation list to rebuild state |
| FFmpeg stitching produced corrupted output | LOW (clips still exist) | Delete corrupted output, re-run ffmpeg with normalized inputs, add validation to prevent future corruption |
| RAI rejection on critical scene | MEDIUM ($3 + dev time) | Analyze rejection reason, rewrite prompt to avoid trigger words, re-submit single scene, update prompt templates to prevent recurrence |
| Pipeline crashed mid-generation | LOW (if state persisted) | Scan DB for "generating" status, query Vertex AI operation status, update DB with current state, resume from last checkpoint |
| Rate limit exceeded (429 error) | LOW (temporary) | Implement exponential backoff (start at 60s), reduce concurrent requests, add request queue with throttling |
| Cost budget exceeded mid-project | MEDIUM (partial delivery) | Mark project as "paused", deliver completed scenes as interim output, get user approval for additional budget before continuing |
| No visual continuity (scenes disconnected) | HIGH (regenerate all) | Extract keyframes from completed scenes, regenerate with image-to-video mode, consider this a full rework ($15-30) |
| Missing/incorrect audio in stitched video | MEDIUM (FFmpeg re-run) | Re-stitch with explicit audio mapping (`-map 0:a`), verify each input clip has audio track, may need to generate ambient audio for silent clips |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Naive polling causes cost explosion | Phase 1: Core Pipeline | Test: Submit 5 scenes, verify <50 API status checks total, cost <$20 |
| No idempotency = duplicate charges | Phase 1: Core Pipeline | Test: Kill process during submission, restart, verify no duplicate operations |
| Synchronous FFmpeg blocks pipeline | Phase 2: Stitching | Test: Stitch 5 clips, verify CLI remains responsive, logs show concurrent operations |
| Not handling RAI rejections | Phase 1: Core Pipeline | Test: Submit prohibited content, verify graceful error + retry suggestion |
| SQLite corruption from crashes | Phase 1: Core Pipeline | Test: kill -9 during DB write, restart, verify DB integrity + state recovery |
| FFmpeg format mismatches = frame loss | Phase 2: Stitching | Test: Stitch clips with different fps/resolution, verify output frame count matches expected |
| Hard-coded prompt limits cause truncation | Phase 1: Core Pipeline | Test: Submit 1000-char prompt, verify rejection or controlled truncation |
| No visual continuity | Phase 3: Visual Continuity | Test: Generate 3-scene sequence, verify character/background consistency across scenes |
| SQLite concurrent write issues | Phase 1: Core Pipeline | Test: WAL mode enabled, 5 concurrent status updates complete without "database locked" |
| Rate limiting not enforced | Phase 1: Core Pipeline | Test: Submit 20 image requests rapidly, verify throttling to <10/min |
| No progress indication | Phase 4: User Experience | Test: User sees per-scene status updating in real-time during generation |
| Cryptic error messages | Phase 4: User Experience | Test: Trigger each error type, verify user-friendly message with action guidance |
| Missing cost transparency | Phase 4: User Experience | Test: User sees cost estimate before submission, running total during generation |
| No cleanup of temp files | Phase 1: Core Pipeline | Test: Run 10 projects, verify temp directory size <100MB |
| Prompt injection risks | Phase 5: Security Hardening | Test: Submit malicious prompts, verify sanitization + rejection |

---

## Sources

**Video Generation Best Practices:**
- [Common Mistakes When Using Veo 3.1: How to Get the Best Results? | Vmake AI](https://vmake.ai/blog/common-mistakes-when-using-veo-3-1-how-to-get-the-best-results)
- [Top 5 AI Video Generator Problems Creators Face and How Higgsfield Solves Them](https://higgsfield.ai/blog/Top-5-AI-Video-Generator-Problems-Creators-Face)
- [5 Common AI Video Generation Problems and Solutions](https://soracreators.ai/blog/5-common-ai-video-generation-problems-and-solutions)

**Vertex AI & Long-Running Operations:**
- [Veo on Vertex AI video generation API | Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation) (OFFICIAL - HIGH confidence)
- [Long-running operations | Vertex AI | Google Cloud](https://cloud.google.com/vertex-ai/docs/general/long-running-operations) (OFFICIAL - HIGH confidence)
- [Design and Implementation of Asynchronous API Model for Long-Running Operations](https://medium.com/@huzefa.qubbawala/design-and-implementation-of-an-asynchronous-api-model-for-long-running-operations-in-rest-api-3303ba6d45a2)

**Rate Limiting & API Best Practices:**
- [API Rate Limiting 2026 | How It Works & Why It Matters](https://www.levo.ai/resources/blogs/api-rate-limiting-guide-2026)
- [API Design Guidance: Long-Running Background Jobs - Tyk](https://tyk.io/blog/api-design-guidance-long-running-background-jobs/)
- [Polling vs. Long Polling vs. SSE vs. WebSockets vs. Webhooks](https://blog.algomaster.io/p/polling-vs-long-polling-vs-sse-vs-websockets-webhooks)

**FFmpeg Video Stitching:**
- [FFmpeg concat: How to merge videos — Shotstack](https://shotstack.io/learn/use-ffmpeg-to-concatenate-video/)
- [How to concatenate videos using ffmpeg | Mux](https://www.mux.com/articles/stitch-multiple-videos-together-with-ffmpeg)
- [Stitching videos from frames with ffmpeg | Starbeamrainbowlabs](https://starbeamrainbowlabs.com/blog/article.php?article=posts/432-ffmpeg-encoding.html)

**SQLite Concurrency & Crash Safety:**
- [File Locking And Concurrency In SQLite Version 3](https://sqlite.org/lockingv3.html) (OFFICIAL - HIGH confidence)
- [SQLite concurrent writes and "database is locked" errors](https://tenthousandmeters.com/blog/sqlite-concurrent-writes-and-database-is-locked-errors/)
- [Optimizing SQLite for Multi-User Apps: Concurrency & Locking](https://www.sqliteforum.com/p/optimizing-sqlite-for-multi-user)

**Prompt Engineering & RAI:**
- [How to Fix Veo Prompt Errors and Get Better AI Video Results in 2026](https://aifreeforever.com/blog/how-to-fix-veo-prompt-errors-and-get-better-ai-video-results)
- [Troubleshooting Bad AI Video Results: How to Refine Your Prompts](https://pyxeljam.com/troubleshooting-bad-ai-video-results-how-to-refine-your-prompts-for-better-outcomes/)
- [Safeguard your models | Responsible Generative AI Toolkit](https://ai.google.dev/responsible/docs/safeguards) (OFFICIAL - HIGH confidence)

**Cost Optimization:**
- [AI Video Generator Costs in 2026: Sora vs Veo 3 Pricing](https://vidpros.com/breaking-down-the-costs-creating-1-minute-videos-with-ai-tools/)
- [AI cost overruns are adding up — with major implications for CIOs](https://www.cio.com/article/4064319/ai-cost-overruns-are-adding-up-with-major-implications-for-cios.html)

**Visual Continuity:**
- [Video diffusion generation: comprehensive review and open problems](https://link.springer.com/article/10.1007/s10462-025-11331-6)
- [Video interpolation: simplify annotation for AI](https://www.innovatiana.com/en/post/interpolation-for-video-annotation)

---

*Pitfalls research for: AI Video Generation Pipeline (Veo + FFmpeg)*
*Researched: 2026-02-14*
