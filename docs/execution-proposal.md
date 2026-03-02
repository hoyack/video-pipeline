# Phase 20 Execution Proposal: Worktrees + Issue-Closing PRs

## Question

Can we execute Phase 20 in git worktrees that create PRs closing the related GitHub issues (#8, #9, #10, #11)?

## Short Answer

**Yes, but the current plans need to be re-split by domain (not layer) to get per-issue PRs.** The current 2-plan split (backend/frontend) can't produce 4 issue-closing PRs without messy cherry-picking. Below are three options ranked by recommendation.

---

## Context

### Phase 20 Scope
Add missing upload endpoints and frontend UI for reference images and audio across all Production Bible entity types.

### GitHub Issue Mapping

| Issue | Entity Domain | Requirements | What's Missing |
|-------|--------------|--------------|----------------|
| #8 | Character (actor refs, wardrobe, appearance) | PBEX-01, PBEX-02 | Upload endpoints + UI |
| #9 | Set (sonic identity audio, reverse-prompt) | PBEX-07, PBEX-08 | Upload endpoints + UI |
| #10 | Prop (reference image) | PBEX-13 | Frontend upload button only (backend exists) |
| #11 | Sound (score themes, SFX audio, playback) | PBEX-16, PBEX-17 | Upload endpoints + UI + AudioPlayer component |

### Current Plan Structure (Layer Split)

```
20-01 (Wave 1, backend)  →  characters.py, sets_props.py, sound.py
      ↓ depends_on
20-02 (Wave 2, frontend) →  AudioPlayer.tsx, client.ts, CharacterDetail.tsx,
                             SetDetail.tsx, SoundDepartment.tsx
```

**Problem:** Both plans touch code for ALL 4 issues. You can't create a PR from plan 20-01 that closes only #8 — it also contains #9, #10, #11 backend work.

---

## Options

### Option A: Re-plan by Domain (Recommended)

Re-split Phase 20 into 4 plans, one per GitHub issue. Each plan is a vertical slice: backend endpoint + frontend UI for one entity domain.

```
20-01 (Wave 1): Character uploads       → closes #8
20-02 (Wave 1): Set + Prop uploads      → closes #9, #10
20-03 (Wave 1): Sound uploads + AudioPlayer → closes #11
```

**Execution strategy:**
- Create one feature branch per plan (3 branches)
- Execute each in an isolated worktree via GSD executor with `isolation: "worktree"`
- Each agent creates its PR on completion
- Plans 20-01, 20-02, 20-03 are independent (no cross-dependencies) since they touch different files

**File overlap analysis:**

| File | #8 (Char) | #9+#10 (Set/Prop) | #11 (Sound) |
|------|-----------|-------------------|-------------|
| `characters.py` | yes | | |
| `sets_props.py` | | yes | |
| `sound.py` | | | yes |
| `client.ts` | yes (3 fns) | yes (2 fns) | yes (2 fns) |
| `CharacterDetail.tsx` | yes | | |
| `SetDetail.tsx` | | yes | |
| `SoundDepartment.tsx` | | | yes |
| `AudioPlayer.tsx` (new) | | | yes |

**Conflict risk:** Only `client.ts` is shared — each plan adds different functions to the same file. Merge conflicts are likely but trivial (additive-only changes to different sections). PRs should merge sequentially.

**Pros:**
- Clean 1:1 mapping between PR and GitHub issue
- Each PR is reviewable in isolation
- Parallel execution possible (Wave 1 for all 3)
- Worktree isolation prevents cross-contamination

**Cons:**
- Requires re-planning Phase 20 (delete current plans, re-run `/gsd:plan-phase 20`)
- `client.ts` merge conflicts between PRs (trivial but present)
- 3 PRs to review instead of 1

**Effort:** ~15 min to re-plan, same execution time

---

### Option B: Single Worktree, Single PR (Simplest)

Execute the current 2-plan structure in a single worktree on one feature branch. Create one PR that closes all 4 issues.

```
git worktree add .claude/worktrees/phase-20 -b feat/entity-media-uploads
# Execute 20-01 (wave 1) → 20-02 (wave 2) sequentially
# Create PR: "Closes #8, Closes #9, Closes #10, Closes #11"
```

**Pros:**
- Zero re-planning needed — execute current plans as-is
- No merge conflicts
- One review covers everything
- Sequential waves work naturally in one worktree

**Cons:**
- Single large PR (~300-400 lines across 8 files)
- All 4 issues close simultaneously — no incremental delivery
- Can't merge partial work if one domain has issues

**Effort:** Minimal — just change GSD branching config and execute

---

### Option C: Single Worktree, Post-Execution Split (Complex)

Execute both plans in a worktree, then cherry-pick commits into per-issue branches.

**Why this doesn't work well:**
- GSD executor commits per-task, not per-domain
- Task 1 of plan 20-01 creates endpoints for Characters AND Wardrobe AND reverse-prompt in one commit
- Splitting after the fact requires `git diff` surgery on shared files
- High risk of missed changes or broken builds per branch

**Not recommended.**

---

## Recommendation

**Option A (re-plan by domain)** if per-issue PRs matter for your workflow — the re-planning cost is low and the result is clean.

**Option B (single PR)** if you just want isolation from master and a reviewable PR — fastest path to done.

### Proposed Next Steps (Option A)

1. Delete current plans: `rm .planning/phases/20-entity-media-uploads/20-0*-PLAN.md`
2. Re-plan with domain split: `/gsd:plan-phase 20` (with guidance to split by issue)
3. Execute with worktree isolation: `/gsd:execute-phase 20` (branching strategy: phase, or manual worktrees)
4. Create 3 PRs, each closing its issue(s)

### Proposed Next Steps (Option B)

1. Set branching strategy: update `.planning/config.json` → `"branching_strategy": "phase"`
2. Execute: `/gsd:execute-phase 20`
3. Create single PR: `gh pr create --title "feat: entity media uploads" --body "Closes #8, #9, #10, #11"`

---

## Technical Notes

### GSD Branching Config
Current setting in `.planning/config.json`:
```json
{ "git": { "branching_strategy": "none" } }
```
Options: `"none"` | `"phase"` (branch per phase) | `"milestone"` (branch per milestone)

### Worktree Isolation
The Agent tool supports `isolation: "worktree"` which creates a temporary git worktree per agent. However, GSD's execute-phase orchestrator manages branching at the orchestrator level, not the agent level. For per-plan worktrees, we'd need to either:
- Run separate `/gsd:execute-phase` invocations per plan (manual)
- Or customize the execution flow

### Wave Dependencies
Current plans have a hard dependency: 20-02 depends on 20-01. If re-planned by domain, all plans become independent (Wave 1) since each touches different backend + frontend files. This actually enables parallel execution — a net improvement.
