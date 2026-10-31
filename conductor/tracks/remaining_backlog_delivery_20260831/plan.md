# Implementation Plan: Remaining Backlog Delivery

## Phase 1 — Audit and recovery

- [x] B1: Reconcile every open PR and issue with current hosted evidence (AC1).
- [x] B2: Verify independent recovery and prune retired branch tips with leases
  while preserving active work (AC5).
- [x] B3: Review inventory and recovery receipts; validate Conductor and
  GitHub cross-references before checkpointing (AC1, AC5).

## Phase 2 — Repository repairs

- [x] B4: Challenge historical-lock, dependency-policy and coverage-upload
  contracts, then repair #656 controls without rewriting old receipts (AC2).
  Evidence: PR #1057 (515529e3), closed #656, and
  [delivery checkpoint](./delivery-checkpoint-20260901.md).
- [x] B5: Test invalidation and plugin incompatibilities, then implement and
  measure opt-in serial test acceleration for #1028 (AC2).
  Later evidence: PR #1059 (79072731), closed #1028, and
  [delivery checkpoint](./delivery-checkpoint-20260901.md).
- [~] B6: Test recipe and installed-source behavior, then complete current
  Spack/EasyBuild packages and unsubmitted upstream packets for #1025 (AC2).
  - [x] B6.1: Qualify an immutable current source archive with provenance,
    runtime/dependency and actual runtime-file integrity witnesses; see the
    2026-10-28 continuation audit and handoff. Accepted in PR #1212 at
    `7f0feacc`; reviewed and merged trees match. This does not close B6.
  - [~] B6.2: Bind current recipes to the qualified source; obtain actual native
    solver/build/installed-module evidence and the required compiler matrix.
    Polars frontend/runtime source declarations were delivered in PR #1214.
    Dated Rust source/bootstrap bindings and stable compiler-role separation
    were accepted in PR #1215 (`e336cd9f`); Arrow providers, native builds
    and the full matrix remain open.
    Source slices: stable compiler roles `07c556ec`, dated source/bootstrap
    binding `1d9be54d`; upstream licensing retained separately. See the dated
    compiler receipt and review. All 15 local tox environments pass; protected
    PR #1215 delivery is accepted; this is not native build acceptance.
- [x] B7: Complete remaining R, governance, identifier and venue preparation;
  retain external acceptance requirements (AC2, AC4).
- [~] B8: Review all changes, run full tox and relevant language-native checks,
  and record failures and subsequent verification without weakening gates (AC2).

## Phase 3 — Protected delivery and issue reconciliation

- [x] B9: Merge repaired PRs sequentially after exact-head checks, signatures
  and resolved-thread verification; record merge evidence (AC3).
- [x] B10: Refresh all issue bodies and evidence links, closing only satisfied
  acceptance criteria and retaining genuine human/external gates (AC4).
- [x] B11: Review post-merge hosted evidence and validate all track records
  before checkpointing delivery (AC3, AC4).

## Phase 4 — Final cleanup and audit

- [~] B12: Preserve and remove integrated local branches and clean worktrees;
  retain unique work, active branches and maintained release lines (AC5).
- [~] B13: Re-query final inventories and document all unfinished requirements
  before assessing the complete user goal (AC6).
  The pre-closeout inventory is bound; an authoritative post-merge inventory
  remains required after this branch and worktree are pruned.
- [ ] B14: Review final receipts and run Conductor/cross-reference validation;
  archive only when this track's acceptance criteria are satisfied (AC6).

## Review fixes — current source qualification

- [x] Require declared build inputs and validate every distribution dependency,
  including marked requirements and extras; preserve failing regression evidence
  and rerun the affected source and package checks before protected delivery.
  Fix: `dffb1114`; 24 qualifier tests and 33 affected integration tests passed.
  Final-head hosted checks and protected delivery remain required for B6.1.

## Review fixes — dated compiler providers

- [x] Replace wildcard imports, correct the class-method binding and preserve
  the failed external OpenSSL probe fallback; regression rejected the original
  import style and all 62 affected tests now pass. Private Spack class loading
  passes on Python 3.15.1. Current-head hosted checks and native build gates
  remain required; this does not complete B6.2.

## Arrow prerequisite source slices

- [~] B6.2-LibCST: Freeze and challenge the Python 3.15 LibCST source, marker,
  licence and compiler contract; implement the bounded readiness packet, then
  qualify the actual provider without inferring the full Arrow/native matrix.
  Source/declaration slice: `227f1583`; 32 affected tests, authenticated source
  and actual private Spack class load pass. The full integrated Arrow candidate
  gate remains pending.

## Build capacity follow-up

The [capacity plan](./capacity-plan.md) retains both EasyBuild generations and
source-built compiler qualification. The integrator has brought the measurement
guard, budget validator and retained pilot evidence into the local candidate.

- [~] B6-CAP1: Reconcile the accepted support scope and corrected admission
  requirement with the terminal capacity packet.
- [~] B6-CAP2: Retain the Polars and Arrow pilot observations; source compiler
  and full native graph measurements remain pending.
- [ ] B6-CAP3: Qualify immutable source staging, cleanup and cache reuse on
  matched inputs without sharing mutable build roots.
- [ ] B6-CAP4: Publish workload-specific budgets only after representative
  successful measurements and explicit headroom are established.

B8 consumes capacity receipts alongside behavioural checks; B12 retains
preservation-backed cleanup ownership. These tasks do not close native acceptance.
