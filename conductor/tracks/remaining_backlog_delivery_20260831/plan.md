# Implementation Plan: Remaining Backlog Delivery

## Phase 1 — Audit and recovery

- [~] B1: Reconcile every open PR and issue with current hosted evidence (AC1).
- [~] B2: Verify independent recovery and prune retired branch tips with leases
  while preserving active work (AC5).
- [~] B3: Review inventory and recovery receipts; validate Conductor and
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
- [~] B7: Complete remaining R, governance, identifier and venue preparation;
  retain external acceptance requirements (AC2, AC4).
- [~] B8: Review all changes, run full tox and relevant language-native checks,
  and record failures and subsequent verification without weakening gates (AC2).

## Phase 3 — Protected delivery and issue reconciliation

- [~] B9: Merge repaired PRs sequentially after exact-head checks, signatures
  and resolved-thread verification; record merge evidence (AC3).
- [~] B10: Refresh all issue bodies and evidence links, closing only satisfied
  acceptance criteria and retaining genuine human/external gates (AC4).
- [~] B11: Review post-merge hosted evidence and validate all track records
  before checkpointing delivery (AC3, AC4).

## Phase 4 — Final cleanup and audit

- [~] B12: Preserve and remove integrated local branches and clean worktrees;
  retain unique work, active branches and maintained release lines (AC5).
- [ ] B13: Re-query final inventories and document all unfinished requirements
  before assessing the complete user goal (AC6).
- [ ] B14: Review final receipts and run Conductor/cross-reference validation;
  archive only when this track's acceptance criteria are satisfied (AC6).
