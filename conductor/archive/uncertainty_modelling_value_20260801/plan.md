# Implementation Plan

## Phase 1 — Contract and evidence

- [x] **UMV1 / #774:** Freeze the exact finite estimand, MoSCoW requirements,
  Mermaid design, information-acquisition boundary and DVSS/VMS disposition.
  (AC-01, AC-05) `c04d4c33`
- [x] **UMV2 / #774:** Add strict schemas plus two-stage nonlinear,
  multistage, infeasible-recourse and pathological fixtures. (AC-01, AC-02,
  AC-04) `c04d4c33`

## Phase 2 — Experimental runtime

- [x] **UMV3 / #775:** Implement direction-aware EV/EEV/RP/WS,
  VSS/EVIU/EVPI, complete ties, feasibility and policy audit. (AC-01–AC-03)
  `c04d4c33`
- [x] **UMV4 / #775:** Add public experimental Python/CLI discovery,
  deterministic serialization and user documentation. (AC-04, AC-05)
  `c04d4c33`

## Phase 3 — Assurance and handoff

- **Migrated:** **UMV5 / #776:** Run 100% changed-module branch coverage, focused/full
  validation and the relevant subagent review panel. The implementation review found
  no unresolved Critical, High or Medium findings. PR #798 exact head
  `aa5d9fd8` completed 42 hosted checks with 38 successes and four governed
  skips, including all six wheel contracts and aggregate coverage, before
  squash merge `c5adca8f`. Repository-delivery subissues #774–#776 may close;
  subagent panel review is recorded in
  `scientific-review-panel-20260801.md`; Rust/R/Julia parity, stable promotion,
  release, parent #594 closure and umbrella #318 closure remain pending.
  (AC-01–AC-05)

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/archive/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
