# Implementation Plan

## Merged review delivery reconciliation

- [x] Record PR #812 exact head `3a2227d12721ffa418d6bf0d7e925ebe70182c59`,
  merge `286f1700b3c06824b6ab56cc6afb84348958190d`, while preserving
  scientific review, parity, promotion, release, and closure as separate gates.

## Phase 1 — Contract and evidence

- [x] **ISP1:** Freeze the finite joint-world estimand, MoSCoW requirements,
  Mermaid design, constraints, exclusions and exact-search assurance. (AC-01,
  AC-05) `57e30b07`
- [x] **ISP2:** Add strict schemas plus complementary, redundant, correlated
  and pathological fixtures with independent reference expectations. (AC-02,
  AC-04) `baa19609`

## Phase 2 — Experimental delivery

- [x] **ISP3:** Implement the exact Python evaluator, deterministic result,
  conditional marginals, Shapley attribution and failure boundaries. (AC-01–AC-03)
  `380d345f`; no-procurement comparator fix `8c8d413e`.
- [x] **ISP4:** Add CLI, public experimental discovery, documentation and
  explicit language/maturity dispositions. (AC-04, AC-05) `d6e01dee`

## Phase 3 — Assurance and handoff

- **Migrated:** **ISP5:** Run focused coverage, full relevant validation and the relevant
  implementation review; retain hosted, scientific, parity, stable, release and
  closure gates. Independent implementation review passed with the
  no-procurement assurance clarification. PR #772 exact head `f1d6f77d` passed
  all hosted checks, including 100% changed-line and changed-branch coverage,
  before squash merge `55771017`; scientific and later gates remain.
  The relevant subagent repository-review panel passed the exact-finite
  experimental scope; see `scientific-review-panel-20260801.md`. Independent
  scientific evidence for stable promotion remains pending.
  (AC-01–AC-05)

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/archive/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
