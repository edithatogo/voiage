# Implementation Plan: Controlled Live Standardized-Dataset Interoperability

## Phase 1 — Source authorization

- **Migrated:** **L1 / AC-01:** Record the approved Croissant and Frictionless source
  packets, digests, rights, citation, terms, and source-selection rationale.
- [x] **L2 / AC-01:** Add fail-closed tests proving a missing approval, changed
  digest, or changed resource cannot run a probe. Existing authoritative-probe
  tests cover explicit opt-in, descriptor/resource digests, materialization,
  and single-receipt enforcement. (2026-08-01)
- [x] **L3 / AC-01:** Run Conductor review and validation checkpoint. Focused
  live-probe tests, full Conductor validation, cross-reference validation, and
  v1 programme integrity all pass; L1 source authorization remains pending.
  (2026-08-01)

## Phase 2 — Controlled probes

- **Migrated:** **L4 / AC-02:** Add opt-in probes and receipt/offline-replay tests.
- **Migrated:** **L5 / AC-03:** Prove the probes terminate at the canonical normalized
  bundle and numerical-equivalence path.
- **Migrated:** **L6 / AC-04:** Publish narrow support documentation and operational
  guidance without claiming general live-data support.
- [x] **L7 / AC-02--AC-04:** Run security, dependency, full tox, hosted checks,
  Conductor review, and issue/Project evidence reconciliation. PR #817 exact
  head `d40f5b617df847fe517759f5892a1562f25bc4d9` passed its required
  matrix before merge `33537325ad7262dc15bcddb4283a6aa51cfdb323`.
  This does not satisfy L1 or authorize network I/O.

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
