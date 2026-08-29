# Implementation Plan: Remote Standardized-Dataset Ingestion Security

## Phase 1 — Threat model and policy

- [x] **R1 / AC-01:** Produce the bounded remote-ingestion threat model and
  source-policy contract. — `f035c50`
- **Migrated:** **R2 / AC-01:** Obtain explicit security-policy approval before enabling
  any remote I/O.
  - [x] Record the decision options, authority boundary and fail-closed
    approval-record schema in the Conductor track.
  - **Migrated:** Obtain a dated decision from an accountable security/infrastructure
    authority; attach the signed or portal-bound evidence without secrets.
  - **Migrated:** Revalidate the threat-model and policy hashes against the approved
    record before changing any gate or enabling implementation.
- **Migrated:** **R3 / AC-01:** Run Conductor review and validation checkpoint.

## Phase 2 — Fail-closed implementation

- **Migrated:** **R4 / AC-02:** Add failing adversarial tests for SSRF, DNS rebinding,
  redirects, archive bombs, cache poisoning, and credential leakage.
- **Migrated:** **R5 / AC-03:** Implement only the approved transport/cache/receipt
  profile with quotas and offline replay.
- **Migrated:** **R6 / AC-04:** Add benchmarks, support matrix, security review, full
  local validation, hosted checks, and evidence reconciliation.

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
