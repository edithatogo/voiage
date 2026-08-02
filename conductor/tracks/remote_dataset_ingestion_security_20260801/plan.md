# Implementation Plan: Remote Standardized-Dataset Ingestion Security

## Phase 1 — Threat model and policy

- [x] **R1 / AC-01:** Produce the bounded remote-ingestion threat model and
  source-policy contract. — `f035c50`
- [ ] **R2 / AC-01:** Obtain explicit security-policy approval before enabling
  any remote I/O.
- [x] **R3 / AC-01:** Run Conductor review and validation checkpoint.
  (2026-08-01; local validators and PR #824 required hosted checks pass;
  remote approval remains pending)

## Phase 2 — Fail-closed implementation

- [ ] **R4 / AC-02:** Add failing adversarial tests for SSRF, DNS rebinding,
  redirects, archive bombs, cache poisoning, and credential leakage.
- [ ] **R5 / AC-03:** Implement only the approved transport/cache/receipt
  profile with quotas and offline replay.
- [ ] **R6 / AC-04:** Add benchmarks, support matrix, security review, full
  local validation, hosted checks, and evidence reconciliation.
