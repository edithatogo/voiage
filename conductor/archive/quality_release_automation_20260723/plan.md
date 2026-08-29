# Implementation Plan

## Phase 1 — Governance and contract reconciliation

- [x] **G1:** Verify the owning issue, native parent/children, Project 28,
  metadata, registry and cross-reference manifest. (AC-01)
- [x] **G2:** Reconcile existing repository artifacts and prior evidence
  without converting issue status into implementation evidence. (AC-01, AC-02)
- [x] **G3:** Freeze workstream estimands, contracts, maturity boundaries and
  explicit exclusions. (AC-02, AC-05, AC-06)
- [x] **G4:** Run automated contract review and full Conductor validation.
  (AC-01, AC-07) — `9ef5271`

## Phase 2 — Evidence before positive claims

- [x] **G5:** Add failing conformance, reference, property and pathological
  tests, or the corresponding reproducible review protocol. (AC-03) — `d06eccc`
- **Migrated:** **G6:** Add versioned schemas, fixtures, diagnostics and provenance
  contracts required by the accepted scope. (AC-02, AC-03)
- [x] **G7:** Record rights, privacy, scientific, practitioner and external
  evidence gates that apply to this workstream. (AC-05, AC-06) — `9ef5271`
- **Migrated:** **G8:** Run an independent evidence and boundary review. (AC-03, AC-07)

## Phase 3 — Delivery or reviewed exclusion

- **Migrated:** **G9:** Implement each accepted repository-owned capability or record a
  reviewed exclusion with migration guidance. (AC-02, AC-06)
- **Migrated:** **G10:** Add Rust/Python/R/Julia/Mojo dispositions and installed
  shared-fixture evidence where executable surfaces are advertised. (AC-04)
- **Migrated:** **G11:** Add documentation, examples, generated surfaces and capability
  discovery that match the evidenced maturity state. (AC-05)
- **Migrated:** **G12:** Run automated implementation review, focused validation and the
  repository harness. (AC-03–AC-07)

## Phase 4 — Programme and hosted closeout

- **Migrated:** **G13:** Reconcile child-issue results, roadmap, todo, registries,
  release targets and remaining external gates. (AC-01, AC-05, AC-06)
  - [x] Reject the regressing `2.0.1-rc.4` identity after verifying published
    `2.0.1`, and select the backward-compatible `2.1.0` release line under the
    owner Option A decision. — `release-lineage-decision-20260821`
  - [x] Synchronize the Rust, Python, R and Julia manifests; obtain exact-head
    hosted checks; create the signed tag; stage and digest-review the private
    draft; then publish only the reviewed stable-core payload. — PR #996,
    `release-2.1.0-publication-receipt-20260821`
  - [x] Record PyPI and GitHub publication receipts separately from external
    registry, archive, and publication-service acceptance. —
    `release-2.1.0-publication-receipt-20260821`
  - **Migrated:** Reconcile each external registry, archive, and publication-service
    decision only after its destination supplies an authoritative receipt.
- [x] **G14:** Run final full local validation and hosted required checks. PR
  #822 exact head `7e12a5fbc6f7091166d7f5d64c6f2b5b45764f72` passed the
  required matrix before merge `0df988125b89f8d0bad08def0bd5b2ea03cd54f5`.
  External publication remains separate. (AC-07)
- **Migrated:** **G15:** Record repository completion separately from merge, release,
  publication, registry acceptance and issue closure. (AC-02, AC-07)

## Review Fixes

- [x] **RF-G5:** Add explicit independent repository references, refresh track
  metadata timestamp and normalize the recorded task SHA. (Conductor review) — `74cbb55`

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
