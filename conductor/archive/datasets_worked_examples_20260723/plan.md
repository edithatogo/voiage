# Implementation Plan

## Phase 1 — Governance and contract reconciliation

- [x] **G1:** Verify the owning issue, native parent/children, Project 28,
  metadata, registry and cross-reference manifest. (AC-01)
- [x] **G2:** Reconcile existing repository artifacts and prior evidence
  without converting issue status into implementation evidence. (AC-01, AC-02)
- [x] **G3:** Freeze workstream estimands, contracts, maturity boundaries and
  explicit exclusions. (AC-02, AC-05, AC-06)
- [x] **G4:** Run automated contract review and full Conductor validation.
  Existing standardized-ingestion reference cases provide the current
  numerical-equivalence contract; full Conductor and cross-reference validation
  pass. (2026-08-01)
  (AC-01, AC-07)

## Phase 2 — Evidence before positive claims

- **Migrated:** **G5:** Add failing conformance, reference, property and pathological
  tests, or the corresponding reproducible review protocol. (AC-03)
- **Migrated:** **G6:** Add versioned schemas, fixtures, diagnostics and provenance
  contracts required by the accepted scope. (AC-02, AC-03)
- **Migrated:** **G7:** Record rights, privacy, scientific, practitioner and external
  evidence gates that apply to this workstream. (AC-05, AC-06)
- **Migrated:** **G8:** Run an independent evidence and boundary review. (AC-03, AC-07)

## Phase 3 — Delivery or reviewed exclusion

- [x] **G9-ChurnRetention:** Deliver decision-grade customer churn retention worked example (#574) and cross-industry dataset snapshots in registry (#575).
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
- **Migrated:** **G14:** Run final full local validation and hosted required checks.
  (AC-07)
- **Migrated:** **G15:** Record repository completion separately from merge, release,
  publication, registry acceptance and issue closure. (AC-02, AC-07)

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
