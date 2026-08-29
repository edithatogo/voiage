# Implementation Plan

## Phase 1 — Governance and contract reconciliation

- [x] **G1:** Verify the owning issue, native parent/children, Project 28,
  metadata, registry and cross-reference manifest. (AC-01) `2fe24c81`
- [x] **G2:** Reconcile existing repository artifacts and prior evidence
  without converting issue status into implementation evidence. (AC-01, AC-02)
  `c9571d89`
- [x] **G2a:** Freeze a hash-bound candidate/frozen classification checkpoint
  for cross-track issues #593–#600 and #619 without completing #566 or
  promoting the canonical registry. (AC-02, AC-05, AC-06) `28f7eb5e`
- [x] **G3:** Freeze workstream estimands, contracts, maturity boundaries and
  explicit exclusions. The hash-bound classification checkpoint freezes the
  candidate/frozen dispositions for #593–#600 and #619; open #566 remains a
  prerequisite for completing the census contract. (AC-02, AC-05, AC-06)
  `28f7eb5e`
- [x] **G4:** Run automated contract review and full Conductor validation.
  Current cross-reference, v1 programme, evidence-ledger and full Conductor
  validation pass. (AC-01, AC-07) `2026-08-01`

## Phase 2 — Evidence before positive claims

- **Migrated:** **G5:** Add failing conformance, reference, property and pathological
  tests, or the corresponding reproducible review protocol. (AC-03)
- **Migrated:** **G6:** Add versioned schemas, fixtures, diagnostics and provenance
  contracts required by the accepted scope. (AC-02, AC-03)
- **Migrated:** **G7:** Record rights, privacy, scientific, practitioner and external
  evidence gates that apply to this workstream. (AC-05, AC-06)
- **Migrated:** **G8:** Run an independent evidence and boundary review. (AC-03, AC-07)

## Phase 3 — Delivery or reviewed exclusion

- [x] **G9-DecisionProblem:** Implement portable DecisionProblem and Intervention data structures (#566) in `voiage/schema.py` matching the versioned DecisionProblemV1 schema contracts and full serialization tests.
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
- [x] **G14:** Run final full local validation and hosted required checks. PR
  #816 exact head `1a159e02af95fc3a6bce46f2bf8909561be0b9bd` passed the
  required matrix before merge `68cb15dfcb8706ab653f8a1631b433a7f63ba322`.
  Residual classification and promotion remain separate. (AC-07)
- **Migrated:** **G15:** Record repository completion separately from merge, release,
  publication, registry acceptance and issue closure. (AC-02, AC-07)

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
