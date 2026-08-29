# Implementation Plan

## Phase 1 — Governance and contract reconciliation

- [x] **G1:** Verify the owning issue, native parent/children, Project 28,
  metadata, registry and cross-reference manifest. (AC-01) — `cd7bfa3`
- [x] **G2:** Reconcile existing repository artifacts and prior evidence
  without converting issue status into implementation evidence. (AC-01, AC-02)
  — `cd7bfa3`
- [x] **G3:** Freeze workstream estimands, contracts, maturity boundaries and
  explicit exclusions. (AC-02, AC-05, AC-06) — `cd7bfa3`
- [x] **G4:** Run automated contract review and full Conductor validation.
  (AC-01, AC-07) — 4fe2d8d

## Phase 2 — Evidence before positive claims

- [x] **G5:** Add failing conformance, reference, property and pathological
  tests, or the corresponding reproducible review protocol. (AC-03) — 23900b8
- [x] **G6:** Add versioned schemas, fixtures, diagnostics and provenance
  contracts required by the accepted scope. (AC-02, AC-03) — `e7622eb`
- [x] **G7:** Record rights, privacy, scientific, practitioner and external
  evidence gates that apply to this workstream. (AC-05, AC-06) — 67857b5
- [x] **G8:** Run an independent evidence and boundary review. (AC-03, AC-07)
  — `f8a3337`

## Phase 3 — Delivery or reviewed exclusion

- [x] **G9:** Implement each accepted repository-owned capability or record a
  reviewed exclusion with migration guidance. (AC-02, AC-06) — 6ab9c88
- [x] **G10:** Add Rust/Python/R/Julia/Mojo dispositions and installed
  shared-fixture evidence where executable surfaces are advertised. (AC-04)
  - [x] Freeze the ordered parity and promotion gates, evidence packet fields,
    and fail-closed disposition. — `82ec400`
  - [x] Execute clean installed Rust/Python/R/Julia shared-fixture runs and
    capture the parity packet; panel and maintainer promotion decisions remain
    external. — `installed-parity-packet-20260803`
  - [x] Refresh exact-head hosted parity with an isolated Julia depot,
    separated R development and installed-package lanes, and immutable
    unsupported receipts for any unavailable runtime. — PR #992 exact head
    `21b6a073`; hosted receipt `hosted-parity-receipt-20260821`
- [x] **G11:** Add documentation, examples, generated surfaces and capability
  discovery that match the evidenced maturity state. (AC-05) — 7a2f296
- [x] **G12:** Run automated implementation review, focused validation and the
  repository harness. (AC-03–AC-07) — `381bc917`; dedicated v1 and full Conductor validation pass

## Phase 4 — Programme and hosted closeout

- [x] **Closeout plan:** Order remaining local, hosted and external blockers
  with options, contingencies and fail-closed archive rules. — 833ed457

- **Migrated:** **G13:** Reconcile child-issue results, roadmap, todo, registries,
  release targets and remaining external gates. (AC-01, AC-05, AC-06)
  - [x] Define distinct registry/release/publication/parent-closure states,
    evidence lanes and authority boundaries. — `b4a1f20`
  - **Migrated:** Refresh each destination from authoritative receipts and reconcile
    child and parent issue state. Live issue receipts captured; Project 28
    membership is not found and requires authorized GitHub reconciliation.
- **Migrated:** **G14:** Run final full local validation and hosted required checks.
  (AC-07)
  - **Migrated:** Bind the final validation packet to the exact release candidate and
    hosted check run. Local candidate receipt captured; exact-head hosted run
    is not available.
- **Migrated:** **G15:** Record repository completion separately from merge, release,
  publication, registry acceptance and issue closure. (AC-02, AC-07)
  - [x] Define the fail-closed parent-closure rule and final receipt fields.
    — `b4a1f20`
  - [x] Record the final receipt matrix with repository and external states.
    — `g15-final-receipt-matrix-20260803`

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/archive/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
