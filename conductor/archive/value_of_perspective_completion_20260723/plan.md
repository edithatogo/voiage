# Implementation Plan

## Phase 1 — Governance and contract reconciliation

- [x] **G1:** Verify the owning issue, native parent/children, Project 28,
  metadata, registry and cross-reference manifest. (AC-01) — `3862d7b`
- [x] **G2:** Reconcile existing repository artifacts and prior evidence
  without converting issue status into implementation evidence. (AC-01, AC-02)
  — `3862d7b`
- [x] **G3:** Freeze workstream estimands, contracts, maturity boundaries and
  explicit exclusions. (AC-02, AC-05, AC-06) — `3862d7b`
- **Migrated:** **G4:** Run automated contract review and full Conductor validation.
  (AC-01, AC-07)

## Phase 2 — Evidence before positive claims

- [x] **G5:** Add failing conformance, reference, property and pathological
  tests, or the corresponding reproducible review protocol. Existing v1
  conformance, tie/pathology, Arrow and interchange tests provide this evidence;
  stable promotion remains disallowed. (AC-03)
- [x] **G6:** Add versioned schemas, fixtures, diagnostics and provenance
  contracts required by the accepted scope. Versioned perspective schemas,
  normative fixtures, catalog and promotion evidence are present and validated.
  (AC-02, AC-03)
- [x] **G7:** Record rights, privacy, scientific, practitioner and external
  evidence gates that apply to this workstream. Metadata and promotion evidence
  retain blocked external/open-data and scientific gates. (AC-05, AC-06)
- [x] **G8:** Run a panel evidence and boundary review using at least two
  relevant subagents: one estimand/boundary reviewer and one numerical/
  conformance reviewer. Both panels passed with no defects or disagreement;
  synthesis preserves perspective-uncertainty weights, pre-constructed finite
  net-benefit inputs, and fail-closed unsupported estimands. This internal panel
  is not external scientific authorization (2026-08-01). (AC-03, AC-07)

## Phase 3 — Delivery or reviewed exclusion

- [x] **G9:** Implement each accepted repository-owned capability or record a
  reviewed exclusion with migration guidance. Directional current-information
  EVoP remains the implemented fixture-backed path; perfect, partial and sample
  perspective estimands are explicitly unsupported with migration guidance
  (commit `2b84dbd9`). (AC-02, AC-06)
- **Migrated:** **G10:** Add Rust/Python/R/Julia/Mojo dispositions and installed
  shared-fixture evidence where executable surfaces are advertised. Explicit
  language dispositions are now versioned; installed shared-fixture evidence
  remains pending and blocks completion. (AC-04)
- [x] **G11:** Add documentation, examples, generated surfaces and capability
  discovery that match the evidenced maturity state. CLI documentation links
  the fixture-backed contract and capability dispositions; v2 export,
  interchange-manifest, Astro check and docs build all pass (commit `2db2694e`).
  (AC-05)
- [x] **G12:** Run automated implementation review, focused validation and the
  repository harness. Panel synthesis, focused tests, Ruff/Bandit, typecheck,
  repository harness and harness tests pass (commit `57908731`). Installed
  parity and hosted gates remain separate. (AC-03–AC-07)

## Phase 4 — Programme and hosted closeout

- [x] **G13:** Reconcile child-issue results, roadmap, todo, registries,
  release targets and remaining external gates. Repository reconciliation
  validators pass; external parity, hosted, promotion, release and issue
  closure gates remain explicitly pending. (AC-01, AC-05, AC-06)
- [x] **G14:** Run final full local validation and hosted required checks after
  panel synthesis and remediation. PR #828 exact head
  `e63f31e4929081201c2ca5df3372ab73c9714eba` completed 32 successful
  checks, three governed skips, and one neutral check before merge
  `168156a3e0910e99babecbf4ec06bbfb86b85f56`. Installed parity and
  promotion remain separate. (AC-07)
- **Migrated:** **G15:** Record repository completion separately from merge, release,
  publication, registry acceptance and issue closure. (AC-02, AC-07)

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
