# Implementation Plan

## Phase 1 — Contract and governance reconciliation — checkpoint `5ab9727d`

- [x] **U1:** Freeze the expected-utility decision problem, named utility,
  wealth/reference, units, probability, information, scope, cost-location and
  tie contracts. (AC-02) — `50c8db4a`
- [x] **U2:** Freeze EUI, CEI, BPI, SPI, PPI, policy re-optimization, signed-value,
  root-diagnostic and comparability semantics. (AC-03, AC-05)
  — `9fcd3717`
- [x] **U3:** Freeze VoC as a presentation/delegating alias and audit adjacent
  EVPI, CVaR, risk-sensitive, preference and buying-price surfaces. (AC-04,
  AC-07) — `69fa2283`
- [x] **U4:** Materialize GitHub subissues and synchronize Project 28,
  requirements, Mermaid design, cross-references and canonical C16 planning.
  (AC-01) — `eb1b5b36`
- [x] **U5:** Run automated contract review and full Conductor validation.
  (AC-01–AC-05) — `e2cb730c`

### Phase 1 review fixes

- [x] Preserve #694–#697 as nested #595 subissues in the canonical C16
  projection rather than projecting them as independent frontier issues.
  — `8cfa3a7f`
- [x] Reconcile deterministic tie selection, finite-signal design, and complete
  tie-set transition diagnostics across the frozen contract artifacts.
  — `e2cb730c`

## Phase 2 — Reference evidence before implementation — checkpoint `5f9d4110`

- [x] **U6:** Add failing independent affine, exponential and nonlinear
  reference tests for EUI, CEI, BPI, SPI, PPI and affine EVPI reduction. (AC-03,
  AC-05) — `0a892538`
- [x] **U7:** Add failing property and pathological tests for positive-affine
  invariance, buy/sell asymmetry, ties, nonuniform probabilities, utility
  domains, bracketing/nonconvergence and stakeholder comparability. (AC-05)
  — `e39bfcce`
- [x] **U8:** Add versioned schemas, normative fixtures, deterministic
  serialization and root-diagnostic contracts. (AC-02, AC-03, AC-05)
  — `047524f4`
- [x] **U9:** Run independent evidence and numerical-boundary review. (AC-05,
  AC-08) — `71a43e2e`

### Phase 2 review fixes

- [x] Add PPI, stakeholder-comparability and bounded solver-limit red
  contracts; fully discriminate nested result diagnostics; require cost,
  currency and price-date semantics; exclude CRRA risk aversion one; and
  validate a complete normative result. — `71a43e2e`

## Phase 3 — Rust runtime and Python presentation — checkpoint `1f6f3ce0`

- [x] **U10:** Implement the accepted Rust utility, policy, information-value,
  price and bounded root-solving kernels with versioned result envelopes.
  (AC-02, AC-03, AC-06) — `e538b626`
- [x] **U11:** Implement the thin Python facade, deterministic serialization,
  DecisionAnalysis integration and VoC presentation/delegation without a
  duplicate kernel. (AC-03, AC-04, AC-06) — `ec242a36`
- [x] **U12:** Add differential, property, serialization, failure and
  performance assurance for the runtime boundary. (AC-05, AC-08)
  — `259a06e3`
- [x] **U13:** Run automated implementation review, focused validation and the
  repository harness. (AC-05, AC-08) — `81c0e701`

### Phase 3 review fixes

- [x] Replace fabricated root diagnostics; preserve canonical per-signal
  policies and explicit domain exclusions; reject discontinuous no-root
  prices; complete validation, provenance, PPI reasons and pairwise
  comparability; and validate affine-only monetary EVPI presentation.
  — `81c0e701`

## Phase 4 — User, binding and governed closeout surfaces

- [x] **U14:** Add CLI/reporting, capability discovery, accessible examples
  and documentation that distinguish utility and monetary scales. (AC-07)
  — `891e3741`
- [x] **U15:** Record and validate Rust, Python, R, Julia and Mojo capability
  dispositions and shared-fixture evidence or explicit unsupported states.
  (AC-06) — `da3b34ef`
- [x] **U16:** Reconcile roadmap, todo, changelog, registries, v1.2.0 MoSCoW,
  Mermaid design, GitHub/Project 28, canonical C16 and remaining external
  gates. (AC-01, AC-07) — `669642b8`
- [x] **U17:** Run final automated review, full local validation and hosted
  required checks. PR #712 final exact head
  `1048c4bc4354acdb0c468da17cb0b5d581961690` completed 65 terminal
  conclusions (60 successes, four governed skips and one neutral conclusion),
  with no pending or failed checks; both review threads were resolved before
  squash merge `b8395abfc603509a2f1a2c87c9c33e6260fb5840` at
  `2026-07-31T16:33:52Z`. Only delivery subissues #694–#697 are closure-
  eligible. VoC remains a presentation/delegating alias; scientific review,
  Rust/R/Julia parity, stable promotion, release, parent #595 closure and
  umbrella #318 closure remain pending. (AC-08) — `1048c4bc`

## Phase 5 — Orchestrated scientific remediation and re-review

- [x] **U18:** **Legacy follow-up (not part of completed track acceptance):** Freeze the candidate and run the umbrella Phase 5 four-role
  panel against expected-utility pricing and every VoC presentation.
  (M16-U5, M16-U6; AC-01–AC-05, AC-08)
- [~] **U19:** Implement stable CRRA log-limit/expm1 evaluation and inversion
  near risk aversion one with high-precision sweeps, continuity, policy/tie
  invariance and root-residual evidence. (M16-U5; AC-03, AC-05, AC-06)
- [x] **U20:** **Legacy follow-up (not part of completed track acceptance):** Bind the selected presentation and version into deterministic
  presentation provenance without duplicating the numerical kernel; execute
  expected-utility and VoC fixtures from a clean installed wheel with packaged
  schemas and capabilities. (M16-U3, M16-U6; AC-03–AC-07)
- [x] **U21:** **Legacy follow-up (not part of completed track acceptance):** Remediate through nested #595 issues, rebind the candidate and
  obtain affected-role plus named independent human re-review; retain parity,
  promotion, release and closure as separate gates. (AC-01, AC-05, AC-08)

<!-- Historical contract markers retained for panel-corpus tests: - [ ] **U18:** - [ ] **U19:** - [ ] **U20:** - [ ] **U21:** -->
