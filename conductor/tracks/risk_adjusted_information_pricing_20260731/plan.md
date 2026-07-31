# Implementation Plan

## Phase 1 — Contract and governance reconciliation — checkpoint `4ec39a3`

- [x] **U1:** Freeze the expected-utility decision problem, named utility,
  wealth/reference, units, probability, information, scope, cost-location and
  tie contracts. (AC-02) — `8aa2ee6`
- [x] **U2:** Freeze EUI, CEI, BPI, SPI, PPI, policy re-optimization, signed-value,
  root-diagnostic and comparability semantics. (AC-03, AC-05)
  — `0177bcb`
- [x] **U3:** Freeze VoC as a presentation/delegating alias and audit adjacent
  EVPI, CVaR, risk-sensitive, preference and buying-price surfaces. (AC-04,
  AC-07) — `67861ed`
- [x] **U4:** Materialize GitHub subissues and synchronize Project 28,
  requirements, Mermaid design, cross-references and canonical C16 planning.
  (AC-01) — `766149f`
- [x] **U5:** Run automated contract review and full Conductor validation.
  (AC-01–AC-05) — `dbf2021`

### Phase 1 review fixes

- [x] Preserve #694–#697 as nested #595 subissues in the canonical C16
  projection rather than projecting them as independent frontier issues.
  — `80e0a61`
- [x] Reconcile deterministic tie selection, finite-signal design, and complete
  tie-set transition diagnostics across the frozen contract artifacts.
  — `dbf2021`

## Phase 2 — Reference evidence before implementation — checkpoint `0aa8018`

- [x] **U6:** Add failing independent affine, exponential and nonlinear
  reference tests for EUI, CEI, BPI, SPI, PPI and affine EVPI reduction. (AC-03,
  AC-05) — `5332ae7`
- [x] **U7:** Add failing property and pathological tests for positive-affine
  invariance, buy/sell asymmetry, ties, nonuniform probabilities, utility
  domains, bracketing/nonconvergence and stakeholder comparability. (AC-05)
  — `f2c4503`
- [x] **U8:** Add versioned schemas, normative fixtures, deterministic
  serialization and root-diagnostic contracts. (AC-02, AC-03, AC-05)
  — `8a7b707`
- [x] **U9:** Run independent evidence and numerical-boundary review. (AC-05,
  AC-08) — `9e51994`

### Phase 2 review fixes

- [x] Add PPI, stakeholder-comparability and bounded solver-limit red
  contracts; fully discriminate nested result diagnostics; require cost,
  currency and price-date semantics; exclude CRRA risk aversion one; and
  validate a complete normative result. — `9e51994`

## Phase 3 — Rust runtime and Python presentation

- [x] **U10:** Implement the accepted Rust utility, policy, information-value,
  price and bounded root-solving kernels with versioned result envelopes.
  (AC-02, AC-03, AC-06) — `dbe1871`
- [x] **U11:** Implement the thin Python facade, deterministic serialization,
  DecisionAnalysis integration and VoC presentation/delegation without a
  duplicate kernel. (AC-03, AC-04, AC-06) — `c179470`
- [x] **U12:** Add differential, property, serialization, failure and
  performance assurance for the runtime boundary. (AC-05, AC-08)
  — `0e17fed`
- [~] **U13:** Run automated implementation review, focused validation and the
  repository harness. (AC-05, AC-08)

## Phase 4 — User, binding and governed closeout surfaces

- [ ] **U14:** Add CLI/reporting, capability discovery, accessible examples
  and documentation that distinguish utility and monetary scales. (AC-07)
- [ ] **U15:** Record and validate Rust, Python, R, Julia and Mojo capability
  dispositions and shared-fixture evidence or explicit unsupported states.
  (AC-06)
- [ ] **U16:** Reconcile roadmap, todo, changelog, registries, v1.2.0 MoSCoW,
  Mermaid design, GitHub/Project 28, canonical C16 and remaining external
  gates. (AC-01, AC-07)
- [ ] **U17:** Run final automated review, full local validation and hosted
  required checks; retain merge, stable promotion, release and issue closure
  as separate gates. (AC-08)
