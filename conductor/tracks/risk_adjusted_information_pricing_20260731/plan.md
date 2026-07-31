# Implementation Plan

## Phase 1 — Contract and governance reconciliation

- [x] **U1:** Freeze the expected-utility decision problem, named utility,
  wealth/reference, units, probability, information, scope, cost-location and
  tie contracts. (AC-02) — `8aa2ee6`
- [x] **U2:** Freeze EUI, CEI, BPI, SPI, PPI, policy re-optimization, signed-value,
  root-diagnostic and comparability semantics. (AC-03, AC-05)
  — `0177bcb`
- [x] **U3:** Freeze VoC as a presentation/delegating alias and audit adjacent
  EVPI, CVaR, risk-sensitive, preference and buying-price surfaces. (AC-04,
  AC-07) — `67861ed`
- [ ] **U4:** Materialize GitHub subissues and synchronize Project 28,
  requirements, Mermaid design, cross-references and canonical C16 planning.
  (AC-01)
- [ ] **U5:** Run automated contract review and full Conductor validation.
  (AC-01–AC-05)

## Phase 2 — Reference evidence before implementation

- [ ] **U6:** Add failing independent affine, exponential and nonlinear
  reference tests for EUI, CEI, BPI, SPI, PPI and affine EVPI reduction. (AC-03,
  AC-05)
- [ ] **U7:** Add failing property and pathological tests for positive-affine
  invariance, buy/sell asymmetry, ties, nonuniform probabilities, utility
  domains, bracketing/nonconvergence and stakeholder comparability. (AC-05)
- [ ] **U8:** Add versioned schemas, normative fixtures, deterministic
  serialization and root-diagnostic contracts. (AC-02, AC-03, AC-05)
- [ ] **U9:** Run independent evidence and numerical-boundary review. (AC-05,
  AC-08)

## Phase 3 — Rust runtime and Python presentation

- [ ] **U10:** Implement the accepted Rust utility, policy, information-value,
  price and bounded root-solving kernels with versioned result envelopes.
  (AC-02, AC-03, AC-06)
- [ ] **U11:** Implement the thin Python facade, deterministic serialization,
  DecisionAnalysis integration and VoC presentation/delegation without a
  duplicate kernel. (AC-03, AC-04, AC-06)
- [ ] **U12:** Add differential, property, serialization, failure and
  performance assurance for the runtime boundary. (AC-05, AC-08)
- [ ] **U13:** Run automated implementation review, focused validation and the
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
