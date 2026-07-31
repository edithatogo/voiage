# Implementation Plan

## Phase 1 — Contract and governance reconciliation — checkpoint `0582920e`

- [x] **U1:** Freeze the expected-utility decision problem, named utility,
  wealth/reference, units, probability, information, scope, cost-location and
  tie contracts. (AC-02) — `52098b24`
- [x] **U2:** Freeze EUI, CEI, BPI, SPI, PPI, policy re-optimization, signed-value,
  root-diagnostic and comparability semantics. (AC-03, AC-05)
  — `a7e3ede7`
- [x] **U3:** Freeze VoC as a presentation/delegating alias and audit adjacent
  EVPI, CVaR, risk-sensitive, preference and buying-price surfaces. (AC-04,
  AC-07) — `ec9bab1c`
- [x] **U4:** Materialize GitHub subissues and synchronize Project 28,
  requirements, Mermaid design, cross-references and canonical C16 planning.
  (AC-01) — `17a2b4a7`
- [x] **U5:** Run automated contract review and full Conductor validation.
  (AC-01–AC-05) — `709732aa`

### Phase 1 review fixes

- [x] Preserve #694–#697 as nested #595 subissues in the canonical C16
  projection rather than projecting them as independent frontier issues.
  — `5362160c`
- [x] Reconcile deterministic tie selection, finite-signal design, and complete
  tie-set transition diagnostics across the frozen contract artifacts.
  — `709732aa`

## Phase 2 — Reference evidence before implementation — checkpoint `5699ce3a`

- [x] **U6:** Add failing independent affine, exponential and nonlinear
  reference tests for EUI, CEI, BPI, SPI, PPI and affine EVPI reduction. (AC-03,
  AC-05) — `020bf66e`
- [x] **U7:** Add failing property and pathological tests for positive-affine
  invariance, buy/sell asymmetry, ties, nonuniform probabilities, utility
  domains, bracketing/nonconvergence and stakeholder comparability. (AC-05)
  — `b0848a69`
- [x] **U8:** Add versioned schemas, normative fixtures, deterministic
  serialization and root-diagnostic contracts. (AC-02, AC-03, AC-05)
  — `a8db46f9`
- [x] **U9:** Run independent evidence and numerical-boundary review. (AC-05,
  AC-08) — `097bd28a`

### Phase 2 review fixes

- [x] Add PPI, stakeholder-comparability and bounded solver-limit red
  contracts; fully discriminate nested result diagnostics; require cost,
  currency and price-date semantics; exclude CRRA risk aversion one; and
  validate a complete normative result. — `097bd28a`

## Phase 3 — Rust runtime and Python presentation — checkpoint `3338f7d1`

- [x] **U10:** Implement the accepted Rust utility, policy, information-value,
  price and bounded root-solving kernels with versioned result envelopes.
  (AC-02, AC-03, AC-06) — `4a6678e4`
- [x] **U11:** Implement the thin Python facade, deterministic serialization,
  DecisionAnalysis integration and VoC presentation/delegation without a
  duplicate kernel. (AC-03, AC-04, AC-06) — `e2fbda10`
- [x] **U12:** Add differential, property, serialization, failure and
  performance assurance for the runtime boundary. (AC-05, AC-08)
  — `0a24dd52`
- [x] **U13:** Run automated implementation review, focused validation and the
  repository harness. (AC-05, AC-08) — `67e08243`

### Phase 3 review fixes

- [x] Replace fabricated root diagnostics; preserve canonical per-signal
  policies and explicit domain exclusions; reject discontinuous no-root
  prices; complete validation, provenance, PPI reasons and pairwise
  comparability; and validate affine-only monetary EVPI presentation.
  — `67e08243`

## Phase 4 — User, binding and governed closeout surfaces

- [x] **U14:** Add CLI/reporting, capability discovery, accessible examples
  and documentation that distinguish utility and monetary scales. (AC-07)
  — `6e70fae0`
- [x] **U15:** Record and validate Rust, Python, R, Julia and Mojo capability
  dispositions and shared-fixture evidence or explicit unsupported states.
  (AC-06) — `38da97f7`
- [x] **U16:** Reconcile roadmap, todo, changelog, registries, v1.2.0 MoSCoW,
  Mermaid design, GitHub/Project 28, canonical C16 and remaining external
  gates. (AC-01, AC-07) — `aa694307`
- [~] **U17:** Run final automated review, full local validation and hosted
  required checks; retain merge, stable promotion, release and issue closure
  as separate gates. (AC-08)
