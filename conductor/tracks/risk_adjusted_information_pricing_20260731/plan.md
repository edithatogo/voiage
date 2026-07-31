# Implementation Plan

## Phase 1 — Contract and governance reconciliation — checkpoint `f629cc34`

- [x] **U1:** Freeze the expected-utility decision problem, named utility,
  wealth/reference, units, probability, information, scope, cost-location and
  tie contracts. (AC-02) — `a6d8c707`
- [x] **U2:** Freeze EUI, CEI, BPI, SPI, PPI, policy re-optimization, signed-value,
  root-diagnostic and comparability semantics. (AC-03, AC-05)
  — `7f208e22`
- [x] **U3:** Freeze VoC as a presentation/delegating alias and audit adjacent
  EVPI, CVaR, risk-sensitive, preference and buying-price surfaces. (AC-04,
  AC-07) — `78d99d53`
- [x] **U4:** Materialize GitHub subissues and synchronize Project 28,
  requirements, Mermaid design, cross-references and canonical C16 planning.
  (AC-01) — `6f788faf`
- [x] **U5:** Run automated contract review and full Conductor validation.
  (AC-01–AC-05) — `5c5ad190`

### Phase 1 review fixes

- [x] Preserve #694–#697 as nested #595 subissues in the canonical C16
  projection rather than projecting them as independent frontier issues.
  — `34d40fc0`
- [x] Reconcile deterministic tie selection, finite-signal design, and complete
  tie-set transition diagnostics across the frozen contract artifacts.
  — `5c5ad190`

## Phase 2 — Reference evidence before implementation — checkpoint `8df63a6f`

- [x] **U6:** Add failing independent affine, exponential and nonlinear
  reference tests for EUI, CEI, BPI, SPI, PPI and affine EVPI reduction. (AC-03,
  AC-05) — `546a455e`
- [x] **U7:** Add failing property and pathological tests for positive-affine
  invariance, buy/sell asymmetry, ties, nonuniform probabilities, utility
  domains, bracketing/nonconvergence and stakeholder comparability. (AC-05)
  — `0b4377f9`
- [x] **U8:** Add versioned schemas, normative fixtures, deterministic
  serialization and root-diagnostic contracts. (AC-02, AC-03, AC-05)
  — `1f769a2f`
- [x] **U9:** Run independent evidence and numerical-boundary review. (AC-05,
  AC-08) — `fc1f4e61`

### Phase 2 review fixes

- [x] Add PPI, stakeholder-comparability and bounded solver-limit red
  contracts; fully discriminate nested result diagnostics; require cost,
  currency and price-date semantics; exclude CRRA risk aversion one; and
  validate a complete normative result. — `fc1f4e61`

## Phase 3 — Rust runtime and Python presentation — checkpoint `ec09d4e8`

- [x] **U10:** Implement the accepted Rust utility, policy, information-value,
  price and bounded root-solving kernels with versioned result envelopes.
  (AC-02, AC-03, AC-06) — `fad9c17a`
- [x] **U11:** Implement the thin Python facade, deterministic serialization,
  DecisionAnalysis integration and VoC presentation/delegation without a
  duplicate kernel. (AC-03, AC-04, AC-06) — `20866c78`
- [x] **U12:** Add differential, property, serialization, failure and
  performance assurance for the runtime boundary. (AC-05, AC-08)
  — `8c717623`
- [x] **U13:** Run automated implementation review, focused validation and the
  repository harness. (AC-05, AC-08) — `48573b54`

### Phase 3 review fixes

- [x] Replace fabricated root diagnostics; preserve canonical per-signal
  policies and explicit domain exclusions; reject discontinuous no-root
  prices; complete validation, provenance, PPI reasons and pairwise
  comparability; and validate affine-only monetary EVPI presentation.
  — `48573b54`

## Phase 4 — User, binding and governed closeout surfaces

- [x] **U14:** Add CLI/reporting, capability discovery, accessible examples
  and documentation that distinguish utility and monetary scales. (AC-07)
  — `ca94c567`
- [x] **U15:** Record and validate Rust, Python, R, Julia and Mojo capability
  dispositions and shared-fixture evidence or explicit unsupported states.
  (AC-06) — `552fb369`
- [x] **U16:** Reconcile roadmap, todo, changelog, registries, v1.2.0 MoSCoW,
  Mermaid design, GitHub/Project 28, canonical C16 and remaining external
  gates. (AC-01, AC-07) — `9ddde39c`
- [~] **U17:** Run final automated review, full local validation and hosted
  required checks; retain merge, stable promotion, release and issue closure
  as separate gates. (AC-08)
