# Implementation Plan

## Phase 1 — Scientific classification and contract

- [x] **E1:** Add the estimation-focused method-registry classifications and
  aliases, explicitly separating variance VOI from decision VOI, sensitivity
  indices and estimator uncertainty. (AC-01) [df061cc0]
- [x] **E2:** Freeze scalar/vector target shape, component units, variance or
  covariance functional, conditioning and sampling models, zero-variance
  behavior and finite-sample diagnostics after scientific review. (AC-01,
  AC-02) [40c91636] *(experimental contract frozen; the stable-promotion
  scientific-review gate remains pending)*
- [x] **E3:** Add versioned input/result schemas and compatibility fixtures.
  (AC-02) [82c72afa]
- [x] **E4:** Run an automated contract review and full Conductor validation.
  (AC-01, AC-02, AC-07) [a1a4af31]

## Phase 2 — Reference evidence before runtime

- [x] **E5:** Add failing analytical and enumerable reference tests for
  `EVPPI_var` and `EVSI_var`. (AC-03) [9026751f]
- [x] **E6:** Add failing property tests for variance decompositions,
  zero/perfect-information limits and supported monotonicity cases. (AC-03,
  AC-04) [144e90a0]
- [ ] **E7:** Add failing error and pathological tests for non-finite,
  degenerate, insufficient-sample and non-convergent inputs. (AC-04)
- [ ] **E8:** Review reference independence, tolerances and fixture provenance.
  (AC-03, AC-04)

## Phase 3 — Runtime and assurance

- [ ] **E9:** Implement the accepted Rust numerical kernels and versioned result
  envelopes, or record a reviewed exclusion if stable implementation is not
  scientifically supportable. (AC-05)
- [ ] **E10:** Add the thin Python façade, typed diagnostics and deterministic
  serialization without numerical fallback. (AC-02, AC-05)
- [ ] **E11:** Add Monte Carlo uncertainty, convergence diagnostics, property
  assurance and benchmarks. (AC-04, AC-08)
- [ ] **E12:** Run automated implementation review, focused tests, differential
  checks and the repository harness. (AC-05, AC-08)

## Phase 4 — User and polyglot surfaces

- [ ] **E13:** Add CLI, reporting, provenance and accessible example/plot
  surfaces. (AC-06)
- [ ] **E14:** Record Rust, Python, R, Julia and Mojo capability dispositions
  and shared-fixture evidence or explicit unsupported failures. (AC-06)
- [ ] **E15:** Reconcile the roadmap, method registry, documentation, bindings,
  v1.2.0 MoSCoW requirements, Mermaid design, canonical C16, GitHub issues,
  Project 28 and Conductor evidence. (AC-06, AC-07, AC-09)
- [ ] **E16:** Run final automated review, full local validation and hosted
  required checks; retain merge, release and issue closure as separate gates.
  (AC-08)
