# Implementation Plan

## Phase 1 — Scientific classification and contract

- [x] **E1:** Add the estimation-focused method-registry classifications and
  aliases, explicitly separating variance VOI from decision VOI, sensitivity
  indices and estimator uncertainty. (AC-01) [3e4f4759]
- [x] **E2:** Freeze scalar/vector target shape, component units, variance or
  covariance functional, conditioning and sampling models, zero-variance
  behavior and finite-sample diagnostics after scientific review. (AC-01,
  AC-02) [4e6cdd3f] *(experimental contract frozen; the stable-promotion
  scientific-review gate remains pending)*
- [x] **E3:** Add versioned input/result schemas and compatibility fixtures.
  (AC-02) [a76b5cac]
- [x] **E4:** Run an automated contract review and full Conductor validation.
  (AC-01, AC-02, AC-07) [12fc17d1]

## Phase 2 — Reference evidence before runtime

- [x] **E5:** Add failing analytical and enumerable reference tests for
  `EVPPI_var` and `EVSI_var`. (AC-03) [a8e3678c]
- [x] **E6:** Add failing property tests for variance decompositions,
  zero/perfect-information limits and supported monotonicity cases. (AC-03,
  AC-04) [7fc07713]
- [x] **E7:** Add failing error and pathological tests for non-finite,
  degenerate, insufficient-sample and non-convergent inputs. (AC-04)
  [7bb8b0d1]
- [x] **E8:** Review reference independence, tolerances and fixture provenance.
  (AC-03, AC-04) [6f53e9cd]

## Phase 3 — Runtime and assurance

- [x] **E9:** Implement the accepted Rust numerical kernels and versioned result
  envelopes, or record a reviewed exclusion if stable implementation is not
  scientifically supportable. (AC-05) [d73a3c0a]
- [x] **E10:** Add the thin Python façade, typed diagnostics and deterministic
  serialization without numerical fallback. (AC-02, AC-05) [4e5f624b]
- [x] **E11:** Add Monte Carlo uncertainty, convergence diagnostics, property
  assurance and benchmarks. (AC-04, AC-08) [e306dd2d]
- [x] **E12:** Run automated implementation review, focused tests, differential
  checks and the repository harness. (AC-05, AC-08) [37f320b9]

## Phase 4 — User and polyglot surfaces

- [x] **E13:** Add CLI, reporting, provenance and accessible example/plot
  surfaces. (AC-06) [f0eaecc5]
- [x] **E14:** Record Rust, Python, R, Julia and Mojo capability dispositions
  and shared-fixture evidence or explicit unsupported failures. (AC-06)
  [656d06e8]
- [x] **E15:** Reconcile the roadmap, method registry, documentation, bindings,
  v1.2.0 MoSCoW requirements, Mermaid design, canonical C16, GitHub issues,
  Project 28 and Conductor evidence. (AC-06, AC-07, AC-09) [23174bc2]
- [~] **E16:** Run final automated review, full local validation and hosted
  required checks; retain merge, release and issue closure as separate gates.
  (AC-08)
