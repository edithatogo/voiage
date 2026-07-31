# Implementation Plan

## Phase 1 — Scientific classification and contract

- [x] **E1:** Add the estimation-focused method-registry classifications and
  aliases, explicitly separating variance VOI from decision VOI, sensitivity
  indices and estimator uncertainty. (AC-01) [d510256f]
- [x] **E2:** Freeze scalar/vector target shape, component units, variance or
  covariance functional, conditioning and sampling models, zero-variance
  behavior and finite-sample diagnostics after scientific review. (AC-01,
  AC-02) [ea1e2620] *(experimental contract frozen; the stable-promotion
  scientific-review gate remains pending)*
- [x] **E3:** Add versioned input/result schemas and compatibility fixtures.
  (AC-02) [bee7fb91]
- [x] **E4:** Run an automated contract review and full Conductor validation.
  (AC-01, AC-02, AC-07) [30425456]

## Phase 2 — Reference evidence before runtime

- [x] **E5:** Add failing analytical and enumerable reference tests for
  `EVPPI_var` and `EVSI_var`. (AC-03) [a9e23ee3]
- [x] **E6:** Add failing property tests for variance decompositions,
  zero/perfect-information limits and supported monotonicity cases. (AC-03,
  AC-04) [86c2fef4]
- [x] **E7:** Add failing error and pathological tests for non-finite,
  degenerate, insufficient-sample and non-convergent inputs. (AC-04)
  [474dfa1f]
- [x] **E8:** Review reference independence, tolerances and fixture provenance.
  (AC-03, AC-04) [18fcb5eb]

## Phase 3 — Runtime and assurance

- [x] **E9:** Implement the accepted Rust numerical kernels and versioned result
  envelopes, or record a reviewed exclusion if stable implementation is not
  scientifically supportable. (AC-05) [a3e87106]
- [x] **E10:** Add the thin Python façade, typed diagnostics and deterministic
  serialization without numerical fallback. (AC-02, AC-05) [1287807e]
- [x] **E11:** Add Monte Carlo uncertainty, convergence diagnostics, property
  assurance and benchmarks. (AC-04, AC-08) [9093cc13]
- [x] **E12:** Run automated implementation review, focused tests, differential
  checks and the repository harness. (AC-05, AC-08) [f28f65ec]

## Phase 4 — User and polyglot surfaces

- [x] **E13:** Add CLI, reporting, provenance and accessible example/plot
  surfaces. (AC-06) [f891ff56]
- [x] **E14:** Record Rust, Python, R, Julia and Mojo capability dispositions
  and shared-fixture evidence or explicit unsupported failures. (AC-06)
  [a8895c90]
- [x] **E15:** Reconcile the roadmap, method registry, documentation, bindings,
  v1.2.0 MoSCoW requirements, Mermaid design, canonical C16, GitHub issues,
  Project 28 and Conductor evidence. (AC-06, AC-07, AC-09) [1a70bc24]

## Phase 5 — Final review and hosted validation

- [x] **RF1:** Apply final automated-review fixes for formatter, CLI registry,
  extension policy and curated package-export contract coverage. (AC-06,
  AC-08) [0f61ea9a]
- [x] **RF2:** Preserve the immutable v1.0 extension-policy snapshot while
  explicitly reconciling the separately governed post-v1 estimation module.
  (AC-06, AC-08) [cde857ac]
- [x] **RF3:** Remove the experimental registry helper from the curated package
  export list after hosted CodeQL identified it as an explicit undefined
  export; retain access through the governed estimation module. (AC-06, AC-08)
  [589a91c7]
- [x] **RF4:** Exercise every changed validation, native-boundary, CLI and plot
  branch after the hosted changed-line policy exposed incomplete assurance;
  reject boolean relative reductions consistently with the other numeric
  fields. (AC-04, AC-06, AC-08) [2b8f7393]
- [x] **RF5:** Reconcile the retained R package's embedded release-workflow
  assertion with the job-scoped write permission required to publish immutable
  source and manual assets. (AC-06, AC-08) [dc6ddd4d]
- [x] **E16:** Run final automated review, full local validation and hosted
  required checks; retain merge, release and issue closure as separate gates.
  (AC-08) [d70fff63]

## Phase 6 — External promotion and delivery gates

- [ ] **E17:** Obtain scientific classification review before any stable
  method-registry promotion, including a reviewed disposition for vector
  covariance scalarization. (AC-01, AC-02)
- [ ] **E18:** Rebase onto the then-current protected base, merge the approved
  implementation and canonical-sync pull requests, and handle stable
  promotion, release and issue closure only through their separate governed
  workflows. (AC-07, AC-09)
