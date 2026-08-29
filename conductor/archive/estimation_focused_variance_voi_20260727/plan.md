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
- [x] **RF6:** Remediate independent review findings by enforcing explicit
  prior-predictive EVSI weighting, input-bound replay digests and scalar
  covariance/functional/unit consistency, with unequal-probability reference,
  runtime and pathology coverage. Preserve experimental maturity and the
  pending vector-scientific-review boundary. (AC-02–AC-05, AC-08)
- [x] **RF6a:** Remediate the fresh boundary review by rejecting EVSI sampling
  models that do not declare prior-predictive averaging, computing bootstrap
  replicate means without revalidating internally generated floating weights,
  and declaring the portable result schema's mandatory semantic-validation
  layer. Add zero-tolerance, non-power-of-two, CLI and schema regressions.
  (AC-02–AC-05, AC-08)
- **Migrated:** **RF7:** Obtain fresh exact-head hosted checks and merge the dedicated
  #619 remediation before reconciling any dependent umbrella pull request.
  (AC-07, AC-08)
- [x] **E16:** Run final automated review, full local validation and hosted
  required checks; retain merge, release and issue closure as separate gates.
  (AC-08) [d70fff63]

## Phase 6 — External promotion and delivery gates

- **Migrated:** **E17:** Obtain scientific classification review before any stable
  method-registry promotion, including a reviewed disposition for vector
  covariance scalarization. (AC-01, AC-02)
- [x] **E18:** Rebase onto the then-current protected base, merge the approved
  implementation and canonical-sync pull requests, and handle stable
  promotion, release and issue closure only through their separate governed
  workflows. (AC-07, AC-09) *(VOIAGE PR #676 exact head
  `5e2c097fbdda8965d1907d7e930e910238fa24da`: 65 terminal contexts, 60
  successes, four governed skips, one neutral CodeQL aggregation, zero bad or
  pending, two resolved review threads; squash merge
  `9495fc3f372b9564701a180c6cf611a3ddc010dd` at
  `2026-07-31T16:57:49Z`. Canonical VOP sync PR #64 exact head
  `6c3fd72358f3feef6c542e0a374d7ea74889f915`: 16 terminal contexts, 15
  successes, one governed skip, zero bad or pending, zero review threads;
  squash merge `cedc6fbb17a5d999cb12bb300a01f87d976ec02e` at
  `2026-08-01T03:38:52Z`. This makes delivery subissues #671--#674 eligible
  for closure but does not satisfy E17 or authorize parent #619, umbrella
  #318, stable-promotion or release closure.)*

## Phase 7 — Orchestrated scientific remediation and re-review

- **Migrated:** **E19:** Under umbrella Phase 5, freeze a candidate-bound review packet
  and run the estimand/domain, estimator-assurance, cross-language/API and
  governance/publication panel roles. (M14-E6–M14-E9; AC-01–AC-04, AC-08)
- [x] **E20:** Bind the declared target, conditioning, design, likelihood,
  sampling model and solver request to runtime inputs and replay provenance;
  distinguish exact, outer, nested and coupled estimator designs. (M14-E6,
  M14-E7; AC-02–AC-05) `9d85df1`
- [x] **E21:** Add truth-known bias, RMSE, coverage, calibration, convergence
  and dependence-preserving nested/coupled assurance plus independently
  executable Rust/Python EVPPI and EVSI fixtures. (M14-E7, M14-E9; AC-03–
  AC-06, AC-08) `9a20b6a`
- [x] **E22a:** Harden the pre-review vector boundary: retain trace,
  determinant and weighted quadratic only as reserved schema vocabulary;
  reject every vector runtime request before native dispatch and every vector
  result envelope during semantic validation; declare that no conformant
  vector result/runtime exists; and add fail-closed schema, capability,
  dispatch and matrix-case tests. This repository-owned remediation does not
  satisfy E22 or the scientific-review gate. (M14-E8a; AC-02–AC-05) `4397f16`
- **Migrated:** **E22:** Complete vector covariance scientific review with PSD policy,
  units, scalarization, regularization, recomputation and multivariate oracles;
  otherwise retain a reviewed vector exclusion. (M14-E8; AC-02–AC-05)
- **Migrated:** **E23:** Remediate findings in nested #619 issues, rebind the candidate,
  obtain fresh affected-role and named independent human review, and retain
  parity, promotion, release and closure as separate gates. (AC-01, AC-07–
  AC-09)

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/tracks/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
