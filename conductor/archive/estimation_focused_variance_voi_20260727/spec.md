# Estimation-Focused Variance-Reduction VOI

## Overview

Add governed value-of-information estimands for analyses whose objective is to
estimate a declared model output accurately rather than choose among actions.
The track introduces variance-reduction EVPPI and EVSI contracts without
changing or aliasing VOIAGE's stable decision-focused EVPPI and EVSI methods.
The planned contract version is v1.2.0 and its MoSCoW source is
`requirements.md`, tracing to canonical requirements M14 and M17.

GitHub issue
[#619](https://github.com/edithatogo/voiage/issues/619) is the native
sub-issue of frontier programme issue
[#318](https://github.com/edithatogo/voiage/issues/318), under programme
[#313](https://github.com/edithatogo/voiage/issues/313). All three are tracked
in [Project 28](https://github.com/users/edithatogo/projects/28).

## Requirements

### R1 — Estimands and boundaries

- Define scalar-target partial-perfect-information variance reduction:
  \[
  EVPPI_{var}(S)
  = Var(g(\theta)) - E[Var(g(\theta)\mid\theta_S)].
  \]
- Define scalar-target sample-information variance reduction:
  \[
  EVSI_{var}(d)
  = Var(g(\theta)) - E_Y[Var(g(\theta)\mid Y,d)].
  \]
- Evaluate the outer expectation using explicit prior-predictive probabilities
  aligned one-for-one with posterior-variance evaluations. Probabilities must
  be finite, nonnegative and sum to one within the estimator's declared
  numerical tolerance; arbitrary positive weights are rejected rather than
  silently normalized.
- Declare whether \(g(\theta)\) is scalar or vector, its component units, the
  prior, parameter subset or study design, sampling model/likelihood,
  conditioning sigma-field and averaging convention, estimator, seed and
  numerical tolerances.
- Keep these estimands distinct from decision-focused EVPPI/EVSI, global
  sensitivity indices, posterior estimator variance and EVSI estimator error.

### R2 — Result contracts

- Return prior variance, expected posterior variance, absolute variance
  reduction and relative variance reduction.
- Carry estimator uncertainty, convergence and degeneracy diagnostics,
  method settings, provenance and deterministic serialization.
- Bind separate deterministic digests to the scientific specification and the
  actual runtime values so replay identity changes when either changes.
- For the executable scalar surface, require each 1-by-1 covariance entry to
  be nonnegative and equal its variance functional, and require functional
  units to be exactly the squared target component units.
- Define behavior for zero prior variance, zero information, perfect
  information, non-finite inputs and finite-sample negative estimates.
- Treat scalar-target variance as the first supported functional. Vector
  targets must return the declared prior and expected posterior covariance
  objects and require a separately reviewed covariance functional—such as
  trace, determinant or a declared weighted quadratic form—to produce a scalar
  reduction. Never select or silently change that functional from data shape.

### R3 — Runtime and surfaces

- Put stable numerical policy in the Rust core if the method is promoted.
- Keep Python as the reference façade and provide explicit capability
  dispositions for Rust, Python, R, Julia and Mojo.
- Add versioned schemas, compatibility fixtures, diagnostics, CLI/reporting
  surfaces and an accessible plot or tabular comparison where scientifically
  justified.
- Prevent base imports from requiring optional estimator dependencies.

### R4 — Assurance

- Test conjugate analytical examples and independently enumerable discrete
  references.
- Test law-of-total-variance identities, zero/perfect-information limits,
  supported monotonicity conditions, reproducibility and serialization.
- Add invalid, degenerate, non-finite, insufficient-sample and
  non-convergence cases.
- Record numerical tolerances, Monte Carlo error, convergence evidence and
  performance baselines before any maturity promotion.

## Acceptance criteria

- **AC-01:** The method registry and documentation distinguish
  `EVPPI_var`/`EVSI_var` from decision-focused EVPPI/EVSI and adjacent
  sensitivity or estimator-error concepts.
- **AC-02:** Versioned input and result schemas declare scalar/vector target
  shape, component units, variance or covariance functional, conditioning and
  sampling models, diagnostics and provenance.
- **AC-03:** Analytical and brute-force fixtures verify the estimands and
  law-of-total-variance identities.
- **AC-04:** Property, edge, error and convergence tests cover the declared
  scientific envelope.
- **AC-05:** Executable runtime evidence or a reviewed exclusion exists for
  each method, with no documentation-only completion claim.
- **AC-06:** Rust/Python/R/Julia/Mojo capability dispositions, CLI, reporting,
  examples and maturity registry agree.
- **AC-07:** GitHub #619, parent #318, programme #313, Project 28, this track
  and the central cross-reference manifest remain bidirectionally linked.
- **AC-08:** Automated review, focused validation, the repository harness and
  hosted required checks are green before repository completion.
- **AC-09:** The v1.2.0 MoSCoW requirements, Mermaid design, canonical C16
  projection, GitHub hierarchy and Project 28 fields remain synchronized.

## Non-functional constraints

- Results must be deterministic for a declared seed within the repository's
  numerical reproducibility policy.
- Public contracts must use finite validation, typed diagnostics and
  versioned serialization.
- The stable installation must remain independent of optional research
  estimators.
- Existing v1 decision-focused method behavior and wire formats must remain
  backward compatible.

## External and human gates

- Scientific review must approve terminology, conditioning and vector-target
  policy before stable registry promotion.
- Language registry publication, release dispatch and external indexing remain
  separate from repository completion.
- GitHub issue closure and merge require the repository's protected-branch
  checks; planning alone does not satisfy implementation acceptance.

## Out of scope

- Replacing decision-focused EVPPI or EVSI.
- Treating Sobol or other global-sensitivity indices as variance VOI.
- Entropy, mutual-information or general decision-loss metrics without a
  separately approved extension.
- Selecting an unreviewed scalarization for vector or functional targets.

## Authoritative inputs

- User-approved feature description in the 2026-07-27 Codex task.
- GitHub issue
  [#619](https://github.com/edithatogo/voiage/issues/619), live revision
  created 2026-07-27.
- Frontier parent
  [#318](https://github.com/edithatogo/voiage/issues/318) and programme
  [#313](https://github.com/edithatogo/voiage/issues/313), live revisions
  updated 2026-07-27.
- `specs/v1/stable-api.json` and `specs/v1/README.md` at repository baseline
  `ceefb515`, which define the existing decision-focused stable surface.
- `conductor/product.md`, `conductor/product-guidelines.md`,
  `conductor/tech-stack.md` and `conductor/workflow.md`.
- `conductor/requirements.md`, `conductor/design.md`, this track's
  `requirements.md` and `design.md`, and canonical cross-repository C16.
