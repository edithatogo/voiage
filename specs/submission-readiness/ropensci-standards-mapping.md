# rOpenSci statistical-software standards mapping for `voiageR`

**Package:** `voiageR` 2.2.0
**Pinned standards:** rOpenSci Statistical Software Peer Review standards
revision `974cd8f0d73961235c74bfa34b78086d39fd8817` and `srr` revision
`d186fe6f93657805ed86177f03333c478e136709`.

## Category disposition

`voiageR` is mapped to the general and Probability Distributions standards.
EVPI and ENBS process an empirical joint distribution represented by finite
draws and compute documented expectation transforms, so the distribution
category applies to that bounded surface. Eight of its fourteen unique items
are implemented; the six inapplicable items concern parametric distribution
objects, named parametric families, or optimisation routines that are not part
of this non-parametric empirical algorithm.

The Bayesian and Monte Carlo category was reviewed and rejected as a package
classification: the R package does not specify priors, fit Bayesian models,
implement a sampler, run chains, estimate posterior distributions, diagnose
convergence, or return posterior objects. It accepts already-produced
uncertainty draws for EVPI and delegates the optional bounded EVSI method to the
separately installed Python package. Calling an optional dependency does not
make the R source package an implementation of that dependency's Bayesian or
Monte Carlo algorithms.

These category boundaries avoid both false compliance claims and blanket
`@srrstatsNA` use to force an inapplicable category through the `srr` threshold.
They must be revisited if `voiageR` later owns parametric distribution objects,
sampling, posterior estimation, or convergence behavior.

## Item-level evidence

- Applicable documentation and input standards are tagged at their R
  implementation evidence in `R/srr-runtime-standards.R` and
  `R/srr-input-standards.R`.
- Numerical floating-point evidence is tagged at the packaged Rust EVPI kernel.
- Testing standards are tagged in `tests/testthat/test-voiageR.R` and exercised
  across the installed native, validation, deterministic-seed, perturbation,
  and shared numerical-reference suites.
- Every non-applicable general standard has its own justified `@srrstatsNA`
  entry in the required `NA_standards` block in
  `R/srr-stats-standards.R`.
- There are no `@srrstatsTODO` tags.

The pinned `srr::srr_stats_pre_submit()` run is the executable completeness
gate. A successful run establishes mapping completeness only; `pkgcheck`,
coverage, vignettes, examples, source installation, and supported-platform
checks remain separate evidence.

## Packaging boundary

The package is now self-contained for EVPI and ENBS: its 2.2.0 source archive
builds a dependency-free Rust static library offline and links registered R
native routines. Python and `reticulate` remain optional for EVPPI and EVSI.
No ambient VOIAGE shared library is required.

The current distribution receipt is
`specs/submission-readiness/r-distribution-evidence-20260902.json`. It binds the
immutable source archive and full manual check to the successful hosted R and
retained-bindings matrix, and verifies that the tested R package, FFI,
numerical-reference and workflow Git objects are unchanged at the recorded
current revision. This is repository distribution evidence, not CRAN or
rOpenSci submission or acceptance.
