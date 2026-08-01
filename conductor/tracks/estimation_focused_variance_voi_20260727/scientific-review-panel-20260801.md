# Scientific review panel — estimation-focused variance VOI

Date: 2026-08-01  
Scope: repository-owned experimental scalar variance-reduction contract  
Panel: estimand/conditioning, numerical/reproducibility, and API/boundary/maturity reviewers

## Disposition

**PASS for the experimental scalar scope.** The panel found no unresolved
Critical, High, or Medium finding that prevents retaining the implemented
Rust/Python scalar population-variance methods as experimental. E17 is
satisfied as a repository review-panel gate only; this does not authorize
stable promotion, release, or issue closure.

## Findings

- The EVPPI_var and EVSI_var estimands, conditioning, population-variance
  convention, sampling assumptions, tolerances, and decision-VOI separation
  are explicit and supported by independent analytical, enumerable, property,
  and pathology fixtures.
- The Rust numerical kernels and seeded bootstrap diagnostics are deterministic
  for the supported scope. `monte_carlo_standard_error` is the standard
  deviation of bootstrap replicate reductions (a bootstrap estimator standard
  error); it should not be interpreted as an integration SD divided by
  `sqrt(B)`.
- Vector targets are fail-closed. The schema declares a covariance functional,
  but runtime execution rejects unsupported vector scalarization rather than
  silently selecting trace, determinant, or a weighted quadratic form.
- R and Julia remain explicitly unsupported for this family because no shared
  estimation-variance C ABI is present; Mojo remains an external integration
  boundary.

## Remaining gates

Stable promotion remains blocked on a separately approved vector covariance
functional, validation of caller-supplied sampling/posterior models and
weighting semantics, cross-language/reference parity, release evidence, and
governed parent/subissue closure. E18 remains pending for those reasons.

Panel members supplied evidence-only reviews; the maintainer retains all
stable-promotion, release, and publication decisions.
