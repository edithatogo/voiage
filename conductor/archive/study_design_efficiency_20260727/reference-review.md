# Reference evidence review

Review date: 2026-07-31. Scope: Phase 2 red tests for issue #571.

## Independence

The COSS reference calculates every signed ENBS value directly in Python as
`evsi - research_cost` and independently applies `max` to the enumerated curve.
It does not call the future COSS implementation, Rust kernel, stable `enbs`,
legacy optimizer or plotting helper. The expected all-negative curve also
detects accidental use of the zero-floored `enbs_simple` compatibility paths.

Expected results are fixed from small enumerable examples rather than copied
from runtime output. Interior, lower/upper boundary, tied, infeasible and
non-monotone cases exercise distinct selection branches. Tie expectations are
stated for each supported policy and do not rely on incidental array sorting.

## Cost provenance

Tests construct research cost explicitly in the same declared `AUD_2026`
value unit, population scale, horizon and discounting context as EVSI. The
context names `trial-cost-v1` as the cost model. Runtime tests do not imply
that omitted opportunity, delay or implementation costs are zero in real
analyses; those inclusions remain the caller's documented cost-model choice.

Signed negative ENBS is retained. A negative maximum chooses the best supplied
study design but does not imply that commissioning a study dominates no study.
A caller must include an explicit zero-cost no-study design when that option is
part of the decision.

## Estimator and selection uncertainty

Marginal EVSI and cost standard errors do not determine ENBS uncertainty
without their covariance. Tests will require either directly estimated ENBS
uncertainty or an explicit unavailable state; the runtime may not assume
independence. Likewise, uncertainty in the selected design requires joint
replicates or externally supplied selection frequencies. Marginal pointwise
standard errors cannot be converted into selection probabilities.

The red suite covers direct ENBS intervals, unavailable uncertainty, selection
probabilities and confidence sets before Phase 3 implementation begins.
Uncertainty remains descriptive and cannot silently replace the deterministic
point-estimate tie policy.

## Efficiency reference

The EVSI/EVPI reference uses exact arithmetic for ordinary and scaled cases.
The two values carry separate context objects so mismatched unit/scaling tests
cannot be bypassed by a shared unvalidated argument. Near-bound excursions are
checked against an explicitly supplied value-scale tolerance and retain the
raw ratio; material theoretical violations must fail closed.

## Review result

Pass. The red suite is intentionally uncollectable until the Phase 3 contract
module and experimental façade are implemented. This expected red state is
evidence of TDD sequencing, not runtime completion.
