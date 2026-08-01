# Independent review remediation — 2026-08-02

An independent review of the merged experimental #619 delivery found one High
and two Medium issues:

1. `EVSI_var` averaged enumerated posterior variances uniformly rather than by
   their prior-predictive probabilities.
2. replay provenance hashed the specification but did not bind the actual
   runtime values.
3. the scalar result model accepted contradictions between covariance entries,
   functionals and units.

The dedicated remediation makes predictive probabilities an explicit runtime
input from CLI and Python through PyO3 to Rust. The Rust kernel rejects
misaligned, non-finite, negative or non-unit-total probability vectors using
the estimator's declared numerical tolerance, and the bootstrap samples the
posterior stage according to that categorical law. Provenance now carries a
separate canonical SHA-256 digest of actual method inputs. Scalar results
require nonnegative 1-by-1 covariance entries equal to their functionals and
units exactly equal to the squared target units.

Unequal-probability analytical/runtime tests and probability, digest,
covariance and unit pathologies cover the corrections. The surface remains
experimental; vector covariance execution and scalarization remain pending
scientific review. Fresh hosted checks and merge remain mandatory and are not
claimed by this local remediation record.

A fresh independent boundary review then found that non-prior-predictive EVSI
specifications were still accepted, that bootstrap could reject valid inputs
at zero tolerance by revalidating internally generated equal weights, and that
portable JSON Schema validation alone could not prove cross-field scalar
covariance, functional and unit equality. The follow-up fails closed unless
EVSI declares prior-predictive averaging, calculates the sampled bootstrap mean
directly, constrains scalar schema shape/nonnegativity/unit syntax, and marks
the Pydantic result contract as the mandatory semantic validator for the
cross-field identities. A clean, read-only independent re-review at exact
signed commit `9e860887293dfcd2f95ce0648e25118026141bc5` reproduced all
original and boundary cases and reported zero unresolved Critical, High or
Medium findings. Hosted exact-head checks and merge remain required.
