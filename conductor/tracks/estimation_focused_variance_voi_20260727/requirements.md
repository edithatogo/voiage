# MoSCoW requirements — planned v1.2.0

## Must

- **M14-E1:** Declare scalar/vector target shape and component units.
- **M14-E2:** Declare the variance/covariance functional, prior, conditioning
  convention and sampling model.
- **M14-E3:** Return estimator uncertainty, convergence/degeneracy diagnostics,
  provenance and deterministic serialization.
- **M14-E4:** Remain distinct from decision EVPPI/EVSI, sensitivity indices and
  estimator standard error.
- **M14-E5:** Weight every EVSI posterior-variance evaluation by an aligned,
  finite, nonnegative prior-predictive probability vector that sums to one
  within the declared estimator tolerance; reject arbitrary unnormalised
  weights and bind replay provenance to the actual runtime inputs.

## Should

- Provide analytical and enumerable references, accessible reporting and
  explicit polyglot capability dispositions.

## Could

- Add separately reviewed vector covariance functionals.

## Won't

- Infer a covariance functional from data shape or alias decision-focused VOI.
