# MoSCoW requirements — planned v1.2.0

## Must

- **M14-E1:** Declare scalar/vector target shape and component units.
- **M14-E2:** Declare the variance/covariance functional, prior, conditioning
  convention and sampling model.
- **M14-E3:** Return estimator uncertainty, convergence/degeneracy diagnostics,
  provenance and deterministic serialization.
- **M14-E4:** Remain distinct from decision EVPPI/EVSI, sensitivity indices and
  estimator standard error.

## Should

- Provide analytical and enumerable references, accessible reporting and
  explicit polyglot capability dispositions.

## Could

- Add separately reviewed vector covariance functionals.

## Won't

- Infer a covariance functional from data shape or alias decision-focused VOI.
