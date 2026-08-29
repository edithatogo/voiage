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
- **M14-E6:** Bind the scientific specification, target data/outcome
  identifiers, sampling-model artifact, design/likelihood identifiers and
  solver request to the executable inputs and replay provenance.
- **M14-E7:** Declare exact-enumeration, independent-outer, nested or coupled
  estimator design; resampling unit; variance convention; confidence method;
  inner/outer counts; dependence handling; and tri-state convergence. Report
  truth-known bias, RMSE, coverage and calibration studies before promotion.
- **M14-E8:** Before vector promotion, require positive-semidefinite and eigen
  tolerances, nonnegative diagonals, declared covariance regularization,
  component units, unit-safe scalarization, functional recomputation and an
  independent multivariate oracle.
- **M14-E8a:** Until M14-E8 receives candidate-bound independent scientific
  review and a named human verdict, treat vector functionals as reserved
  contract vocabulary only: reject every vector runtime request before native
  dispatch and every vector result envelope during semantic validation. Do not
  describe rejected pathological matrices as multivariate oracle coverage.
- **M14-E9:** Provide independently executable portable EVPPI and EVSI fixtures
  consumed by Rust and Python, including finite discrete and normal-normal
  references, exact-enumeration, nested/coupled, rare-outcome, singular,
  nearly-PSD, indefinite, permutation and rescaling cases.

## Should

- Provide analytical and enumerable references, accessible reporting and
  explicit polyglot capability dispositions.

## Could

- Add separately reviewed vector covariance functionals.

## Won't

- Infer a covariance functional from data shape or alias decision-focused VOI.
