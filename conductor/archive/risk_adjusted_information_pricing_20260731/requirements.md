# MoSCoW requirements — planned v1.2.0

## Must

- **M16-U1:** Represent a serializable utility function, parameters,
  normalization, wealth/reference state, risk attitude, payoff units,
  probabilities, information structure, information cost location, current and
  informed policies, and stakeholder/organizational scope.
- **M16-U2:** Return EUI, CEI, BPI, SPI and anchored PPI with declared direction, policy
  switches, signed values, root brackets/residuals/iterations/convergence, and
  explicit comparability or non-comparability.
- **M16-U3:** Treat VoC as the expected-utility value of the same clairvoyant
  policy result, with monetary EVPI equality only under a verified
  positive-affine utility reduction.
- **M16-U4:** Require Rust-authoritative accepted numerical policy,
  deterministic fixtures, independent references, nonlinear counterexamples,
  pathological cases, and explicit Rust/Python/R/Julia/Mojo dispositions.
- **M17-U1:** Keep canonical C16, #595 and its native subissues, Project 28,
  this track, and registered repository projections synchronized without
  overwriting human-authored content.
- **M16-U5:** Evaluate CRRA/power utility and its inversion stably near risk
  aversion one using a declared log-limit/expm1 switching tolerance and error
  budget. Require sweeps around one, payoff-scale sweeps, high-precision
  oracles, continuity and root-residual evidence.
- **M16-U6:** Bind presentation selection and presentation-contract version to
  a deterministic presentation digest derived from the native result. EUI,
  BPI, SPI, PPI and affine-EVPI views must retain a shared numerical kernel but
  carry auditable, distinct presentation provenance.

## Should

- Support affine, exponential, logarithmic and power utilities with bounded,
  deterministic root solving and positive-affine invariance checks.
- Expose CLI/reporting and accessible examples that explain why utility-scale
  EUI, certainty-equivalent prices and monetary EVPI are not interchangeable.

## Could

- Add further reviewed constructed-scale prices as additive named price
  definitions without changing the canonical EUI/CEI/BPI/SPI/PPI result.

## Won't

- Create a duplicate VoC kernel, accept arbitrary serialized utility
  callables, silently clamp signed prices, aggregate incomparable stakeholder
  utilities, or relabel CVaR/preference/acquisition helpers as #595.
