# MoSCoW requirements — planned v1.2.0

## Must

- **M18-U1:** #556 deterministic sensitivity analysis must declare a fixed
  baseline, parameter/scenario coordinates, compared alternatives, outcome
  direction and units before evaluating one-way, two-way or scenario surfaces.
- **M18-U2:** Return every evaluated point, baseline and optimal alternatives,
  incremental outcomes, deterministic range/ranking metrics, complete tie sets
  under declared absolute/relative tolerances, and every observed tie or
  bracketing switch interval with exact-versus-bracket status. Tornado ranking
  must name its grid-extrema or endpoint metric; interpolation must be opt-in,
  estimated and assumption-labelled rather than fabricated.
- **M18-U3:** Keep DSA distinct from PSA, EVPPI and global sensitivity; reject
  non-finite/missing baselines, duplicate or unknown coordinates, malformed
  callback results and unsupported extrapolation. Two-way inputs with correlated
  coordinates must declare a feasible mask or path rather than implying that an
  infeasible Cartesian surface is covariance-aware.

- **M16-U1:** Delegate #595 delivery to
  `risk_adjusted_information_pricing_20260731`, which represents named utility,
  wealth/reference state, risk attitude, units, information/cost location,
  policies, scope and deterministic provenance.
- **M16-U2:** Require EUI, CEI, BPI, SPI, anchored PPI, signed values, policy
  switches, root diagnostics and explicit comparability conditions.
- **M16-U3:** Treat VoC as a presentation of the same clairvoyant-policy result
  and permit monetary EVPI reduction only under verified positive-affine
  utility.
- **M17-U1:** Keep canonical C16, issues/subissues and Project 28 synchronized
  through bounded managed projections.

## Should

- Provide normalized tabular input, accessible tornado plotting, explicit
  estimated interpolation labels and independent analytical/brute-force tests.
- Provide nonlinear-utility counterexamples, probability-price anchors,
  root-finding diagnostics and explicit polyglot dispositions.

## Could

- Add further reviewed presentation labels without new kernels.

## Won't

- Treat deterministic ranges as probability distributions, uncertainty
  attribution or information value.
- Create a duplicate VoC method or overwrite human-authored issue content.
