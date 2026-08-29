# Vector covariance pre-review disposition

## Status and boundary

This candidate retains a **major-revision, fail-closed exclusion** for vector
`EVPPI_var` and `EVSI_var`. The three automated panel roles supplied challenge
evidence; they are not the candidate-bound named independent human scientific
approval required by E22, E23, SR3 and SR10. Vector execution, scientific
acceptance and stable promotion therefore remain prohibited.

The runtime must reject a vector specification before native dispatch. The
semantic result validator must reject every vector result envelope, even when
its dimensions, symmetry and reduction identities appear valid. JSON Schema
may reserve `trace`, `determinant` and `weighted_quadratic` vocabulary for a
future major/minor contract, but no portable or executable vector result is
conformant in this candidate.

## Panel synthesis

The estimand/domain, numerical-assurance and API/governance roles independently
identified the same High findings:

1. The result contract checked vector dimension and symmetry but not PSD,
   nonnegative diagonals, units or functional recomputation, so indefinite
   covariance matrices and arbitrary functionals could validate.
2. Trace is not unit-safe across heterogeneous, unstandardized components.
   Weighted quadratic is defensible only as the variance of a declared linear
   contrast with structured weight and result units. Bare weights are
   insufficient.
3. Determinant is nonlinear, so `det(E[Cov(g | Y)])` is generally not
   `E[det(Cov(g | Y))]`. The current envelope cannot resolve or reproduce that
   estimand and determinant remains excluded.
4. PSD must cover prior covariance, expected posterior covariance and, under
   exact population conditioning, their difference. Finite-sample violations
   require uncertainty diagnostics rather than silent matrix repair.
5. Fixed absolute tolerances and silent eigenvalue clipping are not
   rescaling-safe or replay-transparent. A future initial policy should use
   declared component scales, scale-aware tolerances and
   `regularization = reject`.

## Future review decision packet

A future E22 candidate must select and independently reproduce a narrow
scientific contract before implementation. The minimum packet includes:

- homogeneous or explicitly standardized trace, or a dimensioned declared
  linear contrast; determinant remains separately scoped;
- finite symmetric matrices, nonnegative diagonals and scale-aware PSD/eigen
  checks for prior, expected posterior and their reduction;
- an explicit regularization policy, raw/repaired matrices when repair is ever
  permitted, algorithm/version, thresholds and correction diagnostics;
- native functional recomputation rather than caller-supplied functionals;
- analytical bivariate normal-normal and finite-discrete covariance oracles;
- singular, nearly-PSD, indefinite, individually-PSD-but-indefinite-difference,
  permutation, unit-rescaling and extreme-scale cases; and
- fresh affected-role review plus the named independent human scientific
  verdict bound to the exact candidate and packet hashes.

Rejected matrix cases in this candidate prove only the exclusion boundary;
they are not evidence of implemented multivariate numerical correctness.
