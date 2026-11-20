# Implementation Plan

Each leaf is bounded to at most five files and verified by deterministic tests before consumer pin promotion.

- [x] **CE01** — Define polyhedral/interval credal-set contracts (`CredalPolytope`, `DominanceWitness`, `EAdmissibilityResult`) in `voiage-domain` with exact degeneracy reduction to point-identified expected net benefit when lower and upper bounds coincide.
  - The implementation validates named finite intervals and ordered linear constraints. When all intervals pin a feasible point, it evaluates the affine net benefits directly. [Accepted in #1355](https://github.com/edithatogo/voiage/pull/1355)
  - The native tests distinguish a closed halfspace boundary, an excluded pinned point, nonfinite arithmetic and malformed dominance multipliers. Witness verification for general support follows in CE02.
- [ ] **CE02** — Implement LP / Farkas duality witness verification for E-admissibility and strict credal dominance in `voiage-numerics`, verified against analytical 2D/3D polytope reference oracles and deterministic Hypothesis/proptest suites.
  - [PR #1356](https://github.com/edithatogo/voiage/pull/1356) verifies exact rational points and normalized dominance certificates with an explicit feasible point. Analytic cases cover fractional ties, two- and three-dimensional support, certificate rescaling and false positive margins.
  - [PR #1357](https://github.com/edithatogo/voiage/pull/1357) qualifies the native solver with attained affine ranges, Gamma-maximin, Gamma-maximax and interval dominance. It verifies generated E-admissibility and strict mixed-dominance witnesses before returning them. An explicit shared basis budget can leave the support or individual strategies indeterminate. Exact fractional weights and margins remain intact.
  - The native analytical and Proptest controls passed. Hypothesis controls for the installed Python surface depend on CE04.
- [ ] **CE03** — Implement Tobit left-censored (`<LOQ`) and action-dependent observation likelihoods for sequential EVSI in `voiage-numerics`, verifying Blackwell garbling monotonicity ($\text{EVSI}(Y_{\text{garbled}}) \le \text{EVSI}(Y)$) and zero-information degeneracy.
  - The finite sequential kernel evaluates action-specific observation distributions and selects later acquisition actions from the observed history. Exact probability arithmetic preserves zero-information cases and Blackwell order for declared garblings. A bounded search returns an indeterminate result if the budget expires.
  - The Tobit likelihood and the construction of finite channels from censored observations remain open.
- [ ] **CE04** — Expose Polars/Arrow interchange and Python bindings for credal dominance and endogenous censored EVSI, preserving `draw_id`, `strategy_id`, `perspective`, and `subgroup_id`.
- [ ] **CE05** — Qualify installed-package parity with `edithatogo/tobac` (`src/tobac/information.py`, `src/tobac/partial_identification.py`, `tests/engineering/test_properties.py`) under [issue #1193](https://github.com/edithatogo/voiage/issues/1193) and [issue #1290](https://github.com/edithatogo/voiage/issues/1290).
