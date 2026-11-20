# Implementation Plan

Each leaf is bounded to at most five files and verified by deterministic tests before consumer pin promotion.

- [x] **CE01** — Define polyhedral/interval credal-set contracts (`CredalPolytope`, `DominanceWitness`, `EAdmissibilityResult`) in `voiage-domain` with exact degeneracy reduction to point-identified expected net benefit when lower and upper bounds coincide.
  - The implementation validates named finite intervals and ordered linear constraints. When all intervals pin a feasible point, it evaluates the affine net benefits directly. [Accepted in #1355](https://github.com/edithatogo/voiage/pull/1355)
  - The native tests distinguish a closed halfspace boundary, an excluded pinned point, nonfinite arithmetic and malformed dominance multipliers. Witness verification for general support follows in CE02.
- [ ] **CE02** — Implement LP / Farkas duality witness verification for E-admissibility and strict credal dominance in `voiage-numerics`, verified against analytical 2D/3D polytope reference oracles and deterministic Hypothesis/proptest suites.
  - The candidate verifies exact rational points and normalized dominance certificates with an explicit feasible point. Analytic cases cover fractional ties, two- and three-dimensional support, certificate rescaling and false positive margins. Solver-generated witnesses and decision-rule classification remain pending.
- [ ] **CE03** — Implement Tobit left-censored (`<LOQ`) and action-dependent observation likelihoods for sequential EVSI in `voiage-numerics`, verifying Blackwell garbling monotonicity ($\text{EVSI}(Y_{\text{garbled}}) \le \text{EVSI}(Y)$) and zero-information degeneracy.
- [ ] **CE04** — Expose Polars/Arrow interchange and Python bindings for credal dominance and endogenous censored EVSI, preserving `draw_id`, `strategy_id`, `perspective`, and `subgroup_id`.
- [ ] **CE05** — Qualify installed-package parity with `edithatogo/tobac` (`src/tobac/information.py`, `src/tobac/partial_identification.py`, `tests/engineering/test_properties.py`) under [issue #1193](https://github.com/edithatogo/voiage/issues/1193) and [issue #1290](https://github.com/edithatogo/voiage/issues/1290).
