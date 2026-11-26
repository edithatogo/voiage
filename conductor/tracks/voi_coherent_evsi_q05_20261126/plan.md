# Implementation Plan: Q05 Coherent EVSI Qualification

- [ ] **Q05.1** — Freeze estimator-specific study contracts, per-arm vs. total sample-size schemas, and diagnostic fields (`ess`, `khat`, `decision_error`) in `voiage-domain` and `voiage/contracts/`.
- [ ] **Q05.2** — Harden `evsi_regression`, `evsi_moment_based`, and `evsi_stochastic` in `rust/crates/voiage-numerics/src/evsi*.rs` against `Q03` conjugate Normal-Normal, Beta-Binomial, and Gamma-Poisson witnesses (`V03`, `V05`).
- [ ] **Q05.3** — Implement and verify `evsi_importance_sampling` (`V04`) with explicit Pareto-k / ESS degeneracy diagnostics and fail-closed rejection on unsupported weights.
- [ ] **Q05.4** — Expand validated non-normal study models (`V05`: two-arm binary, Poisson count, time-to-event) and `F02` posterior-sample/diagnostic boundary without making external inference frameworks core dependencies.
- [ ] **Q05.5** — Add deterministic QMC / multilevel / unbiased EVSI estimators (`V07`) with accuracy-vs-cost benchmarks against Monte Carlo.
- [ ] **Q05.6** — Wire Python/PyO3 bindings (`voiage/methods/sample_information.py`), run `cargo test --locked` + installed-wheel `pytest`, and archive track.
