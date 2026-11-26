# Numerical Qualification — Optional Simulation-Outcome Boundary (`O05`, `O07`)

## Scope and Ownership

The optional adapter in `rust/crates/voiage-numerics/src/simulation_outcome_adapter.rs`
maps already-validated `voiage.simulation-outcome.v1` profiles onto core Rust
VOI contracts (`ValueArray` and optional `ParameterSet`) and evaluates EVPI by
dispatching to the existing Rust kernels (`evpi` and `structural_evpi`).

- **No duplicated EVPI math:** `simulation_outcome_evpi` never implements a
  parallel EVPI formula; it delegates equal-weight draws to `voiage_numerics::evpi`
  and weighted draws (`probability` or `relative_weight`) to
  `voiage_numerics::structural_evpi` using one 1-row state plane per draw.
- **No implicit valuation or sign flips:** `ObjectiveDirection::Minimize` and
  `ValueConvention::Loss` return `SimulationOutcomeAdapterError::UnsupportedObjectiveDirection`
  explicitly instead of negating payoff matrices.
- **No silent probability rescaling:** `WeightSemantics::Probability` forwards
  raw declared weights directly to `structural_evpi`. Weights that pass the
  profile's `1e-9` validation window but violate `structural_evpi`'s `1e-12`
  sum-to-one check fail explicitly with `SimulationOutcomeAdapterError::Numerical`.

## Analytical Hand-Calculated Reference Fixtures

All reference cases live in `specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json`
with declared absolute/kernel equivalence tolerance `1e-12`. All fixtures are
labelled synthetic examples demonstrating the cross-domain boundary.

1. **`synthetic_health_equal_weights` (`NZD`, `net_benefit` / `maximize`, `equal`):**
   - Payoff rows `(d0..d3)`: `[10, 4]`, `[2, 8]`, `[6, 6]`, `[0, 10]`.
   - `E[max NB] = (10 + 8 + 6 + 10) / 4 = 8.5`.
   - `max_s E[NB_s] = max((10 + 2 + 6 + 0)/4, (4 + 8 + 6 + 10)/4) = max(4.5, 7.0) = 7.0`.
   - Exact EVPI = `8.5 - 7.0 = 1.5`.
2. **`synthetic_non_health_probability_weights` (`utils`, `utility` / `maximize`, `probability` `[0.2, 0.5, 0.3]`):**
   - Payoff rows `(s0..s2)`: `[15, 5, 10]`, `[4, 12, 8]`, `[6, 2, 16]`.
   - `E[max U] = 0.2*15 + 0.5*12 + 0.3*16 = 3.0 + 6.0 + 4.8 = 13.8`.
   - Strategy means: `6.8`, `7.6`, `10.8` -> maximum baseline expected utility = `10.8`.
   - Exact EVPI = `13.8 - 10.8 = 3.0`.
3. **`synthetic_non_health_relative_weights` (`USD`, `net_benefit` / `maximize`, `relative_weight` `[0.0, 1.0, 3.0, 6.0]`):**
   - Normalized probabilities: `[0.0, 0.1, 0.3, 0.6]`; zero-weight draw `[1000, -500]` contributes `0.0`.
   - Remaining rows: `[20, 10]` (`p=0.1`), `[10, 30]` (`p=0.3`), `[15, 5]` (`p=0.6`).
   - `E[max NB] = 0.1*20 + 0.3*30 + 0.6*15 = 2.0 + 9.0 + 9.0 = 20.0`.
   - Strategy means: `14.0`, `13.0` -> maximum baseline = `14.0`.
   - Exact EVPI = `20.0 - 14.0 = 6.0`.

## Joint-Draw Coupling Invariant vs. Independent-Permutation Counterexample

- **Joint-permutation invariant (`joint_permutation_invariant`):** Reordering the
  complete keyed records `(draw_key, strategy_id)` of `synthetic_health_equal_weights`
  leaves canonical pairing intact and yields identical EVPI (`1.5`, error `<= 1e-12`).
- **Independently permuted strategy (`independent_strategy_permutation_counterexample`):**
  Swapping only `intervention`'s `d0` and `d3` values transforms `[4, 8, 6, 10]`
  into `[10, 8, 6, 4]`. Each strategy's marginal distribution and expected payoff
  remain `4.5` and `7.0`, yet `E[max] = (10 + 8 + 6 + 4)/4 = 7.0`, collapsing EVPI
  to `0.0 != 1.5`. Row-position alignment without draw-key correspondence is
  therefore insufficient and unsafe.

## Explicit Non-Claims and Scope Boundaries

- **Kernel sum-to-one & domain finite-total boundaries:**
  `normalized_relative_weights` uses Neumaier compensated summation for the
  normalization denominator, while downstream `structural_evpi` checks
  `|sum(p) - 1.0| <= 1e-12` using sequential binary64 summation (`structural.rs`),
  and domain validation (`simulation_outcome.rs`) requires the unscaled sum of
  `relative_weight` entries to be finite in binary64. Any profile exceeding
  those bounds fails loudly with `SimulationOutcomeError` or
  `SimulationOutcomeAdapterError::Numerical` rather than silently rescaling.
- Passing EVPI equivalence on `voiage.simulation-outcome.v1` demonstrates only
  that paired finite payoffs and draw weights map faithfully to `ValueArray`
  and `evpi` / `structural_evpi`.
- It does **not** establish EVSI, EVPPI, causal validity, producer RNG
  common-random-number validity, clinical/economic valuation validity, or full
  real-producer calibration.

## Verification Provenance & Specialist Review Remediation

- **Base Source Revision:** `8d393c4417361e3c554bec7d18903c65204ec188` (`O01`–`O04`) + `O05`–`O08` changes on `codex/simulation-outcome-o05-o08-20261126`.
- **Fixture & Artifact Digests (SHA-256):**
  - `specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json`: `0ac9880b48a32eef3d330870dbb103eca263a82ef110ef43361cbf58e0f35911`
  - `voiage-domain-3.1.0.crate`: `0b791cc6427896ea1ee3b1b48dc0289912887fc0fb5bcf6607fbdfc14394d1e7`
  - `voiage-diagnostics-3.1.0.crate`: `f467fece137e7c7fd4f7b3e14fb926c6c05ad34ff3244e42a6b7d8f350c184d9`
  - `voiage-numerics-3.1.0.crate`: `f9409e18ab71881ce39d7304f9eab316d65e679bb35599370ea9218e038d53b1`
- **Executed Commands & Nonzero Test Counts:**
  - `cargo test --locked --manifest-path rust/Cargo.toml -p voiage-numerics --test simulation_outcome_core_handoff --test simulation_outcome_evpi_equivalence`: `6 passed; 0 failed` (`3` core handoff + `3` EVPI equivalence tests, tolerance `1e-12`).
  - `cargo clippy --locked --manifest-path rust/Cargo.toml -p voiage-domain -p voiage-numerics --all-targets -- -D warnings`: `0` warnings.
  - `.venv/bin/python scripts/check_simulation_outcome_consumer.py`: `4 passed; 0 failed` across isolated unpacked crates (`35` resolved packages, zero producer engines).
- **Independent Specialist Numerical Review (`67352e19-2423-4d0a-8177-01092b236730`):**
  - Confirmed all analytical EVPI calculations, `evpi`/`structural_evpi` delegation, `loss`/`minimize` rejection without sign flip, and `1e-9` vs `1e-12` probability tolerance preservation.
  - Remediated 1 Medium & 2 Low findings: added Neumaier compensated summation to `normalized_relative_weights`, added `RelativeWeight` + `Loss`/`Minimize` core-handoff tests in `simulation_outcome_core_handoff.rs`, documented inherited `structural_evpi` / domain finite-total boundaries above, and bound closeout digests/commands here.
