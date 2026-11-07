# Plan — optional simulation-outcome boundary

All eight leaves are pending. Each changes one behavior and touches at most
five files; split a larger change into a linked bounded leaf before execution.
Paths are relative to this repository. Files/test targets labelled **future**
do not exist at the baseline: create the distinguishing test, observe failure,
then implement. Zero collected tests or a missing target is not a pass.

`local` requires diff review and focused checks. `specialist` adds independent
agent review of the coherent public contract or numerical boundary. Reuse that
review for dependent implementation; do not add repeated human approvals.
Existing scientific alignment owns provider revisions and any pin migration.

### O01 — reject unsupported outcome-profile versions

- [ ] Task: validate the declared version of the generic outcome profile and
  return an explicit unsupported-version error.
- Depends on: inventory of current `ValueArray`, `ParameterSet`, and normalized
  input contracts; coordinate the supported subset with scientific alignment.
- Execution: specialist, public input contract.
- Files (5): `rust/crates/voiage-domain/src/lib.rs`,
  `rust/crates/voiage-domain/src/simulation_outcome.rs` (future),
  `rust/crates/voiage-domain/tests/simulation_outcome_profile.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/profile.md` (future),
  `specs/ecosystem/simulation-outcome/v1/profile-fixtures.json` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p voiage-domain
  --test simulation_outcome_profile` (future target). Accepted and unknown
  versions must yield different outcomes; freeze the profile and review any
  necessary extension instead of silently changing an existing schema.

### O02 — enforce paired strategy/draw identity

- [ ] Task: admit exactly one value per declared strategy/draw key and preserve
  the producer-declared joint-draw correspondence.
- Depends on: O01.
- Execution: specialist, joint-sample interpretation.
- Files (3): `rust/crates/voiage-domain/src/simulation_outcome.rs` (future),
  `rust/crates/voiage-domain/tests/simulation_draw_pairing.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/pairing-fixtures.json` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p voiage-domain
  --test simulation_draw_pairing` (future target). Reordering complete keyed
  records preserves the mapping; unlabelled row-position pairing, missing or
  duplicate keys, conflicting weights, or mismatched namespaces fail explicitly.
  The adapter cannot infer whether a producer's declared coupling is valid.

### O03 — reject incompatible value declarations

- [ ] Task: reject input values that disagree with the declared units,
  perspective, or value/objective convention before numerical evaluation.
- Depends on: O01; apply the agreed contract without adding valuation methods.
- Execution: specialist, decision-value semantics.
- Files (3): `rust/crates/voiage-domain/src/simulation_outcome.rs` (future),
  `rust/crates/voiage-domain/tests/simulation_value_conventions.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/value-convention-fixtures.json` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p voiage-domain
  --test simulation_value_conventions` (future target). Matching declarations
  pass unchanged; mixed utility/loss, unit/perspective, or applicable
  currency/population/horizon declarations fail without implicit conversion.

### O04 — align supplied parameter samples by draw key

- [ ] Task: join optional parameter samples to outcome draws by their declared
  identity instead of assuming row-position equivalence.
- Depends on: O01, O02.
- Execution: local against the frozen pairing contract.
- Files (3): `rust/crates/voiage-domain/src/simulation_outcome.rs` (future),
  `rust/crates/voiage-domain/tests/simulation_parameter_alignment.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/parameter-fixtures.json` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p voiage-domain
  --test simulation_parameter_alignment` (future target). Reordered named draws
  align exactly; equal-length but unmatched or duplicated identities fail.

### O05 — convert validated outcomes through an optional native adapter

- [ ] Task: map accepted valued outcomes to existing Rust VOI input contracts
  without changing their values, pairing, weights, or declared conventions.
- Depends on: O01–O04; existing owner retains provider qualification.
- Execution: local against the reviewed public contract.
- Files (5): `rust/Cargo.toml`, `rust/Cargo.lock`,
  `rust/crates/voiage-simulation-adapter/Cargo.toml` (future),
  `rust/crates/voiage-simulation-adapter/src/lib.rs` (future),
  `rust/crates/voiage-simulation-adapter/tests/core_handoff.rs` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p
  voiage-simulation-adapter --test core_handoff` (future package/target).
  This provisional optional package depends on Voiage contracts/kernels, not
  producer engines; reuse an existing appropriate package if the owner inventory
  identifies one and update the concrete test target before implementation.

### O06 — preserve outcome semantics through Arrow/Parquet interchange

- [ ] Task: round-trip the accepted outcome profile without losing identity,
  value declarations, weights, or provenance metadata.
- Depends on: O01–O04; existing normalized-input/interchange conventions.
- Execution: local; return to the contract review if format semantics must change.
- Files (5): `voiage/contracts/normalized_input.py`,
  `voiage/contracts/interchange.py`,
  `tests/test_simulation_outcome_interchange.py` (future),
  `specs/ecosystem/simulation-outcome/v1/interchange-fixture.json` (future),
  `specs/ecosystem/simulation-outcome/v1/profile.md` (future).
- Check: `pytest -q tests/test_simulation_outcome_interchange.py` (future suite).
  Compare keyed values and every required metadata field, including shuffled
  and chunked inputs; drop a required field as a negative control. The existing
  `pytest -q tests/test_normalized_input_bundle.py` is the adjacent regression
  check, not proof that this new profile is supported.

### O07 — preserve an analytical VOI result across the adapter

- [ ] Task: obtain the same EVPI from a hand-calculated paired payoff fixture
  through the adapter and the existing Rust kernel within a declared tolerance.
- Depends on: O02, O03, O05; an independently specified expected result.
- Execution: specialist, numerical/estimand boundary.
- Files (3):
  `rust/crates/voiage-simulation-adapter/tests/native_evpi_equivalence.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json` (future),
  `docs/reviews/simulation-outcome-numerical-qualification.md` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p
  voiage-simulation-adapter --test native_evpi_equivalence` (future target).
  Include a joint-permutation invariant and an independently permuted-strategy
  counterexample whose true EVPI differs. Cover the supported weight convention;
  do not claim EVSI, EVPPI, causal validity, or producer calibration from EVPI.

### O08 — use installed outcome artifacts without producer engines

- [ ] Task: run the optional adapter in an isolated consumer that has no Kairos,
  game, or HEOR engine dependency, while retaining normal default Voiage use.
- Depends on: O05–O07 and accepted provider/consumer contract evidence through
  scientific alignment; no automatic provider pin update.
- Execution: local; specialist review only for a newly changed public boundary.
- Files (5): `scripts/check_simulation_outcome_consumer.py` (future),
  `tests/test_simulation_outcome_consumer_contract.py` (future),
  `tests/fixtures/simulation_outcome_consumer/Cargo.toml` (future),
  `tests/fixtures/simulation_outcome_consumer/src/lib.rs` (future),
  `.github/workflows/ci.yml`.
- Checks: `pytest -q tests/test_simulation_outcome_consumer_contract.py` and
  `python scripts/check_simulation_outcome_consumer.py` (future targets).
  The helper must unpack versioned local package artifacts outside the source
  checkout, inspect the resolved dependency graph, and run `cargo test --locked
  --manifest-path <isolated-consumer>/Cargo.toml`. Record nonzero test counts,
  exact hashes/versions and health/non-health valued fixtures. Synthetic cases
  stay labelled; installed use is not full real-producer qualification.

## Closure and rollback

Record the tested source/tree, artifact and fixture hashes, commands, nonzero
test counts, declared tolerances, and review findings for each completed leaf.
Run relevant required repository/hosted gates for the final implementation
candidate; repeat broad checks only when inputs change or a concrete risk
requires them. Keep failures and unsupported-method decisions visible.

The track remains open until the supported profile, negative controls, and
installed-consumer acceptance pass. Existing scientific alignment and HPC
owners keep their own unfulfilled gates. Roll back by reverting a bounded
adapter change or selecting the previous qualified version, never by dropping
identity/unit checks or silently substituting a simulation/valuation engine.
