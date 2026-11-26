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

The [Paignia/Innovate handoff](innovation-paignia-vop-handoff.md) refines these
same leaves: VP01 maps to O01–O04/O06, VP02 to O05/O07/O08, and VP04's dependency
check to O08. VP03/VOP promotion and study-design feedback retain their existing
scientific-alignment/issue #1286 owners. They do not add prerequisites for every
outcome producer or new task-completion counters.

### O01 — reject unsupported outcome-profile versions

- [x] Task: validate the declared version of the generic outcome profile and
  return an explicit unsupported-version error.
  Implemented `voiage.simulation-outcome.v1` contract in `rust/crates/voiage-domain/src/simulation_outcome.rs`, `specs/ecosystem/simulation-outcome/v1/profile.md`, and `profile-fixtures.json` (synthetic health + non-health cases); verified by `simulation_outcome_profile` (`2 passed`) and specialist contract review.
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

- [x] Task: admit exactly one value per declared strategy/draw key and preserve
  the producer-declared joint-draw correspondence.
  Implemented canonical pairing, strict untrimmed key/namespace validation, `DrawIdentity` receipts, and `Equal`/`Probability`/`RelativeWeight` validation; verified by `simulation_draw_pairing` (`3 passed`) and specialist contract review.
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

- [x] Task: reject input values that disagree with the declared units,
  perspective, or value/objective convention before numerical evaluation.
  Implemented `ValueDeclaration`, `ValueConvention`, and `ObjectiveDirection` checks without implicit conversions; verified by `simulation_value_conventions` (`2 passed`) and specialist contract review.
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

- [x] Task: join optional parameter samples to outcome draws by their declared
  identity instead of assuming row-position equivalence.
  Implemented `AlignedParameterSamples` draw-key alignment and schema consistency checks; verified by `simulation_parameter_alignment` (`3 passed`) after specialist review.
- Depends on: O01, O02.
- Execution: local against the frozen pairing contract.
- Files (3): `rust/crates/voiage-domain/src/simulation_outcome.rs` (future),
  `rust/crates/voiage-domain/tests/simulation_parameter_alignment.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/parameter-fixtures.json` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p voiage-domain
  --test simulation_parameter_alignment` (future target). Reordered named draws
  align exactly; equal-length but unmatched or duplicated identities fail.

### O05 — convert validated outcomes through an optional native adapter

- [x] Task: map accepted valued outcomes to existing Rust VOI input contracts
  without changing their values, pairing, weights, or declared conventions.
  Implemented `SimulationOutcomeHandoff`, `adapt_simulation_outcome`, and `adapt_simulation_outcome_input` in `rust/crates/voiage-numerics/src/simulation_outcome_adapter.rs` (reusing existing workspace crate per frozen 7-crate inventory), preserving values, `DrawIdentity`, weights, `ValueDeclaration`, `ProducerProvenance`, `ValueArray`, and optional `ParameterSet`; verified by `simulation_outcome_core_handoff` (`3 passed; 0 failed`, including `RelativeWeight` and `Minimize`/`Loss` preservation without sign flip).
- Depends on: O01–O04; existing owner retains provider qualification.
- Execution: local against the reviewed public contract.
- Files (4): `conductor/tracks/simulation_outcome_boundary_20261107/plan.md`,
  `rust/crates/voiage-numerics/src/lib.rs`,
  `rust/crates/voiage-numerics/src/simulation_outcome_adapter.rs` (future),
  `rust/crates/voiage-numerics/tests/simulation_outcome_core_handoff.rs` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p
  voiage-numerics --test simulation_outcome_core_handoff` (future target).
  Owner inventory (`tests/test_rust_workspace_contract.py` and
  `tests/test_structure_api_abi_audit.py`) freezes `rust/crates/` at 7
  workspace crates, and `voiage-numerics` already depends on `voiage-domain` +
  `voiage-diagnostics` with zero producer engines; reuse `voiage-numerics`
  instead of creating a new workspace crate.

### O06 — preserve outcome semantics through Arrow/Parquet interchange

- [x] Task: round-trip the accepted outcome profile without losing identity,
  value declarations, weights, or provenance metadata.
  Implemented `SimulationOutcomeProvenance`, `SimulationOutcomeValueDeclaration`, and `SimulationOutcomeProfileMetadata` in `voiage/contracts/normalized_input.py`, and Arrow/Parquet helpers in `voiage/contracts/interchange.py` (`specs/ecosystem/simulation-outcome/v1/interchange-fixture.json`, SHA-256 `efa4c3ec41ab317a7b6daffcf3eaf75fcccc7d54bbe61261f9820378adda7933`, `profile.md`); verified on reinstalled non-editable `voiage==3.1.0` wheel via `run_installed_pytest.py` (`tests/test_simulation_outcome_interchange.py`, `test_normalized_input_bundle.py`, `test_contract_interchange.py`: `70 passed`).
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

- [x] Task: obtain the same EVPI from a hand-calculated paired payoff fixture
  through the adapter and the existing Rust kernel within a declared tolerance.
  Implemented `simulation_outcome_evpi` delegating to `evpi` (`Equal`) and `structural_evpi` (`Probability` / `RelativeWeight`) with tolerance `1e-12` against `specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json` (SHA-256 `0ac9880b48a32eef3d330870dbb103eca263a82ef110ef43361cbf58e0f35911`); verified `simulation_outcome_evpi_equivalence` (`3 passed; 0 failed`) and completed specialist numerical review (`docs/reviews/simulation-outcome-numerical-qualification.md`, Neumaier compensated summation + boundary documentation remediated).
- Depends on: O02, O03, O05; an independently specified expected result.
- Execution: specialist, numerical/estimand boundary.
- Files (3):
  `rust/crates/voiage-numerics/tests/simulation_outcome_evpi_equivalence.rs` (future),
  `specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json` (future),
  `docs/reviews/simulation-outcome-numerical-qualification.md` (future).
- Check: `cargo test --locked --manifest-path rust/Cargo.toml -p
  voiage-numerics --test simulation_outcome_evpi_equivalence` (future target).
  Include a joint-permutation invariant and an independently permuted-strategy
  counterexample whose true EVPI differs. Cover the supported weight convention;
  do not claim EVSI, EVPPI, causal validity, or producer calibration from EVPI.

### O08 — use installed outcome artifacts without producer engines

- [x] Task: run the optional adapter in an isolated consumer that has no Kairos,
  game, or HEOR engine dependency, while retaining normal default Voiage use.
  Added isolated crate `tests/fixtures/simulation_outcome_consumer/` (fixture hashes `01359dd644d099b3d90e0acf32ef88bf6bdc29f21c0d5b313e0b43287e8e29ec`, `22a24a23c983a5f9d6ae99a9cd161be4471c374d43fd1b41ff2c7c563c92c38a`), helper `scripts/check_simulation_outcome_consumer.py` (`4 passed; 0 failed` against unpacked `voiage-domain`/`voiage-diagnostics`/`voiage-numerics` v3.1.0 crates, `35` resolved packages, zero producer engines), `tests/test_simulation_outcome_consumer_contract.py`, and CI job `simulation-outcome-consumer` in `.github/workflows/ci.yml`. Synthetic fixtures remain labelled; installed use is not full real-producer qualification.
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
