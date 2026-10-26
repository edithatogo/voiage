# Specification: Polars-native core containers

## Scope and authority

Baseline: `ca3f05cc8adcc74b99fe236152327d0cd3679553`. Follow root `AGENTS.md`. This track owns the in-memory container change in `voiage/schema.py` and its call sites; it authorises no I/O, packaging or documentation change beyond what the container requires.

Prerequisites: `tabular_contract_v3_20261026` must land first, because the current normative decision record names xarray as the Python reference model. This track implements the decision; it does not renegotiate it.

## Evidence basis

Measured on the baseline, not assumed:

- xarray appears in 13 runtime modules; `voiage/schema.py` holds 36 of 75 references.
- The public contract is a two-dimensional table, not a labelled N-D array: `ValueArray` is samples by strategies with one `net_benefit` variable and a `strategy` coordinate; `ParameterSet` is samples by parameters with one variable per parameter.
- The xarray-specific runtime surface is small and enumerable: `xr.Dataset` construction, `.dims`, `.coords`, `.values`, one `isel`, two `assign_coords`, and type checks.
- There is no `.sel`, `.interp`, `xr.concat`, `xr.apply_ufunc`, `.to_dataframe`, NetCDF, Zarr or pickle use anywhere in `voiage/`.
- Core mathematics is already NumPy/JAX/torch and converts to arrays immediately.
- 25 `.dataset` references exist across 8 test modules; 36 test files mention xarray and 28 import pandas or xarray.
- `voiage/cli.py`, `voiage/core/io.py`, 3 examples and `scripts/validate_core_api_fixtures.py` construct or import xarray outside the container module.

## Required context

- `voiage/schema.py`
- `voiage/metamodels.py`
- `voiage/methods/adaptive.py`, `observational.py`, `network_nma.py`, `calibration.py`, `basic.py`, `sample_information.py`
- `voiage/core/io.py`
- `voiage/analysis.py`, `voiage/main_backends.py`, `voiage/parallel/monte_carlo.py`
- `voiage/cli.py` (constructs an `xr.Dataset` at the baseline) and `voiage/widgets/voi_widgets.py` (type checks only)
- `tests/test_schema.py`, `tests/test_schema_comprehensive.py`, `tests/test_network_nma.py`, `tests/conftest.py` and the other test modules that import xarray (28 modules import pandas or xarray at the baseline)
- `examples/` (10 files mention xarray) and `scripts/validate_core_api_fixtures.py`
- `specs/core-api/examples/v1/value-array.example.json`, `parameter-set.example.json`

## Acceptance criteria

### AC1: Inventory

Classify every xarray reference in the runtime as container, validation, conversion or dead code before changing any of them.

Acceptance: the inventory accounts for all references at the baseline with no unclassified entry; each entry names its owning task; anything already unused is deleted rather than ported.

### AC2: Containers

Replace the xarray container with a Polars frame that preserves the existing matrix semantics.

Acceptance: `ValueArray` holds a wide frame keyed by a sample identifier with one `Float64` column per strategy in declared order; `ParameterSet` holds a wide frame with one `Float64` column per parameter; both validate shape, dtype and non-null policy at construction. `values`/`numpy_values`/`parameters`/`from_numpy` keep their current signatures and return types.

### AC3: Validation parity and fail-closed extras

Keep the structural guarantees the xarray container provided, without inheriting its permissiveness.

Acceptance: missing dimension, wrong dtype, empty frame, duplicate strategy names and length mismatch raise the existing Voiage exception types with equivalent messages; an unrecognised extra variable is rejected explicitly rather than silently retained; the decision to reject extras is recorded in the track. Negative tests exist for each case.

### AC4: Numerical parity

Prove the change is a container change and not a numerical one.

Acceptance: every retained reference case produces a bit-identical NumPy array (`np.array_equal`) before and after; column order is identical; no tolerance is widened; the existing `specs/core-api` example fixtures produce identical output; the parity command, environment and artifact digest are recorded.

### AC5: Call sites

Update the remaining runtime modules without altering numerical paths.

Acceptance: reductions, alignment and stacking use explicit Polars expressions or the existing NumPy conversion boundary; no module gains a new pandas or xarray import; no public function signature changes beyond the container type.

### AC6: Performance record

Measure the cost of the container swap rather than asserting it is neutral.

Acceptance: construction, `to_numpy` extraction and long-form conversion are benchmarked on the same fixture and environment before and after; the receipt states the observed delta and the accepted bound; a regression beyond the bound blocks acceptance.

### AC7: No new persistence

Acceptance: no serialization format is added by this track; the existing JSON wire contract remains the only supported interchange; any proposed Parquet writer is deferred to a separate decision.

### AC8: Residual migration

Ownership: this track owns every xarray reference; pandas references, packaging, docs and workflows belong to `pandas_free_io_20261026`. Where a file holds both, this track edits it first.

Remove xarray from every consumer of the container so the only xarray references left are the migration documentation.

Acceptance: no `import xarray` or `from xarray` remains under `voiage/`, `tests/`, `examples/` or `scripts/`; `voiage/cli.py` and `voiage/core/io.py` build and consume the Polars container; test assertions are converted rather than deleted, and any deletion records a reason; the script `scripts/validate_core_api_fixtures.py` passes without xarray. Declared dependencies stay until `pandas_free_io_20261026`.

## Non-functional and external boundaries

No dependency removal in this track: `pandas` and `xarray` stay declared until `pandas_free_io_20261026`. No binding change: Rust, R and Julia conformance fixtures must pass unchanged, which is the evidence that the wire contract was not touched. No algorithm, RNG, tolerance or default change. Do not introduce a second container abstraction alongside the Polars frame.

## Verification

Freeze exact commands in the first task of each phase. Write failing tests before implementation; test collection of zero is failure. Final candidate runs `tox run-parallel -p 2` under `AGENTS.md`, the numerical reference cases, the binding conformance fixtures, and `scripts/validate_core_api_contract.py`. Required hosted checks apply to the delivered PR revision.
