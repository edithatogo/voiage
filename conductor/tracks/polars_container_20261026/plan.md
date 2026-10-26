# Implementation plan: Polars-native core containers

All tasks are pending. Ownership: this track migrates every xarray reference (runtime, tests, examples, scripts); `pandas_free_io_20261026` migrates every pandas reference and owns the repository-wide zero-reference check. Implementation authorised by the maintainer on 2026-10-26. Prerequisite: `tabular_contract_v3_20261026`.

## Phase 1: Inventory

- [x] T1.1 — Freeze the baseline commit, the exact inventory command (`git grep`) and its output digest; state at most five implementation paths for AC1. (phase-1-baseline.json)
- [x] T1.2 — Classify every runtime xarray reference, including function-level lazy imports (five in `voiage/schema.py`, one each in `voiage/cli.py` and `voiage/methods/basic.py`), as container, validation, conversion or dead code, and assign each an owner task; `voiage/cli.py` is in scope. (AC1). (inventory.md)
- [x] T1.3 — Validate: no unclassified reference; dead references are deleted rather than ported; record command, exit status and digest. (AC1). (evidence.jsonl)

## Phase 2: Containers

- [x] T2.1 — Freeze the current `ValueArray` and `ParameterSet` signatures, exception messages and constructor tests; confirm the expected red result before implementation. (phase-2-baseline.json)
- [x] T2.2 — Replace both containers with wide Polars frames (`frame: pl.DataFrame`, `from_polars`, `to_polars`), supporting 2D `Float64` strategy columns and 3D `pl.Array(pl.Float64, n_perspectives)` multi-perspective columns (`from_numpy_perspectives`, `perspective_names`), and making `ValueArray.values` return `np.ndarray` alongside `numpy_values`/`parameters`/`from_numpy`. (AC2). (voiage/schema.py)
- [x] T2.3 — Validate: shapes, dtypes and null policy are enforced at construction; the public signatures are unchanged; the schema tests pass on the new container. (AC2). (tests/test_schema*.py)

## Phase 3: Validation parity

- [x] T3.1 — Enumerate the structural errors the xarray container raised and freeze their current messages and exception types. (evidence.jsonl)
- [x] T3.2 — Implement equivalent column, dtype, length and duplicate-name validation, and reject unrecognised extra variables explicitly. (AC3). (voiage/schema.py)
- [x] T3.3 — Validate: a negative test exists for each enumerated error; extras fail closed; no error type silently changes. (AC3). (tests/test_schema_validation_parity.py)

## Phase 4: Numerical parity

- [x] T4.1 — Freeze the reference cases and the exact parity command, including the `np.array_equal` assertion and the fixture set. (parity_reference.py)
- [x] T4.2 — Port the remaining call sites in metamodels, methods, analysis and backends without altering numerical paths. (AC5). (voiage/metamodels.py, voiage/methods/*, voiage/parallel/monte_carlo.py, voiage/core/memory_optimization.py)
- [x] T4.3 — Validate: every reference case is bit-identical, column order is identical, no tolerance was widened, and the `specs/core-api` examples produce identical output; record environment and digests. (AC4, AC5). (parity-before.json, parity-after.json, evidence.jsonl)

## Phase 5: Performance record

- [ ] T5.1 — Freeze the benchmark fixture, machine and command; record the pre-change numbers before the container swap is final.
- [ ] T5.2 — Re-measure construction, `to_numpy` extraction and long-form conversion on the final candidate. (AC6).
- [ ] T5.3 — Validate: the receipt states the observed delta and the accepted bound; a regression beyond the bound blocks acceptance. (AC6).

## Phase 6: Boundary confirmation

- [ ] T6.1 — Confirm no persistence format, dependency removal or binding change was introduced.
- [ ] T6.2 — Run the Rust, R and Julia conformance fixtures unchanged against the new container. (AC7).
- [ ] T6.3 — Validate: bindings pass without fixture edits, `specs/core-api` is unchanged, and the full local gate passes. (AC7).

## Phase 7: Residual runtime, test, example and script migration

- [ ] T7.1 — Freeze the exact `git grep` inventory of xarray imports and `.dataset` references outside the Phase 2–4 call sites: `voiage/cli.py` (xr.Dataset construction), `voiage/core/io.py`, the 28 test modules that import pandas or xarray (including `tests/conftest.py`), the 10 examples that mention xarray (3 import it), and `scripts/validate_core_api_fixtures.py`; record digest and the expected red result.
- [ ] T7.2 — Port `voiage/cli.py` and `voiage/core/io.py` to the Polars container; rewrite every `.dataset` assertion and xarray fixture in the tests to the new surface; update the examples and `scripts/validate_core_api_fixtures.py` so none imports xarray. (AC8).
- [ ] T7.3 — Validate: `git grep` for `import xarray` and `from xarray` returns nothing under `voiage/`, `tests/`, `examples/` and `scripts/`; the converted tests keep their original assertions' intent (no assertion deleted without a recorded reason); the fixture validator and the contract validators still pass; record command, environment and digest. (AC8).
## Final acceptance

- [ ] CLOSE.1 — Review every AC (AC1–AC8) against the exact tested revision; the hosted checks must pass before any acceptance claim.
- [ ] CLOSE.2 — Reconcile plan, metadata and append-only evidence with the accepted revision; hand the container and perf receipts to `pandas_free_io_20261026`.
