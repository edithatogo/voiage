# Implementation plan: Polars-native core containers

All tasks are pending. This track is registered planning only; implementation is not authorised. Prerequisite: `tabular_contract_v3_20261026`.

## Phase 1: Inventory

- [ ] T1.1 — Freeze the baseline commit, the exact inventory command (`git grep`) and its output digest; state at most five implementation paths for AC1.
- [ ] T1.2 — Classify every runtime xarray reference as container, validation, conversion or dead code, and assign each an owner task. (AC1).
- [ ] T1.3 — Validate: no unclassified reference; dead references are deleted rather than ported; record command, exit status and digest. (AC1).

## Phase 2: Containers

- [ ] T2.1 — Freeze the current `ValueArray` and `ParameterSet` signatures, exception messages and constructor tests; confirm the expected red result before implementation.
- [ ] T2.2 — Replace both containers with wide Polars frames, preserving column order and the `values`/`numpy_values`/`parameters`/`from_numpy` surface. (AC2).
- [ ] T2.3 — Validate: shapes, dtypes and null policy are enforced at construction; the public signatures are unchanged; the schema tests pass on the new container. (AC2).

## Phase 3: Validation parity

- [ ] T3.1 — Enumerate the structural errors the xarray container raised and freeze their current messages and exception types.
- [ ] T3.2 — Implement equivalent column, dtype, length and duplicate-name validation, and reject unrecognised extra variables explicitly. (AC3).
- [ ] T3.3 — Validate: a negative test exists for each enumerated error; extras fail closed; no error type silently changes. (AC3).

## Phase 4: Numerical parity

- [ ] T4.1 — Freeze the reference cases and the exact parity command, including the `np.array_equal` assertion and the fixture set.
- [ ] T4.2 — Port the remaining call sites in metamodels, methods, analysis and backends without altering numerical paths. (AC5).
- [ ] T4.3 — Validate: every reference case is bit-identical, column order is identical, no tolerance was widened, and the `specs/core-api` examples produce identical output; record environment and digests. (AC4, AC5).

## Phase 5: Performance record

- [ ] T5.1 — Freeze the benchmark fixture, machine and command; record the pre-change numbers before the container swap is final.
- [ ] T5.2 — Re-measure construction, `to_numpy` extraction and long-form conversion on the final candidate. (AC6).
- [ ] T5.3 — Validate: the receipt states the observed delta and the accepted bound; a regression beyond the bound blocks acceptance. (AC6).

## Phase 6: Boundary confirmation

- [ ] T6.1 — Confirm no persistence format, dependency removal or binding change was introduced.
- [ ] T6.2 — Run the Rust, R and Julia conformance fixtures unchanged against the new container. (AC7).
- [ ] T6.3 — Validate: bindings pass without fixture edits, `specs/core-api` is unchanged, and the full local gate passes. (AC7).

## Final acceptance

- [ ] CLOSE.1 — Review every AC against the exact tested revision; the hosted checks must pass before any acceptance claim.
- [ ] CLOSE.2 — Reconcile plan, metadata and append-only evidence with the accepted revision; hand the container and perf receipts to `pandas_free_io_20261026`.
