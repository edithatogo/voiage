# Implementation plan: Pandas-free external I/O and packaging

T1.1 baseline is frozen, T1.2 is merged in PR #1203, and Phase 2 qualifies the migrated CSV/Parquet paths; remaining tasks are pending. Ownership: this track migrates every pandas reference and owns the repository-wide zero-reference check; `polars_container_20261026` migrates every xarray reference, and Phase 7 here cannot close before that track's Phase 7. Maintainer authorised the recommended follow-up on 2026-10-27. Prerequisite `polars_container_20261026` was accepted and merged in PR #1202; the package-level absence/enforcement gate was accepted in merged PR #1207; repository-wide reference acceptance remains open.

## Phase 1: Direct-use removal

- [x] T1.1 — Freeze the current public signatures, return types and the exact `git grep` inventory of pandas imports; record the expected red result. See `phase-1-baseline.json` (36 hits, 42 signatures) and `inventory.md`.
- [x] T1.2 — Port `ecosystem_integration` CSV and Parquet paths and the widget CSV upload to Polars, preserving the public surface. (AC1). Implemented in `01a398ca`; see `t12-acceptance-python315.json`.
- [x] T1.3 — Validate: no `import pandas` remains in either module; existing ecosystem and widget tests pass unchanged; record command, environment and digest. (AC1). Generated template cleanup removes the remaining literal import; existing connector tests pass, with final evidence in Phase 7.

## Phase 2: Read semantics [checkpoint: 27d206a]

- [x] T2.1 — Freeze the current fixture set and add the all-null and mixed-type fixtures before changing readers. Committed in `8c2575bd`; see `phase-2-baseline.json` and `tests/fixtures/tabular_io/read_semantics/`.
- [x] T2.2 — Make dtype inference explicit and define null versus NaN handling across load, arithmetic and export. (AC2). Implemented in `dd5016c6`; native Excel inference remains a Phase 3 gate.
- [x] T2.3 — Validate: the new fixtures produce defined results; no silent cast occurs; the null policy is documented and tested. (AC2). See `phase-2-acceptance-python315.json` and `phase-2-review.md`; Excel inference qualification remains part of Phase 3.

## Phase 3: Excel [checkpoint: 9c9f37cb]

- [x] T3.1 — Record the current Excel read and multi-sheet write engines, the `excel` extra and the existing fixtures. See `phase-3-baseline.json` and `phase-3-engines-python315.json`.
- [x] T3.2 — Substitute Polars-native Excel reading (`python-calamine` raw cells into Polars; named AC3 alternative preserving the frozen mixed-cell/header/time behavior) and multi-sheet writing (`xlsxwriter` with `pl.DataFrame.write_excel(workbook=wb, worksheet=table_name)`), update the `excel` extra to declare both, and record licence and wheel availability on the supported interpreter. (AC3).
- [x] T3.3 — Validate: Excel read and multi-sheet write fixtures pass; a missing engine raises the documented error with no silent fallback; wheel availability on the supported interpreter is demonstrated. (AC3). Local qualification: `phase-3-acceptance-python315.json`, the subsequent review-fix `phase-3-whitespace-python315.json`, and `phase-3-review.md`; hosted/platform checks passed and PR #1205 merged as `5546e9b6` on 2026-10-27.

## Phase 4: Packaging reconciliation [checkpoint: 29a83cfe]

- [x] T4.1 — Freeze the four dependency surfaces and their current disagreement; state the intended single set. See `phase-4-baseline.json`; runtime requirements exclude both legacy packages, while the unpublished producer-test group is isolated from wheel dependencies and default installs.
- [x] T4.2 — Reconcile `pyproject.toml`, `conda-recipe/meta.yaml` (and `conda-recipe/README.md`), `tox.ini` and the dependency-frontier configuration, including `.github/workflows/dependency-preview.yml`; remove pandas and xarray; retain polars and pyarrow; pin the Polars floor and ceiling with a recorded reason and run the suite at the floor and at the latest release. (AC4).
- [x] T4.3 — Validate: the drift check fails on disagreement and passes on the reconciled set; a clean install resolves without pandas or xarray. (AC4). See `phase-4-acceptance-python315.json` and `phase-4-review.md`; producer tools are unpublished/opt-in, native Conda publication and later absence enforcement remain separate gates.

## Phase 5: Enforcement [checkpoint: 4bb03447]

- [x] T5.1 — Write the failing subprocess gate and the lint rule before the package is expected to satisfy them. See `phase-5-baseline.json`; real negative import/closure/shadow controls pass, while three wiring contracts fail before activation.
- [x] T5.2 — Add the no-extras tox environment and the `sys.modules` assertion. (AC5).
- [x] T5.3 — Validate: the gate fails when a pandas import is reintroduced and passes on the candidate; the lint rule has no exemption; the gate runs in hosted CI. (AC5). Local qualification: `phase-5-acceptance-python315.json`, `phase-5-protected-ci-python315.json`, `phase-5-benchmark-compatibility-python315.json` and `phase-5-review.md`; hosted/platform acceptance passed and PR #1207 merged as `1383f9c3`.

## Phase 6: Producer compatibility and boundary [checkpoint: 4cd4601e]

- [x] T6.1 — Freeze the interchange fixtures, including a pandas producer object supplied by a caller. Baseline and red automation contracts: `61b8cbc6`, `phase-6-baseline.json` and `tests/producer_compatibility/producer.json`.
- [x] T6.2 — Verify pandas-producing and Polars-producing objects are both accepted without a Voiage pandas import, and that zero-copy is preferred where available. (AC6). Real installed-wheel lane, import ownership, shared numeric buffers and existing strict-copy policy qualified; implementation `f8675bd4`, qualified in the Phase 6 receipt.
- [x] T6.3 — Validate: non-conforming input fails with the documented diagnostic; the wire contract and binding fixtures are unchanged; no new persistence format exists. (AC6, AC7). See `phase-6-acceptance-python315.json` and `phase-6-review.md`; all 15 local tox environments and hosted checks passed; PR #1208 merged as `f6e6ee39`.

## Phase 7: Residual surfaces and zero-reference end state [checkpoint: ea40c21c]

Local qualification: `phase-7-acceptance-python315.json` and `phase-7-review.md`. Hosted and exact merged-revision CLOSE gates remain open.

- [x] T7.1 — Freeze the repository-wide `git grep` inventory of `\bpandas\b`, `\bxarray\b`, `\bpd\.`, `\bxr\.`, and `itertuples` outside `voiage/`: tests (8 modules mention pandas), docs (9 pages mention pandas), scripts (`hpc_package_smoke.py`, `simple_validation.py`, `validate_easybuild_root_graph.py`), `.github/workflows/dependency-preview.yml`, the EasyBuild and Spack overlays, SBOMs and solver logs; classify each hit as live dependency, live documentation, generated-code template, or frozen historical evidence. Frozen in `5ab25903` / `phase-7-baseline.json` (841 lines, 200 files).
- [x] T7.2 — Port or remove every live hit: the pandas snippets emitted into generated notebook code at `voiage/ecosystem_integration.py` lines 757 (`import pandas as pd`), 827 (`pd.DataFrame(results)`), and 844 (`results_df.itertuples()`) become Polars (`import polars as pl`, `pl.DataFrame(results)`, `results_df.iter_rows(named=True)`); live docs, scripts and the preview workflow drop pandas; live overlay dependency declarations are reconciled with the packaging set; frozen historical evidence (SBOMs, solver logs, dated release records) is left byte-unchanged. (AC8).
- [x] T7.3 — Validate: a test enforces an explicit allow-list (migration guide, translation reference, producer-compatibility tests, frozen historical evidence) and fails on any other `pandas`, `xarray`, `pd.`, `xr.`, or `itertuples` reference, including string literals and generated-code templates; the allow-list is recorded in `reference-policy.json`; historical and current overlay validators and packaging validation pass. (AC8). Local qualification is recorded in the Phase 7 receipt; hosted and final merged-revision gates remain open.

## Final acceptance

- [ ] CLOSE.1 — Run the repository-wide zero-reference check against the exact merged revision and confirm no hit lies outside the allow-list; review every AC (AC1–AC8) against the exact tested revision; hosted checks must pass before any acceptance claim.
- [ ] CLOSE.2 — Reconcile plan, metadata and append-only evidence; publish the packaging set and the no-pandas gate result to `translation_reference_20261026`.
