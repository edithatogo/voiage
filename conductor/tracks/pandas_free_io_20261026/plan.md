# Implementation plan: Pandas-free external I/O and packaging

All tasks are pending. Ownership: this track migrates every pandas reference and owns the repository-wide zero-reference check; `polars_container_20261026` migrates every xarray reference, and Phase 7 here cannot close before that track's Phase 7. This track is registered planning only; implementation is not authorised. Prerequisite: `polars_container_20261026` for the acceptance gate; implementation may run in parallel.

## Phase 1: Direct-use removal

- [ ] T1.1 — Freeze the current public signatures, return types and the exact `git grep` inventory of pandas imports; record the expected red result.
- [ ] T1.2 — Port `ecosystem_integration` CSV and Parquet paths and the widget CSV upload to Polars, preserving the public surface. (AC1).
- [ ] T1.3 — Validate: no `import pandas` remains in either module; existing ecosystem and widget tests pass unchanged; record command, environment and digest. (AC1).

## Phase 2: Read semantics

- [ ] T2.1 — Freeze the current fixture set and add the all-null and mixed-type fixtures before changing readers.
- [ ] T2.2 — Make dtype inference explicit and define null versus NaN handling across load, arithmetic and export. (AC2).
- [ ] T2.3 — Validate: the new fixtures produce defined results; no silent cast occurs; the null policy is documented and tested. (AC2).

## Phase 3: Excel

- [ ] T3.1 — Record the current Excel engine, its extra and the existing fixtures.
- [ ] T3.2 — Substitute the Polars-native engine, update the `excel` extra and record licence and wheel availability on the supported interpreter. (AC3).
- [ ] T3.3 — Validate: Excel fixtures pass; a missing engine raises the documented error with no silent fallback; 3.15.1 wheel availability is demonstrated. (AC3).

## Phase 4: Packaging reconciliation

- [ ] T4.1 — Freeze the four dependency surfaces and their current disagreement; state the intended single set.
- [ ] T4.2 — Reconcile `pyproject.toml`, `conda-recipe/meta.yaml` (and `conda-recipe/README.md`), `tox.ini` and the dependency-frontier configuration, including `.github/workflows/dependency-preview.yml`; remove pandas and xarray; retain polars and pyarrow; pin the Polars floor and ceiling with a recorded reason and run the suite at the floor and at the latest release. (AC4).
- [ ] T4.3 — Validate: the drift check fails on disagreement and passes on the reconciled set; a clean install resolves without pandas or xarray. (AC4).

## Phase 5: Enforcement

- [ ] T5.1 — Write the failing subprocess gate and the lint rule before the package is expected to satisfy them.
- [ ] T5.2 — Add the no-extras tox environment and the `sys.modules` assertion. (AC5).
- [ ] T5.3 — Validate: the gate fails when a pandas import is reintroduced and passes on the candidate; the lint rule has no exemption; the gate runs in hosted CI. (AC5).

## Phase 6: Producer compatibility and boundary

- [ ] T6.1 — Freeze the interchange fixtures, including a pandas producer object supplied by a caller.
- [ ] T6.2 — Verify pandas-producing and Polars-producing objects are both accepted without a Voiage pandas import, and that zero-copy is preferred where available. (AC6).
- [ ] T6.3 — Validate: non-conforming input fails with the documented diagnostic; the wire contract and binding fixtures are unchanged; no new persistence format exists. (AC6, AC7).

## Phase 7: Residual surfaces and zero-reference end state

- [ ] T7.1 — Freeze the repository-wide `git grep -w` inventory of pandas and xarray outside `voiage/`: tests (8 modules mention pandas), docs (9 pages mention pandas), scripts (`hpc_package_smoke.py`, `simple_validation.py`, `validate_easybuild_root_graph.py`), `.github/workflows/dependency-preview.yml`, the EasyBuild and Spack overlays, SBOMs and solver logs; classify each hit as live dependency, live documentation, generated-code template, or frozen historical evidence.
- [ ] T7.2 — Port or remove every live hit: the pandas snippet emitted into generated code at `voiage/ecosystem_integration.py:757` becomes Polars; live docs, scripts and the preview workflow drop pandas; live overlay dependency declarations are reconciled with the packaging set; frozen historical evidence (SBOMs, solver logs, dated release records) is left byte-unchanged. (AC8).
- [ ] T7.3 — Validate: a test enforces an explicit allow-list (migration guide, translation reference, producer-compatibility tests, frozen historical evidence) and fails on any other pandas or xarray reference, including string literals and generated-code templates; the allow-list is recorded in the track; the overlay and packaging validators still pass. (AC8).

## Final acceptance

- [ ] CLOSE.1 — Run the repository-wide zero-reference check against the exact merged revision and confirm no hit lies outside the allow-list; review every AC (AC1–AC8) against the exact tested revision; hosted checks must pass before any acceptance claim.
- [ ] CLOSE.2 — Reconcile plan, metadata and append-only evidence; publish the packaging set and the no-pandas gate result to `translation_reference_20261026`.
