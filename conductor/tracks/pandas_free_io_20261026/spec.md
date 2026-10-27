# Specification: Pandas-free external I/O and packaging

## Scope and authority

Baseline: `ca3f05cc8adcc74b99fe236152327d0cd3679553`. Follow root `AGENTS.md`. This track owns external input/output, the widget upload path, the packaging surfaces and the enforcement gates that keep pandas out.

Prerequisite: `polars_container_20261026`. At the baseline, `voiage/ecosystem_integration.py` imported both `pandas` and `voiage.schema`, and `schema` imported xarray, which imported pandas. The container prerequisite was accepted in PR #1202, merged as `b1c63e7f216f3ffee0871e634142026e2763eea8` on 2026-10-27 with Python 3.15.1 and Polars 2.1.0 qualified. The package-level pandas-free gate remains pending until the later phases remove the remaining dependencies and references.

## Evidence basis

Measured on the baseline:

- Only two runtime modules import pandas: `voiage/ecosystem_integration.py` (CSV, Excel and Parquet readers plus export bundles) and `voiage/widgets/voi_widgets.py` (CSV upload parsing).
- `voiage/core/io.py` already reads and writes CSV with the standard library and NumPy.
- `voiage/ingestion/dataframe.py` is backend-neutral by design, using `__dataframe__` and `__arrow_c_stream__` through PyArrow, and documents no pandas or Polars dependency.
- Eight test modules mention pandas (`test_ecosystem_integration.py` and `test_standardized_ingestion_providers.py` among them); the two I/O modules are the only runtime importers.
- `voiage/ecosystem_integration.py` emits an `import pandas as pd` snippet inside a generated-code string at line 757, `results_df = pd.DataFrame(results)` at line 827, and `for row in results_df.itertuples():` at line 844, which no import check or bare `pandas` word-grep can see.
- pandas or xarray is also named by 9 docs pages, 3 scripts, `.github/workflows/dependency-preview.yml`, `conda-recipe/README.md`, the EasyBuild and Spack overlays, and frozen SBOM and solver-log evidence that must not be rewritten.
- The declared dependency bounds disagree across surfaces on this baseline: `pyproject.toml` and `conda-recipe/meta.yaml` declare `pandas>=1.3,<3.0` and `xarray>=0.19,<2025.0`, while `tox.ini` typecheck pins `pandas>=3.0.3` and `xarray>=2026.7.0`. The contradiction predates this track and must be reconciled, not inherited.

## Required context

- `voiage/ecosystem_integration.py`
- `voiage/widgets/voi_widgets.py`
- `voiage/core/io.py`
- `voiage/ingestion/dataframe.py`
- `pyproject.toml`
- `conda-recipe/meta.yaml`
- `tox.ini`
- `.github/workflows/dependency-frontier.yml` and `.github/workflows/dependency-preview.yml`
- `conda-recipe/README.md`, `packaging/easybuild-overlay/` and `packaging/spack-overlay/` (live declarations only)
- `scripts/hpc_package_smoke.py`, `scripts/simple_validation.py`, `scripts/validate_easybuild_root_graph.py`
- `tests/test_ecosystem_integration.py`
- `tests/test_standardized_ingestion_providers.py`

## Acceptance criteria

### AC1: Direct-use removal

Replace both direct pandas uses with Polars without changing the public surface.

Acceptance: `ecosystem_integration` reads and writes CSV and Parquet through Polars, the export bundle path is preserved, and the widget upload path parses CSV through Polars. Public function names, signatures and return types are unchanged, and `import pandas` no longer appears in either module.

### AC2: Read semantics

Dataframe readers change inference behaviour when the backend changes, so inference must become explicit.

Acceptance: every reader pins its dtype policy with explicit overrides or an explicit unbounded inference length; a fixture covering an all-null column and a mixed-type column produces a defined result rather than a silent cast; `null` versus `NaN` propagation through load, arithmetic and export is documented and tested.

### AC3: Excel substitution

Both the Excel reader (`import_health_data`) and the multi-sheet Excel writer (`export_health_data(..., format_type="xlsx")`) require engine substitutions rather than drop-in replacements.

Acceptance: the read engine (Polars' `calamine`/`fastexcel` default, or a named alternative) and the multi-sheet write engine (`xlsxwriter` via `pl.DataFrame.write_excel(workbook=wb, worksheet=table_name)`) are recorded with their licences, wheel availability and behaviour on the supported interpreter; the `excel` extra declares both the read and write engines actually used; existing Excel read and write fixtures pass; a missing engine fails with the documented error rather than a silent fallback.

### AC4: Packaging reconciliation

Acceptance: `pyproject.toml`, `conda-recipe/meta.yaml`, `tox.ini` and the dependency-frontier configuration agree on one declared set; `pandas` and `xarray` are absent from declared dependencies; `polars` and `pyarrow` remain declared, with `pyarrow` retained because the ingestion interchange depends on it; a drift check fails when the four surfaces disagree.

### AC5: Enforcement

Intent is not a gate, so the absence of pandas must be executable.

Acceptance: a subprocess test asserts that `pandas` and `xarray` are not present in `sys.modules` after `import voiage`; a lint rule rejects a direct `import pandas` or `import xarray` anywhere in `voiage/`, with no exemption, because the owner decision is outright removal; and a tox environment without optional extras runs the package-level gate.

### AC6: External producer compatibility

Acceptance: a pandas DataFrame supplied by an external caller is still accepted through the existing `__dataframe__`/`__arrow_c_stream__` route, without Voiage importing pandas; a Polars frame is accepted zero-copy where the producer supports it; a non-conforming object fails with the documented diagnostic.

### AC7: Boundary

Acceptance: no numerical path, container semantics, wire contract or binding changes in this track; the CSV output format is unchanged except where the track records an explicit, documented difference; no new persistence format is introduced.

### AC8: Zero-reference end state

Ownership split with `polars_container_20261026`: that track owns every xarray reference (runtime, tests, examples, scripts); this track owns every pandas reference and the repository-wide end-state check. Neither track edits the other's references; where one file holds both, the container track edits first and this track rebases. The end-state check cannot pass until the container track's AC8 has landed.

Acceptance: a repository-wide `git grep` for `\bpandas\b`, `\bxarray\b`, `\bpd\.`, `\bxr\.`, and `itertuples` returns only entries on a recorded allow-list; the allow-list contains the v2-to-v3 migration guide, the translation reference, the explicitly marked producer-compatibility tests and frozen historical evidence, and nothing else; a test enforces it, including string literals and generated-code templates; live packaging overlays declare neither package. The Polars floor and ceiling are pinned with a recorded reason and the suite passes at both ends.

## Non-functional and external boundaries

The `specs/core-api` JSON contract and the Rust, R and Julia bindings must remain unchanged and their conformance fixtures must pass. Do not add a pandas compatibility shim, a lazy import fallback or a runtime feature check that keeps pandas reachable. Do not change SciPy, NumPy or scikit-learn bounds as part of this track.

## Verification

Freeze exact commands in the first task of each phase. Write failing tests before implementation; test collection of zero is failure. Final candidate runs `tox run-parallel -p 2` under `AGENTS.md`, the no-pandas subprocess gate, the ingestion interchange tests, the packaging drift check and the binding conformance fixtures. Required hosted checks apply to the delivered PR revision.
