# Specification: Tabular contract v3 and xarray removal

## Scope and authority

Baseline: `8fc7748066ea4391d4971e7c9fb3f51507dcf052`. Follow root `AGENTS.md` and the parent programme repository boundaries. This track owns only the normative contract revision and its fixtures; it authorises no runtime change.

Owner decision recorded on 2026-10-26: Voiage 3.0.0 removes the xarray surface outright. There is no compatibility extra, no deprecation window and no shim: `ValueArray.dataset`, `ParameterSet.dataset`, `from_dataset` and `to_dataset` are removed, replaced by `frame: pl.DataFrame`, `from_polars` and `to_polars`, while `ValueArray.values` returns `np.ndarray` (aliasing `numpy_values` rather than `xr.DataArray`), and the migration guide is the only bridge.

This track exists because the current normative documents forbid the change. `specs/core-api/decision-record.md` and `specs/core-api/foundation.md` state that xarray-labelled arrays are the Python in-memory reference model and that Polars is adapter-only tabular tooling, and require an explicit spec revision before a later track may reinterpret them. Without this track, every implementation task in the sibling tracks is out of contract on arrival.

## Required context

- `specs/core-api/decision-record.md`
- `specs/core-api/foundation.md`
- `specs/core-api/contract-index.md`
- `specs/core-api/extension-evolution.md`
- `specs/v2/stable-api.json`
- `specs/v2/compatibility-policy.json`
- `specs/core-api/examples/v1/value-array.example.json`
- `specs/core-api/examples/v1/parameter-set.example.json`
- `specs/core-api/schemas/v2/normalized-input-bundle.schema.json`
- `tests/test_v2_stable_api_contract.py`
- `docs/astro-site/src/content/docs/user-guide/migration-guide.mdx`
- `voiage/schema.py`

## Acceptance criteria

### AC1: Authority revision

Open the explicit spec revision that `specs/core-api/decision-record.md` and `specs/core-api/foundation.md` require, rather than editing behaviour to fit an unreviewed implementation.

Acceptance: a dated decision revision in `specs/core-api/decision-record.md` and `specs/core-api/foundation.md` records Polars as the Python in-memory reference model for tabular containers, records xarray removal at 3.0.0, and preserves the unchanged decisions — NumPy as execution baseline, optional JAX/torch acceleration, Arrow/Parquet as the cross-language interchange boundary. No normative document still asserts xarray is the current reference model.

### AC2: Versioned contract

Add `specs/v3/stable-api.json` and `specs/v3/compatibility-policy.json` following the existing v1-to-v2 precedent.

Acceptance: v3 supersedes v2 with `classification: "major-breaking-change"`; the four removed symbols (`ValueArray.dataset`, `ParameterSet.dataset`, `from_dataset`, `to_dataset`) are recorded as removed, not deprecated; the Polars container surface (`frame: pl.DataFrame`, `from_polars`, `to_polars`) and `ValueArray.values -> np.ndarray` return-type change are recorded; the v1 and v2 files are byte-unchanged; and a governance test (`tests/test_v3_stable_api_contract.py`) mirroring `tests/test_v2_stable_api_contract.py` pins the v3 boundary.

### AC3: Schema vocabulary reuse

Express the Polars container contract in the vocabulary that already exists, instead of inventing a second one.

Acceptance: the new container terms resolve to the existing `specs/core-api` identifiers (`table_id`, `field_id`, `role: "net_benefit"`, `layout: "wide"`, `dtype: "float64"`), and the existing v1 `value-array.example.json` and `parameter-set.example.json` fixtures round-trip unchanged. Multi-perspective 3D `ValueArray` instances (`from_numpy_perspectives` with `perspective_names`) are explicitly specified as wide strategy columns of `pl.Array(pl.Float64, n_perspectives)` alongside `perspective_names: list[str] | None` so `values` and `numpy_values` return a 3D `(n_samples, n_strategies, n_perspectives)` NumPy array.

### AC4: Migration guide

Document the removal for downstream consumers, since no compatibility path is provided.

Acceptance: every removed symbol has either a replacement recipe (`frame`, `from_polars`, `to_polars`, `numpy_values`) or an explicit "no replacement" statement; the guide states that `ValueArray.values` now returns `np.ndarray` (matching `numpy_values` instead of `xr.DataArray`), that `numpy_values`, `parameters`, `from_numpy`, `from_numpy_perspectives`, and the decision-analysis entry points are unchanged, that `pandas` and `xarray` are no longer installed transitively, and names the supported external ingestion route (`__dataframe__` / `__arrow_c_stream__`).

### AC5: Repeatability and non-goals

Acceptance: the spec states that no new persistence format is introduced, that the JSON wire contract and the Rust/R/Julia binding surfaces are unchanged, and that no algorithm, RNG or numerical behaviour changes. The revision names the exact documents it supersedes.

## Non-functional and external boundaries

No runtime implementation, no dependency edit, no lock regeneration, no release and no publication is included in this track. Do not rewrite v1 or v2 contract files. Do not weaken the Rust-first core or the binding conformance fixtures. Do not add a compatibility extra or a deprecation shim; the owner decision is a clean major break.

## Verification

Freeze exact test commands in the first task of each phase before implementation. Existing context files are starting points, not unrestricted write permission. Add meaningful negative tests before code; test collection of zero is failure. Final candidate runs `tox run-parallel -p 2` under `AGENTS.md` plus the contract validators (`scripts/validate_core_api_contract.py`, `scripts/validate_core_api_fixtures.py`, `scripts/validate_conductor_github_cross_references.py`). Required hosted checks apply to the delivered PR revision.
