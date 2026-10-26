# Specification: pandas and xarray to Polars translation reference

## Scope and authority

Baseline: `ca3f05cc8adcc74b99fe236152327d0cd3679553`. Follow root `AGENTS.md`. This track owns one documentation artefact and its executable tests. It changes no runtime code.

Prerequisite: `polars_container_20261026`. The reference must describe the container that actually shipped, not a proposal.

## Purpose

A syntax table is not a migration reference. The differences that break real ports are semantic: the absent index, `null` versus `NaN`, reduction and grouping order, label alignment, and float formatting. This track records those differences for the idioms that Voiage and its consumers actually use, and binds each entry to a test.

## Required context

- `voiage/schema.py` and the ported call sites from `polars_container_20261026`
- `voiage/ecosystem_integration.py` and `voiage/core/io.py` from `pandas_free_io_20261026`
- `specs/core-api/examples/v1/value-array.example.json`, `parameter-set.example.json`
- `docs/astro-site/src/content/docs/user-guide/migration-guide.mdx`
- `docs/astro-site/src/content/docs/developer-guide/v2-contract-architecture.md`
- Consumer idioms: `scripts/run_empirical_longitudinal.py`, `scripts/run_release_calibration.py` and `scripts/verify_empirical_longitudinal.py` in the `tobac` repository, captured into a committed, digest-pinned manifest (`conductor/tracks/translation_reference_20261026/tobac-consumer-idioms.json`) so verification is hermetic and never requires an external checkout on CI runners

## Acceptance criteria

### AC1: Derived inventory

Derive the idiom list from code that exists rather than from a generic cookbook.

Acceptance: each covered idiom cites the file and line that uses it in Voiage or in the committed `tobac-consumer-idioms.json` manifest (which records the `tobac` source commit SHA, file paths, line ranges, and SHA-256 digests); idioms with no in-repository or consumer-manifest use are excluded or explicitly marked as anticipatory; the inventory records the command used to derive it.

### AC2: Semantic reference

Acceptance: every entry states the pandas or xarray form, the Polars form, and the behavioural difference. At minimum it covers: index absence and positional access, attribute versus bracket column access, row iteration, `groupby` versus `group_by` and ordering guarantees, `concat` and schema alignment, `xr.align` versus explicit join, dimension reduction, `resample`/`rolling` boundary rules, `interp` versus `join_asof`, `NaN` versus `null`, and CSV float formatting.

### AC3: Executable parity

Documentation that is not tested decays, so each row must run.

Acceptance: each semantic entry has a test that exercises both forms and asserts the documented relationship — equal values, equal ordering, documented null behaviour, or a documented and measured difference. Tests state which rows are exact matches and which are bounded approximations.

### AC4: Non-equivalences

Acceptance: the reference names what has no equivalent and what a caller must do instead. At minimum: printf-style `%g` formatting is unavailable in the Polars CSV writer, xarray `attrs` have no container equivalent, automatic label alignment has no counterpart, and N-dimensional interpolation is out of scope. Each non-equivalence states the supported replacement or states plainly that none exists.

### AC5: Publication

Acceptance: the reference is published in the documentation source, linked from the v2-to-v3 migration guide, and the Astro build passes; links resolve; no example calls a symbol removed by the v3 contract.

## Non-functional and external boundaries

No runtime change, no dependency change, no contract change. Do not present untested equivalences as verified. Do not describe the removed compatibility surface as still available. Do not import pandas in tests that run in the pandas-free environment unless the test is explicitly marked and skipped when pandas is absent.

## Verification

Freeze exact commands in the first task of each phase. Write the failing parity test before settling a row's wording. Final candidate runs the translation test module, the Astro docs build and the full local gate under `AGENTS.md`. Required hosted checks apply to the delivered PR revision.
