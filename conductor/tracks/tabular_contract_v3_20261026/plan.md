# Implementation plan: Tabular contract v3 and xarray removal

All tasks are pending. This track is registered planning only; implementation is not authorised. Prerequisites: none. Successor tracks are `polars_container_20261026`, `pandas_free_io_20261026` and `translation_reference_20261026`.

## Phase 1: Authority revision

- [ ] T1.1 — Freeze exact input hashes for `specs/core-api/decision-record.md`, `specs/v2/stable-api.json` and `specs/v2/compatibility-policy.json`; name at most five implementation paths and the expected red/green result before writing prose.
- [ ] T1.2 — Open the decision revision that replaces the xarray reference-model clause, records Polars as the Python in-memory tabular model and records outright removal at 3.0.0. (AC1).
- [ ] T1.3 — Validate: no normative document still asserts xarray is the reference model; NumPy, optional-acceleration and Arrow interchange decisions are unchanged; record source, environment, command, exit status and artifact digest, then review the diff and stop on unresolved failure. (AC1).

## Phase 2: Versioned contract

- [ ] T2.1 — Freeze the v2 contract files and their digests; confirm v1 and v2 remain byte-unchanged throughout this phase.
- [ ] T2.2 — Author `specs/v3/stable-api.json` and `specs/v3/compatibility-policy.json`, recording the four removed symbols as removed rather than deprecated. (AC2).
- [ ] T2.3 — Validate: v3 supersedes v2 as a major breaking change, a governance test mirrors `tests/test_v2_stable_api_contract.py`, and the v1/v2 digests are unchanged; write the failing governance test before the contract files. (AC2).

## Phase 3: Schema vocabulary

- [ ] T3.1 — Freeze the existing `specs/core-api` identifiers and the two v1 example fixtures; state the round-trip command and expected output.
- [ ] T3.2 — Express the Polars container contract using the existing `table_id`/`field_id`/`role`/`layout`/`dtype` vocabulary and record the wide-layout decision. (AC3).
- [ ] T3.3 — Validate: `value-array.example.json` and `parameter-set.example.json` round-trip unchanged; no duplicate vocabulary is introduced; run `scripts/validate_core_api_contract.py` and `scripts/validate_core_api_fixtures.py`; note that the fixtures validator imports xarray until `polars_container_20261026` Phase 7 ports it, so run it under the current environment and do not edit it here. (AC3).

## Phase 4: Migration guidance

- [ ] T4.1 — Enumerate every removed symbol reachable from the package `__init__` and record the replacement, or the absence of one; also record that pandas and xarray stop being installed transitively, so downstream code that relied on them must declare its own dependency.
- [ ] T4.2 — Update `docs/astro-site/src/content/docs/user-guide/migration-guide.mdx` with the v2-to-v3 recipes and the supported external ingestion route. (AC4).
- [ ] T4.3 — Validate: each removed symbol maps to a recipe or an explicit "no replacement"; unchanged helpers are stated as unchanged; the Astro docs build passes and no example still calls a removed symbol. (AC4).

## Phase 5: Boundary record

- [ ] T5.1 — Record the non-goals in the spec and the v3 policy: no new persistence format, no wire-contract change, no binding change, no numerical change.
- [ ] T5.2 — Cross-check that the Rust, R and Julia conformance fixtures and `specs/core-api` JSON are untouched by this track. (AC5).
- [ ] T5.3 — Validate: the contract index and extension-evolution documents reference v3, the cross-reference manifest validates, and the full local gate passes. (AC5).

## Final acceptance

- [ ] CLOSE.1 — Review every AC against the exact merged revision and bind the focused validators to it; no release or publication claim.
- [ ] CLOSE.2 — Reconcile plan, metadata, contract index and append-only evidence with the accepted v3 revision.
