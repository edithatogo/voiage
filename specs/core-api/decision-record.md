# Core API Decision Record

## Decision

The authoritative contract for `voiage` v1 will be written as Markdown semantics paired with JSON Schema for machine-readable structures.

## Rationale

This combination keeps the core contract both reviewable and automatable:

- Markdown captures semantics, invariants, and cross-step workflow rules.
- JSON Schema captures structural requirements that bindings and fixtures can validate directly.

## Target Runtimes

The v1 runtime targets are:

- Python
- R
- Julia

Other languages are explicitly deferred until the contract and fixtures are stable.

## Authority Rule

The written contract is authoritative over incidental Python behavior.

If implementation behavior differs from the written contract, later tracks must either:

1. change the implementation to match the contract, or
2. intentionally revise the contract in a tracked spec update.

## Backend And Interchange Rules

- Polars DataFrames are the Python in-memory reference model for tabular containers (`ValueArray` and `ParameterSet`) from contract version 3.0.0 (spec revision opened 2026-10-26, superseding the v1/v2 xarray reference-model rule and removing `ValueArray.dataset`, `ParameterSet.dataset`, `from_dataset`, and `to_dataset` outright at 3.0.0).
- NumPy is the reference execution baseline.
- JAX is optional acceleration, not public contract surface.
- Arrow/Parquet is the cross-language interchange boundary.

## Scope Boundary

The decision record is stable input for:

- canonical schemas
- conformance fixtures
- Python cleanup against spec

It should not be reinterpreted by later implementation tracks unless a spec revision is explicitly opened.
