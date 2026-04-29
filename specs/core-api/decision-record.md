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

- Xarray-labeled arrays are the Python in-memory reference model.
- NumPy is the reference execution baseline.
- JAX is optional acceleration, not public contract surface.
- Arrow/Parquet is the cross-language interchange boundary.
- Polars is adapter-only tabular tooling, not the canonical computational model.

## Scope Boundary

The decision record is stable input for:

- canonical schemas
- conformance fixtures
- Python cleanup against spec

It should not be reinterpreted by later implementation tracks unless a spec revision is explicitly opened.
