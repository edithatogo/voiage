# Numerical Equivalence and Reproducibility

## Purpose

This document defines the numerical comparison rules, reproducibility metadata,
and provenance expectations that later core-api schema and fixture tracks rely
on.

The intent is to keep deterministic conformance checks exact where possible and
to make approximation explicit where exact equality is not scientifically
justified.

## Numerical Equivalence

Numerical equivalence is the rule set used by conformance fixtures, parity
checks, and downstream bindings when comparing results across runtimes.

The contract distinguishes three comparison classes:

- exact equivalence
- tolerance-based equivalence
- non-comparable results

## Tolerance Rules

### Exact Equivalence

Exact equivalence applies when the contract surface is structural or
categorical:

- schema names and versions
- method family labels
- maturity labels
- analysis identifiers
- booleans
- enum-like fields
- array and object keys where ordering is part of the published contract
- deterministic fixture artifacts that are explicitly declared as exact

Exact equivalence also applies when a result family is deterministic by
construction and the fixture manifest marks the case as `tolerance_policy:
"exact"`.

## Comparable Results

Comparable results are outputs that can be assessed under either exact
equivalence or tolerance-based conformance.

Comparable results must carry enough metadata for a downstream binding to
decide which comparison class applies.

### Tolerance-Based Conformance

Tolerance-based conformance applies to numeric outputs that are expected to vary
slightly across implementations while still remaining scientifically equivalent.

Unless a later schema or fixture manifest states otherwise, the default numeric
comparison rule is:

- absolute tolerance: `1e-10`
- relative tolerance: `1e-8`

Use tolerance-based conformance when:

- the value is derived from floating-point aggregation
- the output depends on backend-specific linear algebra
- the result is approximate by design, such as a surrogate or regression-based
  estimator
- the published fixture or binding contract declares a tolerance envelope

Approximate methods must declare that status explicitly in their result payload
or method metadata. They must not appear exact by omission.

## Non-Comparable Results

Results are non-comparable when a binding cannot meaningfully assert numeric
equivalence under the written contract.

Examples include:

- missing required input artifacts
- unsupported method families
- stochastic outputs without a deterministic fixture mode or declared seed
- backend-dependent paths where the backend is not part of the published
  contract
- outputs whose semantics are intentionally open for extension and therefore
  not yet fixed

Non-comparable results must be reported explicitly rather than silently coerced
into exact or approximate equality.

## Reproducibility and Provenance

Stable result families must carry reproducibility metadata that makes a result
auditable and replayable.

The minimum required metadata is:

- `seed`: the primary random seed used to generate the result, if applicable
- `execution_mode`: one of `deterministic` or `stochastic`
- `deterministic_fixture_mode`: a boolean indicating whether the result was
  generated from a committed fixture
- `provenance`: a structured object that can include package version, backend
  name, input artifact identifiers, and method settings

When a result is generated from a committed conformance fixture, the
`deterministic_fixture_mode` flag must be true and the provenance object must
include the fixture identifier or equivalent artifact reference.

When a result is stochastic or approximate, the provenance object must record
the seed or seed derivation strategy and enough method settings to reproduce
the execution path.

## Provenance Fragment

Later schemas and bindings should treat the following fragment as the minimum
shape for stable result families:

```json
{
  "seed": 12345,
  "execution_mode": "deterministic",
  "deterministic_fixture_mode": true,
  "provenance": {
    "fixture_id": "evpi-normative-001",
    "package_version": "0.0.0",
    "backend": "numpy",
    "method_family": "evpi"
  }
}
```

## Relationship To Diagnostics

Diagnostics are separate from reproducibility metadata.

- reproducibility explains how the result was produced
- diagnostics explain whether the result is trustworthy, degraded, or
  approximate

The next track phase defines the stable diagnostic payloads and capability
metadata that pair with the rules in this document.
