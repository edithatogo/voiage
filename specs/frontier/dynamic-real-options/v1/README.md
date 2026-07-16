# Dynamic Real-Options VOI Experimental Contract v1

This directory holds the fixture-backed experimental contract for dynamic
real-options VOI. It is not part of the stable core API v1 matrix yet;
promotion requires an implementation, deterministic fixtures,
cross-language validation, CLI coverage, and method maturity review.

## Files

- `schemas/dynamic-real-options-set.schema.json` defines the staged-decision
  input metadata surface.
- `schemas/value-of-dynamic-real-options-result.schema.json` defines the
  planned result shape.
- `examples/dynamic-real-options-set.example.json` is a compact illustrative
  input payload.
- `examples/value-of-dynamic-real-options.example.json` is a compact
  illustrative result payload.
- `fixtures/` contains the deterministic staged-evidence conformance fixture
  set used to anchor the planned contract.
- `fixtures/evidence.json` records implementation hashes and explicit
  longitudinal open-data and parity gates.

## Shape

The planned analysis surface treats delay, irreversibility, and policy
lock-in as explicit decision-relevant dimensions rather than as a hidden
sensitivity analysis. The intended net-benefit surface uses:

```text
sample x strategy x decision_stage
```

The expected result should include:

- stage-specific optimal strategies
- expected net benefits by stage and strategy
- waiting-value or option-value summaries
- policy-path or exercise-path comparisons
- timing-sensitivity summaries
- robust or consensus strategy summaries under stage weights
- Pareto or non-dominated strategy sets across decision stages

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare timing scenarios side by side without
changing the surrounding frontier tooling.
