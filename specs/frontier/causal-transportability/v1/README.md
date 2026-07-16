# Causal Identification, Transportability, And External-Validity Experimental Contract v1

This directory holds the fixture-backed frontier contract for
causal-identification, transportability, and external-validity VOI. The Python
runtime and CLI are fixture-backed; promotion still requires open-data
attribution, cross-language validation, and method maturity review.

## Files

- `schemas/causal-transportability-set.schema.json` defines the causal and
  transport-context input surface.
- `schemas/value-of-causal-transportability-result.schema.json` defines the
  fixture-backed result shape.
- `examples/causal-transportability-set.example.json` is a compact
  illustrative input payload.
- `examples/value-of-causal-transportability.example.json` is a compact
  illustrative result payload.
- `fixtures/` contains the deterministic normative fixture set used to anchor
  the contract.

## Shape

The analysis surface treats source-to-target population shifts as an
explicit decision-relevant dimension rather than a hidden modelling
assumption. The intended net-benefit surface uses:

```text
sample x strategy x transport_context
```

The expected result should include:

- source and target population identifiers
- transport weights and validity penalties
- causal-identification, transportability, and external-validity value
  summaries
- expected net benefits by target population and strategy
- target-specific optimal strategies
- robust or consensus strategy summaries under transport weights
- Pareto or non-dominated strategy sets across target populations

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare source and target populations side by side
without changing the surrounding frontier tooling.
