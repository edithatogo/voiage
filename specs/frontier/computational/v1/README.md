# Computational And Model-Refinement Experimental Contract v1

This directory holds the fixture-backed frontier contract for computational VOI
and value of model refinement. The Python runtime and CLI are fixture-backed;
promotion still requires profiling evidence, cross-language validation, and
method maturity review.

## Files

- `schemas/computational-set.schema.json` defines the compute-budget input
  surface.
- `schemas/value-of-computational-result.schema.json` defines the fixture-backed
  result shape.
- `examples/computational-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-computational.example.json` is a compact illustrative
  result payload.
- `fixtures/` contains the deterministic normative fixture set used to anchor
  the contract.

## Shape

The analysis surface treats compute budget, approximation error, and
refinement cost as explicit decision-relevant dimensions rather than a hidden
implementation detail. The intended net-benefit surface uses:

```text
sample x strategy x compute_budget
```

The expected result should include:

- compute-budget-specific optimal strategies
- expected net benefits by budget and strategy
- approximation-error and refinement-value summaries
- compute-cost summaries
- robust or consensus strategy summaries under budget weights
- Pareto or non-dominated strategy sets across compute budgets

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare compute scenarios side by side without
changing the surrounding frontier tooling.
