# Threshold, Tipping-Point, And Robust VOI Experimental Contract v1

This directory holds the planned frontier contract for threshold, tipping-point,
and robust VOI. It is not part of the stable core API v1 matrix yet;
promotion requires an implementation, deterministic fixtures, cross-language
validation, CLI coverage, and method maturity review.

## Files

- `schemas/threshold-set.schema.json` defines the threshold-scenario input
  metadata surface.
- `schemas/value-of-threshold-result.schema.json` defines the planned result
  shape.
- `examples/threshold-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-threshold.example.json` is a compact illustrative result
  payload.

## Shape

The planned analysis surface treats thresholds, reversals, and ambiguity as
explicit decision-relevant dimensions rather than post hoc sensitivity labels.
The intended net-benefit surface uses:

```text
sample x strategy x threshold_profile
```

The planned result should include:

- profile-specific optimal strategies
- expected net benefits by profile and strategy
- threshold-crossing probability summaries
- decision-reversal or tipping-point matrices
- robust strategy summaries under ambiguity weights
- Pareto or non-dominated strategy sets across threshold profiles

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare threshold scenarios side by side without
changing the surrounding frontier tooling.
