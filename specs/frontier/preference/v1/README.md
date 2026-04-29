# Preference Heterogeneity And Individualized Care Experimental Contract v1

This directory holds the planned frontier contract for preference heterogeneity
and value of individualized care. It is not part of the stable core API v1
matrix yet; promotion requires an implementation, deterministic fixtures,
cross-language validation, CLI coverage, and method maturity review.

## Files

- `schemas/preference-set.schema.json` defines the preference-profile input
  metadata surface.
- `schemas/value-of-preference-result.schema.json` defines the planned result
  shape.
- `examples/preference-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-preference.example.json` is a compact illustrative result
  payload.

## Shape

The planned analysis surface treats preference heterogeneity as an explicit
dimension rather than a hidden modelling assumption. The intended net-benefit
surface uses:

```text
sample x strategy x preference_profile
```

The planned result should include:

- profile-specific optimal strategies
- expected net benefits by profile and strategy
- cross-profile regret matrix
- value of switching preference profile
- value of individualized care
- consensus or robust strategy summaries under profile weights
- Pareto or non-dominated strategy sets across preference profiles

The contract is intentionally aligned with the Value of Perspective surface so
that future implementations can compare multiple stakeholder preference profiles
side by side without changing the surrounding frontier tooling.
