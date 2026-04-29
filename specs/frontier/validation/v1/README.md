# Model Validation VOI Experimental Contract v1

This directory holds the planned frontier contract for model-validation VOI.
It is not part of the stable core API v1 matrix yet; promotion requires an
implementation, deterministic fixtures, cross-language validation, CLI
coverage, and method maturity review.

## Files

- `schemas/validation-set.schema.json` defines the validation-profile input
  metadata surface.
- `schemas/value-of-model-validation-result.schema.json` defines the planned
  result shape.
- `examples/validation-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-model-validation.example.json` is a compact illustrative
  result payload.

## Shape

The planned analysis surface treats model validation as an explicit
decision-relevant dimension rather than a post hoc diagnostic. The intended
net-benefit surface uses:

```text
sample x strategy x validation_profile
```

The planned result should include:

- profile-specific optimal strategies
- expected net benefits by profile and strategy
- cross-profile discrepancy or regret matrix
- value of external validation
- value of model-discrepancy reduction
- consensus or robust strategy summaries under validation weights
- Pareto or non-dominated strategy sets across validation profiles

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare validation scenarios side by side without
changing the surrounding frontier tooling.
