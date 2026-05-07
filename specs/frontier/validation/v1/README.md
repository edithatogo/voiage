# Model Validation VOI Fixture-Backed Contract v1

This directory holds the implemented model-validation VOI runtime surface and
its fixture-backed frontier contract. It is not part of the stable core API v1
matrix yet; promotion requires cross-language validation and maturity review.

## Files

- `schemas/validation-set.schema.json` defines the validation-profile input
  metadata surface.
- `schemas/value-of-model-validation-result.schema.json` defines the
  fixture-backed result shape.
- `examples/validation-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-model-validation.example.json` is a compact illustrative
  result payload.

## Shape

The fixture-backed analysis surface treats model validation as an explicit
decision-relevant dimension rather than a post hoc diagnostic. The intended
net-benefit surface uses:

```text
sample x strategy x validation_profile
```

The fixture-backed result includes:

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
