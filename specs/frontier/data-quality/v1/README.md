# Data-Quality, Measurement-Error, Privacy, And Linkage Experimental Contract v1

This directory holds the fixture-backed frontier contract for data-quality,
measurement-error, data-acquisition, privacy, and linkage VOI. The Python
runtime and CLI are fixture-backed; promotion still requires open-data
attribution, cross-language validation, and method maturity review.

## Files

- `schemas/data-quality-set.schema.json` defines the data-quality and privacy
  input surface.
- `schemas/value-of-data-quality-result.schema.json` defines the fixture-backed
  result shape.
- `examples/data-quality-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-data-quality.example.json` is a compact illustrative
  result payload.
- `fixtures/` contains the deterministic normative fixture set used to anchor
  the contract.

## Shape

The analysis surface treats source quality, measurement error,
privacy, and linkage constraints as explicit decision-relevant dimensions
rather than a hidden sensitivity analysis. The intended net-benefit surface
uses:

```text
sample x strategy x data_quality_profile
```

The expected result should include:

- data-quality-profile-specific optimal strategies
- expected net benefits by profile and strategy
- acquisition-cost and privacy-constrained value summaries
- measurement-error and linkage-value summaries
- robust or consensus strategy summaries under profile weights
- Pareto or non-dominated strategy sets across data-quality profiles

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare data scenarios side by side without
changing the surrounding frontier tooling.
