# Value of Perspective Fixture-Backed Contract v1

This directory holds the implemented Value of Perspective runtime surface and
its fixture-backed contract. It is not part of the stable core API v1 matrix
yet; promotion still requires cross-language validation and maturity review.

## Files

- `schemas/perspective-set.schema.json` defines ordered perspective metadata.
- `schemas/value-of-perspective-result.schema.json` defines the calculation
  result shape.
- `examples/value-of-perspective.example.json` is a compact illustrative result.
- `fixtures/` contains the deterministic screening-program conformance
  fixture set used to anchor the CLI contract.

## Shape

The input net-benefit surface uses:

```text
sample x strategy x perspective
```

The regret matrix uses:

```text
row i, column j = regret in perspective i when using the strategy optimal under perspective j
```
