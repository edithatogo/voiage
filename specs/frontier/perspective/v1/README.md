# Value of Perspective Experimental Contract v1

This directory holds the experimental Value of Perspective contract. It is not
part of the stable core API v1 matrix yet; promotion requires deterministic
fixtures, cross-language validation, CLI coverage, and method maturity review.

## Files

- `schemas/perspective-set.schema.json` defines ordered perspective metadata.
- `schemas/value-of-perspective-result.schema.json` defines the calculation
  result shape.
- `examples/value-of-perspective.example.json` is a compact illustrative result.
- `fixtures/` contains the deterministic screening-program conformance
  fixture set used to anchor the experimental CLI contract.

## Shape

The input net-benefit surface uses:

```text
sample x strategy x perspective
```

The regret matrix uses:

```text
row i, column j = regret in perspective i when using the strategy optimal under perspective j
```
