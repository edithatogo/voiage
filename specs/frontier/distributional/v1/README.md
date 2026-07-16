# Distributional And Equity VOI Experimental Contract v1

This directory holds the experimental distributional and equity-weighted VOI
contract. It is not part of the stable core API v1 matrix yet; promotion
requires deterministic fixtures, cross-language validation, CLI coverage, and
method maturity review.

## Files

- `schemas/distributional-equity-result.schema.json` defines the experimental
  result shape.
- `examples/value-of-distributional-equity.example.json` is a deterministic
  example result captured from the current Python implementation.
- `fixtures/` contains the deterministic screening-program fixture set used to
  anchor the contract.
- `fixtures/evidence.json` records deterministic hashes and the external
  open-data and cross-language parity gates.

## Shape

The current experimental result summarizes subgroup-level expected net benefits
and equity-weighted welfare values. It intentionally stays aligned with the
existing Value of Heterogeneity surface while keeping equity weighting explicit.
