# Implementation-Adjusted VOI Experimental Contract v1

This directory holds the experimental implementation-adjusted VOI contract. It
is not part of the stable core API v1 matrix yet; promotion requires
deterministic fixtures, cross-language validation, CLI coverage, and method
maturity review.

## Files

- `schemas/implementation-adjusted-result.schema.json` defines the experimental
  result shape.
- `examples/value-of-implementation.example.json` is a deterministic example
  result captured from the current Python implementation.
- `fixtures/` contains the deterministic screening-program fixture set used to
  anchor the contract.
- `fixtures/evidence.json` records deterministic hashes and the external
  open-data and cross-language parity gates.

## Shape

The current experimental result keeps implementation frictions explicit through
uptake, adherence, coverage, delay, uncertainty, and discounting, while
remaining 2D net-benefit first.
