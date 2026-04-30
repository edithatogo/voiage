# Frontier VOI Contracts

This directory contains the experimental frontier VOI contract families.

## Layout

- `perspective/`: Value of Perspective contracts and fixtures.
- `distributional/`: distributional and equity-weighted VOI contracts and fixtures.
- `implementation/`: implementation-adjusted VOI contracts and fixtures.
- `preference/`: preference heterogeneity and individualized-care contracts.
- `validation/`: model-validation and discrepancy-reduction contracts.
- `threshold/`: threshold, tipping-point, and robust VOI contracts.
- `dynamic-real-options/`: dynamic real-options VOI contracts.
- `fixtures/`: registry for the fixture-backed frontier contract set.

The registry is the discovery layer for deterministic frontier fixtures. Each
family keeps its own schema, examples, and normative payloads under its v1
subtree.

The registry schema lives at `fixtures/manifest.schema.json`.

The validation entrypoint lives at `scripts/validate_frontier_contract.py`.
