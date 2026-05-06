# Frontier VOI Contracts

This directory contains the frontier VOI contract families with mixed maturity
labels.

## Layout

- `perspective/`: Value of Perspective contracts and fixtures.
- `distributional/`: distributional and equity-weighted VOI contracts and fixtures.
- `implementation/`: implementation-adjusted VOI contracts and fixtures.
- `preference/`: preference heterogeneity and individualized-care contracts and fixtures.
- `validation/`: model-validation and discrepancy-reduction contracts.
- `threshold/`: threshold, tipping-point, and robust VOI contracts.
- `dynamic-real-options/`: dynamic real-options VOI contracts.
- `causal-transportability/`: causal-identification, transportability, and external-validity contracts.
- `data-quality/`: data-quality, measurement-error, privacy, and linkage contracts.
- `computational/`: computational VOI and model-refinement contracts.
- `expert-synthesis/`: expert-elicitation and evidence-synthesis design contracts.
- `shared-maturity/`: maturity labels, diagnostics, and reporting conventions.
- `fixtures/`: registry for the fixture-backed frontier contract set.

The registry is the discovery layer for deterministic frontier fixtures and
family manifests. Each family keeps its own schema, examples, and normative
payloads under its v1 subtree.

The registry schema lives at `fixtures/manifest.schema.json`.

The validation entrypoint lives at `scripts/validate_frontier_contract.py`.
