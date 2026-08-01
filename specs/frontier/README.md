# Frontier VOI Contracts

This directory contains the frontier VOI contract families with mixed maturity
labels, plus the shared maturity conventions and deterministic fixture
registry used to keep them aligned.

## Layout

- `perspective/`: Value of Perspective contracts and fixtures.
- `distributional/`: distributional and equity-weighted VOI contracts and fixtures.
- `implementation/`: implementation-adjusted VOI contracts and fixtures.
- `preference/`: preference heterogeneity and individualized-care contracts and fixtures.
- `value-of-flexibility/`: experimental flexible-versus-commitment option-value contract and fixtures.
- `value-of-distributional-information/`: experimental model-family-index
  information-value contract and exact fixtures; distinct from distributional equity.
- `qualitative-information/`: experimental non-cardinal assessment, audit and
  accessible-rendering contracts for ordinal information priorities, dissent,
  missingness, redaction, AI provenance and accountable human review.
- `validation/`: model-validation and discrepancy-reduction contracts.
- `threshold/`: threshold, tipping-point, and robust VOI contracts.
- `dynamic-real-options/`: dynamic real-options VOI contracts.
- `causal-transportability/`: causal-identification, transportability, and external-validity contracts.
- `data-quality/`: data-quality, measurement-error, privacy, and linkage contracts.
- `computational/`: computational VOI and model-refinement contracts.
- `expert-synthesis/`: expert-elicitation and evidence-synthesis design contracts.
- `expected-utility-information-pricing/`: experimental utility-equivalent
  information prices and the delegating VoC presentation.
- `shared-maturity/`: maturity labels, diagnostics, and reporting conventions.
- `study-design-efficiency/`: experimental COSS, EVSI/EVPI and binding fixture evidence.
- `fixtures/`: registry for deterministic frontier fixtures and family manifests.

The registry is the discovery layer for deterministic frontier fixtures and
family manifests. Each family keeps its own schema, examples, and normative
payloads under its v1 subtree, and the shared-maturity family documents the
review labels and diagnostics that keep the maturity transitions explicit.

The registry schema lives at `fixtures/manifest.schema.json`. It supports the
existing split input/output manifests and explicit bundled request/result
fixtures with verified SHA-256 digests.

The validation entrypoint lives at `scripts/validate_frontier_contract.py`.
