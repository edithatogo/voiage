# HEOML `voiage` Extension Outline

This document reserves the VOI-specific extension surface that `voiage` should
expose within the HEOML health economics and outcomes research (HEOR) ecosystem
profile.

## Scope

The extension is for portable HEOR VOI handoff and VOI result metadata. It
should reference public `voiage` schemas and result contracts rather than
private implementation classes.

### Consumed Artifacts

- net-benefit matrices by strategy
- parameter samples aligned to PSA rows
- strategy names
- willingness-to-pay thresholds
- population scaling metadata
- trial or study-design metadata where applicable
- provenance and diagnostics from the producing bundle

### Produced Artifacts

- EVPI results
- EVPPI results
- EVSI results
- ENBS results
- method settings and approximation settings
- diagnostics, warnings, and uncertainty summaries
- plot artifact references when a workflow emits figures

### Optional Integrations

- `lifecourse`: health-economic run bundles and PSA handoff artifacts
- `innovate`: health-intervention diffusion and adoption uncertainty artifacts
- `mars`: optional surrogate/metamodel metadata for regression-style VOI

### Dependency Rules

- The base `voiage` install must not require sibling modules.
- Optional adapters belong behind extras or separate integration packages.
- Pickle is not part of the portable extension contract.

### Promotion Gates

- documented contract
- deterministic compatibility fixtures
- CI validation of the shared fixtures
- version compatibility matrix
- release notes for contract changes

This outline is intentionally minimal. The concrete schemas and fixtures should
be versioned in sibling contract directories as each integration matures.
