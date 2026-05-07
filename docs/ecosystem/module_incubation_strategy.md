# Ecosystem Module Incubation Strategy

## Purpose

`voiage` should be the Value of Information analysis engine in a health
economics and outcomes research (HEOR) ecosystem. It should not own upstream
disease simulation, health-intervention diffusion, evidence registries, or
workflow orchestration. Those concerns should remain in sibling modules and
integrate through artifacts, schemas, adapters, and conformance fixtures.

The ecosystem scope is HEOR: cost-effectiveness analysis, HTA, reimbursement,
implementation uncertainty, evidence synthesis, outcomes modelling, and
health-policy evaluation. Generic cross-domain modelling is not the gap this
ecosystem is intended to fill.

## Current Ecosystem Roles

- `voiage`: EVPI, EVPPI, EVSI, ENBS, VOI study design, VOI metamodel workflows,
  diagnostics, and VOI result contracts.
- `lifecourse`: health-economic model runs, PSA outputs, decision-problem
  metadata, scenario configuration, and reporting bundles.
- `innovate`: health-intervention uptake, implementation diffusion, policy
  adoption, implementation spread, and network diffusion traces.
- `mars`: fixed-API surrogate/metamodel package that can be an optional backend
  for regression-style VOI approximations.
- HEOML: portable health-economic model and output artifact profile, including
  extension namespaces for VOI and sibling modules.

The corresponding scaffolded contracts live at:

- [specs/ecosystem/README.md](../../specs/ecosystem/README.md)
- [specs/ecosystem/voiage-extension.md](../../specs/ecosystem/voiage-extension.md)
- [specs/ecosystem/fixtures/README.md](../../specs/ecosystem/fixtures/README.md)
- [specs/ecosystem/fixtures/manifest.json](../../specs/ecosystem/fixtures/manifest.json)

## Candidate Ecosystem Integrations

- `lifecourse` adapter: consume HEOML-compatible net-benefit and parameter
  artifacts and produce VOI result artifacts.
- `innovate` adapter: consume health-intervention adoption or diffusion
  uncertainty artifacts where implementation uptake is a decision-relevant
  uncertainty source.
- `mars` metamodel backend: use MARS-style response surfaces for EVPPI, EVSI,
  calibration VOI, or sensitivity workflows without modifying the `mars` core
  API.
- HEOML `voiage` extension: define VOI handoff and VOI result metadata that can
  be consumed without importing `voiage`.
- Future workflow/reporting tools: consume VOI result bundles through portable
  HEOR artifacts, not Python object graphs.

## Dependency Policy

- Keep the base `voiage` install independent of `lifecourse`, `innovate`, and
  `mars`.
- Add ecosystem integrations through optional extras only after stable public
  APIs and fixture contracts exist.
- Require smoke CI, Renovate coverage, security checks, documentation, and a
  removal path for each optional adapter dependency.
- Do not require changes to the `mars` core API. If VOI-specific behavior is
  needed, add it in a `voiage` adapter or a future companion package.
- Keep pickle outside the portable interchange contract.

## HEOML Alignment

`voiage` should reserve a HEOML extension namespace for:

- VOI handoff inputs: net-benefit matrices, parameter samples, strategy names,
  WTP thresholds, population scaling, trial designs, and provenance.
- VOI outputs: EVPI, EVPPI, EVSI, ENBS, method settings, diagnostics,
  uncertainty summaries, and plot artifact references.
- Metamodel metadata: backend name, training inputs, validation metrics,
  convergence diagnostics, and reproducibility settings.

The HEOML extension should reference `voiage` public result contracts and
versioned schemas, not private implementation classes.

## Promotion Criteria

1. Document the artifact contract and dependency policy.
2. Add deterministic compatibility fixtures.
3. Add experimental adapters behind optional extras.
4. Add CI smoke checks and fixture validation across repositories.
5. Promote only after version compatibility, docs, release notes, and
   deprecation policy are clear.

## Immediate Follow-Up

- Finish the existing `lifecourse` integration contract.
- Add HEOML-compatible VOI result schema outlines.
- Define an `innovate` adoption-uncertainty fixture.
- Benchmark whether `mars` adds value as an optional EVPPI/EVSI metamodel
  backend before exposing it as a supported option.
- HEOR naming brainstorm: shortlist `calibrate`, `evidence`, `process`,
  `report`, `registry`, `workflow`, `quality`, `engines`, and `heoml`; keep
  PM4Py in the ecosystem-only process-mining bucket; require a CLI surface for
  every future module and an explicit MCP decision where orchestration matters.
