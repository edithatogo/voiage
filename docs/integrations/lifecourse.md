# lifecourse Integration Strategy

## Purpose

This document defines how `voiage` should work with the separate
`edithatogo/lifecourse` microsimulation project.

The intended relationship is complementary:

- `lifecourse` generates disease-model, scenario, PSA, budget impact, equity,
  and policy-evaluation outputs.
- `voiage` consumes decision-analysis artifacts and computes EVPI, EVPPI, EVSI,
  ENBS, VOI plots, and advanced VOI analyses.

The two projects should interoperate through stable artifacts and conformance
fixtures, not by depending on each other's internal implementation details.

The corresponding `lifecourse` planning documents are:

- `/Users/doughnut/GitHub/lifecourse/conductor/tracks/voiage_integration_20260428/`
- `/Users/doughnut/GitHub/lifecourse/docs/voiage_integration_strategy.md`
- `/Users/doughnut/GitHub/lifecourse/docs/interchange/model_interchange_strategy.md`
- `/Users/doughnut/GitHub/lifecourse/heoml/`

The local `voiage` scaffold for the shared profile lives at:

- `specs/integrations/lifecourse/v1/`
- `conductor/tracks/lifecourse-integration-contract_20260429/`
- `specs/integrations/lifecourse/v1/fixtures/normative/screening_program_bundle.json`

The candidate shared artifact profile is HEOML, currently embedded in the
`lifecourse` repository while it is being proven against real run bundles and
`voiage` ingestion.
This profile should be treated as the shared portable run-bundle contract for
lifecourse-compatible VOI handoff, while `voiage` keeps its own result schemas
and method contracts.

## Ownership Boundary

`lifecourse` owns:

- simulation engines and cohorts
- genomic testing pathways
- scenario configuration
- parameter provenance
- calibration and validation targets
- PSA output generation
- reproducible run bundles

`voiage` owns:

- VOI algorithms
- VOI schemas and result contracts
- method diagnostics and maturity metadata
- VOI plotting
- VOI conformance fixtures
- cross-language VOI binding contracts

## Integration Contract

The preferred contract is a `lifecourse` VOI artifact profile that can be mapped
onto `voiage` core types:

- net-benefit matrix maps to `ValueArray`
- strategy names map to `ValueArray.strategy_names`
- parameter samples map to `ParameterSet`
- trial or study design metadata maps to `TrialDesign` where applicable
- EVPI, EVPPI, EVSI, and ENBS outputs map to existing `voiage` result contracts

The artifact profile should include schema version, producer package/version,
run identifier, scenario identifiers, PSA seed policy, willingness-to-pay
thresholds, population scaling metadata, time horizon, discount rate, method
settings, diagnostics, and warnings.
It should also carry HEOML profile provenance fields such as
`heoml_version`, `profile`, and a manifest identifier so downstream consumers
can validate the shared contract before loading artifacts.

## Interchange Formats

The contract should support JSON for metadata and small examples, CSV for simple
tabular compatibility, and Arrow or Parquet for larger tabular artifacts where
downstream tooling benefits from typed interchange.

The first conformance fixture should stay small and JSON/CSV-friendly. Arrow or
Parquet can be added once `lifecourse` produces larger run bundles.

HEOML should provide the broader run-bundle manifest and artifact conventions.
`voiage` should continue to own VOI-specific result schemas and method
contracts that HEOML can reference.

Pickle should not be accepted as a shared interchange format. It is
Python-specific, unsafe for untrusted files, fragile across versions, and not
usable by future non-Python bindings. If a pickle is ever emitted for local
debugging, it should be clearly marked non-portable and outside the compatibility
contract.

## Standards Considered

- ONNX, PMML, and PFA are machine-learning model interchange formats. They are
  useful references for model portability, but they do not naturally represent a
  PSA bundle, health-economic scenario set, or VOI result contract.
- PharmML is relevant to pharmacometrics model exchange but is not a direct fit
  for `lifecourse` genomic-testing policy scenarios or `voiage` VOI result
  payloads.
- OMOP CDM is useful for standardizing observational health data, not for
  exchanging executable health-economic models or VOI outputs.
- TreeAge-style feature mapping may be useful later as an ecosystem reference,
  but it should not become the primary contract between these two projects.

## Dependency Policy

`voiage` should not depend on `lifecourse`.

`lifecourse` should treat `voiage` as optional. If direct Python integration is
used, it should live behind an optional extra such as `lifecourse[voiage]` after
version compatibility is stable.

Because `voiage` includes advanced dependencies for JAX, NumPyro, scikit-learn,
statsmodels, plotting, and CLI workflows, `voiage` should consider whether a
lighter core install is useful for integration consumers:

- core: `ValueArray`, `ParameterSet`, EVPI, EVPPI, and NumPy methods
- jax: JAX and NumPyro acceleration
- metamodels: scikit-learn and statsmodels methods
- plotting: plotting dependencies
- cli: Typer and CLI-specific dependencies if separable
- dev: tests, docs, profiling, mutation, and release tooling

## Adapter Shape

A future adapter should be thin:

1. Load `lifecourse` VOI artifacts.
2. Validate schema/profile metadata.
3. Convert net benefits to `ValueArray`.
4. Convert parameter samples to `ParameterSet`.
5. Call stable public methods such as `evpi`, `evppi`, `evsi`, and `enbs`.
6. Return or write stable `voiage` result payloads with provenance.

The adapter should not import `lifecourse` simulation internals.
The local `load_heoml_run_bundle` helper in `voiage.ecosystem_integration`
implements the manifest-to-artifact half of this flow for the shared fixture
set.
It is exported from the top-level `voiage` package so consumers can load the
shared bundle without reaching into private modules.
It returns a `HeomlRunBundle` object containing the parsed manifest, a
`ValueArray`, a `ParameterSet`, and provenance metadata.

## Promotion Criteria

Treat the integration as experimental until:

- `lifecourse` and `voiage` agree on the v1 artifact profile.
- at least one compatibility fixture is shared by both repositories.
- EVPI and EVPPI parity is proven on a deterministic fixture.
- EVSI and ENBS fixtures include explicit method settings and assumptions.
- dependency extras and optional install behavior are documented.
- both projects record compatibility in changelogs or release notes.

## Recommended `voiage` Improvements

- Keep the core VOI artifact types and NumPy EVPI/EVPPI methods lightweight.
- Split heavier dependencies into extras where practical: JAX/NumPyro,
  metamodels, plotting, CLI, docs, and dev tooling.
- Provide stable result objects for EVPI, EVPPI, EVSI, and ENBS that include
  method settings, scaling metadata, diagnostics, package version, and
  provenance.
- Add an artifact loader that can consume the `lifecourse` profile without
  importing `lifecourse`.
- Add compatibility fixtures under `specs/integrations/lifecourse/v1/fixtures/`.
- Track HEOML compatibility and extraction status as the shared profile matures.

## Related Track

The executable plan for this work is:

`conductor/tracks/lifecourse-integration-contract_20260429/plan.md`
