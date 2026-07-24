# Core API Contract Index

This index maps each stable v1 schema to the canonical example payload that should be used by later fixture and binding tracks.

The complete normative public-surface classification and runtime contract is
defined in `../v1/stable-api.json`. This index records the schema and example
artifacts that support that contract.

The cross-surface evolution rules are defined in
`../v1/compatibility-policy.json` and validated by
`../v1/compatibility-policy.schema.json`. Executable method cases are indexed
by `fixtures/v1/compatibility-manifest.json`.

## Entity Schemas

- `schemas/v1/intervention.schema.json` -> `examples/v1/intervention.example.json`
- `schemas/v1/decision-problem.schema.json` -> `examples/v1/decision-problem.example.json`
- `schemas/v1/trial-design.schema.json` -> `examples/v1/trial-design.example.json`
- `schemas/v1/parameter-set.schema.json` -> `examples/v1/parameter-set.example.json`
- `schemas/v1/value-array.schema.json` -> `examples/v1/value-array.example.json`

## Diagnostic Schema

- `schemas/v1/diagnostics.schema.json` -> `examples/v1/diagnostics.example.json`

## Method Metadata Schema

- `schemas/v1/method-metadata.schema.json` -> `examples/v1/method-metadata.example.json`

## Method Result Schemas

- `schemas/v1/results/evpi.schema.json` -> `examples/v1/evpi.example.json`
- `schemas/v1/results/evppi.schema.json` -> `examples/v1/evppi.example.json`
- `schemas/v1/results/evsi.schema.json` -> `examples/v1/evsi.example.json`
- `schemas/v1/results/enbs.schema.json` -> `examples/v1/enbs.example.json`
- `schemas/v1/results/ceac.schema.json` -> `examples/v1/ceac.example.json`
- `schemas/v1/results/ceaf.schema.json` -> `examples/v1/ceaf.example.json`
- `schemas/v1/results/dominance.schema.json` -> `examples/v1/dominance.example.json`
- `schemas/v1/results/expected-loss.schema.json` -> `examples/v1/expected-loss.example.json`

## Deferred Surfaces

Population VOI, structural VOI, network-meta-analysis VOI, adaptive design VOI, and portfolio optimization are deferred until a later spec revision promotes them into stable scope.

## Additive v2 candidates

These contracts are frozen candidates for v1.1 implementation and binding
conformance. They do not remove the stable v1 readers.

- `schemas/v2/decision-problem.schema.json` ->
  `examples/v2/decision-problem.example.json`
- `schemas/v2/analysis-spec.schema.json` ->
  `examples/v2/analysis-spec.example.json`
- `schemas/v2/perspective-result.schema.json` ->
  `examples/v2/perspective-result.example.json`
