# Core API Contract Index

This index maps each stable v1 schema to the canonical example payload that should be used by later fixture and binding tracks.

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

## Deferred Surfaces

Population VOI, structural VOI, network-meta-analysis VOI, adaptive design VOI, and portfolio optimization are deferred until a later spec revision promotes them into stable scope.
