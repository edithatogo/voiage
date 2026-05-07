# Track Implementation Plan: Canonical Schemas and Core Method Contracts

## Phase 1: Create the Stable Entity Schemas [checkpoint: ]

- [x] Task: Create the versioned schema layout under `specs/core-api/schemas/v1/`.
- [x] Task: Author the stable input/entity schemas for the v1 contract.
  - [x] decision problem identity and intervention metadata
  - [x] parameter-draw / uncertainty payloads
  - [x] net-benefit or value-array style core result inputs
- [x] Task: Create at least one canonical example document for each stable entity schema.

## Phase 2: Create the Core Method Result Contracts [checkpoint: ]

- [x] Task: Author the stable result schemas for EVPI, EVPPI, EVSI, and ENBS.
- [x] Task: If population VOI and sample-size optimization are in the approved stable scope, author their stable result schemas in the same versioned layout.
- [x] Task: For every stable method-result schema, record:
  - [x] required versus optional fields
  - [x] units and invariants
  - [x] nullable-field rules
  - [x] extension-field boundaries

## Phase 3: Validate and Version the Contract Set [checkpoint: ]

- [x] Task: Choose and document the minimal schema-validation path used by the repo; if a new validator dependency is required, update `conductor/tech-stack.md` before implementation.
- [x] Task: Add a narrow validation command or helper so each example document can be checked against its schema deterministically.
- [x] Task: Add a contract index document that maps every stable schema to its semantic description and example payload.

## Execution Notes

- Keep the stable contract language-neutral. Do not encode Python-specific container assumptions into the public layer.
- Leave experimental methods out of stable schemas unless the extension model explicitly allows them.
