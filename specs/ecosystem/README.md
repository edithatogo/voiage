# Ecosystem Contracts

This directory is reserved for health economics and outcomes research (HEOR)
ecosystem-level contracts that connect `voiage` to sibling projects without
importing their internals.

Initial contract targets:

- `lifecourse`: HEOML-compatible health-economic run bundles and VOI handoff
  artifacts.
- `innovate`: health-intervention adoption, implementation diffusion, and
  policy-spread uncertainty artifacts that can feed HEOR VOI workflows.
- `mars`: surrogate/metamodel artifact references and backend metadata for
  optional regression-based VOI workflows.
- HEOML: the shared portable artifact profile and extension namespace model.

Stable contracts should define:

- producer and consumer responsibilities
- schema versions
- required and optional artifacts
- compatibility fixtures
- dependency and optional-extra policy
- diagnostics and provenance fields
- deprecation and migration rules

Current scaffolds:

- [voiage-extension.md](./voiage-extension.md) for the HEOML `voiage`
  extension outline
- [fixtures/README.md](./fixtures/README.md) for planned ecosystem fixture
  families
- [fixtures/manifest.json](./fixtures/manifest.json) for the scaffolded
  versioned manifest
- [process/README.md](./process/README.md) for the HEOR process-mining outline
  and PM4Py ecosystem-only contract
