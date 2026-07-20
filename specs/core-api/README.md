# Core API Spec Scaffold

This directory holds the written foundation for the language-agnostic `voiage` core API.

The machine-readable normative v1.0 public-surface contract lives at
`../v1/stable-api.json`. The artifacts in this directory supply its schemas,
diagnostics, numerical-equivalence rules, and conformance-fixture foundation.

- `foundation.md` defines the purpose, audience, and scope of the contract.
- `decision-record.md` records the non-negotiable design choices that later schema and fixture work must follow.
- `numerical-equivalence.md` defines tolerance rules, reproducibility metadata, and provenance expectations for conformance checks.
- `diagnostics.md` defines the stable warning and diagnostic payloads used to report unsupported capabilities, degraded paths, and approximation caveats.
- `method-metadata.md` defines the stable capability, stability, and maturity metadata contract, including the explicit approximation-status rule.
- `extension-evolution.md` defines the versioning, deprecation, and additive-extension rules for the stable contracts and future namespaces.
- `contract-index.md` lists the versioned contract artifacts that are expected to stay in sync.
- `schemas/v1/` contains the normative schema definitions for the core API result types.
- `examples/v1/` contains matching example payloads used by the contract checks.
- `fixtures/v1/` contains the versioned conformance fixture set, its manifest, and the shared deterministic input bundles used by the normative cases.
- `fixtures/v1/runner.md` defines the language-neutral runner contract and CI strategy for future bindings.
- `scripts/validate_core_api_contract.py` and the associated tests enforce the published contract shape.
- `scripts/validate_core_api_fixtures.py` validates the fixture layout and
  executes the language-neutral compatibility catalog against the reference API.

The intent is to keep the contract small, explicit, and stable enough for downstream schema authoring and conformance fixtures to proceed without re-litigating the core model.
