# Core API Spec Scaffold

This directory holds the written foundation for the language-agnostic `voiage` core API.

- `foundation.md` defines the purpose, audience, and scope of the contract.
- `decision-record.md` records the non-negotiable design choices that later schema and fixture work must follow.
- `numerical-equivalence.md` defines tolerance rules, reproducibility metadata, and provenance expectations for conformance checks.
- `contract-index.md` lists the versioned contract artifacts that are expected to stay in sync.
- `schemas/v1/` contains the normative schema definitions for the core API result types.
- `examples/v1/` contains matching example payloads used by the contract checks.
- `fixtures/v1/` contains the versioned conformance fixture set and its manifest.
- `scripts/validate_core_api_contract.py` and the associated tests enforce the published contract shape.

The intent is to keep the contract small, explicit, and stable enough for downstream schema authoring and conformance fixtures to proceed without re-litigating the core model.
