# Core API v1 Fixtures

This directory holds the versioned fixture set for the core API conformance track.

## Layout

- `normative/inputs/`: deterministic input bundles consumed by the canonical fixture cases.
- `normative/`: deterministic fixtures that define the required behavior surface.
- `illustrative/`: non-normative examples used for documentation and exploratory coverage.
- `compatibility/`: language-neutral executable inputs and golden outcomes for
  normal, edge, and invalid public-API behavior.

The normative `manifest.json` is validated by
`scripts/validate_core_api_contract.py`. The executable
`compatibility-manifest.json` is validated and run by
`scripts/validate_core_api_fixtures.py`; external fixture roots may retain
layout-only support when they do not provide that executable manifest.
The runner contract and CI strategy are documented in `runner.md`.

Normative entries must already carry stable provenance metadata, deterministic execution mode,
and an input/output artifact pair. The validator rejects any manifest entry that does not. Future
work in this track is limited to expanding fixture breadth and, where appropriate, adding
tolerance envelopes for approximate outputs.
