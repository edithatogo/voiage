# Core API v1 Fixtures

This directory holds the versioned fixture set for the core API conformance track.

## Layout

- `normative/inputs/`: deterministic input bundles consumed by the canonical fixture cases.
- `normative/`: deterministic fixtures that define the required behavior surface.
- `illustrative/`: non-normative examples used for documentation and exploratory coverage.

The fixture manifest in this directory is validated by `scripts/validate_core_api_contract.py`.
The layout-only smoke check lives in `scripts/validate_core_api_fixtures.py`.
The runner contract and CI strategy are documented in `runner.md`.

Normative entries must already carry stable provenance metadata, deterministic execution mode,
and an input/output artifact pair. The validator rejects any manifest entry that does not. Future
work in this track is limited to expanding fixture breadth and, where appropriate, adding
tolerance envelopes for approximate outputs.
