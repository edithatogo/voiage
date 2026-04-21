# Core API v1 Fixtures

This directory holds the versioned fixture set for the core API conformance track.

## Layout

- `normative/`: deterministic fixtures that define the required behavior surface.
- `illustrative/`: non-normative examples used for documentation and exploratory coverage.

The fixture manifest in this directory is validated by `scripts/validate_core_api_contract.py`.
Normative entries must already carry stable provenance metadata and deterministic execution mode, and the validator rejects any manifest entry that does not. Future work in this track is limited to expanding fixture breadth and, where appropriate, adding tolerance envelopes for approximate outputs.
