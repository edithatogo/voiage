# Core API v1 Fixtures

This directory holds the versioned fixture set for the core API conformance track.

## Layout

- `normative/`: deterministic fixtures that define the required behavior surface.
- `illustrative/`: non-normative examples used for documentation and exploratory coverage.

The fixture manifest in this directory is validated by `scripts/validate_core_api_contract.py`.
Follow-up tasks in the conformance-fixture track will add provenance metadata, deterministic execution mode, and tolerance envelopes.
