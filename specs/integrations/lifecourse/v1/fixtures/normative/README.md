# lifecourse v1 Normative Fixtures

This directory holds the deterministic compatibility fixture(s) for the
`lifecourse` VOI profile.

The first bundle is intentionally small and reviewable:

- JSON metadata describing the run bundle and method settings
- CSV tables for net benefits and parameter samples
- JSON payloads for expected EVPI and EVPPI outputs

The values are chosen to make the contract self-consistent and easy to check in
CI without depending on `lifecourse` internals.

This deterministic bundle is versioned against the same compatibility anchors as
the rest of the scaffold:

- `voiage` `0.2.0`
- `lifecourse` profile `v1`
- HEOML profile `0.1`

The fixture should be validated exactly, including the manifest compatibility
block and the expected EVPI/EVPPI outputs, before any adapter or release notes
are updated.
