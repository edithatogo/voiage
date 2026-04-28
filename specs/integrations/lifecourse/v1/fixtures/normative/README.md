# lifecourse v1 Normative Fixtures

This directory holds the deterministic compatibility fixture(s) for the
`lifecourse` VOI profile.

The first bundle is intentionally small and reviewable:

- JSON metadata describing the run bundle and method settings
- CSV tables for net benefits and parameter samples
- JSON payloads for expected EVPI and EVPPI outputs

The values are chosen to make the contract self-consistent and easy to check in
CI without depending on `lifecourse` internals.
