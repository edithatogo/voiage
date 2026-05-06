# lifecourse v1 Fixtures

This directory reserves the versioned fixture set for the
`lifecourse`-to-`voiage` compatibility contract.

## Layout

- `normative/`: deterministic fixtures that define the required behavior
  surface once both repositories agree on the profile.
- `illustrative/`: non-normative examples for documentation and exploratory
  coverage.

The first committed fixture should stay compact enough for CI, and it should
cover at least the EVPI/EVPPI path against a stable `lifecourse` artifact
profile. EVSI and ENBS fixtures can follow once study-design metadata is
stable.

The fixture manifest in this directory is intentionally a scaffold until the
shared profile is agreed with `lifecourse`. Future validation should enforce
deterministic provenance, method settings, and a portable interchange format.

The current manifest also records the compatibility anchors for the shared
contract: `voiage` `0.2.0`, `lifecourse` profile `v1`, and HEOML profile
`0.1`. Consumers should reject mismatched bundles before attempting artifact
conversion.

An illustrative result-envelope fixture is also included to document the shared
metadata that EVPI, EVPPI, EVSI, and ENBS payloads should preserve once the
direct integration path is stabilized.

Fixture validation should remain layered:

1. Confirm the manifest version and compatibility block.
2. Confirm the HEOML run-bundle metadata on the normative artifact.
3. Assert EVPI and EVPPI outputs exactly.
4. Check the illustrative result envelope structurally and for versioned
   metadata only.
