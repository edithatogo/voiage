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
