# Adjacent Frontier Shared Maturity Contract v1

This directory defines the shared maturity and reporting contract for the
adjacent frontier VOI families:

- causal-identification, transportability, and external-validity
- data-quality, measurement-error, privacy, and linkage
- computational VOI and model-refinement
- expert-elicitation and evidence-synthesis design

## Shared Maturity Labels

- `planned`: contract defined, no deterministic fixtures yet
- `experimental`: implementation exists, but the family is still being
  validated against the contract surface
- `fixture-backed`: deterministic fixtures and exact contract expectations are
  committed
- `stable`: cross-language conformance, CLI coverage, and maturity review have
  all completed

## Shared Diagnostics

The adjacent families report family-specific diagnostics, but they should all
be able to expose a minimum shared set:

- profile or scenario counts
- strategy counts
- sample or decision-stage counts where relevant
- provenance or reproducibility hints

## Reporting Envelope

All adjacent families reuse the CHEERS-VOI reporting envelope. The shared
reporting object should carry:

- reporting standard
- analysis type
- method family
- method maturity
- analysis and decision identifiers when available
- decision context and population metadata when available
- estimator, seed, provenance, reproducibility, and diagnostics fields

## Fixture-Backed Criteria

An adjacent family becomes fixture-backed once it has:

- a versioned schema and example payload
- a deterministic normative fixture set
- validation coverage that checks the exact payloads
- a registry or track entry that points at the committed artifacts

## Next-Step Requirements By Family

- causal-transportability: deterministic fixtures, cross-language parity, and
  CLI coverage
- data-quality: deterministic fixtures, cross-language parity, and CLI
  coverage
- computational: deterministic fixtures, cross-language parity, and CLI
  coverage
- expert-synthesis: deterministic fixtures, cross-language parity, and CLI
  coverage

## Handoff Path

When a family reaches fixture-backed maturity, it should either:

- stay in the adjacent frontier track with a more specific follow-on phase, or
- hand back into the main frontier track if it becomes a general frontier
  capability rather than an adjacent extension

The remaining adjacent families that are still contract-scoping work remain so
until their runtime implementations are ready.
