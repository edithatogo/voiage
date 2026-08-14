# Value of Perspective Fixture-Backed Contract v1

This directory holds the implemented Value of Perspective runtime surface and
its fixture-backed contract. It is not part of the stable core API v1 matrix
yet; promotion still requires cross-language validation and maturity review.

## Files

- `schemas/perspective-set.schema.json` defines ordered perspective metadata.
- `schemas/value-of-perspective-result.schema.json` defines the calculation
  result shape.
- `examples/value-of-perspective.example.json` is a compact illustrative result.
- `fixtures/` contains the deterministic screening-program conformance
  fixture set used to anchor the CLI contract.
- `fixtures/perspective-catalog.json` records the payer, societal, patient,
  provider, regulator, equity-weighted, and custom stakeholder definitions.
- `fixtures/evidence.json` records the fixture hashes, explicit real-data
  gate, parity state, and stable-promotion boundary.

## Shape

The input net-benefit surface uses:

```text
sample x strategy x perspective
```

The regret matrix uses:

```text
row i, column j = regret in perspective i when using the strategy optimal under perspective j
```

## Capability boundary and migration guidance

The v1 fixture-backed runtime currently implements the directional
current-information estimand (`directional_current_information_evop`) over a
finite sampled net-benefit surface. It deliberately does not claim the
following as implemented Value of Perspective estimands:

| Estimand or policy | Status | Migration path |
| --- | --- | --- |
| Directional current-information EVoP | Implemented, fixture-backed | Use `value_of_perspective` with an explicit reference perspective. |
| Perfect perspective information | Unsupported | Model the resolved perspective as an explicit sampled scenario and submit a new versioned fixture/contract before implementation. |
| Partial or sample perspective information | Unsupported | Use the existing EVPI/EVSI contracts only when their information model applies; do not relabel those results as EVoP. |
| Consensus, maximin, minimax-regret and Pareto summaries | Diagnostic outputs only | Treat them as diagnostics from the directional result, not separate stable estimands. |

Unsupported rows must remain fail-closed. Promotion requires independent
scientific review, cross-language shared-fixture evidence, and updated schema
and provenance records; documentation or mock data alone cannot promote them.
