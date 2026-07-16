# Expert-Elicitation And Evidence-Synthesis Design Experimental Contract v1

This directory holds the fixture-backed frontier contract for expert-elicitation
VOI and evidence-synthesis design VOI. The Python runtime and CLI are
fixture-backed; promotion still requires open-data attribution, cross-language
validation, and method maturity review.

## Files

- `schemas/expert-synthesis-set.schema.json` defines the elicitation and
  synthesis input surface.
- `schemas/value-of-expert-synthesis-result.schema.json` defines the
  fixture-backed
  result shape.
- `examples/expert-synthesis-set.example.json` is a compact illustrative input
  payload.
- `examples/value-of-expert-synthesis.example.json` is a compact illustrative
  result payload.
- `fixtures/` contains the deterministic normative fixture set used to anchor
  the contract.

## Shape

The analysis surface treats elicitation design and evidence-synthesis
design as explicit decision-relevant dimensions rather than a hidden workflow
choice. The intended net-benefit surface uses:

```text
sample x strategy x expert_profile
```

The expected result should include:

- expert-profile-specific optimal strategies
- expected net benefits by profile and strategy
- elicitation-cost and synthesis-design-value summaries
- evidence-synthesis penalty or loss summaries
- robust or consensus strategy summaries under profile weights
- Pareto or non-dominated strategy sets across expert profiles

The contract is intentionally aligned with the Value of Perspective surface so
future implementations can compare elicitation scenarios side by side without
changing the surrounding frontier tooling.
