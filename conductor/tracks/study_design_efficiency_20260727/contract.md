# Frozen runtime contract — planned v1.2.0

This document is normative for the experimental implementation governed by
issue #571. Stable promotion remains subject to scientific review.

## Common study-design context

Every COSS curve and information-efficiency calculation declares one immutable
`StudyDesignContextV1` shared by every value and cost in the result:

| Field | Contract |
| --- | --- |
| `decision_problem_id` | Non-empty identifier for the alternatives, current evidence and decision rule. |
| `value_unit` | Non-empty unit label. Currency includes price year and currency, for example `AUD_2026`; non-monetary utility uses its explicit unit. |
| `population_scale` | Strictly positive multiplier represented by every EVPI, EVSI and cost value. |
| `time_horizon` | Non-empty label for the affected population/time window. |
| `discounting_id` | Non-empty identifier for the discounting convention, including `none` when undiscounted. |
| `study_model_id` | Non-empty identifier for the sampling/design model used to estimate EVSI. |
| `cost_model_id` | Non-empty identifier for the research and opportunity-cost model. |
| `random_seed` | Optional integer. Its absence is recorded and may not be described as reproducible stochastic estimation. |

Values are commensurate only when the decision problem, value unit, population
scale, time horizon and discounting identifier are identical. Conversion is an
upstream responsibility and must be recorded before this contract is invoked.
No implicit currency, population or horizon conversion is permitted.

## Evaluated designs

A `StudyDesignPointInputV1` is one declared design in caller order and contains:

- a unique, non-empty `design_id`;
- a finite, non-negative `sample_size`, expressed as total randomized or
  observed participants unless the study-model provenance says otherwise;
- finite, non-negative `evsi` and `research_cost` values in the common context;
- a `feasible` flag and zero or more stable feasibility reason codes;
- optional finite, non-negative `evsi_standard_error` and
  `cost_standard_error`; and
- optional estimator provenance and allocation, duration, delay, uptake,
  guardrail, opportunity-cost or dependency metadata.

The runtime evaluates the supplied finite set only. It never interpolates or
extrapolates an unevaluated design. A range descriptor is descriptive and must
agree with the enumerated design set; irregular and non-monotone sets remain
valid. Duplicate identifiers or sample sizes are rejected because tie and
boundary semantics would otherwise be ambiguous.

Research cost includes every cost the caller chooses to charge to the study,
including opportunity cost when applicable. The cost-model identifier and
provenance make that choice auditable; the runtime does not invent omitted
costs. Signed net benefit is always `evsi - research_cost` and is never floored
at zero.

## Feasibility, ordering and reproducibility

Only records with `feasible = true` participate in optimum selection. Every
declared record remains in the returned curve, including infeasible records.
When no design is feasible, the result has no selected optimum and carries the
`no_feasible_design` diagnostic rather than selecting a sentinel design.

Returned records preserve caller order. Boundary diagnostics use the smallest
and largest feasible sample sizes, not array position. Identical inputs,
tolerances and seed must produce byte-equivalent canonical JSON apart from
explicitly excluded timing fields. Estimator provenance must identify whether
uncertainty is unavailable, analytic, Monte Carlo, bootstrap or externally
supplied.
