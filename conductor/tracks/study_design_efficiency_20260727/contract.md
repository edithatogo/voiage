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
- a non-negative integer `sample_size`, expressed as total randomized or
  observed participants unless the study-model provenance says otherwise;
- finite, non-negative `evsi` and `research_cost` values in the common context;
- a `feasible` flag and zero or more stable feasibility reason codes;
- optional finite, non-negative `evsi_standard_error`, `cost_standard_error`
  and directly estimated `enbs_standard_error`, plus an optional ordered ENBS
  confidence interval;
- optional estimator provenance and allocation, duration, delay, uptake,
  guardrail, opportunity-cost or dependency metadata.

The runtime evaluates the supplied finite set only. It never interpolates or
extrapolates an unevaluated design. A range descriptor includes inclusive
lower and upper bounds plus an optional positive step and must agree with the
enumerated design set. Irregular and non-monotone sets remain valid. Duplicate
identifiers are rejected. Multiple scientifically distinct designs may share
a sample size; `design_id` preserves identity and boundary semantics use
distinct feasible sample sizes.

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

## COSS result envelope

`CossResultV1` is a versioned, finite-validated envelope with these required
components:

- `schema_version = "1.0"`, method identifier and the complete common context;
- an ordered `evaluated_designs` record for every supplied design;
- an `enumerated_feasible_set` containing the feasible sample sizes and an
  optional `declared_feasible_range` with inclusive lower/upper bounds;
- the selected `optimal_design_id` and `optimal_sample_size`, or both null when
  no design is feasible;
- signed `maximum_enbs`, null only when no design is feasible;
- the declared tie policy and absolute/relative tie tolerances;
- `tied_optimal_design_ids`, including the selected design;
- boundary state, diagnostics, estimator provenance and selection uncertainty;
  and
- a plotting-data view derived solely from the result records.

Each evaluated record contains `design_id`, `sample_size`, `evsi`,
`research_cost`, signed `enbs`, feasibility and reason codes, plus available
standard errors and confidence intervals. If the caller supplies independent
EVSI and cost standard errors but no covariance, ENBS uncertainty is not
silently synthesized; the result records `enbs_uncertainty_unavailable`.
Callers may instead supply an externally estimated ENBS standard error or
interval with its estimator provenance.

### Selection and ties

Selection first finds the greatest signed ENBS among feasible designs. A
design is tied when its ENBS differs from that maximum by no more than

`absolute_tolerance + relative_tolerance * max(abs(maximum_enbs), 1)`.

The default and stable policy is `smallest_sample_size`; ties are then resolved
by sample size and finally by `design_id` in Unicode code-point order. The
contract may also accept `largest_sample_size` and `first_declared`, which are
resolved deterministically. Unknown policies and negative or non-finite
tolerances are rejected. The full tied set is returned in caller order so the
choice is auditable.

### Boundary and feasible-set diagnostics

Boundary state is computed over distinct feasible sample sizes:

- `none` when no design is feasible;
- `both` when exactly one distinct feasible sample size exists;
- `lower` or `upper` when the selected sample size is respectively the
  smallest or largest feasible size; and
- `interior` otherwise.

The result reports gaps, infeasible records inside a declared range, range/set
disagreement and a non-monotone-EVSI diagnostic without altering the optimum.
A negative maximum ENBS remains a valid optimum among the enumerated designs;
whether to commission no study is a separate decision unless an explicit
zero-cost no-study design is included.

### Uncertainty around the optimum

Uncertainty is descriptive and never changes deterministic point-estimate
selection. `selection_uncertainty` records one of `unavailable`, `analytic`,
`monte_carlo`, `bootstrap` or `externally_supplied`, its replicate count when
applicable, an optional selection probability for each evaluated design, and
an optional confidence set of design identifiers. Probabilities must be finite
in `[0, 1]` and sum to one within the declared tolerance when complete. Missing
uncertainty is represented explicitly, not as zero uncertainty.

### Plotting-data contract

`CossPlotDataV1` contains caller-ordered vectors for design identifiers, sample
sizes, EVSI, research cost, signed ENBS, feasibility and available uncertainty
bounds, together with the selected design, tied set and boundary state. It has
no Matplotlib objects, colors, labels dependent on a backend, or rendering side
effects. Plotting adapters consume this data and must expose the zero-ENBS
reference, infeasible designs, uncertainty availability, ties and selected
optimum accessibly.

## EVSI/EVPI efficiency result

`InformationEfficiencyResultV1` is a small derived diagnostic, not an
estimator. Its inputs are finite `evsi` and `evpi` values plus the common
`StudyDesignContextV1`, an absolute tolerance and a relative tolerance. When
values arrive from separate envelopes, their decision-problem, unit,
population-scale, time-horizon and discounting fields must match exactly.

For positive EVPI, the raw dimensionless ratio is `evsi / evpi`. Bounds use

`bound_tolerance = absolute_tolerance + relative_tolerance * max(abs(evpi), 1)`

on the value scale. Inputs outside `[-bound_tolerance,
evpi + bound_tolerance]` are materially inconsistent with the theoretical
`0 <= EVSI <= EVPI` relationship and are rejected. Inputs just outside the
bounds but within tolerance retain their unclamped raw ratio and return
`below_zero_within_tolerance` or `above_one_within_tolerance`. Inputs inside
the bounds return `within_bounds`. This preserves Monte Carlo evidence instead
of silently changing it.

When `abs(evpi) <= bound_tolerance`, EVPI is treated as numerically zero:

- if `abs(evsi) <= bound_tolerance`, `ratio` is null and status is
  `undefined_zero_evpi`; and
- otherwise the inputs are rejected as `positive_evsi_with_zero_evpi` (or the
  corresponding negative inconsistency).

Materially negative EVPI is always rejected. Tolerances must be finite and
non-negative. The result returns `schema_version`, `evsi`, `evpi`, nullable
unclamped `ratio`, `status`, tolerances, common context and diagnostics. It may
also return a display percentage equal to `100 * ratio`, clearly labelled as a
presentation field.

This metric must be named information efficiency or EVSI/EVPI efficiency. It
must never be used for `total_voi / total_cost`, return on investment,
cost-effectiveness, ENBS, power, or the probability that a study changes the
decision. Those quantities require distinct names, units and contracts.

## Experiment-portfolio contract

`CossPortfolioCandidateV1` binds one governed COSS optimum to the portfolio
decision. It requires the primary and secondary metric identifiers, declared
guardrails and failures, heterogeneous-effect, delayed-effect, interference,
sequential-monitoring and multiplicity model identifiers, stopping rules,
study duration and unit, opportunity and implementation-delay costs, expected
policy change, dependencies, exclusions, and resource use. A declared
no-effect or fixed-horizon model is valid only when every model has an exact
`PortfolioModelAssuranceV1` disposition of `no_effect` or
`already_reflected_in_coss` plus non-empty provenance. Missing, duplicate,
extra, or silently ignored model declarations fail closed.

Additional opportunity and implementation-delay costs use
`PortfolioIncrementalCostV1`. Its literal exclusion declaration and non-empty
cost-basis provenance assert that these components are not already present in
the COSS research cost. A caller that cannot establish this disjointness must
rebuild its COSS curve from a non-overlapping cost breakdown rather than use
the portfolio allocator.

The exact allocator enumerates all admissible candidate subsets, admits the
empty portfolio, computes the fixed global maximum, constructs one tolerance
tie set against that maximum, and then applies lower-total-cost and
lexicographic-ID tie breaking. This prevents path-dependent tolerance drift.
It maximizes additive net signed ENBS subject to guardrails, dependencies,
exclusion groups and capacity. For each candidate:

- gross ENBS is gross EVSI minus research cost;
- net EVSI is gross EVSI minus opportunity and implementation-delay costs; and
- net ENBS is net EVSI minus research cost.

The versioned result returns every candidate evaluation, selected studies,
sample size and duration, per-study stopping rules and policy changes, gross
and net EVSI/ENBS totals, used and binding capacities, tolerances and
diagnostics. Relational fields are re-derived during deserialization; forged
totals, unknown resources, or false binding constraints fail closed.

The model assurances declare either that the model has no portfolio-level
effect or that its effect is already reflected in the governed COSS curve;
the allocator never silently estimates or ignores an unassured effect.
Domain-specific interference, multiplicity, delayed-effect or sequential
estimators and experiment-platform adapters may be added later only behind
this contract and with separate assurance.
