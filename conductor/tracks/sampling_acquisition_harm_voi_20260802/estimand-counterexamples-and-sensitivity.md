# Estimand, counterexamples and sensitivity contract

## Timing and probability space

For each design `d` in a declared candidate set `D`, including an explicit no-
sampling comparator `d0`, declare design-indexed potential outcomes and a joint
law for:

- uncertain state `theta`;
- decision-time observable history `O_d`, including any observed information,
  harm and dropout indicators;
- acquisition-harm outcome `H_d`, including affected party and timing; and
- any availability, dropout or missingness caused by harm.

The sampling action occurs before the downstream policy uses `O_d`. Harm may
occur before, during or after observation and may be statistically dependent
on state, information, missingness and design. The policy must be measurable
with respect to the declared time filtration. Distinguish acquisition harm,
sampling-induced downstream change, downstream decision harm, interference and
spillovers. Model latent harm, reporting, misclassification and informative
dropout; without validation data or defensible restrictions, return
`not_identified` or partial-identification bounds.

Let `B_down,d(a, theta)` be downstream value in declared common cardinal units,
excluding every acquisition-harm consequence assigned to `L_d`, and let
`A_d(O_d)` be the admissible action correspondence. Define

```text
W_B(d) = sup over O_d-measurable pi_d with pi_d(O_d) in A_d(O_d) almost surely
         E[B_down,d(pi_d(O_d), theta_d)]
G(d; d0) = W_B(d) - W_B(d0).
```

`G` equals ordinary EVSI only when sampling changes the information available
to the policy and leaves the state, action set and outcome mapping unchanged.
Let `Delta C(d; d0) = C(d) - C(d0)`. If and only if a declared harm valuation
`L_d(H_d)` is policy-independent, additively separable from `B`, represents the
chosen party scope and is in the same cardinal units, define

```text
NIV_H(d; d0) = G(d; d0) - Delta C(d; d0)
                - {E[L_d(H_d)] - E[L_d0(H_d0)]}.
```

If any condition fails, `NIV_H` is undefined. Report `G`, `C`, harm
distributions and feasibility separately and solve the declared constrained or
vector problem. Do not insert an arbitrary exchange rate merely to produce a
number.

The candidate supplies an outcome-component ledger that assigns every health,
action, cost and other valued consequence exactly once to `B_down,d`, `C(d)`
or `L_d`. If harm changes state or available actions, or the components cannot
be partitioned without changing the policy problem, define total joint welfare
once as `J_d(a, theta_d, H_d)` and optimize its increment net of ordinary cost
and subject to harm constraints. Do not subtract `L_d` again. If the supremum
is not attained, report the supremum without a selected policy.

## Zero-harm reduction

For matched state/action/outcome model, feasibility and ordinary incremental
cost,

```text
P(H_d = H_d0 = zero) = 1
  => NIV_H(d; d0) = G(d; d0) - Delta C(d; d0) = ENBS(d; d0).
```

This identity is conditional. It does not equate #571 ENBS with a
sampling-harm method, and it does not remove a consent, ethics or regulatory
gate.

## Risk and catastrophe contract

Each design declares exactly one primary ordering rule and any hard
constraints. Supported research vocabulary is:

- expected commensurate loss;
- chance constraint `P(H_d in catastrophe) <= alpha`;
- upper-tail CVaR/expected shortfall for positive loss, using
  `inf_eta {eta + E[(L-eta)+]/(1-q)}` for `0 <= q < 1`;
- lower-tail CVaR/expected shortfall for signed welfare, using
  `sup_eta {eta - E[(eta-W)+]/beta}` for `0 < beta <= 1`;
- lexicographic rule that excludes any design with a prohibited outcome; or
- a separately reviewed domain criterion.

A catastrophic or absorbing outcome requires party, horizon, union/competing-
risk convention, state definition, probability, severity and reversibility.
The optimization forms above determine behavior at atoms; sign, confidence
level, quantile and interpolation conventions remain explicit metadata. A
finite expected penalty cannot override a hard prohibition. Feasibility is
`feasible` only when the declared upper bound or corresponding posterior/robust
worst case is at or below the threshold, `infeasible` only when the declared
lower bound is above it, and `indeterminate` otherwise. Equality is feasible
under this closed constraint unless the candidate states another reviewed
convention. Apply a declared familywise or simultaneous method across searched
designs and constraints. `not_identified` maps to `indeterminate` unless valid
partial-identification bounds lie wholly on one side of the threshold.
Separate aleatory harm uncertainty from epistemic estimator uncertainty. An
`alpha=0` constraint requires structural exclusion or a logically sufficient
bound; zero observed catastrophes alone cannot establish it.

## Enumerable counterexamples

| Case | Declared values | Required result |
|---|---|---|
| Nonzero baseline comparator | `W_B(d)=20`, `W_B(d0)=12`, `C(d)=5`, `C(d0)=1`, `E[L_d]=3`, `E[L_d0]=1` | `G=8`, incremental cost `4`, incremental harm `2`, so `NIV_H=2` |
| Zero harm | `G=12`, incremental cost `4`, both harms zero | `NIV_H=8`, exactly ordinary ENBS |
| Expected harm reverses commissioning | relative to `d0`, `G=10`, incremental cost `2`, incremental valued harm `9`, and `NIV_H(d0;d0)=0` | `NIV_H=-1`; no sampling dominates |
| Positive scalar but prohibited catastrophe | `G=100`, `C=1`, catastrophe probability `0.001`, expected valued harm `0.1`, `alpha=0` | scalar candidate `98.9` is irrelevant; design infeasible |
| Rare absorbing harm | probability `0.0001` of an irreversible prohibited outcome | reject under lexicographic rule even with positive expected net value |
| Incommensurate affected parties | health-system benefit in currency; participant privacy harm on a noncardinal category | scalar `NIV_H` undefined; retain vector/constrained result |
| Correlated information and harm | design A has two equiprobable states `(information gain, loss)=(8,8),(0,0)`; design B has `(8,0),(0,8)` | both marginals and expected net match; net is always `0` for A and is `+8/-8` for B, so lower-tail welfare ES at mass `0.5` is `0` for A and `-8` for B |
| Harm-induced missingness | severe harm causes dropout before `Y_d` is observed | recompute the information law; do not use the no-dropout EVSI |
| Safe low-information design | `G=4`, `C=1`, no harm versus unsafe `G=8`, `C=1` | selected design depends on declared risk constraints, not gross EVSI alone |
| Positive-loss upper tail | loss is `0` with probability `0.99` and `100` with probability `0.01`; compare loss `200` at probability `0.02` | upper-tail CVaR cannot improve when adverse mass and severity increase; a lower-tail loss convention would give the unsafe direction |
| Partial identification | latent harm prevalence is observed only through an unknown under-reporting rate | return `not_identified` or declared bounds, never a point feasibility claim |
| Downstream injury without double counting | sampling injury reduces downstream health by `5` and removes an action | additive `NIV_H` is undefined; represent the health/action consequence once in `J_d` and retain the separate harm constraint |

## Sensitivity requirements

Every future estimator packet varies, at minimum:

1. harm probability and severity independently;
2. catastrophe threshold and safety budget;
3. valuation or standardization assumptions, with an explicit undefined
   scalar region;
4. dependence between harm, information, state and missingness;
5. affected-party distributions, weights, transfers or constraints without
   hiding individual or subgroup results;
6. expected-loss versus chance, CVaR and lexicographic criteria;
7. under-reporting/misclassification of harm;
8. ordinary research cost separately from acquisition harm;
9. no-sampling and alternative safer acquisition actions; and
10. population scaling, perspective, horizon and discount convention aligned
    to #571; and
11. sequential stopping or harm-budget updates, if adaptive sampling is later
    reviewed.

Sensitivity conclusions must report design switches, no-sampling regions,
constraint margins, expected/tail/catastrophic harm and uncertainty. Aggregate
rankings may not hide a failed hard constraint or an affected party.

## Assurance prerequisites

Before any executable claim, the exact finite reference suite must reproduce
all cases above and prove:

- probability normalization and timing/nonanticipativity;
- explicit `d0` increment identities and invariance to common baseline shifts;
- zero-harm ENBS reduction;
- `0 <= G <= EVPI` for a matched ordinary decision problem, Blackwell
  monotonicity under its assumptions, and action/state permutation invariance;
- identical results under joint-state relabelling;
- monotonicity only for a fixed separable valuation and fixed feasibility;
- evaluated no-sampling status, deterministic complete ties, nondominated sets
  and nullable selection when no reviewed ordering exists;
- empirical bias, RMSE, coverage and constraint-calibration evidence for any
  simulation estimator, including rare-event assurance, one-sided feasibility
  bounds, equality/multiple-constraint handling and indeterminate cases;
- a truth-known replicate unit and predeclared thresholds for false-safe and
  false-infeasible classification, familywise error after design/constraint
  search, tail effective sample size, interval and CVaR coverage, selection-
  induced bias, convergence, seed/replay digest and deterministic replay; and
- failure on non-finite, unnormalized, missing-party, mixed-unit or undeclared
  catastrophe inputs.
