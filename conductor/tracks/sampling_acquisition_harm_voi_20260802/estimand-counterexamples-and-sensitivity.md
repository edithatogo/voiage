# Estimand, counterexamples and sensitivity contract

## Timing and probability space

For each design `d` in a declared feasible set `D`, including no sampling
`d0`, declare a joint law for:

- uncertain state `theta`;
- information outcome `Y_d`;
- acquisition-harm outcome `H_d`, including affected party and timing; and
- any availability, dropout or missingness caused by harm.

The sampling action occurs before the downstream policy uses `Y_d`. Harm may
occur before, during or after observation and may be statistically dependent
on `theta`, `Y_d`, missingness and design. The decision policy cannot condition
on information that is unavailable at its decision time.

Let `B(a, theta)` be downstream value in declared common cardinal units. The
ordinary gross sample information value is

```text
G(d) = E_Y[max_a E[B(a, theta) | Y_d]] - max_a E[B(a, theta)].
```

Let `C(d)` be ordinary research cost. If and only if a declared harm valuation
`L(H_d)` is separable from `B`, represents the chosen affected-party scope and
is in the same cardinal units, define

```text
NIV_H(d) = G(d) - C(d) - E[L(H_d)].
```

If any condition fails, `NIV_H` is undefined. Report `G`, `C`, harm
distributions and feasibility separately and solve the declared constrained or
vector problem. Do not insert an arbitrary exchange rate merely to produce a
number.

## Zero-harm reduction

For matched design, downstream decision model, feasibility and ordinary cost,

```text
P(H_d = zero) = 1  =>  NIV_H(d) = G(d) - C(d) = ENBS(d).
```

This identity is conditional. It does not equate #571 ENBS with a
sampling-harm method, and it does not remove a consent, ethics or regulatory
gate.

## Risk and catastrophe contract

Each design declares exactly one primary ordering rule and any hard
constraints. Supported research vocabulary is:

- expected commensurate loss;
- chance constraint `P(H_d in catastrophe) <= alpha`;
- lower-tail CVaR/expected shortfall under a declared loss convention;
- lexicographic rule that excludes any design with a prohibited outcome; or
- a separately reviewed domain criterion.

A catastrophic or absorbing outcome requires its own state definition,
probability, severity, reversibility and constraint. A finite expected penalty
cannot override a hard prohibition. Uncertain constraint satisfaction is
reported as uncertainty, not silently treated as feasibility.

## Enumerable counterexamples

| Case | Declared values | Required result |
|---|---|---|
| Zero harm | `G=12`, `C=4`, `E[L]=0` | `NIV_H=8`, exactly ordinary ENBS |
| Expected harm reverses commissioning | `G=10`, `C=2`, `E[L]=9` | `NIV_H=-1`; no sampling dominates |
| Positive scalar but prohibited catastrophe | `G=100`, `C=1`, catastrophe probability `0.001`, expected valued harm `0.1`, `alpha=0` | scalar candidate `98.9` is irrelevant; design infeasible |
| Rare absorbing harm | probability `0.0001` of an irreversible prohibited outcome | reject under lexicographic rule even with positive expected net value |
| Incommensurate affected parties | health-system benefit in currency; participant privacy harm on a noncardinal category | scalar `NIV_H` undefined; retain vector/constrained result |
| Correlated information and harm | identical marginal information/harm laws but harm concentrated in the most informative outcome | joint-law tail and conditional diagnostics differ; marginal subtraction is insufficient for nonseparable risk |
| Harm-induced missingness | severe harm causes dropout before `Y_d` is observed | recompute the information law; do not use the no-dropout EVSI |
| Safe low-information design | `G=4`, `C=1`, no harm versus unsafe `G=8`, `C=1` | selected design depends on declared risk constraints, not gross EVSI alone |

## Sensitivity requirements

Every future estimator packet varies, at minimum:

1. harm probability and severity independently;
2. catastrophe threshold and safety budget;
3. valuation or standardization assumptions, with an explicit undefined
   scalar region;
4. dependence between harm, information, state and missingness;
5. affected-party weights or constraints without hiding individual results;
6. expected-loss versus chance, CVaR and lexicographic criteria;
7. under-reporting/misclassification of harm;
8. ordinary research cost separately from acquisition harm;
9. no-sampling and alternative safer acquisition actions; and
10. sequential stopping or harm-budget updates, if adaptive sampling is later
    reviewed.

Sensitivity conclusions must report design switches, no-sampling regions,
constraint margins, expected/tail/catastrophic harm and uncertainty. Aggregate
rankings may not hide a failed hard constraint or an affected party.

## Assurance prerequisites

Before any executable claim, the exact finite reference suite must reproduce
all cases above and prove:

- probability normalization and timing/nonanticipativity;
- zero-harm ENBS reduction;
- identical results under joint-state relabelling;
- monotonicity only for a fixed separable valuation and fixed feasibility;
- no-sampling feasibility and deterministic complete ties;
- empirical bias, RMSE, coverage and constraint-calibration evidence for any
  simulation estimator; and
- failure on non-finite, unnormalized, missing-party, mixed-unit or undeclared
  catastrophe inputs.
