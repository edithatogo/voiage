# Specification — Information-Source Portfolio VOI

## Objective

Implement the smallest scientifically coherent contract for issue #582: select
an ordered feasible sequence of information sources using decision value from a
declared finite joint world model, never independent EVSI scores or additive
knapsack values.

## Scope

The v1 experimental input declares actions, joint worlds, world probabilities,
action values, every source observation in every world, value/cost/time units,
rights and provenance, source cost/latency/privacy/freshness/SLA/coverage, order
constraints, and portfolio limits. Shared worlds encode arbitrary dependence,
redundancy and complementarity among source observations and decision payoffs.

The exact evaluator exhaustively enumerates bounded source sequences. For each
sequence it conditions on the joint observation tuple, re-optimizes the action,
and reports gross value, source cost, delay cost, net value, willingness to pay,
complete action ties and switches. The chosen sequence additionally reports
order-conditional marginal values and exact Shapley attribution of gross
decision value over its source set.

## Acceptance criteria

- **AC-01:** Reject un-cleared rights, incomplete joint observations,
  incomparable units, invalid probabilities, cyclic order constraints and
  infeasible declared limits.
- **AC-02:** Match independent exhaustive references for complementary,
  redundant and correlated finite sources, including complete ties.
- **AC-03:** Return all evaluated feasible sequences, the exact optimum and tie
  set, conditional marginal values, attribution, switches and exact-search
  diagnostics.
- **AC-04:** Provide strict v1 schemas, normative and pathology fixtures,
  deterministic JSON, CLI, public experimental discovery and user docs.
- **AC-05:** Keep the method experimental and record unsupported adaptive
  stopping, probabilistic observation channels, approximation, Rust, R, Julia
  and Mojo execution honestly.

## Explicit exclusions

This version does not infer dependence from marginal EVSI, accept predictive
Data Shapley as decision-source attribution, optimize adaptive stopping rules,
model stochastic acquisition failure outside the declared joint worlds, use an
approximate solver, or claim stable/polyglot support.
