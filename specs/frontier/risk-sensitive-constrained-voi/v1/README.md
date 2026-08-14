# Risk-sensitive constrained perfect-information contract v1

This experimental issue #570 contract compares two matched finite policy
problems: one policy chosen before the state is known and one state-contingent
policy mapping chosen after perfect state resolution. The probability law,
higher-is-better objective values or declared utilities, risk functional,
constraint units and limits, feasible-set rules, and information-cost placement
remain fixed across both problems.

The exact Python evaluator supports expected value, declared expected utility,
lower-tail CVaR/expected shortfall, and minimax regret. Operational constraints
use declared budget, capacity, eligibility, fairness, regulation, carbon,
liquidity, or service-level labels and either deterministic or chance
enforcement. These labels do not change the arithmetic: each constraint still
declares its own unit, sense, limit, statewise policy usage, provenance, and
required satisfaction probability.

The result retains the baseline and post-information policies, all declared-
tolerance objective ties, gross and signed net value, policy switches, risk
diagnostics, exact enumeration counts, and selected-policy constraint slacks.
The selected policy remains an exact objective optimum, with lexicographic
selection only between exact optima, so presentation tolerances cannot make
the gross value of perfect information negative. Constraint-removal
effects are reported as discrete diagnostic evidence. They are explicitly not
continuous local shadow prices or dual multipliers.

The normative fixture enumerates all 27 mappings of three policies over three
states. A deterministic budget ceiling and chance-constrained service level
make the unconstrained statewise choices infeasible, so the result demonstrates
the joint risk/constraint policy problem rather than relabelling an EVPI or
standalone CVaR helper.

Excluded from v1 are imperfect information, continuous/MIP solvers,
intertemporal or nonseparable resource constraints, endogenous feasible sets,
post-information risk-functional changes, stable maturity, and unverified
non-Python parity.
