# Belief-state sequential information value v1

This experimental contract evaluates an exact finite-horizon belief MDP. At
each stage the solver applies a control, propagates the latent state through
the declared action-dependent transition, selects a permitted sensor, observes
the declared signal, and performs Bayes updating. Rewards, sensing costs,
units, horizon, discounting, constraints, fixed stopping rule, policy class,
and global tie tolerances are explicit.

The matched no-information Bellman problem uses the same controls, rewards,
transitions, horizon, discounting, and constraints but never conditions a
future control on an observation. Gross and net information value are reported
separately. The result also reports the one-stage myopic value, full-horizon
nonmyopic value, conditional sensor values relative to the required zero-cost
null sensor, the full selected policy tree, value by horizon, and regret
against a fully observed finite-state policy.

The normative fixture is a counterexample to one-stage acquisition scoring.
Its diagnostic sensor is informative only after the costly `probe`
intervention. The one-stage information value is zero, while the two-stage
closed-loop policy is valuable because it probes and then adapts control to the
posterior. This exercises learning by intervention rather than relabelling the
existing `sequential_voi`, real-options, monitoring, or bandit helpers.

Assurance includes a posterior-martingale check, null-sensor and
no-information reductions, complete tolerance ties, deterministic
serialization, strict result-identity validation, a conservative 50,000-call
recursive-evaluator budget covering adaptive, no-information, myopic,
conditional-sensing and fully observed regret calculations, memoization of
repeated belief states, and exact zero-gap bounds. Every result carries the
strict input contract and its canonical SHA-256 commitment. A standalone
result validation therefore reconstructs the exact finite evaluation and binds
the expansion estimate, horizon curve, policy selections and complete ties,
and transition/learning diagnostics to the committed model instead of trusting
self-asserted flags. Action-dependent transition is reported separately. The
dual-control diagnostic is true only when an action changes a state-informative
observation law and the selected policy has a usable downstream control
response. Action-dependent marginal observation frequency without state
information, or transition dependence alone, cannot trigger that diagnostic.
The contract does not claim that dual control has one unique additive
numerical component.

Python is executable only at experimental maturity. Rust, R, and Julia are
unsupported, Mojo remains an external boundary, and scientific review, stable
promotion, release, and parent-programme closure remain pending. The issue
record cites the primary active-adaptive-management and belief-state/POMDP
sources used to govern the chronology and policy estimand.
