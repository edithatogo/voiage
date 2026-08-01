# Uncertainty-modelling value v1 (experimental)

This exact finite contract compares a deterministic solution obtained from one
declared point-estimate functional with a nonanticipative stochastic-policy
class. For minimization,

\[
VSS = EEV - RP, \qquad EVPI = RP - WS.
\]

For maximization both contrasts reverse. `EVIU` is the VSS presentation in v1:
its comparator is explicitly the deterministic expected-value solution induced
by the declared point estimate. The result returns the EV problem and selected
solution, EEV feasibility, recourse/stochastic-program optimum, statewise
wait-and-see solutions, VSS/EVIU, EVPI, full ties and a policy audit.

Nonanticipativity is structural: each enumerated policy declares exactly one
decision for each shared history node. Histories partition the finite states at
every recourse stage. Feasible state objectives, scenario probabilities,
objective direction and common value units are explicit. The evaluator does
not buy, observe or price information; EVPI is a diagnostic upper contrast.
Later-stage partitions must refine earlier partitions and declared available
information must be cumulative, so crossing histories or forgotten signals
cannot masquerade as a coherent filtration.

The two-stage fixture is a nonlinear point-estimate counterexample. The
deterministic objective at the mean selects a policy whose expected uncertain
cost is worse than the stochastic optimum. The multistage fixture exercises a
three-stage history tree and maximization signs. Induced-policy infeasibility is
reported with null EEV/VSS/EVIU; absence of any relatively-complete recourse
policy fails closed.

Python exact finite execution is experimental. DVSS and VMS are reviewed as
related multistage diagnostics but deferred until separately referenced
contracts are approved. Approximate/external solvers and risk criteria beyond
expected value are unsupported. Rust, R and Julia execution are absent; Mojo
remains an external upstream boundary.
