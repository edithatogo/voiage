# Signed, social and strategic information value v1

This experimental Python contract evaluates a complete finite joint-world law
against named agents, roles, sharing designs and nonanticipative policy
catalogs. It reports signed agent, role and social values without clipping,
together with transfers, costs, winners, losers, avoidance, policy switches,
rights receipts and strictly scoped Blackwell nonnegativity checks.
Portable results retain the declared agent-role map and the exact world,
policy and design identifiers evaluated, allowing semantic validation to
recompute role aggregates, ties, optima, diagnostics and assurance counts.

The installed Rust extension evaluates the admitted policy payoffs, accounting
ledgers and signed values. Python validates the v1 specification and emits
the portable result through the Python API and CLI.

The welfare aggregator is a declared weighted sum over cardinally comparable
agent utilities. That declaration is an assumption, not an empirical finding.
Each design names its comparator, recipient set, controller-facing rights and
purpose receipts, policy-selection mode and feasible finite catalog. The
design's recipient list determines which decision agents can condition their
actions on the signal. Fixed and declared responses are evaluated as supplied.
A verified finite-equilibrium entry requires a complete-catalog verification
receipt. Voiage evaluates the supplied catalog.
Infeasible designs and designs or comparators with transfer/cost accounting
are explicitly outside the Blackwell nonnegativity check.

The normative fixture is a finite construction informed by Li and Pozzi's
negative-information analysis: the selected signal-contingent policy harms the
private recipient by 2 units while adding 2 declared social-welfare units. It
is a contract fixture, not a reproduction or empirical calibration of their
model.

Bayesian persuasion, mechanism design, rational inattention, general games,
continuous worlds and incomplete probability laws remain adjacent or out of
scope. The standalone Rust kernel takes indexed numerical inputs. Full portable
v1 execution is available through the Python API and CLI. The R and Julia
contracts remain pending, and Mojo remains an external upstream boundary.
Scientific review and stable promotion remain separate gates.
