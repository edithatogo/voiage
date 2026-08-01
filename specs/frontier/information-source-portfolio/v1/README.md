# Information-source portfolio VOI v1 (experimental)

This contract values a procurement sequence from one explicit finite joint
world law. Each world supplies its probability, every action value and every
source observation. Grouping worlds by the selected joint observation tuple
therefore preserves declared source dependence, redundancy and complementarity
without adding independent EVSI scores.

For a sequence \(S\), the gross value is

\[
E_{Y_S}\left[\max_a E\{V(a,W)\mid Y_S\}\right]
- \max_a E\{V(a,W)\}.
\]

Net value subtracts declared source costs and the declared linear delay cost.
Willingness to pay is gross value less delay cost, before source price. The
selected sequence exposes prefix-conditional marginal values. Exact Shapley
attribution is computed over the selected source set using this decision-value
game; it is not predictive Data Shapley.

The evaluator exhaustively enumerates at most seven sources and applies budget,
latency, privacy, freshness, SLA, coverage, exclusivity and order constraints.
All source rights and provenance receipts must be cleared before evaluation.
The no-procurement option is an always-available external comparator rather
than a source sequence; assurance therefore reports feasible non-empty
sequences separately and states that source constraints do not apply to the
comparator.

This is experimental Python evidence only. Adaptive stopping, branching
acquisition, probabilistic source channels outside the declared worlds,
approximation, Rust, R and Julia are unsupported; Mojo remains external.
