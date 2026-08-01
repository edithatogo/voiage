# MoSCoW requirements — planned v1.3.0

## Must

- **ISP-M1:** Use one declared finite joint-world law for action values and all
  source observations, with explicit value, cost and time units.
- **ISP-M2:** Enforce cost, latency, privacy, rights, licensing, freshness, SLA,
  coverage, cardinality, exclusivity and order constraints fail closed.
- **ISP-M3:** Exhaustively evaluate bounded feasible source sequences and return
  baseline/resolved policy values, gross/net value, willingness to pay,
  complete ties, switches, conditional marginals and decision-value Shapley
  attribution.
- **ISP-M4:** Supply strict schemas, exact fixtures, provenance, deterministic
  serialization, CLI, discovery, documentation and assurance diagnostics.

## Should

- Preserve every evaluated feasible sequence so independent implementations can
  audit the optimum and deterministic tie policy.
- Distinguish decision-source attribution from predictive Data Shapley.

## Could

- Add separately reviewed probabilistic source channels, adaptive stopping and
  exact dynamic policies in later contract versions.

## Won't

- Treat independent EVSI scores or additive knapsack values as #582 evidence;
  fabricate rights, independence, SLA or provenance; use approximation in v1;
  or claim stable, Rust, R, Julia or Mojo execution.
