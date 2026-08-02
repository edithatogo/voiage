# Scientific review panel — information-source portfolio VOI

Date: 2026-08-01

Scope: exact-finite experimental joint-world information-source portfolio
optimizer

Panel: estimand, numerical/reproducibility, and API/boundary/maturity reviewers

## Disposition

**PASS for the declared experimental scope.** The panel found no unresolved
Critical, High, or Medium finding preventing retention of the Python exact
finite evaluator as experimental. This panel disposition does not authorize
stable promotion, release, or issue closure.

## Findings

- A single finite joint-world law defines action values and all source
  observations, so dependence, redundancy, and complementarity are represented
  directly rather than through additive or marginal assumptions.
- Conditional prefix marginals and exact decision-value Shapley attribution are
  deterministic and tie-aware; Shapley is explicitly not Data Shapley.
- Procurement constraints fail closed, and the no-procurement comparator is
  explicitly separated from source constraints.
- Independent fixtures, exhaustive subset enumeration, order-invariance,
  complementary/redundant source cases, identities, pathologies, and
  deterministic serialization support the exact finite scope.
- Public Python and CLI surfaces, documentation, capability metadata, and
  unsupported Rust/R/Julia/Mojo dispositions agree. Adaptive stopping,
  probabilistic channels, and approximate optimization remain unsupported.

## Remaining gates

Stable promotion remains blocked on independent reference/parity evidence,
cross-language implementations or retained unsupported dispositions,
maintainer approval, release evidence, and governed issue closure.
