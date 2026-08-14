# Residual supported-frontier disposition — 2026-08-01

This record reconciles residual supported-frontier families #596–#600 against
the repository state. It is a repository-owned classification, not a
scientific promotion decision. The issues remain open until their contracts,
evidence, bindings, documentation, and maturity gates are satisfied.

## Decision rule

- **Implemented** requires the named estimand, runtime, deterministic fixtures,
  tests, documentation, registry, and binding dispositions.
- **Successor scope** means adjacent infrastructure exists but the explicit
  contract remains future work.
- **Excluded** requires documented scope exclusion and maintainer approval.

No residual family meets the implemented or excluded criteria in this review;
all five are therefore **successor scope**.

| Issue | Method family | Phase | Disposition |
| --- | --- | --- | --- |
| #596 | Event-localized information value and information density | F596 | Successor scope |
| #597 | Belief-state and intervention-aware sequential information value | F597 | Successor scope |
| #598 | Signed, social, and strategic value of information | F598 | Successor scope |
| #599 | Static/dynamic value-of-heterogeneity decomposition | F599 | Successor scope |
| #600 | Outcome-conditional and low-value sample-information value | F600 | Successor scope |

The five issues remain linked to parent programme #313, frontier parent #318,
and `supported_frontier_method_completion_20260723`. Existing adjacent
helpers and generic solvers are prerequisites only; they do not satisfy the
issue-specific contracts. Scientific review, stable promotion, release, and
Rust/Python/R/Julia/Mojo parity remain separate gates.

## Next actions

1. Create an implementation PR or reviewed bundle per family with normative
   fixtures and an issue-specific acceptance matrix.
2. Record hosted checks and language dispositions in the Conductor ledger.
3. Request scientific review against each issue's primary references.
4. Promote or exclude only after maintainer approval and synchronized issue,
   Conductor, registry, documentation, and release evidence.
