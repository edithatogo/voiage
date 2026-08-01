# Independent implementation review — risk-sensitive constrained VOI

Issue: [#570](https://github.com/edithatogo/voiage/issues/570), delivery
subissues #757, #758 and #761, and
[PR #769](https://github.com/edithatogo/voiage/pull/769).

Review date: 2026-08-01. Reviewed maturity: experimental Python execution
planned for v1.3.0 under canonical C18/M22. This is an independent repository
implementation review, not scientific approval, stable promotion,
cross-language parity, release evidence or parent-issue closure.

## Scope and semantics checked

The review covered the input/result schemas and semantic validators, exact
finite evaluator, normative fixture, CLI/public exports, documentation,
capability and promotion records, evidence hashes, Conductor requirements and
the C18/M22 issue hierarchy.

For a fixed policy, the expected-value and declared-utility objectives use

\[
E[z_\pi]=\sum_s p_s z_{\pi,s}.
\]

The lower-tail CVaR/expected-shortfall objective sorts the submitted finite
values and integrates the worst `1-confidence_level` probability mass,
including a fractional boundary state. Minimax regret maximizes the negative
of the largest declared statewise regret. Expected utility therefore consumes
already declared utility values; it does not invent a utility transform from
monetary outcomes.

Baseline candidates use one policy in every state. Perfect-information
candidates enumerate every state-contingent policy mapping. Deterministic
constraints require satisfaction in every state, while chance constraints
require the probability mass of satisfying states to meet the declared
threshold. Both problems use the same probabilities, objective, constraints,
risk functional and cost-placement rule. The constant baseline mappings are a
subset of the informed feasible set, so exact gross perfect-information value
must be nonnegative. Information cost is deducted afterward and net value
remains signed.

## Findings and remediation

No Critical, High or Medium finding remains open.

- **High — fixed:** tolerance ties were used for policy selection as well as
  presentation. A merely near-optimal lexicographic informed mapping could
  therefore produce negative gross perfect-information value. Commit
  `58a119bc` now selects an exact argmax, uses lexicographic ordering only
  among exact optima, and still reports every declared-tolerance tie. A
  counterexample regression preserves this invariant.
- **High — fixed:** Python `NaN` and infinity satisfy JSON Schema's generic
  numeric type. A non-finite cost could reach a non-portable result. Request
  and result validation now reject non-finite numbers recursively, with input
  and result regressions and refreshed schema/evidence hashes.
- **Medium — fixed:** hosted run `30687309902` showed that the new CLI command
  was absent from the exact public-command registry and the registered family
  was absent from the frontier promotion checklist. Both registries now
  include the experimental #570 surface without permitting a stable claim.
- **Medium — fixed:** hosted run `30688015038` treated four BasedPyright
  `Unknown` traversal warnings in the recursive finite-number validator as a
  failed lint/type job. Commit `18747c8b` explicitly narrows mapping keys and
  values plus list items to `object`; the hosted-equivalent local command now
  reports zero errors and zero warnings.

Independent enumeration confirmed the normative expected-value result, the
fractional lower-tail CVaR result, minimax-regret reduction, deterministic and
chance feasibility, exact-threshold behavior, complete ties and explicit
baseline infeasibility. Constraint-removal effects remain labelled as discrete
evidence rather than local shadow prices.

## Assurance

- 75 focused feature, CLI, frontier and Conductor-governance tests passed.
- The complete hosted-equivalent Ruff check/format, Bandit and `ty` scope
  passed. BasedPyright reported zero errors and zero warnings across
  `voiage/logging.py`, `voiage/contracts` and `scripts/export_v2_contracts.py`.
- Frontier-contract validation passed, including
  `risk_sensitive_constrained_perfect_information`.
- Full Conductor validation passed for 146 tracks with zero errors and zero
  warnings; `git diff --check` passed.
- Two local coverage attempts could not collect because NumPy reported
  `cannot load module more than once per process`, including in an isolated
  Python 3.13 environment. Plain focused pytest passed. This machine-specific
  collection fault is not coverage evidence; hosted exact-head coverage and
  wheel checks remain mandatory.

## Boundaries and remaining gates

The implementation is exact only for bounded finite policy/state sets and the
declared ex-ante deterministic or chance constraints. It does not support
imperfect/sample information, continuous or mixed-integer optimization,
endogenous or intertemporal constraints, local dual shadow prices, or
post-information changes to the risk functional. Python is executable; Rust,
R and Julia remain unsupported for this family, and Mojo remains external.

Hosted exact-head/full-suite and installed-wheel evidence, independent
scientific review, canonical C18 projection reconciliation, polyglot parity,
stable promotion, merge, release and #570/#318 closure remain separate gates.
Subject to those gates, no open implementation finding prevents continued
review of the experimental Python surface.
