# Child evidence reconciliation

Audited against `origin/main` and the live GitHub hierarchy through PR #751 on
2026-08-01. The machine-readable source is `child-dispositions.json`.

## Delivery evidence

| Issue | Disposition | Evidence | Remaining boundary |
|---|---|---|---|
| #556 | Experimental implementation on branch | PR #723; exact v1 schemas/fixture; independent implementation review | Hosted exact-head checks, merge, scientific review, polyglot execution and stable promotion |
| #557 | Experimental implementation on branch | PR #736; exact model-family-index v1 schemas/fixture; independent implementation review | Hosted exact-head checks, merge, scientific terminology/partition review, real probability provenance, polyglot execution and stable promotion |
| #558 | Experimental implementation merged | PRs #743 and #744; portable qualitative assessment/audit/rendering contracts; independent implementation and accessibility review | Practitioner/scientific review, polyglot execution, stable promotion and release |
| #559 | Experimental implementation on branch | PR #723; exact timing-scenario v1 schemas/fixture; independent implementation review | Hosted exact-head checks, merge, scientific review, transition-constrained policies, polyglot execution and stable promotion |
| #560 | Experimental implementation merged | PR #751 merged as `e8aaba82`; exact finite additive-MCDA v1 schemas/fixture, Python/CLI/plots, independent implementation review, all exact-head hosted checks and canonical v1.3 dispatch/consumer runs | Scientific review, Rust/R/Julia parity, stable promotion and release |
| #571 | Experimental implementation merged | PR #679; completed study-design track | Scientific review and stable promotion |
| #595 | Experimental implementation merged | PR #712; 60 exact-head checks passed, 5 intentionally skipped, 0 failed | Scientific review and stable promotion |
| #619 | Experimental implementation merged | PR #676; 60 exact-head checks passed, 5 intentionally skipped, 0 failed | Scientific review, vector covariance and stable promotion |

## Residual accepted families

| Issues | Current evidence | Why it does not satisfy AC-06 |
|---|---|---|
| #570, #572, #582 | Constraint/risk helpers, finite-signal analysis, independent experiment portfolios | They omit the required joint policy, forecast, or source-observation semantics. |
| #593, #594 | Implementation loss, EVPI and real-options helpers | They do not implement the joint implementation/information matrix or EVIU/VSS contract. |
| #596–#600 | Threshold plots, sequential/bandit helpers, strategic/privacy helpers, heterogeneity, aggregate EVSI | They omit event density, belief-state control, signed social value, static/dynamic heterogeneity, or outcome-conditional sample value. |

No residual family has a reviewed exclusion. They therefore remain accepted
planned implementations. Excluding them would materially change issue #318 and
requires an explicit scientific/contract decision rather than an agent-created
blanket disposition.

## Dependency and maturity boundaries

The declared method-census and stable-core workstream tracks remain `new`; the
umbrella cannot claim their classifications complete. Merged child capability
evidence remains experimental. For #560, VOP run `30684893440` and VOIAGE
receiver run `30684980076` additionally prove that the canonical v1.3
projection is synchronized without drift. GitHub issue state, Project status,
schemas, plots, aliases, or adjacent helpers are not by themselves
implementation evidence.
