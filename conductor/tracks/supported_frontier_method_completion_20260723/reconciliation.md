# Child evidence reconciliation

Audited against `origin/main` at `b8395abf` and the live GitHub hierarchy on
2026-07-31. The machine-readable source is `child-dispositions.json`.

## Delivery evidence

| Issue | Disposition | Evidence | Remaining boundary |
|---|---|---|---|
| #571 | Experimental implementation merged | PR #679; completed study-design track | Scientific review and stable promotion |
| #595 | Experimental implementation merged | PR #712; 60 exact-head checks passed, 5 intentionally skipped, 0 failed | Scientific review and stable promotion |
| #619 | Experimental implementation branch | PR #676 | Merge/current-head checks, scientific review, vector covariance and stable promotion |

## Residual accepted families

| Issues | Current evidence | Why it does not satisfy AC-06 |
|---|---|---|
| #556, #557, #559, #560 | PSA/threshold plots, distributional equity, dynamic real options, mock MCDA/Pareto helpers | Adjacent estimands or mocks are not the named contracts. |
| #570, #572, #582 | Constraint/risk helpers, finite-signal analysis, independent experiment portfolios | They omit the required joint policy, forecast, or source-observation semantics. |
| #593, #594 | Implementation loss, EVPI and real-options helpers | They do not implement the joint implementation/information matrix or EVIU/VSS contract. |
| #596–#600 | Threshold plots, sequential/bandit helpers, strategic/privacy helpers, heterogeneity, aggregate EVSI | They omit event density, belief-state control, signed social value, static/dynamic heterogeneity, or outcome-conditional sample value. |
| #558 | No matching runtime or portable workflow | A narrative checklist cannot satisfy the executable qualitative assessment contract. |

No residual family has a reviewed exclusion. They therefore remain accepted
planned implementations. Excluding them would materially change issue #318 and
requires an explicit scientific/contract decision rather than an agent-created
blanket disposition.

## Dependency and maturity boundaries

The declared method-census and stable-core workstream tracks remain `new`; the
umbrella cannot claim their classifications complete. Merged child capability
evidence remains experimental. GitHub issue state, Project status, schemas,
plots, aliases, or adjacent helpers are not implementation evidence.
