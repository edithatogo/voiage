# Residual supported-frontier disposition — 2026-08-01

This record reconciles the residual supported-frontier families #596–#600
against the repository state on 2026-08-01. It is a repository-owned
classification, not a scientific promotion decision. The GitHub issues remain
open until their contracts, evidence, bindings, documentation, and maturity
gates are independently satisfied.

## Decision rule

- **Implemented** means the issue's named estimand and acceptance contract are
  present in the runtime, deterministic fixtures/tests, documentation,
  registry, and required binding dispositions. A related helper, plot, alias,
  or generic solver is not sufficient.
- **Successor scope** means useful adjacent infrastructure exists, but the
  issue's explicit contract remains future work and must stay linked to its
  GitHub issue and Conductor phase.
- **Excluded** means the repository deliberately does not promise the method;
  the exclusion must be documented and the issue closed or superseded with
  maintainer approval.

No residual family meets the implemented or excluded criteria in this review.
All five are therefore classified as **successor scope**.

## Family dispositions

| GitHub issue | Method family | Conductor phase | Disposition | Repository evidence and remaining gate |
| --- | --- | --- | --- | --- |
| [#596](https://github.com/edithatogo/voiage/issues/596) | Event-localized information value and information density | F596 | **Successor scope** | Existing tail/event references and plotting or conditional-value helpers do not establish the required localized estimands, density-integral identities, accuracy symmetries, and Python/Rust/R/Julia/Mojo dispositions. Add enumerable discrete/continuous fixtures, integral-error checks, serialization, and review evidence before implementation closure. |
| [#597](https://github.com/edithatogo/voiage/issues/597) | Belief-state and intervention-aware sequential information value | F597 | **Successor scope** | Existing sequential, monitoring, knowledge-gradient, or generic MDP surfaces are adjacent only. The explicit belief-state/POMDP contract, myopic/non-myopic counterexamples, intervention chronology, policy trajectories, approximation bounds, and binding dispositions remain unimplemented evidence gates. |
| [#598](https://github.com/edithatogo/voiage/issues/598) | Signed, social, and strategic value of information | F598 | **Successor scope** | Existing social, privacy, persuasion, or game-theoretic helpers do not prove signed realized/ex-ante value, welfare transfers, information avoidance, constrained negative values, or public/private/team sharing accounting. Add aligned-agent reductions, conflict/constraint cases, and no-clipping diagnostics before promotion. |
| [#599](https://github.com/edithatogo/voiage/issues/599) | Static/dynamic value-of-heterogeneity decomposition | F599 | **Successor scope** | Existing heterogeneity, subgroup, CATE, and policy surfaces are adjacent and do not establish the static-vs-dynamic decomposition, population-weighting identities, selection-bias diagnostics, or marginal segmentation value. Add enumerable subgroup references and binding/registry evidence. |
| [#600](https://github.com/edithatogo/voiage/issues/600) | Outcome-conditional and low-value sample-information value | F600 | **Successor scope** | Existing EVSI, uncertainty, standard-error, or risk helpers do not establish outcome-conditional VSI estimands, EVSI expectation reconciliation, negative delta-EV cases, threshold monotonicity, or continuous-outcome assurance. Add distributional fixtures, calibration/dispersion diagnostics, and binding dispositions. |

## Conductor and GitHub reconciliation

- Parent programme: #313; parent frontier issue: #318.
- Track: `supported_frontier_method_completion_20260723`.
- The five issues are intentionally not closed by this record. Their
  contracts remain successor work under the corresponding F596–F600 phases.
- Existing adjacent-frontier contract scaffolds and merged implementation
  slices may be used as prerequisites, but do not satisfy the closure
  prohibitions in the issues.
- Stable promotion, release, scientific review, and Rust/Python/R/Julia/Mojo
  parity remain separate gates. Local tests or a passing schema validator must
  not be reported as those gates being complete.

## Next actions

1. Create one implementation PR per family (or an explicitly reviewed bundle)
   with deterministic normative fixtures and the issue-specific acceptance
   matrix.
2. Record hosted checks and cross-language disposition in the supported-
   frontier Conductor evidence ledger.
3. Request scientific review against the primary references named in each
   issue.
4. Promote or exclude only after the maintainer records the decision and the
   GitHub issue, Conductor phase, registry, documentation, and release evidence
   agree.

