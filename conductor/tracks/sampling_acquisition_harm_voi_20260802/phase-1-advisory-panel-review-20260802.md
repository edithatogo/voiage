# Phase 1 automated advisory-panel review

## Status and authority

Three independent automated reviewers examined the exact Phase 1 artifacts:
an estimand/domain reviewer, a statistical-assurance reviewer and a governance/
API reviewer. This is repository advisory evidence only. It is not the named
independent human scientific/domain verdict required by H8 and cannot provide
ethics, regulatory, runtime, promotion, release or publication authorization.

## Initial findings and dispositions

| Severity | Finding | Disposition in H3-R1 |
|---|---|---|
| High | The scalar lacked explicit incremental comparison with `d0`. | Define design-indexed `W_B`, `G(d; d0)`, incremental cost and incremental harm; add a nonzero-baseline example. |
| High | Decision-time information, causal effects and downstream changes were underspecified. | Require design-indexed potential outcomes, observable filtration, policy measurability, interference, spillovers and dropout. |
| High | Positive loss was incorrectly described with lower-tail CVaR. | Define upper-tail positive-loss and lower-tail signed-welfare conventions, atoms and an adverse-mass counterexample. |
| High | Separability, commensurability and affected-party treatment were too weak. | Require policy independence and additive welfare decomposition, perspective/scale/numeraire/horizon/discount provenance, unaggregated party results and noncompensatory rights gates. |
| High | Statistical feasibility could be mistaken for ethics approval. | Use feasible/infeasible/indeterminate mathematical status and a separate accountable authorization gate. |
| High | Source attribution and provenance were defective. | Correct Strong to Heath et al.; add a versioned, content-addressed source manifest and direct final-guideline authorities. |
| High | Repository authorization metadata resembled scientific approval. | Replace it with bounded repository scope authorization and split every downstream gate. |
| Medium | Under-reporting was treated only as sensitivity. | Require a latent-harm observation model, validation/restrictions, partial identification or `not_identified`. |
| Medium | Vector/constrained optimization could fabricate one selected design. | Retain nondominated sets and complete ties; selection is nullable without a reviewed ordering. |
| Medium | Candidate, mathematical and authorized sets were conflated. | Separate evaluated candidates, mathematical constraint status and accountable authorization. |
| Medium | Lifecycle text and H1/H2 status drifted. | Synchronize track index and gate metadata. |
| High on repeat review | Acquisition-harm consequences could appear in both downstream welfare and the separately subtracted harm valuation. | Require a mutually exclusive outcome-component ledger; restrict the scalar to a non-overlapping decomposition and otherwise use total joint welfare exactly once. |

No initial reviewer found evidence supporting a generic executable method. All
runtime discovery remains fail closed. A repeat review must find no unresolved
Critical or High item before the automated Phase 1 checkpoint can complete.

## Final repeat-review result

The estimand/domain, statistical-assurance and governance/API roles repeated
their read-only reviews against exact commit
`eeb50c303246b4865c4a0b92e4669c92dd8196ea`. Each reported zero unresolved
Critical or High findings. Evidence entry `064902d9afd4e7683507ae3b420138ba18ad50ebf0fd2cafb8f1022d56919881`
supersedes the original H1/H2 pins and records the exact validation commands.
This closes only the automated Phase 1 repository checkpoint; H8 remains
pending and runtime remains prohibited.
