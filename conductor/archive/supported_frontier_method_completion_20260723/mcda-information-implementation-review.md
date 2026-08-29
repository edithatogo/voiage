# Independent implementation review — finite additive MCDA information value

Issue: [#560](https://github.com/edithatogo/voiage/issues/560) and delivery
subissues #746–#750 under programme #318.

Review date: 2026-08-01. Reviewed maturity: experimental Python implementation
planned for v1.3.0 under canonical Should requirement M21. This record is an
independent repository review, not scientific approval, stable promotion,
cross-language parity, hosted exact-head evidence, release evidence or issue
closure authorization.

## Scope reviewed

- canonical `M21` and `M21-U1`–`M21-U4`, the MCDA Mermaid design and
  `F560-1`–`F560-5` plan entries;
- `mcda-information-reference-review.md`, including its adjacent-surface and
  independent-reference boundaries;
- installed input/result contracts and semantic validators in
  `voiage/contracts/mcda_information.py`;
- the shared exact evaluator in `voiage/methods/mcda_information.py`;
- portable schemas, correlated normative fixture, pathology fixtures,
  capability metadata, fixture manifest and evidence projection under
  `specs/frontier/mcda-information/v1/`;
- lazy Python discovery, CLI execution, accessible text, public plot exports,
  information-value and rank-acceptability plots, documentation, examples,
  global frontier discovery and promotion registration;
- focused analytical, invariant, pathology, contract, CLI, plot, public-export
  and installed-wheel tests, including the inline external wheel request.

The review compared the implementation with the frozen equations rather than
accepting method names, weighted-score helpers, mocks, schemas or plots as
numerical evidence.

## Equations and assumptions checked

For fixed ex-ante criterion value functions, the evaluator implements

\[
z_{ak}(\omega)=v_k(x_{ak}(\omega)),\qquad
U(a,\omega)=\sum_k w_k(\omega)z_{ak}(\omega),
\]

with finite alternatives and criteria, declared raw units and directions,
two fixed linear anchors, nonnegative weights summing to one, and one submitted
finite joint law over criterion performance and preference weights. The law is
enumerated directly; conditional calculations group its submitted joint states
and therefore do not silently factor outcome and preference uncertainty.

For a declared partition `R`, the reviewed implementation computes

\[
V_0=\max_a E[U(a,\omega)],\qquad
V_R=\sum_r P(R=r)\max_a E[U(a,\omega)\mid R=r],
\]

\[
VOI_R=V_R-V_0,\qquad NVOI_R=VOI_R-c_R.
\]

Each cost is nonnegative, action-specific and expressed both in its original
unit and as a declared conversion to the common aggregate-value unit and basis.
Net value remains signed. Gross value is obtained from the optimization
identity and is not clipped.

The v1 contract requires exactly one criterion, one preference and one joint
action. The joint action must use exactly the outcome keys of the criterion
action and preference keys of the preference action, so it is their joint
refinement. The decomposition was checked against

\[
I_{C,W}=VOI_{C\vee W}-VOI_C-VOI_W,
\]

and the two reported increments are `VOI[C join W] - VOI[C]` and
`VOI[C join W] - VOI[W]`. The normative correlated law independently gives
baseline value `0.63375`, criterion and preference gross value `0`, joint gross
value `0.028`, interaction `0.028`, joint cost `0.01` and signed net joint value
`0.018`.

Statewise opportunity loss is computed from the same aggregate matrix as

\[
L(a,\omega)=\max_b U(b,\omega)-U(a,\omega).
\]

Complete ties use one declared absolute/relative rule. Rank acceptability uses
the frozen fractional complete-tie convention: a member of a tie of size `m`
receives `1/m` at each occupied rank. Expected and statewise Pareto diagnostics
compare the fixed direction-normalized criterion vectors componentwise, with a
declared absolute tolerance; they are not cost-effectiveness dominance and are
not inferred from the additive optimum.

The validators also enforce unique and aligned identifiers, finite positive
state probabilities, probability reconciliation, fixed increasing raw anchors,
direction-consistent value anchors, valid domains, exact alternative/criterion
rows, normalized state-specific weights, known disjoint latent-key families,
the joint-refinement rule, and finite nonnegative action costs. Submitted state
order is retained for presentation, while deterministic semantic single-axis
and `joint-N` partition identifiers make the runtime result conform to the
independent normative fixture.

## Findings by severity

### Critical

None.

### High

None open. Two adversarial findings discovered during this review were fixed
and regression-tested before this final record:

1. The original semantic validator accepted a joint action that did not refine
   the declared criterion and preference actions. It now requires the exact
   union of their key sets and rejects a mismatched joint action.
2. The original schema allowed a zero-mass joint state to create a zero-mass
   conditional partition and leak `ZeroDivisionError`. Installed and projected
   schemas now require strictly positive state probabilities, a regression test
   covers the case, and the public evaluator defensively normalizes arithmetic
   failures to `InputError`.

### Medium

None open. The initial runtime used generic partition identifiers that differed
from the independently authored expected fixture. Runtime identifiers are now
deterministic and semantic, submitted state order is preserved, and a recursive
full-contract test requires identical structure and text plus `2e-12` absolute
numeric agreement with the committed expected result.

### Low

None open in the reviewed implementation scope.

The checked-in evidence projection remains intentionally stale until this
review file exists: its current `contract_only` status and hashes cannot include
this record. Regenerating `fixtures/evidence.json` with
`scripts/export_mcda_information_contracts.py`, then rerunning the formerly
excluded pin test, is a required sequencing step rather than an accepted
finding or maturity claim.

## Evidence commands and results

- `uv run --isolated --with '.[plotting]' --with pytest --with pytest-cov pytest --noconftest tests/test_mcda_information.py tests/test_mcda_information_contract.py tests/test_mcda_information_surfaces.py -q -k 'not contract_evidence_is_sha256_pinned'`
  — **45 passed**. The single omitted test is the sequencing-dependent evidence
  hash assertion described above.
- The same focused suite before the final evidence refresh reached the expected
  single failure at `test_contract_evidence_is_sha256_pinned`: checked-in
  `execution_status` was `contract_only` while the finalized projection must be
  `experimental_python`. No numerical, contract, CLI or plot test failed.
- `uv run ruff check voiage/contracts/mcda_information.py voiage/methods/mcda_information.py voiage/plot/mcda_information.py scripts/export_mcda_information_contracts.py tests/test_mcda_information.py tests/test_mcda_information_contract.py tests/test_mcda_information_surfaces.py`
  — **passed**.
- `uv run python scripts/validate_frontier_contract.py` — **passed**, including
  `validated finite_additive_mcda_perfect_information`.
- `npm run check` from `docs/astro-site` — **passed** with zero errors, warnings
  or hints.
- `git diff --check` — **passed**.
- Adversarial probes confirmed that mismatched joint action keys and zero-mass
  joint states are rejected before conditional evaluation after remediation.

The repository's existing worktree environment had an incomplete namespace-only
`pandas` installation, so focused execution used an isolated `uv` environment.
An attempted isolated packaging-node collection also lacked its test-only
`rfc8785` dependency; this does not substitute for the repository's installed-
wheel test. The installed-wheel test itself is present and must run in the
normal packaging/hosted assurance lane after evidence regeneration.

## Maturity and claim boundaries

The reviewed surface is a finite, compensatory, additive-value presentation of
partial perfect information. It supports fixed ex-ante normalization, a
correlated finite joint law and explicitly declared criterion, preference and
joint partitions. It does not support or approximate:

- AHP pairwise elicitation or consistency diagnostics;
- ELECTRE, PROMETHEE, outranking, vetoes, thresholds or other
  noncompensatory methods;
- multiplicative, multilinear, fuzzy, interval, robust or risk-sensitive
  aggregation;
- alternative-relative, draw-relative or post-information normalization;
- ordinal importance classes treated as cardinal values or weights;
- imperfect-sample EVSI, adaptive research or posterior elicitation models;
- social-choice aggregation, endogenous feasible sets or portfolio selection;
- stable API status or Rust/R/Julia parity. Mojo remains external.

Python is the only executable implementation evidenced here. The fixture is
repository-authored synthetic analytical evidence; it is not empirical MCDA
elicitation validation. Accessible plotting uses hatches, outlines, numeric
annotations, distinct markers, labels and legends rather than colour alone.

## Remaining gates

Before F560-5, #560, stable promotion or release can be reconciled as complete:

1. regenerate and verify the final schema/evidence projection after this review
   record, including the full pin test;
2. run the full local suite and the actual installed-wheel black-box test in its
   declared packaging environment;
3. obtain hosted exact-head required-check evidence for the final commit;
4. retain independent scientific review of the additive-model assumptions,
   terminology, normalization, preference elicitation and empirical validity;
5. retain Rust, R and Julia as unsupported until shared-fixture executable
   parity exists, and Mojo as an external boundary;
6. complete stable-promotion, merge, signing, release and registry gates
   separately; and
7. reconcile Project 28, #560 and parent #318 only from their own governed
   closure evidence.

Subject to the required evidence regeneration and external gates above, the
reviewed experimental Python implementation satisfies the repository-owned
M21 contract. No open implementation finding prevents the final evidence and
hosted-assurance sequence.
