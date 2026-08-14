# Belief-state information fourth-review remediation

## Scope and independence

A fourth fresh reviewer inspected issue #597 / C18-M28 at signed head
`d42922a3f766dd07d34b3560d5681e71c5c0a09e`. The reviewer did not author the
original implementation or any of the first three remediations. Review covered
the fixed-horizon policy tree, result assurance, exact-enumeration budget,
preflight/runtime identity, contracts, schemas, fixtures, documentation and
governance records.

This is an engineering implementation review. It is not independent scientific
approval, hosted exact-head or installed-wheel evidence, polyglot parity,
stable promotion, release evidence or programme closure.

## Finding reported before editing

**Medium — several advertised exact result claims were structurally accepted
without being bound to the evaluated model.** The standalone result validator
accepted all of the following schema-valid mutations:

1. reducing `estimated_bellman_expansions` from 82 to 1;
2. shifting both final-horizon closed-loop and no-information values by 100
   while retaining their difference;
3. replacing the root selected control and complete tie with a fabricated ID;
4. flipping the action-dependent-transition diagnostic; and
5. flipping the usable-downstream-learning-response diagnostic.

The finding was reported to the implementing root before any edit. The root
then authorized scoped test-first remediation. Because this reviewer authored
the correction, a fifth fresh reviewer is required.

## Remediation

Signed commit `08c371af` adds a strict `model_assurance` envelope containing
the complete v1 input contract and its canonical SHA-256 commitment. The
portable result schema resolves that contract through the canonical input
schema. Standalone result validation now validates the commitment, rebuilds
the bounded finite model, reruns the exact evaluator and requires the complete
result to reproduce the committed model.

The library's producer path reuses the already validated source model and does
not perform a second Bellman solve. An independently stored result performs one
bounded reconstruction. Therefore the public 50,000-call contract continues to
describe one exact evaluation rather than silently doubling producer work.

This reconstruction binds the expansion estimate and fixed budget,
value-by-horizon cells and top-level values, policy selections and complete
ties, transition and learning diagnostics, policy tree, martingale assurance,
conditional sensing values and every other model-derived result field. A
changed input contract without its matching digest also fails closed.

## Independent budget verification

- Horizon one: preflight estimate 8; instrumented recursive calls 8.
- Normative horizon two: preflight estimate 82; instrumented calls 82, split
  into 52 adaptive, 12 no-information and 18 fully observed calls.
- The 20-state horizon-four adversary remains rejected before recursion at
  estimate 168,453.
- The null-sensor two-state horizon-twelve boundary remains accepted at 24,649,
  below the fixed 50,000-call budget.

## Local assurance

- 117 focused belief-state tests passed.
- The remediated method has 584 statements and 272 branches with 100% coverage.
- The combined feature, package-export, CLI, supported-frontier and frontier-
  fixture suite passed.
- Ruff and BasedPyright passed with zero findings on the changed Python files.
- Frontier-contract, canonical specialized-VOI projection, GitHub cross-
  reference, append-only ledger and full Conductor validation passed; Conductor
  reported 147 tracks with zero errors and zero warnings.

## Remaining boundary

A fifth fresh independent implementation review, hosted exact-head checks and
merge remain pending. Independent scientific review, Rust/R/Julia parity,
stable promotion, release, parent #597 closure and umbrella #318 closure also
remain pending.
