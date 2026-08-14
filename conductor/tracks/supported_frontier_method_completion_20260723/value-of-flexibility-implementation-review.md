# Value of Flexibility implementation review

Date: 2026-08-01

Scope: issue #559 on PR #723

Review range: `6fd474b1..17f85e58`, followed by the F559-4 remediation diff

## Initial independent result

The independent implementation and contract review failed with two High and
three Medium findings. Scenario probabilities could be omitted and silently
replaced with equal weights. Advertised irreversibility and lock-in controls
lacked governed policy-dependent semantics, with the stage-common lock-in
shift cancelling from the estimand. Results omitted named axes and discarded
provenance, arithmetic overflow did not consistently fail closed, and
`exercise_decisions` implied a sequential path across mutually exclusive
timing scenarios.

## Remediation

- Every public v1 execution requires exact named scenario probabilities,
  including an explicit runtime rejection of `None`.
- Non-zero discount, irreversibility and lock-in controls fail closed in the
  runtime and both input and result schemas until units and policy-dependent
  semantics are separately governed.
- Results preserve decision-stage names, strategy names and deterministic
  provenance through Python, fixtures, schemas and the CLI.
- Weight normalization and adjusted scenario values are finite-validated after
  arithmetic, including overflow cases.
- `exercise_decisions` is explicitly unsupported (`null`) for mutually
  exclusive scenarios; adjacent ordered-scenario choice changes have a
  separate diagnostic name.
- Exact evidence pins the runtime, schemas, fixtures, focused tests, user
  example and contract README.

## Re-review result

PASS. The independent reviewer verified every original finding and the final
schema/evidence consistency fixes. No Critical, High, Medium or Low findings
remain. Fifty-one focused tests passed; Ruff and repository-wide formatting
checks passed.

## Boundary

This is an independent implementation and contract review, not named
scientific approval. Hosted exact-head checks, merge, transition-constrained
dynamic programming, Rust/R/Julia execution, stable promotion, release and
issue closure remain separate gates.
