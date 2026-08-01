# Distribution-Family Information implementation review

Date: 2026-08-01

Scope: issue #557 and native implementation subissues #731–#735

Review range: `b78120e1..dd686cef`, followed by the F557-5 remediation diff

## Initial independent result

The independent implementation review failed with four High and one Medium
finding. The specification adapter bypassed frozen schema constants, arbitrary
conditional means could be certified as exact, candidate family definitions
and parameterizations were absent, free-text comparability always produced a
verified claim, and tiny negative gross values were clipped.

A separate assurance review also found eager-import and capability-registry
drift, stale shared evidence digests, incomplete installed-wheel evidence,
missing result-semantic assertions, and changed-branch coverage below policy.

## Remediation

- The public specification adapter validates the complete installed Draft
  2020-12 input schema and cross-field semantics before evaluating any value.
- Version 1 accepts only exact enumerated conditional expectations, with a
  structured assurance record preserved in the result and estimator evidence.
- Ordered candidate definitions record family or assumption, parameterization,
  within-family integration, definition and parameter sources, data reference,
  and value transformation.
- Comparability requires common population, horizon, discounting, value
  semantics, and cost-location identifiers, affirmative verification, and an
  evidence reference.
- Direct Python calls match JSON fail-closed behavior for strings, metadata,
  probabilities, matrix structure, and the probability-sum tolerance cap.
- Stable summation is used consistently. Gross VDI is never clipped; negative
  or non-finite exact arithmetic raises an error.
- Lazy discovery, frontier capabilities, C16 projection, CLI registry, wheel
  execution, result semantics, documentation, and DSA/VoF shared evidence were
  synchronized without editing the frozen stable extension-policy contract.
- Defensive and semantic branches are exercised at 100 percent line and branch
  coverage in the focused #557 runtime/contract audit.

## Re-review result

PASS. The independent reviewer verified every original finding. A final Medium
finding concerning oversized direct-API probability tolerances was remediated
by enforcing the schema's `1e-6` maximum in shared semantic validation and by a
zero-sum regression test. No Critical, High, Medium, or Low findings remain.

Hosted review automation subsequently identified that malformed direct-call
objects could fail during Python conversion before reaching the shared semantic
validator. The public evaluator now preserves existing `InputError` failures
and normalizes conversion `TypeError`, `ValueError`, and `OverflowError` into
the same fail-closed boundary. Regression cases cover malformed values,
probabilities, model definitions, assurance, comparability, and information
cost. The separate empty-tie warning is unreachable: semantic validation
requires a non-empty finite value row, and the selected finite maximum or
minimum is itself exactly present and therefore always `isclose` to itself.

Focused contract/runtime/surface assurance passed 87 tests. The wider
integration slice passed 146 tests, including wheel, CLI registry, canonical
projection, stable extension policy, and shared DSA/VoF evidence. Ruff, Ty, all
30 frontier contracts, the 95-page documentation build, and full Conductor
validation for 144 tracks passed with zero errors and warnings.

The first whole-suite run reached 3,079 passes, 16 skips, and 93.12 percent
coverage; its sole provider-wheel failure was caused by a full host filesystem
and passed when rerun alone after temporary-file cleanup. A later whole-suite
retry was invalidated by the same storage condition. Hosted exact-head checks
remain the authoritative clean-run gate.

## Boundary

This is an independent implementation and contract review, not named
scientific approval. Hosted exact-head checks, merge, Rust/R/Julia parity,
stable promotion, release, and issue closure remain separate gates.
