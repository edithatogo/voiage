# Phase 1 Automated Review — 2026-08-21

## Scope

- Track initialization boundary: `3898744b`.
- Reviewed implementation through cross-reference commit `1ea5147a`.
- Paths: the track contract and ledger, the platform schema and manifest, the
  validator, focused tests, and the Conductor GitHub cross-reference manifest.

## Findings and fixes

1. **High — released-source provenance mismatch: fixed.** The approved draft
   contained a release commit that did not match the source revision pinned by
   Yggdrasil PR #14292. The specification and machine-readable contract now use
   exact commit `964a0fc334ece9509387cd07d43776adf38be240`.
2. **High — platform-catalogue source mismatch: fixed.** The initial source URL
   named a newer `BinaryBuilderBase` commit than the tree resolved by the pinned
   Yggdrasil manifest. The contract now binds commit
   `76c4aab80ad5019af59af0f42e5669109cd5194b` and tree
   `38ac28858e80c575fc2ff3c7ac73982459c4482d`.
3. **High — additional negative filters were insufficiently constrained:
   fixed.** Filters now declare specificity and evidence kind. The validator
   rejects OS-only or otherwise broad predicates, placeholder evidence,
   architecture-wide non-toolchain exclusions, and hosted-failure claims made
   before a hosted run exists.
4. **Medium — pathological coverage did not exercise every catalogue member or
   malformed JSON: fixed.** The focused suite now removes each of the 18
   classifications in turn and verifies fail-closed reconciliation, in addition
   to malformed, duplicate, aggregate, exclusion, and evidence-overclaim cases.

No Critical, High, or Medium finding remains open for Phase 1.

## Applicable guidance

- `conductor/code_styleguides/python.md`: **Pass** for the Python validator and
  tests. Ruff formatting/checking and `ty` pass. Pylint is not a configured
  repository dependency or declared gate; the authoritative repository
  workflow uses Ruff and `ty`, so the Pylint suggestion is not applicable.
- Platform-specific guides: **Not Applicable**. No manifest-selected platform
  guide intersects the contract-only Phase 1 paths.

## Validation

- Ruff check and format: passed for all changed Python files.
- `ty`: passed for the validator.
- Focused pytest: 20 passed across the platform contract and Conductor
  cross-reference suites on Python 3.14.5.
- Canonical platform validator: 18 classified, 16 included, two excluded.
- Vale: zero errors, warnings, or suggestions across the track prose.
- Full Conductor validation: 150 tracks, zero errors, zero warnings.
- GitHub cross-reference validation: passed.
- Evidence-ledger chain validation and `git diff --check`: passed.

The phase proves a repository-owned candidate contract. It does not prove an
expanded upstream build, Yggdrasil acceptance, JLL generation, Julia General
registration, or indexing.
