# Independent implementation review — #599 heterogeneity value

## Review scope

Fresh independent review of signed implementation head
`33136da569357e85f797a9ae0966a70bc074fdb2`, covering the C0/Cf/P0/Pf and
EVPI/EVSI identities, strict input and result contracts, exact finite Python
execution, public API and CLI, generated schemas and fixtures, documentation,
frontier/governance projections, tests, static analysis and evidence.

The review keeps this delivery experimental. Named scientific approval of the
subgroup specification, selection-bias and sparse-subgroup treatment, stable
promotion, Rust/R/Julia parity, hosted exact-head checks, release, parent #599
closure and umbrella #318 closure remain separate gates.

## Findings before remediation

### High — the portable result was not bound to the evaluated input

The standalone result validator reconstructed the headline values from the
result's own audit tables, but it did not validate the reported selected
actions or complete ties and did not bind the result to the exact input model.
Direct mutations of the current-population selected action and tie, subgroup
selected action and tie, perfect-state and signal selected actions, state ID,
subgroup-selection declaration, and evaluated-state count all passed result
validation at the reviewed head. A coordinated internally consistent rewrite
could therefore describe a different analysis while retaining a valid-looking
portable result.

Required remediation: embed the complete strict input contract in assurance,
bind it with canonical SHA-256, validate that embedded input independently,
and require standalone exact re-evaluation to reproduce the complete result.
Add explicit mutation tests for every accepted fabrication above and for a
coordinated audit/value rewrite.

## Scientific and scope assessment

- The finite enumeration implements the prespecified identities correctly:
  static value is `Cf - C0`, dynamic value is `Pf - P0`, and
  `dynamic - static = EVPIf - EVPI0` under maximization, with the sign-reversed
  equivalent under minimization.
- The sample extension correctly integrates joint state-signal weights before
  optimization and preserves the corresponding
  `segmentation - static = EVSIf - EVSI0` identity.
- Eligibility is explicit and the common-policy problem uses the intersection
  of subgroup-eligible actions. Subgroup policy optimization remains distinct.
- The implementation does not replace the stable descriptive heterogeneity
  helper and does not claim selection adjustment, sparse-subgroup inference,
  cross-language parity or stable maturity.

## Final disposition

The High finding was remediated in signed commit `4b5d8d08`. The result now
embeds the complete strict input contract, binds it with canonical SHA-256 and
requires exact standalone re-evaluation. All previously accepted direct
mutations and a coordinated audit/value rewrite are rejected.

Final validation passed:

- 47 focused heterogeneity-value tests, including 100 randomized independent
  oracle cases and the new commitment/mutation suite;
- 100% statements and 100% branches across the changed contract and evaluator
  (277 statements, 88 branches, no misses or partial branches);
- Ruff check/format and BasedPyright with zero errors or warnings;
- 81 combined feature, public API, frontier and Conductor governance tests;
- frontier manifest, GitHub cross-reference and full Conductor validation,
  with 148 tracks, zero errors and zero warnings.

No Critical, High or Medium implementation-review findings remain. This is
local independent implementation/scientific review evidence only. Hosted
exact-head and installed-wheel checks remain pending, and the named scientific,
parity, promotion, release and closure gates listed above remain open.
