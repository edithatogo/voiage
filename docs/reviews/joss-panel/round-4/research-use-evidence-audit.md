# Round 4 research-use evidence audit

Date: 24 July 2026  
Disposition: no qualifying research use found

This sidecar audit supports the simulated panel. It is not a formal JOSS
editorial finding.

## Result

No attributable execution of `voiage` was found in another author-owned
repository that answered a genuine research question or produced a substantive
research output. The synthetic paper example demonstrates functionality but
does not answer a real research question.

## Repositories examined

- `vop_poc_nz` publishes a compatibility contract and hand-calculated
  references. Its validator does not import or execute `voiage`.
- `closer-to-whom` contains an unfulfilled compatibility adapter tested with a
  mock rather than the installed package.
- `ee_trd` contains an independent predecessor implementation and no `voiage`
  invocation.
- `foi-o` contains no dependency, invocation, or result.
- RuleSpec, Innovate, and Lifecourse contain synthetic contracts or future
  integration plans.
- NHRA Game and OIA Case Study contain dependency locks but no invocation or
  attributable output.

Issue #471 contains the author's validation invitation but no non-author report.

## Strongest truthful statement

The `voiage` conformance suite consumes a versioned interoperability bundle
published by the same author's `vop_poc_nz` project, with hand-calculated
analytical references. The repository also includes fixed-seed synthetic
demonstrations. No attributable use of `voiage` to produce a research result
has yet been documented.

## Shortest evidence-producing route

Run the published `voiage` release against non-synthetic PSA outputs from a
defined `vop_poc_nz` research question. Preserve the package version, input
provenance and digest, command, environment, output, interpretation, and a
clean-room verification workflow in an immutable commit. This would be
same-author developer research use; non-author evidence should still be sought
through issue #471.
