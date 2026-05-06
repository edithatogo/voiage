# lifecourse v1 Examples

This directory reserves human-reviewable example payloads for the
`lifecourse` VOI artifact profile.

The examples should stay small, deterministic, and readable in code review.
They are intended to document the interchange shape before a shared fixture set
is finalized with the `lifecourse` repository.

Planned example coverage:

- run-bundle metadata
- net-benefit tables by strategy
- parameter-sample tables
- VOI-ready artifact payloads with method settings and provenance
- illustrative result envelopes that preserve metadata across EVPI, EVPPI,
  EVSI, and ENBS payloads

These examples should map cleanly to the stable `voiage` core API schemas while
remaining portable across Python, R, Julia, TypeScript, Go, Rust, and .NET
bindings.

The current illustrative result envelope is stored in
`../fixtures/illustrative/voi_result_envelope.json` so the contract stays close
to the deterministic fixture set.

Example payloads should preserve the same compatibility anchors as the fixtures
themselves: `voiage` `0.2.0`, `lifecourse` profile `v1`, and HEOML profile
`0.1`. That keeps the exchange path visible in review and makes fixture
validation repeatable for contributors.

When validating examples, check the manifest compatibility block first, then the
illustrative result envelope, and only then the normative fixture outputs.
