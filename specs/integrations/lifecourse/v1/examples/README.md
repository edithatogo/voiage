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

These examples should map cleanly to the stable `voiage` core API schemas while
remaining portable across Python, R, Julia, TypeScript, Go, Rust, and .NET
bindings.
