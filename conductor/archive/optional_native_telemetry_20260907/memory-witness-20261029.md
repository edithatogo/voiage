# Telemetry large-record witness

Baseline: `54b718be106a24e6419922341052df5e543401b7`.
The existing collector source has SHA-256
`2a3f420bf0a7f52dd1584bd37f4989a81212ef87e85f7168814882bdda41f770`.
This bounded slice owns `rust/crates/voiage-diagnostics/src/otel_export.rs`;
the track record and changelog are integrator-owned. It targets an existing
collector's memory behavior and does not establish a need for network export.

Witness: a one-mebibyte event name must be rejected rather than retained in the
queue. The current implementation is expected to fail that assertion; a repair
must preserve normal-size records and establish an explicit bound covering
capacity, names, IDs, attribute keys and values, rather than just this example.
Run the witness under the resource guard with Python 3.15.1 using
`cargo test --manifest-path rust/Cargo.toml -p voiage-diagnostics --features otel --locked oversized_event_is_not_retained`.
Observe the intended assertion failure before modifying implementation. A
compiler, dependency, or missing-tool failure does not count as a red witness.
This is an adversarial example, not an agreed general telemetry budget.

On Python 3.15.1 and Rust 1.99.1, the guarded red run compiled and ran one
test; it failed on `!collector.record(oversized)` (exit 101). The guarded
focused rerun after the change passed all six collector tests (exit 0). The
implementation caps the queue at 1,024 records, retains at most 32 attributes
per record, and rejects records whose retained string allocation capacities
total more than 64 KiB. This bounds collector-owned retained payloads, not
allocations made by callers before submission or total process memory. Requested
limits above the caps are clamped, an explicit behavior change for oversized
configurations. The guarded locked Rust workspace all-features suite passed
after explicitly setting `PYO3_PYTHON` to this worktree's Python 3.15.1. Its
first attempt failed at build time because PyO3 selected system Python 3.11;
that attempt is not test evidence. The guarded diagnostics crate test also
passed with default features disabled. Full package and hosted gates remain open.

The first full tox run exposed one shared cause in the harness and coverage
suite: the packet still pinned the pre-change collector hash. The tested source
hash is now refreshed in that packet; the original hash remains above. The
failed full run does not qualify the candidate. Rerun affected checks and the
final gate on the corrected candidate.

Hosted Rust workspace CI then rejected two empty-result assertions under
`clippy::assert-is-empty`. The assertions were changed without modifying the
collector behavior and the packet hash refreshed. The guarded workspace
all-targets, all-features Clippy command, six focused collector tests, and the
repository harness subsequently passed on Python 3.15.1. Fresh hosted checks
remain required.
