# Implementation handoff: Optional native telemetry qualification

The [AC1 need assessment](./need-assessment-20261029.md) records the decision
merged in PR #1238 to defer network export. The accepted VOP pilot measured
functional and replay behavior. It supplies neither a trace-export use case
nor a workload-specific acceptance budget. T1.2 has its documented outcome.

On 1 November the maintainer requested an opt-in network trace capability.
The Rust diagnostics crate now offers the `otel-http` feature for applications
with a local OTLP receiver. See the [usage guide](../../../rust/crates/voiage-diagnostics/README.md)
and the [loopback tests](../../../rust/crates/voiage-diagnostics/tests/otlp_loopback.rs).
Consumer-specific overhead targets remain open. The VOP pilot supplied no
measurement for this new path.

The [large-record witness](./memory-witness-20261029.md) records the local
collector repair delivered in PR #1235. The [baseline review](./baseline-review-20261029.md)
predates that repair.

## Review handoff — 1 November 2026

The VOP evidence digest matched the assessment. The guarded
(10 tests, Python 3.15.1), and the execution packet passed its readiness check.
The next integrator operation is to reconcile these records. The loopback implementation supersedes that documentation-only handoff.
