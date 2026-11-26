# Telemetry need assessment — 29 October 2026

Decision: **defer exporter selection and network-export implementation** until
a concrete, measured consumer need and acceptance budget are available. This
is a scoped decision under AC1, not completion of the telemetry feature or
acceptance of AC2–AC5. The existing optional, bounded local collector remains
available; this decision does not remove or qualify it as a network exporter.

The VOP pilot's retained `evidence.jsonl` at SHA-256
`80a066f8faba7cb2b0c8d0ab9d52fc40e2f84c38b8e8b7c53547fa525e1333b5`
records synthetic functional, installed-consumer, replacement and replay
checks, and the recorded repository closeout. It does not report a trace
collection use case, a telemetry comparison, a workload-specific overhead
measurement, or an agreed latency, memory, loss or cardinality budget. The
absence of those items in the reviewed pilot is not evidence that a consumer
has no such need. The pilot's historical Python versions remain historical
evidence, not authorization to use them for current work.

At main `04f2e1beb05a01f5ff12b070f5f10f97673527d2` the diagnostics crate
declares the optional trace feature with no default features. The present
collector retains bounded local events and exposes a caller-owned drain; its
existence does not establish network delivery, context propagation to a local
collector, bounded exporter retries, or observed production demand.

On Python 3.15.1, `python -m pytest tests/test_optional_telemetry_contract.py
tests/test_dependency_promotion_policy.py -q` passed 10 tests (exit 0) under
the resource guard on commit `d306feb790ee429cc60f954b655889bb3d1c7326`.
The two test files and collector source had identical SHA-256 hashes on that
commit and the main revision above: respectively `01c64027862193bbe2d63ff88d86863952603a7b7ae407fd0a9a7c924ca89c4c`,
`d509a55fd836a8c09d468563d33bc41628d0ba6d97a29d3c3c3fa4ce4f84e2cc`, and
`3e86514d3d763ff0a1220f771e89f30452fdf563e69b492eacc62ce889bcf23e`.
The guard's sampled peak was 85392 KiB. These tests inspect existing contracts
and dependency policy; they are not exporter or end-to-end qualification.

Reconsider when a named consumer supplies a bounded trace use case, an
appropriate local or approved test collector, and measurable baseline and
acceptance budgets. Then freeze a phase packet and execute the specified
feature, toolchain, privacy, outage, shutdown, and numerical-equivalence
gates. A synthetic benchmark by itself cannot establish consumer need.
