.. _rust-python-bridge-policy:

=========================================
Rust numerics core Python bridge policy
=========================================

Purpose
=======

This decision record defines the Python bridge policy for the Rust numerics
core. It records the PyO3/maturin decision, the Python facade preservation
rule, the fallback behavior, and the fixture parity and benchmark gates that
must hold before a kernel is called Rust-complete.

The policy is owned by the ``rust-frontier-numerics-migration-completion``
track and is enforced by ``tests/test_rust_migration_matrix.py``.

Decision: Python remains the public facade
==========================================

The Rust numerics core is the canonical deterministic and stochastic kernel
owner, but **Python remains the public facade**. The public API
(``voiage.methods.*``) is the only supported surface for users. The Rust
core is consumed through contract fixtures, benchmarks, and an optional
future PyO3 bridge — never as a direct user-facing import.

This preserves public API compatibility: no breaking change to the Python
facade is permitted unless an additive, explicitly approved change is
documented in the changelog and the migration matrix.

PyO3 and maturin decision
=========================

A runtime PyO3 bridge is **not** part of the base install today. The
decision is deferred until benchmark evidence shows that a Rust-backed
kernel delivers a material speedup over the NumPy reference for a real
workload.

When the bridge is approved:

1. ``maturin`` builds the ``voiage-core`` wheel as an optional
   ``voiage[rust]`` extra so the base install never gains a native
   compilation dependency.
2. ``PyO3`` exposes only the contract functions already validated by the
   fixture parity tests — never new public surfaces.
3. The Python facade delegates to the Rust kernel with a transparent
   fallback to the NumPy reference when the native extension is absent or
   raises.

Until then, the Rust numerics core is exercised by ``cargo test`` and the
fixture parity tests in ``tests/`` so the contract stays live without a
runtime dependency.

Fallback behavior
=================

Every migrated kernel must preserve the Python result envelope exactly. If
the Rust path is unavailable, the Python reference path must produce the
same value, diagnostics, and reporting envelope. A kernel is only marked
``rust_status: complete`` when:

* ``parity_status`` is ``verified`` (fixture parity tests pass in CI), and
* ``benchmark_status`` is ``ci_gated`` (a Rust benchmark runs in CI).

Fixture parity
==============

Fixture parity is the gate between a Rust contract and a Rust-complete
kernel. Each migrated kernel must have a deterministic fixture that is
exercised by both the Python facade and the Rust core, with identical
result envelopes. Parity fixtures live under ``specs/`` and the
corresponding tests live under ``tests/``.

Benchmarks
==========

Rust benchmarks live under ``bindings/rust/benches/``. A kernel is only
``benchmark_status: ci_gated`` when its benchmark runs in the Rust CI job
and the scalar baseline artifact is committed for reproducibility.

Migration priority
==================

The migration matrix classifies every numerical kernel by priority:

* **high** — core VOI kernels (EVPI, EVPPI, EVSI, ENBS) and the partial
  information family already ported to Rust.
* **medium** — frontier-adjacent kernels (structural VOI, NMA VOI, CEAF,
  dominance, heterogeneity) where runtime maturity and benchmark evidence
  may justify a future port.
* **low** — perspective, preference, distributional, threshold, and
  validation VOI where the performance value of a Rust port is low.

Out of scope
============

* Irreversible external submissions, paid cloud actions, or hardware
  purchases without explicit user approval.
* Weakening existing CI, coverage, contract, or docs gates.
* Marking experimental or external-gated work complete without evidence.

See also
========

* :doc:`rust_core_handoff` for the Rust core domain model and handoff
  boundary.
* :doc:`rust_accelerators` for the accelerator abstraction contract.
* ``specs/rust/migration_matrix.json`` for the machine-readable kernel
  inventory.
