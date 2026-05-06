Rust Core Handoff
==================

This guide defines the boundary between the Rust core and the language
bindings/adapters that sit on top of it.

The architecture decision is:

* Rust is the authoritative execution core for deterministic VOI kernels,
  shared result contracts, diagnostics, and reporting envelopes.
* Python remains the primary façade for orchestration, CLI entrypoints,
  plotting, and compatibility wrappers.
* R, Julia, TypeScript, Go, .NET, and any future language packages are thin
  bindings or adapters over the Rust contract. They should translate native
  inputs into the shared Rust shapes and forward the results without
  re-implementing the core numerical policy.

What stays in Rust core
-----------------------

Rust core owns the pieces that must stay deterministic, schema-stable, and
shared across languages:

* numeric kernels and summary calculations for stable VOI methods
* domain types such as ``ValueArray``, ``ParameterSet``, ``TrialDesign``, and
  the typed result envelopes
* diagnostics, method metadata, and CHEERS-style reporting payloads
* serialization behavior for cross-language contracts
* validation for invalid shapes, non-finite values, duplicate IDs, and other
  deterministic contract failures

If a method needs to be reproducible across Python, R, and future bindings, it
belongs in Rust core first.

What stays in bindings or adapters
----------------------------------

Binding packages should stay thin and focused on language-specific concerns:

* parsing native inputs and producing Rust-compatible shapes
* user-facing CLI or package-manager plumbing
* light validation that improves error messages before the Rust call
* language-native packaging, release, and registry integration
* wrappers that preserve the Rust envelope shape without changing field names
  or collection ordering

Binding packages should not become duplicate cores. If a binding starts to own
its own numerical policy, the migration boundary has drifted.

How to add a new deterministic method
-------------------------------------

1. Define the method contract in Rust first.
2. Reuse the existing domain types when possible instead of inventing a new
   wire shape.
3. Return a typed summary or an ``AnalysisEnvelope`` that preserves
   diagnostics and reporting.
4. Keep the calculation deterministic and test it against fixed inputs.
5. Add only the smallest wrapper needed in Python or another binding.
6. Export the method from the Rust crate and document it as the source of
   truth for the public result shape.

The preferred pattern is to keep the method result serde-friendly and to make
the reporting payload CHEERS-aligned from the outset.

How result envelopes and reporting stay stable
----------------------------------------------

Result envelopes are part of the public contract, not an internal detail.
Their stability rules are:

* keep field names stable once they are documented
* prefer additive changes over shape changes
* preserve deterministic collection ordering where the output is serialized
* keep ``Diagnostics`` and ``Reporting`` populated even for simple summary
  methods
* use explicit method metadata and approximation status rather than inferring
  them from context

If a binding needs a language-specific convenience shape, it should wrap the
Rust envelope rather than mutating it. That keeps cross-language testing and
fixture validation aligned.

Compatibility notes
-------------------

* Transitional Python orchestration remains allowed while the Rust core grows,
  but it should be treated as a façade.
* The EVSI summary envelope is already in Rust core; the stochastic EVSI
  kernel is the follow-on work tracked separately in Conductor and should be
  described as kernel-only work underneath the existing summary contract.
* The approximation policy for EVSI belongs to the kernel track, not the
  summary envelope, so any approximation variant must preserve the same
  result shape and reporting contract.
* Experimental or frontier methods may have their own contract pages, but they
  still need to fit the same reporting and envelope stability rules once they
  become release-bound.
* Any new binding must describe how it forwards to the Rust core before it
  introduces extra result-shape behavior of its own.
