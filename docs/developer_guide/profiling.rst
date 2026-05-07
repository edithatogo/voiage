Profiling and Performance
=========================

This guide documents the Rust-core profiling workflow around the committed
scalar CPU baseline. It is intentionally narrow: the goal is to describe the
measured baseline, the JSON artifact contract, and the criteria for deciding
when SIMD or accelerator work deserves a follow-on track.

Baseline scaffold
-----------------

The current benchmark scaffold lives in ``bindings/rust/benches/``:

- ``scalar_cpu_baseline.rs`` runs the deterministic EVPI workload
- ``scalar_cpu_baseline.json`` records the workload, expected value, and
  regression policy
- ``README.md`` explains the local entrypoint and the current comparison rule

The baseline is intentionally scalar and deterministic:

- workload: a fixed two-strategy EVPI matrix
- expected result: ``3.0``
- metric type: ``scalar_cpu``
- comparison rule: exact workload/value match
- regression policy: ``ci-contract-only``

Recommended local check:

.. code-block:: bash

   cargo test --benches scalar_cpu_baseline -- --nocapture

Workflow
--------

Use the baseline in three steps:

1. Run the scalar benchmark and confirm the EVPI result matches the committed
   artifact.
2. Record timing, memory, or throughput measurements in the same artifact
   family using the same workload identity.
3. Compare the new artifact against the committed baseline before promoting a
   new optimization claim.

The key rule is that the workload seed and EVPI value remain stable unless the
track explicitly re-baselines them.

Artifact format
---------------

The committed artifact is JSON and should stay small enough to review in code
review or CI logs.

Example:

.. code-block:: json

   {
     "benchmark_name": "scalar_cpu_baseline",
     "metric_type": "scalar_cpu",
     "workload": {
       "seed": 42,
       "repeats": 10000,
       "net_benefits": [[10.0, 1.0], [2.0, 8.0]]
     },
     "expected": {
       "evpi": 3.0,
       "comparison_rule": "exact",
       "regression_policy": "ci-contract-only"
     },
     "metadata": {
       "phase": "phase-1-scalar-cpu-baseline",
       "notes": [
         "Deterministic baseline for the Rust core performance track.",
         "Timing comparisons are deferred until a stable baseline artifact exists."
       ]
     }
   }

How to read the outputs
-----------------------

The current artifact family is correctness-first. Timing is observed, but the
committed contract only enforces the scalar workload and value.

- ``metric_type`` identifies the measurement family.
- ``expected.evpi`` is the correctness anchor.
- ``expected.comparison_rule`` states how strict the comparison is.
- ``expected.regression_policy`` says whether CI only records the artifact or
  enforces a threshold.
- ``metadata.phase`` records which profiling phase produced the artifact.

When memory and throughput artifacts arrive, they should keep the same JSON
family and add measured fields for the new metric rather than replacing the
scalar contract. The scalar workload and expected EVPI remain the baseline
reference unless the track explicitly re-baselines them.

Promotion criteria for SIMD or accelerators
-------------------------------------------

SIMD, Rayon, and accelerator work should be promoted only when the scalar
baseline and artifact layer already exist.

Open a follow-on track when all of the following are true:

- the scalar baseline is stable and reproducible
- the hot path shows a measurable gain from vectorization or parallelism
- the proposed change preserves the same result semantics and tolerance policy
- the memory/throughput artifacts show a repeatable improvement, not a one-off
- the optimization can be described as an internal execution change rather
  than a new public contract

Practical order:

1. Scalar CPU baseline
2. Memory and throughput measurement
3. Rayon or equivalent multithreading feasibility
4. SIMD feasibility
5. GPU or other accelerator feasibility only if the earlier data justify it

If a candidate optimization needs a different workload, a different result
shape, or a different correctness policy, it belongs in a follow-on track
rather than in the baseline profiling contract.

Related guidance
----------------

See the companion pages for the Rust-core migration boundary and the
acceleration decision policy:

* :doc:`rust_core_handoff`
* :doc:`rust_parallelism`
* :doc:`rust_accelerators`
