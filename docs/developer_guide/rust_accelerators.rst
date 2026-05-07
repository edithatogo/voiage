Rust Accelerators and Feasibility Boundaries
============================================

This note records the workloads that are plausibly worth accelerating in the
Rust core, and the cases where GPU, TPU, or custom-circuit work is not worth
the complexity.

The rule is simple: accelerate only the workloads that are compute-bound,
shape-stable, and already covered by a deterministic contract. Anything that
changes the contract shape or depends on irregular control flow should stay
on the scalar path.

What is plausibly GPU-bound
---------------------------

The Rust core has a few workload families that can plausibly benefit from GPU
execution if the implementation is reworked around large batches and dense
array operations:

* scalar summary kernels over many repeated PSA samples
* large matrix reductions for EVPI-style workloads
* frontier-style sweeps where the input surface is already dense and regular
* repeated dominance or CEAF scans over wide strategy sets
* batched memory/throughput profiling workloads that are dominated by the
  same arithmetic on every sample

These are the cases where the workload is mostly arithmetic, the inputs are
already numeric arrays, and the output can remain a stable summary envelope.
That makes them the only sensible candidates for accelerator feasibility
work in the short term.

Why TPU or custom-circuit support is only conditional
------------------------------------------------------

TPU and custom-circuit support are only useful when the workload is large,
regular, and worth paying the control-flow and deployment cost.

That means they are conditional on all of the following:

* the workload is large enough to amortize device transfer and compilation
  overhead
* the kernel can be expressed as dense tensor math without branching over
  strategy-specific or sample-specific control flow
* the result contract can stay identical to the scalar CPU contract
* the implementation can tolerate a more rigid execution model than the CPU
  baseline

In practice, that makes TPU or custom-circuit support a follow-on feasibility
question, not a default direction. The current Rust core is still a contract-
first library, so the accelerator path must prove it can preserve the same
result envelopes and deterministic tests.

Non-goals
---------

The following are explicitly not worth accelerator effort unless a later
profiling track proves otherwise:

* small scalar workloads with only a handful of strategies or samples
* code paths dominated by validation, serialization, or reporting assembly
* irregular branching logic such as dynamic study-design selection
* methods whose main cost is regression fitting or host-side orchestration
* anything that requires a new public result shape just to use the device
* host-specific optimization tricks that cannot be reproduced on Linux CI

Those cases are either too small to benefit from offloading, or they are
shaped by control flow rather than throughput. They should remain on the CPU
until the profiling evidence says otherwise.

Practical guidance
------------------

When evaluating a Rust accelerator idea, ask these questions in order:

1. Is the workload already covered by a deterministic scalar contract?
2. Is the input shape large and regular enough to batch efficiently?
3. Can the same result envelope be produced without changing the contract?
4. Does the profiling artifact show a real gain in throughput or latency?
5. Does the memory profile still make sense after device transfer overhead?

If the answer to any of those is no, the work should stay in the scalar core
or be pushed into a separate experimental track.
